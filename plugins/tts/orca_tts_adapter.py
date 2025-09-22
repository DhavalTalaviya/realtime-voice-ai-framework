from typing import AsyncIterator, List, Optional, Dict, Any
import asyncio
import numpy as np
import re
import logging
from dataclasses import dataclass
from functools import lru_cache
import time
import threading
import os

logger = logging.getLogger(__name__)

@dataclass
class OrcaConfig:
    """Configuration class for Orca TTS settings."""
    access_key: str
    model_path: Optional[str] = None
    speech_rate: float = 1.0
    random_state: Optional[int] = None
    target_sample_rate: int = 22050  # Orca's native sample rate
    frame_size_ms: int = 20  # milliseconds
    chunk_max_length: int = 200
    sentence_max_length: int = 300
    max_retries: int = 2
    stream_mode: bool = True  # Use streaming by default for better performance
    
    def __post_init__(self):
        if not self.access_key:
            raise ValueError("Orca access_key is required. Get it from https://console.picovoice.ai/")
        self.speech_rate = max(0.5, min(self.speech_rate, 2.0))  # clamp speech rate
        self.frame_size = self.target_sample_rate * self.frame_size_ms // 1000

class AudioProcessor:
    """Handles audio processing operations efficiently for Orca output."""
    
    def __init__(self, target_sr: int = 16000, source_sr: int = 22050):
        self.target_sr = target_sr
        self.source_sr = source_sr
        
    def process_audio(self, audio_data: List[int]) -> np.ndarray:
        """Process Orca audio output (16-bit PCM integers) to float32 array."""
        if not audio_data:
            return np.array([], dtype=np.float32)
        
        # Convert from 16-bit integers to float32
        audio = np.array(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample if needed
        if self.source_sr != self.target_sr:
            audio = self._resample_audio(audio)
        
        return audio
    
    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from Orca's native rate to target rate."""
        if self.source_sr == self.target_sr:
            return audio
            
        try:
            import resampy
            return resampy.resample(
                audio, self.source_sr, self.target_sr,
                filter='kaiser_best', parallel=False
            ).astype(np.float32)
        except ImportError:
            return self._fallback_resample(audio)
    
    def _fallback_resample(self, audio: np.ndarray) -> np.ndarray:
        """Fallback resampling methods."""
        try:
            from scipy import signal
            ratio = self.target_sr / self.source_sr
            new_length = int(len(audio) * ratio)
            return signal.resample(audio, new_length).astype(np.float32)
        except ImportError:
            logger.warning("Using basic interpolation - install resampy/scipy for better quality")
            ratio = self.target_sr / self.source_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_rms: float = 0.15, 
                       target_peak: float = 0.8) -> np.ndarray:
        """Optimized audio normalization."""
        if len(audio) == 0:
            return audio
            
        abs_audio = np.abs(audio)
        peak = np.max(abs_audio)
        
        if peak < 1e-6:
            return audio
            
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Vectorized normalization strategy
        if peak > 0.95:
            scale = target_peak / peak
        elif rms < 0.02:
            scale = min(target_rms / rms, target_peak / peak)
        elif rms > 0.3:
            scale = target_rms / rms
        else:
            scale = 1.0
        
        if scale != 1.0:
            audio = audio * scale
        
        # Soft limiting
        return np.tanh(audio * 0.9).astype(np.float32) * 0.95

class TextProcessor:
    """Handles text preprocessing and chunking efficiently for Orca."""
    
    # Pre-compiled regex patterns for performance
    _WHITESPACE_PATTERN = re.compile(r'\s+')
    _SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')
    _COMMA_PATTERN = re.compile(r'(?<=[,;])\s+')
    
    # Orca-specific text replacements (optimized for ARPAbet pronunciation)
    _TEXT_REPLACEMENTS = {
        re.compile(r'\bDr\.', re.I): 'Doctor',
        re.compile(r'\bMr\.', re.I): 'Mister',
        re.compile(r'\bMrs\.', re.I): 'Misses',
        re.compile(r'\bMs\.', re.I): 'Miss',
        re.compile(r'\bProf\.', re.I): 'Professor',
        re.compile(r'\bAPI\b', re.I): 'A-P-I',
        re.compile(r'\bUI\b', re.I): 'U-I',
        re.compile(r'\bURL\b', re.I): 'U-R-L',
        re.compile(r'&'): ' and ',
        re.compile(r'@'): ' at ',
        re.compile(r'%'): ' percent',
        re.compile(r'\$'): ' dollar ',
        re.compile(r'#'): ' number ',
    }
    
    @classmethod
    @lru_cache(maxsize=128)
    def preprocess_text(cls, text: str) -> str:
        """Cached text preprocessing optimized for Orca."""
        if not text:
            return ""
            
        # Normalize whitespace
        text = cls._WHITESPACE_PATTERN.sub(' ', text.strip())
        
        # Apply Orca-friendly replacements
        for pattern, replacement in cls._TEXT_REPLACEMENTS.items():
            text = pattern.sub(replacement, text)
        
        # Clean up and ensure punctuation
        text = cls._WHITESPACE_PATTERN.sub(' ', text.strip())
        if text and not text[-1] in '.!?;:':
            text += '.'
            
        # Validate characters for Orca (will be validated by Orca itself, but good to check)
        return text
    
    @classmethod
    def split_into_chunks(cls, text: str, max_length: int = 200) -> List[str]:
        """Optimized text chunking for Orca streaming."""
        if len(text) <= 100:
            return [text]
        
        # Try sentence splitting first
        sentences = cls._SENTENCE_PATTERN.split(text)
        chunks = cls._build_chunks(sentences, max_length)
        
        # Fallback to comma splitting for long single sentences
        if len(chunks) == 1 and len(text) > 300:
            parts = cls._COMMA_PATTERN.split(chunks[0])
            if len(parts) > 1:
                chunks = cls._build_chunks(parts, max_length // 2)
        
        return [c for c in chunks if c.strip()]
    
    @staticmethod
    def _build_chunks(parts: List[str], max_length: int) -> List[str]:
        """Build chunks efficiently without repeated string operations."""
        chunks = []
        current_parts = []
        current_length = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            part_length = len(part)
            
            if current_length + part_length + len(current_parts) > max_length and current_parts:
                chunks.append(' '.join(current_parts))
                current_parts = [part]
                current_length = part_length
            else:
                current_parts.append(part)
                current_length += part_length
        
        if current_parts:
            chunks.append(' '.join(current_parts))
            
        return chunks

class OrcaTTSStreaming:
    
    def __init__(self, model_path: Optional[str] = None, 
                 speech_rate: float = 1.0, random_state: Optional[int] = None,
                 config: Optional[OrcaConfig] = None, **kwargs):
        # Backward compatibility: support both config object and individual parameters
        if config is not None:
            self.config = config
        else:
            self.config = OrcaConfig(
                access_key=os.getenv("ORCA_API_KEY"),
                model_path=model_path,
                speech_rate=speech_rate,
                random_state=random_state,
                **kwargs
            )
        
        self._orca = None
        self._orca_stream = None
        self._model_lock = threading.Lock()
        self._warmup_done = False
        self._error_count = 0
        self._last_error_time = 0
        self._circuit_breaker_threshold = 5
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            target_sr=16000,  # Standard output rate
            source_sr=self.config.target_sample_rate
        )
        self.text_processor = TextProcessor()
        
        logger.info(f"Initialized Orca TTS adapter")

    def _lazy_load(self):
        """Thread-safe lazy loading of the Orca engine."""
        if self._orca is not None:
            return
        
        with self._model_lock:
            if self._orca is not None:  # Double-check pattern
                return
            
            try:
                import pvorca
                
                # Create Orca instance
                if self.config.model_path:
                    self._orca = pvorca.create(
                        access_key=self.config.access_key,
                        model_path=self.config.model_path
                    )
                else:
                    self._orca = pvorca.create(access_key=self.config.access_key)
                
                # Update actual sample rate from Orca
                self.config.target_sample_rate = self._orca.sample_rate
                self.audio_processor.source_sr = self.config.target_sample_rate
                
                logger.info(f"Loaded Orca TTS engine @ {self.config.target_sample_rate}Hz")
                
            except ImportError as e:
                raise RuntimeError(f"pvorca library not available: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to load Orca TTS: {e}") from e

    def warmup(self):
        """Sync warmup method for backward compatibility."""
        if self._warmup_done:
            return
            
        try:
            self._lazy_load()
            # Test synthesis with short text
            test_audio = self._orca.synthesize("Test.", speech_rate=self.config.speech_rate)
            if test_audio:
                self._warmup_done = True
                logger.info("Orca TTS warmup completed successfully")
        except Exception as e:
            logger.warning(f"Orca TTS warmup failed: {e}")

    async def warmup_async(self) -> bool:
        """Async warmup with better error handling."""
        if self._warmup_done:
            return True
            
        try:
            await asyncio.to_thread(self._lazy_load)
            # Test synthesis
            test_audio = await asyncio.to_thread(
                self._orca.synthesize, "Test.", speech_rate=self.config.speech_rate
            )
            if test_audio:
                self._warmup_done = True
                logger.info("Orca TTS warmup completed successfully")
                return True
        except Exception as e:
            logger.warning(f"Orca TTS warmup failed: {e}")
            return False

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Main streaming interface with optimized processing."""
        if not text or not text.strip():
            return
            
        # Circuit breaker check
        if self._is_circuit_open():
            logger.warning("Orca TTS circuit breaker is open, skipping synthesis")
            return
            
        try:
            self._lazy_load()
            
            # Preprocess text once
            clean_text = self.text_processor.preprocess_text(text)
            if not clean_text:
                return
            
            logger.debug(f"Synthesizing with Orca: {clean_text[:100]}...")
            
            # Choose synthesis method based on config
            if self.config.stream_mode and len(clean_text) > 50:
                # Use streaming for longer texts
                async for chunk_bytes in self._stream_synthesis(clean_text):
                    yield chunk_bytes
            else:
                # Use single synthesis for short texts
                async for chunk_bytes in self._single_synthesis(clean_text):
                    yield chunk_bytes
                    
            # Reset error count on success
            self._error_count = 0
            
        except Exception as e:
            self._handle_error(e)
            logger.error(f"Orca stream synthesis failed: {e}")

    async def _stream_synthesis(self, text: str) -> AsyncIterator[bytes]:
        """Streaming synthesis using Orca's streaming capabilities."""
        try:
            # Split text into appropriate chunks for streaming
            chunks = self.text_processor.split_into_chunks(text, self.config.chunk_max_length)
            
            # Open stream with specified parameters
            stream = await asyncio.to_thread(
                self._orca.stream_open,
                speech_rate=self.config.speech_rate,
                random_state=self.config.random_state
            )
            
            try:
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    
                    # Add chunk to stream
                    audio_data = await asyncio.to_thread(stream.synthesize, chunk)
                    
                    if audio_data:
                        # Process and stream audio
                        async for audio_bytes in self._process_and_stream_audio(audio_data):
                            yield audio_bytes
                    
                    # Brief pause between chunks
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.01)
                
                # Flush remaining audio
                final_audio = await asyncio.to_thread(stream.flush)
                if final_audio:
                    async for audio_bytes in self._process_and_stream_audio(final_audio):
                        yield audio_bytes
                        
            finally:
                # Always close the stream
                await asyncio.to_thread(stream.close)
                
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")

    async def _single_synthesis(self, text: str) -> AsyncIterator[bytes]:
        """Single synthesis for shorter texts."""
        try:
            # Synthesize all at once
            audio_data = await asyncio.to_thread(
                self._synthesize_with_retry, text, self.config.max_retries
            )
            
            if audio_data is not None:
                async for audio_bytes in self._process_and_stream_audio(audio_data):
                    yield audio_bytes
                    
        except Exception as e:
            logger.error(f"Error in single synthesis: {e}")

    def _synthesize_with_retry(self, text: str, max_retries: int) -> Optional[List[int]]:
        """Synthesize with retry logic for robustness."""
        original_text = text
        
        for attempt in range(max_retries + 1):
            try:
                audio_data = self._orca.synthesize(
                    text=text,
                    speech_rate=self.config.speech_rate,
                    random_state=self.config.random_state
                )
                
                if audio_data and len(audio_data) > 0:
                    return audio_data
                    
            except Exception as e:
                if attempt < max_retries:
                    # Progressively simplify text
                    if attempt == 0:
                        # Remove special characters
                        text = re.sub(r'[^\w\s.,!?-]', '', text)
                    else:
                        # Truncate text
                        text = text[:len(text)//2]
                    logger.warning(f"Orca synthesis attempt {attempt + 1} failed, retrying with simplified text")
                    continue
                else:
                    logger.error(f"Orca synthesis failed after {max_retries} retries: {e}")
                    return None
        
        return None

    async def _process_and_stream_audio(self, audio_data: List[int]) -> AsyncIterator[bytes]:
        """Process and stream audio with proper timing."""
        try:
            # Convert Orca output to numpy array
            audio = self.audio_processor.process_audio(audio_data)
            
            if len(audio) == 0:
                return
            
            # Normalize audio
            audio = self.audio_processor.normalize_audio(audio)
            
            # Stream in frames
            frame_size = self.config.frame_size
            total_frames = len(audio)
            yield_counter = 0
            
            for i in range(0, total_frames, frame_size):
                chunk = audio[i:i + frame_size]
                
                # Pad incomplete frames
                if len(chunk) < frame_size:
                    chunk = np.pad(chunk, (0, frame_size - len(chunk)), 'constant')
                
                yield chunk.astype(np.float32).tobytes()
                
                # Yield control periodically for responsiveness
                yield_counter += 1
                if yield_counter % 5 == 0:  # Every ~100ms
                    await asyncio.sleep(0.001)
                    
        except Exception as e:
            logger.error(f"Error processing Orca audio: {e}")

    def _is_circuit_open(self) -> bool:
        """Circuit breaker pattern for error handling."""
        if self._error_count < self._circuit_breaker_threshold:
            return False
            
        # Reset circuit breaker after 60 seconds
        return time.time() - self._last_error_time < 60

    def _handle_error(self, error: Exception):
        """Centralized error handling."""
        self._error_count += 1
        self._last_error_time = time.time()
        logger.error(f"Orca TTS error (count: {self._error_count}): {error}")

    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        self._error_count = 0
        logger.info("Orca TTS circuit breaker reset")

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self._orca_stream is not None:
                await asyncio.to_thread(self._orca_stream.close)
                self._orca_stream = None
                
            if self._orca is not None:
                await asyncio.to_thread(self._orca.delete)
                self._orca = None
                
        except Exception as e:
            logger.warning(f"Error during Orca cleanup: {e}")
        finally:
            self._warmup_done = False
            logger.info("Orca TTS resources cleaned up")

    def get_valid_characters(self) -> str:
        """Get valid characters supported by Orca."""
        if self._orca is None:
            self._lazy_load()
        return self._orca.valid_characters

    def get_sample_rate(self) -> int:
        """Get Orca's audio sample rate."""
        if self._orca is None:
            self._lazy_load()
        return self._orca.sample_rate

    def get_max_character_limit(self) -> int:
        """Get Orca's maximum character limit per synthesis."""
        if self._orca is None:
            self._lazy_load()
        return self._orca.max_character_limit


# Factory function for easy instantiation (backward compatibility)
def create_orca_tts_streaming(
    access_key: str,
    model_path: Optional[str] = None,
    speech_rate: float = 1.0,
    random_state: Optional[int] = None,
    **kwargs
) -> OrcaTTSStreaming:
    """Create an Orca TTS streaming instance with custom configuration."""
    config = OrcaConfig(
        access_key=access_key,
        model_path=model_path,
        speech_rate=speech_rate,
        random_state=random_state,
        **kwargs
    )
    return OrcaTTSStreaming(config=config)