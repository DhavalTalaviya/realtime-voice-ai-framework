from typing import AsyncIterator, List, Optional, Dict, Any
import asyncio
import numpy as np
import re
import logging
from dataclasses import dataclass
from functools import lru_cache
import time
import torch

logger = logging.getLogger(__name__)

@dataclass
class TTSConfig:
    """Configuration class for TTS settings."""
    model_name: str = "tts_models/en/ljspeech/fast_pitch"
    device: str = "cpu"
    speed: float = 1.2
    target_sample_rate: int = 16000
    frame_size_ms: int = 20  # milliseconds
    chunk_max_length: int = 200
    sentence_max_length: int = 300
    max_retries: int = 2
    
    def __post_init__(self):
        self.speed = max(0.5, min(self.speed, 3.0))  # clamp speed
        self.frame_size = self.target_sample_rate * self.frame_size_ms // 1000

class AudioProcessor:
    """Handles audio processing operations efficiently."""
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self._resampler = None
        
    def resample_audio(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        """High-quality audio resampling with caching."""
        if source_sr == self.target_sr:
            return audio
            
        try:
            import resampy
            return resampy.resample(
                audio, source_sr, self.target_sr,
                filter='kaiser_best', parallel=False
            ).astype(np.float32)
        except ImportError:
            return self._fallback_resample(audio, source_sr)
    
    def _fallback_resample(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        """Fallback resampling methods."""
        try:
            from scipy import signal
            ratio = self.target_sr / source_sr
            new_length = int(len(audio) * ratio)
            return signal.resample(audio, new_length).astype(np.float32)
        except ImportError:
            logger.warning("Using basic interpolation - install resampy/scipy for better quality")
            ratio = self.target_sr / source_sr
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
    """Handles text preprocessing and chunking efficiently."""
    
    # Pre-compiled regex patterns for performance (FIXED - no problematic lookbehind)
    _WHITESPACE_PATTERN = re.compile(r'\s+')
    _SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')  # Simple fixed-width lookbehind
    _COMMA_PATTERN = re.compile(r'(?<=[,;])\s+')
    
    # Common abbreviations that shouldn't trigger sentence breaks
    _ABBREVIATIONS = {
        'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'st.', 'ave.', 
        'blvd.', 'rd.', 'ct.', 'vs.', 'etc.', 'i.e.', 'e.g.'
    }
    
    # Cached replacements for common terms
    _TEXT_REPLACEMENTS = {
        re.compile(r'\bDr\.', re.I): 'Doctor',
        re.compile(r'\bMr\.', re.I): 'Mister', 
        re.compile(r'\bMrs\.', re.I): 'Misses',
        re.compile(r'\bMs\.', re.I): 'Miss',
        re.compile(r'\bProf\.', re.I): 'Professor',
        re.compile(r'\bAPI\b', re.I): 'A P I',
        re.compile(r'\bUI\b', re.I): 'U I',
        re.compile(r'\bURL\b', re.I): 'U R L',
        re.compile(r'&'): ' and ',
        re.compile(r'@'): ' at ',
        re.compile(r'%'): ' percent',
    }
    
    @classmethod
    @lru_cache(maxsize=128)
    def preprocess_text(cls, text: str) -> str:
        """Cached text preprocessing for better performance."""
        if not text:
            return ""
            
        # Normalize whitespace
        text = cls._WHITESPACE_PATTERN.sub(' ', text.strip())
        
        # Apply replacements
        for pattern, replacement in cls._TEXT_REPLACEMENTS.items():
            text = pattern.sub(replacement, text)
        
        # Clean up and ensure punctuation
        text = cls._WHITESPACE_PATTERN.sub(' ', text.strip())
        if text and not text[-1] in '.!?;:':
            text += '.'
            
        return text
    
    @classmethod
    def split_into_chunks(cls, text: str, max_length: int = 200) -> List[str]:
        """Optimized text chunking algorithm with fixed regex."""
        if len(text) <= 100:
            return [text]
        
        # Split on sentence boundaries, then filter out abbreviations
        potential_sentences = cls._SENTENCE_PATTERN.split(text)
        sentences = cls._filter_abbreviation_splits(potential_sentences)
        chunks = cls._build_chunks(sentences, max_length)
        
        # Fallback to comma splitting for long single sentences
        if len(chunks) == 1 and len(text) > 300:
            parts = cls._COMMA_PATTERN.split(chunks[0])
            if len(parts) > 1:
                chunks = cls._build_chunks(parts, max_length // 2)
        
        return [c for c in chunks if c.strip()]
    
    @classmethod
    def _filter_abbreviation_splits(cls, parts: List[str]) -> List[str]:
        """Filter out sentence splits that are actually abbreviations."""
        if len(parts) <= 1:
            return parts
            
        filtered = []
        i = 0
        
        while i < len(parts):
            current = parts[i]
            
            # Check if this part ends with a common abbreviation
            if (i < len(parts) - 1 and 
                current.lower().rstrip().endswith(tuple(cls._ABBREVIATIONS))):
                # Merge with next part
                current = current + ' ' + parts[i + 1]
                i += 2
            else:
                i += 1
            
            if current.strip():
                filtered.append(current)
        
        return filtered
    
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

class CoquiTTSStreaming:
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/fast_pitch", 
                 device: str = "cpu", speed: float = 1.2, config: Optional[TTSConfig] = None):
        # Backward compatibility: support both old and new initialization styles
        if config is not None:
            self.config = config
        else:
            self.config = TTSConfig(
                model_name=model_name,
                device=device,
                speed=speed
            )
        
        self._tts = None
        self._model_sr = 22050
        self._warmup_done = False
        self._error_count = 0
        self._last_error_time = 0
        self._circuit_breaker_threshold = 5
        
        # Initialize processors
        self.audio_processor = AudioProcessor(self.config.target_sample_rate)
        self.text_processor = TextProcessor()
        
        logger.info(f"Initialized TTS with model: {self.config.model_name}")

    def warmup(self):
        """Sync warmup method for backward compatibility."""
        if self._warmup_done:
            return
            
        try:
            self._lazy_load()
            # Warm up with minimal text
            self._tts.tts("Test.", speed=self.config.speed)
            self._warmup_done = True
            logger.info("TTS warmup completed successfully")
        except Exception as e:
            logger.warning(f"TTS warmup failed: {e}")

    async def warmup_async(self) -> bool:
        """Async warmup with better error handling."""
        if self._warmup_done:
            return True
            
        try:
            await asyncio.to_thread(self._lazy_load)
            # Warm up with minimal text
            await asyncio.to_thread(self._tts.tts, "Test.", speed=self.config.speed)
            self._warmup_done = True
            logger.info("TTS warmup completed successfully")
            return True
        except Exception as e:
            logger.warning(f"TTS warmup failed: {e}")
            return False

    def _lazy_load(self):
        """Optimized model loading with better error context."""
        if self._tts is not None:
            return
            
        try:
            from TTS.api import TTS
            self._tts = TTS(
                self.config.model_name, 
                gpu=(self.config.device != "cpu"),
                progress_bar=False
            ).to(self.config.device)
            
            # Detect actual sample rate more robustly
            self._model_sr = self._detect_sample_rate()
            logger.info(f"Loaded TTS model @ {self._model_sr}Hz on {self.config.device}")
            
        except ImportError as e:
            raise RuntimeError(f"TTS library not available: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load TTS model: {e}") from e

    def _detect_sample_rate(self) -> int:
        """Robust sample rate detection."""
        candidates = [
            lambda: self._tts.synthesizer.output_sample_rate,
            lambda: self._tts.output_sample_rate,
            lambda: self._tts.synthesizer.tts_config.audio.sample_rate,
            lambda: 22050  # fallback
        ]
        
        for getter in candidates:
            try:
                sr = getter()
                if isinstance(sr, int) and sr > 0:
                    return sr
            except (AttributeError, TypeError):
                continue
        
        return 22050

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Main streaming interface with circuit breaker pattern."""
        if not text or not text.strip():
            return
            
        # Circuit breaker check
        if self._is_circuit_open():
            logger.warning("TTS circuit breaker is open, skipping synthesis")
            return
            
        try:
            self._lazy_load()
            
            # Preprocess text once
            clean_text = self.text_processor.preprocess_text(text)
            if not clean_text:
                return
            
            # Split into optimal chunks
            chunks = self.text_processor.split_into_chunks(
                clean_text, self.config.chunk_max_length
            )
            
            logger.debug(f"Synthesizing {len(chunks)} chunks: {clean_text[:100]}...")
            
            # Process chunks with optimized pipeline
            async for chunk_bytes in self._process_chunks(chunks):
                yield chunk_bytes
                
            # Reset error count on success
            self._error_count = 0
            
        except Exception as e:
            self._handle_error(e)
            logger.error(f"Stream synthesis failed: {e}")

    async def _process_chunks(self, chunks: List[str]) -> AsyncIterator[bytes]:
        """Optimized chunk processing pipeline."""
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            try:
                # Synthesize audio
                wav = await asyncio.to_thread(
                    self._synthesize_with_retry, chunk, self.config.max_retries
                )
                
                if wav is None or len(wav) == 0:
                    continue
                
                # Stream processed audio
                async for audio_bytes in self._stream_audio(wav):
                    yield audio_bytes
                
                # Add inter-chunk pause
                if i < len(chunks) - 1:
                    yield self._generate_silence(frames=2)
                    
            except Exception as e:
                logger.error(f"Error processing chunk '{chunk[:30]}...': {e}")
                continue

    def _synthesize_with_retry(self, text: str, max_retries: int) -> Optional[np.ndarray]:
        """Optimized synthesis with intelligent retry logic."""
        original_text = text
        
        for attempt in range(max_retries + 1):
            try:
                wav = self._tts.tts(text=text, speed=self.config.speed)
                
                if wav is None:
                    continue
                    
                wav = np.asarray(wav, dtype=np.float32)
                
                # Quality validation
                if len(wav) == 0 or np.all(wav == 0) or np.max(np.abs(wav)) < 0.001:
                    continue
                    
                return wav
                
            except Exception as e:
                if attempt < max_retries:
                    # Progressively simplify text
                    text = re.sub(r'[^\w\s.,!?-]', '', text) if attempt == 0 else text[:len(text)//2]
                    continue
                else:
                    logger.error(f"Synthesis failed after {max_retries} retries: {e}")
                    return None
        
        return None

    async def _stream_audio(self, wav: np.ndarray) -> AsyncIterator[bytes]:
        """Optimized audio streaming with proper async yielding."""
        try:
            # Resample if needed
            if self._model_sr != self.config.target_sample_rate:
                wav = await asyncio.to_thread(
                    self.audio_processor.resample_audio, wav, self._model_sr
                )
            
            # Normalize audio
            wav = self.audio_processor.normalize_audio(wav)
            
            # Stream in frames
            frame_size = self.config.frame_size
            total_frames = len(wav)
            yield_counter = 0
            
            for i in range(0, total_frames, frame_size):
                chunk = wav[i:i + frame_size]
                
                # Pad incomplete frames
                if len(chunk) < frame_size:
                    chunk = np.pad(chunk, (0, frame_size - len(chunk)), 'constant')
                
                yield chunk.astype(np.float32).tobytes()
                
                # Yield control periodically for better responsiveness
                yield_counter += 1
                if yield_counter % 5 == 0:  # Every ~100ms
                    await asyncio.sleep(0.001)
                    
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")

    def _generate_silence(self, frames: int) -> bytes:
        """Generate silence frames efficiently."""
        silence = np.zeros(self.config.frame_size * frames, dtype=np.float32)
        return silence.tobytes()

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
        logger.error(f"TTS error (count: {self._error_count}): {error}")

    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        self._error_count = 0
        logger.info("TTS circuit breaker reset")

    async def cleanup(self):
        """Cleanup resources."""
        if self._tts is not None:
            try:
                # If the TTS object has a cleanup method, call it
                if hasattr(self._tts, 'cleanup'):
                    await asyncio.to_thread(self._tts.cleanup)
                elif hasattr(self._tts, 'close'):
                    await asyncio.to_thread(self._tts.close)
            except Exception as e:
                logger.warning(f"Error during TTS cleanup: {e}")
            finally:
                self._tts = None
                self._warmup_done = False
                logger.info("TTS resources cleaned up")


# Factory function for easy instantiation
def create_tts_streaming(
    model_name: str = "tts_models/en/ljspeech/fast_pitch",
    device: str = "cpu",
    speed: float = 1.2,
    **kwargs
) -> CoquiTTSStreaming:
    """Create a TTS streaming instance with custom configuration."""
    config = TTSConfig(
        model_name=model_name,
        device=device,
        speed=speed,
        **kwargs
    )
    return CoquiTTSStreaming(config=config)