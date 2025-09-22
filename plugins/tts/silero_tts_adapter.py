# plugins/tts/silero_tts_adapter.py
from typing import AsyncIterator, List, Dict, Optional
import asyncio
import numpy as np
import re

class SileroTTSStreaming:
    
    def __init__(self, model_name: str = "v3_en", voice: str = "en_1", speed: float = 1.15):
        self.model_name = model_name  # e.g., 'v3_en'
        self.voice = voice            # e.g., 'en_0' or 'random'
        self.speed = float(max(0.5, min(speed, 2.5)))  # reasonable speed limits
        
        # internal state
        self._model = None
        self._sr_model = 48000
        self._speakers = None
        self._warmup_done = False
        self._torch_optimized = False

    def warmup(self):
        """Preload model and optimize for faster synthesis."""
        try:
            self._lazy_load()
            # warm up with short synthesis
            test_audio = self._synthesize_raw("Hello world.", "en_0")
            if test_audio is not None:
                self._warmup_done = True
                print(f"[Silero TTS] Warmed up successfully")
        except Exception as e:
            print(f"[Silero TTS] Warmup warning: {e}")

    def _lazy_load(self):
        """Load model with enhanced error handling and optimization."""
        if self._model is not None:
            return
            
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required for Silero TTS")
            
        try:
            import omegaconf  # noqa: F401
        except ModuleNotFoundError as e:
            raise RuntimeError("Install: pip install omegaconf==2.3.0") from e

        try:
            print(f"[Silero TTS] Loading {self.model_name}...")
            
            # Load model with specific parameters for better performance
            ret = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="en",
                speaker=self.model_name,
                trust_repo=True,
                verbose=False,  # reduce loading noise
                force_reload=False  # use cache if available
            )

            # parse return value (different versions return different formats)
            if isinstance(ret, tuple):
                if len(ret) == 4:
                    model, _example, sample_rate, speakers = ret
                elif len(ret) == 2:
                    model, _example = ret
                    sample_rate, speakers = 48000, None
                else:
                    model = ret[0]
                    sample_rate, speakers = 48000, None
            else:
                model, sample_rate, speakers = ret, 48000, None
            
            # device = torch.device('cuda')
            # model.to(device)
            self._model = model
            self._sr_model = int(sample_rate or 48000)
            self._speakers = set(speakers) if speakers is not None else None

            # # optimize torch settings for audio synthesis
            # if not self._torch_optimized:
            #     try:
            #         torch.set_num_threads(min(4, torch.get_num_threads()))  # reasonable thread count
            #         torch.set_num_interop_threads(2)
            #         if torch.backends.mkldnn.is_available():
            #             torch.backends.mkldnn.enabled = True
            #         self._torch_optimized = True
            #     except Exception:
            #         pass

            print(f"[Silero TTS] Loaded @ {self._sr_model}Hz, speakers: {len(self._speakers) if self._speakers else 'unknown'}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero model '{self.model_name}': {e}") from e

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream TTS with enhanced processing and quality."""
        if not text or not text.strip():
            return
            
        self._lazy_load()
        
        # preprocess text for better synthesis
        clean_text = self._preprocess_text(text)
        if not clean_text:
            return
            
        print(f"[Silero TTS] Synthesizing: {clean_text}")
        
        # intelligent chunking for streaming
        chunks = self._split_for_streaming(clean_text)
        
        # select and validate voice
        selected_voice = self._pick_voice(self.voice)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            try:
                # synthesize audio chunk
                wav = await asyncio.to_thread(self._synthesize_raw, chunk, selected_voice)
                
                if wav is None or len(wav) == 0:
                    print(f"[Silero TTS] Empty synthesis for: {chunk[:30]}...")
                    continue
                
                # process with improved speed control
                processed_wav = await asyncio.to_thread(
                    self._apply_speed_control, wav, self.speed
                )
                
                if processed_wav is None:
                    continue
                
                # stream the processed audio
                async for audio_chunk in self._stream_audio_frames(processed_wav):
                    yield audio_chunk
                
                # add natural pause between chunks
                if i < len(chunks) - 1 and len(chunks) > 1:
                    pause_duration = self._calculate_pause_duration(chunk)
                    silence = np.zeros(int(16000 * pause_duration), dtype=np.float32)
                    async for silence_chunk in self._stream_audio_frames(silence):
                        yield silence_chunk
                        
            except Exception as e:
                print(f"[Silero TTS] Error processing chunk '{chunk[:30]}...': {e}")
                continue

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for Silero TTS."""
        if not text:
            return ""
            
        # normalize whitespace
        text = " ".join(text.split())
        
        # handle common abbreviations and symbols for better pronunciation
        replacements = {
            # titles and honorifics
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Misses', 
            r'\bMs\.': 'Miss',
            r'\bProf\.': 'Professor',
            
            # addresses
            r'\bSt\.': 'Street',
            r'\bAve\.': 'Avenue',
            r'\bBlvd\.': 'Boulevard', 
            r'\bRd\.': 'Road',
            r'\bCt\.': 'Court',
            r'\bDr\.(?=\s)': 'Drive',  # when used as street type
            
            # technical terms - spell out for clarity
            r'\bAPI\b': 'A P I',
            r'\bUI\b': 'U I', 
            r'\bURL\b': 'U R L',
            r'\bHTTP\b': 'H T T P',
            r'\bHTML\b': 'H T M L',
            r'\bCSS\b': 'C S S',
            r'\bSQL\b': 'S Q L',
            r'\bJSON\b': 'J S O N',
            r'\bXML\b': 'X M L',
            r'\bPDF\b': 'P D F',
            
            # symbols and special characters
            r'&': ' and ',
            r'@': ' at ',
            r'#': ' hash ',
            r'%': ' percent',
            r'\+': ' plus ',
            r'=': ' equals ',
            r'<': ' less than ',
            r'>': ' greater than ',
            
            # currency
            r'\$(\d+)': r'\1 dollars',
            r'€(\d+)': r'\1 euros',
            r'£(\d+)': r'\1 pounds',
            
            # common contractions (Silero handles these better expanded)
            r"won't": 'will not',
            r"can't": 'cannot',
            r"n't": ' not',
            r"'re": ' are',
            r"'ve": ' have',
            r"'ll": ' will',
            r"'d": ' would',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # ensure proper sentence ending for natural prosody
        if text and not text.rstrip().endswith(('.', '!', '?', ';', ':')):
            # add appropriate punctuation based on context
            if '?' in text or any(word in text.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                text = text.rstrip() + '?'
            elif '!' in text or any(word in text.lower() for word in ['wow', 'great', 'amazing', 'excellent']):
                text = text.rstrip() + '!'
            else:
                text = text.rstrip() + '.'
        
        return text

    def _split_for_streaming(self, text: str) -> List[str]:
        """Intelligent text chunking optimized for streaming latency."""
        if len(text) <= 80:  # short text - no need to split
            return [text]
        
        # primary split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
        
        if not sentences:
            return [text]
        
        # group sentences into optimal chunks for streaming
        chunks = []
        current_chunk = ""
        target_chunk_size = 150  # optimal for Silero latency vs quality
        
        for sentence in sentences:
            # if this sentence alone is long, try to split it further
            if len(sentence) > target_chunk_size * 1.5:
                # finalize current chunk if any
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # try to split long sentence on clause boundaries
                clause_splits = re.split(r'(?<=[,;:])\s+', sentence)
                if len(clause_splits) > 1:
                    temp_chunk = ""
                    for clause in clause_splits:
                        if temp_chunk and len(temp_chunk + clause) > target_chunk_size:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = clause
                        else:
                            temp_chunk = temp_chunk + " " + clause if temp_chunk else clause
                    if temp_chunk.strip():
                        current_chunk = temp_chunk.strip()
                else:
                    # can't split further, use as is
                    chunks.append(sentence)
                continue
            
            # check if adding this sentence exceeds target size
            if current_chunk and len(current_chunk + " " + sentence) > target_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def _synthesize_raw(self, text: str, voice: str) -> Optional[np.ndarray]:
        """Raw synthesis with enhanced error handling."""
        if not text.strip():
            return None
            
        try:
            # synthesize with Silero
            wav = self._model.apply_tts(
                text=text,
                speaker=voice,
                sample_rate=self._sr_model,
                put_accent=True,   # better pronunciation
                put_yo=True,       # better Russian support if applicable
            )
            
            # convert to numpy and validate
            wav = np.asarray(wav, dtype=np.float32)
            
            if len(wav) == 0:
                return None
                
            # basic quality checks
            if np.all(wav == 0):  # all silence
                return None
            if np.max(np.abs(wav)) < 0.001:  # too quiet
                return None
                
            return wav
            
        except Exception as e:
            print(f"[Silero TTS] Synthesis error: {e}")
            return None

    def _apply_speed_control(self, wav: np.ndarray, speed: float) -> Optional[np.ndarray]:
        """Apply pitch-preserving speed control using advanced resampling."""
        if speed == 1.0 or wav is None or len(wav) == 0:
            return self._resample_to_16k(wav)
            
        try:
            # first resample to 16kHz
            wav_16k = self._resample_to_16k(wav)
            if wav_16k is None:
                return None
                
            # apply pitch-preserving speed change
            if speed != 1.0:
                wav_16k = self._pitch_preserving_speed_change(wav_16k, speed)
                
            return wav_16k
            
        except Exception as e:
            print(f"[Silero TTS] Speed control error: {e}")
            # fallback to basic resampling
            return self._resample_to_16k(wav)

    def _pitch_preserving_speed_change(self, wav: np.ndarray, speed: float) -> np.ndarray:
        """Pitch-preserving time stretching using overlap-add method."""
        if speed == 1.0:
            return wav
            
        try:
            # try to use librosa for high-quality time stretching
            import librosa
            stretched = librosa.effects.time_stretch(wav, rate=speed)
            return stretched.astype(np.float32)
        except ImportError:
            pass
            
        # fallback: simple overlap-add PSOLA-like algorithm
        return self._simple_time_stretch(wav, speed)

    def _simple_time_stretch(self, wav: np.ndarray, speed: float) -> np.ndarray:
        """Simple time stretching using overlap-add."""
        if speed == 1.0:
            return wav
            
        frame_length = 1024  # ~64ms at 16kHz
        hop_length = frame_length // 4
        
        # calculate output length
        output_length = int(len(wav) / speed)
        output = np.zeros(output_length, dtype=np.float32)
        
        # simple overlap-add time stretching
        input_hop = hop_length * speed
        output_pos = 0
        input_pos = 0.0
        
        window = np.hanning(frame_length)
        
        while output_pos + frame_length < output_length and input_pos + frame_length < len(wav):
            # extract frame from input
            start_idx = int(input_pos)
            if start_idx + frame_length >= len(wav):
                break
                
            frame = wav[start_idx:start_idx + frame_length] * window
            
            # overlap-add to output
            end_pos = min(output_pos + frame_length, output_length)
            frame_end = end_pos - output_pos
            output[output_pos:end_pos] += frame[:frame_end]
            
            # advance positions
            output_pos += hop_length
            input_pos += input_hop
        
        return output

    def _resample_to_16k(self, wav: np.ndarray) -> Optional[np.ndarray]:
        """High-quality resampling to 16kHz."""
        if wav is None or len(wav) == 0:
            return None
            
        if self._sr_model == 16000:
            return wav.astype(np.float32)
            
        try:
            import resampy
            resampled = resampy.resample(
                wav, self._sr_model, 16000, 
                filter='kaiser_best',
                parallel=False
            )
            return resampled.astype(np.float32)
        except ImportError:
            # fallback resampling
            ratio = 16000 / self._sr_model
            new_length = int(len(wav) * ratio)
            indices = np.linspace(0, len(wav) - 1, new_length)
            resampled = np.interp(indices, np.arange(len(wav)), wav)
            return resampled.astype(np.float32)

    async def _stream_audio_frames(self, wav: np.ndarray) -> AsyncIterator[bytes]:
        """Stream audio in properly timed frames."""
        if wav is None or len(wav) == 0:
            return
            
        # normalize audio levels
        wav = self._normalize_audio(wav)
        
        # stream in 20ms frames (320 samples at 16kHz)
        frame_size = 320
        
        for i in range(0, len(wav), frame_size):
            chunk = wav[i:i + frame_size]
            
            # pad incomplete final frame
            if len(chunk) < frame_size:
                padding = np.zeros(frame_size - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, padding])
            
            yield chunk.astype(np.float32).tobytes()
            
            # yield periodically for responsiveness
            if i % (frame_size * 10) == 0:  # every ~200ms
                await asyncio.sleep(0.001)

    def _normalize_audio(self, wav: np.ndarray) -> np.ndarray:
        """Intelligent audio normalization for consistent levels."""
        if len(wav) == 0:
            return wav
            
        # calculate audio statistics
        rms = np.sqrt(np.mean(wav ** 2))
        peak = np.max(np.abs(wav))
        
        if peak < 1e-6:  # essentially silence
            return wav
            
        # target levels for good playback
        target_rms = 0.12
        target_peak = 0.75
        
        # choose normalization approach
        if peak > 0.9:  # prevent clipping
            wav = wav * (target_peak / peak)
        elif rms < 0.03:  # boost quiet audio
            boost = min(target_rms / rms, target_peak / peak)
            wav = wav * boost
        elif rms > 0.25:  # reduce loud audio
            wav = wav * (target_rms / rms)
            
        # soft limiting to prevent artifacts
        wav = np.tanh(wav * 1.2) * 0.85
        
        return wav.astype(np.float32)

    def _calculate_pause_duration(self, text: str) -> float:
        """Calculate natural pause duration based on text content."""
        text = text.strip()
        if not text:
            return 0.2
            
        # longer pauses for sentence endings
        if text.endswith('.'):
            return 0.4
        elif text.endswith('!') or text.endswith('?'):
            return 0.5
        elif text.endswith(',') or text.endswith(';'):
            return 0.25
        elif text.endswith(':'):
            return 0.3
        else:
            return 0.2  # default pause

    def _pick_voice(self, requested: str) -> str:
        """Select and validate voice with better fallback logic."""
        if not self._speakers:
            # no speaker list available, trust the request
            return requested if requested != "random" else "en_0"
            
        # handle random voice selection
        if requested == "random":
            import random
            available = [s for s in self._speakers if s.startswith("en_")]
            if available:
                return random.choice(available)
            return "en_0" if "en_0" in self._speakers else next(iter(self._speakers))
        
        # validate requested voice
        if requested in self._speakers:
            return requested
            
        # fallback logic
        # try to find similar voice (same language)
        if requested.startswith("en_"):
            candidates = [s for s in self._speakers if s.startswith("en_")]
            if candidates:
                return candidates[0]
                
        # try default voices
        defaults = ["en_0", "en_1", "en_2"]
        for default in defaults:
            if default in self._speakers:
                return default
                
        # last resort: use first available
        return next(iter(self._speakers))

def _split_sentences(text: str) -> List[str]:
    """Enhanced sentence splitting utility function."""
    if not text.strip():
        return []
        
    # improved sentence boundary detection
    pattern = r'(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|St|Ave|Blvd|Rd|Ct|vs|etc|i\.e|e\.g)\.)\s*(?<=[.!?])\s+'
    sentences = re.split(pattern, text.strip(), flags=re.IGNORECASE)
    
    # clean and validate sentences
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 2:  # minimum meaningful length
            # ensure proper punctuation
            if not sentence[-1] in '.!?':
                sentence += '.'
            result.append(sentence)
    
    return result