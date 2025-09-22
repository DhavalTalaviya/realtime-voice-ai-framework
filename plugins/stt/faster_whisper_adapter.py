from typing import AsyncIterator, Optional, Dict, Any
import asyncio
import numpy as np
import time
from core.interfaces import STTEngine

WhisperModel = None

class FasterWhisperSTT(STTEngine):
    """
    Faster-Whisper with real-time capabilities:
    """

    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        lang: Optional[str] = "en",
        decode_window_s: float = 2.0,
        decode_hop_ms: int = 300,
        min_emit_delta_chars: int = 2,
        overlap_s: float = 0.5,
        enable_realtime: bool = True,
        realtime_model: str = "tiny.en",
        realtime_hop_ms: int = 150,  # Faster updates for real-time
        vad_threshold: float = 0.003,  # Voice activity threshold
        silence_timeout_s: float = 1.0,  # Finalize after this much silence
    ):
        global WhisperModel
        if WhisperModel is None:
            from faster_whisper import WhisperModel as _WM
            WhisperModel = _WM

        # Main model for final transcription
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=1
        )
        
        # Real-time model for live updates (if enabled)
        self.realtime_model = None
        if enable_realtime and realtime_model != model_name:
            try:
                self.realtime_model = WhisperModel(
                    realtime_model,
                    device=device,
                    compute_type="int8",  # Always use int8 for speed
                    num_workers=1
                )
                print(f"[Whisper] Loaded real-time model: {realtime_model}")
            except Exception as e:
                print(f"[Whisper] Failed to load real-time model, using main model: {e}")
                self.realtime_model = self.model
        else:
            self.realtime_model = self.model
        
        self.lang = lang
        self.decode_window_s = float(decode_window_s)
        self.decode_hop_ms = int(decode_hop_ms)
        self.min_emit_delta_chars = int(min_emit_delta_chars)
        self.overlap_s = float(overlap_s)
        
        # Real-time settings
        self.enable_realtime = enable_realtime
        self.realtime_hop_ms = int(realtime_hop_ms)
        self.vad_threshold = float(vad_threshold)
        self.silence_timeout_s = float(silence_timeout_s)
        
        # Calculate exact buffer limits
        self.max_buffer_samples = int(self.decode_window_s * 16000)
        self.max_buffer_bytes = self.max_buffer_samples * 2
        self.hop_samples = int(16000 * self.decode_hop_ms / 1000)
        self.hop_bytes = self.hop_samples * 2
        self.realtime_hop_samples = int(16000 * self.realtime_hop_ms / 1000)
        self.realtime_hop_bytes = self.realtime_hop_samples * 2
        self.overlap_samples = int(self.overlap_s * 16000)
        
        print(f"[Whisper] ENHANCED: max={self.max_buffer_bytes/1000:.1f}KB, "
              f"hop={self.hop_bytes/1000:.1f}KB, realtime={self.enable_realtime}")
        
        # Performance tracking
        self._decode_times = []
        self._buffer_sizes = []
        self._last_speech_time = 0
        self._last_realtime_text = ""

    async def stream(self, pcm16: AsyncIterator[bytes], sample_rate: int) -> AsyncIterator[str]:
        assert sample_rate == 16000, "Expect 16 kHz PCM16 frames"
        
        buf = bytearray()
        last_emit = ""
        decode_count = 0
        realtime_count = 0
        bytes_since_decode = 0
        bytes_since_realtime = 0
        last_activity_time = time.time()

        async for frame in pcm16:
            if not frame:
                continue
            
            current_time = time.time()
            
            # Add new audio
            buf.extend(frame)
            bytes_since_decode += len(frame)
            bytes_since_realtime += len(frame)
            
            # Check for voice activity
            if self._has_voice_activity(frame):
                last_activity_time = current_time
                self._last_speech_time = current_time
            
            # Enforce strict buffer size limit
            if len(buf) > self.max_buffer_bytes:
                buf = buf[-self.max_buffer_bytes:]
                if decode_count % 10 == 0:
                    print(f"[Whisper] Buffer trimmed to {len(buf)} bytes")

            # Real-time transcription
            if (self.enable_realtime and 
                bytes_since_realtime >= self.realtime_hop_bytes and 
                len(buf) >= self.realtime_hop_bytes):
                
                bytes_since_realtime = 0
                realtime_count += 1
                
                # Use smaller buffer for real-time (faster processing)
                realtime_buffer_size = min(len(buf), int(1.0 * 16000 * 2))  # 1 second max
                realtime_buf = buf[-realtime_buffer_size:]
                
                if len(realtime_buf) >= 640:  # At least 40ms
                    realtime_text = await self._realtime_decode(bytes(realtime_buf), realtime_count)
                    if (realtime_text and 
                        realtime_text != self._last_realtime_text and 
                        len(realtime_text.strip()) > 1):
                        self._last_realtime_text = realtime_text
                        yield f"[REALTIME] {realtime_text}"

            # Full transcription
            if bytes_since_decode >= self.hop_bytes and len(buf) >= self.hop_bytes:
                bytes_since_decode = 0
                decode_count += 1
                
                self._buffer_sizes.append(len(buf))
                
                start_time = time.time()
                text = await self._fixed_window_decode(bytes(buf), decode_count)
                decode_time = (time.time() - start_time) * 1000
                
                self._decode_times.append(decode_time)
                if len(self._decode_times) > 10:
                    self._decode_times = self._decode_times[-5:]
                
                if decode_count % 5 == 0:
                    avg_time = sum(self._decode_times) / len(self._decode_times)
                    avg_buffer = sum(self._buffer_sizes[-5:]) / min(5, len(self._buffer_sizes))
                    print(f"[Whisper] Decode #{decode_count}: {decode_time:.0f}ms "
                          f"(avg: {avg_time:.0f}ms, buffer: {avg_buffer/1000:.1f}KB)")
                
                if text and self._should_emit_fast(text, last_emit):
                    last_emit = text
                    yield f"[FINAL] {text}"

                await asyncio.sleep(0.001)
            
            # Check for silence timeout (finalize incomplete transcription)
            if (current_time - last_activity_time > self.silence_timeout_s and 
                buf and len(buf) >= 1000):
                
                final_text = await self._fixed_window_decode(bytes(buf), decode_count + 1, final=True)
                if final_text and final_text != last_emit:
                    yield f"[FINAL] {final_text}"
                    last_emit = final_text
                    buf.clear()  # Clear buffer after finalizing
                    last_activity_time = current_time  # Reset timeout

        # Final decode with remaining buffer
        if buf and len(buf) >= 1000:
            final_text = await self._fixed_window_decode(bytes(buf), decode_count + 1, final=True)
            if final_text and final_text != last_emit:
                yield f"[FINAL] {final_text}"

    def _has_voice_activity(self, audio_chunk: bytes) -> bool:
        """Simple voice activity detection based on RMS energy."""
        if len(audio_chunk) < 2:
            return False
        
        try:
            if len(audio_chunk) % 2:
                audio_chunk = audio_chunk[:-1]
            
            audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio ** 2)))
            return rms > self.vad_threshold
        except:
            return False

    async def _realtime_decode(self, pcm: bytes, decode_count: int) -> str:
        """Fast real-time decode with minimal accuracy for live feedback."""
        if not pcm or len(pcm) < 640:
            return ""

        try:
            if len(pcm) % 2:
                pcm = pcm[:-1]

            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Quick quality check
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < self.vad_threshold:
                return ""

            # Ultra-fast transcription with aggressive settings for speed
            segments, info = self.realtime_model.transcribe(
                audio,
                language=self.lang or "en",
                temperature=0.0,
                beam_size=1,
                best_of=1,
                patience=0.5,  # Less patience for speed
                length_penalty=1.0,
                repetition_penalty=1.0,
                log_prob_threshold=-1.5,  # More permissive
                no_speech_threshold=0.6,  # Higher threshold
                compression_ratio_threshold=4.0,  # More permissive
                condition_on_previous_text=False,
                vad_filter=False,
                without_timestamps=True,
                word_timestamps=False,
                suppress_blank=True,
            )

            # Collect segments quickly
            text_parts = []
            for segment in segments:
                if segment.text and len(segment.text.strip()) > 0:
                    clean_text = segment.text.strip()
                    if len(clean_text) > 0:
                        text_parts.append(clean_text)

            if not text_parts:
                return ""

            text = " ".join(text_parts)
            text = " ".join(text.split())  # Normalize whitespace
            
            return text

        except Exception as e:
            print(f"[Whisper] Real-time decode error: {e}")
            return ""

    async def _fixed_window_decode(self, pcm: bytes, decode_count: int, final: bool = False) -> str:
        """Fixed-window decode for final accurate transcription."""
        if not pcm or len(pcm) < 640:
            return ""

        try:
            if len(pcm) % 2:
                pcm = pcm[:-1]

            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Ensure we never exceed the window size
            max_samples = self.max_buffer_samples
            if len(audio) > max_samples:
                audio = audio[-max_samples:]
            
            # Quality check
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < 0.003:
                return ""

            # High-quality transcription with balanced parameters
            segments, info = self.model.transcribe(
                audio,
                language=self.lang or "en",
                temperature=0.0,
                beam_size=1,
                best_of=1,
                patience=1.0,
                length_penalty=1.0,
                repetition_penalty=1.0,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.4 if not final else 0.2,
                compression_ratio_threshold=3.0,
                condition_on_previous_text=False,
                vad_filter=False,
                without_timestamps=True,
                word_timestamps=False,
                suppress_blank=True,
            )

            # Collect segments
            text_parts = []
            for segment in segments:
                if segment.text and len(segment.text.strip()) > 0:
                    clean_text = segment.text.strip()
                    if len(clean_text) > 1:
                        text_parts.append(clean_text)

            if not text_parts:
                return ""

            text = " ".join(text_parts)
            text = " ".join(text.split())
            
            # Basic cleanup
            if text and text[0].islower():
                text = text[0].upper() + text[1:]
                
            return text

        except Exception as e:
            print(f"[Whisper] Decode error: {e}")
            return ""

    def _should_emit_fast(self, new: str, old: str) -> bool:
        """Fast emission logic."""
        if not old:
            return bool(new.strip()) and len(new.strip()) >= 2

        if not new or new == old:
            return False

        # Growth check
        if len(new) - len(old) >= self.min_emit_delta_chars:
            return True

        # Word count check
        old_words = len(old.split())
        new_words = len(new.split())
        
        if new_words > old_words:
            return True

        # Sentence completion
        if new.rstrip().endswith(('.', '!', '?')) and not old.rstrip().endswith(('.', '!', '?')):
            return True

        return False