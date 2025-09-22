# plugins/stt/wav2vec2_adapter.py - FIXED BUFFER ISSUE
from typing import AsyncIterator
import asyncio
import numpy as np
import torch
import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from core.interfaces import STTEngine

class Wav2Vec2Streaming(STTEngine):

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "cpu",
        decode_window_s: float = 1.0,
        decode_hop_ms: int = 300,
        min_emit_delta_chars: int = 1,
    ):
        print(f"[Wav2Vec2] Loading model: {model_name}")
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
        self.device = device

        self.decode_window_s = decode_window_s
        self.decode_hop_ms = decode_hop_ms
        self.min_emit_delta_chars = min_emit_delta_chars
        
        # FIXED: Calculate exact buffer sizes
        self.max_buffer_bytes = int(self.decode_window_s * 16000) * 2  # 32KB for 1s
        self.hop_bytes = int(16000 * self.decode_hop_ms / 1000) * 2

        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        
        print(f"[Wav2Vec2] Buffer limit: {self.max_buffer_bytes/1000:.1f}KB, Hop: {self.hop_bytes/1000:.1f}KB")

    async def stream(self, pcm16: AsyncIterator[bytes], sample_rate: int) -> AsyncIterator[str]:
        buf = bytearray()
        last_emit = ""
        decode_count = 0
        bytes_since_decode = 0

        async for frame in pcm16:
            if not frame:
                continue
            
            # Add new frame
            buf.extend(frame)
            bytes_since_decode += len(frame)
            
            # CRITICAL FIX: Enforce buffer limit STRICTLY
            if len(buf) > self.max_buffer_bytes:
                # Remove old data from the beginning
                excess = len(buf) - self.max_buffer_bytes
                buf = buf[excess:]
                print(f"[Wav2Vec2] Trimmed buffer: removed {excess} bytes, now {len(buf)} bytes")

            # Decode timing
            if bytes_since_decode >= self.hop_bytes and len(buf) >= self.hop_bytes:
                bytes_since_decode = 0
                decode_count += 1
                
                start_time = time.time()
                text = self._minimal_decode(bytes(buf))
                decode_time = (time.time() - start_time) * 1000
                
                print(f"[Wav2Vec2] Decode #{decode_count}: {decode_time:.0f}ms (buffer: {len(buf)/1000:.1f}KB)")
                
                if text and text != last_emit:
                    last_emit = text
                    yield text

                await asyncio.sleep(0.001)

        # Final decode
        if buf:
            final_text = self._minimal_decode(bytes(buf))
            if final_text and final_text != last_emit:
                yield final_text

    def _minimal_decode(self, pcm_bytes: bytes) -> str:
        """Fixed decode with proper buffer handling."""
        if not pcm_bytes or len(pcm_bytes) < 640:
            return ""

        try:
            if len(pcm_bytes) % 2:
                pcm_bytes = pcm_bytes[:-1]

            # Create writable copy to avoid the warning
            audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
            audio = torch.from_numpy(audio_array).float() / 32768.0
            
            # Process with fixed size
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.decode_window_s * 16000)
            )

            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device)).logits

            pred_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

            return " ".join(text.strip().split()) if text else ""

        except Exception as e:
            print(f"[Wav2Vec2] Decode error: {e}")
            return ""