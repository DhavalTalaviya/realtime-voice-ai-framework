# plugins/stt/groq_whisper_adapter.py
from typing import AsyncIterator
import asyncio
import numpy as np
import time
import io
import wave
from dataclasses import dataclass
from collections import deque
from core.interfaces import STTEngine

# Try to import aiohttp, fallback to httpx if needed
try:
    import aiohttp
    USE_AIOHTTP = True
except ImportError:
    import httpx
    USE_AIOHTTP = False

@dataclass
class TranscriptionRequest:
    """Request wrapper for Groq API calls."""
    audio_data: bytes
    timestamp: float
    is_final: bool = False
    request_id: int = 0

class GroqWhisperSTT(STTEngine):
    """
    Groq Whisper API adapter with real-time capabilities.
    Fixed multipart/form-data handling.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "whisper-large-v3",
        decode_window_s: float = 2.0,
        decode_hop_ms: int = 300,
        min_emit_delta_chars: int = 2,
        overlap_s: float = 0.5,
        enable_realtime: bool = True,
        realtime_hop_ms: int = 200,
        vad_threshold: float = 0.003,
        silence_timeout_s: float = 1.0,
        # Groq-specific settings
        temperature: float = 0.0,
        language: str = "en",
        # API optimization
        max_concurrent_requests: int = 3,
        request_timeout_s: float = 10.0,
        retry_attempts: int = 2,
        batch_delay_ms: int = 50,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        
        # Transcription settings
        self.decode_window_s = float(decode_window_s)
        self.decode_hop_ms = int(decode_hop_ms)
        self.min_emit_delta_chars = int(min_emit_delta_chars)
        self.overlap_s = float(overlap_s)
        
        # Real-time settings
        self.enable_realtime = enable_realtime
        self.realtime_hop_ms = int(realtime_hop_ms)
        self.vad_threshold = float(vad_threshold)
        self.silence_timeout_s = float(silence_timeout_s)
        
        # Groq API settings
        self.temperature = temperature
        self.language = language
        
        # API optimization
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout_s = request_timeout_s
        self.retry_attempts = retry_attempts
        self.batch_delay_ms = batch_delay_ms
        
        # Calculate buffer sizes
        self.max_buffer_samples = int(self.decode_window_s * 16000)
        self.max_buffer_bytes = self.max_buffer_samples * 2
        self.hop_samples = int(16000 * self.decode_hop_ms / 1000)
        self.hop_bytes = self.hop_samples * 2
        self.realtime_hop_samples = int(16000 * self.realtime_hop_ms / 1000)
        self.realtime_hop_bytes = self.realtime_hop_samples * 2
        
        # Request management
        self.active_requests = 0
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.last_api_call = 0
        
        # Performance tracking
        self._api_times = []
        self._buffer_sizes = []
        self._last_speech_time = 0
        self._last_realtime_text = ""
        self._client = None
        
        # Text cache for deduplication
        self._recent_texts = deque(maxlen=5)
        
        print(f"[GroqWhisper] Initialized: model={model_name}, "
              f"window={decode_window_s}s, realtime={enable_realtime}")

    async def _get_client(self):
        """Get or create HTTP client."""
        if USE_AIOHTTP:
            if self._client is None or self._client.closed:
                timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
                connector = aiohttp.TCPConnector(limit=10, force_close=True)
                # Don't set any default headers here
                self._client = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector
                )
            return self._client
        else:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=self.request_timeout_s,
                    limits=httpx.Limits(max_connections=10)
                )
            return self._client

    async def stream(self, pcm16: AsyncIterator[bytes], sample_rate: int) -> AsyncIterator[str]:
        assert sample_rate == 16000, "Expect 16 kHz PCM16 frames"
        
        buf = bytearray()
        last_emit = ""
        decode_count = 0
        realtime_count = 0
        bytes_since_decode = 0
        bytes_since_realtime = 0
        last_activity_time = time.time()

        try:
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
                
                # Enforce buffer size limit
                if len(buf) > self.max_buffer_bytes:
                    buf = buf[-self.max_buffer_bytes:]

                # Real-time transcription
                if (self.enable_realtime and 
                    bytes_since_realtime >= self.realtime_hop_bytes and 
                    len(buf) >= self.realtime_hop_bytes):
                    
                    bytes_since_realtime = 0
                    realtime_count += 1
                    
                    # Use smaller buffer for real-time
                    realtime_buffer_size = min(len(buf), int(1.0 * 16000 * 2))
                    realtime_buf = buf[-realtime_buffer_size:]
                    
                    if len(realtime_buf) >= 3200:  # At least 200ms
                        text = await self._transcribe_audio(bytes(realtime_buf), is_realtime=True)
                        if text and text != self._last_realtime_text and len(text.strip()) > 1:
                            self._last_realtime_text = text
                            yield f"[REALTIME] {text}"

                # Full transcription
                if bytes_since_decode >= self.hop_bytes and len(buf) >= self.hop_bytes:
                    bytes_since_decode = 0
                    decode_count += 1
                    
                    self._buffer_sizes.append(len(buf))
                    
                    start_time = time.time()
                    text = await self._transcribe_audio(bytes(buf), is_realtime=False)
                    api_time = (time.time() - start_time) * 1000
                    
                    self._api_times.append(api_time)
                    if len(self._api_times) > 10:
                        self._api_times = self._api_times[-5:]
                    
                    if decode_count % 5 == 0:
                        avg_time = sum(self._api_times) / len(self._api_times)
                        print(f"[GroqWhisper] Decode #{decode_count}: {api_time:.0f}ms (avg: {avg_time:.0f}ms)")
                    
                    if text and self._should_emit_fast(text, last_emit):
                        last_emit = text
                        yield f"[FINAL] {text}"

                    await asyncio.sleep(0.001)
                
                # Check for silence timeout
                if (current_time - last_activity_time > self.silence_timeout_s and 
                    buf and len(buf) >= 3200):
                    
                    final_text = await self._transcribe_audio(bytes(buf), is_realtime=False)
                    if final_text and final_text != last_emit:
                        yield f"[FINAL] {final_text}"
                        last_emit = final_text
                        buf.clear()
                        last_activity_time = current_time

            # Final decode
            if buf and len(buf) >= 3200:
                final_text = await self._transcribe_audio(bytes(buf), is_realtime=False)
                if final_text and final_text != last_emit:
                    yield f"[FINAL] {final_text}"
                    
        finally:
            await self._cleanup()

    async def _transcribe_audio(self, audio_data: bytes, is_realtime: bool = False) -> str:
        """Main transcription method with proper multipart handling."""
        if not audio_data or len(audio_data) < 3200:
            return ""
        
        async with self.request_semaphore:
            # Rate limiting
            current_time = time.time()
            time_since_last = (current_time - self.last_api_call) * 1000
            if time_since_last < self.batch_delay_ms:
                await asyncio.sleep((self.batch_delay_ms - time_since_last) / 1000)
            self.last_api_call = time.time()
            
            # Convert PCM to WAV
            wav_data = self._pcm_to_wav(audio_data)
            
            # Retry logic
            for attempt in range(self.retry_attempts):
                try:
                    if USE_AIOHTTP:
                        result = await self._call_with_aiohttp(wav_data, is_realtime)
                    else:
                        result = await self._call_with_httpx(wav_data, is_realtime)
                    
                    if result:
                        # Post-process and deduplicate
                        text = self._post_process_text(result)
                        if text and text not in self._recent_texts:
                            self._recent_texts.append(text)
                            return text
                    return ""
                    
                except Exception as e:
                    print(f"[GroqWhisper] API error (attempt {attempt + 1}): {e}")
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
            
            return ""

    async def _call_with_aiohttp(self, wav_data: bytes, is_realtime: bool) -> str:
        """Call API using aiohttp with proper multipart."""
        client = await self._get_client()
        
        # Create writer for multipart - THIS IS THE KEY FIX
        writer = aiohttp.MultipartWriter('form-data')
        
        # Add the file part
        file_part = writer.append(wav_data)
        file_part.set_content_disposition('form-data', name='file', filename='audio.wav')
        file_part.headers['Content-Type'] = 'audio/wav'
        
        # Add model field
        model_part = writer.append(self.model_name)
        model_part.set_content_disposition('form-data', name='model')
        
        # Add temperature field
        temp_part = writer.append(str(self.temperature))
        temp_part.set_content_disposition('form-data', name='temperature')
        
        # Add language field
        lang_part = writer.append(self.language)
        lang_part.set_content_disposition('form-data', name='language')
        
        # Add response format (text for simplicity)
        format_part = writer.append('text')
        format_part.set_content_disposition('form-data', name='response_format')
        
        # Make request with only Authorization header
        async with client.post(
            self.base_url,
            data=writer,
            headers={'Authorization': f'Bearer {self.api_key}'}
        ) as response:
            if response.status == 200:
                text = await response.text()
                return text.strip()
            elif response.status == 429:
                wait_time = float(response.headers.get('Retry-After', 1))
                print(f"[GroqWhisper] Rate limited, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                return ""
            else:
                error_text = await response.text()
                print(f"[GroqWhisper] API error {response.status}: {error_text}")
                return ""

    async def _call_with_httpx(self, wav_data: bytes, is_realtime: bool) -> str:
        """Call API using httpx as fallback."""
        client = await self._get_client()
        
        # Create files and data dictionaries for httpx
        files = {
            'file': ('audio.wav', io.BytesIO(wav_data), 'audio/wav')
        }
        
        data = {
            'model': self.model_name,
            'temperature': str(self.temperature),
            'language': self.language,
            'response_format': 'text'
        }
        
        response = await client.post(
            self.base_url,
            files=files,
            data=data,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        
        if response.status_code == 200:
            return response.text.strip()
        elif response.status_code == 429:
            wait_time = float(response.headers.get('Retry-After', 1))
            print(f"[GroqWhisper] Rate limited, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
            return ""
        else:
            print(f"[GroqWhisper] API error {response.status_code}: {response.text}")
            return ""

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert PCM16 data to WAV format."""
        if len(pcm_data) % 2:
            pcm_data = pcm_data[:-1]
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(pcm_data)
        
        wav_buffer.seek(0)
        return wav_buffer.read()

    def _has_voice_activity(self, audio_chunk: bytes) -> bool:
        """Simple voice activity detection."""
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

    def _post_process_text(self, text: str) -> str:
        """Post-process transcribed text."""
        if not text:
            return ""
        
        # Clean up
        text = " ".join(text.strip().split())
        
        # Capitalize
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Fix common issues
        text = text.replace(" i ", " I ")
        text = text.replace(" i'm ", " I'm ")
        text = text.replace(" i'll ", " I'll ")
        text = text.replace(" i've ", " I've ")
        text = text.replace(" i'd ", " I'd ")
        
        return text

    def _should_emit_fast(self, new: str, old: str) -> bool:
        """Emission logic."""
        if not old:
            return bool(new.strip()) and len(new.strip()) >= 2

        if not new or new == old:
            return False

        if len(new) - len(old) >= self.min_emit_delta_chars:
            return True

        old_words = len(old.split())
        new_words = len(new.split())
        
        if new_words > old_words:
            return True

        if new.rstrip().endswith(('.', '!', '?')) and not old.rstrip().endswith(('.', '!', '?')):
            return True

        return False

    async def _cleanup(self):
        """Cleanup resources."""
        if self._client:
            if USE_AIOHTTP and not self._client.closed:
                await self._client.close()
            elif not USE_AIOHTTP:
                await self._client.aclose()
            self._client = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        if self._session and not self._session.closed:
            try:
                asyncio.create_task(self._session.close())
            except:
                pass