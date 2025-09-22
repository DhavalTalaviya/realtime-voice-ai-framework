# core/interfaces.py
from typing import AsyncIterator, Protocol, Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod

class STTEngine(Protocol):
    """Speech-to-Text engine interface with streaming support."""
    
    async def stream(self, pcm16: AsyncIterator[bytes], sample_rate: int) -> AsyncIterator[str]:
        """
        Stream transcription results from PCM16 audio frames.
        
        Args:
            pcm16: Async iterator of PCM16 audio bytes (mono, 16kHz typically)
            sample_rate: Sample rate of the audio (usually 16000)
            
        Yields:
            str: Partial transcription results as they become available
        """
        ...

class LLMEngine(Protocol):
    """Large Language Model engine interface with streaming support."""
    
    async def stream_reply(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """
        Generate streaming response from conversation messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Yields:
            str: Token chunks as they are generated
        """
        ...

class TTSEngine(Protocol):
    """Text-to-Speech engine interface with streaming audio output."""
    
    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Generate streaming audio from text input.
        
        Args:
            text: Text to synthesize
            
        Yields:
            bytes: Audio chunks (typically float32 PCM at 16kHz)
        """
        ...
    
    def warmup(self) -> None:
        """
        Optional warmup method to preload models for faster synthesis.
        Should be called once during initialization.
        """
        ...

class VADEngine(Protocol):
    """Voice Activity Detection engine interface."""
    
    def is_speech(self, pcm16: bytes, sample_rate: int) -> bool:
        """
        Detect if audio frame contains speech.
        
        Args:
            pcm16: PCM16 audio bytes for a single frame
            sample_rate: Sample rate of the audio
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        ...

class AudioProcessor(Protocol):
    """Audio preprocessing interface for noise reduction and enhancement."""
    
    def process_frame(self, pcm16: bytes, sample_rate: int) -> bytes:
        """
        Process a single audio frame for enhancement.
        
        Args:
            pcm16: Input PCM16 audio frame
            sample_rate: Sample rate of the audio
            
        Returns:
            bytes: Processed PCM16 audio frame
        """
        ...

class CancelToken:
    """
    Thread-safe cancellation token for interrupting operations.
    Used primarily for TTS barge-in functionality.
    """
    
    def __init__(self) -> None:
        self._cancelled = False
        self._callbacks: List[Callable[[], None]] = []
    
    def cancel(self) -> None:
        """Mark as cancelled and trigger any registered callbacks."""
        if not self._cancelled:
            self._cancelled = True
            # Execute callbacks for cleanup
            for callback in self._callbacks:
                try:
                    callback()
                except Exception:
                    pass  # ignore callback errors
    
    @property
    def cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled
    
    def reset(self) -> None:
        """Reset cancellation state."""
        self._cancelled = False
    
    def add_cancel_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to execute when cancelled."""
        self._callbacks.append(callback)

class SessionStats:
    """Statistics tracking for voice sessions."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.utterances_processed = 0
        self.total_audio_duration_s = 0.0
        self.total_synthesis_duration_s = 0.0
        self.interruptions_count = 0
        self.average_response_latency_ms = 0.0
        self.errors_count = 0
        self._response_times = []
    
    def record_utterance(self, duration_s: float) -> None:
        """Record a processed utterance."""
        self.utterances_processed += 1
        self.total_audio_duration_s += duration_s
    
    def record_synthesis(self, duration_s: float) -> None:
        """Record synthesis duration."""
        self.total_synthesis_duration_s += duration_s
    
    def record_interruption(self) -> None:
        """Record a barge-in interruption."""
        self.interruptions_count += 1
    
    def record_response_time(self, latency_ms: float) -> None:
        """Record response latency."""
        self._response_times.append(latency_ms)
        if self._response_times:
            self.average_response_latency_ms = sum(self._response_times) / len(self._response_times)
    
    def record_error(self) -> None:
        """Record an error occurrence."""
        self.errors_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        return {
            "utterances_processed": self.utterances_processed,
            "total_audio_duration_s": round(self.total_audio_duration_s, 2),
            "total_synthesis_duration_s": round(self.total_synthesis_duration_s, 2),
            "interruptions_count": self.interruptions_count,
            "average_response_latency_ms": round(self.average_response_latency_ms, 1),
            "errors_count": self.errors_count,
        }

# Type aliases for common patterns
MessageList = List[Dict[str, str]]
AudioCallback = Callable[[bytes], Any]
ErrorCallback = Callable[[Exception], None]