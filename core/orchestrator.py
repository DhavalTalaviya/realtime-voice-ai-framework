# core/orchestrator.py - Real-time version
import asyncio
import contextlib
import time
from typing import AsyncIterator, List, Dict, Callable, Optional
from .interfaces import STTEngine, LLMEngine, TTSEngine, VADEngine, CancelToken, SessionStats

class VoiceOrchestrator:
    """
    Real-time streaming voice orchestrator with live transcription during speech.
    """
    
    def __init__(
        self,
        stt: STTEngine,
        llm: LLMEngine,
        tts: TTSEngine,
        vad: Optional[VADEngine] = None,
        frame_ms: int = 20,
        gap_ms: int = 600,
        max_utterance_s: float = 30.0,
        min_utterance_s: float = 0.5,
        silence_padding_ms: int = 200,
        enable_barge_in: bool = True,
        enable_realtime_stt: bool = True,  # Enable real-time transcription
        speech_threshold_frames: int = 5,
    ):
        # Core engines
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vad = vad
        
        # Timing configuration
        self.frame_ms = int(frame_ms)
        self.gap_ms = int(gap_ms)
        self.max_utterance_s = float(max_utterance_s)
        self.min_utterance_s = float(min_utterance_s)
        self.silence_padding_ms = int(silence_padding_ms)
        self.enable_barge_in = bool(enable_barge_in)
        self.enable_realtime_stt = bool(enable_realtime_stt)
        self.speech_threshold_frames = int(speech_threshold_frames)
        
        # Conversation state
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful and concise voice assistant. Keep responses brief and conversational."}
        ]
        
        # Session tracking
        self.stats = SessionStats()
        self._current_state = "IDLE"
        self._session_start_time = 0.0
        
        # Audio state tracking
        self._tts_start_time = 0.0
        self._last_tts_frame_time = 0.0
        
        # Real-time STT state
        self._current_stt_stream = None
        self._stt_audio_queue = None
        self._latest_transcription = ""

    async def run_session(
        self,
        pcm_iter: AsyncIterator[bytes],
        sample_rate: int,
        out_audio_cb: Callable[[bytes], "asyncio.Future[None]"],
        error_callback: Optional[Callable[[Exception], None]] = None,
        transcription_callback: Optional[Callable[[str, bool], None]] = None,  # For real-time updates
    ) -> None:
        """Run a complete voice interaction session with real-time transcription."""
        self._session_start_time = time.time()
        self.stats.reset()
        
        frame_ms = self.frame_ms
        gap_frames = max(1, self.gap_ms // frame_ms)
        max_utterance_frames = int(self.max_utterance_s * 1000 // frame_ms)
        min_utterance_frames = int(self.min_utterance_s * 1000 // frame_ms)
        padding_frames = max(1, self.silence_padding_ms // frame_ms)

        # Shared session state
        class SessionState:
            def __init__(self):
                self.mode = "LISTEN"
                self.silence_count = 0
                self.utterance_frames = 0
                self.speech_frames = 0
                self.cancel_token = CancelToken()
                self.utterance_event = asyncio.Event()
                self.turn_complete_event = asyncio.Event()
                self.tts_active = False
                self.processing_turn = False
                # Real-time STT state
                self.speech_detected = False
                self.stt_streaming = False

        session_state = SessionState()

        # Audio queues
        monitor_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=512)

        async def audio_fanout_task():
            """Distribute incoming audio with real-time STT support."""
            try:
                async for frame in pcm_iter:
                    if not frame:
                        continue
                        
                    current_time = time.time()
                    
                    # Voice activity detection
                    is_speech = False
                    if self.vad:
                        try:
                            is_speech = self.vad.is_speech(frame, sample_rate)
                        except Exception as e:
                            print(f"[VAD] Error: {e}")
                            is_speech = False
                    
                    # Update counters
                    if is_speech:
                        session_state.silence_count = 0
                        session_state.speech_frames += 1
                        
                        # Start real-time STT when speech begins
                        if (not session_state.speech_detected and 
                            session_state.mode == "LISTEN" and 
                            self.enable_realtime_stt):
                            
                            session_state.speech_detected = True
                            print("[STT] Speech detected, starting real-time transcription...")
                            await self._start_realtime_stt(session_state, sample_rate, transcription_callback)
                    else:
                        session_state.silence_count += 1

                    session_state.utterance_frames += 1

                    # Feed frame to real-time STT if active
                    if (session_state.stt_streaming and 
                        self._stt_audio_queue and 
                        session_state.mode == "LISTEN"):
                        try:
                            self._stt_audio_queue.put_nowait(frame)
                        except asyncio.QueueFull:
                            # Drop oldest frames if queue is full
                            try:
                                for _ in range(5):
                                    self._stt_audio_queue.get_nowait()
                                self._stt_audio_queue.put_nowait(frame)
                            except asyncio.QueueEmpty:
                                pass

                    # Distribute to monitor queue
                    try:
                        monitor_queue.put_nowait(frame)
                    except asyncio.QueueFull:
                        try:
                            for _ in range(5):
                                monitor_queue.get_nowait()
                            monitor_queue.put_nowait(frame)
                        except asyncio.QueueEmpty:
                            pass

                    # Utterance boundary detection (only when not processing)
                    if (session_state.mode == "LISTEN" and 
                        not session_state.processing_turn and
                        session_state.speech_detected):  # Only after speech was detected
                        
                        should_end = False
                        
                        # Silence-based boundary
                        if session_state.silence_count >= gap_frames:
                            should_end = True
                            
                        # Max utterance length
                        elif session_state.utterance_frames >= max_utterance_frames:
                            should_end = True
                            print("[Orchestrator] Max utterance length reached")
                            
                        # Check requirements for valid utterance
                        if (should_end and 
                            session_state.utterance_frames >= min_utterance_frames and 
                            session_state.speech_frames >= 5):
                            
                            print(f"[Orchestrator] End of utterance detected (frames: {session_state.utterance_frames}, speech: {session_state.speech_frames})")
                            
                            # Stop real-time STT and finalize transcription
                            await self._stop_realtime_stt(session_state)
                            
                            # Mark as processing to prevent new utterances
                            session_state.processing_turn = True
                            session_state.mode = "THINK"
                            session_state.utterance_event.set()
                            
                            # Reset counters
                            session_state.utterance_frames = 0
                            session_state.speech_frames = 0
                            session_state.silence_count = 0
                            session_state.speech_detected = False

            except Exception as e:
                if error_callback:
                    error_callback(e)
                print(f"[Audio Fanout] Error: {e}")
            finally:
                # Clean up
                await self._stop_realtime_stt(session_state)
                try:
                    await monitor_queue.put(None)
                except Exception:
                    pass

        async def barge_in_monitor():
            """Monitor for speech during TTS with improved false positive prevention."""
            speech_frame_count = 0
            consecutive_speech = 0
            
            while True:
                try:
                    item = await asyncio.wait_for(monitor_queue.get(), timeout=0.1)
                    if item is None:
                        break
                        
                    current_time = time.time()
                    
                    # Only monitor for barge-in during active TTS
                    if (session_state.mode == "SPEAK" and 
                        session_state.tts_active and
                        self.enable_barge_in and 
                        self.vad):
                        
                        try:
                            is_speech = self.vad.is_speech(item, sample_rate)
                            
                            if is_speech:
                                consecutive_speech += 1
                                
                                tts_runtime = current_time - self._tts_start_time
                                since_last_tts = current_time - self._last_tts_frame_time
                                
                                if (consecutive_speech >= self.speech_threshold_frames and
                                    tts_runtime > 1.0 and
                                    since_last_tts > 0.5):
                                    
                                    print(f"[Barge-in] Valid user speech detected")
                                    session_state.cancel_token.cancel()
                                    self.stats.record_interruption()
                                    consecutive_speech = 0
                            else:
                                consecutive_speech = max(0, consecutive_speech - 1)
                                
                        except Exception as e:
                            print(f"[Barge-in] VAD error: {e}")
                            
                except asyncio.TimeoutError:
                    consecutive_speech = max(0, consecutive_speech - 1)
                except Exception as e:
                    print(f"[Barge-in] Monitor error: {e}")
                    break

        async def process_conversation_turn():
            """Process one complete turn using the accumulated real-time transcription."""
            turn_start_time = time.time()
            
            try:
                # Use the latest transcription from real-time STT
                user_text = self._latest_transcription.strip()
                
                # Validate transcription
                if not user_text or len(user_text) < 2:
                    print("[STT] No valid transcription, returning to listen")
                    return
                    
                print(f"[User] {user_text}")
                self.history.append({"role": "user", "content": user_text})
                utterance_duration = time.time() - turn_start_time
                self.stats.record_utterance(utterance_duration)

                # Language Model Processing
                print("[LLM] Generating response...")
                session_state.mode = "THINK"
                self._current_state = "PROCESSING"
                
                try:
                    context_messages = self.history[-10:]
                    reply_chunks = []
                    
                    async for token_chunk in self.llm.stream_reply(context_messages):
                        if token_chunk:
                            reply_chunks.append(token_chunk)
                            await asyncio.sleep(0.001)
                            
                except Exception as e:
                    print(f"[LLM] Error: {e}")
                    self.stats.record_error()
                    reply_chunks = ["I apologize, but I'm having trouble processing that right now."]

                reply_text = "".join(reply_chunks).strip()
                if not reply_text:
                    reply_text = "I'm not sure how to respond to that."
                    
                print(f"[Assistant] {reply_text}")
                self.history.append({"role": "assistant", "content": reply_text})

                # Text-to-Speech
                print("[TTS] Synthesizing speech...")
                session_state.mode = "SPEAK"
                session_state.tts_active = True
                session_state.cancel_token.reset()
                self._current_state = "SPEAKING"
                
                self._tts_start_time = time.time()
                self._last_tts_frame_time = time.time()
                synthesis_start = time.time()
                audio_frame_count = 0
                
                try:
                    async for audio_chunk in self.tts.stream(reply_text):
                        if session_state.cancel_token.cancelled:
                            print("[TTS] Synthesis cancelled by barge-in")
                            break
                            
                        if audio_chunk:
                            audio_frame_count += 1
                            self._last_tts_frame_time = time.time()
                            await out_audio_cb(audio_chunk)
                            
                        if audio_frame_count % 10 == 0:
                            await asyncio.sleep(0.001)
                            
                except Exception as e:
                    print(f"[TTS] Error: {e}")
                    self.stats.record_error()
                finally:
                    session_state.tts_active = False
                    synthesis_duration = time.time() - synthesis_start
                    self.stats.record_synthesis(synthesis_duration)
                    
                    total_latency = (time.time() - turn_start_time) * 1000
                    self.stats.record_response_time(total_latency)
                    
                    print(f"[TTS] Completed {audio_frame_count} frames in {synthesis_duration:.2f}s")

            except Exception as e:
                print(f"[Turn] Unexpected error: {e}")
                self.stats.record_error()
                if error_callback:
                    error_callback(e)
            finally:
                # Reset state
                session_state.mode = "LISTEN"
                session_state.tts_active = False
                session_state.processing_turn = False
                self._current_state = "LISTENING"
                self._latest_transcription = ""  # Clear for next turn
                session_state.turn_complete_event.set()
                print("[Orchestrator] Turn complete, returning to listen mode")

        # Start background tasks
        fanout_task = asyncio.create_task(audio_fanout_task())
        monitor_task = asyncio.create_task(barge_in_monitor())
        
        try:
            self._current_state = "LISTENING"
            print("[Orchestrator] Session started - listening for speech...")
            
            # Main conversation loop
            while True:
                try:
                    # Wait for utterance completion
                    await session_state.utterance_event.wait()
                    session_state.utterance_event.clear()
                    
                    # Process the conversation turn
                    await process_conversation_turn()
                    
                    # Wait for turn to complete before accepting new utterances
                    await session_state.turn_complete_event.wait()
                    session_state.turn_complete_event.clear()
                    
                    print("[Orchestrator] Ready for next utterance")
                    
                except asyncio.CancelledError:
                    print("[Orchestrator] Session cancelled")
                    break
                except Exception as e:
                    print(f"[Orchestrator] Loop error: {e}")
                    if error_callback:
                        error_callback(e)
                    # Force return to LISTEN state on error
                    session_state.mode = "LISTEN"
                    session_state.processing_turn = False
                    session_state.tts_active = False
                    self._current_state = "LISTENING"
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"[Orchestrator] Fatal error: {e}")
            if error_callback:
                error_callback(e)
        finally:
            # Cleanup
            self._current_state = "STOPPED"
            await self._stop_realtime_stt(session_state)
            fanout_task.cancel()
            monitor_task.cancel()
            
            with contextlib.suppress(Exception):
                await asyncio.gather(fanout_task, monitor_task, return_exceptions=True)
            
            session_duration = time.time() - self._session_start_time
            print(f"\n[Session] Completed in {session_duration:.1f}s")
            print(f"[Session] Stats: {self.stats.get_summary()}")

    async def _start_realtime_stt(self, session_state, sample_rate: int, transcription_callback: Optional[Callable]):
        """Start real-time STT streaming."""
        if session_state.stt_streaming:
            return
            
        try:
            # Create audio queue for STT
            self._stt_audio_queue = asyncio.Queue(maxsize=1024)
            session_state.stt_streaming = True
            
            # Create audio generator
            async def stt_audio_generator():
                while session_state.stt_streaming:
                    try:
                        frame = await asyncio.wait_for(self._stt_audio_queue.get(), timeout=0.1)
                        if frame is None:
                            break
                        yield frame
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break
            
            # Start STT stream processing
            asyncio.create_task(self._process_realtime_stt(
                stt_audio_generator(), sample_rate, session_state, transcription_callback
            ))
            
        except Exception as e:
            print(f"[STT] Error starting real-time stream: {e}")
            session_state.stt_streaming = False

    async def _process_realtime_stt(self, audio_gen, sample_rate: int, session_state, transcription_callback):
        """Process real-time STT stream."""
        try:
            async for transcription in self.stt.stream(audio_gen, sample_rate):
                if not session_state.stt_streaming:
                    break
                    
                if transcription and transcription.strip():
                    # Remove [REALTIME] and [FINAL] prefixes if present
                    clean_text = transcription.replace("[REALTIME]", "").replace("[FINAL]", "").strip()
                    
                    if clean_text:
                        is_final = "[FINAL]" in transcription
                        if is_final:
                            self._latest_transcription = clean_text
                            
                        print(f"[STT] {'Final' if is_final else 'Live'}: {clean_text}")
                        
                        # Call the transcription callback if provided
                        if transcription_callback:
                            try:
                                transcription_callback(clean_text, is_final)
                            except Exception as e:
                                print(f"[STT] Callback error: {e}")
                                
        except Exception as e:
            print(f"[STT] Real-time processing error: {e}")
        finally:
            session_state.stt_streaming = False

    async def _stop_realtime_stt(self, session_state):
        """Stop real-time STT streaming."""
        if not session_state.stt_streaming:
            return
            
        session_state.stt_streaming = False
        
        if self._stt_audio_queue:
            try:
                # Signal end of stream
                await self._stt_audio_queue.put(None)
            except Exception:
                pass
            finally:
                self._stt_audio_queue = None

    def get_current_state(self) -> str:
        """Get current orchestrator state."""
        return self._current_state
        
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history (excluding system message)."""
        return [msg for msg in self.history if msg.get("role") != "system"]
        
    def clear_conversation_history(self) -> None:
        """Clear conversation history but keep system message."""
        self.history = [msg for msg in self.history if msg.get("role") == "system"]
        
    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        stats = self.stats.get_summary()
        if self._session_start_time > 0:
            stats["session_duration_s"] = round(time.time() - self._session_start_time, 1)
        return stats