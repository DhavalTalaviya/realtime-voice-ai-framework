# main.py
from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import asyncio
import json
import logging
import sys
import time
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import webrtcvad

from core.orchestrator import VoiceOrchestrator
from core.interfaces import VADEngine

# STT implementations
from plugins.stt.wav2vec2_adapter import Wav2Vec2Streaming
from plugins.stt.faster_whisper_adapter import FasterWhisperSTT
from plugins.stt.groq_stt_adapter import GroqWhisperSTT

# LLM implementation
from plugins.llm.openai_adapter import OpenAIStreaming

# TTS implementations
from plugins.tts.coqui_tts_adapter import CoquiTTSStreaming
from plugins.tts.silero_tts_adapter import SileroTTSStreaming
from plugins.tts.orca_tts_adapter import OrcaTTSStreaming

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced VAD wrapper with error handling
class RobustVAD(VADEngine):
    """WebRTC VAD wrapper with enhanced error handling and adaptive thresholds."""
    
    def __init__(self, aggressiveness: int = 2):
        self.aggressiveness = max(0, min(3, int(aggressiveness)))
        self.vad = webrtcvad.Vad(self.aggressiveness)
        self.frame_count = 0
        self.speech_frames = 0
        
    def is_speech(self, pcm16: bytes, sample_rate: int) -> bool:
        """Detect speech with error handling and statistics tracking."""
        try:
            # Ensure frame is correct size for WebRTC VAD
            expected_frame_sizes = {
                8000: [160, 240, 320],  # 10ms, 15ms, 20ms
                16000: [320, 480, 640], # 10ms, 15ms, 20ms  
                32000: [640, 960, 1280], # 10ms, 15ms, 20ms
                48000: [960, 1440, 1920] # 10ms, 15ms, 20ms
            }
            
            if sample_rate not in expected_frame_sizes:
                return False
                
            frame_size = len(pcm16)
            if frame_size not in expected_frame_sizes[sample_rate]:
                return False
                
            is_speech = self.vad.is_speech(pcm16, sample_rate)
            
            # Update statistics
            self.frame_count += 1
            if is_speech:
                self.speech_frames += 1
                
            return is_speech
            
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return False
    
    def get_speech_ratio(self) -> float:
        """Get ratio of speech frames to total frames."""
        if self.frame_count == 0:
            return 0.0
        return self.speech_frames / self.frame_count

def build_stt_engine(
    engine_type: str, 
    model_name: str, 
    device: str, 
    decode_window_s: float, 
    decode_hop_ms: int
) -> object:
    """Factory function to build STT engines with error handling."""
    engine_type = (engine_type or "wav2vec2").lower()
    
    try:
        if engine_type == "wav2vec2":
            logger.info(f"Loading Wav2Vec2 STT: {model_name}")
            return Wav2Vec2Streaming(
                model_name=model_name,
                device=device,
                decode_window_s=decode_window_s,
                decode_hop_ms=decode_hop_ms,
                min_emit_delta_chars=2
            )
        elif engine_type in ("faster-whisper", "whisper"):
            logger.info(f"Loading Faster-Whisper STT: {model_name}")
            return FasterWhisperSTT(
                model_name=model_name or "tiny",
                device=device,
                compute_type="int8",
                decode_window_s=decode_window_s,
                decode_hop_ms=decode_hop_ms,
                overlap_s=0.5,
                realtime_model="tiny.en",    # Fast model for real-time
                enable_realtime=True,        # Enable real-time updates
                realtime_hop_ms=150,         # Update every 150ms
                silence_timeout_s=1.0       # Finalize after 1s of silence
            )
        elif engine_type == "groq":
            logger.info(f"Loading Groq STT")
            return GroqWhisperSTT(
                api_key=os.getenv("GROQ_API_KEY"),
                model_name="whisper-large-v3",
                enable_realtime=True,
                decode_window_s=2.0,
                realtime_hop_ms=200,  # Balanced for API latency
                max_concurrent_requests=3,
                vad_threshold=0.003
            )
        else:
            raise ValueError(f"Unknown STT engine: {engine_type}")
            
    except Exception as e:
        logger.error(f"Failed to load STT engine '{engine_type}': {e}")
        raise

def build_llm_engine(
        engine_type: str,
        model_name: str,
    ) -> object:
    """Factory function to build LLM engine."""
    try:
        engine_type = (engine_type or "nvidia").lower()
        if engine_type == "nvidia":
            logger.info(f"Loading Nvidia LLM")
            return OpenAIStreaming(os.getenv("NVIDIA_API_KEY"), os.getenv("NVIDIA_BASE_URL"), model_name)
        elif engine_type == "deepseek":
            logger.info(f"Loading DeepSeek LLM")
            return OpenAIStreaming(os.getenv("DEEPSEEK_API_KEY"), os.getenv("DEEPSEEK_BASE_URL"), model_name)
        elif engine_type == "groq":
            logger.info(f"Loading Groq LLM")
            return OpenAIStreaming(os.getenv("GROQ_API_KEY"), os.getenv("GROQ_BASE_URL"), model_name)
        elif engine_type == "anthropic":
            logger.info(f"Loading Anthropic LLM")
            return OpenAIStreaming(os.getenv("ANTHROPIC_API_KEY"), model_name=model_name, provider="anthropic")
        else:
            logger.info("Loading OpenAI LLM adapter")
            return OpenAIStreaming(model_name)
    except Exception as e:
        logger.error(f"Failed to load LLM engine: {e}")
        raise

def build_tts_engine(
    engine_type: str, 
    model_name: str, 
    device: str, 
    speed: float, 
    voice: str
) -> object:
    """Factory function to build TTS engines with error handling and warmup."""
    engine_type = (engine_type or "coqui").lower()
    
    try:
        if engine_type == "coqui":
            logger.info(f"Loading Coqui TTS: {model_name}")
            tts = CoquiTTSStreaming(
                model_name=model_name,
                device=device,
                speed=speed
            )
        elif engine_type == "silero":
            logger.info(f"Loading Silero TTS: {voice}")
            tts = SileroTTSStreaming(
                voice=voice,
                speed=speed
            )
        elif engine_type == "orca":
            logger.info(f"Loading Orca TTS: {voice}")
            tts = OrcaTTSStreaming(
                speech_rate=1.2,
                stream_mode=True,  # Use streaming for better performance
                chunk_max_length=150
            )
        else:
            raise ValueError(f"Unknown TTS engine: {engine_type}")
            
        # Warmup TTS engine
        logger.info("Warming up TTS engine...")
        if hasattr(tts, 'warmup'):
            tts.warmup()
        logger.info("TTS engine ready")
        
        return tts
        
    except Exception as e:
        logger.error(f"Failed to load TTS engine '{engine_type}': {e}")
        raise

def create_app(orchestrator: VoiceOrchestrator, args) -> FastAPI:
    """Create FastAPI application with WebSocket endpoint."""
    app = FastAPI(
        title="Voice Agent Streaming Server",
        description="Real-time voice AI with STT, LLM, and TTS streaming",
        version="1.0.0"
    )

    @app.get("/")
    async def get_client():
        """Serve the HTML client interface."""
        # Read client HTML from the same directory
        try:
            with open("client.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        except FileNotFoundError:
            return HTMLResponse(
                content="<h1>Client not found</h1><p>Please ensure client.html is in the same directory.</p>",
                status_code=404
            )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "orchestrator_state": orchestrator.get_current_state(),
            "session_stats": orchestrator.get_session_stats()
        }

    @app.get("/stats")
    async def get_stats():
        """Get session statistics."""
        return {
            "session_stats": orchestrator.get_session_stats(),
            "conversation_length": len(orchestrator.get_conversation_history())
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Enhanced WebSocket endpoint with status updates."""
        await websocket.accept()
        client_ip = websocket.client.host if websocket.client else "unknown"
        logger.info(f"Client connected: {client_ip}")

        async def send_status_update(status: str, data: dict = None):
            """Send status update to client."""
            try:
                message = {
                    "type": "status",
                    "status": status,
                    "timestamp": time.time(),
                    "data": data or {}
                }
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send status update: {e}")

        async def error_callback(error: Exception):
            """Enhanced error callback with client notification."""
            logger.error(f"Orchestrator error: {error}")
            await send_status_update("error", {"message": str(error)})

        async def audio_input_stream():
            """Stream audio from client with status tracking."""
            frame_count = 0
            try:
                while True:
                    try:
                        data = await websocket.receive_bytes()
                        
                        if data == b"__END__":
                            logger.info("Received end signal from client")
                            await send_status_update("disconnecting")
                            break
                            
                        if len(data) < 100:  # Too small to be valid audio
                            continue
                        
                        frame_count += 1
                        
                        # Send periodic status updates
                        if frame_count % 100 == 0:  # Every ~2 seconds at 20ms frames
                            await send_status_update("listening", {
                                "frames_received": frame_count,
                                "orchestrator_state": orchestrator.get_current_state()
                            })
                            
                        yield data
                        
                    except WebSocketDisconnect:
                        logger.info("Client disconnected")
                        break
                    except Exception as e:
                        logger.error(f"Error receiving audio: {e}")
                        await send_status_update("error", {"message": f"Audio error: {e}"})
                        break
                        
            except Exception as e:
                logger.error(f"Audio input stream error: {e}")
            finally:
                logger.info(f"Audio input stream ended. Total frames: {frame_count}")
                await send_status_update("stream_ended", {"total_frames": frame_count})

        async def audio_output_callback(audio_chunk: bytes):
            """Send TTS audio to client with status updates."""
            try:
                await websocket.send_bytes(audio_chunk)
                
                # Update status to speaking when first audio chunk is sent
                if orchestrator.get_current_state() == "SPEAKING":
                    await send_status_update("speaking")
                    
            except WebSocketDisconnect:
                logger.info("Client disconnected during audio output")
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

        # Send initial status
        await send_status_update("connected", {
            "server_info": {
                "stt_engine": args.stt,
                "tts_engine": args.tts,
                "sample_rate": args.sample_rate
            }
        })

        # Create a modified orchestrator wrapper that sends status updates
        class StatusAwareOrchestrator:
            def __init__(self, base_orchestrator):
                self.base = base_orchestrator
                self._last_state = "IDLE"
                
            async def run_session(self, pcm_iter, sample_rate, out_audio_cb, error_callback=None):
                """Wrapper that monitors state changes and sends updates."""
                
                # Create a wrapper for the audio callback that tracks state
                async def enhanced_audio_callback(audio_chunk):
                    await out_audio_cb(audio_chunk)
                    
                    # Check for state changes
                    current_state = self.base.get_current_state()
                    if current_state != self._last_state:
                        self._last_state = current_state
                        status_map = {
                            "LISTENING": "listening",
                            "PROCESSING": "processing", 
                            "SPEAKING": "speaking",
                            "IDLE": "idle",
                            "STOPPED": "stopped"
                        }
                        await send_status_update(status_map.get(current_state, "unknown"), {
                            "orchestrator_state": current_state
                        })
                
                # Enhanced error callback
                async def enhanced_error_callback(error):
                    await send_status_update("error", {"message": str(error)})
                    if error_callback:
                        await error_callback(error)
                
                # Monitor state changes during session
                state_monitor_task = asyncio.create_task(self._monitor_state_changes())
                
                try:
                    await self.base.run_session(
                        pcm_iter=pcm_iter,
                        sample_rate=sample_rate, 
                        out_audio_cb=enhanced_audio_callback,
                        error_callback=enhanced_error_callback
                    )
                finally:
                    state_monitor_task.cancel()
                    await send_status_update("session_ended")
                    
            async def _monitor_state_changes(self):
                """Monitor orchestrator state changes and send updates."""
                try:
                    while True:
                        current_state = self.base.get_current_state()
                        if current_state != self._last_state:
                            self._last_state = current_state
                            status_map = {
                                "LISTENING": "listening",
                                "PROCESSING": "processing",
                                "SPEAKING": "speaking", 
                                "IDLE": "idle",
                                "STOPPED": "stopped"
                            }
                            await send_status_update(status_map.get(current_state, "unknown"), {
                                "orchestrator_state": current_state,
                                "stats": self.base.get_session_stats()
                            })
                        await asyncio.sleep(0.5)  # Check twice per second
                except asyncio.CancelledError:
                    pass
                    
            def get_current_state(self):
                return self.base.get_current_state()
                
            def get_session_stats(self):
                return self.base.get_session_stats()

        # Create status-aware wrapper
        status_orchestrator = StatusAwareOrchestrator(orchestrator)

        class PerformanceMonitor:
            def __init__(self):
                self.stt_times = []
                self.response_times = []
            
            def log_stt_time(self, decode_time_ms):
                self.stt_times.append(decode_time_ms)
                if len(self.stt_times) % 5 == 0:
                    avg = sum(self.stt_times[-5:]) / 5
                    print(f"[PERF] STT Average (last 5): {avg:.0f}ms")
            
            def log_response_time(self, total_time_ms):
                self.response_times.append(total_time_ms)
                print(f"[PERF] Total Response Time: {total_time_ms:.0f}ms")

        perf_monitor = PerformanceMonitor()

        # Run the orchestrator session
        try:
            await status_orchestrator.run_session(
                pcm_iter=audio_input_stream(),
                sample_rate=16000,
                out_audio_cb=audio_output_callback,
                error_callback=error_callback
            )
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during session")
        except Exception as e:
            logger.error(f"Session error: {e}")
            await send_status_update("session_error", {"message": str(e)})
        finally:
            try:
                await send_status_update("disconnected")
                await websocket.close()
            except Exception:
                pass
            logger.info(f"Client session ended: {client_ip}")

    return app

def main():
    """Main entry point with comprehensive argument parsing and error handling."""
    parser = argparse.ArgumentParser(
        prog="voiceagent-streaming",
        description="Real-time voice AI agent with streaming STT, LLM, and TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Live streaming command
    live_parser = subparsers.add_parser(
        "live", 
        help="Start streaming voice agent server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # STT Configuration
    stt_group = live_parser.add_argument_group("Speech-to-Text Options")
    stt_group.add_argument(
        "--stt", 
        choices=["wav2vec2", "faster-whisper", "groq"],
        default="wav2vec2", 
        help="STT engine to use"
    )
    stt_group.add_argument(
        "--stt-model", 
        default="facebook/wav2vec2-large-960h-lv60-self",
        help="STT model name or path"
    )
    stt_group.add_argument(
        "--stt-device", 
        default="cpu",
        help="Device for STT processing (cpu/cuda)"
    )
    stt_group.add_argument(
        "--stt-decode-window-s", 
        type=float, 
        default=6.0,
        help="STT decode window duration in seconds"
    )
    stt_group.add_argument(
        "--stt-decode-hop-ms", 
        type=int, 
        default=500,
        help="STT decode hop interval in milliseconds"
    )

    # LLM Configuration
    llm_group = live_parser.add_argument_group("LLM Options")
    llm_group.add_argument(
        "--llm", 
        choices=["nvidia", "deepseek", "groq", "anthropic"],
        default="nvidia", 
        help="LLM engine to use"
    )
    llm_group.add_argument(
        "--llm-model",
        help="LLM model name"
    )
    
    # TTS Configuration
    tts_group = live_parser.add_argument_group("Text-to-Speech Options")
    tts_group.add_argument(
        "--tts", 
        choices=["coqui", "silero", "orca"],
        default="coqui", 
        help="TTS engine to use"
    )
    tts_group.add_argument(
        "--tts-model", 
        default="tts_models/en/ljspeech/fast_pitch",
        help="TTS model name"
    )
    tts_group.add_argument(
        "--tts-voice", 
        default="en_0",
        help="Voice ID (for Silero TTS)"
    )
    tts_group.add_argument(
        "--tts-speed", 
        type=float, 
        default=1.2,
        help="TTS speech speed multiplier"
    )
    
    # Audio Processing Options
    audio_group = live_parser.add_argument_group("Audio Processing Options")
    audio_group.add_argument(
        "--sample-rate", 
        type=int, 
        default=16000,
        help="Audio sample rate in Hz"
    )
    audio_group.add_argument(
        "--frame-ms", 
        type=int, 
        default=20,
        help="Audio frame duration in milliseconds"
    )
    audio_group.add_argument(
        "--gap-ms", 
        type=int, 
        default=600,
        help="Silence gap for utterance boundary detection"
    )
    audio_group.add_argument(
        "--vad-aggressiveness", 
        type=int, 
        choices=[0, 1, 2, 3],
        default=2,
        help="VAD aggressiveness level (0-3, higher = more aggressive)"
    )
    
    # Orchestrator Options
    orchestrator_group = live_parser.add_argument_group("Orchestrator Options")
    orchestrator_group.add_argument(
        "--max-utterance-s", 
        type=float, 
        default=30.0,
        help="Maximum utterance duration in seconds"
    )
    orchestrator_group.add_argument(
        "--min-utterance-s", 
        type=float, 
        default=0.5,
        help="Minimum utterance duration in seconds"
    )
    orchestrator_group.add_argument(
        "--disable-barge-in", 
        action="store_true",
        help="Disable barge-in interruption during TTS"
    )
    orchestrator_group.add_argument(
        "--speech-threshold-frames", 
        type=int, 
        default=3,
        help="Number of consecutive speech frames needed for barge-in"
    )
    
    # Server Options
    server_group = live_parser.add_argument_group("Server Options")
    server_group.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Server host address"
    )
    server_group.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Server port"
    )
    server_group.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command != "live":
        parser.print_help()
        return 1

    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate arguments
    if args.sample_rate not in [8000, 16000, 32000, 48000]:
        logger.error("Sample rate must be 8000, 16000, 32000, or 48000 Hz")
        return 1
        
    if not (0.5 <= args.tts_speed <= 3.0):
        logger.error("TTS speed must be between 0.5 and 3.0")
        return 1

    try:
        # Build engines with comprehensive error handling
        logger.info("Initializing voice agent components...")
        
        # Build STT engine
        logger.info("Loading STT engine...")
        stt_engine = build_stt_engine(
            args.stt, 
            args.stt_model, 
            args.stt_device,
            args.stt_decode_window_s, 
            args.stt_decode_hop_ms
        )
        
        # Build LLM engine
        logger.info("Loading LLM engine...")
        llm_engine = build_llm_engine(
            args.llm,
            args.llm_model
        )
        
        # Build TTS engine
        logger.info("Loading TTS engine...")
        tts_engine = build_tts_engine(
            args.tts, 
            args.tts_model,
            args.stt_device, 
            args.tts_speed,
            args.tts_voice
        )
        
        # Build VAD engine
        logger.info("Initializing VAD...")
        vad_engine = RobustVAD(aggressiveness=args.vad_aggressiveness)
        
        # Create orchestrator
        logger.info("Creating voice orchestrator...")
        orchestrator = VoiceOrchestrator(
            stt=stt_engine,
            llm=llm_engine,
            tts=tts_engine,
            vad=vad_engine,
            frame_ms=args.frame_ms,
            gap_ms=args.gap_ms,
            max_utterance_s=args.max_utterance_s,
            min_utterance_s=args.min_utterance_s,
            enable_barge_in=not args.disable_barge_in,
            speech_threshold_frames=args.speech_threshold_frames,
            enable_realtime_stt=True
        )
        
        # Create FastAPI app
        app = create_app(orchestrator, args)
        
        # Print startup information
        logger.info("=" * 60)
        logger.info("Voice Agent Streaming Server")
        logger.info("=" * 60)
        logger.info(f"STT Engine: {args.stt} ({args.stt_model})")
        logger.info(f"TTS Engine: {args.tts} (speed: {args.tts_speed}x)")
        logger.info(f"VAD Aggressiveness: {args.vad_aggressiveness}")
        logger.info(f"Server: http://{args.host}:{args.port}")
        logger.info(f"WebSocket: ws://{args.host}:{args.port}/ws")
        logger.info(f"Barge-in: {'Enabled' if not args.disable_barge_in else 'Disabled'}")
        logger.info("=" * 60)
        logger.info("Ready for connections!")
        
        # Start server
        uvicorn.run(
            app, 
            host=args.host, 
            port=args.port,
            log_level=args.log_level.lower(),
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        return 0
    except Exception as e:
        logger.error(f"Startup error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())