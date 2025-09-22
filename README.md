# Voice Agent - Production-Grade Real-Time Voice AI

A comprehensive streaming voice AI framework with real-time STT, LLM, and TTS capabilities. Built for production deployments with low-latency requirements and enterprise-grade reliability.

## Key Features

- **Real-time streaming**: 20ms audio frames with immediate transcription feedback
- **Barge-in support**: Users can interrupt AI responses naturally
- **Multi-provider support**: Swap between STT/LLM/TTS providers without code changes
- **Production-ready**: Circuit breakers, error recovery, and performance monitoring
- **WebSocket streaming**: Full-duplex audio communication
- **Plugin architecture**: Extensible design for custom implementations

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Client    │◄──►│ Orchestrator │◄──►│  Plugins    │
│ (Browser)   │    │   (FastAPI)  │    │ STT/LLM/TTS │
└─────────────┘    └──────────────┘    └─────────────┘
      │                     │                   │
   WebSocket           State Machine        Provider APIs
   Audio Stream        Management           (Local/Cloud)
```

### Core Components

- **VoiceOrchestrator**: Central coordinator managing conversation flow and real-time audio
- **Plugin System**: Clean interfaces for STT, LLM, and TTS providers  
- **WebSocket Server**: FastAPI-based streaming server with status monitoring
- **Web Client**: HTML/JavaScript interface for testing and interaction

## Quick Start

### Prerequisites

- Python 3.10+
- Audio input device (microphone)
- GPU recommended for local models

### Installation

```bash
git clone https://github.com/DhavalTalaviya/realtime-voice-ai-framework
cd voice-agent
pip install -e .
```

### Environment Setup

Create `.env` file with your API keys:
```env
# Required for specific providers
NVIDIA_API_KEY=your_nvidia_key
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key
DEEPSEEK_API_KEY=your_deepseek_key
ORCA_API_KEY=your_orca_key

# API Base URLs
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
GROQ_BASE_URL=https://api.groq.com/openai/v1
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### Run the Server

```bash
# Start with default configuration
python -m apps.server_ws.main live

# Customize providers and models
python -m apps.server_ws.main live \
  --stt faster-whisper \
  --stt-model tiny \
  --llm anthropic \
  --llm-model claude-3-sonnet-20240229 \
  --tts silero \
  --tts-speed 1.2
  --disable-barge-in
```

Open http://localhost:8000 in your browser and start talking!

## Provider Options

### Speech-to-Text (STT)
- **wav2vec2** (default): `facebook/wav2vec2-large-960h-lv60-self` - Local streaming model
- **faster-whisper**: Local Whisper with real-time streaming, default model `tiny`
- **groq**: Groq Whisper API using `whisper-large-v3`

### Language Models (LLM)  
- **nvidia** (default): `nvidia/llama-3.3-nemotron-super-49b-v1.5`
- **anthropic**: Claude models via Anthropic API
- **groq**: Groq inference API
- **deepseek**: DeepSeek models

### Text-to-Speech (TTS)
- **coqui** (default): `tts_models/en/ljspeech/fast_pitch` - Local synthesis
- **silero**: Local Silero models, default voice `en_0`  
- **orca**: Picovoice Orca cloud TTS with streaming

## Configuration Examples

### Cloud-First Setup
```bash
python -m apps.server_ws.main live \
  --stt groq \
  --llm nvidia \
  --tts orca \
  --tts-speed 1.2
```

### Offline-Capable Setup
```bash
python -m apps.server_ws.main live \
  --stt faster-whisper \
  --stt-model base \
  --tts coqui \
  --tts-model tts_models/en/ljspeech/fast_pitch
```

### Development/Testing
```bash
python -m apps.server_ws.main live \
  --stt faster-whisper \
  --stt-model tiny \
  --llm groq \
  --tts silero \
  --log-level DEBUG
```

## Command Line Options

### STT Configuration
```bash
--stt {wav2vec2,faster-whisper,groq}  # STT engine choice
--stt-model MODEL_NAME                # Specific model name
--stt-device {cpu,cuda}               # Processing device
--stt-decode-window-s SECONDS         # Decode window duration (default: 6.0)
--stt-decode-hop-ms MILLISECONDS      # Decode hop interval (default: 500)
```

### LLM Configuration  
```bash
--llm {nvidia,deepseek,groq,anthropic}  # LLM provider
--llm-model MODEL_NAME                  # Specific model name
```

### TTS Configuration
```bash
--tts {coqui,silero,orca}         # TTS engine choice
--tts-model MODEL_NAME            # Model name (for Coqui)
--tts-voice VOICE_ID              # Voice ID (for Silero)
--tts-speed MULTIPLIER            # Speech speed (default: 1.2)
```

### Audio Processing
```bash
--sample-rate {8000,16000,32000,48000}  # Audio sample rate (default: 16000)
--frame-ms MILLISECONDS                 # Frame duration (default: 20)
--gap-ms MILLISECONDS                   # Silence gap threshold (default: 600)
--vad-aggressiveness {0,1,2,3}          # VAD sensitivity (default: 2)
```

### Orchestrator Options
```bash
--max-utterance-s SECONDS        # Max speech length (default: 30.0)
--min-utterance-s SECONDS        # Min speech length (default: 0.5)
--disable-barge-in               # Disable interruption capability
--speech-threshold-frames COUNT  # Frames needed for barge-in (default: 3)
```

### Server Options
```bash
--host HOST           # Server host (default: 0.0.0.0)
--port PORT           # Server port (default: 8000)
--log-level LEVEL     # Logging level (default: INFO)
```

## API Endpoints

- `GET /`: Web client interface
- `GET /health`: Health check with orchestrator state
- `GET /stats`: Session statistics
- `WebSocket /ws`: Real-time audio streaming

## Agent Customization

Edit the agent configuration in `agent.py`:

```python
agent = Agent(
    model="your-model",
    agent_name="CustomerService",
    company_name="Your Company",
    agent_goal="provide excellent customer support",
    trading_hours="9am-5pm weekdays",
    address="123 Main St, City",
    service_types="technical support, billing",
    service_modalities="phone, chat"
)
```

Global behavioral rules are defined in `global_rules.py`.

## Dependencies

Core requirements from `pyproject.toml`:
```
fastapi>=0.111
uvicorn[standard]>=0.30
websockets>=12.0
aiohttp>=3.9
python-dotenv>=1.0
numpy>=1.24
soundfile>=0.12
resampy>=0.4.3
librosa>=0.10.0.post2
webrtcvad>=2.0.10
transformers>=4.41
torch>=2.2
TTS>=0.22.0
openai>=1.30.0
```

Optional dependencies:
```bash
pip install -e ".[offline]"  # For faster-whisper
pip install -e ".[dev]"      # For development tools
```

## Monitoring

### Built-in Monitoring
- Real-time WebSocket status updates
- Session statistics tracking
- Performance metrics collection
- Comprehensive structured logging

### Health Checks
```bash
curl http://localhost:8000/health
# Returns: orchestrator state, session stats

curl http://localhost:8000/stats  
# Returns: detailed session metrics
```

## Architecture Details

### Audio Pipeline
1. **Input**: 16kHz PCM16 audio in 20ms frames (320 samples)
2. **VAD**: WebRTC voice activity detection with configurable aggressiveness
3. **STT**: Streaming transcription with real-time updates
4. **LLM**: Streaming text generation with conversation context
5. **TTS**: Audio synthesis with frame-based streaming output

### State Management
The orchestrator manages these states:
- `IDLE`: Waiting for connection
- `LISTENING`: Accepting audio input
- `PROCESSING`: Running STT + LLM
- `SPEAKING`: Playing TTS output
- `STOPPED`: Session ended

### Error Handling
- Circuit breaker patterns prevent cascade failures
- Automatic retry logic with exponential backoff
- Graceful degradation when providers fail
- Session recovery and state cleanup

## Development

### Adding New Providers

1. Implement the appropriate interface from `core/interfaces.py`:
   - `STTEngine` for speech-to-text
   - `LLMEngine` for language models  
   - `TTSEngine` for text-to-speech

2. Add your implementation to the appropriate `plugins/` directory

3. Update the factory functions in `main.py`

### Testing
```bash
pip install -e ".[dev]"
pytest tests/
ruff check .
mypy .
```

## License

MIT License - see LICENSE file for details.
