
import os, asyncio
from typing import AsyncIterator, Dict, List
from core.interfaces import LLMEngine
from agent import Agent

class OpenAIStreaming(LLMEngine):
    def __init__(
        self,
        api_key:str = os.getenv("NVIDIA_API_KEY"), 
        base_url:str = os.getenv("NVIDIA_BASE_URL"), 
        model_name: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        provider:str = "openai"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model_name
        self.provider = provider.lower()

        self.agent = Agent(model=self.model, api_key=api_key, base_url=base_url, provider=self.provider)

    async def stream_reply(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        user_last = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
        reply = self.agent.chat(user_last)
        step = 80
        for i in range(0, len(reply), step):
            yield reply[i:i+step]
            await asyncio.sleep(0)
