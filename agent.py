from openai import OpenAI
import anthropic
from global_rules import get_global_rules_text
from master_prompt import VOICE_AGENT_PERSONA, MASTER_PROMPT_TEMPLATE

class Agent:
    def __init__(self, model: str, api_key: str, base_url: str = None,
                 provider: str = "openai",  # New parameter
                 agent_name: str = "VoiceAssistant",
                 company_name: str = "CompanyName",
                 agent_goal: str = "help users via voice interactions",
                 trading_hours: str = "9am–5pm weekdays",
                 address: str = "123 Main St, Hometown",
                 service_types: str = "general inquiries",
                 service_modalities: str = "phone, chat"):
        
        self.provider = provider.lower()
        self.model = model
        
        # Initialize client based on provider
        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        else:  # Default to OpenAI-compatible
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)

        self.persona = VOICE_AGENT_PERSONA.format(
            agent_name=agent_name,
            company_name=company_name,
            agent_goal=agent_goal,
            trading_hours=trading_hours,
            address=address,
            service_types=service_types,
            service_modalities=service_modalities
        )
        self.global_rules = get_global_rules_text()
        self.history = []

    def chat(self, user_text: str) -> str:
        # Append user turn
        self.history.append({"role": "user", "content": user_text})

        # Build system prompt with recent context summary
        recent = self.history[-6:]
        conversation_context = "".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)
        system_prompt = MASTER_PROMPT_TEMPLATE.format(
            persona=self.persona,
            global_rules=self.global_rules,
            conversation_context=conversation_context
        )

        if self.provider == "anthropic":
            return self._chat_anthropic(system_prompt, recent)
        else:
            return self._chat_openai(system_prompt, recent)

    def _chat_openai(self, system_prompt: str, recent_history: list) -> str:
        """Handle OpenAI-compatible API calls"""
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(recent_history)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=0.6,
            top_p=0.8,
            max_tokens=1000,
            frequency_penalty=0.5,
            presence_penalty=0,
            stream=True
        )
        buffer = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                buffer += delta or ""

        # Remove hidden thinking tags if any
        answer = buffer.split("</think>")[-1].strip()
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def _chat_anthropic(self, system_prompt: str, recent_history: list) -> str:
        """Handle Anthropic API calls"""
        # Anthropic doesn't use system messages in the messages array
        # Instead, system prompt goes as a separate parameter
        
        # Filter out system messages from recent_history for Anthropic
        messages = [msg for msg in recent_history if msg["role"] != "system"]
        
        # Ensure messages alternate properly (Anthropic requirement)
        if messages and messages[0]["role"] == "assistant":
            messages = messages[1:]  # Remove leading assistant message if any
            
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,  # System prompt goes here
            messages=messages,
            temperature=0.6,
            top_p=0.8,
            max_tokens=1000,
            stream=True
        )
        
        buffer = ""
        for chunk in response:
            if chunk.type == "content_block_delta":
                buffer += chunk.delta.text or ""

        # Remove hidden thinking tags if any
        answer = buffer.split("</think>")[-1].strip()
        self.history.append({"role": "assistant", "content": answer})
        return answer