import os
import httpx
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class ProviderHTTP:
    base_url: str
    chat_endpoint: str

@dataclass
class ChatRequest:
    model: str
    messages: List[Dict[str, str]]  # [{"role":"system|user|assistant","content":"..."}]
    temperature: float = 0.2
    max_tokens: int = 1500
    stop: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None

class BaseProvider:
    async def chat(self, req: ChatRequest) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

class OpenAIProvider(BaseProvider):
    def __init__(self, cfg: ProviderHTTP, api_key: Optional[str] = None, organization: Optional[str] = None):
        self.cfg = cfg
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv("OPENAI_ORG")

    async def chat(self, req: ChatRequest) -> Tuple[str, Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        url = self.cfg.base_url.rstrip("/") + self.cfg.chat_endpoint
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        payload = {
            "model": req.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.stop:
            payload["stop"] = req.stop
        if req.extra:
            payload.update(req.extra)

        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return text, data

class OllamaProvider(BaseProvider):
    def __init__(self, cfg: ProviderHTTP):
        self.cfg = cfg

    async def chat(self, req: ChatRequest) -> Tuple[str, Dict[str, Any]]:
        url = self.cfg.base_url.rstrip("/") + self.cfg.chat_endpoint
        payload = {
            "model": req.model,
            "messages": req.messages,
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            }
        }
        if req.stop:
            payload["options"]["stop"] = req.stop
        if req.extra:
            payload["options"].update(req.extra)

        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message") or {}
            content = msg.get("content", "")
            return content, data

def build_provider(name: str, http_cfg: dict) -> BaseProvider:
    n = (name or "").lower()
    if n == "openai":
        return OpenAIProvider(
            ProviderHTTP(
                base_url=http_cfg.get("base_url", "https://api.openai.com/v1"),
                chat_endpoint=http_cfg.get("chat_endpoint", "/chat/completions"),
            ),
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=http_cfg.get("organization") or os.getenv("OPENAI_ORG"),
        )
    elif n == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", http_cfg.get("base_url", "http://localhost:11434"))
        return OllamaProvider(
            ProviderHTTP(
                base_url=base,
                chat_endpoint=http_cfg.get("chat_endpoint", "/api/chat"),
            )
        )
    else:
        raise ValueError(f"Unknown provider: {name}")
