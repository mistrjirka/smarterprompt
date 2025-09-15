import os
import httpx
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

@dataclass
class ChatRequest:
    model: str
    messages: List[Dict[str, str]]  # [{"role":"system|user|assistant","content":"..."}]
    temperature: float = 0.2
    max_tokens: int = 1500
    stop: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None

class LangChainProvider:
    """Wrapper for LangChain chat models to maintain compatibility with existing interface"""
    
    def __init__(self, chat_model: BaseChatModel):
        self.chat_model = chat_model
    
    def _convert_messages(self, messages: List[Dict[str, str]]):
        """Convert dict messages to LangChain message objects"""
        converted = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "user":
                converted.append(HumanMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                # Default to human message
                converted.append(HumanMessage(content=content))
        
        return converted
    
    async def chat(self, req: ChatRequest) -> Tuple[str, Dict[str, Any]]:
        """Async chat method compatible with existing interface"""
        try:
            # Convert messages to LangChain format
            lc_messages = self._convert_messages(req.messages)
            
            # LangChain automatically handles thinking models (o1, DeepSeek R1, etc.)
            # No need for manual detection - let LangChain handle reasoning content
            
            # Use ainvoke for async call
            response = await self.chat_model.ainvoke(lc_messages)
            
            # Extract content - LangChain automatically handles reasoning content stripping
            content = response.content
            
            # Create response metadata similar to original format
            metadata = {
                "model": getattr(self.chat_model, 'model_name', req.model),
                "usage": getattr(response, 'usage_metadata', {}),
                "response_metadata": getattr(response, 'response_metadata', {})
            }
            
            return content, metadata
            
        except Exception as e:
            raise RuntimeError(f"LangChain chat error: {e}")

def build_provider(name: str, http_cfg: dict, model: str, temperature: float = 0.2, 
                  max_tokens: int = 1500, stop: Optional[List[str]] = None) -> LangChainProvider:
    """Build a LangChain-based provider"""
    n = (name or "").lower()
    
    if n == "openai":
        chat_model = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=http_cfg.get("organization") or os.getenv("OPENAI_ORG"),
            base_url=http_cfg.get("base_url"),
        )
        
    elif n == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", http_cfg.get("base_url", "http://localhost:11434"))
        chat_model = ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            stop=stop,
            base_url=base_url,
        )
    else:
        raise ValueError(f"Unknown provider: {name}")
    
    return LangChainProvider(chat_model)

def get_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Get list of available Ollama models"""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return sorted(models)
        else:
            return ["llama3", "deepseek-r1:3b", "qwen2.5:7b"]  # fallback models
    except Exception:
        return ["llama3", "deepseek-r1:3b", "qwen2.5:7b"]  # fallback models
