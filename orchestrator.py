from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import os
import yaml

from util import try_parse_json, squeeze
from providers import build_provider, BaseProvider, ChatRequest

MAIN_SYSTEM_PROMPT = """You are the Main AI. Follow the user's task prompt precisely.
- Write clearly and concisely.
- If the task asks for code, provide fully working, self-contained code blocks with filenames.
- Respect any style or formatting requirements given by the user.

Refinement phase rules:
- You will receive a critique from a Judge AI and possibly user feedback.
- Fix ALL must-fix points. Address suggestions if useful, without adding unrelated content.

At the end of your FINAL turn for this task, output a section titled exactly:
## Full Deliverable
Include the complete, final, self-contained deliverable (including all code/files) that the user can copy/paste/run without other context.
"""

JUDGE_SYSTEM_PROMPT = """You are the Judge AI. Evaluate the Main AI's answer vs the original user prompt.

Return a JSON object ONLY, with fields:
{
  "score": float in [0,1],
  "pass": boolean,
  "reasons": [ "brief", ... ],
  "required_changes": [ "must fix 1", ... ],
  "suggestions": [ "nice to have 1", ... ]
}"""

REFINE_USER_PROMPT = """Refine your previous answer to fully address the Judge critique and the user's feedback.
- Apply all REQUIRED changes first.
- Then apply suggestions that improve clarity/quality without breaking constraints.
- Keep focused strictly on the user's goals.
- End with the '## Full Deliverable' section."""

FINALIZE_USER_PROMPT = """Finalize the deliverable now.

Provide:
1) A single coherent **final answer**.
2) **## Full Deliverable** with all files (filenames + contents).
3) **## Transcript Summary** (key steps and decisions as bullets).
"""

@dataclass
class RoleConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    stop: List[str] = field(default_factory=list)

@dataclass
class ProvidersBundle:
    main_ai: BaseProvider
    judge_ai: BaseProvider
    main_cfg: RoleConfig
    judge_cfg: RoleConfig
    openai_cfg: dict = field(default_factory=dict)
    ollama_cfg: dict = field(default_factory=dict)

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        path = "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def build_bundle(cfg: Dict[str, Any]) -> ProvidersBundle:
    openai_cfg = cfg.get("openai", {}) or {}
    ollama_cfg = cfg.get("ollama", {}) or {}

    m = cfg.get("main_ai", {}) or {}
    j = cfg.get("judge_ai", {}) or {}

    main_provider = build_provider(m.get("provider", "openai"),
                                   openai_cfg if m.get("provider") == "openai" else ollama_cfg)
    judge_provider = build_provider(j.get("provider", "ollama"),
                                    openai_cfg if j.get("provider") == "openai" else ollama_cfg)

    main_cfg = RoleConfig(
        provider=m.get("provider", "openai"),
        model=m.get("model", "gpt-4o-mini"),
        temperature=float(m.get("temperature", 0.2)),
        max_tokens=int(m.get("max_tokens", 2000)),
        stop=list(m.get("stop", []) or []),
    )
    judge_cfg = RoleConfig(
        provider=j.get("provider", "ollama"),
        model=j.get("model", "llama3"),
        temperature=float(j.get("temperature", 0.1)),
        max_tokens=int(j.get("max_tokens", 1500)),
        stop=list(j.get("stop", []) or []),
    )

    return ProvidersBundle(
        main_ai=main_provider,
        judge_ai=judge_provider,
        main_cfg=main_cfg,
        judge_cfg=judge_cfg,
        openai_cfg=openai_cfg,
        ollama_cfg=ollama_cfg,
    )

@dataclass
class ChatTurn:
    role: str     # "you" | "main" | "judge" | "system" | "orchestrator"
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)

class ReviewLoop:
    def __init__(self, bundle: ProvidersBundle):
        self.bundle = bundle
        self.original_prompt: Optional[str] = None
        self.transcript: List[ChatTurn] = []
        self.main_ctx: List[Dict[str, str]] = [{"role": "system", "content": MAIN_SYSTEM_PROMPT}]
        self.judge_ctx: List[Dict[str, str]] = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
        self.latest_main: Optional[str] = None
        self.latest_judge_json: Optional[Dict[str, Any]] = None

    def record(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None):
        self.transcript.append(ChatTurn(role=role, content=content, meta=meta or {}))

    async def run_main(self, user_prompt: str) -> str:
        if self.original_prompt is None:
            self.original_prompt = user_prompt
        self.record("you", user_prompt)
        self.main_ctx.append({"role": "user", "content": f"User task:\n{user_prompt}"})
        text, raw = await self.bundle.main_ai.chat(ChatRequest(
            model=self.bundle.main_cfg.model,
            messages=self.main_ctx,
            temperature=self.bundle.main_cfg.temperature,
            max_tokens=self.bundle.main_cfg.max_tokens,
            stop=self.bundle.main_cfg.stop or None,
        ))
        text = squeeze(text)
        self.latest_main = text
        self.record("main", text, {"provider": self.bundle.main_cfg.provider, "model": self.bundle.main_cfg.model})
        self.main_ctx.append({"role": "assistant", "content": text})
        return text

    async def run_judge(self) -> Dict[str, Any]:
        assert self.latest_main, "No main answer to judge."
        user_msg = f"""Original user prompt:
---
{self.original_prompt}

Main AI answer:
---
{text_clip(self.latest_main, 12000)}
"""
        self.judge_ctx.append({"role": "user", "content": user_msg})
        text, raw = await self.bundle.judge_ai.chat(ChatRequest(
            model=self.bundle.judge_cfg.model,
            messages=self.judge_ctx,
            temperature=self.bundle.judge_cfg.temperature,
            max_tokens=self.bundle.judge_cfg.max_tokens,
            stop=self.bundle.judge_cfg.stop or None,
        ))
        parsed = try_parse_json(text) or {
            "score": 0.0, "pass": False,
            "reasons": ["Judge did not produce valid JSON."],
            "required_changes": ["Return valid JSON with required_changes."],
            "suggestions": []
        }
        self.latest_judge_json = parsed
        self.record("judge", text, {"parsed": parsed, "provider": self.bundle.judge_cfg.provider, "model": self.bundle.judge_cfg.model})
        self.judge_ctx.append({"role": "assistant", "content": text})
        return parsed

    async def refine(self, user_feedback: Optional[str] = None) -> str:
        if user_feedback:
            self.record("you", f"(feedback) {user_feedback}")
        critique = self.latest_judge_json or {}
        refine_prompt = f"""{REFINE_USER_PROMPT}

Judge JSON:
{critique}

User feedback:
{user_feedback or "None"}"""
        self.main_ctx.append({"role": "user", "content": refine_prompt})
        text, raw = await self.bundle.main_ai.chat(ChatRequest(
            model=self.bundle.main_cfg.model,
            messages=self.main_ctx,
            temperature=self.bundle.main_cfg.temperature,
            max_tokens=self.bundle.main_cfg.max_tokens,
            stop=self.bundle.main_cfg.stop or None,
        ))
        text = squeeze(text)
        self.latest_main = text
        self.record("main", text, {"phase": "refined"})
        self.main_ctx.append({"role": "assistant", "content": text})
        return text

    async def finalize(self) -> str:
        summary = make_summary(self.transcript, 1800)
        fin_prompt = f"""{FINALIZE_USER_PROMPT}

Original prompt:
{self.original_prompt}

Very short conversation summary:
{summary}
"""
        self.main_ctx.append({"role": "user", "content": fin_prompt})
        text, raw = await self.bundle.main_ai.chat(ChatRequest(
            model=self.bundle.main_cfg.model,
            messages=self.main_ctx,
            temperature=self.bundle.main_cfg.temperature,
            max_tokens=max(2048, self.bundle.main_cfg.max_tokens),
            stop=self.bundle.main_cfg.stop or None,
        ))
        text = squeeze(text)
        self.latest_main = text
        self.record("main", text, {"phase": "final"})
        self.main_ctx.append({"role": "assistant", "content": text})
        return text

    def export(self) -> List[Dict[str, Any]]:
        return [{"role": t.role, "content": t.content, "meta": t.meta} for t in self.transcript]

def text_clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " ..."

def make_summary(transcript: List[ChatTurn], max_chars: int = 2000) -> str:
    parts = []
    for t in transcript:
        tag = {"you": "You", "main": "Main AI", "judge": "Judge AI", "orchestrator": "Orchestrator"}.get(t.role, t.role)
        parts.append(f"{tag}: {text_clip(t.content.replace('\n', ' '), 180)}")
    joined = "\n".join(parts)
    return joined if len(joined) <= max_chars else (joined[:max_chars] + " ...")
