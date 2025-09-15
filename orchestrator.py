from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass, field
import os
import yaml
import json

# LangChain/LangGraph imports
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from util import try_parse_json, squeeze
from providers import build_provider, LangChainProvider, ChatRequest

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
    main_ai: LangChainProvider
    judge_ai: LangChainProvider
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

    main_provider = build_provider(
        main_cfg.provider,
        openai_cfg if main_cfg.provider == "openai" else ollama_cfg,
        main_cfg.model,
        main_cfg.temperature,
        main_cfg.max_tokens,
        main_cfg.stop
    )
    judge_provider = build_provider(
        judge_cfg.provider,
        openai_cfg if judge_cfg.provider == "openai" else ollama_cfg,
        judge_cfg.model,
        judge_cfg.temperature,
        judge_cfg.max_tokens,
        judge_cfg.stop
    )

    return ProvidersBundle(
        main_ai=main_provider,
        judge_ai=judge_provider,
        main_cfg=main_cfg,
        judge_cfg=judge_cfg,
        openai_cfg=openai_cfg,
        ollama_cfg=ollama_cfg,
    )

# LangGraph State Definition
class ReviewState(TypedDict):
    original_prompt: Optional[str]
    current_answer: Optional[str]
    judge_feedback: Optional[Dict[str, Any]]
    user_feedback: Optional[str]
    transcript: List[Dict[str, Any]]
    phase: str  # "main", "judge", "refine", "finalize"
    iteration: int
    roles_switched: bool  # True if roles have been swapped

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


# LangGraph-based Review Workflow
class LangGraphReviewLoop:
    def __init__(self, bundle: ProvidersBundle, message_callback=None, status_callback=None):
        self.bundle = bundle
        self.graph = self._build_graph()
        self.current_node = None
        self.last_state = None
        self.message_callback = message_callback
        self.status_callback = status_callback
    
    def _get_current_providers(self, state: ReviewState) -> tuple[LangChainProvider, LangChainProvider]:
        """Get current main and judge providers based on iteration count"""
        # Switch roles every 2 iterations (every 2 rounds)
        should_switch = (state["iteration"] // 2) % 2 == 1
        
        if should_switch:
            # Roles are switched: original judge becomes main, original main becomes judge
            return self.bundle.judge_ai, self.bundle.main_ai
        else:
            # Normal roles
            return self.bundle.main_ai, self.bundle.judge_ai
    
    def _get_role_info(self, state: ReviewState) -> dict:
        """Get information about current role assignments"""
        should_switch = (state["iteration"] // 2) % 2 == 1
        
        if should_switch:
            return {
                "main_model": f"{self.bundle.judge_cfg.provider}:{self.bundle.judge_cfg.model}",
                "judge_model": f"{self.bundle.main_cfg.provider}:{self.bundle.main_cfg.model}",
                "roles_switched": True,
                "switch_iteration": (state["iteration"] // 2) * 2
            }
        else:
            return {
                "main_model": f"{self.bundle.main_cfg.provider}:{self.bundle.main_cfg.model}",
                "judge_model": f"{self.bundle.judge_cfg.provider}:{self.bundle.judge_cfg.model}",
                "roles_switched": False,
                "switch_iteration": None
            }
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        graph = StateGraph(ReviewState)
        
        # Add nodes
        graph.add_node("main_ai", self._main_ai_node)
        graph.add_node("judge_ai", self._judge_ai_node)
        graph.add_node("refine", self._refine_node)
        graph.add_node("finalize", self._finalize_node)
        
        # Add conditional edges
        graph.add_edge(START, "main_ai")
        graph.add_edge("main_ai", "judge_ai")
        graph.add_conditional_edges(
            "judge_ai",
            self._should_refine,
            {
                "refine": "refine",
                "finalize": "finalize",
                "end": END
            }
        )
        graph.add_edge("refine", "main_ai")  # Fixed: refine should go back to main_ai
        graph.add_edge("finalize", END)
        
        return graph.compile()
    
    async def _main_ai_node(self, state: ReviewState) -> ReviewState:
        """Main AI generation node"""
        self.current_node = "main_ai"  # Track current active node
        self.last_state = state  # Update state for status callbacks
        
        # Call status callback for real-time updates
        if self.status_callback:
            self.status_callback()
        
        if not state.get("original_prompt"):
            raise ValueError("No original prompt provided")
        
        # Get the correct provider based on role switching
        main_provider, _ = self._get_current_providers(state)
        role_info = self._get_role_info(state)
        
        messages = [{"role": "system", "content": MAIN_SYSTEM_PROMPT}]
        
        if state["phase"] == "main":
            messages.append({"role": "user", "content": f"User task:\n{state['original_prompt']}"})
        else:
            # Refinement phase
            critique = state.get("judge_feedback", {})
            user_feedback = state.get("user_feedback", "")
            
            refine_prompt = f"""{REFINE_USER_PROMPT}

Judge JSON:
{critique}

User feedback:
{user_feedback or "None"}"""
            messages.append({"role": "user", "content": refine_prompt})
        
        # Use appropriate config based on role switching
        current_cfg = self.bundle.judge_cfg if role_info["roles_switched"] else self.bundle.main_cfg
        
        req = ChatRequest(
            model=current_cfg.model,
            messages=messages,
            temperature=current_cfg.temperature,
            max_tokens=current_cfg.max_tokens,
            stop=current_cfg.stop or None,
        )
        
        text, raw = await main_provider.chat(req)
        text = squeeze(text)
        
        # Call message callback for real-time display
        if self.message_callback:
            meta = {
                "provider": current_cfg.provider, 
                "model": current_cfg.model, 
                "phase": state["phase"],
                "roles_switched": role_info["roles_switched"],
                "actual_model": role_info["main_model"]
            }
            self.message_callback("main", text, meta)
        
        # Update state
        new_state = state.copy()
        new_state["current_answer"] = text
        new_state["transcript"].append({
            "role": "main",
            "content": text,
            "meta": {
                "provider": current_cfg.provider, 
                "model": current_cfg.model, 
                "phase": state["phase"],
                "roles_switched": role_info["roles_switched"],
                "actual_model": role_info["main_model"]
            }
        })
        new_state["roles_switched"] = role_info["roles_switched"]
        
        return new_state
    
    async def _judge_ai_node(self, state: ReviewState) -> ReviewState:
        """Judge AI evaluation node"""
        self.current_node = "judge_ai"  # Track current active node
        self.last_state = state  # Update state for status callbacks
        
        # Call status callback for real-time updates
        if self.status_callback:
            self.status_callback()
        
        current_answer = state.get("current_answer")
        if not current_answer:
            raise ValueError("No answer to judge")
        
        # Get the correct provider based on role switching
        _, judge_provider = self._get_current_providers(state)
        role_info = self._get_role_info(state)
        
        user_msg = f"""Original user prompt:
---
{state['original_prompt']}

Main AI answer:
---
{text_clip(current_answer, 12000)}
"""
        
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        
        # Use appropriate config based on role switching
        current_cfg = self.bundle.main_cfg if role_info["roles_switched"] else self.bundle.judge_cfg
        
        req = ChatRequest(
            model=current_cfg.model,
            messages=messages,
            temperature=current_cfg.temperature,
            max_tokens=current_cfg.max_tokens,
            stop=current_cfg.stop or None,
        )
        
        text, raw = await judge_provider.chat(req)
        parsed = try_parse_json(text) or {
            "score": 0.0, "pass": False,
            "reasons": ["Judge did not produce valid JSON."],
            "required_changes": ["Return valid JSON with required_changes."],
            "suggestions": []
        }
        
        # Call message callback for real-time display
        if self.message_callback:
            meta = {
                "parsed": parsed, 
                "provider": current_cfg.provider, 
                "model": current_cfg.model,
                "roles_switched": role_info["roles_switched"],
                "actual_model": role_info["judge_model"]
            }
            self.message_callback("judge", text, meta)
        
        # Update state
        new_state = state.copy()
        new_state["judge_feedback"] = parsed
        new_state["transcript"].append({
            "role": "judge",
            "content": text,
            "meta": {
                "parsed": parsed, 
                "provider": current_cfg.provider, 
                "model": current_cfg.model,
                "roles_switched": role_info["roles_switched"],
                "actual_model": role_info["judge_model"]
            }
        })
        
        return new_state
    
    async def _refine_node(self, state: ReviewState) -> ReviewState:
        """Refinement node - updates phase for next main_ai call"""
        self.current_node = "refine"  # Track current active node
        self.last_state = state  # Update state for status callbacks
        
        # Call status callback for real-time updates
        if self.status_callback:
            self.status_callback()
        
        # Notify about refinement
        if self.message_callback:
            self.message_callback("orchestrator", f"Refining response (iteration {state['iteration'] + 1})...", {})
        
        new_state = state.copy()
        new_state["phase"] = "refine"
        new_state["iteration"] += 1
        return new_state
    
    async def _finalize_node(self, state: ReviewState) -> ReviewState:
        """Finalization node"""
        self.current_node = "finalize"  # Track current active node
        self.last_state = state  # Update state for status callbacks
        
        # Call status callback for real-time updates
        if self.status_callback:
            self.status_callback()
        
        # Notify about finalization
        if self.message_callback:
            self.message_callback("orchestrator", "Finalizing response...", {})
        
        summary = self._make_state_summary(state)
        
        fin_prompt = f"""{FINALIZE_USER_PROMPT}

Original prompt:
{state['original_prompt']}

Very short conversation summary:
{summary}
"""
        
        messages = [
            {"role": "system", "content": MAIN_SYSTEM_PROMPT},
            {"role": "user", "content": fin_prompt}
        ]
        
        req = ChatRequest(
            model=self.bundle.main_cfg.model,
            messages=messages,
            temperature=self.bundle.main_cfg.temperature,
            max_tokens=max(2048, self.bundle.main_cfg.max_tokens),
            stop=self.bundle.main_cfg.stop or None,
        )
        
        text, raw = await self.bundle.main_ai.chat(req)
        text = squeeze(text)
        
        # Display the final response
        if self.message_callback:
            self.message_callback("main", text, {"phase": "final"})
        
        # Update state
        new_state = state.copy()
        new_state["current_answer"] = text
        new_state["phase"] = "final"
        new_state["transcript"].append({
            "role": "main",
            "content": text,
            "meta": {"phase": "final"}
        })
        
        return new_state
    
    def _should_refine(self, state: ReviewState) -> str:
        """Conditional logic for refinement"""
        judge_feedback = state.get("judge_feedback") or {}
        
        # If judge says it passes and no user feedback, we're done
        if judge_feedback.get("pass", False) and not state.get("user_feedback"):
            if state["iteration"] >= 1:  # At least one refinement cycle
                return "finalize"
            else:
                return "refine"  # At least one refinement
        
        # If we've done too many iterations, finalize
        if state["iteration"] >= 3:
            return "finalize"
        
        # Otherwise, refine
        return "refine"
    
    def _make_state_summary(self, state: ReviewState, max_chars: int = 2000) -> str:
        """Create summary from state transcript"""
        parts = []
        for t in state["transcript"]:
            tag = {"you": "You", "main": "Main AI", "judge": "Judge AI", "orchestrator": "Orchestrator"}.get(t["role"], t["role"])
            cleaned_content = t["content"].replace('\n', ' ')
            parts.append(f"{tag}: {text_clip(cleaned_content, 180)}")
        joined = "\n".join(parts)
        return joined if len(joined) <= max_chars else (joined[:max_chars] + " ...")
    
    async def run_workflow(self, user_prompt: str, user_feedback: Optional[str] = None):
        """Run the complete review workflow"""
        initial_state: ReviewState = {
            "original_prompt": user_prompt,
            "current_answer": None,
            "judge_feedback": None,
            "user_feedback": user_feedback,
            "transcript": [{"role": "you", "content": user_prompt, "meta": {}}],
            "phase": "main",
            "iteration": 0,
            "roles_switched": False
        }
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        # Store for export functionality  
        self.last_state = final_state
        self.current_node = "completed"  # Mark as completed
        return final_state
    
    def get_graph_visualization(self, format: str = "mermaid") -> str:
        """Get graph visualization in different formats with current state highlighting"""
        base_graph = self.graph.get_graph()
        
        if format == "mermaid":
            mermaid_str = base_graph.draw_mermaid()
            # Add current node highlighting if available
            if self.current_node and self.current_node != "completed":
                # Add highlighting to the current active node
                mermaid_str += f"\n    classDef active fill:#f96,stroke:#333,stroke-width:4px"
                mermaid_str += f"\n    class {self.current_node} active"
            return mermaid_str
        elif format == "ascii":
            ascii_str = base_graph.draw_ascii()
            if self.current_node and self.current_node != "completed":
                ascii_str += f"\n\n[ACTIVE NODE: {self.current_node}]"
            return ascii_str
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_graph_image(self) -> bytes:
        """Get graph as PNG image"""
        return self.graph.get_graph().draw_mermaid_png()
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        status = {
            "current_node": self.current_node or "not_started",
            "is_active": self.current_node is not None and self.current_node != "completed",
            "is_completed": self.current_node == "completed"
        }
        
        if self.last_state:
            role_info = self._get_role_info(self.last_state)
            status.update({
                "iteration": self.last_state.get("iteration", 0),
                "phase": self.last_state.get("phase", "unknown"),
                "has_judge_feedback": bool(self.last_state.get("judge_feedback")),
                "has_user_feedback": bool(self.last_state.get("user_feedback")),
                "roles_switched": role_info["roles_switched"],
                "main_model": role_info["main_model"],
                "judge_model": role_info["judge_model"],
                "switch_iteration": role_info["switch_iteration"]
            })
        
        return status

def text_clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " ..."

def make_summary(transcript: List[ChatTurn], max_chars: int = 2000) -> str:
    parts = []
    for t in transcript:
        tag = {"you": "You", "main": "Main AI", "judge": "Judge AI", "orchestrator": "Orchestrator"}.get(t.role, t.role)
        cleaned_content = t.content.replace('\n', ' ')
        parts.append(f"{tag}: {text_clip(cleaned_content, 180)}")
    joined = "\n".join(parts)
    return joined if len(joined) <= max_chars else (joined[:max_chars] + " ...")
