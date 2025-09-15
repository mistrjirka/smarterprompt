#!/usr/bin/env python3
"""
SmarterPrompt - Unified CLI with TUI Monitoring
Usage: python main.py "Your prompt here"
"""

import asyncio
import sys
import json
import time
import os
import threading
from typing import Optional, Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align
from rich.live import Live



from orchestrator import load_config, build_bundle, LangGraphReviewLoop
from providers import get_ollama_models

# Load .env if present
load_dotenv()

class WorkflowGraphRenderer:
    """Renders the LangGraph workflow as ASCII art"""
    
    def __init__(self):
        self.nodes = {
            "main_ai": {"state": "inactive", "label": "Main AI"},
            "judge_ai": {"state": "inactive", "label": "Judge AI"},
            "refine": {"state": "inactive", "label": "Refine"},
            "finalize": {"state": "inactive", "label": "Finalize"}
        }
        self.current_node = None
        self.iteration = 0
        
    def update_status(self, current_node: Optional[str], iteration: int = 0):
        """Update which node is currently active"""
        self.current_node = current_node
        self.iteration = iteration
        
        # Reset all nodes
        for node in self.nodes.values():
            node["state"] = "inactive"
        
        # Set current node as active
        if current_node and current_node in self.nodes:
            self.nodes[current_node]["state"] = "active"
    
    def render(self) -> str:
        """Render the workflow graph as ASCII art"""
        lines = []
        
        # Title with iteration info
        title = f"LangGraph Workflow - Iteration {self.iteration}"
        lines.append(f"┌─── {title} ───┐")
        
        # Main flow: START → main_ai → judge_ai → finalize → END
        flow_line = "│ START ──"
        
        # Main AI
        if self.nodes["main_ai"]["state"] == "active":
            flow_line += "● main_ai"
        else:
            flow_line += "○ main_ai"
        
        flow_line += " ── "
        
        # Judge AI  
        if self.nodes["judge_ai"]["state"] == "active":
            flow_line += "● judge_ai"
        else:
            flow_line += "○ judge_ai"
        
        flow_line += " ──┐"
        lines.append(flow_line)
        
        # Decision branch
        lines.append("│                                │")
        
        # Finalize path
        finalize_line = "│                                ├─ "
        if self.nodes["finalize"]["state"] == "active":
            finalize_line += "● finalize"
        else:
            finalize_line += "○ finalize"
        finalize_line += " ── END"
        lines.append(finalize_line)
        
        # Refine loop
        refine_line = "│                                └─ "
        if self.nodes["refine"]["state"] == "active":
            refine_line += "● refine"
        else:
            refine_line += "○ refine"
        refine_line += " ──┐"
        lines.append(refine_line)
        
        lines.append("│                                     │")
        lines.append("└─────────────────────────────────────┘")
        
        # Add legend
        lines.append("")
        lines.append("● Active  ○ Inactive")
        
        return "\n".join(lines)

class SmarterPromptTUI:
    """Unified TUI interface for SmarterPrompt"""
    
    def __init__(self):
        self.console = Console()
        self.graph_renderer = WorkflowGraphRenderer()
        self.messages = []
        self.workflow_status = {
            "state": "initializing",
            "current_node": None,
            "iteration": 0,
            "roles_switched": False,
            "start_time": time.time(),
            "models": {}
        }
        self.workflow_task = None

        
    def add_message(self, role: str, content: str, meta: Optional[dict] = None):
        """Add message to the display"""
        message = {
            "role": role,
            "content": content,
            "meta": meta or {},
            "timestamp": time.time()
        }
        self.messages.append(message)
    
    def update_workflow_status(self, **kwargs):
        """Update workflow status"""
        self.workflow_status.update(kwargs)
        
        # Update graph renderer
        self.graph_renderer.update_status(
            self.workflow_status.get("current_node"),
            self.workflow_status.get("iteration", 0)
        )
    
    def render_header(self) -> Panel:
        """Render the header panel"""
        return Panel(
            Align.center(Text("🤖 SmarterPrompt - AI Workflow System", style="bold blue")),
            style="bold blue"
        )
    
    def render_status_panel(self) -> Panel:
        """Render the status panel"""
        state = self.workflow_status.get("state", "unknown")
        current_node = self.workflow_status.get("current_node", "unknown")
        iteration = self.workflow_status.get("iteration", 0)
        roles_switched = self.workflow_status.get("roles_switched", False)
        
        # Status indicators
        if state == "running":
            status_text = "[bold green]● RUNNING[/]"
            border_style = "green"
        elif state == "completed":
            status_text = "[bold blue]✓ COMPLETED[/]"
            border_style = "blue"
        elif state == "error":
            status_text = "[bold red]✗ ERROR[/]"
            border_style = "red"
        else:
            status_text = f"[yellow]{state.upper()}[/]"
            border_style = "yellow"
        
        # Build status info
        info_lines = [
            f"State: {status_text}",
            f"Current Node: [bold]{current_node}[/]",
            f"Iteration: [bold]{iteration}[/]",
        ]
        
        if roles_switched:
            info_lines.append("Roles: [yellow]● SWITCHED[/]")
        
        # Add models info
        models = self.workflow_status.get("models", {})
        if models:
            info_lines.append("")
            info_lines.append("Models:")
            info_lines.append(f"  Main: [cyan]{models.get('main', 'unknown')}[/]")
            info_lines.append(f"  Judge: [cyan]{models.get('judge', 'unknown')}[/]")
        
        # Add timing info
        start_time = self.workflow_status.get("start_time")
        if start_time:
            elapsed = time.time() - start_time
            info_lines.append(f"Elapsed: [dim]{elapsed:.1f}s[/]")
        
        return Panel(
            "\n".join(info_lines),
            title="Workflow Status",
            border_style=border_style
        )
    
    def render_graph_panel(self) -> Panel:
        """Render the workflow graph panel"""
        graph_text = self.graph_renderer.render()
        
        return Panel(
            graph_text,
            title="Workflow Graph",
            border_style="cyan"
        )
    
    def render_messages_panel(self) -> Panel:
        """Render the messages panel with scrolling support"""
        if not self.messages:
            return Panel(
                Align.center("Waiting for workflow to start...\n\n[dim]Navigation: ↑/↓ scroll, Enter to expand message[/]"),
                title="AI Responses",
                border_style="dim"
            )
        

        
        # Show more messages and prioritize final responses
        messages_per_page = 6  # Show more messages
        
        # Always show the most recent messages (including final response)
        start_idx = max(0, len(self.messages) - messages_per_page)
        end_idx = len(self.messages)
        
        visible_messages = self.messages[start_idx:end_idx]
        
        content_lines = []
        
        for i, msg in enumerate(visible_messages):
            actual_index = start_idx + i
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", 0)
            meta = msg.get("meta", {})
            
            # Format timestamp
            dt = datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%H:%M:%S")
            
            # Role colors
            role_colors = {
                "you": "blue",
                "main": "green", 
                "judge": "yellow",
                "orchestrator": "magenta"
            }
            color = role_colors.get(role, "white")
            
            # Role names
            role_names = {
                "you": "You",
                "main": "Main AI",
                "judge": "Judge AI",
                "orchestrator": "Orchestrator"
            }
            role_name = role_names.get(role, role.title())
            
            # Add prefix for final responses
            if meta.get("phase") == "final":
                prefix = "🎯 "
            else:
                prefix = "   "
            
            # Check if content is truncated
            is_truncated = len(content) > 400
            display_content = content[:400] + "... [dim](truncated)[/]" if is_truncated else content
            
            content_lines.append(f"{prefix}[{color}][{time_str}] {role_name}:[/]")
            
            # Add judge score if available
            if role == "judge" and "parsed" in meta:
                parsed = meta["parsed"]
                if isinstance(parsed, dict):
                    score = parsed.get("score", "N/A")
                    passed = parsed.get("pass", False)
                    status = "✅ PASS" if passed else "❌ NEEDS WORK"
                    content_lines.append(f"    [dim]Score: {score} | {status}[/]")
            
            # Add role switch info
            if meta.get("roles_switched"):
                content_lines.append(f"    [yellow]🔄 Roles switched - using: {meta.get('actual_model', 'unknown')}[/]")
            
            # Add phase info for final responses
            if meta.get("phase") == "final":
                content_lines.append(f"    [bold green]✨ FINAL RESPONSE ✨[/]")
            
            # Add content (with word wrapping)
            words = display_content.split()
            if words:
                line = "    "
                for word in words:
                    # Handle rich markup in word wrapping
                    word_len = len(word) - word.count("[") * 2 if "[" in word else len(word)
                    
                    if len(line) + word_len > 75:
                        content_lines.append(line)
                        line = "    " + word
                    else:
                        line += " " + word if line != "    " else word
                if line.strip():
                    content_lines.append(line)
            
            content_lines.append("")
        
        title = f"Recent Messages ({len(self.messages)} total)"
        if len(self.messages) > messages_per_page:
            title += f" - Showing last {min(messages_per_page, len(self.messages))}"
        
        panel_content = "\n".join(content_lines)
        
        return Panel(
            panel_content,
            title=title,
            border_style="magenta"
        )
    

    
    def render_layout(self) -> Layout:
        """Create the main layout"""
        layout = Layout()
        
        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=2)
        )
        
        # Header
        layout["header"].update(self.render_header())
        
        # Body - split into top and bottom
        layout["body"].split_column(
            Layout(name="top_row", size=15),
            Layout(name="messages")
        )
        
        # Top row - status and graph side by side
        layout["top_row"].split_row(
            Layout(self.render_status_panel(), name="status"),
            Layout(self.render_graph_panel(), name="graph")
        )
        
        # Messages
        layout["messages"].update(self.render_messages_panel())
        
        # Footer
        state = self.workflow_status.get("state", "unknown")
        if state == "completed":
            footer_text = "[dim]Workflow completed! Press Ctrl+C to exit. Check transcript.json for full messages."
        else:
            footer_text = "[dim]Press Ctrl+C to exit | Real-time workflow monitoring"
        
        layout["footer"].update(
            Panel(footer_text, style="dim")
        )
        
        return layout
    
    async def run_workflow(self, prompt: str):
        """Run the workflow in background"""
        try:
            self.update_workflow_status(state="configuring")
            
            # Load configuration
            config = load_config()
            
            # Get Ollama models
            try:
                ollama_models = get_ollama_models()
            except Exception as e:
                self.add_message("system", f"Warning: Could not get Ollama models: {e}")
                ollama_models = ['llama3', 'gemma2']
            
            # Configure models
            if ollama_models:
                main_model = ollama_models[0]
                judge_model = ollama_models[1] if len(ollama_models) > 1 else ollama_models[0]
            else:
                main_model = "llama3"
                judge_model = "gemma2"
            
            workflow_config = {
                "main_ai": {
                    "provider": "ollama",
                    "model": main_model,
                    "temperature": 0.2,
                    "max_tokens": 2000,
                    "stop": []
                },
                "judge_ai": {
                    "provider": "ollama", 
                    "model": judge_model,
                    "temperature": 0.1,
                    "max_tokens": 1500,
                    "stop": []
                },
                "openai": config.get("openai", {}),
                "ollama": config.get("ollama", {})
            }
            
            self.update_workflow_status(
                state="building_workflow",
                models={
                    "main": f"ollama:{main_model}",
                    "judge": f"ollama:{judge_model}"
                }
            )
            
            bundle = build_bundle(workflow_config)
            
            def message_callback(role: str, content: str, meta: Optional[dict] = None):
                self.add_message(role, content, meta)
            
            def status_callback():
                pass  # Status updates handled by monitoring thread
            
            # Create workflow
            workflow = LangGraphReviewLoop(
                bundle,
                message_callback=message_callback,
                status_callback=status_callback
            )
            
            self.add_message("you", prompt)
            self.update_workflow_status(state="running", current_node="main_ai")
            
            # Start status monitoring thread
            def monitor_status():
                while self.workflow_status.get("state") == "running":
                    try:
                        current_status = workflow.get_current_status()
                        self.update_workflow_status(
                            iteration=current_status.get('iteration', 0),
                            roles_switched=current_status.get('roles_switched', False)
                        )
                        time.sleep(0.5)
                    except:
                        pass
            
            status_thread = threading.Thread(target=monitor_status, daemon=True)
            status_thread.start()
            
            # Run workflow
            final_state = await workflow.run_workflow(prompt)
            
            # Get final status
            status = workflow.get_current_status()
            iteration = status.get('iteration', 0)
            roles_switched = status.get('roles_switched', False)
            
            self.update_workflow_status(
                state="completed",
                current_node="finished",
                iteration=iteration,
                roles_switched=roles_switched
            )
            

            
            # Save transcript
            transcript_file = "transcript.json"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.update_workflow_status(state="error", error=str(e))
            self.add_message("system", f"Error: {e}")
    
    async def run_tui(self, prompt: str):
        """Main TUI loop"""
        # Start workflow in background
        self.workflow_task = asyncio.create_task(self.run_workflow(prompt))
        
        # Run TUI display
        with Live(self.render_layout(), refresh_per_second=2, console=self.console) as live:
            try:
                while True:
                    # Update display
                    live.update(self.render_layout())
                    
                    # Check if workflow is done
                    if self.workflow_task.done():
                        # Show completion for a moment, then exit to browser
                        await asyncio.sleep(3)
                        break
                    
                    await asyncio.sleep(0.5)
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Stopping workflow...[/]")
                if self.workflow_task and not self.workflow_task.done():
                    self.workflow_task.cancel()
                    try:
                        await self.workflow_task
                    except asyncio.CancelledError:
                        pass

def show_message_browser(messages: List[Dict[str, Any]]):
    """Simple message browser after workflow completion"""
    console = Console()
    console.clear()
    
    console.print("\n[bold green]✅ Workflow Completed![/]")
    console.print(f"[dim]Total messages: {len(messages)}[/]\n")
    
    # Show all messages in a readable format
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", 0)
        meta = msg.get("meta", {})
        
        dt = datetime.fromtimestamp(timestamp)
        time_str = dt.strftime("%H:%M:%S")
        
        role_names = {
            "you": "You", "main": "Main AI", 
            "judge": "Judge AI", "orchestrator": "Orchestrator"
        }
        role_name = role_names.get(role, role.title())
        
        # Role colors
        role_colors = {
            "you": "blue", "main": "green", 
            "judge": "yellow", "orchestrator": "magenta"
        }
        color = role_colors.get(role, "white")
        
        console.print(f"\n[bold]{i+1}. [{color}][{time_str}] {role_name}:[/][/]")
        
        # Special highlighting for final response
        if meta.get("phase") == "final":
            console.print("[bold green]🎯 FINAL RESPONSE[/]")
        
        # Judge score
        if role == "judge" and "parsed" in meta:
            parsed = meta["parsed"]
            if isinstance(parsed, dict):
                score = parsed.get("score", "N/A")
                passed = parsed.get("pass", False)
                status = "✅ PASS" if passed else "❌ NEEDS WORK"
                console.print(f"[dim]Score: {score} | {status}[/]")
        
        # Role switching info
        if meta.get("roles_switched"):
            console.print(f"[yellow]🔄 Roles switched - using: {meta.get('actual_model', 'unknown')}[/]")
        
        console.print()
        
        # Content with smart truncation
        if len(content) > 800:
            console.print(content[:800])
            console.print(f"[dim]... (truncated, {len(content)} chars total. See transcript.json for full content)[/]")
        else:
            console.print(content)
        
        console.print("[dim]" + "─" * 60 + "[/]")
    
    console.print(f"\n[green]✨ Full transcript saved to: transcript.json[/]")
    console.print("[dim]Press Enter to exit...[/]")
    
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass

async def run_interactive_mode(tui: SmarterPromptTUI, prompt: str):
    """Run TUI workflow then show message browser"""
    # Run the workflow with TUI monitoring
    await tui.run_tui(prompt)
    
    # After workflow completes, show message browser
    if tui.messages:
        show_message_browser(tui.messages)

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        console = Console()
        console.print("[red]Usage:[/] python main.py \"Your prompt here\"")
        console.print("        python main.py --no-browser \"Your prompt here\"")
        console.print("\n[yellow]Examples:[/]")
        console.print('python main.py "Write a Python function to calculate fibonacci numbers"')
        console.print('python main.py --no-browser "Just run workflow, no message browser"')
        console.print("\n[dim]Default: Run workflow with TUI monitoring, then show message browser[/]")
        sys.exit(1)
    
    show_browser = True
    if sys.argv[1] == "--no-browser":
        show_browser = False
        if len(sys.argv) < 3:
            console = Console()
            console.print("[red]Error:[/] Prompt required after --no-browser")
            sys.exit(1)
        prompt = " ".join(sys.argv[2:])
    else:
        prompt = " ".join(sys.argv[1:])
    
    tui = SmarterPromptTUI()
    
    try:
        # Always run TUI workflow
        asyncio.run(tui.run_tui(prompt))
        
        # Show message browser by default (unless --no-browser)
        if show_browser and tui.messages:
            show_message_browser(tui.messages)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()