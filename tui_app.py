#!/usr/bin/env python3
"""
TUI (Terminal User Interface) version of SmarterPrompt
Uses rich library for interactive command-line interface
"""

import asyncio
import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich import print as rprint

from orchestrator import load_config, build_bundle, ReviewLoop, LangGraphReviewLoop, ProvidersBundle
from providers import get_ollama_models

# Load .env if present
load_dotenv()

class TUIApp:
    def __init__(self):
        self.console = Console()
        self.config = load_config()
        self.session_loop: Optional[ReviewLoop] = None
        self.langgraph_loop: Optional[LangGraphReviewLoop] = None
        self.current_bundle: Optional[ProvidersBundle] = None
        self.transcript = []
        self.workflow_type = "langgraph"
        
    def show_header(self):
        """Display application header"""
        header = Panel.fit(
            "[bold blue]🤖 SmarterPrompt TUI[/bold blue]\n"
            "[dim]AI Review Loop with Role Switching[/dim]",
            border_style="blue"
        )
        self.console.print(header)
        self.console.print()

    def get_available_models(self, provider: str) -> List[str]:
        """Get available models for a provider"""
        if provider == 'ollama':
            try:
                return get_ollama_models()
            except Exception as e:
                self.console.print(f"[red]Error getting Ollama models: {e}[/red]")
                return ['llama3', 'gemma2']
        elif provider == 'openai':
            return ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o1-preview', 'o1-mini']
        return []

    def configure_models(self):
        """Interactive model configuration"""
        self.console.print("[bold cyan]🔧 Model Configuration[/bold cyan]")
        
        # Main AI configuration
        self.console.print("\n[yellow]Main AI Configuration:[/yellow]")
        main_provider = Prompt.ask(
            "Main provider", 
            choices=["ollama", "openai"], 
            default="ollama"
        )
        
        main_models = self.get_available_models(main_provider)
        if main_models:
            self.console.print(f"Available models: {', '.join(main_models)}")
            main_model = Prompt.ask(
                "Main model", 
                choices=main_models,
                default=main_models[0] if main_models else "llama3"
            )
        else:
            main_model = Prompt.ask("Main model", default="llama3")
        
        main_temp = float(Prompt.ask("Main temperature", default="0.2"))
        main_tokens = int(Prompt.ask("Main max tokens", default="2000"))
        
        # Judge AI configuration
        self.console.print("\n[yellow]Judge AI Configuration:[/yellow]")
        judge_provider = Prompt.ask(
            "Judge provider", 
            choices=["ollama", "openai"], 
            default="ollama"
        )
        
        judge_models = self.get_available_models(judge_provider)
        if judge_models:
            self.console.print(f"Available models: {', '.join(judge_models)}")
            judge_model = Prompt.ask(
                "Judge model", 
                choices=judge_models,
                default=judge_models[1] if len(judge_models) > 1 else judge_models[0] if judge_models else "llama3"
            )
        else:
            judge_model = Prompt.ask("Judge model", default="llama3")
        
        judge_temp = float(Prompt.ask("Judge temperature", default="0.1"))
        judge_tokens = int(Prompt.ask("Judge max tokens", default="1500"))
        
        # Workflow type
        self.workflow_type = Prompt.ask(
            "Workflow type", 
            choices=["langgraph", "original"], 
            default="langgraph"
        )
        
        # Build configuration bundle
        config = {
            "main_ai": {
                "provider": main_provider,
                "model": main_model,
                "temperature": main_temp,
                "max_tokens": main_tokens,
                "stop": []
            },
            "judge_ai": {
                "provider": judge_provider,
                "model": judge_model,
                "temperature": judge_temp,
                "max_tokens": judge_tokens,
                "stop": []
            },
            "openai": self.config.get("openai", {}),
            "ollama": self.config.get("ollama", {})
        }
        
        self.current_bundle = build_bundle(config)
        
        # Show configuration summary
        table = Table(title="Configuration Summary", show_header=True, header_style="bold magenta")
        table.add_column("Role", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Temperature", style="blue")
        table.add_column("Max Tokens", style="red")
        
        table.add_row("Main AI", main_provider, main_model, str(main_temp), str(main_tokens))
        table.add_row("Judge AI", judge_provider, judge_model, str(judge_temp), str(judge_tokens))
        
        self.console.print()
        self.console.print(table)
        self.console.print(f"[dim]Workflow: {self.workflow_type}[/dim]")
        self.console.print()

    def display_message(self, role: str, content: str, meta: Optional[dict] = None):
        """Display a message in the console"""
        role_colors = {
            'you': 'green',
            'main': 'blue', 
            'judge': 'yellow',
            'orchestrator': 'magenta'
        }
        
        role_names = {
            'you': 'You',
            'main': 'Main AI',
            'judge': 'Judge AI', 
            'orchestrator': 'Orchestrator'
        }
        
        color = role_colors.get(role, 'white')
        name = role_names.get(role, role.title())
        
        # Truncate very long messages for display
        display_content = content
        if len(content) > 1000:
            display_content = content[:1000] + "\n[dim]... (truncated, full content saved to transcript)[/dim]"
        
        panel = Panel(
            display_content,
            title=f"[{color}]{name}[/{color}]",
            title_align="left",
            border_style=color,
            padding=(0, 1)
        )
        
        self.console.print(panel)
        
        # Show metadata if present and relevant
        if meta and role == 'judge' and 'parsed' in meta:
            parsed = meta['parsed']
            if isinstance(parsed, dict):
                score = parsed.get('score', 'N/A')
                passed = parsed.get('pass', False)
                status = "✅ PASS" if passed else "❌ NEEDS WORK"
                self.console.print(f"[dim]Judge Score: {score} | Status: {status}[/dim]")
        
        # Show role switching info
        if meta and meta.get('roles_switched'):
            self.console.print(f"[dim]🔄 Roles switched - using: {meta.get('actual_model', 'unknown')}[/dim]")
        
        self.console.print()

    async def run_workflow(self, user_prompt: str):
        """Run the selected workflow"""
        if not self.current_bundle:
            self.console.print("[red]Please configure models first![/red]")
            return
        
        self.transcript = []
        
        # Store user prompt
        self.transcript.append({"role": "you", "content": user_prompt, "meta": {}})
        self.display_message("you", user_prompt)
        
        if self.workflow_type == "langgraph":
            await self.run_langgraph_workflow(user_prompt)
        else:
            await self.run_original_workflow(user_prompt)

    async def run_langgraph_workflow(self, user_prompt: str):
        """Run LangGraph workflow with progress tracking"""
        # Create workflow with callbacks
        def message_callback(role: str, content: str, meta: Optional[dict] = None):
            self.transcript.append({"role": role, "content": content, "meta": meta or {}})
            self.display_message(role, content, meta)
        
        def status_callback():
            # Could add status updates here if needed
            pass
        
        if not self.current_bundle:
            self.console.print("[red]No configuration bundle available![/red]")
            return
            
        self.langgraph_loop = LangGraphReviewLoop(
            self.current_bundle,
            message_callback=message_callback,
            status_callback=status_callback
        )
        
        self.display_message("orchestrator", "Starting LangGraph workflow...")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("Running workflow...", total=None)
                
                final_state = await self.langgraph_loop.run_workflow(user_prompt)
                
                progress.update(task, description="Workflow completed!")
            
            # Get status info
            status = self.langgraph_loop.get_current_status()
            iteration = status.get('iteration', 0)
            roles_switched = status.get('roles_switched', False)
            
            completion_msg = f"Workflow completed after {iteration} iterations."
            if roles_switched:
                completion_msg += " Roles were switched during execution."
            
            self.display_message("orchestrator", completion_msg)
            
        except Exception as e:
            self.console.print(f"[red]Error running workflow: {e}[/red]")

    async def run_original_workflow(self, user_prompt: str):
        """Run original workflow"""
        try:
            if not self.current_bundle:
                self.console.print("[red]No configuration bundle available![/red]")
                return
                
            self.session_loop = ReviewLoop(self.current_bundle)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                
                # Main AI
                main_task = progress.add_task("Running Main AI...", total=None)
                main_ans = await self.session_loop.run_main(user_prompt)
                self.transcript.append({"role": "main", "content": main_ans, "meta": {}})
                self.display_message("main", main_ans)
                progress.remove_task(main_task)
                
                # Judge AI
                judge_task = progress.add_task("Running Judge AI...", total=None)
                judge_json = await self.session_loop.run_judge()
                self.transcript.append({"role": "judge", "content": str(judge_json), "meta": {"parsed": judge_json}})
                self.display_message("judge", str(judge_json), {"parsed": judge_json})
                progress.remove_task(judge_task)
                
                # Refine
                refine_task = progress.add_task("Refining response...", total=None)
                refined = await self.session_loop.refine()
                self.transcript.append({"role": "main", "content": refined, "meta": {"phase": "refined"}})
                self.display_message("main", refined, {"phase": "refined"})
                progress.remove_task(refine_task)
            
            self.display_message("orchestrator", "Review cycle complete!")
            
        except Exception as e:
            self.console.print(f"[red]Error running workflow: {e}[/red]")

    async def iterate_with_feedback(self, feedback: str):
        """Iterate with user feedback"""
        if not self.session_loop and not self.langgraph_loop:
            self.console.print("[red]No active workflow to iterate![/red]")
            return
        
        self.display_message("you", f"Feedback: {feedback}")
        
        try:
            if self.workflow_type == "langgraph" and self.langgraph_loop:
                # Re-run with feedback
                final_state = await self.langgraph_loop.run_workflow(
                    self.transcript[0]["content"],  # Original prompt
                    user_feedback=feedback
                )
                self.display_message("orchestrator", "Iteration completed with feedback!")
            elif self.session_loop:
                refined = await self.session_loop.refine(feedback)
                self.transcript.append({"role": "main", "content": refined, "meta": {"phase": "feedback_refined"}})
                self.display_message("main", refined, {"phase": "feedback_refined"})
                self.display_message("orchestrator", "Iteration completed with feedback!")
        except Exception as e:
            self.console.print(f"[red]Error during iteration: {e}[/red]")

    def export_transcript(self):
        """Export transcript to JSON file"""
        if not self.transcript:
            self.console.print("[red]No transcript to export![/red]")
            return
        
        filename = "transcript_tui.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.transcript, f, ensure_ascii=False, indent=2)
            self.console.print(f"[green]Transcript exported to {filename}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error exporting transcript: {e}[/red]")

    def show_menu(self):
        """Show main menu options"""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Option", style="cyan")
        table.add_column("Description", style="white")
        
        table.add_row("1", "Configure models")
        table.add_row("2", "Run workflow")
        table.add_row("3", "Iterate with feedback")
        table.add_row("4", "Export transcript")
        table.add_row("5", "Show current config")
        table.add_row("q", "Quit")
        
        panel = Panel(table, title="[bold]Menu Options[/bold]", border_style="cyan")
        self.console.print(panel)

    def show_current_config(self):
        """Show current configuration"""
        if not self.current_bundle:
            self.console.print("[red]No configuration set![/red]")
            return
        
        table = Table(title="Current Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Workflow Type", self.workflow_type)
        table.add_row("Main Provider", self.current_bundle.main_cfg.provider)
        table.add_row("Main Model", self.current_bundle.main_cfg.model)
        table.add_row("Judge Provider", self.current_bundle.judge_cfg.provider)
        table.add_row("Judge Model", self.current_bundle.judge_cfg.model)
        
        self.console.print(table)

    async def run(self):
        """Main application loop"""
        self.show_header()
        
        while True:
            self.show_menu()
            
            choice = Prompt.ask("\n[bold]Choose an option[/bold]", default="1")
            self.console.print()
            
            if choice == "1":
                self.configure_models()
            elif choice == "2":
                if not self.current_bundle:
                    self.console.print("[red]Please configure models first![/red]")
                    continue
                
                # Better multiline input handling
                self.console.print("[bold cyan]Enter your task prompt[/bold cyan]")
                self.console.print("[dim]Press Ctrl+D when finished, or type END on a new line[/dim]")
                
                lines = []
                try:
                    while True:
                        try:
                            line = input("> ")
                            if line.strip().upper() == "END":
                                break
                            lines.append(line)
                        except EOFError:
                            break
                except KeyboardInterrupt:
                    self.console.print("[red]Input cancelled[/red]")
                    continue
                
                prompt = "\n".join(lines).strip()
                if prompt:
                    await self.run_workflow(prompt)
                else:
                    self.console.print("[red]Please enter a valid prompt![/red]")
            elif choice == "3":
                # Better multiline feedback input
                self.console.print("[bold cyan]Enter your feedback[/bold cyan]")
                self.console.print("[dim]Press Ctrl+D when finished, or type END on a new line[/dim]")
                
                lines = []
                try:
                    while True:
                        try:
                            line = input("> ")
                            if line.strip().upper() == "END":
                                break
                            lines.append(line)
                        except EOFError:
                            break
                except KeyboardInterrupt:
                    self.console.print("[red]Input cancelled[/red]")
                    continue
                
                feedback = "\n".join(lines).strip()
                if feedback:
                    await self.iterate_with_feedback(feedback)
                else:
                    self.console.print("[red]Please enter valid feedback![/red]")
            elif choice == "4":
                self.export_transcript()
            elif choice == "5":
                self.show_current_config()
            elif choice.lower() == "q":
                self.console.print("[bold green]Thanks for using SmarterPrompt TUI! 👋[/bold green]")
                break
            else:
                self.console.print("[red]Invalid option! Please try again.[/red]")
            
            if choice != "q":
                Prompt.ask("\n[dim]Press Enter to continue...[/dim]", default="")
                self.console.clear()
                self.show_header()

def main():
    """Entry point"""
    app = TUIApp()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()