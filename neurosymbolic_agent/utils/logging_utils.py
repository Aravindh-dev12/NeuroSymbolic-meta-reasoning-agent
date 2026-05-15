"""
utils/logging_utils.py — Structured logging with Rich console output.
"""
from __future__ import annotations

import sys
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# Rich console with custom theme
theme = Theme({
    "neural": "bold cyan",
    "symbolic": "bold magenta",
    "meta": "bold yellow",
    "constitutional": "bold red",
    "memory": "bold green",
    "planning": "bold blue",
    "success": "bold green",
    "failure": "bold red",
    "warning": "bold orange1",
})

console = Console(theme=theme)


def setup_logging(log_file: str = "logs/agent.log", level: str = "INFO") -> None:
    """Configure loguru logging."""
    import os
    os.makedirs("logs", exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
        colorize=True,
    )
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
        rotation="10 MB",
        retention="7 days",
    )


def print_routing_decision(task: str, path: str, confidence: float, reason: str) -> None:
    """Pretty-print routing decision."""
    color = "neural" if path == "neural" else "symbolic" if path == "symbolic" else "meta"
    console.print(Panel(
        f"[bold]Task:[/bold] {task[:80]}...\n"
        f"[bold]Path:[/bold] [{color}]{path.upper()}[/{color}]\n"
        f"[bold]Confidence:[/bold] {confidence:.2%}\n"
        f"[bold]Reason:[/bold] {reason}",
        title="[meta]🧠 Routing Decision[/meta]",
        border_style="yellow",
    ))


def print_self_improvement(round_n: int, critique: str, corrected: bool) -> None:
    """Pretty-print self-improvement round."""
    status = "[success]✓ Corrected[/success]" if corrected else "[warning]⚠ Needs more work[/warning]"
    console.print(Panel(
        f"[bold]Round:[/bold] {round_n}\n"
        f"[bold]Critique:[/bold] {critique[:200]}\n"
        f"[bold]Status:[/bold] {status}",
        title="[meta]🔄 Self-Improvement Loop[/meta]",
        border_style="cyan",
    ))


def print_constitutional_check(violated: bool, violations: list[str]) -> None:
    """Pretty-print constitutional check result."""
    if violated:
        console.print(Panel(
            "\n".join(f"• {v}" for v in violations),
            title="[constitutional]⚖️ Constitutional Violations[/constitutional]",
            border_style="red",
        ))
    else:
        console.print("[success]✓ Constitutional check passed[/success]")


def print_final_answer(answer: str, confidence: float, path_used: str) -> None:
    """Pretty-print final answer."""
    console.print(Panel(
        f"{answer}\n\n"
        f"[dim]Confidence: {confidence:.2%} | Path: {path_used}[/dim]",
        title="[success]✅ Final Answer[/success]",
        border_style="green",
    ))
