"""Command line interface for the NeuroSymbolic Meta-Reasoning Agent."""
from __future__ import annotations

import argparse
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# The existing modules use package-local absolute imports such as
# ``from agent.meta_controller import ...``. Keep that runtime contract intact
# for installed console scripts.
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

console = Console()


def console_safe(value) -> str:
    encoding = getattr(console.file, "encoding", None) or sys.stdout.encoding or "utf-8"
    return str(value).encode(encoding, errors="replace").decode(encoding, errors="replace")


def default_config_path() -> str:
    return str(PACKAGE_DIR / "configs" / "agent_config.yaml")


def apply_runtime_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "backend", None):
        os.environ["LLM_BACKEND"] = args.backend
    if getattr(args, "model", None):
        backend = getattr(args, "backend", None) or os.getenv("LLM_BACKEND", "local")
        if backend == "local":
            os.environ["LOCAL_MODEL_NAME"] = args.model
        elif backend == "openai":
            os.environ["OPENAI_MODEL"] = args.model
        elif backend == "anthropic":
            os.environ["ANTHROPIC_MODEL"] = args.model
    if getattr(args, "no_telemetry", False):
        os.environ["ENABLE_TELEMETRY"] = "false"
    if getattr(args, "no_fallback", False):
        os.environ["ALLOW_HEURISTIC_FALLBACK"] = "false"


def build_agent(args: argparse.Namespace):
    apply_runtime_overrides(args)
    from main import NeuroSymbolicAgent

    return NeuroSymbolicAgent(config_path=args.config)


def cmd_run(args: argparse.Namespace) -> int:
    task = args.task or " ".join(args.task_words or [])
    if not task.strip():
        console.print("[red]Provide a task, for example: neuro-agent run \"Solve 2x + 3 = 7\"[/red]")
        return 2

    agent = build_agent(args)
    try:
        answer = agent.run(task.strip())
        console.print("\n[bold]Final answer[/bold]")
        console.print(console_safe(answer))
    finally:
        agent.cleanup()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    agent = build_agent(args)
    try:
        console.print("\n[bold]NeuroSymbolic Agent interactive mode[/bold]")
        console.print("[dim]Type 'exit', 'quit', or Ctrl+C to stop.[/dim]\n")
        while True:
            try:
                task = input("Task > ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if task.lower() in {"exit", "quit", "q"}:
                break
            if task:
                agent.run(task)
                print()
    finally:
        agent.cleanup()
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from llm.local_llm_manager import LocalLLMManager

    table = Table(title="Local open-weight model catalog")
    table.add_column("Name", style="cyan")
    table.add_column("Runtime")
    table.add_column("Model ID")
    table.add_column("Context")

    table.add_row("auto", "auto", "Prefer Ollama when available, otherwise llama.cpp", "-")
    for name, config in sorted(LocalLLMManager.PRODUCTION_MODELS.items()):
        table.add_row(
            name,
            config.backend.value,
            config.model_id,
            str(config.context_length),
        )
    console.print(table)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []

    def import_check(module_name: str, label: str) -> None:
        try:
            __import__(module_name)
            checks.append((label, True, "installed"))
        except Exception as exc:
            checks.append((label, False, str(exc)))

    import_check("torch", "PyTorch")
    import_check("sentence_transformers", "SentenceTransformers")
    import_check("z3", "Z3")
    import_check("chromadb", "ChromaDB")
    import_check("llama_cpp", "llama.cpp Python")

    ollama_path = shutil.which("ollama")
    if ollama_path:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            checks.append(("Ollama", result.returncode == 0, "running" if result.returncode == 0 else "installed but not reachable"))
        except Exception as exc:
            checks.append(("Ollama", False, str(exc)))
    else:
        checks.append(("Ollama", False, "not installed"))

    table = Table(title="Agent environment doctor")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")
    for label, ok, detail in checks:
        table.add_row(label, "OK" if ok else "WARN", detail)
    console.print(table)
    return 0 if all(ok for _, ok, _ in checks[:3]) else 1


def cmd_serve(args: argparse.Namespace) -> int:
    apply_runtime_overrides(args)
    app_path = PROJECT_ROOT / "app.py"
    if not app_path.exists():
        console.print("[red]app.py was not found; the Gradio dashboard is unavailable in this install.[/red]")
        return 1
    runpy.run_path(str(app_path), run_name="__main__")
    return 0


def build_parser(prog: str = "neuro-agent") -> argparse.ArgumentParser:
    def add_runtime_options(target: argparse.ArgumentParser) -> None:
        target.add_argument("--config", default=argparse.SUPPRESS, help="Path to agent_config.yaml")
        target.add_argument("--backend", choices=["local", "anthropic", "openai"], default=argparse.SUPPRESS, help="LLM backend override")
        target.add_argument("--model", default=argparse.SUPPRESS, help="Model override, for example auto, ollama-qwen2.5, llama3-8b")
        target.add_argument("--no-telemetry", action="store_true", default=argparse.SUPPRESS, help="Disable telemetry for this run")
        target.add_argument("--no-fallback", action="store_true", default=argparse.SUPPRESS, help="Disable offline heuristic fallback when local LLM startup fails")

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run the NeuroSymbolic Meta-Reasoning Agent from the command line.",
    )
    parser.add_argument("--config", default=default_config_path(), help="Path to agent_config.yaml")
    parser.add_argument("--backend", choices=["local", "anthropic", "openai"], help="LLM backend override")
    parser.add_argument("--model", help="Model override, for example auto, ollama-qwen2.5, llama3-8b")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable telemetry for this run")
    parser.add_argument("--no-fallback", action="store_true", help="Disable offline heuristic fallback when local LLM startup fails")

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Execute a single task")
    add_runtime_options(run_parser)
    run_parser.add_argument("task_words", nargs="*", help="Task text")
    run_parser.add_argument("--task", help="Task text, useful for scripts")
    run_parser.set_defaults(func=cmd_run)

    chat_parser = subparsers.add_parser("chat", aliases=["start", "activate"], help="Start interactive mode")
    add_runtime_options(chat_parser)
    chat_parser.set_defaults(func=cmd_chat)

    models_parser = subparsers.add_parser("models", help="List local/open-weight model options")
    models_parser.set_defaults(func=cmd_models)

    doctor_parser = subparsers.add_parser("doctor", help="Check local runtime dependencies")
    doctor_parser.set_defaults(func=cmd_doctor)

    serve_parser = subparsers.add_parser("serve", help="Launch the Gradio dashboard")
    add_runtime_options(serve_parser)
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    prog = Path(sys.argv[0]).stem if argv is None else "neuro-agent"
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        return cmd_chat(args)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
