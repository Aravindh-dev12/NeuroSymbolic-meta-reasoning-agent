"""
main.py — Production-grade NeuroSymbolic AGI Agent entry point.

Orchestrates the full pipeline with local LLM support:
  Task → MetaController → ReasoningEngine → SelfImprovement → ConstitutionalCheck → Output
Supports local LLMs (llama.cpp, Ollama) and cloud LLMs (Anthropic, OpenAI).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from rich.console import Console

from agent.meta_controller import MetaController
from agent.reasoning_engine import ReasoningEngine
from agent.self_improvement import SelfImprovementLoop
from constitutional.checker import ConstitutionalChecker
from constitutional.principles import PrinciplesRegistry
from constitutional.reward_hacking_detector import RewardHackingDetector
from llm.local_llm_manager import create_llm_manager, LocalLLMManager
from memory.memory_manager import MemoryManager
from neural.embedder import Embedder
from neural.classifier import TaskClassifier
from planning.hierarchical_planner import HierarchicalPlanner
from planning.fallback_strategies import FallbackStrategyEngine, FallbackType
from symbolic.knowledge_base import KnowledgeBase
from utils.config import load_config, load_principles
from utils.logging_utils import setup_logging
from utils.telemetry import TelemetryManager, TimingContext
from utils.trace_recorder import TraceRecorder

console = Console()


class NeuroSymbolicAgent:
    """
    Production-grade NeuroSymbolic AGI Agent.

    Full pipeline:
    1. Embed task → retrieve memory context
    2. MetaController routes task (neural/symbolic/hybrid) using local/cloud LLM
    3. ReasoningEngine executes on selected path
    4. SelfImprovementLoop critiques and corrects
    5. ConstitutionalChecker validates output
    6. Store episode in memory → return final answer
    """

    def __init__(self, config_path: str = "configs/agent_config.yaml"):
        self.config = load_config(config_path)
        principles_raw = load_principles(self.config.constitutional.principles_file)

        setup_logging(
            self.config.logging.log_file,
            self.config.logging.level,
        )
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # ── Telemetry ───────────────────────────────────────────────────────
        self.telemetry = TelemetryManager(
            log_file="logs/telemetry.jsonl",
            enable_telemetry=self.config.agent.enable_telemetry,
        )

        # ── Local LLM Manager (if using local backend) ─────────────────────
        self.local_llm: Optional[LocalLLMManager] = None
        if self.config.agent.llm_backend == "local":
            logger.info(f"[Agent] Initializing local LLM: {self.config.agent.local_model_name}")
            self.local_llm = create_llm_manager(
                model_name=self.config.agent.local_model_name,
                use_gpu=True,
            )

        # ── Core components ───────────────────────────────────────────────────
        self.embedder = Embedder(
            model_name=self.config.memory.embedding_model,
            device=self.config.neural.device,
        )
        self.classifier = TaskClassifier(
            self.embedder,
            hidden_dim=self.config.neural.classifier_hidden_dim,
            device=self.config.neural.device,
        )
        self.memory = MemoryManager(
            working_memory_capacity=self.config.memory.working_memory_capacity,
            episodic_memory_path=self.config.memory.episodic_memory_path,
            retrieval_top_k=self.config.memory.retrieval_top_k,
            similarity_threshold=self.config.memory.similarity_threshold,
            embedding_dim=self.embedder.embedding_dim,
            vector_db_type=self.config.memory.vector_db_type,
            persist_directory=self.config.memory.persist_directory,
            enable_compression=self.config.memory.enable_compression,
        )
        self.kb = KnowledgeBase()
        
        # Determine model based on backend
        if self.config.agent.llm_backend == "local":
            model_name = self.config.agent.local_model_name
        elif self.config.agent.llm_backend == "anthropic":
            model_name = self.config.agent.anthropic_model
        else:
            model_name = self.config.agent.llm_model
        
        self.meta_controller = MetaController(
            model=model_name,
            backend=self.config.agent.llm_backend,
            confidence_threshold=self.config.agent.confidence_threshold,
            embedder=self.embedder,
            classifier=self.classifier,
            local_llm=self.local_llm,
            api_key=self.config.agent.api_key,
        )
        self.reasoning_engine = ReasoningEngine(
            model=model_name,
            knowledge_base=self.kb,
        )
        registry = PrinciplesRegistry(principles=principles_raw)
        self.constitutional_checker = ConstitutionalChecker(
            registry=registry,
            max_violations_before_halt=self.config.constitutional.max_violations_before_halt,
            strict_mode=self.config.constitutional.strict_mode,
        )
        self.hacking_detector = RewardHackingDetector()
        self.self_improvement = SelfImprovementLoop(
            model=model_name,
            max_rounds=self.config.agent.max_self_improvement_rounds,
            constitutional_checker=self.constitutional_checker,
            memory_manager=self.memory,
            hacking_detector=self.hacking_detector,
        )
        self.planner = HierarchicalPlanner(
            model=model_name,
            max_depth=self.config.agent.max_planning_depth,
        )
        self.fallback_engine = FallbackStrategyEngine()
        self.trace_recorder = TraceRecorder(self.config.logging.trace_file)

        logger.info(f"[Agent] {self.config.agent.name} v{self.config.agent.version} ready")
        logger.info(f"[Agent] LLM Backend: {self.config.agent.llm_backend}")
        console.print(f"\n[meta]🧬 {self.config.agent.name} v{self.config.agent.version} initialised[/meta]\n")
        console.print(f"[dim]Backend: {self.config.agent.llm_backend} | Model: {model_name}[/dim]\n")

    def run(self, task: str) -> str:
        """
        Main entry point. Execute the full meta-reasoning pipeline on a task.
        Returns the final answer string.
        """
        trace_id = str(uuid.uuid4())[:8]
        trace = self.trace_recorder.start_trace(trace_id, task)

        # Start telemetry session
        metrics = self.telemetry.start_session(task)

        console.print(f"\n[bold]📋 Task:[/bold] {task}\n")

        try:
            result = self._run_pipeline(task, trace_id, metrics)
            self.telemetry.end_session(success=True)
            return result
        except Exception as e:
            logger.error(f"[Agent] Pipeline error: {e}", exc_info=True)
            self.telemetry.end_session(success=False, error_message=str(e))
            self.trace_recorder.complete_trace(
                final_answer=f"Error: {e}",
                final_confidence=0.0,
                path_used="error",
                success=False,
            )
            return f"I encountered an error processing your task: {e}"

    def _run_pipeline(self, task: str, trace_id: str, metrics) -> str:
        # ── Step 1: Embed task + retrieve memory context ──────────────────────
        with TimingContext(self.telemetry, "memory_retrieval"):
            task_embedding = self.embedder.embed(task)
            memory_ctx = self.memory.get_context(task_embedding, recent_n=3)

        self.trace_recorder.add_step(
            "memory_retrieval",
            input_summary=task,
            output_summary=memory_ctx.context_summary[:200],
        )

        # ── Step 2: Meta-controller routing ───────────────────────────────────
        with TimingContext(self.telemetry, "routing"):
            routing = self.meta_controller.route(task, memory_context=memory_ctx.context_summary)

        self.telemetry.record_routing(routing.path, routing.confidence)

        self.trace_recorder.add_step(
            "routing",
            input_summary=task,
            output_summary=f"path={routing.path}, confidence={routing.confidence:.2%}",
            path=routing.path,
            confidence=routing.confidence,
            task_type=routing.task_type,
        )

        if self.config.agent.verbose:
            console.print(f"[routing]🔀 Path: {routing.path} | Confidence: {routing.confidence:.2%}[/routing]")
            console.print(f"[dim]{routing.reasoning}[/dim]\n")

        # ── Step 3: Optional hierarchical planning ────────────────────────────
        if routing.needs_planning:
            plan = self.planner.decompose(task)
            console.print(f"[planning]📐 Plan: {len(plan.subtasks)} subtasks[/planning]")
            # For now, execute the full task with plan context injected
            task_with_plan = (
                f"{task}\n\n[Execution plan: "
                + "; ".join(s.description for s in plan.subtasks) + "]"
            )
        else:
            task_with_plan = task

        # ── Step 4: Execute reasoning on selected path ────────────────────────
        with TimingContext(self.telemetry, "reasoning"):
            reasoning_result = self.reasoning_engine.execute(
                task=task_with_plan,
                path=routing.path,
                memory_context=memory_ctx.context_summary,
                extracted_facts=routing.facts_extracted,
            )

        self.trace_recorder.add_step(
            "reasoning",
            input_summary=task_with_plan[:200],
            output_summary=reasoning_result.answer[:200],
            path=reasoning_result.path_used,
            confidence=reasoning_result.confidence,
        )

        # ── Step 5: Self-improvement loop ─────────────────────────────────────
        with TimingContext(self.telemetry, "self_improvement"):
            final_answer, final_confidence, rounds_used, all_steps = self.self_improvement.improve(
                task=task,
                initial_answer=reasoning_result.answer,
                initial_confidence=reasoning_result.confidence,
                reasoning_steps=reasoning_result.reasoning_steps,
                path_used=reasoning_result.path_used,
            )

        self.telemetry.record_self_improvement(rounds_used)

        if self.config.agent.verbose:
            console.print(f"[improvement]🔄 {rounds_used} improvement round(s) completed[/improvement]\n")

        self.trace_recorder.add_step(
            "self_improvement",
            input_summary=reasoning_result.answer[:200],
            output_summary=final_answer[:200],
            rounds=rounds_used,
            confidence_delta=final_confidence - reasoning_result.confidence,
        )

        # ── Step 6: Final constitutional check ───────────────────────────────
        with TimingContext(self.telemetry, "constitutional_check"):
            final_check = self.constitutional_checker.check(
                output=final_answer,
                confidence=final_confidence,
                reasoning_steps=all_steps,
                improvement_round=rounds_used,
            )
        final_confidence = max(0.05, final_confidence + final_check.confidence_adjustment)

        if self.config.agent.verbose:
            if not final_check.passed:
                console.print(f"[warning]⚠️ Constitutional violations: {[v.description for v in final_check.violations]}[/warning]\n")

        # ── Step 7: Store episode in memory ───────────────────────────────────
        self.memory.store_episode(
            task=task,
            answer=final_answer,
            path_used=reasoning_result.path_used,
            confidence=final_confidence,
            success=final_check.passed,
            reasoning_steps=all_steps,
            embedding=task_embedding,
            tags=[routing.task_type, routing.path],
        )

        # ── Step 8: Complete trace and return ─────────────────────────────────
        self.trace_recorder.complete_trace(
            final_answer=final_answer,
            final_confidence=final_confidence,
            path_used=reasoning_result.path_used,
            self_improvement_rounds=rounds_used,
            constitutional_violations=[v.principle_id for v in final_check.violations],
            success=final_check.passed,
        )

        if self.config.agent.verbose:
            console.print(f"\n[bold]✅ Answer:[/bold] {final_answer}")
            console.print(f"[dim]Confidence: {final_confidence:.2%} | Path: {reasoning_result.path_used}[/dim]\n")

        return final_answer

    def cleanup(self):
        """Cleanup resources."""
        if self.local_llm:
            self.local_llm.cleanup()
        self.memory.cleanup()
        logger.info("[Agent] Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Production-grade NeuroSymbolic AGI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local LLM (default)
  python main.py --task "All birds can fly. Penguins are birds. Can penguins fly?"
  
  # Use Anthropic Claude
  python main.py --backend anthropic --task "Classify sentiment: The concert was breathtaking"
  
  # Use specific local model
  python main.py --backend local --model mistral-7b --task "Solve: What is 15 * 7 + 23?"
  
  # Interactive mode
  python main.py --interactive
  
  # Environment variables
  export LLM_BACKEND=local
  export LOCAL_MODEL_NAME=llama3-8b
  python main.py --task "Your task here"
        """,
    )
    parser.add_argument("--task", type=str, help="Task to solve")
    parser.add_argument("--config", type=str, default="configs/agent_config.yaml")
    parser.add_argument("--backend", type=str, choices=["local", "anthropic", "openai"], 
                       help="LLM backend (overrides config)")
    parser.add_argument("--model", type=str, help="Model name (overrides config)")
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable telemetry")
    args = parser.parse_args()

    # Override config with command-line args
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.model:
        if args.backend == "local":
            os.environ["LOCAL_MODEL_NAME"] = args.model
        else:
            os.environ["ANTHROPIC_MODEL"] = args.model
    if args.no_telemetry:
        os.environ["ENABLE_TELEMETRY"] = "false"

    agent = NeuroSymbolicAgent(config_path=args.config)

    try:
        if args.interactive:
            console.print("\n[meta]🧬 NeuroSymbolic AGI Agent — Interactive Mode[/meta]")
            console.print(f"[dim]Backend: {agent.config.agent.llm_backend}[/dim]")
            console.print("[dim]Type 'exit' to quit[/dim]\n")
            while True:
                try:
                    task = input("Task > ").strip()
                    if task.lower() in ("exit", "quit", "q"):
                        break
                    if task:
                        agent.run(task)
                        print()
                except (KeyboardInterrupt, EOFError):
                    break
        elif args.task:
            agent.run(args.task)
        else:
            # Demo tasks
            console.print("[dim]Running demo tasks...\n[/dim]")
            demo_tasks = [
                "All mammals breathe air. Whales are mammals. Do whales breathe air?",
                "Analyse the sentiment of: The product quality was surprisingly disappointing.",
                "Solve: What is 15 * 7 + 23?",
            ]
            for task in demo_tasks:
                agent.run(task)
                print("\n" + "─" * 60 + "\n")
    finally:
        agent.cleanup()


if __name__ == "__main__":
    main()
