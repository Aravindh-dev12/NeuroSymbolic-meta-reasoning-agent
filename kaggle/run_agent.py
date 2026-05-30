"""Kaggle entrypoint for the NeuroSymbolic Meta-Reasoning Agent."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "neurosymbolic_agent"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

os.environ.setdefault("LLM_BACKEND", "local")
os.environ.setdefault("LOCAL_MODEL_NAME", "agentic-rules")
os.environ.setdefault("VECTOR_DB_TYPE", "sqlite")
os.environ.setdefault("ENABLE_TELEMETRY", "false")

from main import NeuroSymbolicAgent


def main() -> None:
    task = os.getenv(
        "AGENT_TASK",
        "All mammals breathe air. Whales are mammals. Do whales breathe air?",
    )
    config_path = PACKAGE_DIR / "configs" / "agent_config.yaml"
    agent = NeuroSymbolicAgent(config_path=str(config_path))
    try:
        print(agent.run(task))
    finally:
        agent.cleanup()


if __name__ == "__main__":
    main()
