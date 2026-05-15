"""
llm package — Local LLM integration for production AGI agent.
"""
from .local_llm_manager import (
    LLMBackend,
    LocalLLMManager,
    ModelConfig,
    create_llm_manager,
)

__all__ = [
    "LLMBackend",
    "LocalLLMManager",
    "ModelConfig",
    "create_llm_manager",
]
