"""
utils/config.py — Production-grade configuration loader with environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class AgentConfig(BaseModel):
    name: str = "NeuroSymbolicMetaAgent"
    version: str = "2.1.0"
    llm_backend: str = "local"  # "local", "anthropic", "openai"
    local_model_name: str = "auto"
    llm_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-4-20250514"
    confidence_threshold: float = 0.75
    max_self_improvement_rounds: int = 3
    max_planning_depth: int = 5
    verbose: bool = True
    enable_telemetry: bool = True
    allow_heuristic_fallback: bool = True
    api_key: Optional[str] = None  # For cloud LLMs
    
    @model_validator(mode="after")
    def validate_api_key(self):
        if self.llm_backend in ["anthropic", "openai"] and not self.api_key:
            raise ValueError(f"API key required for {self.llm_backend} backend")
        return self


class RoutingConfig(BaseModel):
    symbolic_task_types: list[str] = Field(default_factory=list)
    neural_task_types: list[str] = Field(default_factory=list)
    hybrid_task_types: list[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    working_memory_capacity: int = 20
    episodic_memory_path: str = "logs/episodic_store.json"
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.6
    vector_db_type: str = "chroma"  # "chroma", "faiss", "sqlite"
    persist_directory: str = "data/vector_db"
    enable_compression: bool = True


class SymbolicConfig(BaseModel):
    solver: str = "z3"
    timeout_seconds: int = 10
    max_iterations: int = 1000


class NeuralConfig(BaseModel):
    embedding_dim: int = 384
    classifier_hidden_dim: int = 256
    device: str = "cpu"
    batch_size: int = 32


class ConstitutionalConfig(BaseModel):
    principles_file: str = "configs/constitutional_principles.yaml"
    max_violations_before_halt: int = 2
    strict_mode: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"
    trace_file: str = "logs/reasoning_traces.jsonl"
    log_file: str = "logs/agent.log"
    enable_structured_logging: bool = True
    log_to_console: bool = True
    max_log_size_mb: int = 100
    backup_count: int = 5


class Config(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    symbolic: SymbolicConfig = Field(default_factory=SymbolicConfig)
    neural: NeuralConfig = Field(default_factory=NeuralConfig)
    constitutional: ConstitutionalConfig = Field(default_factory=ConstitutionalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: str = "configs/agent_config.yaml") -> Config:
    """Load and validate configuration from YAML file with environment variable overrides."""
    path = Path(config_path)
    raw_config = {}
    
    if path.exists():
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f) or {}
    
    # Apply environment variable overrides
    env_overrides = {
        "agent": {
            "llm_backend": os.getenv("LLM_BACKEND"),
            "local_model_name": os.getenv("LOCAL_MODEL_NAME"),
            "llm_model": os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL"),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL"),
            "api_key": os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"),
            "confidence_threshold": _parse_float(os.getenv("CONFIDENCE_THRESHOLD")),
            "max_self_improvement_rounds": _parse_int(os.getenv("MAX_SELF_IMPROVEMENT_ROUNDS")),
            "max_planning_depth": _parse_int(os.getenv("MAX_PLANNING_DEPTH")),
            "verbose": _parse_bool(os.getenv("VERBOSE")),
            "enable_telemetry": _parse_bool(os.getenv("ENABLE_TELEMETRY")),
            "allow_heuristic_fallback": _parse_bool(os.getenv("ALLOW_HEURISTIC_FALLBACK")),
        },
        "memory": {
            "vector_db_type": os.getenv("VECTOR_DB_TYPE"),
            "persist_directory": os.getenv("VECTOR_DB_DIR"),
            "enable_compression": _parse_bool(os.getenv("ENABLE_MEMORY_COMPRESSION")),
        },
        "neural": {
            "device": os.getenv("NEURAL_DEVICE"),
            "classifier_hidden_dim": _parse_int(os.getenv("CLASSIFIER_HIDDEN_DIM")),
            "batch_size": _parse_int(os.getenv("BATCH_SIZE")),
        },
        "constitutional": {
            "max_violations_before_halt": _parse_int(os.getenv("MAX_CONSTITUTIONAL_VIOLATIONS")),
            "strict_mode": _parse_bool(os.getenv("CONSTITUTIONAL_STRICT_MODE")),
        },
        "logging": {
            "level": os.getenv("LOG_LEVEL"),
            "log_to_console": _parse_bool(os.getenv("LOG_TO_CONSOLE")),
        },
    }
    
    # Merge environment overrides
    for section, overrides in env_overrides.items():
        if section not in raw_config:
            raw_config[section] = {}
        for key, value in overrides.items():
            if value is not None:
                raw_config[section][key] = value
    
    return Config(**raw_config)


def _parse_float(value: Optional[str]) -> Optional[float]:
    """Parse environment variable as float."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    """Parse environment variable as boolean."""
    if value is None:
        return None
    return value.lower() in ("true", "1", "yes", "on")


def _parse_int(value: Optional[str]) -> Optional[int]:
    """Parse environment variable as integer."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_principles(principles_path: str = "configs/constitutional_principles.yaml") -> list[dict]:
    """Load constitutional principles from YAML."""
    path = Path(principles_path)
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("principles", [])
