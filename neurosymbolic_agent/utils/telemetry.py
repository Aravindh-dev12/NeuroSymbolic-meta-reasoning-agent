"""
utils/telemetry.py — Production-grade telemetry and monitoring system.
Tracks agent performance, resource usage, and reasoning metrics.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from threading import Lock

import psutil
from loguru import logger


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    event_type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a reasoning session."""
    task: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_ms: Optional[float] = None
    memory_retrieval_time_ms: Optional[float] = None
    routing_time_ms: Optional[float] = None
    reasoning_time_ms: Optional[float] = None
    self_improvement_rounds: int = 0
    self_improvement_time_ms: Optional[float] = None
    constitutional_check_time_ms: Optional[float] = None
    path_used: Optional[str] = None
    final_confidence: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TelemetryManager:
    """
    Production-grade telemetry manager for monitoring agent performance.
    Tracks timing, resource usage, and custom events.
    """

    def __init__(
        self,
        log_file: str = "logs/telemetry.jsonl",
        enable_telemetry: bool = True,
        track_resources: bool = True,
    ):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.enable_telemetry = enable_telemetry
        self.track_resources = track_resources
        self._lock = Lock()
        self._events: list[TelemetryEvent] = []
        self._current_metrics: Optional[PerformanceMetrics] = None
        
        if self.enable_telemetry:
            logger.info(f"[Telemetry] Initialized with log file: {log_file}")

    def start_session(self, task: str) -> PerformanceMetrics:
        """Start a new telemetry session for a task."""
        metrics = PerformanceMetrics(
            task=task,
            start_time=time.time(),
        )
        self._current_metrics = metrics
        return metrics

    def end_session(
        self,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> PerformanceMetrics:
        """End the current telemetry session."""
        if self._current_metrics is None:
            logger.warning("[Telemetry] No active session to end")
            return PerformanceMetrics(task="unknown", start_time=0)

        self._current_metrics.end_time = time.time()
        self._current_metrics.total_duration_ms = (
            self._current_metrics.end_time - self._current_metrics.start_time
        ) * 1000
        self._current_metrics.success = success
        self._current_metrics.error_message = error_message

        # Log the session
        if self.enable_telemetry:
            self._log_event("session_complete", self._current_metrics.total_duration_ms, {
                "task": self._current_metrics.task,
                "path_used": self._current_metrics.path_used,
                "confidence": self._current_metrics.final_confidence,
                "success": success,
                "self_improvement_rounds": self._current_metrics.self_improvement_rounds,
                "error": error_message,
            })

        # Persist to file
        self._persist_session(self._current_metrics)

        metrics = self._current_metrics
        self._current_metrics = None
        return metrics

    def record_timing(self, stage: str, duration_ms: float):
        """Record timing for a specific stage."""
        if self._current_metrics is None:
            return

        if stage == "memory_retrieval":
            self._current_metrics.memory_retrieval_time_ms = duration_ms
        elif stage == "routing":
            self._current_metrics.routing_time_ms = duration_ms
        elif stage == "reasoning":
            self._current_metrics.reasoning_time_ms = duration_ms
        elif stage == "self_improvement":
            self._current_metrics.self_improvement_time_ms = duration_ms
        elif stage == "constitutional_check":
            self._current_metrics.constitutional_check_time_ms = duration_ms

        if self.enable_telemetry:
            self._log_event(f"timing_{stage}", duration_ms)

    def record_routing(self, path: str, confidence: float):
        """Record routing decision."""
        if self._current_metrics is None:
            return
        self._current_metrics.path_used = path
        self._current_metrics.final_confidence = confidence

    def record_self_improvement(self, rounds: int):
        """Record self-improvement rounds."""
        if self._current_metrics is None:
            return
        self._current_metrics.self_improvement_rounds = rounds

    def _log_event(
        self,
        event_type: str,
        duration_ms: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Log a telemetry event."""
        event = TelemetryEvent(
            event_type=event_type,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._events.append(event)

    def _persist_session(self, metrics: PerformanceMetrics):
        """Persist session metrics to log file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps({
                    "task": metrics.task,
                    "start_time": datetime.fromtimestamp(metrics.start_time).isoformat(),
                    "end_time": datetime.fromtimestamp(metrics.end_time or 0).isoformat(),
                    "total_duration_ms": metrics.total_duration_ms,
                    "memory_retrieval_time_ms": metrics.memory_retrieval_time_ms,
                    "routing_time_ms": metrics.routing_time_ms,
                    "reasoning_time_ms": metrics.reasoning_time_ms,
                    "self_improvement_rounds": metrics.self_improvement_rounds,
                    "self_improvement_time_ms": metrics.self_improvement_time_ms,
                    "constitutional_check_time_ms": metrics.constitutional_check_time_ms,
                    "path_used": metrics.path_used,
                    "final_confidence": metrics.final_confidence,
                    "success": metrics.success,
                    "error_message": metrics.error_message,
                    "metadata": metrics.metadata,
                }) + "\n")
        except Exception as e:
            logger.error(f"[Telemetry] Failed to persist session: {e}")

    def get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage."""
        if not self.track_resources:
            return {}

        process = psutil.Process()
        try:
            return {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "num_fds": process.num_fds() if hasattr(process, 'num_fds') else 0,
            }
        except Exception as e:
            logger.warning(f"[Telemetry] Failed to get resource usage: {e}")
            return {}

    def get_summary(self) -> dict[str, Any]:
        """Get telemetry summary."""
        with self._lock:
            event_counts = {}
            for event in self._events:
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

            return {
                "total_events": len(self._events),
                "event_counts": event_counts,
                "current_session": (
                    {
                        "task": self._current_metrics.task,
                        "duration_ms": (time.time() - self._current_metrics.start_time) * 1000,
                    }
                    if self._current_metrics
                    else None
                ),
                "resource_usage": self.get_resource_usage(),
            }

    def clear_events(self):
        """Clear all accumulated events."""
        with self._lock:
            self._events.clear()
            logger.info("[Telemetry] Events cleared")


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, telemetry: TelemetryManager, stage: str):
        self.telemetry = telemetry
        self.stage = stage
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            self.telemetry.record_timing(self.stage, duration_ms)
        return False
