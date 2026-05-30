"""
llm/local_llm_manager.py — Production-grade local LLM manager.
Supports multiple backends: llama.cpp, Ollama, vLLM with automatic model downloading.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve

import torch
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


class LLMBackend(Enum):
    """Supported LLM backends."""
    LLAMA_CPP = "llama_cpp"
    OLLAMA = "ollama"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    HEURISTIC = "heuristic"


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    backend: LLMBackend
    model_id: str  # HuggingFace model ID or Ollama model name
    quantization: str = "q4_k_m"  # For llama.cpp
    context_length: int = 8192
    gpu_layers: int = -1  # -1 = all layers on GPU
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    local_path: Optional[str] = None


class LocalLLMManager:
    """
    Production-grade local LLM manager with automatic model downloading,
    caching, and multi-backend support.
    """

    # Pre-configured production models
    PRODUCTION_MODELS = {
        "agentic-rules": ModelConfig(
            name="agentic-rules",
            backend=LLMBackend.HEURISTIC,
            model_id="offline-structured-agentic-fallback",
            context_length=8192,
        ),
        "llama3-8b": ModelConfig(
            name="llama3-8b",
            backend=LLMBackend.LLAMA_CPP,
            model_id="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
            quantization="q4_k_m",
            context_length=8192,
        ),
        "llama3-70b": ModelConfig(
            name="llama3-70b",
            backend=LLMBackend.LLAMA_CPP,
            model_id="Meta-Llama-3-70B-Instruct.Q4_K_M.gguf",
            quantization="q4_k_m",
            context_length=8192,
        ),
        "mistral-7b": ModelConfig(
            name="mistral-7b",
            backend=LLMBackend.LLAMA_CPP,
            model_id="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            quantization="q4_k_m",
            context_length=8192,
        ),
        "mixtral-8x7b": ModelConfig(
            name="mixtral-8x7b",
            backend=LLMBackend.LLAMA_CPP,
            model_id="mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            quantization="q4_k_m",
            context_length=32768,
        ),
        "ollama-llama3": ModelConfig(
            name="ollama-llama3",
            backend=LLMBackend.OLLAMA,
            model_id="llama3",
            context_length=8192,
        ),
        "ollama-llama3.1": ModelConfig(
            name="ollama-llama3.1",
            backend=LLMBackend.OLLAMA,
            model_id="llama3.1:8b",
            context_length=131072,
        ),
        "ollama-qwen2.5": ModelConfig(
            name="ollama-qwen2.5",
            backend=LLMBackend.OLLAMA,
            model_id="qwen2.5:7b",
            context_length=32768,
        ),
        "ollama-deepseek-r1": ModelConfig(
            name="ollama-deepseek-r1",
            backend=LLMBackend.OLLAMA,
            model_id="deepseek-r1:7b",
            context_length=32768,
        ),
        "ollama-mistral": ModelConfig(
            name="ollama-mistral",
            backend=LLMBackend.OLLAMA,
            model_id="mistral:7b",
            context_length=32768,
        ),
        "ollama-phi4-mini": ModelConfig(
            name="ollama-phi4-mini",
            backend=LLMBackend.OLLAMA,
            model_id="phi4-mini",
            context_length=131072,
        ),
        "transformers-tinyllama": ModelConfig(
            name="transformers-tinyllama",
            backend=LLMBackend.TRANSFORMERS,
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            context_length=4096,
        ),
        "transformers-qwen2.5-0.5b": ModelConfig(
            name="transformers-qwen2.5-0.5b",
            backend=LLMBackend.TRANSFORMERS,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            context_length=32768,
        ),
    }

    MODEL_DOWNLOAD_URLS = {
        "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "Meta-Llama-3-70B-Instruct.Q4_K_M.gguf": "https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.Q4_K_M.gguf",
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    }

    def __init__(
        self,
        model_config: ModelConfig,
        models_dir: str = "models",
        cache_dir: str = "cache",
        use_gpu: bool = True,
    ):
        self.config = model_config
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()
        
        logger.info(f"[LocalLLM] Initializing with backend: {model_config.backend.value}")
        logger.info(f"[LocalLLM] GPU available: {self.use_gpu}")
        
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the selected backend."""
        if self.config.backend == LLMBackend.LLAMA_CPP:
            self._init_llama_cpp()
        elif self.config.backend == LLMBackend.OLLAMA:
            self._init_ollama()
        elif self.config.backend == LLMBackend.TRANSFORMERS:
            self._init_transformers()
        elif self.config.backend == LLMBackend.HEURISTIC:
            logger.warning("[LocalLLM] Using offline heuristic fallback; install Ollama or llama.cpp for real local LLM inference.")
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _init_llama_cpp(self):
        """Initialize llama.cpp backend."""
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("[LocalLLM] llama-cpp-python not installed. Run: pip install llama-cpp-python")
            raise

        model_path = self._get_or_download_model()
        
        logger.info(f"[LocalLLM] Loading model from {model_path}")
        
        n_gpu_layers = self.config.gpu_layers if self.use_gpu else 0
        if self.config.gpu_layers == -1 and self.use_gpu:
            n_gpu_layers = -1  # All layers on GPU
        
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=self.config.context_length,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        
        logger.info("[LocalLLM] llama.cpp backend ready")

    def _init_ollama(self):
        """Initialize Ollama backend."""
        # Check if Ollama is running
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                logger.warning("[LocalLLM] Ollama not running. Start with: ollama serve")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.error("[LocalLLM] Ollama not found. Install from: https://ollama.ai")
            raise

        # Pull model if not available
        try:
            subprocess.run(
                ["ollama", "pull", self.config.model_id],
                capture_output=True,
                check=True,
            )
            logger.info(f"[LocalLLM] Ollama model {self.config.model_id} ready")
        except subprocess.CalledProcessError:
            logger.error(f"[LocalLLM] Failed to pull Ollama model: {self.config.model_id}")
            raise

    def _init_transformers(self):
        """Initialize HuggingFace Transformers backend."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.error("[LocalLLM] transformers not installed. Run: pip install transformers")
            raise

        logger.info(f"[LocalLLM] Loading model: {self.config.model_id}")
        
        device = "cuda" if self.use_gpu else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map="auto" if self.use_gpu else None,
            torch_dtype=torch.float16 if self.use_gpu else torch.float32,
            trust_remote_code=True,
        )
        
        if not self.use_gpu:
            self.model = self.model.to(device)
        
        logger.info("[LocalLLM] Transformers backend ready")

    def _get_or_download_model(self) -> Path:
        """Get model path, downloading if necessary."""
        model_filename = self.config.model_id
        model_path = self.models_dir / model_filename
        
        if model_path.exists():
            logger.info(f"[LocalLLM] Model found at {model_path}")
            return model_path
        
        # Download model
        if model_filename not in self.MODEL_DOWNLOAD_URLS:
            raise ValueError(f"No download URL for model: {model_filename}")
        
        url = self.MODEL_DOWNLOAD_URLS[model_filename]
        logger.info(f"[LocalLLM] Downloading model from {url}")
        
        # Download with progress bar
        def download_with_progress():
            with tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=model_filename,
            ) as progress:
                def report_progress(block_num, block_size, total_size):
                    if progress.total is None and total_size > 0:
                        progress.total = total_size
                    progress.update(block_num * block_size - progress.n)
                
                urlretrieve(url, model_path, reporthook=report_progress)
        
        download_with_progress()
        logger.info(f"[LocalLLM] Model downloaded to {model_path}")
        
        return model_path

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generate text using the local LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            json_mode: Force JSON output
            
        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        
        with self._lock:
            if self.config.backend == LLMBackend.LLAMA_CPP:
                return self._generate_llama_cpp(
                    prompt, system_prompt, max_tokens, temperature, top_p, stop, json_mode
                )
            elif self.config.backend == LLMBackend.OLLAMA:
                return self._generate_ollama(
                    prompt, system_prompt, max_tokens, temperature, top_p, stop, json_mode
                )
            elif self.config.backend == LLMBackend.TRANSFORMERS:
                return self._generate_transformers(
                    prompt, system_prompt, max_tokens, temperature, top_p, stop, json_mode
                )
            elif self.config.backend == LLMBackend.HEURISTIC:
                return self._generate_heuristic(prompt, system_prompt, json_mode)
            else:
                raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _generate_heuristic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_mode: bool,
    ) -> str:
        """Structured offline fallback used when no local LLM runtime is available."""
        system = (system_prompt or "").lower()
        prompt_lower = prompt.lower()

        if "meta-controller" in system:
            symbolic_terms = [
                "solve", "equation", "all ", "if ", "then", "logic", "proof",
                "constraint", "derivative", "integral", "system of equations",
            ]
            neural_terms = ["sentiment", "summarize", "summarise", "classify", "translate", "write"]
            needs_planning = any(term in prompt_lower for term in ["plan", "multi-step", "complex", "optimize", "puzzle"])
            if any(term in prompt_lower for term in symbolic_terms):
                path = "symbolic"
                task_type = "symbolic_reasoning"
                confidence = 0.72
            elif any(term in prompt_lower for term in neural_terms):
                path = "neural"
                task_type = "neural_language_task"
                confidence = 0.62
            else:
                path = "hybrid"
                task_type = "general_reasoning"
                confidence = 0.55
            return json.dumps({
                "path": path,
                "confidence": confidence,
                "task_type": task_type,
                "reasoning": "Offline fallback routed the task using deterministic agentic heuristics.",
                "needs_planning": needs_planning,
                "facts_extracted": [],
                "subtask_hints": [],
            })

        if "neural inference engine" in system:
            label = "general"
            confidence = 0.5
            output = "I can process this with limited offline language heuristics, but a local open-weight model will produce a stronger answer."
            if "sentiment" in prompt_lower:
                positive = sum(word in prompt_lower for word in ["good", "great", "excellent", "fast", "intuitive", "love"])
                negative = sum(word in prompt_lower for word in ["bad", "poor", "slow", "disappointing", "hate", "high"])
                if positive > negative:
                    output, label, confidence = "Positive sentiment.", "sentiment_analysis", 0.66
                elif negative > positive:
                    output, label, confidence = "Negative sentiment.", "sentiment_analysis", 0.66
                else:
                    output, label, confidence = "Mixed or neutral sentiment.", "sentiment_analysis", 0.55
            return json.dumps({
                "output": output,
                "confidence": confidence,
                "task_type": label,
                "reasoning": "Generated by the offline fallback because no local LLM runtime was available.",
            })

        if "hybrid reasoning engine" in system:
            symbolic = self._extract_after(prompt, "Symbolic reasoning result:")
            neural = self._extract_after(prompt, "Neural reasoning result:")
            answer = symbolic if symbolic and "could not" not in symbolic.lower() else neural
            return json.dumps({
                "answer": answer.strip() or "No reliable answer was produced by the fallback engines.",
                "confidence": 0.55,
                "reasoning": "Selected the strongest available symbolic/neural fallback result.",
            })

        if "self-critic" in system:
            return json.dumps({
                "has_issues": False,
                "issues": [],
                "failure_modes": ["none"],
                "suggested_correction": "",
                "corrected_answer": self._extract_after(prompt, "Agent's answer:") or prompt,
                "corrected_confidence": 0.55,
                "improvement_achieved": False,
                "reasoning_trace": ["Offline fallback critique found no deterministic issue."],
            })

        return "{}" if json_mode else "Offline fallback did not have a specialized handler for this prompt."

    @staticmethod
    def _extract_after(text: str, marker: str) -> str:
        if marker not in text:
            return ""
        fragment = text.split(marker, 1)[1]
        for next_marker in ("\n\n", "Neural confidence:", "Proof steps:"):
            if next_marker in fragment:
                fragment = fragment.split(next_marker, 1)[0]
        return fragment.strip()

    def _generate_llama_cpp(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[list[str]],
        json_mode: bool,
    ) -> str:
        """Generate using llama.cpp."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        if json_mode:
            full_prompt += "\n\nRespond ONLY with valid JSON, no other text."
        
        response = self.model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False,
        )
        
        return response["choices"][0]["text"].strip()

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[list[str]],
        json_mode: bool,
    ) -> str:
        """Generate using Ollama."""
        import requests
        
        payload = {
            "model": self.config.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if stop:
            payload["options"]["stop"] = stop
        
        if json_mode:
            payload["format"] = "json"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        
        return response.json()["response"].strip()

    def _generate_transformers(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[list[str]],
        json_mode: bool,
    ) -> str:
        """Generate using Transformers."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.context_length - max_tokens,
        )
        
        if self.use_gpu:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated.strip()

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "name": self.config.name,
            "backend": self.config.backend.value,
            "model_id": self.config.model_id,
            "context_length": self.config.context_length,
            "use_gpu": self.use_gpu,
            "quantization": self.config.quantization,
        }

    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.use_gpu:
            torch.cuda.empty_cache()
        
        logger.info("[LocalLLM] Cleanup complete")


def create_llm_manager(
    model_name: str = "llama3-8b",
    models_dir: str = "models",
    use_gpu: bool = True,
) -> LocalLLMManager:
    """
    Factory function to create a LocalLLMManager with a pre-configured model.
    
    Args:
        model_name: Name of the production model to use
        models_dir: Directory to store downloaded models
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Configured LocalLLMManager instance
    """
    if model_name in ("auto", "default"):
        model_name = _select_auto_model()

    if model_name not in LocalLLMManager.PRODUCTION_MODELS:
        available = ", ".join(LocalLLMManager.PRODUCTION_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    config = LocalLLMManager.PRODUCTION_MODELS[model_name]
    return LocalLLMManager(config, models_dir=models_dir, use_gpu=use_gpu)


def _select_auto_model() -> str:
    """Prefer a local Ollama runtime when it is already installed/running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            return "ollama-qwen2.5"
    except Exception:
        pass
    return "llama3-8b"
