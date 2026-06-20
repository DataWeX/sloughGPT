"""
Llama.cpp Inference Engine
High-performance local inference using llama.cpp with GGUF models.

GPU Support:
- Auto-detects GPU capability (Metal/CUDA)
- Falls back to CPU when GPU is too slow or unavailable
- Supports high-end GPU switching via MAN_FORCE_GPU env var
"""

import os
import sys
import time
import threading
import json
import subprocess
import requests
import platform
from pathlib import Path
from typing import Optional, List, AsyncIterator, Dict, Any, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("man.inference.llama_engine")

LLAMA_CPP_AVAILABLE = False
LLAMA_CPP_PYTHON_AVAILABLE = False
LLAMA_CPP_ERROR: Optional[str] = None
Llama = None

try:
    from llama_cpp import Llama as _Llama
    from llama_cpp.llama import LlamaCompletionTokenChunk

    LLAMA_CPP_AVAILABLE = True
    LLAMA_CPP_PYTHON_AVAILABLE = True
    Llama = _Llama
except ImportError as e:
    LLAMA_CPP_ERROR = str(e)
    logger.warning(f"llama-cpp-python not installed: {e}. Falling back to llama-cli subprocess.")

LLAMA_CLI_PATH = "/usr/local/bin/llama-cli"
LLAMA_CLI_AVAILABLE = Path(LLAMA_CLI_PATH).exists() if not LLAMA_CPP_PYTHON_AVAILABLE else False

_MODEL_CACHE: Dict[str, "LlamaInferenceEngine"] = {}
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_SIZE = 10  # Default max cached models

_INFERENCE_STATS = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_time": 0.0,
    "cache_hits": 0,
    "latencies_ms": [],  # Last 100 latencies for histogram
    "request_ids": [],  # Track recent request IDs
    "total_errors": 0,
    "last_error": None,
    "streaming_requests": 0,  # Streaming request count
}

_REQUEST_COUNTER = 0
_COUNTER_LOCK = threading.Lock()


def _generate_request_id() -> str:
    """Generate a unique request ID."""
    global _REQUEST_COUNTER
    with _COUNTER_LOCK:
        _REQUEST_COUNTER += 1
        return f"req_{_REQUEST_COUNTER}_{int(time.time() * 1000)}"


@dataclass
class GPUInfo:
    """Information about a detected GPU."""

    name: str
    backend: str
    vram_mb: float
    has_tensor_ops: bool
    recommended: bool
    reason: str = ""


def detect_gpu() -> Optional[GPUInfo]:
    """
    Detect available GPU and determine if it should be used.

    Returns GPUInfo if a capable GPU is detected, None otherwise.

    GPU is recommended when:
    - Has tensor ops (Metal GPU family >= 3, or modern CUDA)
    - Has sufficient VRAM (>2GB for small models)
    """
    system = platform.system()

    if system == "Darwin":
        return _detect_metal_gpu()
    elif system == "Linux":
        return _detect_cuda_gpu()

    return None


def _detect_metal_gpu() -> Optional[GPUInfo]:
    """Detect Apple Metal GPU and check capabilities."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        total_ram_gb = int(result.stdout.strip()) / (1024**3)
    except Exception:
        total_ram_gb = 0

    machine = platform.machine()
    has_apple_silicon = machine in ("arm64", "aarch64")

    if not has_apple_silicon:
        gpu_name = "Unknown"
        has_amd_gpu = False
        vram_mb = 1536

        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout
            if "Radeon" in output:
                has_amd_gpu = True
                if "Pro 5" in output:
                    gpu_name = "AMD Radeon Pro 555X/560X"
                else:
                    gpu_name = "AMD Radeon GPU"
                if "4 GB" in output:
                    vram_mb = 4096
                elif "x8" in output:
                    vram_mb = 4096
        except Exception:
            pass

        if has_amd_gpu:
            return GPUInfo(
                name=gpu_name,
                backend="metal",
                vram_mb=vram_mb,
                has_tensor_ops=False,
                recommended=False,
                reason="AMD GPU on Intel Mac lacks Metal tensor ops (simdgroup_mm) - CPU is faster",
            )

        return GPUInfo(
            name="Intel Mac (Integrated GPU)",
            backend="metal",
            vram_mb=1536,
            has_tensor_ops=False,
            recommended=False,
            reason="Intel integrated GPU too slow for LLM inference",
        )

    gpu_names = {
        "M1": "Apple M1",
        "M2": "Apple M2",
        "M3": "Apple M3",
        "M4": "Apple M4",
        "M5": "Apple M5",
    }

    try:
        cpubrand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        cpubrand = "Unknown"

    for chip, name in gpu_names.items():
        if chip in cpubrand:
            has_tensor = "M3" in cpubrand or "M4" in cpubrand or "M5" in cpubrand

            recommended = has_tensor and total_ram_gb >= 8

            reason = ""
            if not has_tensor:
                reason = "No tensor ops (M1/M2 chips slower than CPU for LLM)"
            elif total_ram_gb < 8:
                reason = f"Limited RAM ({total_ram_gb:.0f}GB)"
            else:
                reason = "Full tensor support"

            return GPUInfo(
                name=name,
                backend="metal",
                vram_mb=total_ram_gb * 1024,
                has_tensor_ops=has_tensor,
                recommended=recommended,
                reason=reason,
            )

    return GPUInfo(
        name="Unknown Apple Silicon",
        backend="metal",
        vram_mb=total_ram_gb * 1024,
        has_tensor_ops=False,
        recommended=False,
        reason="Unknown chip, assuming no tensor ops",
    )


def _detect_cuda_gpu() -> Optional[GPUInfo]:
    """Detect NVIDIA CUDA GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                name, mem = lines[0].split(",")
                vram_mb = float(mem.strip().replace("MiB", "").strip())
                compute_capable = vram_mb >= 4096
                return GPUInfo(
                    name=name.strip(),
                    backend="cuda",
                    vram_mb=vram_mb,
                    has_tensor_ops=compute_capable,
                    recommended=compute_capable,
                    reason="CUDA with tensor cores"
                    if compute_capable
                    else f"Limited VRAM ({vram_mb:.0f}MB)",
                )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def auto_select_backend(model_size_gb: float = 1.5) -> int:
    """
    Auto-select GPU layers based on detected GPU capability.

    Returns 0 (CPU) by default for safety. GPU only if:
    - has_tensor_ops is True
    - GPU has sufficient VRAM (>2GB for model)

    Args:
        model_size_gb: Size of the model in GB

    Returns:
        Number of GPU layers to use (0 = CPU only)
    """
    force_gpu = os.environ.get("MAN_FORCE_GPU", "").lower()
    if force_gpu in ("1", "true", "yes"):
        logger.info("Force GPU enabled via MAN_FORCE_GPU")
        return 99

    force_cpu = os.environ.get("MAN_FORCE_CPU", "").lower()
    if force_cpu in ("1", "true", "yes"):
        logger.info("Force CPU enabled via MAN_FORCE_CPU")
        return 0

    gpu = detect_gpu()

    if gpu is None:
        logger.info("No GPU detected, using CPU")
        return 0

    if not gpu.has_tensor_ops:
        logger.info(f"GPU {gpu.name} lacks tensor ops - using CPU instead ({gpu.reason})")
        return 0

    model_size_gb = max(0.5, model_size_gb)
    if gpu.vram_mb < model_size_gb * 1024 * 1.5:
        logger.info(
            f"GPU VRAM ({gpu.vram_mb:.0f}MB) too small for model ({model_size_gb:.1f}GB) - using CPU"
        )
        return 0

    logger.info(f"GPU {gpu.name} with tensor ops - using GPU acceleration")
    return 99


@dataclass
class LlamaInferenceConfig:
    """Configuration for llama.cpp inference.

    GPU behavior:
    - n_gpu_layers=-1: Auto-detect (recommended)
    - n_gpu_layers=0: CPU only
    - n_gpu_layers=99: Full GPU offload
    - n_gpu_layers=N: Offload N layers to GPU
    """

    model_path: str
    n_ctx: int = 4096
    n_threads: int = 6
    n_gpu_layers: int = -1
    n_batch: int = 512
    repeat_penalty: float = 1.1
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0
    verbose: bool = False

    def __post_init__(self):
        if self.n_gpu_layers == -1:
            model_size_gb = 1.5
            if Path(self.model_path).exists():
                try:
                    model_size_gb = Path(self.model_path).stat().st_size / (1024**3)
                except Exception:
                    pass
            self.n_gpu_layers = auto_select_backend(model_size_gb)


class LlamaInferenceEngine:
    """
    Inference engine using llama.cpp for high-performance CPU/GPU inference.

    Features:
    - GGUF model support
    - Streaming generation
    - KV cache (via llama.cpp)
    - Configurable thread count
    - Memory efficient
    - Fallback to llama-cli subprocess if llama-cpp-python unavailable
    """

    def __init__(self, config: LlamaInferenceConfig):
        self.config = config
        self._llama: Optional[Llama] = None
        self._cli_engine: Optional[LlamaCLIInferenceEngine] = None
        self._lock = threading.Lock()
        self._model_loaded = False

        gpu = detect_gpu()
        if gpu:
            logger.info(
                f"GPU Info: {gpu.name}, VRAM: {gpu.vram_mb:.0f}MB, "
                f"Tensor ops: {gpu.has_tensor_ops}, Recommended: {gpu.recommended}"
            )

        if not LLAMA_CPP_PYTHON_AVAILABLE:
            if LLAMA_CLI_AVAILABLE:
                logger.info("llama-cpp-python not available, using llama-cli fallback")
                self._cli_engine = LlamaCLIInferenceEngine(
                    model_path=config.model_path,
                    n_ctx=config.n_ctx,
                    n_threads=config.n_threads,
                    n_gpu_layers=config.n_gpu_layers,
                )
            else:
                raise ImportError(
                    "Neither llama-cpp-python nor llama-cli available. "
                    "Install llama-cpp-python or brew install llama.cpp"
                )

    def load_model(self) -> bool:
        """Load the model into memory."""
        if self._model_loaded:
            return True

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False

        try:
            logger.info(f"Loading model: {model_path}")
            self._llama = Llama(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch,
                rope_freq_base=self.config.rope_freq_base,
                rope_freq_scale=self.config.rope_freq_scale,
                verbose=self.config.verbose,
            )
            self._model_loaded = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text synchronously."""
        if self._cli_engine:
            return self._cli_engine.generate(
                prompt, max_tokens, temperature, top_p, repeat_penalty, stop
            )

        if not self._model_loaded:
            if not self.load_model():
                return ""

        with self._lock:
            try:
                start = time.perf_counter()
                result = self._llama(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop or [],
                    echo=False,
                )
                elapsed = time.perf_counter() - start
                text = result["choices"][0]["text"]
                _INFERENCE_STATS["total_requests"] += 1
                _INFERENCE_STATS["total_tokens"] += len(text.split())
                _INFERENCE_STATS["total_time"] += elapsed
                _INFERENCE_STATS["latencies_ms"].append(elapsed * 1000)
                if len(_INFERENCE_STATS["latencies_ms"]) > 100:
                    _INFERENCE_STATS["latencies_ms"] = _INFERENCE_STATS["latencies_ms"][-100:]
                return text
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                _INFERENCE_STATS["total_errors"] += 1
                _INFERENCE_STATS["last_error"] = str(e)
                return ""

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """Generate text with streaming."""
        if self._cli_engine:
            text = self._cli_engine.generate(
                prompt, max_tokens, temperature, top_p, repeat_penalty, stop
            )
            for word in text.split():
                yield word + " "
            return

        if not self._model_loaded:
            if not self.load_model():
                return

        def _stream():
            with self._lock:
                try:
                    _INFERENCE_STATS["streaming_requests"] += 1
                    start = time.perf_counter()
                    tokens_yielded = 0
                    for chunk in self._llama(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=repeat_penalty,
                        stop=stop or [],
                        echo=False,
                        stream=True,
                    ):
                        token = chunk["choices"][0]["text"]
                        tokens_yielded += 1
                        yield token
                    elapsed = time.perf_counter() - start
                    _INFERENCE_STATS["total_requests"] += 1
                    _INFERENCE_STATS["total_tokens"] += tokens_yielded
                    _INFERENCE_STATS["latencies_ms"].append(elapsed * 1000)
                    if len(_INFERENCE_STATS["latencies_ms"]) > 100:
                        _INFERENCE_STATS["latencies_ms"] = _INFERENCE_STATS["latencies_ms"][-100:]
                except Exception as e:
                    logger.error(f"Streaming failed: {e}")
                    _INFERENCE_STATS["total_errors"] += 1
                    _INFERENCE_STATS["last_error"] = str(e)

        return _stream()

    def benchmark(
        self, prompt: str = "The quick brown fox", num_tokens: int = 50
    ) -> Dict[str, Any]:
        """Benchmark inference speed."""
        if self._cli_engine:
            return self._cli_engine.benchmark(prompt, num_tokens)

        if not self._model_loaded:
            if not self.load_model():
                return {"error": "Failed to load model"}

        start = time.perf_counter()
        text = self.generate(prompt, max_tokens=num_tokens, temperature=0)
        end = time.perf_counter()

        elapsed = end - start
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

        return {
            "tokens_generated": len(text.split()),
            "time_seconds": elapsed,
            "tokens_per_second": tokens_per_sec,
            "text_preview": text[:100],
            "backend": "llama-cpp-python",
        }

    def warmup(self, num_tokens: int = 10) -> bool:
        """Warmup the model with a dummy inference."""
        if self._cli_engine:
            return True

        if not self._model_loaded:
            if not self.load_model():
                return False

        try:
            self._llama(
                "warmup",
                max_tokens=num_tokens,
                temperature=0,
                echo=False,
            )
            logger.info("Model warmup complete")
            return True
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            return False

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        """Generate text for multiple prompts in sequence."""
        results = []
        for prompt in prompts:
            text = self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            results.append(text)
        return results

    def batch_stream_generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[AsyncIterator[str]]:
        """Generate streaming text for multiple prompts."""
        streams = []
        for prompt in prompts:
            stream = self.generate_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            streams.append(stream)
        return streams

    def unload(self):
        """Unload model from memory."""
        with self._lock:
            if self._cli_engine:
                self._cli_engine.unload()
                self._cli_engine = None
            if self._llama is not None:
                del self._llama
                self._llama = None
                self._model_loaded = False
                logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False

    def profile_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """Profile generation with detailed timing breakdown."""
        if self._cli_engine:
            return self._cli_engine.benchmark(prompt, max_tokens)

        if not self._model_loaded:
            if not self.load_model():
                return {"error": "Failed to load model"}

        times = {}

        t0 = time.perf_counter()
        result = self._llama(
            prompt,
            max_tokens=max_tokens,
            echo=False,
            stream=False,
            **kwargs,
        )
        t1 = time.perf_counter()

        text = result["choices"][0]["text"]
        tokens = len(text.split())
        times["generation_ms"] = (t1 - t0) * 1000
        times["tokens_per_second"] = tokens / (t1 - t0) if (t1 - t0) > 0 else 0
        times["tokens_generated"] = tokens

        return {
            "prompt": prompt[:50],
            "times": times,
            "text_preview": text[:100],
            "config": {
                "max_tokens": max_tokens,
                "n_ctx": self.config.n_ctx,
                "n_threads": self.config.n_threads,
                "n_gpu_layers": self.config.n_gpu_layers,
            },
        }


class InferenceProfiler:
    """Profiler for tracking inference performance."""

    def __init__(self):
        self._profiles: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, float]] = None

    def start(self, name: str = "default"):
        """Start a profiling session."""
        self._current = {"name": name, "start": time.perf_counter()}
        return self

    def end(self) -> float:
        """End profiling session and return elapsed time in ms."""
        if self._current is None:
            return 0
        elapsed = (time.perf_counter() - self._current["start"]) * 1000
        self._profiles.append(
            {
                "name": self._current["name"],
                "elapsed_ms": elapsed,
            }
        )
        self._current = None
        return elapsed

    def get_profiles(self) -> List[Dict[str, Any]]:
        """Get all profile results."""
        return self._profiles

    def clear(self):
        """Clear all profiles."""
        self._profiles.clear()

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._profiles:
            return {}
        total = sum(p["elapsed_ms"] for p in self._profiles)
        return {
            "total_profiles": len(self._profiles),
            "total_ms": total,
            "avg_ms": total / len(self._profiles),
            "min_ms": min(p["elapsed_ms"] for p in self._profiles),
            "max_ms": max(p["elapsed_ms"] for p in self._profiles),
        }


class OllamaInferenceEngine:
    """
    Inference engine using Ollama API as backend.

    This provides a reference implementation to benchmark against while
    building our native inference core.
    """

    def __init__(
        self,
        model: str = "llama3.2:1b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self._available = None

    def check_connection(self) -> bool:
        """Check if Ollama is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Generate text via Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def benchmark(
        self, prompt: str = "The quick brown fox", num_tokens: int = 50
    ) -> Dict[str, Any]:
        """Benchmark inference speed."""
        start = time.perf_counter()
        result = self.generate(prompt, max_tokens=num_tokens, temperature=0)
        end = time.perf_counter()

        elapsed = end - start
        tokens = result.get("eval_count", 0)

        return {
            "tokens_generated": tokens,
            "time_seconds": elapsed,
            "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
            "text": result.get("response", "")[:100],
            "backend": "ollama",
            "model": self.model,
        }


class LlamaCLIInferenceEngine:
    """
    Inference engine using llama-cli subprocess.

    This is a fallback when llama-cpp-python is not available.
    Less efficient due to subprocess overhead but provides direct llama.cpp access.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 6,
        n_gpu_layers: int = -1,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads

        if n_gpu_layers == -1:
            model_size_gb = 1.5
            if Path(model_path).exists():
                try:
                    model_size_gb = Path(model_path).stat().st_size / (1024**3)
                except Exception:
                    pass
            self.n_gpu_layers = auto_select_backend(model_size_gb)
        else:
            self.n_gpu_layers = n_gpu_layers

        self._process: Optional[subprocess.Popen] = None
        self._input_buffer = ""
        self._output_buffer = ""
        self._lock = threading.Lock()

    def _start_interactive(self):
        """Start llama-cli in interactive mode."""
        if self._process is not None:
            return

        cmd = [
            LLAMA_CLI_PATH,
            "-m",
            self.model_path,
            "-c",
            str(self.n_ctx),
            "-t",
            str(self.n_threads),
            "-ngl",
            str(self.n_gpu_layers),
            "-i",  # interactive mode
            "--log-disable",
        ]

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text using llama-completion (non-interactive)."""
        if not Path(self.model_path).exists():
            logger.error(f"Model not found: {self.model_path}")
            return ""

        cmd = [
            "llama-completion",
            "-m", self.model_path,
            "-c", str(self.n_ctx),
            "-t", str(self.n_threads),
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--repeat-penalty", str(repeat_penalty),
            "--no-display-prompt",
        ]

        try:
            # Use shell to pipe input (echo)
            input_cmd = f"echo {repr(prompt)}"
            cmd_str = " ".join([
                input_cmd,
                "| llama-simple",
                "-m", self.model_path,
                "-c", str(self.n_ctx),
                "-t", str(self.n_threads),
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "-no-cnv",  # Non-conversation mode
            ])
            
            result = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

            output = result.stdout + result.stderr
            logger.info(f"llama output[:300]: {output[:300]}")
            
            # Parse: output is "prompt + generated text" + perf lines
            if prompt in output:
                # Get rest after prompt
                generated = output.split(prompt)[-1].strip()
            else:
                generated = output
            
            # Remove trailing performance metrics
            lines = generated.split("\n")
            clean = [l for l in lines if l.strip() and not l.strip().startswith(("~", "main:", "llama_", "load", "common_", "print_", "init_", "load_"))]
            return " ".join(clean).strip() if clean else generated.strip()
        except subprocess.TimeoutExpired:
            logger.error("llama-completion timed out")
            return ""


    def benchmark(
        self, prompt: str = "The quick brown fox", num_tokens: int = 50
    ) -> Dict[str, Any]:
        """Benchmark inference speed."""
        start = time.perf_counter()
        text = self.generate(prompt, max_tokens=num_tokens, temperature=0)
        end = time.perf_counter()

        elapsed = end - start
        tokens = len(text.split()) if text else 0

        return {
            "tokens_generated": tokens,
            "time_seconds": elapsed,
            "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
            "text_preview": text[:100] if text else "",
            "backend": "llama-cli",
        }

    def unload(self):
        """Clean up subprocess."""
        if self._process:
            self._process.terminate()
            self._process = None

    @property
    def is_loaded(self) -> bool:
        return Path(self.model_path).exists()


def get_cached_engine(model_path: str, **kwargs) -> LlamaInferenceEngine:
    """Get or create a cached inference engine for the model."""
    global _MODEL_CACHE

    with _CACHE_LOCK:
        if model_path not in _MODEL_CACHE:
            config = LlamaInferenceConfig(model_path=model_path, **kwargs)
            _MODEL_CACHE[model_path] = LlamaInferenceEngine(config)
            _MODEL_CACHE[model_path].load_model()
        else:
            _INFERENCE_STATS["cache_hits"] += 1
        return _MODEL_CACHE[model_path]


def clear_model_cache():
    """Clear the model cache and free memory."""
    global _MODEL_CACHE
    with _CACHE_LOCK:
        for engine in _MODEL_CACHE.values():
            engine.unload()
        _MODEL_CACHE.clear()


def get_inference_stats() -> Dict[str, Any]:
    """Get inference statistics."""
    stats = dict(_INFERENCE_STATS)
    if stats["total_time"] > 0:
        stats["avg_tokens_per_second"] = stats["total_tokens"] / stats["total_time"]
    else:
        stats["avg_tokens_per_second"] = 0.0
    stats["cached_models"] = list(_MODEL_CACHE.keys())
    return stats


def reset_inference_stats() -> None:
    """Reset inference statistics to initial state."""
    global _INFERENCE_STATS
    _INFERENCE_STATS = {
        "total_requests": 0,
        "total_tokens": 0,
        "total_time": 0.0,
        "cache_hits": 0,
        "latencies_ms": [],
        "request_ids": [],
        "total_errors": 0,
        "last_error": None,
        "streaming_requests": 0,
    }


def get_latency_histogram() -> Dict[str, Any]:
    """Get latency histogram for recent requests."""
    latencies = _INFERENCE_STATS.get("latencies_ms", [])
    if not latencies:
        return {"count": 0, "p50": 0, "p90": 0, "p99": 0}

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    return {
        "count": n,
        "p50": sorted_latencies[int(n * 0.5)] if n > 0 else 0,
        "p90": sorted_latencies[int(n * 0.9)] if n > 0 else 0,
        "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
        "min": min(latencies),
        "max": max(latencies),
        "avg": sum(latencies) / n,
    }


def get_metrics_summary() -> Dict[str, Any]:
    """Get a comprehensive summary of all inference metrics."""
    stats = get_inference_stats()
    latency = get_latency_histogram()
    memory = get_memory_usage()

    return {
        "requests": {
            "total": stats.get("total_requests", 0),
            "streaming": stats.get("streaming_requests", 0),
            "errors": stats.get("total_errors", 0),
            "tokens": stats.get("total_tokens", 0),
        },
        "performance": {
            "avg_tokens_per_second": round(stats.get("avg_tokens_per_second", 0), 2),
            "total_time_seconds": round(stats.get("total_time", 0), 2),
            "latency_p50_ms": round(latency.get("p50", 0), 2),
            "latency_p90_ms": round(latency.get("p90", 0), 2),
            "latency_p99_ms": round(latency.get("p99", 0), 2),
        },
        "memory": {
            "rss_mb": round(memory.get("rss_mb", 0), 1) if "error" not in memory else None,
            "cache_hits": stats.get("cache_hits", 0),
            "cached_models": stats.get("cached_models", []),
        },
        "last_error": stats.get("last_error"),
    }


def get_model_info(model_path: str) -> Dict[str, Any]:
    """Get information about a GGUF model file."""
    path = Path(model_path)
    if not path.exists():
        return {"error": f"Model not found: {model_path}"}

    try:
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_bytes / (1024**3)

        return {
            "path": str(path),
            "name": path.name,
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 2),
            "size_gb": round(size_gb, 2),
            "exists": True,
        }
    except Exception as e:
        return {"error": str(e)}


def list_cached_models() -> List[str]:
    """List all currently cached model paths."""
    with _CACHE_LOCK:
        return list(_MODEL_CACHE.keys())


def preload_models(model_paths: List[str]) -> Dict[str, bool]:
    """Preload multiple models into cache."""
    results = {}
    for path in model_paths:
        try:
            engine = get_cached_engine(path)
            results[path] = engine is not None
        except Exception as e:
            logger.error(f"Failed to preload {path}: {e}")
            results[path] = False
    return results


def is_model_cached(model_path: str) -> bool:
    """Check if a model is currently cached."""
    with _CACHE_LOCK:
        return model_path in _MODEL_CACHE


def unload_model(model_path: str) -> bool:
    """Unload a specific model from cache."""
    global _MODEL_CACHE
    with _CACHE_LOCK:
        if model_path in _MODEL_CACHE:
            _MODEL_CACHE[model_path].unload()
            del _MODEL_CACHE[model_path]
            logger.info(f"Unloaded model: {model_path}")
            return True
        return False


def get_cache_stats() -> Dict[str, Any]:
    """Get detailed cache statistics."""
    with _CACHE_LOCK:
        models = list(_MODEL_CACHE.keys())
        loaded = [m for m in models if _MODEL_CACHE[m].is_loaded]

        return {
            "total_cached": len(models),
            "loaded": len(loaded),
            "unloaded": len(models) - len(loaded),
            "models": models,
            "max_size": _MAX_CACHE_SIZE,
        }


def set_max_cache_size(size: int) -> None:
    """Set maximum number of models to cache."""
    global _MAX_CACHE_SIZE
    _MAX_CACHE_SIZE = max(1, size)
    logger.info(f"Max cache size set to {_MAX_CACHE_SIZE}")


def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage for the process."""
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "vms_mb": mem_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
        }
    except ImportError:
        return {"error": "psutil not installed"}


def profile_memory(f):
    """Decorator to profile memory usage of a function."""

    def wrapper(*args, **kwargs):
        mem_before = get_memory_usage()
        result = f(*args, **kwargs)
        mem_after = get_memory_usage()
        if "error" not in mem_before and "error" not in mem_after:
            result["memory_delta_mb"] = mem_after["rss_mb"] - mem_before["rss_mb"]
        return result

    return wrapper


def get_inference_health() -> Dict[str, Any]:
    """Get health status of the inference system."""
    gpu = detect_gpu()
    memory = get_memory_usage()
    stats = get_inference_stats()

    return {
        "status": "healthy",
        "gpu_available": gpu is not None,
        "gpu_name": gpu.name if gpu else None,
        "gpu_recommended": gpu.recommended if gpu else None,
        "llama_cpp_available": LLAMA_CPP_PYTHON_AVAILABLE,
        "llama_cli_available": LLAMA_CLI_AVAILABLE,
        "cached_models": len(list_cached_models()),
        "memory_rss_mb": round(memory.get("rss_mb", 0), 1) if "error" not in memory else None,
        "total_requests": stats.get("total_requests", 0),
        "total_errors": stats.get("total_errors", 0),
    }


def find_gguf_models(search_paths: Optional[List[str]] = None) -> List[Path]:
    """Find all GGUF model files in common locations."""
    if search_paths is None:
        search_paths = [
            str(Path.home() / "models"),
            str(Path.home() / ".ollama" / "models"),
            "/tmp",
            "./models",
        ]

    models = []
    for path_str in search_paths:
        path = Path(path_str)
        if path.exists():
            models.extend(path.glob("**/*.gguf"))

    return models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test llama.cpp inference")
    parser.add_argument("--model", "-m", help="Path to GGUF model")
    parser.add_argument("--prompt", "-p", default="Hello world", help="Prompt")
    parser.add_argument("--tokens", "-t", type=int, default=50, help="Max tokens")
    parser.add_argument("--threads", type=int, default=6, help="CPU threads")
    parser.add_argument(
        "--gpu-layers",
        "-g",
        type=int,
        default=-1,
        help="GPU layers (-1=auto, 0=CPU only, N=layers)",
    )
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--detect-gpu", action="store_true", help="Show GPU detection info")

    args = parser.parse_args()

    if args.detect_gpu:
        gpu = detect_gpu()
        if gpu:
            print(f"GPU: {gpu.name}")
            print(f"Backend: {gpu.backend}")
            print(f"VRAM: {gpu.vram_mb:.0f}MB")
            print(f"Tensor ops: {gpu.has_tensor_ops}")
            print(f"Recommended: {gpu.recommended}")
            print(f"Reason: {gpu.reason}")
        else:
            print("No GPU detected")
        sys.exit(0)

    if args.list:
        models = find_gguf_models()
        print("Available GGUF models:")
        for m in models:
            size = m.stat().st_size / (1024 * 1024)
            print(f"  {m} ({size:.1f} MB)")
        sys.exit(0)

    if not args.model:
        models = find_gguf_models()
        if models:
            args.model = str(models[0])
            print(f"Using model: {args.model}")
        else:
            print("No model found. Use --model or download a GGUF model.")
            sys.exit(1)

    config = LlamaInferenceConfig(
        model_path=args.model,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
    )

    engine = LlamaInferenceEngine(config)

    print(f"Using GPU layers: {config.n_gpu_layers}")

    print(f"Generating {args.tokens} tokens...")
    result = engine.benchmark(args.prompt, args.tokens)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Speed: {result['tokens_per_second']:.1f} tok/s")
        print(f"Time: {result['time_seconds']:.2f}s")
        print(f"Output: {result['text_preview']}...")


class AutoInferenceEngine:
    """
    Automatically selects the best available inference backend.

    Priority:
    1. Ollama (if available and has matching model)
    2. llama-cpp-python (if installed)
    3. llama-cli (fallback)

    For GGUF files, converts to Ollama model name or uses llama-cli.
    """

    def __init__(self, model_path: Optional[str] = None, ollama_model: Optional[str] = None):
        self._ollama: Optional[OllamaInferenceEngine] = None
        self._llama: Optional[LlamaInferenceEngine] = None
        self._backend = "none"

        if ollama_model:
            self._ollama = OllamaInferenceEngine(model=ollama_model)
            if self._ollama.check_connection():
                self._backend = "ollama"
                logger.info(f"Using Ollama backend: {ollama_model}")
                return

        ollama = OllamaInferenceEngine()
        if ollama.check_connection():
            if ollama_model:
                self._ollama = OllamaInferenceEngine(model=ollama_model)
                self._backend = "ollama"
                logger.info(f"Using Ollama backend: {ollama_model}")
                return
            available = ollama.list_models()
            if available:
                self._ollama = OllamaInferenceEngine(model=available[0])
                self._backend = "ollama"
                logger.info(f"Using Ollama backend: {available[0]}")
                return

        if model_path and Path(model_path).exists():
            config = LlamaInferenceConfig(model_path=model_path)
            self._llama = LlamaInferenceEngine(config)
            self._backend = "llama"
            logger.info(f"Using llama.cpp backend: {model_path}")
            return

        raise RuntimeError("No inference backend available")

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        if self._backend == "ollama" and self._ollama:
            result = self._ollama.generate(prompt, max_tokens=max_tokens, **kwargs)
            return result.get("response", "")
        elif self._llama:
            return self._llama.generate(prompt, max_tokens=max_tokens, **kwargs)
        return ""

    def benchmark(self, prompt: str, num_tokens: int = 50) -> Dict[str, Any]:
        if self._backend == "ollama" and self._ollama:
            return self._ollama.benchmark(prompt, num_tokens)
        elif self._llama:
            return self._llama.benchmark(prompt, num_tokens)
        return {"error": "No backend"}

    @property
    def backend(self) -> str:
        return self._backend
