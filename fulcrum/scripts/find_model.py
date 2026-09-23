#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
fulcrum find: Model discovery and hardware requirement calculator.
Helps find the right open-weight model based on user needs and calculates
exact VRAM and GCP GPU requirements for inference serving.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelMetadata:
    id: str
    name: str
    organisation: str
    parameters_billion: float
    active_parameters_billion: float
    default_context: int
    max_context: int
    architecture: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    license: str
    primary_tasks: List[str]
    strengths: str
    supports_fp8: bool
    gated: bool


# Comprehensive catalogue of leading open-weight models
MODELS_CATALOGUE: List[ModelMetadata] = [
    # Google Gemma series
    ModelMetadata(
        id="google/gemma-3-1b-it",
        name="Gemma 3 1B Instruct",
        organisation="Google",
        parameters_billion=1.0,
        active_parameters_billion=1.0,
        default_context=8192,
        max_context=32768,
        architecture="gemma3",
        num_layers=18,
        num_kv_heads=1,
        head_dim=256,
        license="Gemma Terms of Use",
        primary_tasks=["general", "mobile", "edge", "summarisation"],
        strengths="Ultra-low latency, mobile & edge deployments, basic conversational tasks.",
        supports_fp8=True,
        gated=True,
    ),
    ModelMetadata(
        id="google/gemma-3-4b-it",
        name="Gemma 3 4B Instruct",
        organisation="Google",
        parameters_billion=4.0,
        active_parameters_billion=4.0,
        default_context=8192,
        max_context=65536,
        architecture="gemma3",
        num_layers=28,
        num_kv_heads=4,
        head_dim=256,
        license="Gemma Terms of Use",
        primary_tasks=["general", "coding", "agentic", "multilingual"],
        strengths="Exceptional quality-to-size ratio, low latency on single L4 or T4.",
        supports_fp8=True,
        gated=True,
    ),
    ModelMetadata(
        id="google/gemma-3-12b-it",
        name="Gemma 3 12B Instruct",
        organisation="Google",
        parameters_billion=12.0,
        active_parameters_billion=12.0,
        default_context=8192,
        max_context=65536,
        architecture="gemma3",
        num_layers=40,
        num_kv_heads=8,
        head_dim=256,
        license="Gemma Terms of Use",
        primary_tasks=["general", "coding", "reasoning", "multilingual"],
        strengths="Strong reasoning and coding, fits comfortably on a single 24GB L4 GPU.",
        supports_fp8=True,
        gated=True,
    ),
    ModelMetadata(
        id="google/gemma-3-27b-it",
        name="Gemma 3 27B Instruct",
        organisation="Google",
        parameters_billion=27.0,
        active_parameters_billion=27.0,
        default_context=8192,
        max_context=131072,
        architecture="gemma3",
        num_layers=46,
        num_kv_heads=16,
        head_dim=128,
        license="Gemma Terms of Use",
        primary_tasks=["general", "coding", "reasoning", "agentic"],
        strengths="Near 70B class reasoning capability; runs in FP8 or AWQ on single L4, or BF16 on 2x L4 / A100.",
        supports_fp8=True,
        gated=True,
    ),
    ModelMetadata(
        id="google/gemma-2-9b-it",
        name="Gemma 2 9B Instruct",
        organisation="Google",
        parameters_billion=9.2,
        active_parameters_billion=9.2,
        default_context=8192,
        max_context=8192,
        architecture="gemma2",
        num_layers=42,
        num_kv_heads=8,
        head_dim=256,
        license="Gemma Terms of Use",
        primary_tasks=["general", "coding", "reasoning"],
        strengths="Proven battle-tested performance, excellent instruction following.",
        supports_fp8=True,
        gated=True,
    ),

    # Meta Llama series
    ModelMetadata(
        id="meta-llama/Llama-3.2-1B-Instruct",
        name="Llama 3.2 1B Instruct",
        organisation="Meta",
        parameters_billion=1.2,
        active_parameters_billion=1.2,
        default_context=8192,
        max_context=131072,
        architecture="llama",
        num_layers=16,
        num_kv_heads=8,
        head_dim=64,
        license="Llama 3.2 Community",
        primary_tasks=["general", "edge", "summarisation", "on-device"],
        strengths="Compact on-device model with huge 128k context window.",
        supports_fp8=True,
        gated=True,
    ),
    ModelMetadata(
        id="meta-llama/Llama-3.2-3B-Instruct",
        name="Llama 3.2 3B Instruct",
        organisation="Meta",
        parameters_billion=3.2,
        active_parameters_billion=3.2,
        default_context=8192,
        max_context=131072,
        architecture="llama",
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
        license="Llama 3.2 Community",
        primary_tasks=["general", "coding", "agentic"],
        strengths="High throughput, low latency, versatile tool calling.",
        supports_fp8=True,
        gated=True,
    ),
    ModelMetadata(
        id="meta-llama/Llama-3.1-8B-Instruct",
        name="Llama 3.1 8B Instruct",
        organisation="Meta",
        parameters_billion=8.0,
        active_parameters_billion=8.0,
        default_context=8192,
        max_context=131072,
        architecture="llama",
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        license="Llama 3.1 Community",
        primary_tasks=["general", "coding", "reasoning", "agentic"],
        strengths="Industry benchmark for 8B models, 128k context, strong ecosystem support.",
        supports_fp8=True,
        gated=True,
    ),
    ModelMetadata(
        id="meta-llama/Llama-3.3-70B-Instruct",
        name="Llama 3.3 70B Instruct",
        organisation="Meta",
        parameters_billion=70.6,
        active_parameters_billion=70.6,
        default_context=8192,
        max_context=131072,
        architecture="llama",
        num_layers=80,
        num_kv_heads=8,
        head_dim=128,
        license="Llama 3.3 Community",
        primary_tasks=["general", "coding", "reasoning", "agentic", "analysis"],
        strengths="State-of-the-art 70B open weight model matching Llama 3.1 405B on key metrics.",
        supports_fp8=True,
        gated=True,
    ),

    # Qwen series
    ModelMetadata(
        id="Qwen/Qwen2.5-Coder-7B-Instruct",
        name="Qwen 2.5 Coder 7B Instruct",
        organisation="Alibaba Cloud",
        parameters_billion=7.6,
        active_parameters_billion=7.6,
        default_context=8192,
        max_context=131072,
        architecture="qwen2",
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
        license="Apache-2.0",
        primary_tasks=["coding", "general", "agentic"],
        strengths="Top-tier code generation and repository-level reasoning for its size.",
        supports_fp8=True,
        gated=False,
    ),
    ModelMetadata(
        id="Qwen/Qwen2.5-Coder-32B-Instruct",
        name="Qwen 2.5 Coder 32B Instruct",
        organisation="Alibaba Cloud",
        parameters_billion=32.5,
        active_parameters_billion=32.5,
        default_context=8192,
        max_context=131072,
        architecture="qwen2",
        num_layers=64,
        num_kv_heads=8,
        head_dim=128,
        license="Apache-2.0",
        primary_tasks=["coding", "general", "agentic", "math"],
        strengths="Rivals commercial coding models like GPT-4o on coding benchmarks.",
        supports_fp8=True,
        gated=False,
    ),
    ModelMetadata(
        id="Qwen/Qwen2.5-7B-Instruct",
        name="Qwen 2.5 7B Instruct",
        organisation="Alibaba Cloud",
        parameters_billion=7.6,
        active_parameters_billion=7.6,
        default_context=8192,
        max_context=131072,
        architecture="qwen2",
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
        license="Apache-2.0",
        primary_tasks=["general", "coding", "reasoning", "multilingual"],
        strengths="Broad language coverage, strong structured JSON output, Apache 2.0 license.",
        supports_fp8=True,
        gated=False,
    ),
    ModelMetadata(
        id="Qwen/Qwen2.5-32B-Instruct",
        name="Qwen 2.5 32B Instruct",
        organisation="Alibaba Cloud",
        parameters_billion=32.5,
        active_parameters_billion=32.5,
        default_context=8192,
        max_context=131072,
        architecture="qwen2",
        num_layers=64,
        num_kv_heads=8,
        head_dim=128,
        license="Apache-2.0",
        primary_tasks=["general", "reasoning", "coding", "multilingual"],
        strengths="Sweet spot between performance and compute footprint under Apache 2.0.",
        supports_fp8=True,
        gated=False,
    ),
    ModelMetadata(
        id="Qwen/Qwen2.5-72B-Instruct",
        name="Qwen 2.5 72B Instruct",
        organisation="Alibaba Cloud",
        parameters_billion=72.7,
        active_parameters_billion=72.7,
        default_context=8192,
        max_context=131072,
        architecture="qwen2",
        num_layers=80,
        num_kv_heads=8,
        head_dim=128,
        license="Qwen License",
        primary_tasks=["general", "reasoning", "coding", "math", "analysis"],
        strengths="One of the highest scoring dense open-weight models available.",
        supports_fp8=True,
        gated=False,
    ),
    ModelMetadata(
        id="Qwen/Qwen3.8-27B",
        name="Qwen 3.8 27B",
        organisation="Alibaba Cloud",
        parameters_billion=27.0,
        active_parameters_billion=27.0,
        default_context=8192,
        max_context=262144,
        architecture="qwen3",
        num_layers=16,
        num_kv_heads=4,
        head_dim=256,
        license="Apache-2.0",
        primary_tasks=["general", "reasoning", "coding", "agentic", "multilingual"],
        strengths="Hybrid Gated DeltaNet attention, native multimodal vision-language, MTP, and flexible thinking control.",
        supports_fp8=True,
        gated=False,
    ),
    ModelMetadata(
        id="Qwen/Qwen3-8B",
        name="Qwen 3 8B",
        organisation="Alibaba Cloud",
        parameters_billion=8.2,
        active_parameters_billion=8.2,
        default_context=8192,
        max_context=131072,
        architecture="qwen3",
        num_layers=36,
        num_kv_heads=4,
        head_dim=128,
        license="Apache-2.0",
        primary_tasks=["general", "coding", "reasoning", "agentic"],
        strengths="High-efficiency dense 8B model with 128k context, strong reasoning and tool-calling.",
        supports_fp8=True,
        gated=False,
    ),

    # DeepSeek Reasoning & Distillation series
    ModelMetadata(
        id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        name="DeepSeek R1 Distill Qwen 7B",
        organisation="DeepSeek",
        parameters_billion=7.6,
        active_parameters_billion=7.6,
        default_context=8192,
        max_context=131072,
        architecture="qwen2",
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
        license="MIT",
        primary_tasks=["reasoning", "math", "coding"],
        strengths="Superb reasoning, step-by-step mathematical problem solving, MIT licensed.",
        supports_fp8=True,
        gated=False,
    ),
    ModelMetadata(
        id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        name="DeepSeek R1 Distill Qwen 14B",
        organisation="DeepSeek",
        parameters_billion=14.7,
        active_parameters_billion=14.7,
        default_context=8192,
        max_context=131072,
        architecture="qwen2",
        num_layers=48,
        num_kv_heads=8,
        head_dim=128,
        license="MIT",
        primary_tasks=["reasoning", "math", "coding"],
        strengths="Outperforms o1-preview on math benchmarks while running on a single L4 in FP8 or 2x L4 in BF16.",
        supports_fp8=True,
        gated=False,
    ),
    ModelMetadata(
        id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        name="DeepSeek R1 Distill Qwen 32B",
        organisation="DeepSeek",
        parameters_billion=32.5,
        active_parameters_billion=32.5,
        default_context=8192,
        max_context=131072,
        architecture="qwen2",
        num_layers=64,
        num_kv_heads=8,
        head_dim=128,
        license="MIT",
        primary_tasks=["reasoning", "math", "coding", "logic"],
        strengths="Comparable reasoning quality to full OpenAI o1-mini on AIME and MATH benchmarks.",
        supports_fp8=True,
        gated=False,
    ),
    ModelMetadata(
        id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        name="DeepSeek R1 Distill Llama 70B",
        organisation="DeepSeek",
        parameters_billion=70.6,
        active_parameters_billion=70.6,
        default_context=8192,
        max_context=131072,
        architecture="llama",
        num_layers=80,
        num_kv_heads=8,
        head_dim=128,
        license="Llama 3.3 Community",
        primary_tasks=["reasoning", "math", "coding", "scientific"],
        strengths="Frontier reasoning capabilities distilled into dense 70B architecture.",
        supports_fp8=True,
        gated=False,
    ),

    # Mistral AI series
    ModelMetadata(
        id="mistralai/Mistral-Nemo-Instruct-2407",
        name="Mistral NeMo 12B Instruct",
        organisation="Mistral AI",
        parameters_billion=12.2,
        active_parameters_billion=12.2,
        default_context=8192,
        max_context=128000,
        architecture="mistral",
        num_layers=40,
        num_kv_heads=8,
        head_dim=128,
        license="Apache-2.0",
        primary_tasks=["general", "coding", "multilingual"],
        strengths="Co-developed with NVIDIA, efficient inference, native FP8 support, Apache 2.0.",
        supports_fp8=True,
        gated=False,
    ),
    ModelMetadata(
        id="mistralai/Mistral-Small-24B-Instruct-2501",
        name="Mistral Small 24B Instruct",
        organisation="Mistral AI",
        parameters_billion=24.0,
        active_parameters_billion=24.0,
        default_context=8192,
        max_context=32768,
        architecture="mistral",
        num_layers=56,
        num_kv_heads=8,
        head_dim=128,
        license="Apache-2.0",
        primary_tasks=["general", "reasoning", "agentic", "coding"],
        strengths="Advanced reasoning and multilingual performance under Apache 2.0 license.",
        supports_fp8=True,
        gated=False,
    ),

    # Microsoft Phi series
    ModelMetadata(
        id="microsoft/phi-4",
        name="Phi-4 14B",
        organisation="Microsoft",
        parameters_billion=14.7,
        active_parameters_billion=14.7,
        default_context=4096,
        max_context=16384,
        architecture="phi3",
        num_layers=40,
        num_kv_heads=10,
        head_dim=128,
        license="MIT",
        primary_tasks=["reasoning", "math", "coding", "stem"],
        strengths="Industry-leading mathematical and scientific reasoning at 14B parameters. MIT license.",
        supports_fp8=True,
        gated=False,
    ),
]


def calculate_memory_and_hardware(
    model: ModelMetadata,
    precision: str = "bf16",
    context_tokens: int = 8192,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Computes precise memory requirement for weights, KV cache, and runtime activations.
    Maps requirement to standard GCP GPU instances.
    """
    bytes_per_param = {
        "bf16": 2.0,
        "fp16": 2.0,
        "fp8": 1.0,
        "awq": 0.55,
        "gptq": 0.55,
        "int4": 0.55,
    }.get(precision.lower(), 2.0)

    # Model weight footprint with 5% overhead for tensor buffer padding
    weights_vram_gb = (model.parameters_billion * 1e9 * bytes_per_param) / (1024**3) * 1.05

    # Standard KV cache sizing formula:
    # 2 (keys + values) * num_layers * num_kv_heads * head_dim * bytes_per_element * context_tokens * batch_size
    kv_bytes_per_elem = 1.0 if precision.lower() == "fp8" else 2.0
    kv_cache_bytes = (
        2
        * model.num_layers
        * model.num_kv_heads
        * model.head_dim
        * kv_bytes_per_elem
        * context_tokens
        * batch_size
    )
    kv_cache_vram_gb = kv_cache_bytes / (1024**3)

    # Runtime activation and CUDA workspace headroom (~15% of total or minimum 2.5GB)
    base_activation_headroom_gb = max(2.5, (weights_vram_gb + kv_cache_vram_gb) * 0.15)

    # vLLM V1 with PyTorch 2.5+ Ahead-of-Time CUDA Graph compilation and Torch Dynamo
    # allocates ~6.0 GiB of JIT workspace per GPU during the warm-up profiling pass.
    # For models >= 20B parameters, allocating CUDA graphs and dynamic FP8 scale buffers requires
    # a dedicated memory allowance of at least 6 GB per GPU unless eager mode is enforced.
    jit_workspace_per_gpu_gb = 6.0 if model.parameters_billion >= 20.0 else 0.0

    # Select recommended GCP GPU configuration
    gcp_options = [
        {"gpu": "nvidia-l4", "count": 1, "vram": 24, "machine": "g2-standard-8", "cost_hr": "~$0.70/hr"},
        {"gpu": "nvidia-l4", "count": 2, "vram": 48, "machine": "g2-standard-24", "cost_hr": "~$1.40/hr"},
        {"gpu": "nvidia-l4", "count": 4, "vram": 96, "machine": "g2-standard-48", "cost_hr": "~$2.80/hr"},
        {"gpu": "nvidia-l4", "count": 8, "vram": 192, "machine": "g2-standard-96", "cost_hr": "~$5.60/hr"},
        {"gpu": "nvidia-tesla-a100", "count": 1, "vram": 40, "machine": "a2-highgpu-1g", "cost_hr": "~$3.67/hr"},
        {"gpu": "nvidia-a100-80gb", "count": 1, "vram": 80, "machine": "a2-ultragpu-1g", "cost_hr": "~$5.00/hr"},
        {"gpu": "nvidia-a100-80gb", "count": 2, "vram": 160, "machine": "a2-ultragpu-2g", "cost_hr": "~$10.00/hr"},
        {"gpu": "nvidia-a100-80gb", "count": 4, "vram": 320, "machine": "a2-ultragpu-4g", "cost_hr": "~$20.00/hr"},
        {"gpu": "nvidia-h100-80gb", "count": 1, "vram": 80, "machine": "a3-highgpu-1g", "cost_hr": "~$10.50/hr"},
        {"gpu": "nvidia-h100-80gb", "count": 8, "vram": 640, "machine": "a3-highgpu-8g", "cost_hr": "~$84.00/hr"},
    ]

    selected_gcp = None
    for opt in gcp_options:
        # Avoid under-provisioning >=20B models on <=2x L4 where CUDA Graph capture OOMs
        if model.parameters_billion >= 20.0 and opt["gpu"] == "nvidia-l4" and opt["count"] < 4:
            continue

        needed = weights_vram_gb + kv_cache_vram_gb + base_activation_headroom_gb + (jit_workspace_per_gpu_gb * opt["count"])
        if opt["vram"] >= needed:
            selected_gcp = opt
            break

    if not selected_gcp:
        selected_gcp = gcp_options[-1]

    # Select recommended Cloud TPU configuration on Vertex AI (TPU v5e, v6e Trillium, v5p)
    # vLLM on TPU targets --gpu-memory-utilization 0.85 to retain HBM for XLA AOT graph memory.
    tpu_options = [
        {"tpu": "tpu-v5e", "generation": "v5e", "chips": 1, "hbm": 16, "machine": "ct5lp-hightpu-1t", "cost_hr": "~$1.20/hr"},
        {"tpu": "tpu-v5e", "generation": "v5e", "chips": 4, "hbm": 64, "machine": "ct5lp-hightpu-4t", "cost_hr": "~$4.80/hr"},
        {"tpu": "tpu-v5e", "generation": "v5e", "chips": 8, "hbm": 128, "machine": "ct5lp-hightpu-8t", "cost_hr": "~$9.60/hr"},
        {"tpu": "tpu-v6e", "generation": "v6e", "chips": 1, "hbm": 32, "machine": "ct6e-standard-1t", "cost_hr": "~$2.20/hr"},
        {"tpu": "tpu-v6e", "generation": "v6e", "chips": 4, "hbm": 128, "machine": "ct6e-standard-4t", "cost_hr": "~$8.80/hr"},
        {"tpu": "tpu-v6e", "generation": "v6e", "chips": 8, "hbm": 256, "machine": "ct6e-standard-8t", "cost_hr": "~$17.60/hr"},
        {"tpu": "tpu-v5p", "generation": "v5p", "chips": 4, "hbm": 380, "machine": "ct5p-hightpu-4t", "cost_hr": "~$16.80/hr"},
        {"tpu": "tpu-v5p", "generation": "v5p", "chips": 8, "hbm": 760, "machine": "ct5p-hightpu-8t", "cost_hr": "~$33.60/hr"},
    ]

    selected_tpu = None
    tpu_needed = weights_vram_gb + kv_cache_vram_gb + base_activation_headroom_gb
    for opt in tpu_options:
        if (opt["hbm"] * 0.85) >= tpu_needed:
            selected_tpu = opt
            break

    if not selected_tpu:
        selected_tpu = tpu_options[-1]

    total_vram_needed_gb = weights_vram_gb + kv_cache_vram_gb + base_activation_headroom_gb + (jit_workspace_per_gpu_gb * selected_gcp["count"])
    activation_headroom_gb = base_activation_headroom_gb + (jit_workspace_per_gpu_gb * selected_gcp["count"])

    # Generate vLLM command flags
    tensor_parallel = selected_gcp["count"]
    quant_flag = ""
    if precision.lower() == "fp8":
        quant_flag = " --quantization fp8"
    elif precision.lower() in ["awq", "gptq"]:
        quant_flag = f" --quantization {precision.lower()}"

    eager_flag = ""
    if model.parameters_billion >= 20.0 and selected_gcp["gpu"] == "nvidia-l4" and selected_gcp["count"] <= 2:
        eager_flag = " --enforce-eager"

    vllm_cmd = (
        f"vllm serve {model.id}"
        f" --tensor-parallel-size {tensor_parallel}"
        f" --max-model-len {min(model.max_context, context_tokens)}"
        f"{quant_flag}"
        f"{eager_flag}"
        " --gpu-memory-utilization 0.95"
        " --enable-chunked-prefill"
        " --enable-prefix-caching"
    )

    return {
        "precision": precision,
        "weights_vram_gb": round(weights_vram_gb, 2),
        "kv_cache_vram_gb": round(kv_cache_vram_gb, 2),
        "activation_headroom_gb": round(activation_headroom_gb, 2),
        "total_vram_needed_gb": round(total_vram_needed_gb, 2),
        "recommended_gcp": selected_gcp,
        "recommended_tpu": selected_tpu,
        "vllm_command": vllm_cmd,
    }


def search_models(
    task: Optional[str] = None,
    max_params: Optional[float] = None,
    vram_limit_gb: Optional[float] = None,
    precision: str = "bf16",
    license_type: Optional[str] = None,
    query: Optional[str] = None,
    context_window: int = 8192,
) -> List[Dict[str, Any]]:
    """Filters the catalogue according to user requirements."""
    results = []

    for model in MODELS_CATALOGUE:
        # Filter by search query
        if query:
            q = query.lower()
            match_name = q in model.name.lower() or q in model.id.lower() or q in model.strengths.lower()
            if not match_name:
                continue

        # Filter by primary task
        if task:
            t = task.lower()
            if not any(t in pt.lower() for pt in model.primary_tasks):
                continue

        # Filter by max parameter count
        if max_params and model.parameters_billion > max_params:
            continue

        # Filter by license
        if license_type:
            lt = license_type.lower()
            if lt == "permissive" and not ("apache" in model.license.lower() or "mit" in model.license.lower()):
                continue
            elif lt in ["apache", "apache-2.0"] and "apache" not in model.license.lower():
                continue
            elif lt == "mit" and "mit" not in model.license.lower():
                continue

        # Calculate memory requirements
        calc = calculate_memory_and_hardware(
            model=model,
            precision=precision,
            context_tokens=context_window,
            batch_size=8,
        )

        # Filter by VRAM limit
        if vram_limit_gb and calc["total_vram_needed_gb"] > vram_limit_gb:
            continue

        item = {
            "metadata": asdict(model),
            "hardware_calc": calc,
        }
        results.append(item)

    # Sort results by parameter count ascending
    results.sort(key=lambda x: x["metadata"]["parameters_billion"])
    return results


def format_table(results: List[Dict[str, Any]], accelerator: str = "all") -> str:
    """Formats model recommendations into a clear ASCII table."""
    if not results:
        return "No open-weight models found matching the specified criteria."

    lines = []
    if accelerator == "tpu":
        header = f"{'MODEL ID':<36} | {'PARAMS':<7} | {'MEM (GB)':<8} | {'RECOMMENDED CLOUD TPU':<25} | {'EST COST':<10}"
    elif accelerator == "gpu":
        header = f"{'MODEL ID':<36} | {'PARAMS':<7} | {'MEM (GB)':<8} | {'RECOMMENDED GCP GPU':<26} | {'EST COST':<10}"
    else:
        header = f"{'MODEL ID':<34} | {'PARAMS':<7} | {'GPU INSTANCE':<22} | {'TPU INSTANCE':<22} | {'LICENSE':<12}"
    
    separator = "-" * len(header)
    lines.append(separator)
    lines.append(header)
    lines.append(separator)

    for item in results:
        m = item["metadata"]
        h = item["hardware_calc"]
        g = h["recommended_gcp"]
        t = h.get("recommended_tpu", {})
        gpu_str = f"{g['count']}x {g['gpu'].replace('nvidia-', '')} ({g['machine']})"
        tpu_str = f"{t.get('chips', 1)}x {t.get('generation', 'v5e')} ({t.get('machine', 'n/a')})"
        params_str = f"{m['parameters_billion']}B"
        vram_str = f"{h['total_vram_needed_gb']} GB"
        license_str = m["license"][:12]
        
        if accelerator == "tpu":
            lines.append(f"{m['id']:<36} | {params_str:<7} | {vram_str:<8} | {tpu_str:<25} | {t.get('cost_hr', 'n/a'):<10}")
        elif accelerator == "gpu":
            lines.append(f"{m['id']:<36} | {params_str:<7} | {vram_str:<8} | {gpu_str:<26} | {g['cost_hr']:<10}")
        else:
            lines.append(f"{m['id']:<34} | {params_str:<7} | {gpu_str:<22} | {tpu_str:<22} | {license_str:<12}")

    lines.append(separator)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fulcrum find: Discover open-weight, downloadable, self-deployable models and calculate GCP hardware requirements. (Excludes proprietary API-only models.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find models suitable for a 24GB VRAM GPU (NVIDIA L4)
  python3 fulcrum/scripts/find_model.py --vram-limit 24

  # Find models with Cloud TPU recommendations
  python3 fulcrum/scripts/find_model.py --accelerator tpu

  # Find coding models under Apache-2.0 license
  python3 fulcrum/scripts/find_model.py --task coding --license permissive

  # Find reasoning models with FP8 precision
  python3 fulcrum/scripts/find_model.py --task reasoning --precision fp8

  # Output in JSON format
  python3 fulcrum/scripts/find_model.py --query gemma --format json
""",
    )
    parser.add_argument("--query", "-q", type=str, help="Search term across model name, ID, or strengths")
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["general", "coding", "reasoning", "math", "multilingual", "agentic", "edge"],
        help="Target capability or workload type",
    )
    parser.add_argument(
        "--accelerator",
        "-a",
        choices=["all", "gpu", "tpu"],
        default="all",
        help="Accelerator architecture to display in recommendations (all, gpu, or tpu)",
    )
    parser.add_argument("--max-params", "-p", type=float, help="Maximum parameter count in billions (e.g. 14, 32, 70)")
    parser.add_argument("--vram-limit", "-v", type=float, help="Maximum available GPU VRAM in GB (e.g. 24, 48, 80)")
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp8", "awq", "gptq"],
        default="bf16",
        help="Inference precision (default: bf16)",
    )
    parser.add_argument(
        "--license",
        choices=["permissive", "apache", "mit", "all"],
        default="all",
        help="License restriction filter (default: all)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=8192,
        help="Target context window in tokens for KV cache calculation (default: 8192)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    args = parser.parse_args()

    results = search_models(
        task=args.task,
        max_params=args.max_params,
        vram_limit_gb=args.vram_limit,
        precision=args.precision,
        license_type=args.license if args.license != "all" else None,
        query=args.query,
        context_window=args.context_window,
    )

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print(format_table(results, accelerator=args.accelerator))
        if results:
            print("\nTop Recommendation Details:")
            top = results[0]
            m = top["metadata"]
            h = top["hardware_calc"]
            g = h["recommended_gcp"]
            t = h.get("recommended_tpu", {})
            print(f"  Model:             {m['name']} ({m['id']})")
            print(f"  Strengths:         {m['strengths']}")
            print(f"  GCP GPU Machine:   {g['machine']} ({g['count']}x {g['gpu']}, {g['vram']} GB VRAM total) - {g['cost_hr']}")
            if t:
                print(f"  Cloud TPU Machine: {t.get('machine')} ({t.get('chips')}x {t.get('tpu')}, {t.get('hbm')} GB HBM total) - {t.get('cost_hr')}")
            print(f"  Memory Split:      Weights: {h['weights_vram_gb']}GB | KV Cache: {h['kv_cache_vram_gb']}GB | Headroom: {h['activation_headroom_gb']}GB")
            print(f"  vLLM Run Command:")
            print(f"    {h['vllm_command']}")


if __name__ == "__main__":
    main()

