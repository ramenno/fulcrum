#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
fulcrum optimise: Inference optimisation advisor and benchmark tool.
Formulates optimal vLLM configurations for throughput, latency, and cost,
and benchmarks live endpoints to measure TTFT, TPOT, and tokens/sec.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple


def get_gpu_architecture(gpu_type: str) -> str:
    """Classifies GPU/TPU architecture to determine feature compatibility."""
    g = gpu_type.lower()
    if "tpu" in g or g.startswith("ct") or g in ("v5e", "v6e", "v5p"):
        return "tpu"
    elif "h100" in g or "h200" in g or "b200" in g:
        return "hopper"
    elif "l4" in g or "l40" in g or "4090" in g:
        return "ada"
    elif "a100" in g or "a10" in g or "3090" in g:
        return "ampere"
    elif "t4" in g:
        return "turing"
    return "unknown"


def formulate_recommendations(
    model: str,
    gpu_type: str,
    gpu_count: int,
    workload: str,
    max_context: int,
    expected_concurrency: int,
    engine: str = "vllm",
) -> Dict[str, Any]:
    """Generates optimal vLLM or Strata MoE parameters based on hardware capabilities and workload goals."""
    arch = get_gpu_architecture(gpu_type)
    flags: Dict[str, Any] = {}
    rationales: List[str] = []

    if engine.lower() == "strata":
        mid = model.lower()
        is_massive_moe = any(h in mid for h in ("125b", "141b", "8x22b", "flash-next", "70b", "72b", "671b"))
        vram_gb = 22 * gpu_count
        ram_gb = 96 if is_massive_moe else 48
        machine_rec = "g2-standard-32 (128 GB RAM)" if is_massive_moe else "g2-standard-16 (64 GB RAM)"

        flags["--vram"] = vram_gb
        flags["--ram"] = ram_gb
        flags["--vram-reserve-mib"] = 2048
        flags["--expert-cache"] = "auto"
        flags["--evict"] = "lfu"
        flags["--kv"] = "int8"
        flags["--prefill"] = "auto"
        flags["--max-context"] = max_context
        flags["--port"] = 8000

        rationales.append(
            f"3-Tier MoE Memory Hierarchy: Pin attention, router, shared experts, and hot routed experts in {vram_gb} GiB GPU VRAM (--vram {vram_gb}), "
            f"stage warm experts in {ram_gb} GiB Host System RAM (--ram {ram_gb} on {machine_rec}), and stream cold experts from Local NVMe SSD (/mnt/nvme)."
        )
        rationales.append(
            "Reserved 2,048 MiB VRAM headroom (--vram-reserve-mib 2048) for CUDA context, workspace scratch buffers, and asynchronous PCIe Gen4 H2D expert transfers."
        )
        rationales.append(
            "Enabled LFU (Least-Frequently-Used) expert cache eviction (--expert-cache auto --evict lfu) to retain high-activation power-law experts resident in GPU VRAM."
        )
        rationales.append(
            "Enabled INT8 KV cache quantisation (--kv int8) to halve KV cache memory footprint in VRAM and leave maximum capacity for hot MoE expert slots."
        )

        if workload == "latency":
            flags["--spec"] = 3
            flags["--mtp"] = "auto"
            flags["--spec-min-p"] = 0.1
            flags["--skip-w"] = 0.02
            flags["--prompt-skip-w"] = 0.03
            rationales.append(
                "Enabled Multi-Token Prediction / speculative decoding (--spec 3 --mtp auto --spec-min-p 0.1) to amortise expert weight transfers across multiple drafted tokens."
            )
            rationales.append(
                "Configured low-weight expert skipping (--skip-w 0.02 --prompt-skip-w 0.03) to bypass tail router experts with negligible routing weight (<2%), cutting PCIe transfer stalls."
            )
        elif workload == "throughput":
            flags["--pcie-frac"] = 0.85
            rationales.append(
                "Allocated 85% PCIe Gen4 bandwidth budget (--pcie-frac 0.85) for asynchronous layer-ahead expert prefetching under sustained batch generation."
            )
        else:
            rationales.append(
                "Hybrid prefill mode (--prefill auto) dynamically balances batched CPU/GPU expert execution during prompt ingestion."
            )

        cmd_parts = [f"strata-cli serve \\\n    --model {model}"]
        for k, v in flags.items():
            if isinstance(v, bool):
                if v:
                    cmd_parts.append(k)
            else:
                cmd_parts.append(f"{k} {v}")
        full_command = " \\\n    ".join(cmd_parts)

        return {
            "engine": "strata",
            "model": model,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "workload": workload,
            "flags": flags,
            "rationales": rationales,
            "command": full_command,
        }

    # Tensor Parallelism
    flags["--tensor-parallel-size"] = gpu_count
    unit_name = "Cloud TPU chips" if arch == "tpu" else "GPUs"
    rationales.append(f"Set tensor parallelism to {gpu_count} to partition weights across all available {unit_name}.")

    # Context length capping
    flags["--max-model-len"] = max_context
    rationales.append(f"Cap context length to {max_context} to avoid unnecessary KV cache block reservation.")

    if arch == "tpu":
        # Cloud TPU specific recommendations
        flags["--gpu-memory-utilization"] = 0.90
        flags["--block-size"] = 16
        rationales.append(
            "Set --gpu-memory-utilization to 0.90 dynamically to balance KV cache memory against HBM reservations for AOT XLA execution graphs and intermediate activations."
        )
        rationales.append(
            "Pre-Flight KV-Cache Sizing: Verify that model weights plus KV cache at max-model-len do not exceed total available HBM (Total HBM * 0.90 - Weight Memory). On 4-chip slices (125 GiB HBM) for >=27B-32B models, clamp --max-model-len to 4096 to prevent container startup failure."
        )
        rationales.append(
            "Omit --swap-space: CPU swap space for KV cache offloading is unsupported in vLLM Cloud TPU builds (vllm.entrypoints.openai.api_server) and causes fatal argparse exit code 2. All KV blocks are managed directly in TPU HBM via Pallas PagedAttention."
        )
        rationales.append(
            "Pallas PagedAttention: Run with environment variable VLLM_TPU_ENABLE_PALLAS=1 and PJRT_DEVICE=TPU to enable hardware-accelerated KV cache management in HBM without shape recompilations."
        )
        rationales.append(
            "Static Sequence Length Bucketing: Pre-compile computation graphs for static batch sizes and sequence lengths (e.g. 128, 512, 1024, 2048, 4096) to prevent Just-In-Time (JIT) XLA compilation latency stalls."
        )
        rationales.append(
            "Vertex AI Endpoint Settings: Use container 'vllm/vllm-tpu:latest' with --container-shared-memory-size-mb=32768 (32 GiB) and --container-deployment-timeout-seconds=7200 (2 hours) to avoid container timeout during graph compilation."
        )
        rationales.append(
            "Cardinal Rule: Do NOT pass --accelerator=type=...,count=... during endpoint deployment; Cloud TPU machine types (ct5lp-*, ct6e-*, ct5p-*) have hardwired topologies."
        )
        rationales.append(
            "Weight Staging: Stage model weights in a regional Cloud Storage bucket (GCS_MODEL_PATH) co-located with the endpoint region to avoid Hugging Face rate limits and container download timeouts."
        )
    else:
        # GPU Memory Utilisation
        if workload == "throughput":
            flags["--gpu-memory-utilization"] = 0.95
            rationales.append("Set GPU memory utilisation to 0.95 to maximise KV cache space for high concurrency.")
        else:
            flags["--gpu-memory-utilization"] = 0.92
            rationales.append("Set GPU memory utilisation to 0.92 to maintain stable headroom for CUDA graphs and activations.")

        # Quantisation decisions based on architecture
        supports_native_fp8 = arch in ["ada", "hopper"]
        if supports_native_fp8 and workload in ["throughput", "balanced"]:
            flags["--quantization"] = "fp8"
            rationales.append(
                f"GPU architecture ({arch.title()}) natively accelerates FP8 compute and weights, doubling throughput with minimal accuracy loss."
            )
        elif arch == "ampere" and workload == "throughput":
            rationales.append(
                "Ampere architecture does not natively accelerate FP8 in vLLM. Retaining BF16 or evaluate pre-quantised AWQ for 4-bit serving."
            )

    # Batching parameters tailored to workload
    if workload == "latency":
        flags["--max-num-batched-tokens"] = 1024 if arch in ["turing", "ada"] else 2048
        flags["--max-num-seqs"] = min(32, max(4, expected_concurrency * 2))
        rationales.append(
            f"Constrained batch size (--max-num-batched-tokens={flags['--max-num-batched-tokens']}, --max-num-seqs={flags['--max-num-seqs']}) to minimise queuing and guarantee lowest Time Per Output Token (TPOT)."
        )
    elif workload == "throughput":
        flags["--max-num-batched-tokens"] = 4096 if arch == "ada" else 8192
        flags["--max-num-seqs"] = max(128, expected_concurrency * 2)
        rationales.append(
            f"Expanded batch limits (--max-num-batched-tokens={flags['--max-num-batched-tokens']}, --max-num-seqs={flags['--max-num-seqs']}) to maximise compute saturation and overall tokens/second."
        )
    else:  # balanced
        flags["--max-num-batched-tokens"] = 2048 if arch in ["ada", "tpu"] else 4096
        flags["--max-num-seqs"] = max(64, expected_concurrency)
        rationales.append("Balanced batching parameters chosen for good single-stream responsiveness under multi-tenant load.")

    # Chunked prefill (vital for preventing long prompts from blocking concurrent generation)
    flags["--enable-chunked-prefill"] = True
    rationales.append(
        "Enabled chunked prefill: slices large prompt prefills into smaller chunks, eliminating decode jitter for concurrent requests."
    )

    # Prefix caching (essential for multi-turn conversations and system prompts)
    flags["--enable-prefix-caching"] = True
    rationales.append(
        "Enabled prefix caching: reuses KV cache for shared system prompts and multi-turn chat prefixes, yielding near-instant TTFT on repeated prefixes."
    )

    # Speculative decoding recommendation
    speculative_draft = None
    mid = model.lower()
    if "gemma-3-12b" in mid:
        speculative_draft = "google/gemma-3-1b-it"
    elif "llama-3.1-70b" in mid or "llama-3.3-70b" in mid:
        speculative_draft = "meta-llama/Llama-3.1-8B-Instruct"
    elif "qwen2.5-32b" in mid:
        speculative_draft = "Qwen/Qwen2.5-1.5B-Instruct"

    if speculative_draft and workload in ["latency", "balanced"]:
        rationales.append(
            f"Consider speculative decoding with draft model '{speculative_draft}' (--speculative-model {speculative_draft} --num-speculative-tokens 3) to achieve 1.5x-2.2x speedup on single-stream generation."
        )

    # Constrained VRAM and CUDA Graph memory protection (only relevant for GPUs)
    if arch != "tpu":
        is_large_model = any(k in mid for k in ["20b", "24b", "27b", "32b", "70b", "72b"])
        is_tight_vram = is_large_model and ("l4" in gpu_type.lower() and gpu_count <= 2)
        if is_tight_vram:
            flags["--enforce-eager"] = True
            flags["--max-model-len"] = min(max_context, 4096)
            rationales.append(
                "Enforced eager execution (--enforce-eager): Disables Torch Dynamo and Ahead-of-Time CUDA Graph memory profiling. "
                "Reclaims ~5–6 GiB of VRAM per GPU on constrained hardware (e.g. 27B model on 2x L4), preventing fatal OutOfMemoryError during initialization."
            )
            rationales.append(
                "Capped context window to 4096 tokens to reduce initial graph/activation buffer allocations under tight VRAM envelopes."
            )
            rationales.append(
                "Recommended environment variable: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent virtual memory address fragmentation."
            )
            rationales.append(
                "Recommended checkpoint format: Prefer pre-quantised checkpoints (native FP8 safetensors) over dynamic runtime FP8 to avoid dynamic activation scaling buffer overhead."
            )

    # Assemble runnable CLI command
    cmd_parts = [f"vllm serve {model}"]
    for k, v in flags.items():
        if isinstance(v, bool):
            if v:
                cmd_parts.append(k)
        else:
            cmd_parts.append(f"{k} {v}")

    full_command = " \\\n    ".join(cmd_parts)

    return {
        "engine": "vllm",
        "model": model,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "workload": workload,
        "flags": flags,
        "rationales": rationales,
        "command": full_command,
    }


def send_chat_completion_stream(
    url: str,
    headers: Dict[str, str],
    model: str,
    prompt: str,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """Sends a single streaming request to measure Time To First Token (TTFT) and decode latency."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers=headers,
        method="POST",
    )

    start_time = time.perf_counter()
    ttft: Optional[float] = None
    tokens_generated = 0
    first_token_time: Optional[float] = None

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: ") or decoded == "data: [DONE]":
                    continue
                chunk_str = decoded[6:]
                try:
                    chunk = json.loads(chunk_str)
                    choices = chunk.get("choices", [])
                    if choices and "delta" in choices[0]:
                        delta = choices[0]["delta"]
                        if "content" in delta and delta["content"]:
                            now = time.perf_counter()
                            if ttft is None:
                                ttft = now - start_time
                                first_token_time = now
                            tokens_generated += 1
                except json.JSONDecodeError:
                    continue

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        # Calculate decode time per token (excluding prefill time)
        decode_duration = (end_time - first_token_time) if first_token_time else total_duration
        tpot = (decode_duration / max(1, tokens_generated - 1)) if tokens_generated > 1 else 0.0

        return {
            "success": True,
            "ttft_ms": round((ttft or total_duration) * 1000, 2),
            "tpot_ms": round(tpot * 1000, 2),
            "total_ms": round(total_duration * 1000, 2),
            "tokens": tokens_generated,
            "throughput_tps": round(tokens_generated / max(0.001, total_duration), 2),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
            "total_ms": 0.0,
            "tokens": 0,
            "throughput_tps": 0.0,
        }


def run_benchmark(
    endpoint: str,
    model: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs concurrent synthetic load against an OpenAI-compatible endpoint."""
    test_prompt = "Write a concise explanation of how transformer attention mechanisms operate."

    # Resolve target URL and authentication
    if "aiplatform.googleapis.com" in endpoint:
        url = endpoint if endpoint.endswith(":rawPredict") else f"{endpoint.rstrip('/')}:rawPredict"
    elif endpoint.startswith("http://") or endpoint.startswith("https://"):
        url = endpoint if endpoint.endswith("/chat/completions") else f"{endpoint.rstrip('/')}/chat/completions"
    else:
        # Check if numeric endpoint ID or resource name from state database
        from db import get_deployment
        dep = get_deployment(endpoint)
        if dep and dep.get("platform") == "vertex":
            proj = dep.get("project")
            reg = dep.get("region", "us-central1")
            epid = dep.get("endpoint_id")
            url = f"https://{reg}-aiplatform.googleapis.com/v1/projects/{proj}/locations/{reg}/endpoints/{epid}:rawPredict"
        else:
            url = f"http://{endpoint}:8000/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    token = auth_token or os.environ.get("GCP_TOKEN") or os.environ.get("BEARER_TOKEN")
    if not token and "aiplatform.googleapis.com" in url:
        try:
            token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
        except Exception as e:
            print(f"[Warning] Could not obtain gcloud access token: {e}", file=sys.stderr)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"Beginning benchmark against {url} with {concurrency} concurrent streams ({num_requests} total requests)...")

    results: List[Dict[str, Any]] = []
    start_bench = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_chat_completion_stream, url, headers, model, test_prompt, max_tokens)
            for _ in range(num_requests)
        ]
        for f in as_completed(futures):
            res = f.result()
            results.append(res)

    total_bench_duration = time.perf_counter() - start_bench

    successful = [r for r in results if r.get("success")]
    if not successful:
        return {
            "success": False,
            "error": "All benchmark requests failed.",
            "details": [r.get("error") for r in results],
        }

    ttft_values = sorted([r["ttft_ms"] for r in successful])
    tpot_values = sorted([r["tpot_ms"] for r in successful])
    total_tokens = sum([r["tokens"] for r in successful])

    def p(vals: List[float], percentile: float) -> float:
        idx = int(len(vals) * percentile)
        return vals[min(idx, len(vals) - 1)]

    summary = {
        "success": True,
        "total_requests": num_requests,
        "successful_requests": len(successful),
        "concurrency": concurrency,
        "total_time_seconds": round(total_bench_duration, 2),
        "total_tokens_generated": total_tokens,
        "aggregate_throughput_tokens_per_sec": round(total_tokens / total_bench_duration, 2),
        "ttft_ms": {
            "min": ttft_values[0],
            "p50": p(ttft_values, 0.50),
            "p90": p(ttft_values, 0.90),
            "p99": p(ttft_values, 0.99),
            "max": ttft_values[-1],
        },
        "tpot_ms": {
            "min": tpot_values[0],
            "p50": p(tpot_values, 0.50),
            "p90": p(tpot_values, 0.90),
            "p99": p(tpot_values, 0.99),
            "max": tpot_values[-1],
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fulcrum optimise: Compute optimal vLLM configurations and benchmark live endpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Formulate optimal vLLM flags for throughput on an L4 GPU
  python3 fulcrum/scripts/optimise_deployment.py advise --model google/gemma-3-12b-it --gpu-type nvidia-l4 --workload throughput

  # Optimise for minimum latency on an A100 GPU
  python3 fulcrum/scripts/optimise_deployment.py advise --model meta-llama/Llama-3.1-8B-Instruct --gpu-type nvidia-tesla-a100 --workload latency

  # Benchmark a running vLLM instance
  python3 fulcrum/scripts/optimise_deployment.py benchmark --endpoint http://34.12.34.56:8000/v1 --model google/gemma-3-12b-it --concurrency 4
""",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: advise
    advise_parser = subparsers.add_parser("advise", help="Calculate optimal vLLM or Strata flags for target hardware and workload")
    advise_parser.add_argument("--model", "-m", type=str, required=True, help="Model ID (e.g. google/gemma-3-12b-it)")
    advise_parser.add_argument("--gpu-type", type=str, default="nvidia-l4", help="GPU type (nvidia-l4, nvidia-tesla-a100, nvidia-h100-80gb)")
    advise_parser.add_argument("--gpu-count", type=int, default=1, help="GPU count (default: 1)")
    advise_parser.add_argument("--workload", "-w", choices=["throughput", "latency", "balanced"], default="balanced", help="Optimisation goal")
    advise_parser.add_argument("--max-context", type=int, default=8192, help="Target context window length (default: 8192)")
    advise_parser.add_argument("--concurrency", "-c", type=int, default=16, help="Expected concurrent request count (default: 16)")
    advise_parser.add_argument("--engine", choices=["vllm", "strata"], default="vllm", help="Inference engine runtime: vllm (default) or strata (3-tier MoE)")
    advise_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # Subcommand: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark latency and throughput against a live vLLM endpoint")
    bench_parser.add_argument("--endpoint", "-e", type=str, required=True, help="Base API URL (e.g. http://<IP>:8000/v1)")
    bench_parser.add_argument("--model", "-m", type=str, required=True, help="Model name deployed on the endpoint")
    bench_parser.add_argument("--concurrency", "-c", type=int, default=4, help="Concurrent client requests (default: 4)")
    bench_parser.add_argument("--requests", "-n", type=int, default=12, help="Total requests to send (default: 12)")
    bench_parser.add_argument("--max-tokens", type=int, default=128, help="Max output tokens per stream (default: 128)")
    bench_parser.add_argument("--token", type=str, help="Bearer authentication token (auto-resolved via gcloud for Vertex endpoints)")
    bench_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    args = parser.parse_args()

    if args.subcommand == "advise":
        rec = formulate_recommendations(
            model=args.model,
            gpu_type=args.gpu_type,
            gpu_count=args.gpu_count,
            workload=args.workload,
            max_context=args.max_context,
            expected_concurrency=args.concurrency,
            engine=args.engine,
        )

        if args.format == "json":
            print(json.dumps(rec, indent=2))
        else:
            print("=" * 80)
            print(f"  /fulcrum optimise: Tuning Formulation for {args.model}")
            print("=" * 80)
            print(f"Inference Engine:    {'Strata MoE (3-Tier Dynamic Expert Streaming)' if args.engine == 'strata' else 'vLLM'}")
            print(f"Hardware Target:     {args.gpu_count}x {args.gpu_type}")
            print(f"Optimisation Goal:   {args.workload.capitalize()}")
            print(f"Context Window:      {args.max_context} tokens")
            print(f"Expected Stream Load: {args.concurrency} concurrent sequences")
            print("-" * 80)
            print("Key Architectural Rationales:")
            for idx, r in enumerate(rec["rationales"], 1):
                print(f"  {idx}. {r}")
            print("-" * 80)
            print(f"Optimised {'Strata' if args.engine == 'strata' else 'vLLM'} Serving Command:")
            print(f"  {rec['command']}")
            print("=" * 80)

    elif args.subcommand == "benchmark":
        summary = run_benchmark(
            endpoint=args.endpoint,
            model=args.model,
            concurrency=args.concurrency,
            num_requests=args.requests,
            max_tokens=args.max_tokens,
            auth_token=args.token,
        )

        if args.format == "json":
            print(json.dumps(summary, indent=2))
        else:
            if not summary["success"]:
                print(f"Benchmark failed: {summary.get('error')}", file=sys.stderr)
                sys.exit(1)

            print("=" * 80)
            print(f"  /fulcrum optimise: Live Endpoint Benchmark Results ({args.model})")
            print("=" * 80)
            print(f"Endpoint:            {args.endpoint}")
            print(f"Concurrency:         {summary['concurrency']} streams")
            print(f"Total Requests:      {summary['successful_requests']} / {summary['total_requests']}")
            print(f"Total Tokens:        {summary['total_tokens_generated']}")
            print(f"Aggregate Throughput:{summary['aggregate_throughput_tokens_per_sec']} tokens/sec")
            print("-" * 80)
            print(f"Time To First Token (TTFT):")
            print(f"  P50: {summary['ttft_ms']['p50']} ms | P90: {summary['ttft_ms']['p90']} ms | P99: {summary['ttft_ms']['p99']} ms")
            print(f"Time Per Output Token (TPOT):")
            print(f"  P50: {summary['tpot_ms']['p50']} ms | P90: {summary['tpot_ms']['p90']} ms | P99: {summary['tpot_ms']['p99']} ms")
            print("=" * 80)


if __name__ == "__main__":
    main()

