#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
fulcrum benchmark: Performance benchmarking suite for open-weight model deployments.
Measures prefill latency (TTFT), decode latency (TPOT / ITL), inter-token jitter,
generation and prompt throughput, concurrency scaling, and vLLM KV cache efficiency
across Vertex AI Endpoints and Compute Engine instances.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import local state database manager
sys.path.insert(0, str(Path(__file__).parent))
try:
    import db
except ImportError:
    from fulcrum.scripts import db


# Pre-configured synthetic workload profiles for open-weight models
WORKLOAD_PROFILES: Dict[str, Dict[str, Any]] = {
    "chat": {
        "description": "Balanced conversational profile (~150 prompt tokens, 150 max output tokens).",
        "prompt": (
            "You are a helpful and knowledgeable AI assistant. Please explain the architecture "
            "of Transformer self-attention, describing queries, keys, values, and how multi-head "
            "attention enables representation learning across diverse linguistic contexts."
        ),
        "max_tokens": 150,
        "expected_prompt_tokens": 40,
    },
    "rag": {
        "description": "Prefill-heavy RAG profile (~1,000 prompt tokens, 80 max output tokens) to stress prompt attention.",
        "prompt": (
            "Review the following technical context and answer the question succinctly.\n\n"
            "Context: "
            + (
                "Deep learning models for natural language processing rely heavily on sequence-to-sequence "
                "modelling, encoder-decoder architectures, and decoder-only causal language modelling. In causal "
                "language models, tokens attend only to preceding tokens in the sequence through masked self-attention. "
                "FlashAttention and FlashAttention-2 optimise exact attention on modern GPU tensor cores by tiling "
                "softmax computation and minimising memory read/write operations between High Bandwidth Memory (HBM) "
                "and SRAM on-chip memory. PagedAttention, introduced in vLLM, manages key-value (KV) cache memory "
                "by dividing sequences into non-contiguous physical memory blocks, completely eliminating external "
                "memory fragmentation and enabling up to 4x higher serving concurrency under heavy traffic. "
            )
            * 8
            + "\n\nQuestion: How does PagedAttention eliminate memory fragmentation in modern LLM serving engines?"
        ),
        "max_tokens": 80,
        "expected_prompt_tokens": 450,
    },
    "code": {
        "description": "Decode-heavy reasoning/code generation profile (~80 prompt tokens, 450 max output tokens).",
        "prompt": (
            "Implement a production-grade LRU Cache in Python using a doubly linked list and a hash map. "
            "Include get, put, and evict operations with O(1) time complexity, type annotations, docstrings, "
            "and defensive edge case handling for zero capacity."
        ),
        "max_tokens": 450,
        "expected_prompt_tokens": 45,
    },
    "reasoning": {
        "description": "Decode-heavy analytical reasoning profile (~100 prompt tokens, 500 max output tokens).",
        "prompt": (
            "Provide a step-by-step mathematical proof that the square root of 2 is irrational using proof by "
            "contradiction. Thoroughly explain each algebraic transformation, the fundamental theorem of arithmetic, "
            "and why parity constraints lead to an impossible premise."
        ),
        "max_tokens": 500,
        "expected_prompt_tokens": 40,
    },
}


def resolve_endpoint_target(
    identifier: Optional[str] = None,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
    """
    Resolves the benchmark URL, model name, and associated deployment record.
    Supports deployment IDs from deployments.json as well as direct HTTP/HTTPS URLs.
    """
    dep = None
    if identifier:
        dep = db.get_deployment(identifier)
        if not dep:
            raise ValueError(f"Deployment '{identifier}' not found in local database.")

    if dep:
        model_name = dep.get("model") or model
        platform = dep.get("platform", "vertex")
        if platform == "vertex":
            proj = dep.get("project")
            reg = dep.get("region", "us-central1")
            endpoint_name = dep.get("endpoint_id") or dep.get("endpoint_numeric_id") or dep.get("endpoint_name") or dep.get("id")
            # Vertex AI rawPredict route for streaming SSE
            target_url = f"https://{reg}-aiplatform.googleapis.com/v1/projects/{proj}/locations/{reg}/endpoints/{endpoint_name}:rawPredict"
        else:
            ip = dep.get("ip_address") or dep.get("ip") or "127.0.0.1"
            port = dep.get("port", 8000)
            target_url = f"http://{ip}:{port}/v1/chat/completions"
        return target_url, model_name, dep.get("id"), dep

    if not endpoint:
        raise ValueError("Either --id <deployment_name> or --endpoint <url> must be specified.")
    if not model:
        raise ValueError("--model <name> must be specified when using direct --endpoint.")

    # Format direct endpoint URL
    if "aiplatform.googleapis.com" in endpoint:
        target_url = endpoint if endpoint.endswith(":rawPredict") else f"{endpoint.rstrip('/')}:rawPredict"
    elif endpoint.startswith("http://") or endpoint.startswith("https://"):
        target_url = endpoint if endpoint.endswith("/chat/completions") else f"{endpoint.rstrip('/')}/chat/completions"
    else:
        target_url = f"http://{endpoint}:8000/v1/chat/completions"

    return target_url, model, None, None


def get_auth_token(target_url: str, user_token: Optional[str] = None) -> Optional[str]:
    """Retrieves Google Cloud IAM token if querying Vertex AI endpoint, or uses supplied token."""
    if user_token:
        return user_token
    token = os.environ.get("GCP_TOKEN") or os.environ.get("BEARER_TOKEN")
    if token:
        return token
    if "aiplatform.googleapis.com" in target_url:
        try:
            token = subprocess.check_output(
                ["gcloud", "auth", "print-access-token"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return token
        except Exception:
            pass
    return None


def query_prometheus_metrics(target_url: str) -> Optional[Dict[str, float]]:
    """
    Attempts to scrape Prometheus metrics from vLLM (/metrics endpoint)
    if accessible on Compute Engine or local deployments.
    """
    if "aiplatform.googleapis.com" in target_url:
        return None
    try:
        # Extract host and port
        if "/v1/" in target_url:
            base_url = target_url.split("/v1/")[0]
        else:
            base_url = target_url.rsplit(":", 1)[0]
        metrics_url = f"{base_url}/metrics"
        req = urllib.request.Request(metrics_url, headers={"User-Agent": "fulcrum-benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            text = resp.read().decode("utf-8")
            metrics: Dict[str, float] = {}
            for line in text.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    k, v = parts[0], parts[1]
                    try:
                        metrics[k] = float(v)
                    except ValueError:
                        pass
            return metrics
    except Exception:
        return None


def simulate_synthetic_stream(
    prompt: str,
    max_tokens: int,
    base_ttft_ms: float = 45.0,
    base_tpot_ms: float = 12.0,
) -> Dict[str, Any]:
    """
    Simulates a streaming response for offline testing or mock verification.
    Models realistic prefill and decode delays with jitter.
    """
    tokens_to_generate = min(max_tokens, 120)
    # Estimate prompt token count
    prompt_tokens = max(10, len(prompt.split()) * 4 // 3)

    # Micro sleep for realistic thread scheduling simulation without stalling tests
    time.sleep(0.001)
    ttft_sec = base_ttft_ms / 1000.0

    itls: List[float] = []
    # Generate tokens with Gaussian jitter around base_tpot_ms
    for _ in range(tokens_to_generate - 1):
        jitter = (hash(str(time.perf_counter())) % 20 - 10) / 10.0
        itl_ms = max(2.0, base_tpot_ms + jitter)
        itls.append(itl_ms / 1000.0)

    decode_duration = sum(itls)
    total_duration = ttft_sec + decode_duration
    tpot = (decode_duration / max(1, tokens_to_generate - 1)) if tokens_to_generate > 1 else 0.0

    return {
        "success": True,
        "ttft_ms": round(ttft_sec * 1000.0, 2),
        "tpot_ms": round(tpot * 1000.0, 2),
        "itls_ms": [round(x * 1000.0, 2) for x in itls],
        "itl_jitter_ms": round(statistics.stdev([x * 1000.0 for x in itls]) if len(itls) > 1 else 0.0, 2),
        "total_ms": round(total_duration * 1000.0, 2),
        "prompt_tokens": prompt_tokens,
        "tokens": tokens_to_generate,
        "throughput_tps": round(tokens_to_generate / max(0.001, total_duration), 2),
        "error": None,
    }


def send_streaming_request(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_sec: float = 120.0,
    mock: bool = False,
) -> Dict[str, Any]:
    """
    Sends an SSE streaming chat completion request and records granular latency markers:
    - Time To First Token (TTFT / prefill latency)
    - Inter-Token Latencies (ITL) for every received chunk
    - Time Per Output Token (TPOT / decode latency)
    - End-to-End request turnaround time
    """
    if mock:
        prompt_content = payload["messages"][0]["content"]
        max_tokens = payload.get("max_tokens", 128)
        return simulate_synthetic_stream(prompt_content, max_tokens)

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    token_timestamps: List[float] = []
    tokens_generated = 0
    prompt_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            for line in response:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: ") or decoded == "data: [DONE]":
                    continue
                chunk_str = decoded[6:]
                try:
                    chunk = json.loads(chunk_str)
                    # Check for usage metadata if available
                    usage = chunk.get("usage")
                    if usage and "prompt_tokens" in usage:
                        prompt_tokens = usage["prompt_tokens"]

                    choices = chunk.get("choices", [])
                    if choices and "delta" in choices[0]:
                        delta = choices[0]["delta"]
                        if "content" in delta and delta["content"]:
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                            token_timestamps.append(now)
                            tokens_generated += 1
                except json.JSONDecodeError:
                    continue

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        if tokens_generated == 0:
            return {
                "success": False,
                "error": "Stream closed without generating content tokens.",
                "ttft_ms": 0.0,
                "tpot_ms": 0.0,
                "itls_ms": [],
                "itl_jitter_ms": 0.0,
                "total_ms": round(total_duration * 1000.0, 2),
                "prompt_tokens": prompt_tokens,
                "tokens": 0,
                "throughput_tps": 0.0,
            }

        ttft = (first_token_time - start_time) if first_token_time else total_duration
        decode_duration = (end_time - first_token_time) if first_token_time else total_duration
        tpot = (decode_duration / max(1, tokens_generated - 1)) if tokens_generated > 1 else 0.0

        itls_ms: List[float] = []
        for i in range(1, len(token_timestamps)):
            itls_ms.append(round((token_timestamps[i] - token_timestamps[i - 1]) * 1000.0, 2))

        itl_jitter = statistics.stdev(itls_ms) if len(itls_ms) > 1 else 0.0

        if prompt_tokens == 0:
            prompt_content = payload["messages"][0]["content"]
            prompt_tokens = max(10, len(prompt_content.split()) * 4 // 3)

        return {
            "success": True,
            "ttft_ms": round(ttft * 1000.0, 2),
            "tpot_ms": round(tpot * 1000.0, 2),
            "itls_ms": itls_ms,
            "itl_jitter_ms": round(itl_jitter, 2),
            "total_ms": round(total_duration * 1000.0, 2),
            "prompt_tokens": prompt_tokens,
            "tokens": tokens_generated,
            "throughput_tps": round(tokens_generated / max(0.001, total_duration), 2),
            "error": None,
        }

    except urllib.error.HTTPError as e:
        err_msg = f"HTTP {e.code}: {e.reason}"
        try:
            body = e.read().decode("utf-8")
            err_msg += f" - {body[:200]}"
        except Exception:
            pass
        return {
            "success": False,
            "error": err_msg,
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
            "itls_ms": [],
            "itl_jitter_ms": 0.0,
            "total_ms": 0.0,
            "prompt_tokens": 0,
            "tokens": 0,
            "throughput_tps": 0.0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
            "itls_ms": [],
            "itl_jitter_ms": 0.0,
            "total_ms": 0.0,
            "prompt_tokens": 0,
            "tokens": 0,
            "throughput_tps": 0.0,
        }


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculates comprehensive percentile distribution for latency values."""
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    vals = sorted(values)
    n = len(vals)

    def p(pct: float) -> float:
        idx = int(n * pct)
        return vals[min(idx, n - 1)]

    return {
        "min": round(vals[0], 2),
        "mean": round(statistics.mean(vals), 2),
        "p50": round(p(0.50), 2),
        "p90": round(p(0.90), 2),
        "p95": round(p(0.95), 2),
        "p99": round(p(0.99), 2),
        "max": round(vals[-1], 2),
    }


def execute_concurrency_tier(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    concurrency: int,
    num_requests: int,
    mock: bool = False,
) -> Dict[str, Any]:
    """Runs a single load tier with specified concurrency and request count."""
    start_tier = time.perf_counter()
    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_streaming_request, url, headers, payload, 120.0, mock)
            for _ in range(num_requests)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    wall_time_sec = max(0.001, time.perf_counter() - start_tier)
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        return {
            "success": False,
            "concurrency": concurrency,
            "total_requests": num_requests,
            "successful_requests": 0,
            "failed_requests": len(failed),
            "errors": [r.get("error") for r in failed[:3]],
        }

    ttfts = [r["ttft_ms"] for r in successful]
    tpots = [r["tpot_ms"] for r in successful]
    e2es = [r["total_ms"] for r in successful]
    all_itls = [itl for r in successful for itl in r.get("itls_ms", [])]

    total_output_tokens = sum(r["tokens"] for r in successful)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in successful)

    generation_throughput_tps = round(total_output_tokens / wall_time_sec, 2)
    prompt_throughput_tps = round(total_prompt_tokens / max(0.001, sum(ttfts) / 1000.0), 2)
    request_throughput_rps = round(len(successful) / wall_time_sec, 2)

    return {
        "success": True,
        "concurrency": concurrency,
        "total_requests": num_requests,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "wall_time_seconds": round(wall_time_sec, 2),
        "total_output_tokens": total_output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "generation_throughput_tokens_per_sec": generation_throughput_tps,
        "prompt_throughput_tokens_per_sec": prompt_throughput_tps,
        "request_throughput_rps": request_throughput_rps,
        "ttft_ms": compute_percentiles(ttfts),
        "tpot_ms": compute_percentiles(tpots),
        "e2e_latency_ms": compute_percentiles(e2es),
        "itl_jitter_ms": round(statistics.stdev(all_itls), 2) if len(all_itls) > 1 else 0.0,
    }


def run_benchmark_suite(
    target_url: str,
    model: str,
    workload: str = "chat",
    custom_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    concurrency: int = 4,
    num_requests: int = 12,
    sweep: bool = False,
    sweep_concurrencies: Optional[List[int]] = None,
    auth_token: Optional[str] = None,
    mock: bool = False,
    deployment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes complete benchmark suite across target endpoint.
    Supports single concurrency level or automated concurrency saturation sweep.
    """
    # Select prompt and max tokens
    if workload in WORKLOAD_PROFILES and not custom_prompt:
        prof = WORKLOAD_PROFILES[workload]
        prompt = prof["prompt"]
        tokens = max_tokens or prof["max_tokens"]
    else:
        prompt = custom_prompt or WORKLOAD_PROFILES["chat"]["prompt"]
        tokens = max_tokens or 150

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "fulcrum-benchmark/1.0",
    }
    token = auth_token if auth_token else (get_auth_token(target_url) if not mock else None)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Build OpenAI-compatible chat completion payload with SSE streaming
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": 0.7,
        "stream": True,
    }

    # Pre-benchmark Prometheus metric capture (if not in mock mode)
    pre_metrics = query_prometheus_metrics(target_url) if not mock else None

    benchmark_start_iso = datetime.now(timezone.utc).isoformat()
    tier_results: List[Dict[str, Any]] = []

    if sweep:
        concurrencies = sweep_concurrencies or [1, 2, 4, 8, 16]
        for c in concurrencies:
            tier_reqs = max(c * 2, 8)
            tier_res = execute_concurrency_tier(target_url, headers, payload, c, tier_reqs, mock)
            tier_results.append(tier_res)
    else:
        tier_res = execute_concurrency_tier(target_url, headers, payload, concurrency, num_requests, mock)
        tier_results.append(tier_res)

    # Post-benchmark Prometheus metric capture (if not in mock mode)
    post_metrics = query_prometheus_metrics(target_url) if not mock else None

    # Aggregate evaluation and saturation analysis
    primary_tier = tier_results[-1] if not sweep else next((t for t in tier_results if t.get("concurrency") == concurrency), tier_results[0])

    saturation_analysis = None
    if sweep and len(tier_results) > 1:
        # Detect saturation knee (concurrency level yielding maximum throughput)
        valid_tiers = [t for t in tier_results if t.get("success")]
        if valid_tiers:
            peak_tier = max(valid_tiers, key=lambda x: x["generation_throughput_tokens_per_sec"])
            baseline_tier = valid_tiers[0]
            scaling_factor = round(
                peak_tier["generation_throughput_tokens_per_sec"]
                / max(0.01, baseline_tier["generation_throughput_tokens_per_sec"]),
                2,
            )
            ttft_inflation = round(
                (peak_tier["ttft_ms"]["p50"] - baseline_tier["ttft_ms"]["p50"])
                / max(0.01, baseline_tier["ttft_ms"]["p50"])
                * 100.0,
                1,
            )
            saturation_analysis = {
                "optimal_concurrency": peak_tier["concurrency"],
                "peak_throughput_tps": peak_tier["generation_throughput_tokens_per_sec"],
                "baseline_concurrency": baseline_tier["concurrency"],
                "baseline_throughput_tps": baseline_tier["generation_throughput_tokens_per_sec"],
                "throughput_scaling_factor": scaling_factor,
                "ttft_inflation_pct_at_peak": ttft_inflation,
            }

    report = {
        "benchmark_id": f"fulcrum-bench-{int(time.time())}",
        "timestamp": benchmark_start_iso,
        "target_url": target_url,
        "model": model,
        "workload_profile": workload,
        "max_tokens": tokens,
        "sweep_mode": sweep,
        "primary_concurrency": concurrency,
        "results": tier_results if sweep else tier_results[0],
        "saturation_analysis": saturation_analysis,
        "metrics_telemetry": {
            "kv_cache_usage_pre": pre_metrics.get("vllm:gpu_cache_usage_factor") if pre_metrics else None,
            "kv_cache_usage_post": post_metrics.get("vllm:gpu_cache_usage_factor") if post_metrics else None,
            "requests_waiting_post": post_metrics.get("vllm:num_requests_waiting") if post_metrics else None,
        },
    }

    # Save to state database if deployment ID is known
    if deployment_id:
        try:
            bench_summary = {
                "last_run": benchmark_start_iso,
                "throughput_tps": primary_tier.get("generation_throughput_tokens_per_sec", 0.0),
                "ttft_p50_ms": primary_tier.get("ttft_ms", {}).get("p50", 0.0),
                "tpot_p50_ms": primary_tier.get("tpot_ms", {}).get("p50", 0.0),
                "workload": workload,
                "concurrency": concurrency,
            }
            db.update_deployment(deployment_id, {"last_benchmark": bench_summary})
        except Exception:
            pass

    return report


def render_ascii_report(report: Dict[str, Any]) -> str:
    """Formats benchmark results into an executive terminal ASCII dashboard."""
    lines: List[str] = []
    lines.append("=" * 84)
    lines.append(f"  /fulcrum benchmark: Open-Weight Model Serving Diagnostics ({report['model']})")
    lines.append("=" * 84)
    lines.append(f"Target URL:        {report['target_url']}")
    lines.append(f"Workload Profile:  {report['workload_profile'].upper()} (max_tokens: {report['max_tokens']})")
    lines.append(f"Benchmark Timestamp:{report['timestamp']}")

    sweep_mode = report.get("sweep_mode", False)

    if not sweep_mode:
        tier = report["results"]
        if not tier.get("success"):
            lines.append("-" * 84)
            lines.append("Benchmark Execution Failed!")
            for err in tier.get("errors", []):
                lines.append(f"  Error: {err}")
            lines.append("=" * 84)
            return "\n".join(lines)

        lines.append(f"Concurrency Load:  {tier['concurrency']} concurrent streams ({tier['total_requests']} requests)")
        lines.append(f"Total Completed:   {tier['successful_requests']} / {tier['total_requests']} in {tier['wall_time_seconds']}s")
        lines.append("-" * 84)
        lines.append("KEY OPEN-WEIGHT INFERENCE METRICS:")
        lines.append(f"  * Generation Throughput:  {tier['generation_throughput_tokens_per_sec']:>8.2f} tokens/sec  (Tokens: {tier['total_output_tokens']})")
        lines.append(f"  * Prompt/Prefill Rate:    {tier['prompt_throughput_tokens_per_sec']:>8.2f} tokens/sec  (Tokens: {tier['total_prompt_tokens']})")
        lines.append(f"  * Request Throughput:     {tier['request_throughput_rps']:>8.2f} reqs/sec")
        lines.append(f"  * Inter-Token Jitter:     {tier['itl_jitter_ms']:>8.2f} ms std-dev")
        lines.append("-" * 84)
        lines.append("LATENCY PERCENTILE DISTRIBUTION (milliseconds):")
        lines.append(f"  METRIC                    MIN       P50       P90       P95       P99       MAX")
        lines.append(f"  --------------------------------------------------------------------------------")
        ttft = tier["ttft_ms"]
        tpot = tier["tpot_ms"]
        e2e = tier["e2e_latency_ms"]
        lines.append(f"  TTFT (Prefill Latency)  {ttft['min']:>7.1f} {ttft['p50']:>9.1f} {ttft['p90']:>9.1f} {ttft['p95']:>9.1f} {ttft['p99']:>9.1f} {ttft['max']:>9.1f}")
        lines.append(f"  TPOT (Decode Latency)   {tpot['min']:>7.1f} {tpot['p50']:>9.1f} {tpot['p90']:>9.1f} {tpot['p95']:>9.1f} {tpot['p99']:>9.1f} {tpot['max']:>9.1f}")
        lines.append(f"  E2E Request Latency     {e2e['min']:>7.1f} {e2e['p50']:>9.1f} {e2e['p90']:>9.1f} {e2e['p95']:>9.1f} {e2e['p99']:>9.1f} {e2e['max']:>9.1f}")

    else:
        # Render multi-concurrency sweep table
        lines.append("-" * 84)
        lines.append("CONCURRENCY SATURATION SWEEP RESULTS:")
        lines.append(f"  CONC | REQS | OUTPUT TPS | PROMPT TPS | RPS   | TTFT P50  | TPOT P50  | E2E P50")
        lines.append(f"  -----+------+------------+------------+-------+-----------+-----------+---------")
        for tier in report["results"]:
            if tier.get("success"):
                c = tier["concurrency"]
                r = tier["successful_requests"]
                out_tps = tier["generation_throughput_tokens_per_sec"]
                p_tps = tier["prompt_throughput_tokens_per_sec"]
                rps = tier["request_throughput_rps"]
                ttft_p50 = tier["ttft_ms"]["p50"]
                tpot_p50 = tier["tpot_ms"]["p50"]
                e2e_p50 = tier["e2e_latency_ms"]["p50"]
                lines.append(f"  {c:>4} | {r:>4} | {out_tps:>10.1f} | {p_tps:>10.1f} | {rps:>5.1f} | {ttft_p50:>7.1f}ms | {tpot_p50:>7.1f}ms | {e2e_p50:>7.1f}ms")

        sat = report.get("saturation_analysis")
        if sat:
            lines.append("-" * 84)
            lines.append("SATURATION ANALYSIS:")
            lines.append(
                f"  * Optimal Concurrency Knee: {sat['optimal_concurrency']} streams "
                f"yielding {sat['peak_throughput_tps']} tokens/sec ({sat['throughput_scaling_factor']}x scaling over single-stream)."
            )
            lines.append(
                f"  * Prefill Queuing Penalty: TTFT inflated by +{sat['ttft_inflation_pct_at_peak']}% at peak saturation."
            )

    telem = report.get("metrics_telemetry", {})
    if telem.get("kv_cache_usage_post") is not None:
        lines.append("-" * 84)
        lines.append("VLLM ENGINE TELEMETRY:")
        lines.append(f"  * KV Cache Usage Factor: {telem['kv_cache_usage_post'] * 100:.1f}%")
        lines.append(f"  * Requests Waiting:      {telem.get('requests_waiting_post', 0)}")

    lines.append("=" * 84)
    return "\n".join(lines)


def render_markdown_report(report: Dict[str, Any]) -> str:
    """Renders benchmark findings as GitHub-flavoured Markdown document."""
    md: List[str] = []
    md.append(f"# Benchmark Report: `{report['model']}`")
    md.append("")
    md.append(f"- **Benchmark ID**: `{report['benchmark_id']}`")
    md.append(f"- **Target URL**: `{report['target_url']}`")
    md.append(f"- **Workload Profile**: `{report['workload_profile']}`")
    md.append(f"- **Execution Timestamp**: {report['timestamp']}")
    md.append("")

    if not report.get("sweep_mode", False):
        tier = report["results"]
        md.append("## Executive Metrics")
        md.append("")
        md.append("| Metric | Measurement | Description |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| **Generation Throughput** | **{tier['generation_throughput_tokens_per_sec']} tokens/sec** | Aggregate output tokens generated per wall-clock second |")
        md.append(f"| **Prompt / Prefill Rate** | **{tier['prompt_throughput_tokens_per_sec']} tokens/sec** | Processing rate for prompt token ingestion |")
        md.append(f"| **Request Throughput** | **{tier['request_throughput_rps']} RPS** | Requests fully served per second |")
        md.append(f"| **Inter-Token Jitter** | **{tier['itl_jitter_ms']} ms** | Standard deviation of per-token arrival delays |")
        md.append("")
        md.append("## Latency Distribution (ms)")
        md.append("")
        md.append("| Latency Stage | Min | P50 | P90 | P95 | P99 | Max | Mean |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        ttft = tier["ttft_ms"]
        tpot = tier["tpot_ms"]
        e2e = tier["e2e_latency_ms"]
        md.append(f"| **TTFT (Prefill)** | {ttft['min']} | {ttft['p50']} | {ttft['p90']} | {ttft['p95']} | {ttft['p99']} | {ttft['max']} | {ttft['mean']} |")
        md.append(f"| **TPOT (Decode)** | {tpot['min']} | {tpot['p50']} | {tpot['p90']} | {tpot['p95']} | {tpot['p99']} | {tpot['max']} | {tpot['mean']} |")
        md.append(f"| **E2E Latency** | {e2e['min']} | {e2e['p50']} | {e2e['p90']} | {e2e['p95']} | {e2e['p99']} | {e2e['max']} | {e2e['mean']} |")
    else:
        md.append("## Concurrency Saturation Sweep")
        md.append("")
        md.append("| Concurrency | Success Reqs | Output TPS | Prompt TPS | RPS | TTFT P50 | TPOT P50 | E2E P50 |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for tier in report["results"]:
            if tier.get("success"):
                md.append(
                    f"| {tier['concurrency']} | {tier['successful_requests']} | {tier['generation_throughput_tokens_per_sec']} | "
                    f"{tier['prompt_throughput_tokens_per_sec']} | {tier['request_throughput_rps']} | "
                    f"{tier['ttft_ms']['p50']} ms | {tier['tpot_ms']['p50']} ms | {tier['e2e_latency_ms']['p50']} ms |"
                )

    sat = report.get("saturation_analysis")
    if sat:
        md.append("")
        md.append("## Saturation Interpretation")
        md.append(
            f"- **Optimal Concurrency**: The deployment achieves peak throughput at **{sat['optimal_concurrency']} concurrent streams** "
            f"({sat['peak_throughput_tps']} tokens/sec), representing a **{sat['throughput_scaling_factor']}x** scaling efficiency."
        )
        md.append(
            f"- **Queue Inflation**: Under peak load, median TTFT inflated by **{sat['ttft_inflation_pct_at_peak']}%**, indicating "
            "KV cache queuing threshold."
        )

    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fulcrum benchmark: Measure TTFT, TPOT, ITL jitter, throughput, and concurrency scaling for open-weight models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark deployment tracked in state database by ID
  python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-qwen3-8-27b

  # Benchmark a direct Vertex AI Endpoint with chat workload
  python3 fulcrum/scripts/benchmark_deployment.py --endpoint https://us-central1-aiplatform.googleapis.com/.../endpoints/123:rawPredict --model Qwen/Qwen3.8-27B

  # Run a concurrency saturation sweep (1, 2, 4, 8, 16 streams)
  python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-gemma-3-12b-it --sweep

  # Test prefill performance on a RAG workload
  python3 fulcrum/scripts/benchmark_deployment.py --id fulcrum-gemma-3-12b-it --workload rag --concurrency 4

  # Run offline synthetic verification without an active server
  python3 fulcrum/scripts/benchmark_deployment.py --model google/gemma-3-12b-it --endpoint http://mock --mock
""",
    )

    parser.add_argument("--id", "-i", type=str, help="Deployment ID from local deployments.json registry")
    parser.add_argument("--endpoint", "-e", type=str, help="Target API endpoint URL or host:port")
    parser.add_argument("--model", "-m", type=str, help="Model name identifier (e.g. google/gemma-3-12b-it)")
    parser.add_argument(
        "--workload",
        "-w",
        choices=["chat", "rag", "code", "reasoning", "custom"],
        default="chat",
        help="Synthetic workload profile (default: chat)",
    )
    parser.add_argument("--prompt", type=str, help="Custom prompt text for benchmark evaluation")
    parser.add_argument("--max-tokens", type=int, help="Maximum generated tokens per stream")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Concurrent client streams (default: 4)")
    parser.add_argument("--requests", "-n", type=int, default=12, help="Total requests to dispatch (default: 12)")
    parser.add_argument("--sweep", action="store_true", help="Perform automated multi-concurrency saturation sweep")
    parser.add_argument("--token", type=str, help="Bearer token (auto-obtained via gcloud for Vertex endpoints)")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text", help="Output format")
    parser.add_argument("--report-file", type=str, help="Save Markdown benchmark report to file path")
    parser.add_argument("--mock", action="store_true", help="Simulate synthetic inference responses for offline testing")

    args = parser.parse_args()

    try:
        target_url, resolved_model, dep_id, _ = resolve_endpoint_target(
            identifier=args.id,
            endpoint=args.endpoint,
            model=args.model,
        )
    except Exception as e:
        if args.mock and args.model:
            target_url = "http://mock-service/v1/chat/completions"
            resolved_model = args.model
            dep_id = None
        else:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    report = run_benchmark_suite(
        target_url=target_url,
        model=resolved_model,
        workload=args.workload,
        custom_prompt=args.prompt,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        num_requests=args.requests,
        sweep=args.sweep,
        auth_token=args.token,
        mock=args.mock,
        deployment_id=dep_id,
    )

    if args.report_file:
        md_content = render_markdown_report(report)
        with open(args.report_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Benchmark report saved to: {args.report_file}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(report, indent=2))
    elif args.format == "markdown":
        print(render_markdown_report(report))
    else:
        print(render_ascii_report(report))


if __name__ == "__main__":
    main()
