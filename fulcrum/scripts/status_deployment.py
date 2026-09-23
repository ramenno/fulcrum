#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
fulcrum status: Deployment monitor and utilisation inspector.
Checks status of deployments on Vertex AI Endpoints (predict state, deployed models)
and Compute Engine (VM state, vLLM /health, and Prometheus /metrics).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import local state database manager
sys.path.insert(0, str(Path(__file__).parent))
try:
    import db
except ImportError:
    from fulcrum.scripts import db


def query_vertex_status(endpoint_name: str, region: str, project: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Queries Vertex AI endpoint via gcloud to check status and retrieve deployed model info."""
    try:
        proc = subprocess.run(
            [
                "gcloud",
                "ai",
                "endpoints",
                "describe",
                endpoint_name,
                f"--region={region}",
                f"--project={project}",
                "--format=json",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if proc.returncode == 0:
            info = json.loads(proc.stdout)
            deployed = info.get("deployedModels", [])
            state = "ACTIVE" if deployed else "NO_MODELS"
            deployed_id = deployed[0].get("id") if deployed else None
            uri = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{endpoint_name}:predict"
            return state, deployed_id, uri
    except Exception:
        pass
    return "UNKNOWN", None, None


def query_vertex_operation(operation_id: str, region: str, project: str) -> Tuple[str, bool, Optional[str]]:
    """Queries Vertex AI operation via gcloud ai operations describe to check completion status."""
    op_num = operation_id.split("/")[-1]
    try:
        proc = subprocess.run(
            [
                "gcloud",
                "ai",
                "operations",
                "describe",
                op_num,
                f"--region={region}",
                f"--project={project}",
                "--format=json",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            is_done = data.get("done", False)
            if is_done:
                err = data.get("error")
                if err:
                    return "FAILED", True, err.get("message")
                return "SUCCEEDED", True, None
            meta = data.get("metadata", {})
            state = meta.get("genericMetadata", {}).get("state", "RUNNING")
            return state, False, None
    except Exception:
        pass
    return "UNKNOWN", False, None


def query_gce_status(instance_name: str, zone: str, project: str) -> Tuple[str, Optional[str]]:
    """Queries GCP Compute Engine via gcloud to retrieve live VM status and external IP."""
    try:
        proc = subprocess.run(
            [
                "gcloud",
                "compute",
                "instances",
                "describe",
                instance_name,
                f"--zone={zone}",
                f"--project={project}",
                "--format=json",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if proc.returncode == 0:
            info = json.loads(proc.stdout)
            status = info.get("status", "UNKNOWN")
            ip = None
            interfaces = info.get("networkInterfaces", [])
            if interfaces:
                access_configs = interfaces[0].get("accessConfigs", [])
                if access_configs:
                    ip = access_configs[0].get("natIP")
            return status, ip
    except Exception:
        pass
    return "UNKNOWN", None


def query_vllm_metrics(ip: str, port: int) -> Dict[str, Any]:
    """Scrapes health and Prometheus metrics from the running vLLM service on GCE."""
    base_url = f"http://{ip}:{port}"
    res: Dict[str, Any] = {
        "health": "UNREACHABLE",
        "kv_cache_usage_pct": None,
        "requests_running": 0,
        "requests_waiting": 0,
        "active_models": [],
    }

    # 1. Health check
    try:
        req = urllib.request.Request(f"{base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                res["health"] = "HEALTHY"
    except Exception:
        return res

    # 2. Query loaded models
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                models_data = json.loads(response.read().decode("utf-8"))
                res["active_models"] = [m.get("id") for m in models_data.get("data", [])]
    except Exception:
        pass

    # 3. Query Prometheus metrics for KV cache and request concurrency
    try:
        req = urllib.request.Request(f"{base_url}/metrics", method="GET")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                content = response.read().decode("utf-8")
                cache_match = re.search(r"vllm:gpu_cache_usage_factor(?:\{[^\}]*\})?\s+([\d\.]+)", content)
                if cache_match:
                    res["kv_cache_usage_pct"] = round(float(cache_match.group(1)) * 100, 1)

                running_match = re.search(r"vllm:num_requests_running(?:\{[^\}]*\})?\s+([\d\.]+)", content)
                if running_match:
                    res["requests_running"] = int(float(running_match.group(1)))

                waiting_match = re.search(r"vllm:num_requests_waiting(?:\{[^\}]*\})?\s+([\d\.]+)", content)
                if waiting_match:
                    res["requests_waiting"] = int(float(waiting_match.group(1)))
    except Exception:
        pass

    return res


def inspect_deployment(deployment: Dict[str, Any]) -> Dict[str, Any]:
    """Inspects live state for a single deployment and updates record if changed."""
    platform = deployment.get("platform", "gce")
    name = deployment.get("endpoint_id") or deployment.get("endpoint_numeric_id") or deployment.get("endpoint_name") or deployment.get("instance_name") or deployment.get("id")
    record_id = deployment.get("id") or deployment.get("endpoint_name") or name
    project = deployment.get("project") or db.detect_gcp_project() or ""
    region = deployment.get("region", "us-central1")
    zone = deployment.get("zone", "us-central1-a")
    port = deployment.get("port", 8000)

    # 1. Vertex AI Endpoint Inspection
    if platform == "vertex":
        vertex_state, deployed_id, predict_uri = query_vertex_status(name, region, project)
        operation_id = deployment.get("operation_id")
        op_state = None
        op_done = False
        op_err = None

        if operation_id and vertex_state != "ACTIVE":
            op_state, op_done, op_err = query_vertex_operation(operation_id, region, project)
            if not op_done and op_state in ("RUNNING", "PENDING"):
                vertex_state = "PROVISIONING"
            elif op_done and op_state == "FAILED":
                vertex_state = "FAILED"

        live_status = vertex_state if vertex_state != "UNKNOWN" else deployment.get("status", "RECORDED")
        updates: Dict[str, Any] = {}
        if deployed_id and deployed_id != deployment.get("deployed_model_id"):
            updates["deployed_model_id"] = deployed_id
        if live_status != deployment.get("status"):
            updates["status"] = live_status
        if updates:
            db.update_deployment(record_id, updates)

        return {
            **deployment,
            "platform": "vertex",
            "live_status": live_status,
            "operation_id": operation_id,
            "operation_state": op_state,
            "operation_error": op_err,
            "deployed_model_id": deployed_id or deployment.get("deployed_model_id"),
            "predict_uri": predict_uri or deployment.get("predict_uri"),
            "vllm_metrics": {
                "health": "MANAGED" if live_status == "ACTIVE" else ("PROVISIONING" if live_status == "PROVISIONING" else "INITIALIZING"),
                "kv_cache_usage_pct": None,
                "requests_running": 0,
                "requests_waiting": 0,
            },
        }

    # 2. Compute Engine Inspection
    gce_status, live_ip = query_gce_status(name, zone, project)
    ip = live_ip or deployment.get("ip_address")

    metrics = {
        "health": "UNKNOWN",
        "kv_cache_usage_pct": None,
        "requests_running": 0,
        "requests_waiting": 0,
    }

    if ip and gce_status == "RUNNING":
        metrics = query_vllm_metrics(ip, port)

    updates: Dict[str, Any] = {}
    if gce_status != "UNKNOWN" and gce_status != deployment.get("status"):
        updates["status"] = gce_status
    if live_ip and live_ip != deployment.get("ip_address"):
        updates["ip_address"] = live_ip

    if updates:
        db.update_deployment(name, updates)

    return {
        **deployment,
        "platform": platform,
        "live_status": gce_status if gce_status != "UNKNOWN" else deployment.get("status", "RECORDED"),
        "live_ip": ip,
        "vllm_metrics": metrics,
    }


def format_table(inspections: List[Dict[str, Any]]) -> str:
    """Renders deployments and live utilisation metrics in an ASCII table."""
    if not inspections:
        db_path = db.get_db_path()
        return f"No tracked deployments found in local database ({db_path}).\nDeploy models using '/fulcrum run' to track them here."

    lines = []
    header = f"{'RESOURCE NAME':<22} | {'PLATFORM':<8} | {'MODEL':<24} | {'STATUS':<11} | {'HEALTH':<10} | {'KV CACHE':<8} | {'BENCHMARK':<17} | {'ENDPOINT / URI':<28}"
    separator = "-" * len(header)
    lines.append(separator)
    lines.append(header)
    lines.append(separator)

    for item in inspections:
        name = (item.get("endpoint_name") or item.get("instance_name") or item.get("id", "unknown"))[:22]
        plat = item.get("platform", "gce").upper()[:8]
        model = item.get("model", "unknown").split("/")[-1][:24]
        status = item.get("live_status", "UNKNOWN")[:11]
        vmetrics = item.get("vllm_metrics", {})
        health = vmetrics.get("health", "UNKNOWN")[:10]

        cache_pct = vmetrics.get("kv_cache_usage_pct")
        cache_str = f"{cache_pct}%" if cache_pct is not None else "N/A"

        bench = item.get("last_benchmark")
        if bench:
            bench_str = f"{bench.get('throughput_tps', 0):.1f} tps"[:17]
        else:
            bench_str = "Not run"

        if plat == "VERTEX":
            reg = item.get("region", "us-central1")
            endpoint_str = f"Vertex AI ({reg})"[:28]
        else:
            ip = item.get("live_ip") or item.get("ip_address")
            port = item.get("port", 8000)
            endpoint_str = f"{ip}:{port}" if ip else "Pending IP"
            endpoint_str = endpoint_str[:28]

        lines.append(f"{name:<22} | {plat:<8} | {model:<24} | {status:<11} | {health:<10} | {cache_str:<8} | {bench_str:<17} | {endpoint_str:<28}")

    lines.append(separator)
    return "\n".join(lines)


def print_verbose_details(inspections: List[Dict[str, Any]]) -> None:
    """Prints in-depth diagnostic details for each inspected deployment."""
    for item in inspections:
        plat = item.get("platform", "gce")
        name = item.get("endpoint_name") or item.get("instance_name") or item.get("id", "unknown")
        vmetrics = item.get("vllm_metrics", {})

        print(f"\n--- Deployment Diagnostics: {name} ({plat.upper()}) ---")
        print(f"  Model ID:          {item.get('model')}")
        print(f"  Target Platform:   {plat.upper()}")
        print(f"  GCP Project:       {item.get('project')}")
        print(f"  Hardware:          {item.get('gpu_count', 1)}x {item.get('gpu_type', 'nvidia-l4')} ({item.get('machine_type')})")
        print(f"  Status:            {item.get('live_status')}")
        print(f"  Health:            {vmetrics.get('health')}")

        bench = item.get("last_benchmark")
        if bench:
            print(
                f"  Last Benchmark:    {bench.get('throughput_tps', 0):.1f} tps "
                f"(TTFT P50: {bench.get('ttft_p50_ms', 0):.1f}ms, TPOT P50: {bench.get('tpot_p50_ms', 0):.1f}ms) "
                f"[{bench.get('workload', 'chat')}, c={bench.get('concurrency', 1)}] - {bench.get('last_run')}"
            )

        if plat == "vertex":
            reg = item.get("region", "us-central1")
            proj = item.get("project", "PROJECT_ID")
            uri = item.get("predict_uri") or f"https://{reg}-aiplatform.googleapis.com/v1/projects/{proj}/locations/{reg}/endpoints/{name}:predict"
            if item.get("operation_id"):
                print(f"  Operation ID:      {item.get('operation_id')}")
            if item.get("operation_state"):
                print(f"  Operation State:   {item.get('operation_state')}")
            if item.get("operation_error"):
                print(f"  Operation Error:   {item.get('operation_error')}")
            print(f"  Vertex Predict URI: {uri}")
            print(f"  Curl Test (IAM OAuth2):")
            print(f"    curl -X POST \\")
            print(f"      -H 'Authorization: Bearer $(gcloud auth print-access-token)' \\")
            print(f"      -H 'Content-Type: application/json' \\")
            print(f"      '{uri}' \\")
            print(f"      -d '{{\"instances\": [{{\"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}]}}'")
        else:
            ip = item.get("live_ip") or item.get("ip_address")
            port = item.get("port", 8000)
            if vmetrics.get("kv_cache_usage_pct") is not None:
                print(f"  KV Cache Load:     {vmetrics.get('kv_cache_usage_pct')}% utilised")
            print(f"  Active Streams:    {vmetrics.get('requests_running', 0)} running, {vmetrics.get('requests_waiting', 0)} queued")
            if ip:
                print(f"  Chat Endpoint:     http://{ip}:{port}/v1/chat/completions")
                print(f"  Health Endpoint:   http://{ip}:{port}/health")
                print(f"  Metrics Endpoint:  http://{ip}:{port}/metrics")
                print(f"  Curl Test:")
                print(f"    curl http://{ip}:{port}/v1/chat/completions \\")
                print(f"      -H 'Content-Type: application/json' \\")
                print(f"      -d '{{\"model\": \"{item.get('model')}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fulcrum status: Inspect current status and live utilisation metrics of deployments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status and utilisation of all deployed models
  python3 fulcrum/scripts/status_deployment.py

  # Filter by specific deployment ID with detailed diagnostics
  python3 fulcrum/scripts/status_deployment.py --id fulcrum-gemma-3-12b-it --verbose

  # Output as JSON
  python3 fulcrum/scripts/status_deployment.py --format json

  # Continuously poll metrics every 5 seconds
  python3 fulcrum/scripts/status_deployment.py --watch 5
""",
    )
    parser.add_argument("--id", "--instance-name", "-i", dest="identifier", type=str, help="Target specific deployment ID")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output display format (default: table)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Display full diagnostic information and curl endpoints")
    parser.add_argument("--watch", "-w", type=int, help="Poll and refresh status every N seconds")

    args = parser.parse_args()

    while True:
        deployments = db.load_deployments()
        if args.identifier:
            deployments = [
                d for d in deployments
                if d.get("id") == args.identifier
                or d.get("instance_name") == args.identifier
                or d.get("endpoint_name") == args.identifier
            ]

        inspections = [inspect_deployment(d) for d in deployments]

        if args.format == "json":
            print(json.dumps(inspections, indent=2))
        else:
            if args.watch:
                print("\033[H\033[J", end="")
                print(f"[fulcrum status] Live Monitor (polling every {args.watch}s - Press Ctrl+C to stop)")
            print(format_table(inspections))
            if args.verbose and inspections:
                print_verbose_details(inspections)

        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
