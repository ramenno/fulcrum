#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
fulcrum run: Automated GCP deployment orchestrator for open-weight models.
Prioritises Vertex AI Endpoints as the primary default whenever viable.
Falls back to Compute Engine (GCE) or Kubernetes Engine (GKE) only when
Vertex AI Endpoints are not viable or explicitly requested.
"""

import argparse
from datetime import datetime, timezone
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

try:
    import check_quota
except ImportError:
    from fulcrum.scripts import check_quota

try:
    import find_model
except ImportError:
    from fulcrum.scripts import find_model


# Accelerator mapping for Vertex AI
VERTEX_ACCELERATOR_MAP = {
    "nvidia-l4": "NVIDIA_L4",
    "nvidia-tesla-a100": "NVIDIA_TESLA_A100",
    "nvidia-a100-80gb": "NVIDIA_A100_80GB",
    "nvidia-h100-80gb": "NVIDIA_H100_80GB",
}

# Machine mappings for Vertex AI prediction
VERTEX_MACHINE_MAP = {
    ("nvidia-l4", 1): "g2-standard-8",
    ("nvidia-l4", 2): "g2-standard-24",
    ("nvidia-l4", 4): "g2-standard-48",
    ("nvidia-l4", 8): "g2-standard-96",
    ("nvidia-tesla-a100", 1): "a2-highgpu-1g",
    ("nvidia-tesla-a100", 2): "a2-highgpu-2g",
    ("nvidia-tesla-a100", 4): "a2-highgpu-4g",
    ("nvidia-tesla-a100", 8): "a2-highgpu-8g",
    ("nvidia-a100-80gb", 1): "a2-ultragpu-1g",
    ("nvidia-a100-80gb", 2): "a2-ultragpu-2g",
    ("nvidia-a100-80gb", 4): "a2-ultragpu-4g",
    ("nvidia-a100-80gb", 8): "a2-ultragpu-8g",
    ("nvidia-h100-80gb", 8): "a3-highgpu-8g",
}

# Default machine mappings for GCE GPU accelerators
GCE_MACHINE_MAP = {
    ("nvidia-l4", 1): "g2-standard-8",
    ("nvidia-l4", 2): "g2-standard-24",
    ("nvidia-l4", 4): "g2-standard-48",
    ("nvidia-l4", 8): "g2-standard-96",
    ("nvidia-tesla-a100", 1): "a2-highgpu-1g",
    ("nvidia-tesla-a100", 2): "a2-highgpu-2g",
    ("nvidia-tesla-a100", 4): "a2-highgpu-4g",
    ("nvidia-tesla-a100", 8): "a2-highgpu-8g",
    ("nvidia-a100-80gb", 1): "a2-ultragpu-1g",
    ("nvidia-a100-80gb", 2): "a2-ultragpu-2g",
    ("nvidia-a100-80gb", 4): "a2-ultragpu-4g",
    ("nvidia-a100-80gb", 8): "a2-ultragpu-8g",
    ("nvidia-h100-80gb", 1): "a3-highgpu-1g",
    ("nvidia-h100-80gb", 8): "a3-highgpu-8g",
}

# Rigid 1:1 hardware topology for Google Cloud G2 instances (NVIDIA L4)
G2_L4_TOPOLOGY = {
    1: "g2-standard-8",
    2: "g2-standard-24",
    4: "g2-standard-48",
    8: "g2-standard-96",
}

# Cloud TPU Machine Types on Vertex AI Prediction Endpoints
# Accelerators are hardwired into the machine definition (e.g. ct5lp-* for v5e, ct6e-* for v6e)
TPU_MACHINE_MAP = {
    ("v5e", 1): "ct5lp-hightpu-1t",
    ("v5e", 4): "ct5lp-hightpu-4t",
    ("v5e", 8): "ct5lp-hightpu-8t",
    ("tpu-v5e", 1): "ct5lp-hightpu-1t",
    ("tpu-v5e", 4): "ct5lp-hightpu-4t",
    ("tpu-v5e", 8): "ct5lp-hightpu-8t",
    ("v6e", 1): "ct6e-standard-1t",
    ("v6e", 4): "ct6e-standard-4t",
    ("v6e", 8): "ct6e-standard-8t",
    ("tpu-v6e", 1): "ct6e-standard-1t",
    ("tpu-v6e", 4): "ct6e-standard-4t",
    ("tpu-v6e", 8): "ct6e-standard-8t",
    ("v5p", 4): "ct5p-hightpu-4t",
    ("v5p", 8): "ct5p-hightpu-8t",
    ("tpu-v5p", 4): "ct5p-hightpu-4t",
    ("tpu-v5p", 8): "ct5p-hightpu-8t",
}

# Production vLLM-TPU container image for Vertex AI Endpoints
VERTEX_TPU_VLLM_CONTAINER = "vllm/vllm-tpu:latest"


def is_tpu_machine(machine_type: Optional[str]) -> bool:
    """Returns True if the machine type is a Vertex AI Cloud TPU machine type (e.g. ct5lp-*, ct6e-*, ct5p-*)."""
    if not machine_type:
        return False
    mt = machine_type.lower()
    return mt.startswith("ct5lp-") or mt.startswith("ct6e-") or mt.startswith("ct5p-") or mt.startswith("ct")


def resolve_machine_type(gpu_type: str, gpu_count: int, platform: str = "vertex") -> str:
    """
    Resolves the exact GCP machine type adhering to rigid hardware topologies.
    Guarantees G2 instances always pair strictly:
      1x L4 -> g2-standard-8
      2x L4 -> g2-standard-24 (g2-standard-16 is strictly rejected by GCP API)
      4x L4 -> g2-standard-48
      8x L4 -> g2-standard-96
    Supports Cloud TPU instances on Vertex AI:
      ct5lp-hightpu-1t/4t/8t (v5e)
      ct6e-standard-1t/4t/8t (v6e)
      ct5p-hightpu-4t/8t (v5p)
    """
    g_lower = gpu_type.lower()
    if "tpu" in g_lower or g_lower in ("v5e", "v6e", "v5p"):
        key = (g_lower, gpu_count)
        if key in TPU_MACHINE_MAP:
            return TPU_MACHINE_MAP[key]
        if "v6e" in g_lower:
            return f"ct6e-standard-{gpu_count}t"
        if "v5p" in g_lower:
            return f"ct5p-hightpu-{gpu_count}t"
        return f"ct5lp-hightpu-{gpu_count}t"

    if "l4" in g_lower:
        return G2_L4_TOPOLOGY.get(gpu_count, f"g2-standard-{gpu_count * 8}")

    mapping = VERTEX_MACHINE_MAP if platform == "vertex" else GCE_MACHINE_MAP
    if (g_lower, gpu_count) in mapping:
        return mapping[(g_lower, gpu_count)]

    if "a100" in g_lower:
        if "80" in g_lower:
            return f"a2-ultragpu-{gpu_count}g"
        return f"a2-highgpu-{gpu_count}g"
    if "h100" in g_lower:
        return "a3-highgpu-8g"

    return f"n1-standard-{gpu_count * 8}"


# Recommended official Google vLLM container on Vertex AI Model Garden (GPU)
VERTEX_VLLM_CONTAINER = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:latest"


def sanitize_name(name: str) -> str:
    """Sanitises model or instance names to conform to RFC 1035 labels for GCP."""
    cleaned = re.sub(r"[^a-z0-9-]", "-", name.lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned[:50]


def detect_gcp_project() -> Optional[str]:
    """Attempts to read active GCP project from gcloud configuration or environment."""
    return db.detect_gcp_project()


def resolve_gcp_project(explicit_project: Optional[str] = None) -> str:
    """
    Resolves the GCP project ID.
    Prioritises explicit CLI argument, then auto-detects from gcloud.
    Prompts the user only if no project is set in gcloud.
    """
    return db.resolve_gcp_project(explicit_project)


def estimate_gpu_needs(model_id: str) -> Tuple[str, int]:
    """Infers suitable GPU type and count from model parameter hints."""
    mid = model_id.lower()
    if "671b" in mid:
        # DeepSeek-R1 full MoE requires multi-node cluster (>8 GPUs)
        return "nvidia-h100-80gb", 16
    elif "70b" in mid or "72b" in mid:
        return "nvidia-a100-80gb", 1
    elif "27b" in mid or "32b" in mid or "24b" in mid:
        return "nvidia-l4", 4
    elif "12b" in mid or "14b" in mid:
        return "nvidia-l4", 1
    elif any(f"{x}b" in mid for x in [1, 2, 3, 4, 7, 8, 9]):
        return "nvidia-l4", 1
    else:
        return "nvidia-l4", 1


def evaluate_platform_viability(
    model_id: str,
    gpu_type: str,
    gpu_count: int,
    requested_platform: str = "auto",
) -> Tuple[str, str]:
    """
    Evaluates whether Vertex AI Endpoint is viable for the requested model and hardware.
    Returns (selected_platform, rationale).
    Priority: Vertex Endpoint -> Compute Engine (GCE) -> Kubernetes Engine (GKE).
    """
    req = requested_platform.lower()
    if req == "vertex":
        return "vertex", "Vertex AI Endpoint explicitly requested by user."
    if req == "gce":
        return "gce", "Compute Engine (GCE) explicitly requested by user."
    if req == "gke":
        return "gke", "Google Kubernetes Engine (GKE) explicitly requested by user."

    # Auto-detection mode: Vertex AI Endpoint is the primary choice whenever viable.
    # Check 1: Multi-node distributed inference (>8 GPUs)
    if gpu_count > 8 or "671b" in model_id.lower():
        return (
            "gke",
            f"Model {model_id} requires multi-node distributed serving ({gpu_count} GPUs), "
            "which exceeds Vertex AI single-node prediction limits. Falling back to GKE cluster.",
        )

    # Check 1b: Cloud TPU support on Vertex AI
    if "tpu" in gpu_type.lower() or gpu_type.lower() in ("v5e", "v6e", "v5p"):
        return (
            "vertex",
            f"Vertex AI Endpoint is fully viable for Cloud TPU ({gpu_type}) serving using vLLM-TPU container.",
        )

    # Check 2: Accelerator support on Vertex AI
    vertex_accel = VERTEX_ACCELERATOR_MAP.get(gpu_type.lower())
    if not vertex_accel:
        return (
            "gce",
            f"Accelerator '{gpu_type}' does not map to a standard Vertex AI accelerator. "
            "Falling back to Compute Engine (GCE).",
        )

    # Check 3: Machine type support
    if (gpu_type.lower(), gpu_count) not in VERTEX_MACHINE_MAP:
        return (
            "gce",
            f"Configuration {gpu_count}x {gpu_type} is not standard on Vertex AI Endpoints. "
            "Falling back to Compute Engine (GCE).",
        )

    # Default to Vertex AI Endpoint
    return (
        "vertex",
        "Vertex AI Endpoint is fully viable (single-node GPU envelope with official vLLM container) "
        "and selected as the primary deployment platform.",
    )


def extract_operation_id_from_output(text: str) -> Optional[str]:
    """
    Extracts Vertex AI Operation ID or full operation resource name from command output.
    Handles standard gcloud operation notifications, e.g.:
      - Operation [projects/123/locations/europe-west4/operations/1810106235958067200] has not finished in 1800 seconds
      - Operation [projects/123/locations/us-central1/operations/1810106235958067200] is in progress.
      - {"name": "projects/123/locations/europe-west4/operations/1810106235958067200"}
    """
    if not text:
        return None
    # Match standard Vertex AI operations/<id> pattern
    m = re.search(r"operations/([0-9]+)", text)
    if m:
        return m.group(1)
    # Match standalone 16-22 digit operation ID
    m2 = re.search(r"\b([0-9]{16,22})\b", text)
    if m2:
        return m2.group(1)
    return None


def dispatch_vertex_deploy_model_async(
    project: str,
    region: str,
    endpoint_id: str,
    model_id: str,
    display_name: str,
    machine_type: str,
    vertex_accel: Optional[str] = None,
    gpu_count: int = 1,
    is_tpu: bool = False,
    access_token: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Dispatches Vertex AI model deployment asynchronously via REST API.
    Bypasses gcloud's 1,800s client socket timeout by immediately returning the Operation ID.
    Returns: (operation_name, operation_id) or (None, None) on failure.
    """
    token = access_token
    if not token:
        try:
            p = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            if p.returncode == 0 and p.stdout.strip():
                token = p.stdout.strip()
        except Exception:
            token = None

    if not token:
        return None, None

    machine_spec: Dict[str, Any] = {"machineType": machine_type}
    if not is_tpu and vertex_accel:
        machine_spec["acceleratorType"] = vertex_accel.upper().replace("-", "_")
        machine_spec["acceleratorCount"] = int(gpu_count)

    payload = {
        "deployedModel": {
            "model": f"projects/{project}/locations/{region}/models/{model_id}",
            "displayName": display_name,
            "dedicatedResources": {
                "machineSpec": machine_spec,
                "minReplicaCount": 1,
                "maxReplicaCount": 1,
            },
        },
        "trafficSplit": {
            "0": 100,
        },
    }

    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{endpoint_id}:deployModel"
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            op_name = data.get("name", "")
            op_id = extract_operation_id_from_output(op_name)
            return op_name, op_id
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8") if e.fp else ""
        print(f"Notice: Async deployModel API returned HTTP {e.code}: {err_body}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"Notice: Async deployModel dispatch failed: {e}", file=sys.stderr)
        return None, None


def poll_vertex_operation(
    operation_id: str,
    project: str,
    region: str,
    timeout_seconds: int = 7200,
    interval_seconds: int = 15,
    logger_func=print,
    max_poll_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Polls a Vertex AI Long-Running Operation (LRO) until completion or timeout.
    Decoupled from gcloud's 1,800s client socket timeout, providing real-time stage telemetry.
    Raises RuntimeError on operation failure or TimeoutError on timeout.
    """
    start_time = time.time()
    op_num = operation_id.split("/")[-1]
    logger_func(f"🚀 [ASYNC DEPLOYMENT DISPATCHED]: Handed off monitoring to operation poller for Operation ID: {op_num}")
    logger_func(f"   Monitoring endpoint deployment (timeout: {timeout_seconds // 60}m, poll interval: {interval_seconds}s)...")

    iterations = 0
    while True:
        if max_poll_iterations is not None and iterations >= max_poll_iterations:
            break

        elapsed = int(time.time() - start_time)
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Vertex AI deployment operation {op_num} exceeded timeout limit of {timeout_seconds} seconds ({timeout_seconds // 60}m)."
            )

        mins = elapsed // 60
        secs = elapsed % 60
        time_str = f"{mins}m {secs:02d}s"

        cmd = [
            "gcloud", "ai", "operations", "describe", op_num,
            f"--project={project}", f"--region={region}",
            "--format=json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        op_data = {}
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                op_data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                op_data = {}

        if op_data:
            is_done = op_data.get("done", False)
            if is_done:
                err = op_data.get("error")
                if err:
                    err_msg = err.get("message", "Unknown Google Cloud error")
                    err_code = err.get("code", "UNKNOWN")
                    raise RuntimeError(
                        f"Vertex AI deployment operation {op_num} FAILED [Code {err_code}]: {err_msg}"
                    )
                logger_func(f"✅ [DEPLOYMENT COMPLETE]: Operation {op_num} finished successfully in {time_str}!")
                return op_data

            meta = op_data.get("metadata", {})
            state = meta.get("genericMetadata", {}).get("state", "RUNNING")
            logger_func(f"   ⏱️  [Elapsed: {time_str}] Operation {op_num} in progress... (State: {state})")
        else:
            logger_func(f"   ⏱️  [Elapsed: {time_str}] Operation {op_num} in progress... (polling Vertex AI API)")

        iterations += 1
        if max_poll_iterations is not None and iterations >= max_poll_iterations:
            return op_data

        time.sleep(interval_seconds)
    return {}


def generate_vertex_commands(
    endpoint_name: str,
    model_id: str,
    project: str,
    region: str,
    machine_type: str,
    vertex_accel: str,
    gpu_count: int,
    tensor_parallel: int,
    max_model_len: int,
    quantization: str,
    hf_token: Optional[str] = None,
    enforce_eager: bool = False,
    is_tpu: bool = False,
    gcs_model_path: Optional[str] = None,
    container_image: Optional[str] = None,
    gpu_memory_utilization: float = 0.90,
) -> Dict[str, str]:
    """Generates gcloud CLI commands for provisioning on Vertex AI Endpoints (supporting GPUs and Cloud TPUs)."""
    quant_flag = f",--quantization={quantization}" if quantization and quantization != "none" else ""
    
    if is_tpu:
        container_image = container_image or VERTEX_TPU_VLLM_CONTAINER
        shm_size = 32768
        timeout_sec = 7200
        env_vars = [
            "PJRT_DEVICE=TPU",
            "VLLM_TPU_ENABLE_PALLAS=1",
            "RAY_DEDUP_LOGS=0",
            "RAY_USAGE_STATS_ENABLED=0",
            "PYTHONUNBUFFERED=1",
        ]
        if gcs_model_path:
            env_vars.append(f"GCS_MODEL_PATH={gcs_model_path}")
        if hf_token:
            env_vars.append(f"HF_TOKEN={hf_token}")
        env_vars_str = ",".join(env_vars)
        token_env = f' \\\n    --container-env-vars="{env_vars_str}"'

        served_model = "/tmp/model" if gcs_model_path else model_id
        container_args = (
            f"--model={served_model},"
            f"--port=8080,"
            f"--tensor-parallel-size={tensor_parallel},"
            f"--max-model-len={max_model_len},"
            f"--gpu-memory-utilization={gpu_memory_utilization:.2f},"
            f"--block-size=16,"
            f"--trust-remote-code"
            f"{quant_flag}"
        )
    else:
        container_image = container_image or VERTEX_VLLM_CONTAINER
        shm_size = 16384
        timeout_sec = 1800
        env_vars = ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
        if hf_token:
            env_vars.append(f"HF_TOKEN={hf_token}")
        env_vars_str = ",".join(env_vars)
        token_env = f' \\\n    --container-env-vars="{env_vars_str}"'

        # Layer 3 guardrail: automatic --enforce-eager to prevent CUDA graph OOM on >=20B models on <=2x L4
        mid = model_id.lower()
        is_large_model = any(f"{x}b" in mid for x in range(20, 100)) or "r1" in mid
        auto_eager = enforce_eager or (is_large_model and ("l4" in vertex_accel.lower()) and gpu_count <= 2)
        eager_flag = ",--enforce-eager" if auto_eager else ""

        container_args = (
            f"--model={model_id},"
            f"--port=8080,"
            f"--tensor-parallel-size={tensor_parallel},"
            f"--max-model-len={max_model_len},"
            f"--gpu-memory-utilization={gpu_memory_utilization:.2f}"
            f"{quant_flag}"
            f"{eager_flag}"
        )

    create_endpoint_cmd = (
        f"ENDPOINT_ID=$(gcloud ai endpoints create \\\n"
        f"    --project={project} \\\n"
        f"    --region={region} \\\n"
        f"    --display-name={endpoint_name} \\\n"
        f'    --format="value(name.basename())")'
    )

    upload_model_cmd = (
        f"MODEL_ID=$(gcloud ai models upload \\\n"
        f"    --project={project} \\\n"
        f"    --region={region} \\\n"
        f"    --display-name={endpoint_name}-model \\\n"
        f'    --container-image-uri="{container_image}" \\\n'
        f'    --container-command="python3,-m,vllm.entrypoints.openai.api_server" \\\n'
        f"    --container-ports=8080 \\\n"
        f'    --container-predict-route="/v1/chat/completions" \\\n'
        f'    --container-health-route="/health" \\\n'
        f"    --container-shared-memory-size-mb={shm_size} \\\n"
        f"    --container-deployment-timeout-seconds={timeout_sec} \\\n"
        f'    --container-args="{container_args}"'
        f"{token_env} \\\n"
        f'    --format="value(model)")'
    )

    # Precompute REST payload for Option B asynchronous dispatch
    rest_accel = f', "acceleratorType": "{vertex_accel}", "acceleratorCount": {gpu_count}' if not is_tpu else ""
    rest_payload = (
        f'{{"deployedModel": {{"model": "projects/{project}/locations/{region}/models/${{MODEL_ID}}", '
        f'"displayName": "{endpoint_name}-deployed", '
        f'"dedicatedResources": {{"machineSpec": {{"machineType": "{machine_type}"{rest_accel}}}, '
        f'"minReplicaCount": 1, "maxReplicaCount": 1}}}}, "trafficSplit": {{"0": 100}}}}'
    )

    # Cardinal Rule: Do NOT pass --accelerator when machine_type is a TPU (ct*), as accelerators are hardwired.
    accel_flag = "" if is_tpu else f"    --accelerator=type={vertex_accel},count={gpu_count} \\\n"

    deploy_model_cmd = (
        f"# Option A: Standard gcloud CLI deployment\n"
        f"gcloud ai endpoints deploy-model ${{ENDPOINT_ID}} \\\n"
        f"    --project={project} \\\n"
        f"    --region={region} \\\n"
        f"    --model=${{MODEL_ID}} \\\n"
        f"    --display-name={endpoint_name}-deployed \\\n"
        f"    --machine-type={machine_type} \\\n"
        f"{accel_flag}"
        f"    --min-replica-count=1 \\\n"
        f"    --max-replica-count=1 \\\n"
        f"    --traffic-split=0=100\n\n"
        f"# Option B: Asynchronous REST API dispatch (decoupled from 1,800s client socket timeout)\n"
        f'DEPLOY_OP=$(curl -s -X POST \\\n'
        f'    -H "Authorization: Bearer $(gcloud auth print-access-token)" \\\n'
        f'    -H "Content-Type: application/json" \\\n'
        f'    "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/${{ENDPOINT_ID}}:deployModel" \\\n'
        f"    -d '{rest_payload}' | grep -o '\"name\": \"[^\"]*' | cut -d'\"' -f4)\n"
        f'echo "Operation ID: ${{DEPLOY_OP##*/}}"\n'
        f'# Track operation status beyond 1,800s:\n'
        f'gcloud ai operations describe ${{DEPLOY_OP##*/}} --project={project} --region={region}'
    )

    sample_request = (
        f'# Option A: Standard OpenAI-compatible format via rawPredict (Recommended)\n'
        f'curl -X POST \\\n'
        f'    -H "Authorization: Bearer $(gcloud auth print-access-token)" \\\n'
        f'    -H "Content-Type: application/json" \\\n'
        f'    "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/${{ENDPOINT_ID}}:rawPredict" \\\n'
        f'    -d \'{{"model": "{model_id}", "messages": [{{"role": "user", "content": "Explain quantum computing in one sentence."}}], "max_tokens": 100}}\'\n\n'
        f'# Option B: Vertex AI standard predict envelope\n'
        f'curl -X POST \\\n'
        f'    -H "Authorization: Bearer $(gcloud auth print-access-token)" \\\n'
        f'    -H "Content-Type: application/json" \\\n'
        f'    "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/${{ENDPOINT_ID}}:predict" \\\n'
        f'    -d \'{{"instances": [{{"messages": [{{"role": "user", "content": "Explain quantum computing in one sentence."}}]}}]}}\''
    )

    return {
        "create_endpoint": create_endpoint_cmd,
        "upload_model": upload_model_cmd,
        "deploy_model": deploy_model_cmd,
        "sample_request": sample_request,
    }


def generate_vertex_terraform(
    endpoint_name: str,
    model_id: str,
    project: str,
    region: str,
    machine_type: str,
    vertex_accel: str,
    gpu_count: int,
    tensor_parallel: int,
    max_model_len: int,
    quantization: str,
    hf_token: Optional[str] = None,
    is_tpu: bool = False,
    gcs_model_path: Optional[str] = None,
    container_image: Optional[str] = None,
    gpu_memory_utilization: float = 0.90,
) -> str:
    """Generates Terraform HCL configuration for Vertex AI Endpoint."""
    quant_arg = f'      "--quantization={quantization}",\n' if quantization and quantization != "none" else ""
    container_image = container_image or (VERTEX_TPU_VLLM_CONTAINER if is_tpu else VERTEX_VLLM_CONTAINER)
    
    env_blocks = []
    if is_tpu:
        env_blocks.append('    env {\n      name  = "PJRT_DEVICE"\n      value = "TPU"\n    }')
        env_blocks.append('    env {\n      name  = "VLLM_TPU_ENABLE_PALLAS"\n      value = "1"\n    }')
        if gcs_model_path:
            env_blocks.append(f'    env {{\n      name  = "GCS_MODEL_PATH"\n      value = "{gcs_model_path}"\n    }}')
    else:
        env_blocks.append('    env {\n      name  = "PYTORCH_CUDA_ALLOC_CONF"\n      value = "expandable_segments:True"\n    }')
    if hf_token:
        env_blocks.append(f'    env {{\n      name  = "HF_TOKEN"\n      value = "{hf_token}"\n    }}')
    
    token_block = "\n" + "\n".join(env_blocks) + "\n" if env_blocks else ""

    if is_tpu:
        machine_spec_body = f'      machine_type      = "{machine_type}"\n'
    else:
        machine_spec_body = (
            f'      machine_type      = "{machine_type}"\n'
            f'      accelerator_type  = "{vertex_accel}"\n'
            f'      accelerator_count = {gpu_count}\n'
        )

    return f"""# Terraform Configuration for Vertex AI Endpoint Serving: {model_id}
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 5.0"
    }}
  }}
}}

provider "google" {{
  project = "{project}"
  region  = "{region}"
}}

resource "google_vertex_ai_endpoint" "endpoint" {{
  name         = "{endpoint_name}"
  display_name = "{endpoint_name}"
  project      = "{project}"
  location     = "{region}"
}}

resource "google_vertex_ai_model" "model" {{
  display_name = "{endpoint_name}-model"
  project      = "{project}"
  location     = "{region}"

  container_spec {{
    image_uri     = "{container_image}"
    predict_route = "/v1/chat/completions"
    health_route  = "/health"
    ports {{
      container_port = 8080
    }}
    args = [
      "--model={model_id}",
      "--tensor-parallel-size={tensor_parallel}",
      "--max-model-len={max_model_len}",
      "--gpu-memory-utilization={gpu_memory_utilization:.2f}",
{quant_arg}    ]
{token_block}  }}
}}

resource "google_vertex_ai_endpoint_deployed_model" "deployment" {{
  endpoint_id  = google_vertex_ai_endpoint.endpoint.id
  model_id     = google_vertex_ai_model.model.id
  display_name = "{endpoint_name}-deployed"

  dedicated_resources {{
    machine_spec {{
{machine_spec_body}    }}
    min_replica_count = 1
    max_replica_count = 1
  }}

  traffic_split = {{
    "0" = 100
  }}
}}

output "endpoint_id" {{
  description = "The ID of the deployed Vertex AI Endpoint"
  value       = google_vertex_ai_endpoint.endpoint.id
}}

output "raw_predict_uri" {{
  description = "Vertex AI rawPredict URI for OpenAI-compatible streaming chat"
  value       = "https://{region}-aiplatform.googleapis.com/v1/${{google_vertex_ai_endpoint.endpoint.id}}:rawPredict"
}}

output "predict_uri" {{
  description = "Vertex AI Prediction URI"
  value       = "https://{region}-aiplatform.googleapis.com/v1/${{google_vertex_ai_endpoint.endpoint.id}}:predict"
}}
"""


def generate_startup_script(
    model: str,
    port: int,
    tensor_parallel: int,
    max_model_len: int,
    quantization: str,
    hf_token: Optional[str] = None,
) -> str:
    """Generates shell script embedded into GCE instance metadata to launch vLLM via Docker."""
    token_export = f'export HF_TOKEN="{hf_token}"\n' if hf_token else ""
    token_docker_env = '-e HF_TOKEN="$HF_TOKEN" ' if hf_token else ""
    quant_flag = f"--quantization {quantization} " if quantization and quantization != "none" else ""

    return f"""#!/bin/bash
set -euxo pipefail

echo "=== Fulcrum vLLM Compute Engine Node Initialization ==="
{token_export}
# Ensure NVIDIA driver and docker are ready
nvidia-smi || exit 1

# Pull official vLLM Docker image
docker pull vllm/vllm-openai:latest

# Run vLLM container
docker run -d \\
    --name vllm-server \\
    --restart unless-stopped \\
    --gpus all \\
    --ipc=host \\
    --net=host \\
    {token_docker_env}\\
    -p {port}:{port} \\
    -v ~/.cache/huggingface:/root/.cache/huggingface \\
    vllm/vllm-openai:latest \\
    --model {model} \\
    --port {port} \\
    --tensor-parallel-size {tensor_parallel} \\
    --max-model-len {max_model_len} \\
    --gpu-memory-utilization 0.95 \\
    {quant_flag}\\
    --trust-remote-code

echo "=== Initialization Complete. Probing /health on port {port}... ==="
for i in $(seq 1 60); do
    if curl -s "http://localhost:{port}/health" > /dev/null; then
        echo "vLLM service is healthy!"
        exit 0
    fi
    sleep 5
done
echo "Warning: Service startup did not complete within 5 minutes."
"""


def generate_terraform(
    instance_name: str,
    project: str,
    zone: str,
    machine_type: str,
    gpu_type: str,
    gpu_count: int,
    disk_size: int,
    startup_script: str,
    port: int,
    spot: bool = False,
) -> str:
    """Generates Terraform HCL configuration for Compute Engine fallback."""
    spot_block = (
        '    provisioning_model  = "SPOT"\n'
        '    instance_termination_action = "STOP"\n'
        if spot
        else ""
    )

    return f"""# Terraform Configuration for Compute Engine Fallback: {instance_name}
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 5.0"
    }}
  }}
}}

provider "google" {{
  project = "{project}"
  zone    = "{zone}"
}}

resource "google_compute_firewall" "allow_vllm" {{
  name    = "fulcrum-allow-vllm-{port}"
  network = "default"

  allow {{
    protocol = "tcp"
    ports    = ["{port}"]
  }}

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["fulcrum-vllm-node"]
}}

resource "google_compute_instance" "vllm_node" {{
  name         = "{instance_name}"
  machine_type = "{machine_type}"
  zone         = "{zone}"
  tags         = ["fulcrum-vllm-node"]

  boot_disk {{
    initialize_params {{
      image = "deeplearning-platform-release/common-cu124-debian-11"
      size  = {disk_size}
      type  = "pd-ssd"
    }}
  }}

  guest_accelerator {{
    type  = "{gpu_type}"
    count = {gpu_count}
  }}

  scheduling {{
    on_host_maintenance = "TERMINATE"
{spot_block}  }}

  network_interface {{
    network = "default"
    access_config {{}}
  }}

  metadata_startup_script = <<-EOT
{startup_script}
  EOT
}}

output "instance_ip" {{
  value = google_compute_instance.vllm_node.network_interface[0].access_config[0].nat_ip
}}
"""


def generate_gke_manifest(
    model_id: str,
    app_name: str,
    gpu_count: int,
    gpu_type: str,
    tensor_parallel: int,
    max_model_len: int,
    port: int = 8000,
    hf_token: Optional[str] = None,
) -> str:
    """Generates Kubernetes Deployment and Service manifest for GKE multi-node cluster serving."""
    token_env = (
        f"""        - name: HF_TOKEN
          value: "{hf_token}"
"""
        if hf_token
        else ""
    )

    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  labels:
    app: {app_name}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:latest
        args:
        - "--model={model_id}"
        - "--tensor-parallel-size={tensor_parallel}"
        - "--max-model-len={max_model_len}"
        - "--port={port}"
        env:
{token_env}        resources:
          limits:
            nvidia.com/gpu: "{gpu_count}"
        ports:
        - containerPort: {port}
---
apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
spec:
  type: LoadBalancer
  selector:
    app: {app_name}
  ports:
  - port: {port}
    targetPort: {port}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fulcrum run: Provision open-weight models on GCP (Vertex AI Endpoint primary, GCE/GKE fallback).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy Gemma 3 12B on Vertex AI Endpoint (default / auto-selected)
  python3 scripts/launch_gcp.py --model google/gemma-3-12b-it

  # Explicitly force Compute Engine fallback
  python3 scripts/launch_gcp.py --model google/gemma-3-12b-it --platform gce

  # Generate Terraform HCL for Vertex AI
  python3 scripts/launch_gcp.py --model meta-llama/Llama-3.1-8B-Instruct --format terraform

  # Dry-run inspection without provisioning
  python3 scripts/launch_gcp.py --model Qwen/Qwen2.5-Coder-7B-Instruct --dry-run
""",
    )
    parser.add_argument("--model", "-m", type=str, required=True, help="Hugging Face model ID to serve")
    parser.add_argument(
        "--platform",
        choices=["auto", "vertex", "gce", "gke"],
        default="auto",
        help="Target platform: auto (Vertex priority whenever viable), vertex, gce, or gke (default: auto)",
    )
    parser.add_argument("--instance-name", "--name", "-n", dest="instance_name", type=str, help="Resource display name (auto-derived if omitted)")
    parser.add_argument("--project", "-p", type=str, help="Google Cloud project ID")
    parser.add_argument("--zone", "-z", type=str, default="us-central1-a", help="Compute zone (default: us-central1-a)")
    parser.add_argument("--region", "-r", type=str, help="GCP region (auto-derived from zone if omitted)")
    parser.add_argument("--gpu-type", type=str, help="Accelerator type (nvidia-l4, nvidia-tesla-a100, nvidia-a100-80gb)")
    parser.add_argument("--gpu-count", type=int, help="Number of GPUs (1, 2, 4, 8)")
    parser.add_argument("--machine-type", type=str, help="Machine type (auto-selected from GPU if omitted)")
    parser.add_argument("--disk-size", type=int, default=200, help="Boot disk size in GB for GCE (default: 200)")
    parser.add_argument("--port", type=int, default=8000, help="API server port for GCE (default: 8000)")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Maximum context sequence length (default: 8192)")
    parser.add_argument(
        "--quantization",
        choices=["none", "fp8", "awq", "gptq"],
        default="none",
        help="Quantisation mode (default: none)",
    )
    parser.add_argument("--hf-token", type=str, help="Hugging Face API token for gated models (or set HF_TOKEN env)")
    parser.add_argument("--tensor-parallel-size", type=int, help="Tensor parallelism degrees (defaults to GPU/TPU count)")
    parser.add_argument(
        "--tpu-type",
        choices=["v5e", "v6e", "v5p"],
        help="Target Cloud TPU generation (v5e Lite Pod, v6e Trillium, or v5p) on Vertex AI Endpoint",
    )
    parser.add_argument(
        "--tpu-chips",
        type=int,
        choices=[1, 4, 8],
        help="Number of TPU chips (1, 4, 8) in the slice",
    )
    parser.add_argument(
        "--gcs-model-path",
        type=str,
        help="Google Cloud Storage URI where model weights are staged (e.g. gs://bucket/models/Qwen2.5-14B)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of GPU/TPU memory allocated for vLLM (default: 0.90, dynamically adjusted by memory budget)",
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Deploy as a Spot instance for up to 60-70%% cost savings",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Bypass PyTorch CUDA graph memory capture to save ~5-6 GiB VRAM per GPU (auto-enabled for >=20B models on <=2x L4)",
    )
    parser.add_argument(
        "--container-image",
        type=str,
        help="Custom serving container image URI in Google Artifact Registry",
    )
    parser.add_argument("--format", choices=["bash", "json", "terraform"], default="bash", help="Output format")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record deployment details into the local state database even in dry-run mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display provisioning plan and generated commands without executing them directly on GCP",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--check-quota",
        dest="check_quota",
        action="store_true",
        default=True,
        help="Perform pre-flight GPU/TPU quota verification before provisioning (default: True)",
    )
    parser.add_argument(
        "--no-check-quota",
        dest="check_quota",
        action="store_false",
        help="Skip pre-flight quota inspection",
    )
    parser.add_argument(
        "--strict-quota",
        action="store_true",
        help="Strictly abort deployment if quota check fails or is unverified",
    )

    args = parser.parse_args()

    # Determine GCP project and region (auto-detects from gcloud, asks user only if unset)
    project = resolve_gcp_project(args.project)
    has_explicit_region = bool(args.region) or (args.zone != "us-central1-a" and "-" in args.zone)
    region = args.region or (args.zone.rsplit("-", 1)[0] if "-" in args.zone else "us-central1")

    # Evaluate whether TPU deployment is requested or implied by machine type
    is_tpu = bool(
        args.tpu_type
        or (args.machine_type and is_tpu_machine(args.machine_type))
        or (args.gpu_type and "tpu" in args.gpu_type.lower())
    )

    tpu_sweep_report = None
    if is_tpu:
        tpu_gen = args.tpu_type or (
            "v6e" if "ct6e" in (args.machine_type or "")
            else ("v5p" if "ct5p" in (args.machine_type or "") else "v5e")
        )
        tpu_chips = args.tpu_chips or (
            8 if "8t" in (args.machine_type or "")
            else (4 if "4t" in (args.machine_type or "") else (4 if tpu_gen == "v5p" else 1))
        )
        gpu_type = f"tpu-{tpu_gen}"
        gpu_count = tpu_chips
        tensor_parallel = args.tensor_parallel_size or tpu_chips

        # Proactive Multi-Region Cloud TPU Quota Discovery:
        # When --accelerator tpu is specified without an explicit --region, do not assume us-central1.
        # Query aiplatform.googleapis.com across known TPU regions, rank by available headroom,
        # and automatically route to the region with verified capacity.
        if not has_explicit_region and getattr(args, "check_quota", True) and check_quota:
            tpu_sweep_report = check_quota.sweep_tpu_quota(
                project=project,
                tpu_gen=tpu_gen,
                required_chips=gpu_count,
            )
            discovered_region = tpu_sweep_report.get("selected_region")
            if discovered_region and tpu_sweep_report.get("status") == "SUFFICIENT":
                if discovered_region != region:
                    print(
                        f"\nℹ️  [TPU QUOTA DISCOVERY]: No explicit --region provided. Discovered verified capacity in "
                        f"'{discovered_region}' ({tpu_sweep_report['regional_headroom'].get(discovered_region, 0):.0f} chips available "
                        f"for {tpu_sweep_report['accelerator']}). Routing deployment from '{region}' to '{discovered_region}'.\n",
                        file=sys.stderr,
                    )
                    region = discovered_region
                    args.zone = f"{region}-a"
            elif discovered_region and discovered_region != region:
                region = discovered_region
                args.zone = f"{region}-a"
    else:
        # Infer GPU requirements if not explicitly supplied
        inferred_gpu, inferred_count = estimate_gpu_needs(args.model)
        gpu_type = args.gpu_type or inferred_gpu
        gpu_count = args.gpu_count or inferred_count
        tensor_parallel = args.tensor_parallel_size or min(gpu_count, 8)

    # Evaluate platform viability (Vertex Endpoint priority)
    platform, rationale = evaluate_platform_viability(
        model_id=args.model,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        requested_platform=args.platform,
    )

    # Derive resource name
    short_model = args.model.split("/")[-1]
    resource_name = args.instance_name or f"fulcrum-{sanitize_name(short_model)}"
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Pre-flight quota inspection across Vertex AI and GCE
    quota_report = None
    if getattr(args, "check_quota", True) and check_quota:
        if is_tpu and tpu_sweep_report and not has_explicit_region:
            quota_report = {
                "status": tpu_sweep_report["status"],
                "platform": platform,
                "accelerator": tpu_sweep_report["accelerator"],
                "region": region,
                "project": project,
                "required": gpu_count,
                "metric": tpu_sweep_report["metric"],
                "limit": tpu_sweep_report["regional_headroom"].get(region),
                "usage": 0.0,
                "available": tpu_sweep_report["regional_headroom"].get(region),
                "console_url": tpu_sweep_report["console_url"],
                "alternative_regions": [r for r in tpu_sweep_report["candidate_regions"] if r != region],
                "message": tpu_sweep_report["message"],
                "tpu_sweep": tpu_sweep_report,
            }
        else:
            target_accel = tpu_gen if is_tpu else gpu_type
            quota_report = check_quota.check_accelerator_quota(
                project=project,
                region=region,
                platform=platform,
                accelerator=target_accel,
                required_count=gpu_count,
                is_tpu=is_tpu,
            )
        if getattr(args, "strict_quota", False) and quota_report["status"] != "SUFFICIENT":
            print(f"\n❌ [STRICT QUOTA FAILURE]: Quota for {quota_report['accelerator']} in {region} is {quota_report['status']}.", file=sys.stderr)
            print(quota_report["message"], file=sys.stderr)
            sys.exit(1)
        if not getattr(args, "dry_run", False) and quota_report["status"] == "INSUFFICIENT":
            print(f"\n❌ [ABORTING PROVISIONING]: Insufficient quota for {quota_report['accelerator']} in {region}.", file=sys.stderr)
            print(quota_report["message"], file=sys.stderr)
            sys.exit(1)

    # Pre-Flight KV-Cache & HBM/VRAM Memory Budget Calculation:
    # Sizing model weights and available accelerator memory against requested sequence length.
    # If the model's sequence length exceeds Max Safe Context, automatically clamp --max-model-len
    # and dynamically set --gpu-memory-utilization 0.90.
    preflight_hw = args.machine_type or (resolve_machine_type(gpu_type, gpu_count, platform="vertex") if is_tpu else gpu_type)
    mem_budget = None
    if find_model:
        mem_budget = find_model.calculate_preflight_memory_budget(
            model_id=args.model,
            hardware_type=preflight_hw,
            hardware_count=gpu_count,
            requested_max_model_len=args.max_model_len,
            quantization=args.quantization,
            gpu_memory_utilization=args.gpu_memory_utilization or 0.90,
            hf_token=hf_token,
            is_tpu=is_tpu,
        )
        if mem_budget.get("clamped"):
            print(
                f"\nℹ️  [MEMORY BUDGET PRE-FLIGHT]: {mem_budget['rationale']}\n",
                file=sys.stderr,
            )

    effective_max_model_len = mem_budget["effective_max_model_len"] if mem_budget else args.max_model_len
    effective_gpu_memory_utilization = mem_budget["gpu_memory_utilization"] if mem_budget else (args.gpu_memory_utilization or 0.90)

    # =========================================================================
    # TARGET 1: VERTEX AI ENDPOINT (PRIMARY CHOICE)
    # =========================================================================
    if platform == "vertex":
        if is_tpu:
            vertex_accel = f"TPU_{tpu_gen.upper()}"
            machine_type = args.machine_type or resolve_machine_type(gpu_type, gpu_count, platform="vertex")
        else:
            vertex_accel = VERTEX_ACCELERATOR_MAP.get(gpu_type.lower(), "NVIDIA_L4")
            machine_type = args.machine_type or resolve_machine_type(gpu_type, gpu_count, platform="vertex")

        vertex_cmds = generate_vertex_commands(
            endpoint_name=resource_name,
            model_id=args.model,
            project=project,
            region=region,
            machine_type=machine_type,
            vertex_accel=vertex_accel,
            gpu_count=gpu_count,
            tensor_parallel=tensor_parallel,
            max_model_len=effective_max_model_len,
            quantization=args.quantization,
            hf_token=hf_token,
            enforce_eager=args.enforce_eager,
            is_tpu=is_tpu,
            gcs_model_path=args.gcs_model_path,
            container_image=args.container_image,
            gpu_memory_utilization=effective_gpu_memory_utilization,
        )

        accel_desc = f"{gpu_count}x Cloud TPU {tpu_gen.upper()} chips" if is_tpu else f"{gpu_count}x {vertex_accel}"

        record_data = {
            "id": resource_name,
            "platform": "vertex",
            "platform_rationale": rationale,
            "endpoint_name": resource_name,
            "model": args.model,
            "project": project,
            "region": region,
            "zone": args.zone,
            "machine_type": machine_type,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "accelerator_type": "tpu" if is_tpu else "gpu",
            "accelerator": accel_desc,
            "vertex_accelerator": vertex_accel,
            "quantization": args.quantization,
            "memory_budget": mem_budget,
            "spot": args.spot,
            "status": "RECORDED" if args.dry_run else "PROVISIONING",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if args.format == "terraform":
            print(
                generate_vertex_terraform(
                    endpoint_name=resource_name,
                    model_id=args.model,
                    project=project,
                    region=region,
                    machine_type=machine_type,
                    vertex_accel=vertex_accel,
                    gpu_count=gpu_count,
                    tensor_parallel=tensor_parallel,
                    max_model_len=effective_max_model_len,
                    quantization=args.quantization,
                    hf_token=hf_token,
                    is_tpu=is_tpu,
                    gcs_model_path=args.gcs_model_path,
                    container_image=args.container_image,
                    gpu_memory_utilization=effective_gpu_memory_utilization,
                )
            )
            if args.record:
                db.add_deployment(record_data)
            return

        if args.format == "json":
            out = {
                "platform": "vertex",
                "platform_rationale": rationale,
                "endpoint_name": resource_name,
                "project": project,
                "region": region,
                "machine_type": machine_type,
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "accelerator_type": "tpu" if is_tpu else "gpu",
                "accelerator": accel_desc,
                "quota_check": quota_report,
                "memory_budget": mem_budget,
                "commands": vertex_cmds,
            }
            if args.record:
                db.add_deployment(record_data)
            print(json.dumps(out, indent=2))
            return

        # Default bash presentation for Vertex
        serving_container = args.container_image or (VERTEX_TPU_VLLM_CONTAINER if is_tpu else VERTEX_VLLM_CONTAINER)

        print("=" * 80)
        print(f"  /fulcrum run: Vertex AI Endpoint Deployment Specification for {args.model}")
        print("=" * 80)
        print(f"Target Platform:   Vertex AI Endpoint (Managed Prediction Service)")
        print(f"Selection Reason:  {rationale}")
        print(f"Target Project:    {project}")
        print(f"Region:            {region}")
        print(f"Machine Type:      {machine_type}")
        print(f"Accelerator:       {accel_desc}")
        if quota_report:
            q_icon = "✅" if quota_report["status"] == "SUFFICIENT" else ("❌" if quota_report["status"] == "INSUFFICIENT" else "⚠️")
            q_headroom = f" ({quota_report['available']:.0f} available / {quota_report['limit']:.0f} limit)" if quota_report.get("limit") is not None else ""
            print(f"Quota Status:      {q_icon} {quota_report['status']}{q_headroom}")
        print(f"Endpoint Name:     {resource_name}")
        print(f"Serving Container: {serving_container}")
        print(f"Quantisation:      {args.quantization}")
        if mem_budget:
            budget_str = f"{mem_budget['effective_max_model_len']} tokens (Safe Context: {mem_budget['max_safe_context']}, Util: {mem_budget['gpu_memory_utilization']:.2f})"
            clamp_tag = " [CLAMPED]" if mem_budget["clamped"] else " [SAFE]"
            print(f"Memory Budget:     {budget_str}{clamp_tag}")
        if quota_report and quota_report["status"] != "SUFFICIENT":
            print("-" * 80)
            print(f"⚠️  [QUOTA NOTICE]: {quota_report['message']}")
            print(f"    Direct Console Link: {quota_report['console_url']}")
        print("-" * 80)

        if args.dry_run:
            if args.record:
                db.add_deployment(record_data)
                print(f"[fulcrum] Recorded deployment '{resource_name}' in local database ({db.get_db_path()}).\n")

            print("\n[DRY-RUN MODE] Displaying deployment plan without provisioning infrastructure.\n")
            print("Step 1: Create Vertex AI Endpoint")
            print(vertex_cmds["create_endpoint"])
            print("\nStep 2: Upload Model to Vertex Model Registry (vLLM container)")
            print(vertex_cmds["upload_model"])
            step3_title = "Step 3: Deploy Model to Endpoint (TPU Machine Specification)" if is_tpu else "Step 3: Deploy Model to Endpoint with GPU Acceleration"
            print(f"\n{step3_title}")
            print(vertex_cmds["deploy_model"])
            print("\nStep 4: Sample Prediction via Google Cloud IAM")
            print(vertex_cmds["sample_request"])
            print("=" * 80)
        else:
            print(f"\n[EXECUTION MODE] Commencing Vertex AI deployment for {resource_name}...")
            db.add_deployment(record_data)

            print("Step 1: Checking / Creating Vertex AI Endpoint...")
            query_proc = subprocess.run(
                ["gcloud", "ai", "endpoints", "list", f"--project={project}", f"--region={region}",
                 f"--filter=displayName={resource_name}", "--format=value(name.basename())",
                 "--sort-by=~createTime", "--limit=1"],
                capture_output=True, text=True, check=False
            )
            existing_ep_id = query_proc.stdout.strip().splitlines()[-1] if query_proc.stdout.strip() else ""
            if existing_ep_id:
                ep_id = existing_ep_id
                res1_code = 0
                print(f"Found existing Endpoint ID: {ep_id}")
            else:
                create_ep_cmd = f'gcloud ai endpoints create --project={project} --region={region} --display-name={resource_name} --format="value(name.basename())"'
                ep_proc = subprocess.run(create_ep_cmd, shell=True, capture_output=True, text=True, check=False)
                res1_code = ep_proc.returncode
                ep_id = ep_proc.stdout.strip().splitlines()[-1] if ep_proc.stdout.strip() else ""
                if not ep_id:
                    q = subprocess.run(
                        ["gcloud", "ai", "endpoints", "list", f"--project={project}", f"--region={region}",
                         f"--filter=displayName={resource_name}", "--format=value(name.basename())",
                         "--sort-by=~createTime", "--limit=1"],
                        capture_output=True, text=True, check=False
                    )
                    ep_id = q.stdout.strip().splitlines()[-1] if q.stdout.strip() else resource_name
                print(f"Created Endpoint ID: {ep_id}")
            record_data["endpoint_id"] = ep_id

            quant_flag = f",--quantization={args.quantization}" if args.quantization and args.quantization != "none" else ""

            if is_tpu:
                container_image = args.container_image or VERTEX_TPU_VLLM_CONTAINER
                shm_size = 32768
                timeout_sec = 7200
                env_vars = [
                    "PJRT_DEVICE=TPU",
                    "VLLM_TPU_ENABLE_PALLAS=1",
                    "RAY_DEDUP_LOGS=0",
                    "RAY_USAGE_STATS_ENABLED=0",
                    "PYTHONUNBUFFERED=1",
                ]
                if args.gcs_model_path:
                    env_vars.append(f"GCS_MODEL_PATH={args.gcs_model_path}")
                if hf_token:
                    env_vars.append(f"HF_TOKEN={hf_token}")
                env_vars_str = ",".join(env_vars)
                token_env = f'--container-env-vars="{env_vars_str}"'

                served_model = "/tmp/model" if args.gcs_model_path else args.model
                # Note: --swap-space is unsupported in vLLM Cloud TPU builds (vllm.entrypoints.openai.api_server)
                # and causes immediate argparse failure (exit code 2). KV blocks are managed directly in TPU HBM.
                container_args = (
                    f"--model={served_model},"
                    f"--port=8080,"
                    f"--tensor-parallel-size={tensor_parallel},"
                    f"--max-model-len={effective_max_model_len},"
                    f"--gpu-memory-utilization={effective_gpu_memory_utilization:.2f},"
                    f"--block-size=16,"
                    f"--trust-remote-code"
                    f"{quant_flag}"
                )
            else:
                container_image = args.container_image or VERTEX_VLLM_CONTAINER
                shm_size = 16384
                timeout_sec = 1800
                # Layer 3 guardrail: automatic --enforce-eager to prevent CUDA graph OOM on >=20B models on <=2x L4
                mid = args.model.lower()
                is_large_model = any(f"{x}b" in mid for x in range(20, 100)) or "r1" in mid
                auto_eager = args.enforce_eager or (is_large_model and ("l4" in vertex_accel.lower()) and gpu_count <= 2)
                eager_flag = ",--enforce-eager" if auto_eager else ""

                env_vars = ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
                if hf_token:
                    env_vars.append(f"HF_TOKEN={hf_token}")
                env_vars_str = ",".join(env_vars)
                token_env = f'--container-env-vars="{env_vars_str}"'

                container_args = (
                    f"--model={args.model},"
                    f"--port=8080,"
                    f"--tensor-parallel-size={tensor_parallel},"
                    f"--max-model-len={effective_max_model_len},"
                    f"--gpu-memory-utilization={effective_gpu_memory_utilization:.2f}"
                    f"{quant_flag}"
                    f"{eager_flag}"
                )

            print("Step 2: Checking / Uploading model to Vertex Model Registry...")
            query_model = subprocess.run(
                ["gcloud", "ai", "models", "list", f"--project={project}", f"--region={region}",
                 f"--filter=displayName={resource_name}-model", "--format=value(name.basename())",
                 "--sort-by=~createTime", "--limit=1"],
                capture_output=True, text=True, check=False
            )
            existing_model_id = query_model.stdout.strip().splitlines()[-1] if query_model.stdout.strip() else ""
            if existing_model_id:
                model_id = existing_model_id
                res2_code = 0
                print(f"Found existing Model ID: {model_id}")
            else:
                upload_cmd = (
                    f'gcloud ai models upload --project={project} --region={region} '
                    f'--display-name={resource_name}-model '
                    f'--container-image-uri="{container_image}" '
                    f'--container-command="python3,-m,vllm.entrypoints.openai.api_server" '
                    f'--container-ports=8080 '
                    f'--container-predict-route="/v1/chat/completions" '
                    f'--container-health-route="/health" '
                    f'--container-shared-memory-size-mb={shm_size} '
                    f'--container-deployment-timeout-seconds={timeout_sec} '
                    f'--container-args="{container_args}" '
                    f'{token_env} '
                    f'--format="value(model)"'
                )
                model_proc = subprocess.run(upload_cmd, shell=True, capture_output=True, text=True, check=False)
                res2_code = model_proc.returncode
                if res2_code != 0 and model_proc.stderr:
                    print(f"Warning: Model upload returned code {res2_code}:\n{model_proc.stderr.strip()}", file=sys.stderr)
                model_id = model_proc.stdout.strip().splitlines()[-1] if model_proc.stdout.strip() else ""
                if not model_id:
                    q_m = subprocess.run(
                        ["gcloud", "ai", "models", "list", f"--project={project}", f"--region={region}",
                         f"--filter=displayName={resource_name}-model", "--format=value(name.basename())",
                         "--sort-by=~createTime", "--limit=1"],
                        capture_output=True, text=True, check=False
                    )
                    model_id = q_m.stdout.strip().splitlines()[-1] if q_m.stdout.strip() else f"{resource_name}-model"
                print(f"Uploaded Model ID: {model_id}")
            record_data["vertex_model_id"] = model_id

            print("Step 3: Deploying model to endpoint (Asynchronous Dispatch)...")
            # 1. Attempt asynchronous REST API dispatch to decouple from gcloud's 1,800s client socket timeout
            op_name, op_id = dispatch_vertex_deploy_model_async(
                project=project,
                region=region,
                endpoint_id=ep_id,
                model_id=model_id,
                display_name=f"{resource_name}-deployed",
                machine_type=machine_type,
                vertex_accel=vertex_accel,
                gpu_count=gpu_count,
                is_tpu=is_tpu,
            )

            res3_code = 1
            if op_id:
                record_data["operation_id"] = op_id
                record_data["status"] = "DEPLOYING"
                db.add_deployment(record_data)
                try:
                    poll_vertex_operation(
                        operation_id=op_id,
                        project=project,
                        region=region,
                        timeout_seconds=timeout_sec,
                    )
                    res3_code = 0
                except Exception as e:
                    print(f"\nDeployment error during operation polling: {e}", file=sys.stderr)
                    res3_code = 1
            else:
                # 2. Resilient CLI fallback: invoke gcloud and intercept any 1,800s client timeout
                print("Notice: Dispatching via gcloud CLI with automatic 1,800s timeout interception...")
                accel_param = "" if is_tpu else f"--accelerator=type={gpu_type.lower()},count={gpu_count} "
                deploy_cmd = (
                    f"gcloud ai endpoints deploy-model {ep_id} "
                    f"--project={project} "
                    f"--region={region} "
                    f"--model={model_id} "
                    f"--display-name={resource_name}-deployed "
                    f"--machine-type={machine_type} "
                    f"{accel_param}"
                    f"--min-replica-count=1 "
                    f"--max-replica-count=1 "
                    f"--traffic-split=0=100"
                )
                res3 = subprocess.run(deploy_cmd, shell=True, capture_output=True, text=True, check=False)
                combined_output = f"{res3.stdout}\n{res3.stderr}"
                if res3.stdout:
                    print(res3.stdout)

                if res3.returncode == 0:
                    res3_code = 0
                else:
                    # Check if gcloud exited due to the 1,800s client timeout:
                    # "Operation [projects/.../operations/<id>] has not finished in 1800 seconds"
                    recovered_op_id = extract_operation_id_from_output(combined_output)
                    if recovered_op_id:
                        print(f"\nℹ️  [TIMEOUT RECOVERY]: Intercepted active Vertex Operation ID {recovered_op_id} from gcloud client timeout.")
                        print(f"   Decoupling from client socket timeout and handing off to operation poller...")
                        record_data["operation_id"] = recovered_op_id
                        record_data["status"] = "DEPLOYING"
                        db.add_deployment(record_data)
                        try:
                            poll_vertex_operation(
                                operation_id=recovered_op_id,
                                project=project,
                                region=region,
                                timeout_seconds=timeout_sec,
                            )
                            res3_code = 0
                        except Exception as e:
                            print(f"\nDeployment error during operation polling: {e}", file=sys.stderr)
                            res3_code = 1
                    else:
                        if res3.stderr:
                            print(res3.stderr, file=sys.stderr)
                        res3_code = res3.returncode

            if res1_code == 0 and res2_code == 0 and res3_code == 0:
                record_data["status"] = "RUNNING"
                db.add_deployment(record_data)
                print(f"\nSuccessfully deployed {resource_name} to Vertex AI Endpoint!")
            else:
                record_data["status"] = "FAILED"
                db.add_deployment(record_data)
                print(f"\nVertex deployment encountered errors. Check quotas and permissions.", file=sys.stderr)
                sys.exit(1)
        return

    # =========================================================================
    # TARGET 2: GOOGLE KUBERNETES ENGINE (GKE) FALLBACK (FOR MULTI-NODE)
    # =========================================================================
    if platform == "gke":
        print("=" * 80)
        print(f"  /fulcrum run: GKE Cluster Deployment Specification for {args.model}")
        print("=" * 80)
        print(f"Target Platform:   Google Kubernetes Engine (GKE Multi-Node Cluster)")
        print(f"Selection Reason:  {rationale}")
        print(f"Target Project:    {project}")
        print(f"Region:            {region}")
        print(f"Required GPUs:     {gpu_count}x {gpu_type}")
        print("-" * 80)

        gke_yaml = generate_gke_manifest(
            model_id=args.model,
            app_name=resource_name,
            gpu_count=gpu_count,
            gpu_type=gpu_type,
            tensor_parallel=tensor_parallel,
            max_model_len=effective_max_model_len,
            port=args.port,
            hf_token=hf_token,
        )

        record_data = {
            "id": resource_name,
            "platform": "gke",
            "platform_rationale": rationale,
            "instance_name": resource_name,
            "model": args.model,
            "project": project,
            "region": region,
            "zone": args.zone,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "memory_budget": mem_budget,
            "status": "RECORDED",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if args.format == "json":
            print(json.dumps({"platform": "gke", "manifest": gke_yaml, "memory_budget": mem_budget}, indent=2))
            return

        print("\nKubernetes Manifest (apply with kubectl apply -f <manifest.yaml>):")
        print(gke_yaml)
        if args.record:
            db.add_deployment(record_data)
            print(f"\n[fulcrum] Recorded deployment '{resource_name}' in local database ({db.get_db_path()}).")
        return

    # =========================================================================
    # TARGET 3: COMPUTE ENGINE (GCE) FALLBACK
    # =========================================================================
    machine_type = args.machine_type or GCE_MACHINE_MAP.get((gpu_type, gpu_count), f"g2-standard-{gpu_count * 8}")
    machine_type = args.machine_type or resolve_machine_type(gpu_type, gpu_count, platform="gce")
    instance_name = resource_name
    startup_script = generate_startup_script(
        model=args.model,
        port=args.port,
        tensor_parallel=tensor_parallel,
        max_model_len=effective_max_model_len,
        quantization=args.quantization,
        hf_token=hf_token,
    )
    script_path = f"/tmp/{instance_name}-startup.sh"

    firewall_cmd = (
        f"gcloud compute firewall-rules create fulcrum-allow-vllm-{args.port} \\\n"
        f"    --project={project} \\\n"
        f"    --direction=INGRESS \\\n"
        f"    --priority=1000 \\\n"
        f"    --network=default \\\n"
        f"    --action=ALLOW \\\n"
        f"    --rules=tcp:{args.port} \\\n"
        f"    --source-ranges=0.0.0.0/0 \\\n"
        f"    --target-tags=fulcrum-vllm-node"
    )

    spot_flags = "    --provisioning-model=SPOT \\\n    --instance-termination-action=STOP \\\n" if args.spot else ""

    create_cmd = (
        f"gcloud compute instances create {instance_name} \\\n"
        f"    --project={project} \\\n"
        f"    --zone={args.zone} \\\n"
        f"    --machine-type={machine_type} \\\n"
        f"    --accelerator=type={gpu_type},count={gpu_count} \\\n"
        f"{spot_flags}"
        f"    --maintenance-policy=TERMINATE \\\n"
        f"    --image-family=common-cu124-debian-11 \\\n"
        f"    --image-project=deeplearning-platform-release \\\n"
        f"    --boot-disk-size={args.disk_size}GB \\\n"
        f"    --boot-disk-type=pd-ssd \\\n"
        f"    --tags=fulcrum-vllm-node \\\n"
        f"    --metadata-from-file=startup-script={script_path}"
    )

    if args.format == "terraform":
        print(
            generate_terraform(
                instance_name=instance_name,
                project=project,
                zone=args.zone,
                machine_type=machine_type,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                disk_size=args.disk_size,
                startup_script=startup_script,
                port=args.port,
                spot=args.spot,
            )
        )
        return

    if args.format == "json":
        data = {
            "platform": "gce",
            "platform_rationale": rationale,
            "instance_name": instance_name,
            "project": project,
            "zone": args.zone,
            "machine_type": machine_type,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "spot": args.spot,
            "disk_size_gb": args.disk_size,
            "port": args.port,
            "firewall_command": firewall_cmd,
            "create_command": create_cmd,
            "startup_script": startup_script,
            "quota_check": quota_report,
            "memory_budget": mem_budget,
        }
        print(json.dumps(data, indent=2))
        return

    # Default bash presentation for GCE
    print("=" * 80)
    print(f"  /fulcrum run: Compute Engine Fallback Specification for {args.model}")
    print("=" * 80)
    print(f"Target Platform:   Compute Engine (GCE GPU Virtual Machine)")
    print(f"Fallback Reason:   {rationale}")
    print(f"Target Project:    {project}")
    print(f"Zone:              {args.zone}")
    print(f"Machine Type:      {machine_type}")
    print(f"Accelerator:       {gpu_count}x {gpu_type}")
    if quota_report:
        q_icon = "✅" if quota_report["status"] == "SUFFICIENT" else ("❌" if quota_report["status"] == "INSUFFICIENT" else "⚠️")
        q_headroom = f" ({quota_report['available']:.0f} available / {quota_report['limit']:.0f} limit)" if quota_report.get("limit") is not None else ""
        print(f"Quota Status:      {q_icon} {quota_report['status']}{q_headroom}")
    print(f"Spot Instance:     {'Enabled (up to 70% discount)' if args.spot else 'Disabled (standard)'}")
    print(f"Instance Name:     {instance_name}")
    print(f"Service Port:      {args.port}")
    print(f"Quantisation:      {args.quantization}")
    if mem_budget:
        budget_str = f"{mem_budget['effective_max_model_len']} tokens (Safe Context: {mem_budget['max_safe_context']}, Util: {mem_budget['gpu_memory_utilization']:.2f})"
        clamp_tag = " [CLAMPED]" if mem_budget["clamped"] else " [SAFE]"
        print(f"Memory Budget:     {budget_str}{clamp_tag}")
    if quota_report and quota_report["status"] != "SUFFICIENT":
        print("-" * 80)
        print(f"⚠️  [QUOTA NOTICE]: {quota_report['message']}")
        print(f"    Direct Console Link: {quota_report['console_url']}")
    print("-" * 80)

    record_data = {
        "id": instance_name,
        "platform": "gce",
        "platform_rationale": rationale,
        "instance_name": instance_name,
        "model": args.model,
        "project": project,
        "region": region,
        "zone": args.zone,
        "machine_type": machine_type,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "port": args.port,
        "quantization": args.quantization,
        "memory_budget": mem_budget,
        "spot": args.spot,
        "firewall_rule": f"fulcrum-allow-vllm-{args.port}",
        "status": "RECORDED" if args.dry_run else "PROVISIONING",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if args.dry_run:
        if args.record:
            db.add_deployment(record_data)
            print(f"[fulcrum] Recorded deployment '{instance_name}' in local database ({db.get_db_path()}).\n")

        print("\n[DRY-RUN MODE] Displaying deployment plan without provisioning infrastructure.\n")
        print("Step 1: Write Startup Script")
        print(f"cat << 'EOF' > {script_path}")
        print(startup_script.strip())
        print("EOF\n")

        print("Step 2: Create Firewall Rule (Allow Port 8000)")
        print(firewall_cmd)
        print("\nStep 3: Provision Compute Engine Instance with GPU")
        print(create_cmd)
        print("\nStep 4: Check Service Readiness")
        print(f'INSTANCE_IP=$(gcloud compute instances describe {instance_name} --zone={args.zone} --format="get(networkInterfaces[0].accessConfigs[0].natIP)")')
        print(f'curl -i "http://${{INSTANCE_IP}}:{args.port}/health"')
        print("\nStep 5: Send Sample Inference Request")
        print(
            f"""curl "http://${{INSTANCE_IP}}:{args.port}/v1/chat/completions" \\
    -H "Content-Type: application/json" \\
    -d '{{"model": "{args.model}", "messages": [{{"role": "user", "content": "Explain quantum computing in one sentence."}}]}}'"""
        )
        print("=" * 80)
    else:
        print(f"\n[EXECUTION MODE] Commencing GCE provisioning for {instance_name}...")
        db.add_deployment(record_data)

        with open(script_path, "w") as f:
            f.write(startup_script)
        print(f"Wrote startup script to {script_path}")

        print("Executing firewall creation...")
        subprocess.run(firewall_cmd, shell=True, check=False)

        print("Executing instance creation...")
        res = subprocess.run(create_cmd, shell=True, check=False)
        if res.returncode == 0:
            print(f"\nSuccessfully launched instance {instance_name} on GCP!")
            try:
                ip_proc = subprocess.run(
                    [
                        "gcloud",
                        "compute",
                        "instances",
                        "describe",
                        instance_name,
                        f"--zone={args.zone}",
                        f"--project={project}",
                        "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                ip_addr = ip_proc.stdout.strip()
                if ip_addr:
                    record_data["ip_address"] = ip_addr
            except Exception:
                pass
            record_data["status"] = "RUNNING"
            db.add_deployment(record_data)
            print(f"[fulcrum] Registered active deployment '{instance_name}' in local database ({db.get_db_path()}).")
        else:
            record_data["status"] = "FAILED"
            db.add_deployment(record_data)
            print(f"\nFailed to launch instance (exit code {res.returncode}). Check quotas and permissions.", file=sys.stderr)
            sys.exit(res.returncode)


if __name__ == "__main__":
    main()
