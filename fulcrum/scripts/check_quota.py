#!/usr/bin/env python3
"""
Pre-Flight Quota Inspection Utility for /fulcrum.

Audits GPU and Cloud TPU quota availability across Google Cloud Vertex AI
and Compute Engine prior to attempting infrastructure deployment.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Register parent fulcrum package for imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from fulcrum.scripts import db
except ImportError:
    db = None

# Known high-allocation Google Cloud accelerator regions
PREFERRED_ACCELERATOR_REGIONS = ["us-central1", "europe-west4", "us-east4", "us-west1"]

# Mapping of normalized accelerator keys to GCE and Vertex quota metric identifiers
QUOTA_METRIC_MAP: Dict[str, Dict[str, Any]] = {
    "nvidia-l4": {
        "display_name": "NVIDIA L4",
        "is_tpu": False,
        "gce_metric": "NVIDIA_L4_GPUS",
        "vertex_metric": "custom_model_serving_nvidia_l4_gpus",
        "console_filter": "NVIDIA_L4",
    },
    "nvidia_l4": {
        "display_name": "NVIDIA L4",
        "is_tpu": False,
        "gce_metric": "NVIDIA_L4_GPUS",
        "vertex_metric": "custom_model_serving_nvidia_l4_gpus",
        "console_filter": "NVIDIA_L4",
    },
    "nvidia-tesla-a100": {
        "display_name": "NVIDIA A100 (40GB/80GB)",
        "is_tpu": False,
        "gce_metric": "NVIDIA_A100_GPUS",
        "vertex_metric": "custom_model_serving_nvidia_a100_gpus",
        "console_filter": "NVIDIA_A100",
    },
    "nvidia-a100-80gb": {
        "display_name": "NVIDIA A100 80GB",
        "is_tpu": False,
        "gce_metric": "NVIDIA_A100_80GB_GPUS",
        "vertex_metric": "custom_model_serving_nvidia_a100_gpus",
        "console_filter": "NVIDIA_A100",
    },
    "nvidia-h100-80gb": {
        "display_name": "NVIDIA H100 80GB",
        "is_tpu": False,
        "gce_metric": "NVIDIA_H100_GPUS",
        "vertex_metric": "custom_model_serving_nvidia_h100_gpus",
        "console_filter": "NVIDIA_H100",
    },
    "v5e": {
        "display_name": "Cloud TPU v5e",
        "is_tpu": True,
        "gce_metric": "TPU_V5_LITE_PODSLICE_CHIPS",
        "vertex_metric": "custom_model_serving_tpu_v5e_chips",
        "console_filter": "TPU v5",
    },
    "v6e": {
        "display_name": "Cloud TPU v6e (Trillium)",
        "is_tpu": True,
        "gce_metric": "TPU_V6E_CHIPS",
        "vertex_metric": "CustomModelServingV6ETPUPerProjectPerRegion",
        "console_filter": "CustomModelServingV6ETPU",
    },
    "v5p": {
        "display_name": "Cloud TPU v5p",
        "is_tpu": True,
        "gce_metric": "TPU_V5P_CHIPS",
        "vertex_metric": "custom_model_serving_tpu_v5p_chips",
        "console_filter": "TPU v5p",
    },
}


def normalize_accelerator_key(accel: str) -> str:
    """Normalize input accelerator string to canonical dictionary key."""
    clean = accel.strip().lower().replace("_", "-")
    if clean.startswith("ct5lp"):
        return "v5e"
    if clean.startswith("ct6e"):
        return "v6e"
    if clean.startswith("ct5p"):
        return "v5p"
    if "l4" in clean:
        return "nvidia-l4"
    if "a100-80" in clean or "ultragpu" in clean:
        return "nvidia-a100-80gb"
    if "a100" in clean:
        return "nvidia-tesla-a100"
    if "h100" in clean:
        return "nvidia-h100-80gb"
    return clean


def build_console_quota_url(project: str, service: str = "aiplatform.googleapis.com") -> str:
    """Generate direct hyperlink to Google Cloud Console Quotas page."""
    return f"https://console.cloud.google.com/iam-admin/quotas?project={project}&service={service}"


def query_gce_regional_quotas(project: str, region: str) -> Optional[List[Dict[str, Any]]]:
    """
    Query regional quotas using gcloud compute regions describe.
    Returns list of quota entries or None if command fails.
    """
    cmd = [
        "gcloud",
        "compute",
        "regions",
        "describe",
        region,
        f"--project={project}",
        "--format=json(quotas)",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            return data.get("quotas", [])
    except Exception:
        pass
    return None


def evaluate_quota_headroom(
    quotas_list: List[Dict[str, Any]],
    target_metric: str,
    required_count: int,
) -> Optional[Dict[str, float]]:
    """Search quotas list for target metric and calculate available headroom."""
    target_clean = target_metric.upper().replace("-", "_")
    for q in quotas_list:
        metric_name = q.get("metric", "").upper()
        if metric_name == target_clean or target_clean in metric_name:
            limit = float(q.get("limit", 0.0))
            usage = float(q.get("usage", 0.0))
            available = max(0.0, limit - usage)
            return {
                "limit": limit,
                "usage": usage,
                "available": available,
            }
    return None


def check_accelerator_quota(
    project: str,
    region: str,
    platform: str,
    accelerator: str,
    required_count: int,
    is_tpu: bool = False,
    mock_quotas: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Perform pre-flight quota inspection for GPU or Cloud TPU resource.

    Returns structured report containing status ('SUFFICIENT', 'INSUFFICIENT',
    'UNVERIFIED'), numeric limits, available headroom, and guidance.
    """
    norm_key = normalize_accelerator_key(accelerator)
    spec = QUOTA_METRIC_MAP.get(norm_key, {
        "display_name": accelerator,
        "is_tpu": is_tpu,
        "gce_metric": accelerator.upper(),
        "vertex_metric": f"custom_model_serving_{accelerator.lower()}",
        "console_filter": accelerator,
    })

    display_name = spec["display_name"]
    target_metric = spec["vertex_metric"] if platform == "vertex" else spec["gce_metric"]
    console_service = "aiplatform.googleapis.com" if platform == "vertex" else "compute.googleapis.com"
    console_url = build_console_quota_url(project, console_service)

    # 1. Inspect live or mocked regional quotas
    quotas_data = mock_quotas
    if quotas_data is None:
        quotas_data = query_gce_regional_quotas(project, region)

    # 2. If quota data could not be retrieved, return graceful UNVERIFIED status
    if quotas_data is None:
        return {
            "status": "UNVERIFIED",
            "platform": platform,
            "accelerator": display_name,
            "region": region,
            "project": project,
            "required": required_count,
            "metric": target_metric,
            "limit": None,
            "usage": None,
            "available": None,
            "console_url": console_url,
            "alternative_regions": [r for r in PREFERRED_ACCELERATOR_REGIONS if r != region],
            "message": (
                f"Could not automatically query {display_name} quota in {region} via gcloud "
                f"(service account may lack compute.regions.get permission or offline). "
                f"Please manually confirm quota in Google Cloud Console before provisioning: {console_url}"
            ),
        }

    # 3. Locate matching metric in regional quota response
    metric_check = spec["gce_metric"]
    headroom = evaluate_quota_headroom(quotas_data, metric_check, required_count)

    # If specific metric not found, inspect fallback generic GPU metric
    if headroom is None and not spec.get("is_tpu", False):
        headroom = evaluate_quota_headroom(quotas_data, "GPUS_ALL_REGIONS", required_count)
        if headroom:
            metric_check = "GPUS_ALL_REGIONS"

    if headroom is None:
        # Metric not present in regional quotas report
        return {
            "status": "UNVERIFIED",
            "platform": platform,
            "accelerator": display_name,
            "region": region,
            "project": project,
            "required": required_count,
            "metric": target_metric,
            "limit": None,
            "usage": None,
            "available": None,
            "console_url": console_url,
            "alternative_regions": [r for r in PREFERRED_ACCELERATOR_REGIONS if r != region],
            "message": (
                f"Quota metric '{target_metric}' was not explicitly enumerated for {region}. "
                f"Verify quota allocation in Cloud Console: {console_url}"
            ),
        }

    limit = headroom["limit"]
    usage = headroom["usage"]
    available = headroom["available"]

    if available >= required_count:
        return {
            "status": "SUFFICIENT",
            "platform": platform,
            "accelerator": display_name,
            "region": region,
            "project": project,
            "required": required_count,
            "metric": metric_check,
            "limit": limit,
            "usage": usage,
            "available": available,
            "console_url": console_url,
            "alternative_regions": [r for r in PREFERRED_ACCELERATOR_REGIONS if r != region],
            "message": (
                f"Quota verified: {available:.0f} available units of {display_name} "
                f"(limit: {limit:.0f}, usage: {usage:.0f}) satisfy required count of {required_count} in {region}."
            ),
        }
    else:
        alt_regions = [r for r in PREFERRED_ACCELERATOR_REGIONS if r != region]
        return {
            "status": "INSUFFICIENT",
            "platform": platform,
            "accelerator": display_name,
            "region": region,
            "project": project,
            "required": required_count,
            "metric": metric_check,
            "limit": limit,
            "usage": usage,
            "available": available,
            "console_url": console_url,
            "alternative_regions": alt_regions,
            "message": (
                f"Insufficient quota: Required {required_count} units of {display_name} in {region}, "
                f"but only {available:.0f} available (limit: {limit:.0f}, usage: {usage:.0f}). "
                f"Request a quota increase at: {console_url} or try alternative regions: {', '.join(alt_regions)}."
            ),
        }


def format_quota_report(report: Dict[str, Any]) -> str:
    """Format structured quota report into human-readable British English string."""
    status = report["status"]
    status_icon = "✅" if status == "SUFFICIENT" else ("❌" if status == "INSUFFICIENT" else "⚠️")

    lines = [
        f"{status_icon} Pre-Flight Quota Check: {status}",
        f"  Target Accelerator: {report['accelerator']} (Required: {report['required']})",
        f"  Target Region:      {report['region']} (Project: {report['project']})",
    ]
    if report["limit"] is not None:
        lines.append(f"  Quota Limit:        {report['limit']:.0f}")
        lines.append(f"  Current Usage:      {report['usage']:.0f}")
        lines.append(f"  Available Headroom: {report['available']:.0f}")

    lines.append(f"  Details:            {report['message']}")
    if status != "SUFFICIENT":
        lines.append(f"  Quota Console URL:  {report['console_url']}")
        lines.append(f"  Recommended Hubs:   {', '.join(report['alternative_regions'])}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight quota inspection for Google Cloud accelerator deployments."
    )
    parser.add_argument("--platform", choices=["vertex", "gce"], default="vertex", help="Target GCP platform")
    parser.add_argument("--accelerator", required=True, help="Accelerator type (e.g. nvidia-l4, v6e, nvidia-tesla-a100)")
    parser.add_argument("--count", type=int, default=1, help="Required number of accelerators or chips")
    parser.add_argument("--region", default="us-central1", help="Target GCP region (default: us-central1)")
    parser.add_argument("--project", default=None, help="GCP project ID (auto-detected if omitted)")
    parser.add_argument("--is-tpu", action="store_true", help="Flag if target accelerator is Cloud TPU")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if quota is INSUFFICIENT or UNVERIFIED")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    args = parser.parse_args()

    project = args.project
    if not project and db:
        project = db.resolve_gcp_project()
    if not project:
        project = "your-gcp-project"

    report = check_accelerator_quota(
        project=project,
        region=args.region,
        platform=args.platform,
        accelerator=args.accelerator,
        required_count=args.count,
        is_tpu=args.is_tpu,
    )

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(format_quota_report(report))

    if args.strict and report["status"] != "SUFFICIENT":
        sys.exit(1)


if __name__ == "__main__":
    main()

