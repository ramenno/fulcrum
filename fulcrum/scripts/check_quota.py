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

# Known TPU serving regions across Google Cloud (aiplatform.googleapis.com)
KNOWN_TPU_REGIONS = ["europe-west4", "us-central1", "asia-southeast1", "us-east4"]

# Mapping of normalized accelerator keys to GCE and Vertex quota metric identifiers.
# Explicitly distinguishes Cloud TPU generations:
#   - custom_model_serving_tpu_v6e (Trillium)
#   - custom_model_serving_tpu_v5e (v5e)
#   - custom_model_serving_tpu_v5p (v5p)
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
    # Cloud TPU v5e (Lite Pod)
    "v5e": {
        "display_name": "Cloud TPU v5e",
        "is_tpu": True,
        "tpu_gen": "v5e",
        "gce_metric": "TPU_V5_LITE_PODSLICE_CHIPS",
        "vertex_metric": "custom_model_serving_tpu_v5e",
        "quota_info_id": "CustomModelServingV5ETPUPerProjectPerRegion",
        "console_filter": "custom_model_serving_tpu_v5e",
    },
    "custom_model_serving_tpu_v5e": {
        "display_name": "Cloud TPU v5e",
        "is_tpu": True,
        "tpu_gen": "v5e",
        "gce_metric": "TPU_V5_LITE_PODSLICE_CHIPS",
        "vertex_metric": "custom_model_serving_tpu_v5e",
        "quota_info_id": "CustomModelServingV5ETPUPerProjectPerRegion",
        "console_filter": "custom_model_serving_tpu_v5e",
    },
    # Cloud TPU v6e (Trillium)
    "v6e": {
        "display_name": "Cloud TPU v6e (Trillium)",
        "is_tpu": True,
        "tpu_gen": "v6e",
        "gce_metric": "TPU_V6E_CHIPS",
        "vertex_metric": "custom_model_serving_tpu_v6e",
        "quota_info_id": "CustomModelServingV6ETPUPerProjectPerRegion",
        "console_filter": "custom_model_serving_tpu_v6e",
    },
    "custom_model_serving_tpu_v6e": {
        "display_name": "Cloud TPU v6e (Trillium)",
        "is_tpu": True,
        "tpu_gen": "v6e",
        "gce_metric": "TPU_V6E_CHIPS",
        "vertex_metric": "custom_model_serving_tpu_v6e",
        "quota_info_id": "CustomModelServingV6ETPUPerProjectPerRegion",
        "console_filter": "custom_model_serving_tpu_v6e",
    },
    # Cloud TPU v5p
    "v5p": {
        "display_name": "Cloud TPU v5p",
        "is_tpu": True,
        "tpu_gen": "v5p",
        "gce_metric": "TPU_V5P_CHIPS",
        "vertex_metric": "custom_model_serving_tpu_v5p",
        "quota_info_id": "CustomModelServingV5pTPUPerProjectPerRegion",
        "console_filter": "custom_model_serving_tpu_v5p",
    },
    "custom_model_serving_tpu_v5p": {
        "display_name": "Cloud TPU v5p",
        "is_tpu": True,
        "tpu_gen": "v5p",
        "gce_metric": "TPU_V5P_CHIPS",
        "vertex_metric": "custom_model_serving_tpu_v5p",
        "quota_info_id": "CustomModelServingV5pTPUPerProjectPerRegion",
        "console_filter": "custom_model_serving_tpu_v5p",
    },
}


def normalize_accelerator_key(accel: str) -> str:
    """Normalize input accelerator string to canonical dictionary key."""
    clean = accel.strip().lower().replace("_", "-")
    if "custom-model-serving-tpu-v6e" in clean or "v6e" in clean or "ct6e" in clean or "trillium" in clean:
        return "v6e"
    if "custom-model-serving-tpu-v5p" in clean or "v5p" in clean or "ct5p" in clean:
        return "v5p"
    if "custom-model-serving-tpu-v5e" in clean or "v5e" in clean or "ct5lp" in clean:
        return "v5e"
    if clean.startswith("tpu"):
        return "v6e"
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


def query_vertex_cloud_quotas(project: str, tpu_gen: str) -> Optional[Dict[str, float]]:
    """
    Query aiplatform.googleapis.com Cloud Quotas for TPU generation across regions.
    Uses gcloud beta quotas info describe with a short timeout.
    Returns mapping of region -> available quota limit or None if unqueryable.
    """
    norm_key = normalize_accelerator_key(tpu_gen)
    spec = QUOTA_METRIC_MAP.get(norm_key, QUOTA_METRIC_MAP.get("v6e", {}))
    quota_id = spec.get("quota_info_id")
    if not quota_id:
        return None

    cmd = [
        "gcloud",
        "beta",
        "quotas",
        "info",
        "describe",
        quota_id,
        "--service=aiplatform.googleapis.com",
        f"--project={project}",
        "--quiet",
        "--format=json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            regional_limits: Dict[str, float] = {}
            for entry in data.get("dimensionsInfos", []):
                dims = entry.get("dimensions", {})
                reg = dims.get("region")
                details = entry.get("details", {})
                if reg and "value" in details:
                    try:
                        regional_limits[reg] = float(details["value"])
                    except (ValueError, TypeError):
                        pass
                # Any locations in applicableLocations without details default to 0 quota
                if not details:
                    for loc in entry.get("applicableLocations", []):
                        if loc not in regional_limits:
                            regional_limits[loc] = 0.0
            return regional_limits
    except Exception:
        pass
    return None


def sweep_tpu_quota(
    project: str,
    tpu_gen: str = "v6e",
    required_chips: int = 1,
    candidate_regions: Optional[List[str]] = None,
    mock_regional_quotas: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Proactively audit Vertex AI TPU quota across all known candidate regions.
    Ranks regions by available quota headroom and identifies the best deployment target.
    """
    regions = list(candidate_regions or KNOWN_TPU_REGIONS)
    norm_key = normalize_accelerator_key(tpu_gen)
    spec = QUOTA_METRIC_MAP.get(norm_key, QUOTA_METRIC_MAP["v6e"])
    target_metric = spec.get("vertex_metric", "custom_model_serving_tpu_v6e")

    # 1. Fetch quota numbers per region
    regional_headroom: Dict[str, Optional[float]] = {}
    if mock_regional_quotas is not None:
        for r in regions:
            regional_headroom[r] = mock_regional_quotas.get(r, 0.0)
    else:
        # Live query via Cloud Quotas / aiplatform.googleapis.com
        cloud_quotas = query_vertex_cloud_quotas(project, norm_key)
        if cloud_quotas is not None:
            for r in regions:
                regional_headroom[r] = cloud_quotas.get(r, 0.0)
        else:
            # If Cloud Quotas cannot be queried (offline, sandboxed, or lack permissions),
            # mark candidate regions as unverified
            for r in regions:
                regional_headroom[r] = None

    # 2. Rank candidate regions by headroom
    tier1 = []  # Headroom >= required_chips
    tier2 = []  # Headroom > 0 but < required_chips
    tier3 = []  # Unverified (None)
    tier4 = []  # Verified 0

    for r in regions:
        val = regional_headroom.get(r)
        if val is None:
            tier3.append((r, val))
        elif val >= required_chips:
            tier1.append((r, val))
        elif val > 0:
            tier2.append((r, val))
        else:
            tier4.append((r, val))

    tier1.sort(key=lambda x: x[1], reverse=True)
    tier2.sort(key=lambda x: x[1], reverse=True)

    ranked = tier1 + tier2 + tier3 + tier4

    if tier1:
        selected_region, best_headroom = tier1[0]
        status = "SUFFICIENT"
        message = (
            f"Automated regional sweep discovered {selected_region} with verified headroom "
            f"({best_headroom:.0f} chips available, requires {required_chips}) for {spec['display_name']}."
        )
    elif tier2:
        selected_region, best_headroom = tier2[0]
        status = "INSUFFICIENT"
        message = (
            f"Regional sweep found candidate regions, but available headroom is insufficient "
            f"(best is {selected_region} with {best_headroom:.0f} chips, requires {required_chips})."
        )
    elif tier3 and not tier4:
        selected_region = regions[0]
        status = "UNVERIFIED"
        message = (
            f"Regional TPU quota could not be automatically queried across candidate regions. "
            f"Defaulting to {selected_region}."
        )
    else:
        # All regions have 0 quota
        selected_region = regions[0]
        status = "INSUFFICIENT"
        message = (
            f"All audited TPU regions ({', '.join(regions)}) have 0 quota for {spec['display_name']}. "
            f"Request quota via Cloud Console: {build_console_quota_url(project, 'aiplatform.googleapis.com')}"
        )

    return {
        "status": status,
        "selected_region": selected_region,
        "accelerator": spec["display_name"],
        "tpu_gen": norm_key,
        "metric": target_metric,
        "required": required_chips,
        "regional_headroom": regional_headroom,
        "ranked_regions": ranked,
        "candidate_regions": regions,
        "project": project,
        "console_url": build_console_quota_url(project, "aiplatform.googleapis.com"),
        "message": message,
    }


def format_tpu_sweep_report(report: Dict[str, Any]) -> str:
    """Format structured multi-region TPU quota sweep report into human-readable British English string."""
    status = report["status"]
    status_icon = "✅" if status == "SUFFICIENT" else ("❌" if status == "INSUFFICIENT" else "⚠️")
    lines = [
        f"{status_icon} Proactive Multi-Region Cloud TPU Quota Discovery: {status}",
        f"  Target Accelerator: {report['accelerator']} (Required: {report['required']} chips)",
        f"  Target Metric:      {report['metric']} (Project: {report['project']})",
        "",
        "  Regional Quota Headroom Ranking:",
    ]
    for idx, (r, val) in enumerate(report["ranked_regions"], start=1):
        mark = "  <-- Selected for Deployment" if r == report["selected_region"] and status == "SUFFICIENT" else ""
        if val is None:
            val_str = "Unverified (offline or permission required)"
        else:
            val_str = f"{val:.0f} chips available"
        lines.append(f"    {idx}. {r:<18} {val_str:<32}{mark}")

    lines.append("")
    lines.append(f"  Routing Decision:   {report['message']}")
    if status != "SUFFICIENT":
        lines.append(f"  Quota Console URL:  {report['console_url']}")
    return "\n".join(lines)


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

    # 1. If querying Vertex AI Cloud TPU without mocks, attempt Cloud Quotas API directly
    if (is_tpu or spec.get("is_tpu", False)) and platform == "vertex" and mock_quotas is None:
        vertex_limits = query_vertex_cloud_quotas(project, norm_key)
        if vertex_limits is not None and region in vertex_limits:
            limit = vertex_limits[region]
            usage = 0.0
            available = limit
            if available >= required_count:
                return {
                    "status": "SUFFICIENT",
                    "platform": platform,
                    "accelerator": display_name,
                    "region": region,
                    "project": project,
                    "required": required_count,
                    "metric": target_metric,
                    "limit": limit,
                    "usage": usage,
                    "available": available,
                    "console_url": console_url,
                    "alternative_regions": [r for r in KNOWN_TPU_REGIONS if r != region],
                    "message": (
                        f"Quota verified via Cloud Quotas: {available:.0f} available units of {display_name} "
                        f"satisfy required count of {required_count} in {region}."
                    ),
                }
            else:
                alt_regions = [r for r in KNOWN_TPU_REGIONS if r != region]
                return {
                    "status": "INSUFFICIENT",
                    "platform": platform,
                    "accelerator": display_name,
                    "region": region,
                    "project": project,
                    "required": required_count,
                    "metric": target_metric,
                    "limit": limit,
                    "usage": usage,
                    "available": available,
                    "console_url": console_url,
                    "alternative_regions": alt_regions,
                    "message": (
                        f"Insufficient quota via Cloud Quotas: Required {required_count} units of {display_name} in {region}, "
                        f"but only {available:.0f} available (limit: {limit:.0f}). "
                        f"Request a quota increase at: {console_url} or try alternative regions: {', '.join(alt_regions)}."
                    ),
                }

    # 2. Inspect live or mocked regional quotas
    quotas_data = mock_quotas
    if quotas_data is None:
        quotas_data = query_gce_regional_quotas(project, region)

    # 3. If quota data could not be retrieved, return graceful UNVERIFIED status
    if quotas_data is None:
        alt_list = [r for r in KNOWN_TPU_REGIONS if r != region] if (is_tpu or spec.get("is_tpu", False)) else [r for r in PREFERRED_ACCELERATOR_REGIONS if r != region]
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
            "alternative_regions": alt_list,
            "message": (
                f"Could not automatically query {display_name} quota in {region} via gcloud "
                f"(service account may lack compute.regions.get permission or offline). "
                f"Please manually confirm quota in Google Cloud Console before provisioning: {console_url}"
            ),
        }

    # 4. Locate matching metric in regional quota response
    metric_check = spec["gce_metric"]
    headroom = evaluate_quota_headroom(quotas_data, metric_check, required_count)

    # If specific metric not found, inspect fallback generic GPU metric
    if headroom is None and not spec.get("is_tpu", False):
        headroom = evaluate_quota_headroom(quotas_data, "GPUS_ALL_REGIONS", required_count)
        if headroom:
            metric_check = "GPUS_ALL_REGIONS"

    if headroom is None:
        # Metric not present in regional quotas report
        alt_list = [r for r in KNOWN_TPU_REGIONS if r != region] if (is_tpu or spec.get("is_tpu", False)) else [r for r in PREFERRED_ACCELERATOR_REGIONS if r != region]
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
            "alternative_regions": alt_list,
            "message": (
                f"Quota metric '{target_metric}' was not explicitly enumerated for {region}. "
                f"Verify quota allocation in Cloud Console: {console_url}"
            ),
        }

    limit = headroom["limit"]
    usage = headroom["usage"]
    available = headroom["available"]

    alt_list = [r for r in KNOWN_TPU_REGIONS if r != region] if (is_tpu or spec.get("is_tpu", False)) else [r for r in PREFERRED_ACCELERATOR_REGIONS if r != region]

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
            "alternative_regions": alt_list,
            "message": (
                f"Quota verified: {available:.0f} available units of {display_name} "
                f"(limit: {limit:.0f}, usage: {usage:.0f}) satisfy required count of {required_count} in {region}."
            ),
        }
    else:
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
            "alternative_regions": alt_list,
            "message": (
                f"Insufficient quota: Required {required_count} units of {display_name} in {region}, "
                f"but only {available:.0f} available (limit: {limit:.0f}, usage: {usage:.0f}). "
                f"Request a quota increase at: {console_url} or try alternative regions: {', '.join(alt_list)}."
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
    parser.add_argument("--accelerator", default="v6e", help="Accelerator type (e.g. nvidia-l4, v6e, nvidia-tesla-a100)")
    parser.add_argument("--count", type=int, default=1, help="Required number of accelerators or chips")
    parser.add_argument("--region", default=None, help="Target GCP region (default: us-central1 for GPU, swept for TPU)")
    parser.add_argument("--project", default=None, help="GCP project ID (auto-detected if omitted)")
    parser.add_argument("--is-tpu", action="store_true", help="Flag if target accelerator is Cloud TPU")
    parser.add_argument("--tpu-type", choices=["v5e", "v6e", "v5p"], help="Target TPU generation")
    parser.add_argument("--sweep", action="store_true", help="Perform proactive multi-region quota discovery across candidate regions")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if quota is INSUFFICIENT or UNVERIFIED")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    args = parser.parse_args()

    project = args.project
    if not project and db:
        project = db.resolve_gcp_project()
    if not project:
        project = "your-gcp-project"

    is_tpu = args.is_tpu or bool(args.tpu_type) or "tpu" in args.accelerator.lower() or args.accelerator.lower() in ("v5e", "v6e", "v5p")
    tpu_gen = args.tpu_type or args.accelerator

    # Proactive Multi-Region Quota Sweeping
    if args.sweep or (is_tpu and not args.region):
        sweep_report = sweep_tpu_quota(
            project=project,
            tpu_gen=tpu_gen,
            required_chips=args.count,
        )
        if args.format == "json":
            print(json.dumps(sweep_report, indent=2))
        else:
            print(format_tpu_sweep_report(sweep_report))

        if args.strict and sweep_report["status"] != "SUFFICIENT":
            sys.exit(1)
        return

    target_region = args.region or "us-central1"

    report = check_accelerator_quota(
        project=project,
        region=target_region,
        platform=args.platform,
        accelerator=args.accelerator,
        required_count=args.count,
        is_tpu=is_tpu,
    )

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(format_quota_report(report))

    if args.strict and report["status"] != "SUFFICIENT":
        sys.exit(1)


if __name__ == "__main__":
    main()

