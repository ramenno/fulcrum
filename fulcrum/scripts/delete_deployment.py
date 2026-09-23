#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
fulcrum delete: Resource deprovisioning and teardown orchestrator.
Safely tears down Vertex AI Endpoints (undeploy model, delete endpoint, delete model),
Compute Engine instances, associated firewall rules, and updates the local registry.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import local state database manager
sys.path.insert(0, str(Path(__file__).parent))
try:
    import db
except ImportError:
    from fulcrum.scripts import db


def is_firewall_shared(firewall_rule: str, deleting_instances: List[str], all_deployments: List[Dict[str, Any]]) -> bool:
    """Checks whether any retained deployment still relies on the specified firewall rule."""
    for d in all_deployments:
        name = d.get("instance_name") or d.get("id")
        if name in deleting_instances:
            continue
        if d.get("firewall_rule") == firewall_rule:
            return True
    return False


def build_deletion_plan(
    target_deployments: List[Dict[str, Any]],
    all_deployments: List[Dict[str, Any]],
    keep_firewall: bool = False,
) -> List[Dict[str, Any]]:
    """Builds an ordered list of deletion operations for Vertex endpoints or GCE instances."""
    plan = []
    deleting_names = [d.get("endpoint_name") or d.get("instance_name") or d.get("id") for d in target_deployments]
    handled_firewalls = set()

    for d in target_deployments:
        platform = d.get("platform", "gce")
        project = d.get("project") or db.detect_gcp_project() or ""
        region = d.get("region", "us-central1")
        zone = d.get("zone", "us-central1-a")

        # 1. Vertex AI Deprovisioning Plan
        if platform == "vertex":
            ep_name = d.get("endpoint_id") or d.get("endpoint_numeric_id") or d.get("endpoint_name") or d.get("id")
            deployed_id = d.get("deployed_model_id") or f"{ep_name}-deployed"
            model_id = d.get("vertex_model_id") or f"{ep_name}-model"

            undeploy_cmd = (
                f"gcloud ai endpoints undeploy-model {ep_name} \\\n"
                f"    --project={project} \\\n"
                f"    --region={region} \\\n"
                f"    --deployed-model-id={deployed_id} \\\n"
                f"    --quiet"
            )
            plan.append({
                "type": "vertex_undeploy",
                "name": f"{ep_name} (undeploy model)",
                "project": project,
                "region": region,
                "command": undeploy_cmd,
                "deployment": d,
            })

            delete_ep_cmd = (
                f"gcloud ai endpoints delete {ep_name} \\\n"
                f"    --project={project} \\\n"
                f"    --region={region} \\\n"
                f"    --quiet"
            )
            plan.append({
                "type": "vertex_endpoint",
                "name": ep_name,
                "project": project,
                "region": region,
                "command": delete_ep_cmd,
                "deployment": d,
            })

            delete_model_cmd = (
                f"gcloud ai models delete {model_id} \\\n"
                f"    --project={project} \\\n"
                f"    --region={region} \\\n"
                f"    --quiet"
            )
            plan.append({
                "type": "vertex_model",
                "name": model_id,
                "project": project,
                "region": region,
                "command": delete_model_cmd,
                "deployment": d,
            })
            continue

        # 2. GKE Deprovisioning Plan
        if platform == "gke":
            app_name = d.get("instance_name") or d.get("id")
            k8s_cmd = f"kubectl delete deployment {app_name} && kubectl delete service {app_name}-service"
            plan.append({
                "type": "gke_manifest",
                "name": app_name,
                "project": project,
                "region": region,
                "command": k8s_cmd,
                "deployment": d,
            })
            continue

        # 3. Compute Engine Deprovisioning Plan
        name = d.get("instance_name") or d.get("id")
        fw_rule = d.get("firewall_rule")

        instance_cmd = (
            f"gcloud compute instances delete {name} \\\n"
            f"    --project={project} \\\n"
            f"    --zone={zone} \\\n"
            f"    --quiet"
        )
        plan.append({
            "type": "instance",
            "name": name,
            "project": project,
            "zone": zone,
            "command": instance_cmd,
            "deployment": d,
        })

        if fw_rule and not keep_firewall and fw_rule not in handled_firewalls:
            if not is_firewall_shared(fw_rule, deleting_names, all_deployments):
                fw_cmd = (
                    f"gcloud compute firewall-rules delete {fw_rule} \\\n"
                    f"    --project={project} \\\n"
                    f"    --quiet"
                )
                plan.append({
                    "type": "firewall",
                    "name": fw_rule,
                    "project": project,
                    "zone": None,
                    "command": fw_cmd,
                    "deployment": d,
                })
                handled_firewalls.add(fw_rule)

    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fulcrum delete: Deprovision and delete resources created by /fulcrum run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview deletion commands for a specific deployment (dry-run)
  python3 fulcrum/scripts/delete_deployment.py --id fulcrum-gemma-3-12b-it

  # Execute deletion of a specific deployment
  python3 fulcrum/scripts/delete_deployment.py --id fulcrum-gemma-3-12b-it --execute

  # Tear down ALL tracked deployments across Vertex and GCE
  python3 fulcrum/scripts/delete_deployment.py --all --execute
""",
    )
    parser.add_argument("--id", "-i", dest="identifier", type=str, help="Resource name or deployment ID to delete")
    parser.add_argument("--all", "-a", action="store_true", help="Delete ALL deployments recorded in the database")
    parser.add_argument("--keep-firewall", action="store_true", help="Preserve GCE firewall rules during teardown")
    parser.add_argument("--format", choices=["bash", "json"], default="bash", help="Output format (default: bash)")
    parser.add_argument("--execute", action="store_true", help="Execute the deletion commands directly via gcloud")

    args = parser.parse_args()

    if not args.identifier and not args.all:
        parser.error("You must specify either --id <deployment-id> or --all.")

    all_deployments = db.load_deployments()

    if args.all:
        targets = all_deployments
    else:
        targets = [
            d for d in all_deployments
            if d.get("id") == args.identifier
            or d.get("instance_name") == args.identifier
            or d.get("endpoint_name") == args.identifier
        ]

    if not targets:
        print(f"No matching deployments found for identifier '{args.identifier}'.", file=sys.stderr)
        sys.exit(1)

    plan = build_deletion_plan(targets, all_deployments, keep_firewall=args.keep_firewall)

    if args.format == "json":
        out = {
            "target_count": len(targets),
            "operations": plan,
        }
        print(json.dumps(out, indent=2))
        return

    print("=" * 80)
    print("  /fulcrum delete: Deprovisioning Plan")
    print("=" * 80)
    print(f"Deployments to delete: {len(targets)}")
    for t in targets:
        p = t.get("platform", "gce").upper()
        tid = t.get("endpoint_name") or t.get("instance_name") or t.get("id")
        print(f"  - [{p}] {tid} ({t.get('model')})")
    print("-" * 80)

    if not args.execute:
        print("\n[DRY-RUN MODE] Run with --execute to perform teardown on GCP.\n")
        for idx, op in enumerate(plan, 1):
            print(f"Step {idx}: Delete {op['type'].upper()} '{op['name']}'")
            print(op["command"])
            print()
        print("Local Database Action:")
        for t in targets:
            tid = t.get("id") or t.get("instance_name") or t.get("endpoint_name")
            print(f"  - Remove record '{tid}' from {db.get_db_path()}")
        print("=" * 80)
    else:
        print("\n[EXECUTION MODE] Executing deprovisioning operations...")
        success = True
        for idx, op in enumerate(plan, 1):
            print(f"[{idx}/{len(plan)}] Executing: {op['type']} -> {op['name']}...")
            res = subprocess.run(op["command"], shell=True, check=False)
            if res.returncode != 0:
                print(f"Warning: Command failed with code {res.returncode}. Continuing teardown...", file=sys.stderr)
                success = False

        # Remove from local database
        for t in targets:
            tid = t.get("id") or t.get("instance_name") or t.get("endpoint_name")
            db.remove_deployment(tid)
            print(f"Removed '{tid}' from local deployment database.")

        if success:
            print("\nSuccessfully completed all deprovisioning operations!")
        else:
            print("\nDeprovisioning finished with warnings. Check GCP console for remaining resources.", file=sys.stderr)


if __name__ == "__main__":
    main()
