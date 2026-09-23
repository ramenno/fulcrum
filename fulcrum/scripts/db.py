#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
fulcrum state manager: Local JSON database tracking active and historical deployments.
Manages metadata for Compute Engine instances, models, ports, and lifecycle states.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_db_path() -> Path:
    """
    Determines path to deployments.json database.
    Prefers FULCRUM_DB_PATH (or legacy SOW_DB_PATH) env var, then ~/.config/fulcrum/deployments.json,
    falling back to .fulcrum/deployments.json in current working directory.
    """
    env_path = os.environ.get("FULCRUM_DB_PATH") or os.environ.get("SOW_DB_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    primary_dir = Path.home() / ".config" / "fulcrum"
    try:
        primary_dir.mkdir(parents=True, exist_ok=True)
        test_file = primary_dir / ".perm_check"
        test_file.touch()
        test_file.unlink()
        return primary_dir / "deployments.json"
    except (OSError, PermissionError):
        pass

    fallback_dir = Path.cwd() / ".fulcrum"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir / "deployments.json"


def load_deployments() -> List[Dict[str, Any]]:
    """Loads all tracked deployment records from local JSON database."""
    db_file = get_db_path()
    if not db_file.exists():
        return []
    try:
        with open(db_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "deployments" in data:
                return data["deployments"]
            return []
    except (json.JSONDecodeError, OSError):
        return []


list_deployments = load_deployments


def save_deployments(deployments: List[Dict[str, Any]]) -> None:
    """Atomically writes deployment list to local JSON database."""
    db_file = get_db_path()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = db_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(deployments, f, indent=2, ensure_ascii=False)
    temp_file.replace(db_file)


def add_deployment(record: Dict[str, Any]) -> Dict[str, Any]:
    """Adds or updates a deployment record in the database."""
    deployments = load_deployments()
    identifier = record.get("id") or record.get("instance_name")
    if not identifier:
        raise ValueError("Deployment record must contain an 'id' or 'instance_name'")

    if "created_at" not in record:
        record["created_at"] = datetime.now(timezone.utc).isoformat()
    if "status" not in record:
        record["status"] = "PROVISIONED"

    updated = False
    for i, item in enumerate(deployments):
        if item.get("id") == identifier or item.get("instance_name") == identifier:
            deployments[i] = {**item, **record}
            updated = True
            break

    if not updated:
        deployments.append(record)

    save_deployments(deployments)
    return record


def get_deployment(identifier: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single deployment by id, instance_name, endpoint_name, or endpoint_id."""
    deployments = load_deployments()
    for item in deployments:
        if (
            item.get("id") == identifier
            or item.get("instance_name") == identifier
            or item.get("endpoint_name") == identifier
            or item.get("endpoint_id") == identifier
        ):
            return item
    return None


def update_deployment(identifier: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Updates fields of an existing deployment record."""
    deployments = load_deployments()
    for i, item in enumerate(deployments):
        if (
            item.get("id") == identifier
            or item.get("instance_name") == identifier
            or item.get("endpoint_name") == identifier
            or item.get("endpoint_id") == identifier
        ):
            deployments[i] = {**item, **updates}
            save_deployments(deployments)
            return deployments[i]
    return None


def remove_deployment(identifier: str) -> Optional[Dict[str, Any]]:
    """Deletes a deployment record from the database."""
    deployments = load_deployments()
    target = None
    remaining = []
    for item in deployments:
        if (
            item.get("id") == identifier
            or item.get("instance_name") == identifier
            or item.get("endpoint_name") == identifier
            or item.get("endpoint_id") == identifier
        ):
            target = item
        else:
            remaining.append(item)

    if target:
        save_deployments(remaining)
    return target


def clear_deployments() -> None:
    """Removes all deployments from the database."""
    save_deployments([])


def detect_gcp_project() -> Optional[str]:
    """Attempts to read active GCP project from gcloud configuration or environment."""
    try:
        import subprocess
        proc = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            if line and "(unset)" not in line and not line.startswith("WARNING:"):
                return line
    except Exception:
        pass
    env_proj = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if env_proj and "(unset)" not in env_proj and env_proj.strip():
        return env_proj.strip()
    return None


def resolve_gcp_project(explicit_project: Optional[str] = None) -> str:
    """
    Resolves active GCP project:
    1. Returns explicit_project if provided.
    2. Runs gcloud config get-value project.
    3. Asks user interactively only if no project is set.
    """
    if explicit_project and explicit_project.strip() and explicit_project != "YOUR_GCP_PROJECT_ID":
        return explicit_project.strip()

    detected = detect_gcp_project()
    if detected:
        return detected

    if sys.stdin.isatty():
        try:
            val = input("No GCP project is currently set in gcloud config. Please enter your GCP Project ID: ").strip()
            if val:
                return val
        except (EOFError, KeyboardInterrupt):
            pass

    print(
        "Error: No GCP Project configured in gcloud. Please run 'gcloud config set project <PROJECT_ID>' "
        "or pass '--project <PROJECT_ID>'.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    path = get_db_path()
    count = len(load_deployments())
    detected_proj = detect_gcp_project()
    print(f"Deployments Database: {path} ({count} active entries)")
    print(f"Active GCP Project:   {detected_proj or 'None (unset)'}")

