from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_SLUG = "rishavutkarsh/himraah-text-sft-approved"
PREFLIGHT_KERNEL_ID = "rishavutkarsh/himraah-gemma-e2b-preflight"
WHEELHOUSE_SLUG = "rishavutkarsh/himraah-transformers-wheels"
GEMMA_MODEL_SOURCE = "google/gemma-4/Transformers/gemma-4-e2b-it/1"
REQUIRED_EXPORT_FILES = {
    "train.jsonl",
    "eval.jsonl",
    "eval_prompts.jsonl",
    "eval_rubric.jsonl",
    "manifest.json",
    "training_config.json",
}
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"kaggle[_-]?key", re.I),
    re.compile(r"api[_-]?key\s*[:=]", re.I),
    re.compile(r"token\s*[:=]\s*[A-Za-z0-9_\-]{20,}", re.I),
]
LOCAL_PATH_PATTERN = re.compile(r"C:\\Users\\|/Users/|/home/", re.I)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short", "--branch"], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def assert_clean_repo() -> str:
    status = git_status()
    if status.strip() != "## main...origin/main":
        raise SystemExit(f"BLOCKED: expected clean HimRaah repo on main...origin/main, got:\n{status}")
    return status


def validate_export(export_dir: Path) -> dict[str, Any]:
    missing = sorted(name for name in REQUIRED_EXPORT_FILES if not (export_dir / name).exists())
    if missing:
        raise SystemExit(f"BLOCKED: export is missing required files: {missing}")
    manifest = read_json(export_dir / "manifest.json")
    if manifest.get("project") != "himraah":
        raise SystemExit(f"BLOCKED: wrong export project: {manifest.get('project')}")
    if manifest.get("dataset_name") != "himraah-text-sft-approved":
        raise SystemExit(f"BLOCKED: wrong export dataset_name: {manifest.get('dataset_name')}")
    review = manifest.get("review_report_snapshot", {})
    if review.get("gate_status") != "APPROVED_FOR_SFT" or review.get("sft_allowed") is not True:
        raise SystemExit("BLOCKED: export review snapshot is not APPROVED_FOR_SFT")
    for key in ["source_dataset_hashes", "training_config_sha256", "export_file_manifest", "export_bundle_sha256"]:
        if not manifest.get(key):
            raise SystemExit(f"BLOCKED: export manifest missing {key}")
    exported_commit = manifest.get("git_commit")
    current_commit = git_commit()
    if exported_commit not in {current_commit, "unavailable"}:
        raise SystemExit(f"BLOCKED: export git_commit {exported_commit} does not match current {current_commit}")
    return manifest


def scan_export(export_dir: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in sorted(export_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(export_dir).as_posix()
        if rel not in REQUIRED_EXPORT_FILES:
            findings.append({"path": rel, "reason": "file is outside approved export allowlist"})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append({"path": rel, "reason": f"possible secret pattern: {pattern.pattern}"})
        if LOCAL_PATH_PATTERN.search(text):
            findings.append({"path": rel, "reason": "possible unintended local filesystem path"})
    return findings


def validate_kernel_metadata(metadata_path: Path) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    expected = {
        "id": PREFLIGHT_KERNEL_ID,
        "code_file": "gemma_lora_preflight.py",
        "enable_gpu": True,
        "enable_internet": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SystemExit(f"BLOCKED: preflight metadata {key}={metadata.get(key)!r}, expected {value!r}")
    dataset_sources = set(metadata.get("dataset_sources", []))
    if DATASET_SLUG not in dataset_sources:
        raise SystemExit(f"BLOCKED: missing dataset source {DATASET_SLUG}")
    if WHEELHOUSE_SLUG not in dataset_sources:
        raise SystemExit(f"BLOCKED: missing wheelhouse source {WHEELHOUSE_SLUG}")
    model_sources = set(metadata.get("model_sources", []))
    if GEMMA_MODEL_SOURCE not in model_sources:
        raise SystemExit(f"BLOCKED: missing model source {GEMMA_MODEL_SOURCE}")
    return metadata


def write_kaggle_dataset_metadata(export_dir: Path, manifest: dict[str, Any]) -> Path:
    metadata = {
        "title": "HimRaah Text SFT Approved Export",
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "Approved HimRaah route-pack text SFT/eval export for Kaggle preflight and Gemma LoRA.",
        "description": (
            "Immutable HimRaah export generated from the approved Gangotri-Chirbasa-Bhojbasa-Gomukh route pack. "
            f"Bundle sha256: {manifest['export_bundle_sha256']}."
        ),
        "keywords": ["gemma", "himraah", "offline", "route-pack", "sft"],
    }
    path = export_dir / "dataset-metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def write_run_log(out_file: Path, payload: dict[str, Any]) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and prepare the HimRaah Kaggle preflight package.")
    parser.add_argument("--export-dir", default=str(ROOT / "exports" / "himraah-text-sft-approved"))
    parser.add_argument("--metadata-path", default=str(ROOT / "kaggle_gemma_preflight" / "kernel-metadata.json"))
    parser.add_argument("--run-log", default=str(ROOT / "outputs" / "preflight_package_gate.json"))
    parser.add_argument("--write-kaggle-dataset-metadata", action="store_true")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    status = assert_clean_repo()
    manifest = validate_export(export_dir)
    scan_findings = scan_export(export_dir)
    metadata = validate_kernel_metadata(Path(args.metadata_path))
    dataset_metadata_path = ""
    if args.write_kaggle_dataset_metadata:
        dataset_metadata_path = str(write_kaggle_dataset_metadata(export_dir, manifest))
        scan_findings = scan_export(export_dir)
    blockers = [finding for finding in scan_findings if finding["path"] != "dataset-metadata.json"]
    if blockers:
        payload = {"ok": False, "scan_findings": scan_findings}
        write_run_log(Path(args.run_log), payload)
        raise SystemExit(f"BLOCKED: export scan found issues: {json.dumps(scan_findings, indent=2)}")
    payload = {
        "ok": True,
        "stage": "kaggle_preflight_package_gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_status": status,
        "git_commit": git_commit(),
        "export_dir": str(export_dir),
        "dataset_slug": DATASET_SLUG,
        "export_bundle_sha256": manifest["export_bundle_sha256"],
        "exported_at_utc": manifest.get("exported_at_utc"),
        "export_file_manifest": manifest["export_file_manifest"],
        "kernel_metadata": metadata,
        "dataset_metadata_path": dataset_metadata_path,
        "scan_findings": scan_findings,
        "non_evaluative_boundary": "Preflight tiny generation is a plumbing canary only; no safety or quality claims.",
    }
    write_run_log(Path(args.run_log), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
