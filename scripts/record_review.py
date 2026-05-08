from __future__ import annotations

import argparse
import json
from pathlib import Path


VALID_REVIEWERS = {"safety", "source_grounding", "training_eval"}
VALID_STATUSES = {"PENDING", "APPROVE", "APPROVE_WITH_CHANGES", "BLOCK"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a HimRaah reviewer gate decision.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--reviewer", required=True, choices=sorted(VALID_REVIEWERS))
    parser.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    report_path = Path(args.dataset_dir) / "review_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("reviewers", {})
    report["reviewers"][args.reviewer] = {
        "status": args.status,
        "required_before_sft": True,
        "notes": args.notes,
    }

    validation_ok = report.get("validation", {}).get("valid") is True
    reviewer_statuses = [report["reviewers"].get(name, {}).get("status") for name in sorted(VALID_REVIEWERS)]
    if "BLOCK" in reviewer_statuses:
        report["gate_status"] = "BLOCKED_BY_REVIEW"
        report["sft_allowed"] = False
    elif validation_ok and all(status == "APPROVE" for status in reviewer_statuses):
        report["gate_status"] = "APPROVED_FOR_SFT"
        report["sft_allowed"] = True
    elif validation_ok and any(status == "APPROVE_WITH_CHANGES" for status in reviewer_statuses):
        report["gate_status"] = "BLOCKED_REVIEW_CHANGES_REQUIRED"
        report["sft_allowed"] = False
    elif validation_ok:
        report["gate_status"] = "BLOCKED_PENDING_THREE_REVIEWS"
        report["sft_allowed"] = False
    else:
        report["gate_status"] = "BLOCKED_VALIDATION_FAILED"
        report["sft_allowed"] = False

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate_status": report["gate_status"], "sft_allowed": report["sft_allowed"]}, indent=2))


if __name__ == "__main__":
    main()
