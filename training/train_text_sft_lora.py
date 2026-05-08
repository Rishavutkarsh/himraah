from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="HimRaah text SFT entrypoint with mandatory review gate.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from himraah_dataset.validators import validate_dataset

    validation = validate_dataset(dataset_dir, update_report=True)
    if not validation["valid"]:
        raise SystemExit(f"BLOCKED: dataset validation failed at training launch: {validation['errors']}")

    report_path = dataset_dir / "review_report.json"
    if not report_path.exists():
        raise SystemExit(f"BLOCKED: missing review report at {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("gate_status") != "APPROVED_FOR_SFT" or not report.get("sft_allowed"):
        reviewers = report.get("reviewers", {})
        raise SystemExit(
            "BLOCKED: HimRaah SFT is not allowed until validation passes and all three reviewer gates approve. "
            f"Current gate_status={report.get('gate_status')}; reviewers={reviewers}"
        )

    if args.dry_run:
        print("DRY RUN OK: dataset is approved for SFT. Training would start here.")
        return

    raise SystemExit("Real Gemma SFT training is not implemented in this gate stub yet.")


if __name__ == "__main__":
    main()
