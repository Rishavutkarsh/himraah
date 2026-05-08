from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from himraah_dataset.generator import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the HimRaah route-pack dataset.")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed" / "starter"))
    args = parser.parse_args()
    report = build_dataset(Path(args.out_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
