"""Classify one or more images from the command line.

  uv run python predict.py path/to/leaf.jpg
  uv run python predict.py data/images/black_sigatoka/*.jpg --artifact artifacts/banana-clf-0.1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from common import Classifier


def latest_artifact(root: Path) -> Path:
    candidates = sorted(d for d in root.glob("*") if (d / "config.json").exists())
    if not candidates:
        raise SystemExit(f"no artifact in {root} -- run train.py first")
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--artifact", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of a table")
    args = ap.parse_args()

    artifact = args.artifact or latest_artifact(Path("artifacts"))
    clf = Classifier(artifact)
    if not args.json:
        print(f"model {clf.version}  (temperature {clf.temperature:.3f})\n")

    for path in args.images:
        if not path.exists():
            print(f"{path}: not found")
            continue
        with Image.open(path) as im:
            result = clf.predict(im, top_k=args.top_k)
        if args.json:
            print(json.dumps({"image": str(path), **result}))
        else:
            d = result["diagnosis"]
            print(f"{path.name}")
            print(f"  {d['display']:34s} {d['confidence'] * 100:5.1f}%")
            for alt in d["alternatives"]:
                print(f"    {alt['label']:32s} {alt['confidence'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
