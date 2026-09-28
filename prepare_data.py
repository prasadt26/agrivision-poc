"""Build data/images/<class>/*.jpg + data/manifest.csv from a dataset source.

This is the dataset swap point. Everything downstream reads only manifest.csv,
so replacing the dataset means re-running this script with a different source --
train.py, predict.py and app.py do not change.

Sources
  --parquet FILE.parquet    HuggingFace-style parquet with image/label columns
  --hf-id   ORG/NAME        download the parquet shard for that dataset
  --image-dir DIR           a folder of class-named subfolders (the client-data path)

Grouping
  Near-duplicate frames of the same physical plant must not straddle a split, or
  the test set becomes a memory test. This dataset carries no capture-session key
  (filenames are just class prefix + index), so groups are recovered with a
  perceptual hash: images within Hamming distance --hash-threshold of each other,
  in the same class, are unioned into one group.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


# ---------------------------------------------------------------- loading

def load_from_parquet(path: Path) -> tuple[list[bytes], list[str], list[str]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = table.column_names
    img_col = next((c for c in ("image", "img", "images") if c in cols), None)
    lab_col = next((c for c in ("label", "labels", "class") if c in cols), None)
    if img_col is None or lab_col is None:
        sys.exit(f"parquet needs an image and a label column; found {cols}")

    # Integer labels are decoded through the HF feature metadata when present.
    names: list[str] | None = None
    meta = table.schema.metadata or {}
    if b"huggingface" in meta:
        try:
            feats = json.loads(meta[b"huggingface"].decode())["info"]["features"]
            names = feats[lab_col].get("names")
        except (KeyError, ValueError, TypeError):
            names = None

    raw_labels = table.column(lab_col).to_pylist()
    labels = [
        names[v] if names is not None and isinstance(v, int) else str(v)
        for v in raw_labels
    ]

    blobs, stems = [], []
    for cell in table.column(img_col).to_pylist():
        if isinstance(cell, dict):
            blobs.append(cell["bytes"])
            stems.append(Path(cell.get("path") or "").stem or "")
        else:
            blobs.append(cell)
            stems.append("")
    return blobs, labels, stems


def load_from_image_dir(root: Path) -> tuple[list[bytes], list[str], list[str]]:
    blobs, labels, stems = [], [], []
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not class_dirs:
        sys.exit(f"{root} has no class subfolders")
    for d in class_dirs:
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in IMG_EXT:
                blobs.append(f.read_bytes())
                labels.append(d.name)
                stems.append(f.stem)
    return blobs, labels, stems


def fetch_hf_parquet(hf_id: str, dest: Path) -> Path:
    api = f"https://huggingface.co/api/datasets/{hf_id}/parquet/default/train"
    with urllib.request.urlopen(api) as r:
        shards = json.loads(r.read().decode())
    if not shards:
        sys.exit(f"no parquet shards listed for {hf_id}")
    if len(shards) > 1:
        print(f"note: {len(shards)} shards available, using the first only", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {shards[0]}")
    urllib.request.urlretrieve(shards[0], dest)
    return dest


# ---------------------------------------------------------------- grouping

def dhash(img: Image.Image, size: int = 8) -> np.uint64:
    """64-bit difference hash: robust to resize/compression, sensitive to content."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_matrix(hashes: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor.outer(hashes, hashes).astype(np.uint64)
    out = np.zeros(x.shape, dtype=np.uint8)
    for shift in range(0, 64, 8):
        out += _POPCOUNT[((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)]
    return out


def group_by_hash(hashes: list[np.uint64], labels: list[str], threshold: int) -> list[int]:
    """Union-find over near-identical hashes, within each class."""
    n = len(hashes)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)

    for lab, idx in by_label.items():
        arr = np.array([hashes[i] for i in idx], dtype=np.uint64)
        dist = _hamming_matrix(arr)
        ii, jj = np.where(np.triu(dist <= threshold, k=1))
        for a, b in zip(ii, jj):
            union(idx[a], idx[b])

    roots = {}
    groups = []
    for i in range(n):
        r = find(i)
        groups.append(roots.setdefault(r, len(roots)))
    return groups


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet", type=Path)
    src.add_argument("--hf-id", type=str)
    src.add_argument("--image-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--hash-threshold", type=int, default=4,
                    help="Hamming distance for near-duplicate grouping; 0 disables")
    ap.add_argument("--clean", action="store_true", help="wipe --out first")
    args = ap.parse_args()

    out: Path = args.out
    if args.clean and out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    if args.hf_id:
        pq_path = fetch_hf_parquet(args.hf_id, out / "_source" / "train0.parquet")
        blobs, labels, stems = load_from_parquet(pq_path)
    elif args.parquet:
        blobs, labels, stems = load_from_parquet(args.parquet)
    else:
        blobs, labels, stems = load_from_image_dir(args.image_dir)

    print(f"{len(blobs)} images, {len(set(labels))} classes")

    rows, hashes = [], []
    seen: set[str] = set()
    for i, (blob, lab, stem) in enumerate(tqdm(list(zip(blobs, labels, stems)), desc="writing")):
        cls = slug(lab)
        base = slug(stem) if stem else f"img_{i:06d}"
        name = base
        k = 1
        while f"{cls}/{name}" in seen:
            name = f"{base}_{k}"
            k += 1
        seen.add(f"{cls}/{name}")

        rel = Path("images") / cls / f"{name}.jpg"
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(io.BytesIO(blob)) as im:
            im = im.convert("RGB")
            w, h = im.size
            hashes.append(dhash(im))
            im.save(dst, "JPEG", quality=95)

        rows.append({"path": rel.as_posix(), "label": cls, "label_raw": lab,
                     "width": w, "height": h})

    df = pd.DataFrame(rows)
    if args.hash_threshold > 0:
        print(f"grouping near-duplicates (hamming <= {args.hash_threshold})...")
        df["group"] = group_by_hash(hashes, df["label"].tolist(), args.hash_threshold)
    else:
        df["group"] = range(len(df))

    df.to_csv(out / "manifest.csv", index=False)

    n_groups = df["group"].nunique()
    print(f"\nmanifest: {out / 'manifest.csv'}")
    print(f"  {len(df)} images in {n_groups} groups "
          f"({len(df) / max(n_groups, 1):.1f} images per group)")
    print(f"  resolution: {df.width.min()}-{df.width.max()} x {df.height.min()}-{df.height.max()}")
    print("\nper class (images / groups):")
    for lab, g in df.groupby("label"):
        print(f"  {lab:32s} {len(g):5d} / {g['group'].nunique():5d}")
    if n_groups < len(df):
        print(f"\n{len(df) - n_groups} images collapsed into a near-duplicate group. "
              "Splitting by group is what keeps the test number honest.")


if __name__ == "__main__":
    main()
