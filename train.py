"""Train the classifier from data/manifest.csv and write a versioned artifact.

Reads only the manifest, so swapping the dataset needs no change here.

  uv run python train.py --epochs 10

Splits are group-aware: every image in a near-duplicate group lands on one side
only. A random per-image split on this data would report a number several points
higher than anything reachable in the field.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import build_transform, create_model


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, labels: list[str], transform):
        self.paths = [root / p for p in df["path"]]
        self.targets = [labels.index(l) for l in df["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
            return self.transform(img), self.targets[i]


def split_by_group(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """5 stratified group folds -> fold 0 test, fold 1 val, remainder train."""
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = [te for _, te in skf.split(df, df["label"], groups=df["group"])]
    test_idx, val_idx = folds[0], folds[1]
    train_idx = np.concatenate(folds[2:])
    parts = {"train": df.iloc[train_idx], "val": df.iloc[val_idx], "test": df.iloc[test_idx]}

    overlap = set(parts["train"]["group"]) & (set(parts["val"]["group"]) | set(parts["test"]["group"]))
    assert not overlap, f"group leaked across splits: {sorted(overlap)[:5]}"
    return parts


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """One scalar on held-out logits. A fine-tuned CNN's raw softmax is not a probability."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


@torch.no_grad()  # not inference_mode: these logits are reused under autograd to fit temperature
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(device)).cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--image-size", type=int, default=224,
                    help="224 suits the 256px PFSD images; raise to 384 for higher-res client data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=3e-5)
    ap.add_argument("--class-weight-power", type=float, default=0.5,
                    help="0 = unweighted, 0.5 = sqrt inverse frequency (default), "
                         "1 = full inverse frequency (over-corrects on PFSD)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--name", default="banana-clf")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--limit", type=int, default=0, help="subsample N images for a smoke test")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = args.data / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found -- run prepare_data.py first")
    df = pd.read_csv(manifest)
    if args.limit:
        per_class = max(2, args.limit // df["label"].nunique())
        keep: list[int] = []
        for _, g in df.groupby("label"):
            keep.extend(g.sample(min(len(g), per_class), random_state=args.seed).index)
        df = df.loc[sorted(keep)].reset_index(drop=True)

    labels = sorted(df["label"].unique())
    parts = split_by_group(df, args.seed)

    print(f"device={device}  arch={args.arch}  size={args.image_size}  classes={len(labels)}")
    for k, v in parts.items():
        print(f"  {k:5s} {len(v):5d} images  {v['group'].nunique():5d} groups")

    train_tf = build_transform(args.image_size, train=True)
    eval_tf = build_transform(args.image_size, train=False)
    loaders = {}
    for k, v in parts.items():
        ds = ManifestDataset(v, args.data, labels, train_tf if k == "train" else eval_tf)
        loaders[k] = DataLoader(ds, batch_size=args.batch_size, shuffle=(k == "train"),
                                num_workers=args.workers, pin_memory=(device.type == "cuda"),
                                persistent_workers=args.workers > 0)

    model = create_model(args.arch, len(labels)).to(device)
    classifier_params = {id(p) for p in model.get_classifier().parameters()}

    # Class-weighted loss: PSEUDOSTEM WEEVIL has 2736 images, PANAMA DISEASE 102.
    # Unweighted, the rare classes are free to ignore -- but full inverse frequency
    # (power 1.0) over-corrects hard on this dataset: it gives panama_disease ~26x the
    # weight of pseudostem_weevil, and the model then abandons the dominant class
    # (recall 0.37 at precision 0.91) while 76% of its panama calls are wrong.
    # Power 0.5 -- sqrt inverse frequency -- keeps the rare classes represented without
    # inverting the prior. Use 0.0 for no weighting, 1.0 for the naive version.
    counts = Counter(parts["train"]["label"])
    freq = np.array([counts[l] for l in labels], dtype=np.float64)
    w = (freq.sum() / (len(labels) * freq)) ** args.class_weight_power
    weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    print("class weights: " + "  ".join(f"{l}={v:.2f}" for l, v in zip(labels, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) in classifier_params], "lr": args.lr_head},
        {"params": [p for p in model.parameters() if id(p) not in classifier_params], "lr": args.lr_backbone},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    def set_backbone_grad(on: bool) -> None:
        for p in model.parameters():
            if id(p) not in classifier_params:
                p.requires_grad_(on)

    best_f1, best_state = -1.0, None
    history = []

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        set_backbone_grad(not frozen)

        model.train()
        running, seen = 0.0, 0
        t0 = time.time()
        bar = tqdm(loaders["train"], desc=f"epoch {epoch}/{args.epochs}{' [frozen]' if frozen else ''}")
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
            bar.set_postfix(loss=f"{running / seen:.3f}")
        sched.step()

        logits, targets = collect_logits(model, loaders["val"], device)
        val_f1 = f1_score(targets.numpy(), logits.argmax(1).numpy(), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": running / seen,
                        "val_macro_f1": round(float(val_f1), 4),
                        "seconds": round(time.time() - t0, 1)})
        print(f"  train_loss={running / seen:.4f}  val_macro_f1={val_f1:.4f}  ({time.time() - t0:.0f}s)")

        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  new best (macro-F1 {best_f1:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibrate on val, then report on test -- never the other way round.
    val_logits, val_targets = collect_logits(model, loaders["val"], device)
    temperature = fit_temperature(val_logits, val_targets)
    print(f"\ntemperature = {temperature:.3f}")

    test_logits, test_targets = collect_logits(model, loaders["test"], device)
    y_true = test_targets.numpy()
    raw_probs = test_logits.softmax(1).numpy()
    cal_probs = (test_logits / temperature).softmax(1).numpy()
    y_pred = cal_probs.argmax(1)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=labels,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    ece_raw = expected_calibration_error(raw_probs, y_true)
    ece_cal = expected_calibration_error(cal_probs, y_true)

    out_dir = args.out / f"{args.name}-{args.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps({
        "version": f"{args.name}-{args.version}",
        "arch": args.arch,
        "image_size": args.image_size,
        "labels": labels,
        "temperature": round(temperature, 4),
    }, indent=2))

    metrics = {
        "version": f"{args.name}-{args.version}",
        "split": {k: {"images": len(v), "groups": int(v["group"].nunique())} for k, v in parts.items()},
        "class_weight_power": args.class_weight_power,
        "epochs": args.epochs,
        "test_macro_f1": round(float(macro_f1), 4),
        "test_accuracy": round(float(report["accuracy"]), 4),
        "ece_before_calibration": round(ece_raw, 4),
        "ece_after_calibration": round(ece_cal, 4),
        "temperature": round(temperature, 4),
        "per_class": {l: {k: round(float(v), 4) for k, v in report[l].items()} for l in labels},
        "history": history,
        "caveats": [
            "Measured on a held-out group split of the public PFSD-Musa dataset, "
            "not on field photos. Expect a drop on phone images of a real canopy.",
            "Source images are 256x256, already downsampled; fine lesion detail is "
            "partly gone, so early-stage Sigatoka performance is understated here "
            "and unvalidated at full resolution.",
            "No healthy and no not-a-leaf class exists in this dataset, so the model "
            "assigns one of 8 diseases to every image, including a photo of a shoe.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")

    print(f"\ntest macro-F1 {macro_f1:.4f}   accuracy {report['accuracy']:.4f}")
    print(f"ECE {ece_raw:.4f} -> {ece_cal:.4f} after calibration\n")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    print(f"artifact: {out_dir}")


if __name__ == "__main__":
    main()
