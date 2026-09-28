"""Streamlit demo surface.

  uv run streamlit run streamlit_app.py

Uses the same Classifier and the same preprocessing as train.py and the API, so
what you see here is what the model actually does.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from common import Classifier

ARTIFACTS = Path("artifacts")
DATA = Path("data")

st.set_page_config(page_title="AgriVision POC", page_icon="🌿", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def load_classifier(artifact: str) -> Classifier:
    return Classifier(artifact)


@st.cache_data
def load_metrics(artifact: str) -> dict:
    p = Path(artifact) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def sample_paths(limit_per_class: int = 4) -> dict[str, list[str]]:
    manifest = DATA / "manifest.csv"
    if not manifest.exists():
        return {}
    df = pd.read_csv(manifest)
    out: dict[str, list[str]] = {}
    for label, group in df.groupby("label"):
        picks = group.sample(min(limit_per_class, len(group)), random_state=7)
        out[label] = [str(DATA / p) for p in picks["path"]]
    return out


def find_artifacts() -> list[str]:
    return [str(d) for d in sorted(ARTIFACTS.glob("*")) if (d / "config.json").exists()]


def green_shade(df: pd.DataFrame) -> pd.DataFrame:
    """Heatmap CSS for the confusion matrix, without matplotlib.

    pandas' Styler.background_gradient imports matplotlib, which is a whole plotting
    stack pulled in for one colour ramp — and one more thing that can be missing in a
    deployed environment. The sqrt keeps small off-diagonal counts visible; on a matrix
    this skewed a linear ramp washes the confusions out, and the confusions are the
    reason anyone opens this table.
    """
    vmax = float(df.to_numpy().max()) or 1.0

    def css(v) -> str:
        t = (float(v) / vmax) ** 0.5
        r, g, b = round(247 - 220 * t), round(252 - 160 * t), round(245 - 200 * t)
        return f"background-color: rgb({r},{g},{b}); color: {'#fff' if t > 0.55 else '#111'}"

    return df.map(css)


# ------------------------------------------------------------------ sidebar

artifacts = find_artifacts()

st.sidebar.title("Model")
if not artifacts:
    st.sidebar.error("No artifact found.")
    st.title("Banana disease POC")
    st.warning(
        "No trained model yet. Run:\n\n"
        "```\n"
        "uv run python prepare_data.py --hf-id Project-AgML/PFSD_Musa_banana_disease_classification\n"
        "uv run python train.py --epochs 10\n"
        "```"
    )
    st.stop()

choice = st.sidebar.selectbox("Artifact", artifacts, index=len(artifacts) - 1,
                              format_func=lambda p: Path(p).name)
clf = load_classifier(choice)
metrics = load_metrics(choice)

st.sidebar.caption(f"{clf.arch} · {clf.size}px · {len(clf.labels)} classes")
st.sidebar.caption(f"temperature {clf.temperature:.3f}")

if metrics:
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Macro-F1", f"{metrics.get('test_macro_f1', 0):.3f}")
    c2.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.3f}")
    st.sidebar.metric("Calibration error (ECE)", f"{metrics.get('ece_after_calibration', 0):.3f}",
                      delta=f"{metrics.get('ece_after_calibration', 0) - metrics.get('ece_before_calibration', 0):+.3f}",
                      delta_color="inverse")
    st.sidebar.caption("Held-out **group** split of public PFSD-Musa data — not field photos.")

with st.sidebar.expander("Read this before quoting a number"):
    for c in metrics.get("caveats", ["Metrics unavailable."]):
        st.markdown(f"- {c}")

# ------------------------------------------------------------------ main

st.title("Banana disease POC")
st.caption("Upload a photo, or try one from the dataset. Indicative only — not for field decisions.")

tab_upload, tab_sample, tab_detail = st.tabs(["Upload", "Dataset sample", "Model detail"])


def show_result(img: Image.Image, truth: str | None = None) -> None:
    left, right = st.columns([1, 1.15], gap="medium")
    with left:
        st.image(img, width='stretch')
    with right:
        result = clf.predict(img, top_k=len(clf.labels))
        d = result["diagnosis"]
        st.subheader(d["display"])
        st.progress(min(d["confidence"], 1.0), text=f"{d['confidence'] * 100:.1f}% confidence")

        if truth:
            if d["label"] == truth:
                st.success(f"Matches the dataset label: {truth.replace('_', ' ')}")
            else:
                st.error(f"Dataset label is {truth.replace('_', ' ')}")

        st.caption("All classes, calibrated")
        for alt in [{"label": d["label"], "confidence": d["confidence"]}] + d["alternatives"]:
            st.progress(min(alt["confidence"], 1.0),
                        text=f"{alt['label'].replace('_', ' ')} — {alt['confidence'] * 100:.1f}%")

        st.caption(f"model `{result['model_version']}`")

    st.info("This model has no *healthy* class and no *not-a-leaf* class, so it assigns "
            "one of its 8 diseases to every image — including a photo of a shoe.", icon="⚠️")


with tab_upload:
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp", "bmp"],
                          label_visibility="collapsed")
    if up is not None:
        show_result(Image.open(up).convert("RGB"))
    else:
        st.caption("JPEG, PNG or WebP.")

with tab_sample:
    samples = sample_paths()
    if not samples:
        st.caption("No local dataset found — run prepare_data.py to enable this tab.")
    else:
        label = st.selectbox("Class", sorted(samples), format_func=lambda s: s.replace("_", " ").title())
        paths = samples[label]
        idx = st.radio("Sample", range(len(paths)), horizontal=True,
                       format_func=lambda i: f"#{i + 1}", label_visibility="collapsed")
        show_result(Image.open(paths[idx]).convert("RGB"), truth=label)

with tab_detail:
    if not metrics:
        st.caption("No metrics.json in this artifact.")
    else:
        st.subheader("Per-class performance")
        per = metrics.get("per_class", {})
        st.dataframe(
            pd.DataFrame(per).T.rename(columns={
                "precision": "Precision", "recall": "Recall",
                "f1-score": "F1", "support": "Images"}),
            width='stretch',
        )
        st.caption("Low-support classes (Panama disease has ~100 images total) carry "
                   "unreliable numbers however good they look.")

        cm_path = Path(choice) / "confusion_matrix.csv"
        if cm_path.exists():
            st.subheader("Confusion matrix")
            st.caption("Rows = dataset label, columns = prediction.")
            cm = pd.read_csv(cm_path, index_col=0)
            st.dataframe(cm.style.apply(green_shade, axis=None),
                         width='stretch')

        st.subheader("Split")
        st.dataframe(pd.DataFrame(metrics.get("split", {})).T, width='stretch')
