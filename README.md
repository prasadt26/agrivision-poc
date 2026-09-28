# AgriVision POC — banana disease classifier

A minimal, working slice: **image in → disease label + calibrated confidence out**, with a
Streamlit page to try it. Deliberately *not* the full design in the diagnosis engine plan
(kept out of this repo) — no guard classes, no abstain gate, no severity, no risk
scorecard, no review queue.

Trained on [Project-AgML/PFSD_Musa_banana_disease_classification](https://huggingface.co/datasets/Project-AgML/PFSD_Musa_banana_disease_classification)
(CC-BY-4.0, commercial use permitted with attribution).

## Setup

```bash
uv sync
```

## Run

```bash
# 1. build data/images/ + data/manifest.csv  (~30 s)
uv run python prepare_data.py --hf-id Project-AgML/PFSD_Musa_banana_disease_classification

# 2. train  (~75 min on CPU, ~15 min on a Colab T4; writes artifacts/banana-clf-0.1.0/)
uv run python train.py --epochs 20

# 3. demo
uv run streamlit run streamlit_app.py
```

Also available:

```bash
uv run python predict.py data/images/black_sigatoka/*.jpg   # CLI
uv run uvicorn app:app --port 8000                          # JSON API + plain upload page
```

Add `--limit 240 --epochs 1` to `train.py` for a fast smoke test.

A trained artifact (`artifacts/banana-clf-0.2.0/`) is committed, so steps 1 and 2 are only
needed if you want to retrain. To just run the demo, `uv sync` then step 3.

## Deploying to Streamlit Community Cloud

The repo is deploy-ready: `requirements.txt` is inference-only and the trained weights are
committed under `artifacts/`, so there is nothing to fetch at boot.

1. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick this repo.
2. Main file path: `streamlit_app.py` · Branch: `main`.
3. Deploy. First build takes a few minutes while the CPU torch wheel installs.

Two things that will bite you if changed:

- **Don't loosen the `torch==2.14.0+cpu` pin in `requirements.txt`.** Plain `torch` on Linux
  pulls ~2.5 GB of CUDA libraries and the build runs out of disk. See the comment in that file.
- **Keep `artifacts/` committed.** `streamlit_app.py` scans `artifacts/*/config.json` and
  stops with a "no trained model" warning if it finds nothing — there is no download fallback.

The free tier gives ~1 GB of RAM. EfficientNet-B0 at 224 px fits comfortably, but the model
loads once per session via `@st.cache_resource`; a heavier backbone may not fit.

## Google Colab (recommended for retraining — ~10 min on a T4 vs ~75 min on local CPU)

Open [`AgriVision_POC.ipynb`](AgriVision_POC.ipynb) at [colab.research.google.com](https://colab.research.google.com)
(*File → Upload notebook*), set **Runtime → Change runtime type → T4 GPU**, and run the cells
in order. Cell 1 asks you to upload `agrivision_poc.zip` (in this folder).

Notes specific to Colab, already handled in the notebook:

- **Don't `pip install torch`** — Colab ships a CUDA build; installing from PyPI replaces it
  with a slower one.
- **Don't `uv sync`** — it builds a virtualenv the notebook kernel can't see. Use plain `pip`
  for the two missing packages (`timm`, `streamlit`).
- Use `--workers 2`; a Colab VM has 2 vCPUs, and asking for more slows the loader down.
- Streamlit is reached through a free Cloudflare tunnel (no signup) since Colab can't serve a
  port directly.
- Optionally symlink `artifacts/` to Drive so weights survive a runtime disconnect.

If you change any source file, rebuild the zip before re-uploading:

```bash
python -c "import zipfile; z=zipfile.ZipFile('agrivision_poc.zip','w',zipfile.ZIP_DEFLATED); [z.write(f,f'poc/{f}') for f in ['prepare_data.py','common.py','train.py','predict.py','app.py','streamlit_app.py','pyproject.toml','README.md']]; z.close()"
```

Putting this in a git repo removes the upload step entirely — the notebook has a `git clone`
cell ready to swap in.

## Swapping the dataset

This is the whole point of the layout. Everything downstream reads **only**
`data/manifest.csv` (`path, label, group`), so a new dataset means re-running step 1:

```bash
# client folder of class-named subdirectories
uv run python prepare_data.py --image-dir /path/to/client_photos --clean
uv run python train.py --epochs 20 --version 1.0.0

# a different HuggingFace dataset
uv run python prepare_data.py --hf-id some-org/some-dataset --clean
```

`train.py`, `predict.py`, `app.py` and `streamlit_app.py` need no edits — class names,
class count and the label order are all derived from the manifest. If the new images are
higher resolution than PFSD's 256 px, raise the input size:
`uv run python train.py --image-size 384`.

## Files

| File | Role |
|---|---|
| `prepare_data.py` | dataset → `data/images/` + `manifest.csv`. **The swap point.** |
| `common.py` | model + preprocessing, shared by training and serving |
| `train.py` | group-aware split, fine-tune, temperature calibration, metrics |
| `streamlit_app.py` | demo page |
| `predict.py` | CLI inference |
| `app.py` | FastAPI `POST /v1/diagnose` |
| `requirements.txt` | inference-only deps for Streamlit Cloud; `pyproject.toml` is the full dev set |

`common.py` exists so training and serving cannot resize differently — the most common
silent failure in a vision service.

## What this dataset does and does not support

Findings from probing the actual parquet, not the dataset card:

- **6,700 images collapse into ~2,029 near-duplicate groups.** 70% are repeat frames of the
  same scene; `potassium_deficiency` is 1,530 images but only 245 distinct scenes. The split
  is therefore by group, not by image. A random per-image split would report a number several
  points higher than anything reachable in the field.
- **All images are 256×256, already downsampled and squashed.** The design doc's 384 px input
  — justified by early Sigatoka presenting as streaks a few pixels wide — is not reachable
  here. Early-stage detection is understated by this POC and unvalidated at full resolution.
- **No `healthy` class and no `not_a_leaf` class.** The model assigns one of 8 diseases to
  every image, including a photo of a shoe. This is the first gap to close with client data.
- **Classes span different plant parts** — pseudostem weevil (stem), fruit-scarring beetle
  (fruit), soft rot (stem/corm), Sigatoka (leaf). It is a plant-part mix, not a leaf classifier,
  so a leaf-coverage quality gate would wrongly reject valid inputs.
- **Three classes are mutually confusable and one of them is notifiable.** Measured on the
  first real run: `bacterial_soft_rot`, `pseudostem_weevil` and `panama_disease` all present as
  a damaged pseudostem or corm, and the model mixes them freely — 76% of its `panama_disease`
  calls were actually weevil. That is the worst direction for the error to run, since Fusarium
  TR4 is notifiable. Whether these three should collapse into one `pseudostem_disorder →
  escalate` class is a client decision, not a modelling one.
- **Class weighting on this dataset needs a light hand.** `--class-weight-power 1.0` (full
  inverse frequency) spans 26× between the largest and smallest class and makes the model
  abandon the dominant one; the default `0.5` spans 5.2×. Re-check this after any dataset swap —
  the right value depends on how skewed the new class counts are.
- **Two classes fall below the design doc's ≥200-images-per-class bar:** `panama_disease` (102)
  and `banana_fruit_scarring_beetle` (150). Panama disease / Fusarium TR4 is also vascular and
  notifiable — the design doc is explicit that it must never be claimed confidently from a
  photo. Treat its output here as a placeholder, not a capability.

Read `artifacts/<version>/metrics.json`; its `caveats` field carries these forward.

## Attribution

Medhi & Deb, "PSFD-Musa: A dataset of banana plant, stem, fruit, leaf, and disease,"
*Data in Brief* 43:108427, 2022. doi:10.17632/4wyymrcpyz.1 — CC-BY-4.0.
