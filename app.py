"""Minimal demo API + upload page.

  uv run uvicorn app:app --reload --port 8000

POST /v1/diagnose  multipart field "image"  ->  JSON
GET  /healthz                              ->  model version
GET  /                                     ->  upload page

The response nests under "diagnosis" so the fuller contract in the design doc
(severity, risk, recommendation, abstain) can be added as sibling keys later
without breaking anything built against this.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from common import Classifier

ARTIFACTS = Path("artifacts")
MAX_BYTES = 12 * 1024 * 1024

app = FastAPI(title="AgriVision POC", version="0.1.0")
_clf: Classifier | None = None


def get_classifier() -> Classifier:
    global _clf
    if _clf is None:
        dirs = sorted(d for d in ARTIFACTS.glob("*") if (d / "config.json").exists())
        if not dirs:
            raise HTTPException(503, "no trained artifact found -- run train.py first")
        _clf = Classifier(dirs[-1])
    return _clf


@app.on_event("startup")
def warm() -> None:
    try:
        get_classifier()
        print(f"loaded {_clf.version}")
    except HTTPException as e:
        print(f"startup warning: {e.detail}")


@app.get("/healthz")
def healthz() -> dict:
    try:
        clf = get_classifier()
        return {"status": "ok", "model_version": clf.version, "classes": len(clf.labels)}
    except HTTPException:
        return {"status": "degraded", "model_version": None}


@app.post("/v1/diagnose")
async def diagnose(image: UploadFile = File(...)) -> JSONResponse:
    blob = await image.read()
    if not blob:
        raise HTTPException(400, "empty upload")
    if len(blob) > MAX_BYTES:
        raise HTTPException(413, f"image exceeds {MAX_BYTES // 1024 // 1024} MB")
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "not a decodable image")

    clf = get_classifier()
    result = clf.predict(img)
    result["filename"] = image.filename
    result["note"] = ("POC model: 8 disease classes only. It has no healthy class and "
                      "no not-a-leaf class, so it always returns a disease.")
    return JSONResponse(result)


PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgriVision POC</title>
<style>
 :root{color-scheme:light dark;--fg:#1a1a18;--bg:#faf9f7;--mut:#6b6b66;--line:#dedcd6;--acc:#2f6b4f}
 @media (prefers-color-scheme:dark){:root{--fg:#ece9e3;--bg:#161715;--mut:#9a9a93;--line:#32332f;--acc:#7fbf9b}}
 *{box-sizing:border-box}
 body{margin:0;padding:32px 16px;font:16px/1.55 ui-sans-serif,system-ui,sans-serif;color:var(--fg);background:var(--bg)}
 main{max-width:560px;margin:auto}
 h1{font-size:1.4rem;margin:0 0 4px}
 p.sub{color:var(--mut);font-size:.9rem;margin:0 0 24px}
 label.drop{display:block;border:1.5px dashed var(--line);border-radius:10px;padding:28px;text-align:center;cursor:pointer;color:var(--mut)}
 label.drop:hover{border-color:var(--acc);color:var(--fg)}
 input[type=file]{display:none}
 img#prev{max-width:100%;border-radius:10px;margin-top:16px;display:none}
 .row{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid var(--line)}
 .row:last-child{border:0}
 .row b{font-weight:600}
 .bar{height:4px;background:var(--line);border-radius:2px;overflow:hidden;margin-top:6px}
 .bar i{display:block;height:100%;background:var(--acc)}
 #out{margin-top:22px}
 .top{font-size:1.15rem;font-weight:600}
 .mut{color:var(--mut);font-size:.82rem}
 footer{margin-top:28px;color:var(--mut);font-size:.78rem}
</style>
<main>
  <h1>Banana disease POC</h1>
  <p class="sub">Upload a photo. 8 classes, trained on public PFSD-Musa data.</p>
  <label class="drop" for="f">Choose or drop an image<input id="f" type="file" accept="image/*"></label>
  <img id="prev" alt="">
  <div id="out"></div>
  <footer>Indicative only. No healthy or not-a-leaf class exists in this model, so
  every image returns a disease. Not for field decisions.</footer>
</main>
<script>
const f=document.getElementById('f'),out=document.getElementById('out'),prev=document.getElementById('prev');
f.onchange=async()=>{
  const file=f.files[0]; if(!file) return;
  prev.src=URL.createObjectURL(file); prev.style.display='block';
  out.innerHTML='<p class="mut">Classifying…</p>';
  const fd=new FormData(); fd.append('image',file);
  try{
    const r=await fetch('/v1/diagnose',{method:'POST',body:fd});
    const j=await r.json();
    if(!r.ok){out.innerHTML='<p class="mut">'+(j.detail||'failed')+'</p>';return;}
    const d=j.diagnosis, pct=v=>(v*100).toFixed(1)+'%';
    let h='<div class="top">'+d.display+' &middot; '+pct(d.confidence)+'</div>'
        + '<div class="bar"><i style="width:'+pct(d.confidence)+'"></i></div>';
    if(d.alternatives.length){
      h+='<p class="mut" style="margin:18px 0 4px">Alternatives</p>';
      for(const a of d.alternatives)
        h+='<div class="row"><span>'+a.label.replace(/_/g,' ')+'</span><b>'+pct(a.confidence)+'</b></div>';
    }
    h+='<p class="mut" style="margin-top:16px">model '+j.model_version+'</p>';
    out.innerHTML=h;
  }catch(e){out.innerHTML='<p class="mut">request failed</p>';}
};
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE
