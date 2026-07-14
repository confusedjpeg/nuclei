from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any
import random

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.pipeline import load_model, preprocess, infer, draw_contours, extract_features, base64_encode


model_lock = None
ml_models: Dict[str, Any] = {}

SAMPLE_IMAGES_DIR = PROJECT_ROOT / "dsb2018" / "test" / "images"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading nuclei segmentation model on {device} ...")
    model = load_model(device)
    ml_models["segmenter"] = model
    ml_models["device"] = device
    print("Model loaded. Ready for inference.")
    yield
    ml_models.clear()


app = FastAPI(
    title="BioVision-CellSegmenter",
    description="Automated nuclei segmentation & feature extraction pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": "segmenter" in ml_models,
        "device": str(ml_models.get("device", "unknown")),
    }


@app.get("/sample")
async def get_sample():
    """Return a random test image from the DSB 2018 test set."""
    if not SAMPLE_IMAGES_DIR.is_dir():
        raise HTTPException(500, "Sample images directory not found")
    images = sorted(SAMPLE_IMAGES_DIR.glob("*.tif"))
    if not images:
        raise HTTPException(500, "No sample images available")
    path = random.choice(images)
    return FileResponse(str(path), media_type="image/tiff", filename=path.name)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), return_annotated: bool = True):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Unsupported content type: {file.content_type}")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")

    img_array = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "Could not decode image. Supported formats: PNG, JPEG, TIFF")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    model = ml_models["segmenter"]
    device = ml_models["device"]

    processed = preprocess(img_rgb)
    mask = infer(model, processed, device)

    MIN_AREA = 120
    features = extract_features(mask, min_area=MIN_AREA)

    result = {
        "filename": file.filename,
        "image_shape": {"height": img_rgb.shape[0], "width": img_rgb.shape[1], "channels": img_rgb.shape[2]},
        "features": {
            "cell_count": features["cell_count"],
            "avg_nucleus_area_px": round(features["avg_nucleus_area_px"], 2),
            "total_nucleus_area_px": round(features["total_nucleus_area_px"], 2),
            "coverage_ratio": round(features["coverage_ratio"], 4),
        },
    }

    if return_annotated:
        annotated = draw_contours(img_rgb, mask, min_area=MIN_AREA)
        _, buffer = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        result["annotated_image_base64"] = base64_encode(buffer)
        _, orig_buffer = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        result["original_image_base64"] = base64_encode(orig_buffer)

    return JSONResponse(result)


STATIC_DIR = PROJECT_ROOT / "app" / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(
            str(STATIC_DIR / "index.html"),
            media_type="text/html",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
