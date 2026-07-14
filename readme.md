# BioVision-CellSegmenter

Automated nuclei segmentation and morphometric analysis for microscopy images. Built on a U-Net (ResNet34) trained on the DSB 2018 dataset.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in a browser.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Frontend UI |
| `/health` | GET | Server + model status |
| `/sample` | GET | Random test image (TIFF) |
| `/analyze` | POST | Upload image for nuclei segmentation |

### POST /analyze

**Parameters:**
- `file` — image file (PNG, JPEG, TIFF)
- `return_annotated` — bool, include base64-encoded annotated image (default: true)

**Response:**
```json
{
  "filename": "sample.tif",
  "image_shape": {"height": 256, "width": 256, "channels": 3},
  "features": {
    "cell_count": 47,
    "avg_nucleus_area_px": 142.32,
    "total_nucleus_area_px": 6689.04,
    "coverage_ratio": 0.1021
  },
  "annotated_image_base64": "iVBORw0KGgo..."
}
```

## Frontend

Single-page UI served at `/` with:
- Drag-and-drop image upload
- Original/annotated image comparison
- Metrics display (count, area, coverage)
- Base64 export with copy-to-clipboard
- "Run Sample Test" for quick verification
- Full diagnostic panel for pipeline validation

## Evaluation

```bash
python tests/test_api.py
```

Tests health, inference, response schema, and batch processing across 6 checks. Requires the server to be running.

## Project Structure

```
app/
  main.py          # FastAPI server + endpoints
  static/
    index.html     # Frontend UI
  __init__.py
src/
  pipeline.py      # Model loading, preprocessing, inference, feature extraction
  dataset.py       # PyTorch Dataset + DataLoader
  train.py         # Training loop
dsb2018/           # Dataset (test images)
models/
  unet_nuclei.pth  # Trained weights
tests/
  test_api.py      # API evaluation script
```

## Model

U-Net with ResNet34 encoder, trained on the DSB 2018 nuclei segmentation dataset (447 training images). Input: 256×256 RGB. Output: binary mask with sigmoid threshold at 0.5.

- Validation Dice: 0.920
- Validation IoU: 0.856
