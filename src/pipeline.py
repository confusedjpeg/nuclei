from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.measure import label, regionprops

import segmentation_models_pytorch as smp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "unet_nuclei.pth"


def load_model(device: torch.device | None = None) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(gray)
    enhanced_rgb = np.stack([enhanced] * 3, axis=-1)
    return enhanced_rgb.astype(np.float32) / 255.0


@torch.inference_mode()
def infer(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).byte()
    return pred.squeeze().cpu().numpy()


def draw_contours(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    canvas = original.copy()
    if canvas.ndim == 2 or canvas.shape[2] == 1:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 1)
    return canvas


def extract_features(mask: np.ndarray, pixel_area: float = 1.0) -> dict:
    labeled = label(mask, connectivity=2)
    props = regionprops(labeled)
    areas = [p.area * pixel_area for p in props]
    return {
        "cell_count": len(props),
        "avg_nucleus_area_px": float(np.mean(areas)) if areas else 0.0,
        "total_nucleus_area_px": float(np.sum(areas)),
        "image_area_px": mask.shape[0] * mask.shape[1],
        "coverage_ratio": float(np.sum(areas) / (mask.shape[0] * mask.shape[1])),
    }


def run_pipeline(
    image_path: str | Path,
    model: nn.Module | None = None,
    device: torch.device | None = None,
    return_annotated: bool = True,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(device)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    processed = preprocess(img_rgb)
    mask = infer(model, processed, device)

    features = extract_features(mask)

    result = {
        "image_path": str(image_path),
        "image_shape": list(img_rgb.shape),
        "features": features,
    }

    if return_annotated:
        annotated = draw_contours(img_rgb, mask)
        _, buffer = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        result["annotated_base64"] = base64_encode(buffer)

    return result


def base64_encode(buffer: np.ndarray) -> str:
    import base64
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


if __name__ == "__main__":
    test_dir = PROJECT_ROOT / "dsb2018" / "test" / "images"
    images = sorted(test_dir.glob("*.tif"))
    if not images:
        print("No test images found")
        exit(1)

    print(f"Found {len(images)} test images. Running pipeline on first image ...")
    device = torch.device("cpu")
    model = load_model(device)

    result = run_pipeline(images[0], model=model, device=device, return_annotated=True)
    print(f"Image: {result['image_path']}")
    print(f"Shape: {result['image_shape']}")
    print(f"Features:")
    for k, v in result["features"].items():
        print(f"  {k}: {v}")
    print(f"Annotated image base64 length: {len(result['annotated_base64'])} chars")
