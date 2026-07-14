"""
BioVision-CellSegmenter — API Evaluation Script

Validates all API endpoints and pipeline correctness.
Run with: python tests/test_api.py [--host HOST] [--port PORT]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://{}:{}"

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "dsb2018" / "test" / "images"

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m– SKIP\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"


def report(name: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    detail_str = f"  ({detail})" if detail else ""
    print(f"  {tag}  {name}{detail_str}")
    return ok


def main(host: str = "127.0.0.1", port: int = 8000):
    base = BASE_URL.format(host, port)
    print(f"\n{BOLD}BioVision-CellSegmenter — API Evaluation{RESET}")
    print(f"{'='*60}")
    print(f"  Target: {base}")
    print()

    total = 0
    passed = 0

    # ── 1. Health Check ──────────────────────────────────
    print(f" {BOLD}[1] Health Endpoint{RESET}")
    total += 1
    try:
        t0 = time.time()
        r = requests.get(f"{base}/health", timeout=10)
        dt = time.time() - t0
        data = r.json()
        ok = r.status_code == 200 and data.get("status") == "ok"
        if report("GET /health", ok, f"{r.status_code} ({dt:.2f}s)"):
            passed += 1
            print(f"       device: {data.get('device', 'unknown')}")
    except requests.RequestException as e:
        report("GET /health", False, str(e))
    print()

    # ── 2. Static Frontend ───────────────────────────────
    total += 1
    print(f" {BOLD}[2] Frontend Serving{RESET}")
    try:
        r = requests.get(f"{base}/", timeout=10)
        ok = r.status_code == 200 and "text/html" in r.headers.get("content-type", "")
        if report("GET /", ok, f"{r.status_code} ({len(r.content)} bytes)"):
            passed += 1
    except requests.RequestException as e:
        report("GET /", False, str(e))
    print()

    # ── 3. Sample Image Endpoint ─────────────────────────
    total += 1
    print(f" {BOLD}[3] Sample Image{RESET}")
    try:
        r = requests.get(f"{base}/sample", timeout=15)
        sample_bytes = r.content
        ok = r.status_code == 200 and len(sample_bytes) > 1000
        if report("GET /sample", ok, f"{r.status_code} ({len(sample_bytes)} bytes)"):
            passed += 1
            sample_type = r.headers.get("content-type", "unknown")
            print(f"       type: {sample_type}")
    except requests.RequestException as e:
        report("GET /sample", False, str(e))
        sample_bytes = None
    print()

    # ── 4. Analyze (inference) ───────────────────────────
    total += 1
    print(f" {BOLD}[4] Analyze Endpoint (Inference){RESET}")

    # Use a local test image if available, otherwise fetch sample
    if sample_bytes is None:
        report("POST /analyze", False, "no sample image available")
    else:
        try:
            files = {"file": ("sample.tif", sample_bytes, "image/tiff")}
            params = {"return_annotated": "true"}
            t0 = time.time()
            r = requests.post(f"{base}/analyze", files=files, params=params, timeout=60)
            dt = time.time() - t0
            data = r.json()

            ok = r.status_code == 200
            report("POST /analyze", ok, f"{r.status_code} ({dt:.2f}s)")
            if ok:
                passed += 1
                fc = data.get("features", {})
                print(f"       filename: {data.get('filename', 'N/A')}")
                print(f"       shape: {data.get('image_shape', 'N/A')}")
                print(f"       cells: {fc.get('cell_count', '?')}")
                print(f"       avg area: {fc.get('avg_nucleus_area_px', '?')} px²")
                print(f"       coverage: {fc.get('coverage_ratio', '?')*100 if fc.get('coverage_ratio') else '?'}%")
                has_b64 = bool(data.get("annotated_image_base64"))
                print(f"       annotated: {'✓' if has_b64 else '✗'} ({len(data.get('annotated_image_base64',''))} chars base64)")
            else:
                print(f"       error: {data.get('detail', r.text[:200])}")
        except requests.RequestException as e:
            report("POST /analyze", False, str(e))
    print()

    # ── 5. Response Schema Validation ────────────────────
    total += 1
    print(f" {BOLD}[5] Response Schema Validation{RESET}")
    if sample_bytes is None:
        report("Schema", False, "no sample available")
    else:
        try:
            files = {"file": ("sample.tif", sample_bytes, "image/tiff")}
            r = requests.post(f"{base}/analyze", files=files, params={"return_annotated": "true"}, timeout=60)
            if r.status_code != 200:
                report("Schema", False, f"HTTP {r.status_code}")
            else:
                data = r.json()
                checks = [
                    ("filename (str)", isinstance(data.get("filename"), str)),
                    ("image_shape (dict)", isinstance(data.get("image_shape"), dict)),
                    ("features (dict)", isinstance(data.get("features"), dict)),
                    ("cell_count (int)", isinstance(data.get("features", {}).get("cell_count"), int)),
                    ("avg_nucleus_area_px (float)", isinstance(data.get("features", {}).get("avg_nucleus_area_px"), (int, float))),
                    ("coverage_ratio (float)", isinstance(data.get("features", {}).get("coverage_ratio"), (int, float))),
                    ("annotated_image_base64 (str, non-empty)", bool(data.get("annotated_image_base64"))),
                ]
                all_ok = all(c[1] for c in checks)
                if report("Schema", all_ok, f"{sum(1 for c in checks if c[1])}/{len(checks)} checks passed"):
                    passed += 1
                    for name, ok in checks:
                        print(f"       {'✓' if ok else '✗'} {name}")
        except requests.RequestException as e:
            report("Schema", False, str(e))
    print()

    # ── 6. Batch / Pipeline Test ─────────────────────────
    total += 1
    print(f" {BOLD}[6] Batch Test (local images){RESET}")
    if SAMPLE_IMAGES_DIR.is_dir():
        images = sorted(SAMPLE_IMAGES_DIR.glob("*.tif"))[:5]
        if not images:
            report("Batch", False, "no local images found")
        else:
            success = 0
            times = []
            for img_path in images:
                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    files = {"file": (img_path.name, img_bytes, "image/tiff")}
                    t0 = time.time()
                    r = requests.post(f"{base}/analyze", files=files, timeout=60)
                    dt = time.time() - t0
                    times.append(dt)
                    if r.status_code == 200 and r.json().get("features", {}).get("cell_count", -1) >= 0:
                        success += 1
                except requests.RequestException:
                    pass

            all_ok = success == len(images)
            avg_t = sum(times) / len(times) if times else 0
            if report("Batch", all_ok, f"{success}/{len(images)} passed, avg {avg_t:.2f}s"):
                passed += 1
    else:
        report("Batch", False, f"test images not found at {SAMPLE_IMAGES_DIR}")
    print()

    # ── Summary ──────────────────────────────────────────
    print(f"{'='*60}")
    pct = (passed / total) * 100 if total else 0
    print(f"  {BOLD}Result: {passed}/{total} tests passed ({pct:.0f}%){RESET}")
    if passed == total:
        print(f"  {PASS} All checks passed — pipeline is operational.")
    else:
        print(f"  {FAIL} Some checks failed — review details above.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--host" and i + 1 < len(args):
            host = args[i + 1]
        elif arg == "--port" and i + 1 < len(args):
            port = int(args[i + 1])

    sys.exit(main(host, port))
