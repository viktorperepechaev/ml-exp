"""Validation-only OptDigits experiment. The writer-independent test is not read.

Data: UCI original 32x32 binary digits, official training/validation partitions.
Representation: pooled pixels plus Euler curves or matched geometric controls.
Run from the shared workspace using work/real-env/bin/python.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy import ndimage
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/real_data"
OUT = ROOT / "outputs/real_topology"


def read_original(partition: str) -> tuple[np.ndarray, np.ndarray]:
    if partition not in {"tra", "cv"}:
        raise ValueError("Pilot is restricted to official training and validation.")
    with zipfile.ZipFile(DATA / "optdigits.zip") as z:
        compressed = z.read(f"optdigits-orig.{partition}.Z")
    raw = subprocess.run(["gzip", "-dc"], input=compressed, capture_output=True, check=True).stdout.decode("ascii")
    lines = [s.strip() for s in raw.splitlines()]
    images, labels = [], []
    i = 0
    while i < len(lines):
        if len(lines[i]) == 32 and set(lines[i]) <= {"0", "1"}:
            rows = lines[i:i + 32]
            if len(rows) != 32 or not all(len(s) == 32 and set(s) <= {"0", "1"} for s in rows):
                raise ValueError("Malformed 32x32 image")
            label = lines[i + 32]
            if len(label) != 1 or not label.isdigit():
                raise ValueError("Malformed digit label")
            images.append(np.frombuffer("".join(rows).encode(), dtype=np.uint8).reshape(32, 32) == ord("1"))
            labels.append(int(label))
            i += 33
        else:
            i += 1
    expected = {"tra": 1934, "cv": 946}[partition]
    if len(labels) != expected:
        raise ValueError(f"Expected {expected} records; got {len(labels)}")
    return np.stack(images), np.array(labels)


def cells(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euler characteristic of the union of closed foreground pixel squares."""
    count = mask.sum(axis=(1, 2))
    # Unique edges and vertices, including the image boundary.
    pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)))
    horizontal = np.logical_or(pad[:, :-1, 1:-1], pad[:, 1:, 1:-1]).sum(axis=(1, 2))
    vertical = np.logical_or(pad[:, 1:-1, :-1], pad[:, 1:-1, 1:]).sum(axis=(1, 2))
    vertices = (pad[:, :-1, :-1] | pad[:, 1:, :-1] | pad[:, :-1, 1:] | pad[:, 1:, 1:]).sum(axis=(1, 2))
    euler = vertices - horizontal - vertical + count
    perimeter = 4 * count - 2 * ((mask[:, :-1] & mask[:, 1:]).sum(axis=(1, 2)) + (mask[:, :, :-1] & mask[:, :, 1:]).sum(axis=(1, 2)))
    return euler.astype(float), count.astype(float), perimeter.astype(float)


def gradient_histograms(images: np.ndarray) -> np.ndarray:
    # Matched 128-dimensional geometric control; ordinary unsigned cell gradients.
    # Filter each image independently: applying 3D Sobel would mix sample rows.
    gx = np.stack([ndimage.sobel(im.astype(float), axis=1, mode="constant") for im in images])
    gy = np.stack([ndimage.sobel(im.astype(float), axis=0, mode="constant") for im in images])
    magnitude = np.hypot(gx, gy)
    orientation = (np.arctan2(gy, gx) % np.pi) * (8 / np.pi)
    lower = np.floor(orientation).astype(int) % 8
    fraction = orientation - np.floor(orientation)
    histograms = []
    for row in range(4):
        for col in range(4):
            sl = (slice(None), slice(row * 8, (row + 1) * 8), slice(col * 8, (col + 1) * 8))
            hist = np.zeros((len(images), 8))
            sample_ids = np.broadcast_to(np.arange(len(images))[:, None, None], lower[sl].shape)
            np.add.at(hist, (sample_ids, lower[sl]), magnitude[sl] * (1 - fraction[sl]))
            np.add.at(hist, (sample_ids, (lower[sl] + 1) % 8), magnitude[sl] * fraction[sl])
            hist /= np.sqrt((hist * hist).sum(axis=1, keepdims=True) + 1e-6)
            histograms.append(hist)
    return np.column_stack(histograms)


def representations(images: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    start = time.perf_counter()
    pixels = images.reshape(-1, 8, 4, 8, 4).mean(axis=(2, 4)).reshape(len(images), -1)
    yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
    ect, area, perimeter = [], [], []
    for angle in np.arange(8) * (2 * np.pi / 8):
        height = xx * np.cos(angle) + yy * np.sin(angle)
        for threshold in np.linspace(height.min(), height.max(), 17)[1:]:
            chi, volume, border = cells(images & (height <= threshold + 1e-10))
            ect.append(chi)
            area.append(volume)
            perimeter.append(border)
    chi, volume, border = cells(images)
    components = np.array([ndimage.label(im, structure=np.ones((3, 3)))[1] for im in images])
    holes = components - chi
    top_global = np.stack([components, holes, chi], axis=1)
    intensity = np.stack([volume, border, images.mean(axis=1).std(axis=1), images.mean(axis=2).std(axis=1)], axis=1)
    arrays = {
        "pixels": pixels,
        "euler": np.stack(ect, axis=1),
        "area": np.stack(area, axis=1),
        "perimeter": np.stack(perimeter, axis=1),
        "global_topology": top_global,
        "global_geometry": intensity,
        "gradients": gradient_histograms(images),
    }
    return arrays, {"feature_seconds": time.perf_counter() - start, "n": len(images), "hole_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(holes, return_counts=True))}, "component_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(components, return_counts=True))}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    train_img, train_y = read_original("tra")
    valid_img, valid_y = read_original("cv")
    cache = DATA / "optdigits_trainval_features.npz"
    if cache.exists():
        with np.load(cache) as z:
            train = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
            valid = {k[3:]: z[k] for k in z.files if k.startswith("va_")}
        audit = json.loads((OUT / "optdigits_feature_audit.json").read_text())
    else:
        train, ta = representations(train_img)
        valid, va = representations(valid_img)
        audit = {"train": ta, "validation": va, "test_read": False, "foreground_convention": "closed pixel squares: 8-connected foreground and 4-connected background"}
        np.savez_compressed(cache, **{f"tr_{k}": v for k, v in train.items()}, **{f"va_{k}": v for k, v in valid.items()})
        (OUT / "optdigits_feature_audit.json").write_text(json.dumps(audit, indent=2))
    normalized_train, normalized_valid = {}, {}
    for key in train:
        scaler = StandardScaler().fit(train[key])
        normalized_train[key] = scaler.transform(train[key])
        normalized_valid[key] = scaler.transform(valid[key])
    rows = []
    def save():
        (OUT / "optdigits_validation_results.json").write_text(json.dumps({"protocol": "official tra=1934, cv=946; no writer-independent records read", "feature_audit": audit, "elapsed_seconds": time.perf_counter() - t0, "runs": rows}, indent=2))
    # Match directional feature count (128) and normalize every group from train.
    branches = ["none", "euler", "area", "perimeter", "gradients", "global_topology", "global_geometry"]
    for model in ("logistic", "rbf_svm"):
        for branch in branches:
            weights = [0.] if branch == "none" else ([0.1, 0.3, 1.0] if model == "rbf_svm" else [0.3, 1.0])
            for weight in weights:
                x = normalized_train["pixels"]
                v = normalized_valid["pixels"]
                if branch != "none":
                    x = np.column_stack([x, weight * normalized_train[branch]])
                    v = np.column_stack([v, weight * normalized_valid[branch]])
                for c in ([0.01, 0.1, 1., 10.] if model == "logistic" else [1., 10., 100.]):
                    gammas = [None] if model == "logistic" else [0.003, 0.01, 0.03]
                    for gamma in gammas:
                        start = time.perf_counter()
                        if model == "logistic":
                            clf = LogisticRegression(C=c, max_iter=1500, solver="lbfgs")
                        else:
                            clf = SVC(C=c, gamma=gamma)
                        clf.fit(x, train_y)
                        pred = clf.predict(v)
                        result = {"model": model, "branch": branch, "weight": weight, "C": c, "gamma": gamma, "features": x.shape[1], "accuracy": accuracy_score(valid_y, pred), "errors": int((pred != valid_y).sum()), "seconds": time.perf_counter() - start}
                        rows.append(result)
                        save()
            best = max([r for r in rows if r["model"] == model and r["branch"] == branch], key=lambda r:r["accuracy"])
            print(json.dumps(best), flush=True)
    save()


if __name__ == "__main__":
    main()
