import os

# Must be set before torch initialises CUDA: reduces fragmentation, which is a
# common cause of OOM in validation where mask tensor sizes vary per batch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

OOM_MARKERS = (
    "out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_NOT_INITIALIZED",  # cuDNN often reports OOM this way
)


def is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or any(m in str(err) for m in OOM_MARKERS)


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def validate(args):
    """Run Ultralytics validation. Returns (metrics, device, half) actually used."""
    val_kwargs = dict(data=str(args.data), conf=args.conf, workers=args.workers, plots=not args.no_plots)

    if torch.cuda.is_available():
        half = args.half
        gpu_name = torch.cuda.get_device_name(int(str(args.device).split(",")[0]))
        if half and "GTX 16" in gpu_name:
            print(f">>> {gpu_name} gives NaN outputs in FP16; using FP32 instead.")
            half = False
        batch = args.batch
        while batch >= 1:
            # Fresh model each attempt: val() fuses and casts the model in place.
            model = YOLO(args.weights)
            print(f"\n>>> GPU validation: device={args.device} batch={batch} half={half}")
            try:
                return model.val(device=args.device, batch=batch, half=half, **val_kwargs), args.device, half
            except RuntimeError as err:  # torch.cuda.OutOfMemoryError subclasses RuntimeError
                if not is_oom(err):
                    raise
            # Cleanup happens outside the except block so the traceback (which
            # holds references to GPU tensors) has already been released.
            del model
            free_gpu()
            print(f">>> CUDA out of memory at batch={batch}, retrying with batch={batch // 2}")
            batch //= 2
        print("\n>>> Out of memory even at batch=1.")
    else:
        print(">>> No CUDA device available.")

    if args.no_cpu_fallback:
        raise SystemExit("GPU validation failed and CPU fallback is disabled.")
    print(">>> Falling back to CPU (slow).")
    free_gpu()
    return YOLO(args.weights).val(device="cpu", batch=args.batch, **val_kwargs), "cpu", False


# ----------------------------------------------------------------------------- IoU


def list_val_images(data: dict) -> list[Path]:
    """Resolve the 'val' entry of a checked data.yaml into image paths (dirs, .txt lists or lists of either)."""
    specs = data["val"] if isinstance(data["val"], (list, tuple)) else [data["val"]]
    files = []
    for s in specs:
        p = Path(s)
        if p.is_dir():
            files += [f for f in p.rglob("*") if f.suffix[1:].lower() in IMG_FORMATS]
        elif p.suffix == ".txt":
            for line in p.read_text().splitlines():
                line = line.strip()
                if line:
                    f = Path(line)
                    files.append(f if f.is_absolute() else (p.parent / f).resolve())
        elif p.suffix[1:].lower() in IMG_FORMATS:
            files.append(p)
    return sorted(files)


def load_gt_masks(label_path: Path, h: int, w: int) -> dict[int, np.ndarray]:
    """Rasterise YOLO-seg labels into one binary mask per class (union of that class's instances)."""
    masks: dict[int, np.ndarray] = {}
    if not label_path.exists():
        return masks
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, coords = int(float(parts[0])), np.array(parts[1:], dtype=np.float64)
        m = masks.setdefault(c, np.zeros((h, w), np.uint8))
        if len(coords) == 4:  # box-only label: x_center y_center width height
            xc, yc, bw, bh = coords * [w, h, w, h]
            pts = np.array([[xc - bw / 2, yc - bh / 2], [xc + bw / 2, yc - bh / 2],
                            [xc + bw / 2, yc + bh / 2], [xc - bw / 2, yc + bh / 2]])
        else:  # polygon: x1 y1 x2 y2 ...
            pts = coords[: len(coords) // 2 * 2].reshape(-1, 2) * [w, h]
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return masks


def compute_iou(args, data: dict, device, half: bool):
    """Accumulate per-class intersection and union of predicted vs GT masks over the whole val set."""
    nc = len(data["names"])
    files = list_val_images(data)
    inter = np.zeros(nc, np.int64)
    union = np.zeros(nc, np.int64)
    tdev = torch.device("cpu") if device == "cpu" else torch.device(f"cuda:{str(device).split(',')[0]}")

    model = YOLO(args.weights)
    # retina_masks=True returns masks at the original image resolution, matching the GT rasterisation.
    results = model.predict(source=[str(f) for f in files], stream=True, batch=1, conf=args.iou_conf,
                            retina_masks=True, device=device, half=half, verbose=False)
    for r in results:
        h, w = r.orig_shape
        gt = load_gt_masks(Path(img2label_paths([r.path])[0]), h, w)
        if r.masks is not None and len(r.masks):
            pm = r.masks.data > 0.5
            pcls = r.boxes.cls.long()
            pred_classes = set(pcls.tolist())
        else:
            pm, pcls, pred_classes = None, None, set()
        for c in pred_classes | set(gt):
            p = pm[pcls == c].any(0) if c in pred_classes else torch.zeros((h, w), dtype=torch.bool, device=tdev)
            g = torch.from_numpy(gt[c]).to(tdev).bool() if c in gt else torch.zeros_like(p)
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
    return inter, union, len(files)


def run_iou(args, data, device, half):
    try:
        return compute_iou(args, data, device, half)
    except RuntimeError as err:
        if device == "cpu" or not is_oom(err):
            raise
    free_gpu()
    print(">>> CUDA out of memory during IoU pass, redoing it on CPU.")
    return compute_iou(args, data, "cpu", False)


# ----------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default="runs/segment/150000_seg_parcelle-5/weights/best.pt")
    parser.add_argument("--data", type=Path, default="synth-gen-test-2000/data.yaml", help="Path to the data file.")
    parser.add_argument(
        "--conf", type=float, default=0.001,
        help="Discard predictions below this confidence before computing P/R/mAP (Ultralytics val default: 0.001).",
    )
    parser.add_argument(
        "--iou-conf", type=float, default=0.25,
        help="Confidence threshold for the masks used in the IoU computation (Ultralytics predict default: 0.25).",
    )
    parser.add_argument("--device", default="0", help="CUDA device index, e.g. 0 or 1.")
    parser.add_argument("--batch", type=int, default=32, help="Starting batch size; halved on every OOM.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument(
        "--half", action="store_true",
        help="FP16 inference (less memory). Produces NaN/zero metrics on GTX 16xx cards, so ignored there.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip confusion matrix / PR curve plots.")
    parser.add_argument("--no-cpu-fallback", action="store_true", help="Exit instead of falling back to CPU.")
    args = parser.parse_args()

    metrics, device, half = validate(args)
    free_gpu()

    data = check_det_dataset(str(args.data))
    names = data["names"]
    print(f"\n>>> IoU pass on device={device} (conf >= {args.iou_conf})")
    inter, union, n_images = run_iou(args, data, device, half)

    print("\n=== Results ===")
    for name, m in (("Box ", metrics.box), ("Mask", metrics.seg)):
        print(
            f"{name} P {m.mp:.4f}  R {m.mr:.4f}  "
            f"mAP50-95 {m.map:.4f}  mAP50 {m.map50:.4f}  mAP75 {m.map75:.4f}"
        )
    fmt = lambda xs: "[" + ", ".join(f"{float(x):.4f}" for x in xs) + "]"
    print(f"Per-class box  P: {fmt(metrics.box.p)}  R: {fmt(metrics.box.r)}  mAP50-95: {fmt(metrics.box.maps)}")
    print(f"Per-class mask P: {fmt(metrics.seg.p)}  R: {fmt(metrics.seg.r)}  mAP50-95: {fmt(metrics.seg.maps)}")
    # Per-image dicts are large; available as metrics.box.image_metrics / metrics.seg.image_metrics

    print(f"\n=== Mask IoU vs GT ({n_images} images, predictions with conf >= {args.iou_conf}) ===")
    seen = union > 0
    for c in np.flatnonzero(seen):
        print(f"  {names[c]:<20} IoU {inter[c] / union[c]:.4f}")
    if seen.any():
        print(f"  {'mean over classes':<20} mIoU {(inter[seen] / union[seen]).mean():.4f}")
        print(f"  {'all pixels pooled':<20} IoU {inter.sum() / union.sum():.4f}")
    else:
        print("  No GT or predicted masks found.")
    return metrics


if __name__ == "__main__":
    main()
