
#!/usr/bin/env python3
"""
Fine-tune a YOLO26 model with Ultralytics.

Defaults to the semantic-segmentation checkpoint (yolo26n-sem.pt) and task,
matching the PNG-mask dataset built earlier in this project. Pass --model /
--task to fine-tune a detection or instance-segmentation checkpoint instead.

Usage
-----
    python3 finetune.py --data /path/to/data.yaml
    python3 finetune.py --data /path/to/data.yaml --model yolo26s-sem.pt --epochs 100 --imgsz 1024

Note on multiprocessing (Python 3.14+, Linux default 'forkserver')
--------------------------------------------------------------------
The Ultralytics dataloader spawns worker processes. On Python 3.14 the default
start method changed from 'fork' to 'forkserver', which re-imports this file in
each worker. Because the training call lives inside `if __name__ == '__main__':`
below, this is already safe -- do not move `train_model(...)` to module level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics import settings

import mlflow

settings.update({"mlflow": True})


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="path to data.yaml describing the dataset")
    ap.add_argument("--model", type=str, default="yolo26n-sem.pt",
                    help="starting checkpoint or model yaml (default: yolo26n-sem.pt, "
                         "the semantic-segmentation nano checkpoint)")
    ap.add_argument("--task", type=str, default=None,
                    choices=["detect", "segment", "semantic", "classify", "pose", "obb"],
                    help="override task; by default Ultralytics infers it from --model")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="training image size; raise this (e.g. 1024) if the dataset "
                         "has thin features like boundary/cote lines that shrink away "
                         "at 640")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default=None,
                    help="e.g. '0', '0,1', 'cpu'; default lets Ultralytics choose")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=5,
                    help="early-stopping patience (epochs with no improvement)")
    ap.add_argument("--project", type=str, default=None,
                    help="output root directory (default: Ultralytics' own runs/ dir)")
    ap.add_argument("--name", type=str, default=None, help="run name subfolder")
    ap.add_argument("--resume", action="store_true",
                    help="resume training from the last checkpoint in --name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True,
                    help="enable automatic mixed precision (default on; Ultralytics "
                         "auto-disables it if its GPU sanity check fails)")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--freeze", type=int, default=None,
                    help="freeze the first N layers (useful for small fine-tuning sets)")
    ap.add_argument("--val", dest="val", action="store_true", default=True,
                    help="run validation each epoch (default on)")
    ap.add_argument("--no-val", dest="val", action="store_false")
    ap.add_argument("--exist-ok", action="store_true",
                    help="allow overwriting an existing run of the same --name")
    return ap.parse_args()

def log_total_losses(trainer):
    if getattr(trainer, "rank", -1) not in (-1, 0):
        return

    train_loss = sum(
        float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
        for v in trainer.tloss.values()
    )

    metrics = trainer.metrics or {}

    val_loss = sum(
        float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
        for k, v in metrics.items()
        if k.startswith("val/") and k.endswith("_loss")
    )

    mlflow.log_metrics(
        {
            "train/total_loss": train_loss,
            "val/total_loss": val_loss,
        },
        step=trainer.epoch,
    )


def train_model(args) -> Path:
    if not args.data.exists():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    model = YOLO(model=args.model, task=args.task)

    model.add_callback("on_fit_epoch_end", log_total_losses)

    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        amp=args.amp,
        val=args.val,
        resume=args.resume,
        exist_ok=args.exist_ok,
        multi_scale=0.25,
        dropout=0.2
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.project is not None:
        train_kwargs["project"] = args.project
    if args.name is not None:
        train_kwargs["name"] = args.name
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    results = model.train(**train_kwargs)

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest checkpoint: {best}")
    return best


if __name__ == "__main__":
    train_model(parse_args())
