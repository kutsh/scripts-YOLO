from ultralytics import YOLO
import argparse
from PIL import Image
from pathlib import Path
from utils import list_files_recursive
from typing import Any
from tqdm import tqdm
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("--input-dir", type=Path, help="path to input image")
args.add_argument("--output-dir", type=Path, help="path to save output images")
args.add_argument("--model-dir", type=Path, default="runs/segment/train-24", help="path to the model directory")
args = args.parse_args()

# Load a model
model = YOLO(args.model_dir / "weights/best.pt")


def parse_args(args: dict[str, Any]):
    input_files = list_files_recursive(args.input_dir, [".png", ".jpg", ".jpeg"])
    return input_files, args.input_dir.resolve(), args.output_dir.resolve()

def predict_batch(input_files, input_dir, output_dir):
    for file in tqdm(input_files):
        try:
            results = model(file)

            try:
                masks = results[0].masks.data.cpu().numpy()
            except:
                continue

            path = output_dir / file.resolve().relative_to(input_dir)
            path = path.parent
            path.mkdir(parents=True, exist_ok=True)

            for i in range(len(masks)):
                individual_mask = masks[i]
                if np.sum(individual_mask) != 0:
                    individual_mask = Image.fromarray(individual_mask)
                    individual_mask.save(path / (f"mask_{i}_" + file.name))
            results[0].save(path / ("overlay_" + file.name))
        except:
            continue

if __name__ == '__main__':
    predict_batch(*parse_args(args))
