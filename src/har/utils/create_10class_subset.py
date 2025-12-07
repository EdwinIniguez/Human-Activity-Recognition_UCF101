"""
Create a filtered subset pickle containing only the selected 10 classes.

This script will:
- locate the original `ucf101_2d.pkl` (tries a few likely paths)
- read `selected_10_classes.csv` (labels to keep)
- filter and remap labels to 0..9 (order preserved as in CSV)
- write a new pickle `Dataset/ucf101_2d_10cls.pkl` (or root if Dataset/ missing)
- write `annotations_summary_10cls.csv` and `label_mapping_10cls.json`

Usage (PowerShell):
python scripts\create_10class_subset.py

"""
import os
import sys
import csv
import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
PICKLE_PATH = DATASET_DIR / "2d-skels" / "ucf101_2d.pkl"
SELECTED_CSV = ROOT / "selected_10_classes.csv"

def read_selected_labels(csv_path):
    labels = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            labels.append(int(r["label"]))
    return labels


def save_summary(anns, out_csv_path):
    # produce a small summary CSV similar to annotations_summary.csv
    cols = ["frame_dir", "label", "total_frames", "img_shape", "keypoint_shape"]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for a in anns:
            frame_dir = a.get("frame_dir", "")
            label = a.get("label", "")
            total_frames = a.get("total_frames", "")
            img_shape = a.get("img_shape", "")
            kp = a.get("keypoint")
            kp_shape = None
            try:
                if kp is not None:
                    # kp shape could be (M,T,V,C) or similar; don't expand large arrays
                    kp_shape = getattr(kp, "shape", None) or (len(kp),)
            except Exception:
                kp_shape = "?"
            writer.writerow([frame_dir, label, total_frames, img_shape, kp_shape])


def main():
    print("Starting 10-class subset creation...")

    if not SELECTED_CSV.exists():
        print(f"Error: selected classes CSV not found at {SELECTED_CSV}")
        sys.exit(1)

    selected = read_selected_labels(SELECTED_CSV)
    print(f"Loaded {len(selected)} selected labels: {selected}")

    pkl = PICKLE_PATH
    if pkl is None:
        print("Could not find `ucf101_2d.pkl` in common locations. Please place it in the repo root or in Dataset/.")
        sys.exit(1)

    print(f"Loading original pickle from: {pkl}")
    with open(pkl, "rb") as f:
        data = pickle.load(f)

    anns = data.get("annotations") or data.get("annotations_list") or []
    if not anns:
        # try keys
        # if data is a dict mapping split->list
        if isinstance(data, dict):
            # try common keys
            for k in ["annotations", "train", "val", "test", "all"]:
                if k in data and isinstance(data[k], list):
                    anns = data[k]
                    break

    if not anns:
        print("No annotations list found in pickle. Inspect the pickle structure manually.")
        sys.exit(1)

    print(f"Total annotations found: {len(anns)}")

    selected_set = set(selected)
    mapping = {old: new for new, old in enumerate(selected)}
    filtered = []
    for a in anns:
        lab = a.get("label")
        if lab in selected_set:
            new_a = dict(a)
            new_a["label"] = mapping[lab]
            filtered.append(new_a)

    print(f"Filtered annotations count: {len(filtered)}")

    out_pickle_dir = DATASET_DIR if DATASET_DIR.exists() else ROOT
    out_pickle = out_pickle_dir / "ucf101_2d_10cls.pkl"
    out_summary = out_pickle_dir / "annotations_summary_10cls.csv"
    mapping_file = out_pickle_dir / "label_mapping_10cls.json"

    out_data = dict(data)  # shallow copy
    # replace annotations wherever they were stored; prefer original key if present
    if "annotations" in out_data:
        out_data["annotations"] = filtered
    else:
        # add a top-level annotations key
        out_data["annotations"] = filtered

    with open(out_pickle, "wb") as f:
        pickle.dump(out_data, f)
    print(f"Saved filtered pickle to: {out_pickle}")

    save_summary(filtered, out_summary)
    print(f"Saved summary CSV to: {out_summary}")

    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved label mapping to: {mapping_file}")

    print("Done. Next: use `Dataset/ucf101_2d_10cls.pkl` in your notebook or training script, set labels to 0..9.")


if __name__ == "__main__":
    main()
