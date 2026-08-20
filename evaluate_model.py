"""Generate a real, machine-readable evaluation report for Crop Disease AI.

Expected dataset layout:

evaluation_dataset/
  Tomato___Early_blight/
    image1.jpg
  Tomato___Late_blight/
    image2.jpg
  ...

Folder names must match the values in model/class_names.json.
Run:
    python evaluate_model.py --data evaluation_dataset

Outputs:
  model_performance/evaluation_metrics.json
  model_performance/classification_report.csv
  model_performance/confusion_matrix.csv

The Streamlit app reads evaluation_metrics.json and never invents metrics.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import load_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classes(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data[str(i)] for i in range(len(data))]
    if isinstance(data, list):
        return data
    raise ValueError("class_names.json must contain a list or numeric-key dictionary")


def collect_dataset(root: Path, classes):
    rows = []
    class_set = set(classes)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in class_set:
            print(f"[WARN] Skipping folder not present in class_names.json: {folder.name}")
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append((path, folder.name))
    if not rows:
        raise ValueError(f"No labeled images found under {root}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Labeled test/evaluation dataset root")
    parser.add_argument("--model", default="model/crop_disease_model.h5")
    parser.add_argument("--classes", default="model/class_names.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_root = Path(args.data)
    classes_path = Path(args.classes)
    out_dir = Path("model_performance")
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(classes_path)
    rows = collect_dataset(data_root, classes)

    model = load_model(args.model, compile=False)

    y_true = []
    y_pred = []
    image_paths = []

    batch_images = []
    batch_labels = []
    batch_paths = []

    def flush():
        if not batch_images:
            return
        x = np.asarray(batch_images, dtype=np.float32)
        preds = model.predict(x, verbose=0)
        pred_indices = np.argmax(preds, axis=1)
        for true_label, pred_idx, image_path in zip(batch_labels, pred_indices, batch_paths):
            y_true.append(true_label)
            y_pred.append(classes[int(pred_idx)])
            image_paths.append(str(image_path))
        batch_images.clear()
        batch_labels.clear()
        batch_paths.clear()

    for path, label in rows:
        try:
            image = Image.open(path).convert("RGB").resize((224, 224))
            batch_images.append(np.asarray(image) / 255.0)
            batch_labels.append(label)
            batch_paths.append(path)
            if len(batch_images) >= args.batch_size:
                flush()
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")

    flush()

    labels_for_report = classes
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels_for_report, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels_for_report, average="weighted", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_for_report,
        target_names=labels_for_report,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for label in labels_for_report:
        values = report.get(label, {})
        per_class[label] = {
            "precision": float(values.get("precision", 0.0)),
            "recall": float(values.get("recall", 0.0)),
            "f1": float(values.get("f1-score", 0.0)),
            "support": int(values.get("support", 0)),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels_for_report)
    pd.DataFrame(cm, index=labels_for_report, columns=labels_for_report).to_csv(
        out_dir / "confusion_matrix.csv"
    )

    report_df = pd.DataFrame(per_class).T.reset_index().rename(columns={"index": "class"})
    report_df.to_csv(out_dir / "classification_report.csv", index=False)

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "dataset_samples": int(len(y_true)),
        "classes": classes,
        "per_class": per_class,
        "dataset_path": str(data_root),
        "model_path": args.model,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
    print(f"Images: {len(y_true)}")
    print(f"Saved: {out_dir / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
