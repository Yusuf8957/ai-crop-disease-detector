from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from treatment_info import treatment_info

try:
    import cv2
except ImportError:
    cv2 = None


CROP_TYPE_LABELS = {0: "other", 1: "pepper", 2: "potato", 3: "tomato"}
SUPPORTED_CROPS_TEXT = "Tomato, Potato, or Bell Pepper"


@dataclass(frozen=True)
class DiagnosisResult:
    is_supported_crop: bool
    predicted_crop: str
    predicted_class: str | None = None
    disease_name: str | None = None
    confidence: float | None = None
    pesticide: str | None = None
    treatment: str | None = None
    precaution: str | None = None
    severity_pct: float | None = None
    severity_label: str | None = None
    severity_advice: str | None = None
    message: str = ""
    top_predictions: list[tuple[str, str, float]] | None = None


@lru_cache(maxsize=1)
def load_diagnosis_assets() -> tuple[Any, dict[str, str], Any]:
    disease_model = load_model("model/crop_disease_model.h5")
    with open("model/class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    crop_type_classifier = load_model("model/crop_type_classifier.h5")
    return disease_model, class_names, crop_type_classifier


def prepare_model_input(img: Image.Image) -> np.ndarray:
    img_resized = img.convert("RGB").resize((224, 224))
    return np.expand_dims(np.array(img_resized) / 255.0, axis=0)


def calculate_severity(img_pil: Image.Image) -> float | None:
    if cv2 is None:
        return None

    img = np.array(img_pil.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_brown = np.array([5, 30, 20])
    upper_brown = np.array([30, 255, 220])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    leaf_mask = cv2.bitwise_or(green_mask, brown_mask)
    leaf_area = int(np.count_nonzero(leaf_mask))
    diseased_area = int(np.count_nonzero(brown_mask))

    if leaf_area == 0:
        return 0.0

    severity_pct = (diseased_area / leaf_area) * 100
    return round(min(severity_pct, 100.0), 1)


def severity_details(severity_pct: float | None) -> tuple[str | None, str | None]:
    if severity_pct is None:
        return None, None
    if severity_pct < 25:
        return "Mild", "Monitor the plant; no urgent action needed."
    if severity_pct < 60:
        return "Moderate", "Spray the recommended pesticide soon."
    return "Severe", "Spray immediately; infection is spreading fast."


def detect_crop(img_array: np.ndarray, crop_type_classifier: Any) -> str:
    crop_prediction = crop_type_classifier.predict(img_array, verbose=0)[0]
    other_confidence = crop_prediction[0]

    if other_confidence > 0.60:
        return "other"

    supported_index = int(np.argmax(crop_prediction[1:]) + 1)
    return CROP_TYPE_LABELS.get(supported_index, "other")


def diagnose_image(img: Image.Image, include_top_predictions: bool = True) -> DiagnosisResult:
    disease_model, class_names, crop_type_classifier = load_diagnosis_assets()
    img = img.convert("RGB")
    img_array = prepare_model_input(img)

    predicted_crop = detect_crop(img_array, crop_type_classifier)
    if predicted_crop == "other":
        return DiagnosisResult(
            is_supported_crop=False,
            predicted_crop="other",
            message=(
                f"This does not look like a supported crop leaf ({SUPPORTED_CROPS_TEXT}). "
                "Please upload a clear leaf photo."
            ),
        )

    prediction = disease_model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[str(predicted_index)]
    confidence = float(prediction[predicted_index] * 100)

    info = treatment_info.get(predicted_class, {})
    severity_pct = calculate_severity(img)
    severity_label, severity_advice = severity_details(severity_pct)

    top_predictions = None
    if include_top_predictions:
        top_indices = np.argsort(prediction)[-3:][::-1]
        top_predictions = []
        for i in top_indices:
            class_key = class_names[str(int(i))]
            display_name = treatment_info.get(class_key, {}).get("name", class_key)
            top_predictions.append((class_key, display_name, float(prediction[i] * 100)))

    return DiagnosisResult(
        is_supported_crop=True,
        predicted_crop=predicted_crop,
        predicted_class=predicted_class,
        disease_name=info.get("name", predicted_class),
        confidence=confidence,
        pesticide=info.get("pesticide", "Not available"),
        treatment=info.get("treatment", "Treatment information not available."),
        precaution=info.get("precaution", "No specific precaution listed."),
        severity_pct=severity_pct,
        severity_label=severity_label,
        severity_advice=severity_advice,
        top_predictions=top_predictions,
    )


def format_diagnosis_text(result: DiagnosisResult) -> str:
    if not result.is_supported_crop:
        return result.message

    severity_line = "Severity: Not available"
    if result.severity_pct is not None:
        severity_line = (
            f"Severity: {result.severity_pct}%"
            f" ({result.severity_label}) - {result.severity_advice}"
        )

    confidence_note = ""
    if result.confidence is not None and result.confidence < 60:
        confidence_note = "\n\nNote: Confidence is low. Try a clearer photo."

    return (
        f"Disease: {result.disease_name}\n"
        f"Crop: {result.predicted_crop.title()}\n"
        f"Confidence: {result.confidence:.1f}%\n"
        f"{severity_line}\n\n"
        f"Pesticide: {result.pesticide}\n\n"
        f"Treatment: {result.treatment}\n\n"
        f"Precaution: {result.precaution}"
        f"{confidence_note}"
    )

