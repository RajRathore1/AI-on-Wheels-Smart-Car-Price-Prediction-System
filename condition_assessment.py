import os

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = os.path.join("models", "damage_yolov8n.pt")

# Higher weight = more impact on price per detected instance.
DAMAGE_SEVERITY = {
    "Broken part": 1.0,
    "Missing part": 1.0,
    "Cracked": 0.9,
    "Corrosion": 0.8,
    "Dent": 0.6,
    "Scratch": 0.4,
    "Paint chip": 0.3,
    "Flaking": 0.3,
}

POINTS_PER_DETECTION = 15
CONFIDENCE_THRESHOLD = 0.35
MIN_PRICE_MULTIPLIER = 0.6


@st.cache_resource
def load_damage_model():
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)


def assess_condition(image: Image.Image):
    """Run damage detection on a PIL image and return detections + a condition score."""
    model = load_damage_model()
    results = model.predict(np.array(image.convert("RGB")), conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_name = results.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        detections.append({"class": cls_name, "confidence": confidence, "bbox": xyxy})

    damage_score = min(100, sum(DAMAGE_SEVERITY.get(d["class"], 0.5) * POINTS_PER_DETECTION for d in detections))
    condition_score = round(100 - damage_score)
    price_multiplier = MIN_PRICE_MULTIPLIER + (1 - MIN_PRICE_MULTIPLIER) * (condition_score / 100)

    annotated = draw_detections(image, detections)

    return {
        "detections": detections,
        "condition_score": condition_score,
        "price_multiplier": round(price_multiplier, 3),
        "annotated_image": annotated,
    }


def draw_detections(image: Image.Image, detections):
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f"{d['class']} {d['confidence']:.0%}"
        draw.rectangle([x1, y1, x2, y2], outline="#ff3860", width=3)
        draw.rectangle([x1, max(0, y1 - 18), x1 + 8 * len(label), y1], fill="#ff3860")
        draw.text((x1 + 2, max(0, y1 - 17)), label, fill="white")
    return annotated


def condition_label(condition_score: int) -> str:
    if condition_score >= 90:
        return "Excellent"
    if condition_score >= 75:
        return "Good"
    if condition_score >= 55:
        return "Fair"
    return "Poor"
