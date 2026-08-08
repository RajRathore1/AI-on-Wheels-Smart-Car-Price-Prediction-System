import difflib
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

MODELS_DIR = "models"

TOP_K = 5
MIN_MATCH_RATIO = 0.55

# Substrings matched against ImageNet category names to gate the fine-grained
# brand/model lookup -- avoids confidently "recognizing" a car in photos of
# unrelated objects, since nearest-neighbor search always returns *something*.
VEHICLE_KEYWORDS = [
    "car", "truck", "van", "bus", "jeep", "limousine", "ambulance", "cab",
    "wagon", "convertible", "racer", "trailer", "tow", "tractor", "moped",
    "golfcart", "go-kart", "minibus", "wheel", "garbage truck", "fire engine",
    "snowplow", "jinrikisha", "model t", "recreational vehicle", "school bus",
]


def _get_preprocess():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@st.cache_resource
def load_backbone():
    import torch
    from torchvision import models

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
    backbone = models.mobilenet_v2(weights=weights)
    backbone.classifier = torch.nn.Identity()
    backbone.eval()
    return backbone


@st.cache_resource
def load_classifier_backbone():
    from torchvision import models

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
    backbone = models.mobilenet_v2(weights=weights)
    backbone.eval()
    return backbone, weights.meta["categories"]


def looks_like_vehicle(image: Image.Image, top_k: int = 5) -> bool:
    """Cheap sanity check: does the general ImageNet classifier think this photo
    contains a vehicle at all? Prevents confidently guessing a car brand/model
    for photos of unrelated objects."""
    import torch

    backbone, categories = load_classifier_backbone()
    tensor = _get_preprocess()(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = backbone(tensor)
    top_indices = torch.topk(logits[0], top_k).indices.tolist()
    top_labels = [categories[i].lower() for i in top_indices]
    return any(keyword in label for label in top_labels for keyword in VEHICLE_KEYWORDS)


@st.cache_resource
def load_reference_data():
    import pickle

    with open(os.path.join(MODELS_DIR, "car_pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    ref_embeddings = np.load(os.path.join(MODELS_DIR, "car_reference_embeddings.npy"))
    ref_classes = np.load(os.path.join(MODELS_DIR, "car_reference_classes.npy"))
    with open(os.path.join(MODELS_DIR, "car_label_map.json"), encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    return pca, ref_embeddings, ref_classes, label_map


def _embed_image(image: Image.Image):
    import torch

    backbone = load_backbone()
    tensor = _get_preprocess()(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        vec = backbone(tensor).numpy()
    return vec


def recognize_brand_model(image: Image.Image):
    """Best-effort recognition of the car's brand + model from a photo via nearest-neighbor
    lookup against a reference set built from the brands in our own price dataset (a few
    images per model). Accuracy is modest (few-shot over ~1,500 classes) -- treat results
    as a starting suggestion, not a certain answer."""
    pca, ref_embeddings, ref_classes, label_map = load_reference_data()
    vec = _embed_image(image)
    reduced = pca.transform(vec)
    reduced = reduced / np.linalg.norm(reduced, axis=1, keepdims=True)

    sims = (reduced @ ref_embeddings.T)[0]
    top_idx = np.argsort(sims)[::-1][:TOP_K]

    results = []
    for i in top_idx:
        cls = int(ref_classes[i])
        info = label_map[cls]
        results.append({"brand": info["company"], "model": info["model"], "similarity": float(sims[i])})
    return results


def match_to_price_dataset(brand: str, model: str, df: pd.DataFrame):
    """Map a recognized brand/model onto entries that actually exist in our price dataset,
    so the UI can auto-select real dropdown options. The reference set is now keyed by our
    own company names, so the brand is an exact match -- only the model needs fuzzy
    matching against that brand's model names."""
    companies = df["company"].unique().tolist()
    if brand not in companies:
        # Reference set is built from our own brands, so this should be rare; fall back to
        # a case-insensitive lookup rather than guessing a different brand entirely.
        lower_map = {c.lower(): c for c in companies}
        if brand.lower() not in lower_map:
            return None
        brand = lower_map[brand.lower()]

    candidate_names = df[df["company"] == brand]["name"].unique().tolist()
    name_matches = difflib.get_close_matches(model, candidate_names, n=1, cutoff=0.45)
    matched_name = name_matches[0] if name_matches else None

    return {"company": brand, "name": matched_name}
