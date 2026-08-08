import difflib
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

MODELS_DIR = "models"

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TOP_K = 5
MIN_MATCH_RATIO = 0.55


@st.cache_resource
def load_backbone():
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
    backbone = models.mobilenet_v2(weights=weights)
    backbone.classifier = torch.nn.Identity()
    backbone.eval()
    return backbone


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
    backbone = load_backbone()
    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        vec = backbone(tensor).numpy()
    return vec


def recognize_brand_model(image: Image.Image):
    """Best-effort recognition of the car's brand + model from a photo via nearest-neighbor
    lookup against a small reference set (few images per class), using a frozen ImageNet
    backbone. Accuracy is modest (few-shot, thousands of classes) -- treat results as a
    starting suggestion, not a certain answer."""
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
        results.append({"brand": info["brand"], "model": info["model"], "similarity": float(sims[i])})
    return results


def match_to_price_dataset(brand: str, model: str, df: pd.DataFrame):
    """Fuzzy-match a recognized brand/model to the closest entries actually present in our
    own price dataset, so the UI can auto-select real dropdown options."""
    companies = df["company"].unique().tolist()
    brand_matches = difflib.get_close_matches(brand.title(), companies, n=1, cutoff=MIN_MATCH_RATIO)
    if not brand_matches:
        # try case-insensitive exact match as a fallback
        lower_map = {c.lower(): c for c in companies}
        brand_matches = [lower_map[brand.lower()]] if brand.lower() in lower_map else []

    if not brand_matches:
        return None

    matched_company = brand_matches[0]
    candidate_names = df[df["company"] == matched_company]["name"].unique().tolist()
    name_matches = difflib.get_close_matches(f"{brand} {model}", candidate_names, n=1, cutoff=0.3)
    matched_name = name_matches[0] if name_matches else None

    return {"company": matched_company, "name": matched_name}
