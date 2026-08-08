import difflib
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

MODELS_DIR = "models"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

TOP_K = 5

# Zero-shot prompts for the "is this actually a car?" gate. CLIP scores the image
# against both groups, so no separate classifier model is needed.
VEHICLE_PROMPTS = [
    "a photo of a car",
    "a photo of a car parked outside",
    "a photo of the front of a car",
    "a photo of a van or SUV",
]
NON_VEHICLE_PROMPTS = [
    "a photo of an object that is not a vehicle",
    "a screenshot of a document or diagram",
    "a photo of a person",
    "a photo of a building or landscape",
    "a photo of tools or hardware",
]


@st.cache_resource
def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, processor, torch


@st.cache_resource
def load_reference_data():
    ref_embeddings = np.load(os.path.join(MODELS_DIR, "car_reference_embeddings.npy"))
    ref_classes = np.load(os.path.join(MODELS_DIR, "car_reference_classes.npy"))
    with open(os.path.join(MODELS_DIR, "car_label_map.json"), encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    return ref_embeddings, ref_classes, label_map


def _embed_image(image: Image.Image) -> np.ndarray:
    model, processor, torch = load_clip()
    inputs = processor(images=[image.convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        # transformers 5.x returns a pooled output object; its pooler_output is
        # already the projected CLIP image embedding.
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
    vec = feats[0].numpy()
    return vec / np.linalg.norm(vec)


@st.cache_data(show_spinner=False)
def _embed_prompts(prompts: tuple) -> np.ndarray:
    model, processor, torch = load_clip()
    inputs = processor(text=list(prompts), return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
    vecs = feats.numpy()
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def looks_like_vehicle(image: Image.Image) -> bool:
    """Zero-shot check that the photo actually contains a car, so we don't confidently
    name a car model for a photo of something else entirely."""
    return _vec_looks_like_vehicle(_embed_image(image))


def _vec_looks_like_vehicle(vec: np.ndarray) -> bool:
    veh = _embed_prompts(tuple(VEHICLE_PROMPTS)) @ vec
    non = _embed_prompts(tuple(NON_VEHICLE_PROMPTS)) @ vec
    return float(veh.max()) > float(non.max())


def recognize_from_images(images):
    """Recognize a car from several photos of it. Averaging the embeddings across angles
    is more robust than trusting any single shot, and photos that don't look like a car
    are ignored so a stray non-car image can't drag the result off.

    Returns (results, used_count) -- results is None when no photo looked like a car."""
    vecs = [_embed_image(img) for img in images]
    car_vecs = [v for v in vecs if _vec_looks_like_vehicle(v)]
    if not car_vecs:
        return None, 0

    mean_vec = np.mean(car_vecs, axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)
    return _match_vector(mean_vec), len(car_vecs)


def _match_vector(vec: np.ndarray):
    """Return the best-matching distinct (brand, model) candidates, best first.

    Deliberately returns several options rather than one: measured on held-out photos
    the single best guess is right ~50% of the time, but the correct car is somewhere
    in the top 3 candidates ~65% of the time -- so letting the user pick from a short
    list beats silently committing to one answer.

    (Weighted voting across neighbours was measured too and made things worse -- with
    only a handful of reference photos per model it pulls in wrong classes.)"""
    ref_embeddings, ref_classes, label_map = load_reference_data()
    sims = ref_embeddings @ vec

    results, seen = [], set()
    for i in np.argsort(sims)[::-1]:
        cls = int(ref_classes[i])
        if cls in seen:
            continue
        seen.add(cls)
        info = label_map[cls]
        results.append({"brand": info["company"], "model": info["model"], "similarity": float(sims[i])})
        if len(results) == TOP_K:
            break
    return results


def recognize_brand_model(image: Image.Image):
    """Recognize the car's brand + model from a photo via nearest-neighbour lookup
    against a reference set built from the brands in our own price dataset, using CLIP
    image embeddings. Good but not perfect -- results are shown as a suggestion the
    user can correct."""
    return _match_vector(_embed_image(image))


def match_to_price_dataset(brand: str, model: str, df: pd.DataFrame):
    """Map a recognized brand/model onto entries that actually exist in our price dataset,
    so the UI can auto-select real dropdown options. The reference set is keyed by our own
    company names, so the brand is an exact match -- only the model needs fuzzy matching
    against that brand's model names."""
    companies = df["company"].unique().tolist()
    if brand not in companies:
        lower_map = {c.lower(): c for c in companies}
        if brand.lower() not in lower_map:
            return None
        brand = lower_map[brand.lower()]

    candidate_names = df[df["company"] == brand]["name"].unique().tolist()
    name_matches = difflib.get_close_matches(model, candidate_names, n=1, cutoff=0.45)
    matched_name = name_matches[0] if name_matches else None

    return {"company": brand, "name": matched_name}
