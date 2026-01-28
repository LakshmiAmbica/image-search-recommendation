# image_recommender.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO

# -----------------------------
# Page config (FAST START)
# -----------------------------
st.set_page_config(page_title="Image Search Recommendation", layout="wide")

# -----------------------------
# Load CSV (lightweight)
# -----------------------------
CSV_PATH = "clean_data.csv"
df = pd.read_csv(CSV_PATH)

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.columns = df.columns.str.strip().str.lower()

df["text_data"] = (
    df["name"].fillna("") + " " +
    df["category"].fillna("") + " " +
    df["brand"].fillna("")
)

# -----------------------------
# UI
# -----------------------------
st.title("🖼️ Image-Based Product Recommendation")

if "prepared" not in st.session_state:
    st.session_state.prepared = False

# -----------------------------
# BUTTON → LOAD EVERYTHING HEAVY
# -----------------------------
if st.button("🚀 Prepare Recommendation Engine"):
    with st.spinner("Loading model, images, and computing embeddings..."):

        # 🔥 IMPORT HEAVY LIBRARIES HERE
        import torch
        from PIL import Image
        from transformers import CLIPProcessor, CLIPModel

        device = "cpu"

        # Load CLIP
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

        # Image loader
        def load_image(url):
            try:
                r = requests.get(url, timeout=5)
                img = Image.open(BytesIO(r.content)).convert("RGB")
                return img
            except:
                return None

        df["image_obj"] = df["imageurl"].apply(load_image)
        df = df[df["image_obj"].notnull()].reset_index(drop=True)

        # Image embeddings
        image_embeddings = []
        for img in df["image_obj"]:
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            image_embeddings.append(emb.cpu().numpy()[0])

        image_embeddings = np.array(image_embeddings)

        # Text embeddings
        inputs = processor(
            text=df["text_data"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        with torch.no_grad():
            text_embeddings = model.get_text_features(**inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        text_embeddings = text_embeddings.cpu().numpy()

        # Save everything
        st.session_state.df = df
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.image_embeddings = image_embeddings
        st.session_state.text_embeddings = text_embeddings
        st.session_state.prepared = True

    st.success(f"System ready with {len(df)} products!")

# -----------------------------
# STOP UNTIL READY
# -----------------------------
if not st.session_state.prepared:
    st.info("Click **Prepare Recommendation Engine** to start")
    st.stop()

# Restore
df = st.session_state.df
model = st.session_state.model
processor = st.session_state.processor
image_embeddings = st.session_state.image_embeddings
text_embeddings = st.session_state.text_embeddings

from PIL import Image
import torch

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, width=200)

    inputs = processor(images=query_img, return_tensors="pt")
    with torch.no_grad():
        q_img_emb = model.get_image_features(**inputs)
        q_img_emb = q_img_emb / q_img_emb.norm(p=2, dim=-1, keepdim=True)

    q_img_emb = q_img_emb.cpu().numpy()

    img_similarity = image_embeddings @ q_img_emb.T

    hint = st.text_input("Optional hint (e.g. shoe, watch, makeup)")

    if hint.strip():
        inputs = processor(
            text=[hint],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        with torch.no_grad():
            q_text_emb = model.get_text_features(**inputs)
            q_text_emb = q_text_emb / q_text_emb.norm(p=2, dim=-1, keepdim=True)

        q_text_emb = q_text_emb.cpu().numpy()
        final_score = 0.6 * img_similarity + 0.4 * (text_embeddings @ q_text_emb.T)
    else:
        final_score = img_similarity

    top_idx = final_score.squeeze().argsort()[::-1][:5]

    st.subheader("Recommended Products")
    for i in top_idx:
        prod = df.iloc[i]
        st.image(prod["image_obj"], width=150)
        st.write(f"**Name:** {prod.get('name','')}")
        st.write(f"**Category:** {prod.get('category','')}")
        st.write(f"**Brand:** {prod.get('brand','')}")
        st.markdown("---")


