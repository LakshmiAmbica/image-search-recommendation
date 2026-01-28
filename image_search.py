# image_recommender.py

import streamlit as st
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# -----------------------------
# Device setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load CSV
# -----------------------------
CSV_PATH = "data/test_products.csv"
df = pd.read_csv(CSV_PATH)

# Clean columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip().str.lower()

# -----------------------------
# Load images safely
# -----------------------------
@st.cache_data
def load_image(url):
    try:
        r = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img
    except:
        return None

df["image_obj"] = df["imageurl"].apply(load_image)

# 🚨 Remove products whose images failed
df = df[df["image_obj"].notnull()].reset_index(drop=True)

st.write("✅ Valid products:", len(df))

# -----------------------------
# Load CLIP model
# -----------------------------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -----------------------------
# Image embeddings
# -----------------------------
@st.cache_data
def get_image_embeddings(images):
    embeddings = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy()[0])
    return np.array(embeddings)

# -----------------------------
# Text embeddings (SHORT TEXT ONLY)
# -----------------------------
@st.cache_data
def get_text_embeddings(texts):
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,   # 🔥 IMPORTANT
        max_length=77
    ).to(device)

    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

    return emb.cpu().numpy()

# -----------------------------
# Prepare text data (NO DESCRIPTION)
# -----------------------------
df["text_data"] = (
    df["name"].fillna("") + " " +
    df["category"].fillna("") + " " +
    df["brand"].fillna("")
)

st.write("⏳ Computing embeddings...")
image_embeddings = get_image_embeddings(df["image_obj"])
text_embeddings = get_text_embeddings(df["text_data"].tolist())
st.write("✅ Embeddings ready!")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🖼️ Image-Based Product Recommendation")

uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Uploaded Image", width=200)

    # Query image embedding
    inputs = processor(images=query_img, return_tensors="pt").to(device)
    with torch.no_grad():
        q_img_emb = model.get_image_features(**inputs)
        q_img_emb = q_img_emb / q_img_emb.norm(p=2, dim=-1, keepdim=True)
    q_img_emb = q_img_emb.cpu().numpy()

    # Image similarity
    img_similarity = image_embeddings @ q_img_emb.T

    # Optional text hint (VERY helpful)
    hint = st.text_input("Optional hint (e.g. shoe, watch, makeup)")

    if hint.strip():
        q_text_emb = get_text_embeddings([hint])
        txt_similarity = text_embeddings @ q_text_emb.T

        # Combine image + text similarity
        final_score = 0.6 * img_similarity + 0.4 * txt_similarity
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
