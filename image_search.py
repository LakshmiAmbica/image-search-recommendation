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
# Device setup (CPU ONLY)
# -----------------------------
device = "cpu"

# -----------------------------
# Load CSV (ALL PRODUCTS)
# -----------------------------
CSV_PATH = "clean_data.csv"
df = pd.read_csv(CSV_PATH)

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.columns = df.columns.str.strip().str.lower()

# -----------------------------
# Load image safely
# -----------------------------
@st.cache_data
def load_image(url):
    try:
        r = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img
    except:
        return None

# -----------------------------
# Load CLIP (DELAYED)
# -----------------------------
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

# -----------------------------
# Embedding functions (UNCHANGED)
# -----------------------------
@st.cache_data
def get_image_embeddings(images, processor, model):
    embeddings = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy()[0])
    return np.array(embeddings)

@st.cache_data
def get_text_embeddings(texts, processor, model):
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)

    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

    return emb.cpu().numpy()

# -----------------------------
# Prepare text data
# -----------------------------
df["text_data"] = (
    df["name"].fillna("") + " " +
    df["category"].fillna("") + " " +
    df["brand"].fillna("")
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🖼️ Image-Based Product Recommendation")

# session state
if "prepared" not in st.session_state:
    st.session_state.prepared = False

# -----------------------------
# BUTTON (CRITICAL FIX)
# -----------------------------
if st.button("🚀 Prepare Recommendation Engine"):
    with st.spinner("Loading model, images, and computing embeddings..."):

        # Load CLIP ONLY here
        model, processor = load_clip()

        # Load images
        df["image_obj"] = df["imageurl"].apply(load_image)
        df = df[df["image_obj"].notnull()].reset_index(drop=True)

        # Compute embeddings
        image_embeddings = get_image_embeddings(df["image_obj"], processor, model)
        text_embeddings = get_text_embeddings(df["text_data"].tolist(), processor, model)

        # Store in session
        st.session_state.df = df
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.image_embeddings = image_embeddings
        st.session_state.text_embeddings = text_embeddings
        st.session_state.prepared = True

    st.success(f"System ready with {len(df)} products!")

# -----------------------------
# STOP until prepared
# -----------------------------
if not st.session_state.prepared:
    st.info("Click **Prepare Recommendation Engine** to start")
    st.stop()

# Restore from session
df = st.session_state.df
model = st.session_state.model
processor = st.session_state.processor
image_embeddings = st.session_state.image_embeddings
text_embeddings = st.session_state.text_embeddings

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Uploaded Image", width=200)

    inputs = processor(images=query_img, return_tensors="pt").to(device)
    with torch.no_grad():
        q_img_emb = model.get_image_features(**inputs)
        q_img_emb = q_img_emb / q_img_emb.norm(p=2, dim=-1, keepdim=True)

    q_img_emb = q_img_emb.cpu().numpy()

    # Image similarity
    img_similarity = image_embeddings @ q_img_emb.T

    # Optional text hint
    hint = st.text_input("Optional hint (e.g. shoe, watch, makeup)")

    if hint.strip():
        q_text_emb = get_text_embeddings([hint], processor, model)
        txt_similarity = text_embeddings @ q_text_emb.T
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


