import streamlit as st
from PIL import Image
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from io import BytesIO
import requests

# -----------------------------
# Device setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load CLIP model (Cached)
# -----------------------------
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# -----------------------------
# Load CSV dataset
# -----------------------------
@st.cache_data
def load_dataset(csv_path="data/test_products.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # remove spaces in headers
    df = df.fillna('')  # fill NaNs
    if 'ImageURL' in df.columns:
        df['ImageURL'] = df['ImageURL'].str.strip().str.replace('"', '')  # clean URLs
    return df

# -----------------------------
# Image Loading Helper
# -----------------------------
@st.cache_data
def load_image_from_url(url):
    try:
        url = url.strip().replace('"', '')  # remove spaces and quotes
        if not url:
            return None
        headers = {"User-Agent": "Mozilla/5.0"}  # fix blocked requests
        response = requests.get(url, headers=headers, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print("Failed to load image:", url, e)
        return None

# -----------------------------
# Feature Extraction Helper
# -----------------------------
def extract_features_safe(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, 'text_embeds') and outputs.text_embeds is not None:
        return outputs.text_embeds
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state.mean(dim=1)
    if isinstance(outputs, (tuple, list)):
        if len(outputs) > 1:
            return outputs[1]
        elif len(outputs) > 0:
            return outputs[0].mean(dim=1) if outputs[0].dim() == 3 else outputs[0]
    return outputs

# -----------------------------
# Compute dataset features
# -----------------------------
@st.cache_data
def get_dataset_features(data):
    model, processor = load_clip_model()
    
    literals = (data['Brand'] + " " + data['Name'] + " " + data['Category']).astype(str).tolist()
    batch_features = []
    valid_indices = []
    batch_size = 64
    
    total = len(literals)
    for i in range(0, total, batch_size):
        batch_text = literals[i:i+batch_size]
        batch_indices = data.index[i:i+batch_size].tolist()
        inputs = processor(text=batch_text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            raw_text_output = model.get_text_features(**inputs)
            text_features = extract_features_safe(raw_text_output)
            if isinstance(text_features, torch.Tensor):
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                batch_features.append(text_features.cpu().numpy())
                valid_indices.extend(batch_indices)
    
    if batch_features:
        all_features = np.concatenate(batch_features, axis=0)
        return all_features, valid_indices
    return np.array([]), []

# -----------------------------
# Recommend by image
# -----------------------------
def recommend_by_image(uploaded_image, data=None, top_n=5):
    if data is None:
        data = load_dataset()
    
    with st.spinner("Indexing product catalog..."):
        dataset_features, valid_indices = get_dataset_features(data)
    
    if len(dataset_features) == 0:
        st.error("Could not extract features from product catalog.")
        return pd.DataFrame()
    
    model, processor = load_clip_model()
    inputs = processor(images=uploaded_image, return_tensors="pt").to(device)
    with torch.no_grad():
        raw_output = model.get_image_features(**inputs)
        query_features = extract_features_safe(raw_output)
        query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
    
    query_features_np = query_features.cpu().numpy()
    similarities = (dataset_features @ query_features_np.T).squeeze()
    
    top_indices_local = similarities.argsort()[::-1][:top_n]
    top_df_indices = [valid_indices[i] for i in top_indices_local]
    safe_indices = [i for i in top_df_indices if i in data.index]
    
    return data.loc[safe_indices].copy()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🖼️ Image Recommendation (Local CSV)")
uploaded_file = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"])
top_n = st.slider("Number of recommendations", 1, 10, 5)

if uploaded_file:
    uploaded_img = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_img, width=250)
    
    if st.button("Get Recommendations"):
        results_df = recommend_by_image(uploaded_img, top_n=top_n)
        
        if results_df.empty:
            st.warning("No recommendations found.")
        else:
            st.subheader("Recommended Products")
            
            img_col = "ImageURL"
            name_col = "Name"
            brand_col = "Brand"
            
            # Display recommendations in a 3-column gallery
            cols_per_row = 3
            for i in range(0, len(results_df), cols_per_row):
                row_items = results_df.iloc[i:i+cols_per_row]
                cols = st.columns(cols_per_row)
                for j, (_, row) in enumerate(row_items.iterrows()):
                    with cols[j]:
                        img_url = row[img_col].strip()
                        img = load_image_from_url(img_url)
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.text("Image not available")
                        st.markdown(f"**{row[name_col]}**")
                        st.markdown(f"Brand: {row[brand_col]}")
                        if "Category" in row:
                            st.markdown(f"Category: {row['Category']}")


