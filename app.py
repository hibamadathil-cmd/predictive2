# ── app.py  ——  Streamlit DR Detection App ────────────────────────────────────
# Run locally: streamlit run app.py

import streamlit as st
import torch
import torch.nn.functional as F
import timm
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
IMG_SIZE = 224
DEVICE = torch.device("cpu")

@st.cache_resource
def load_model():
    import torch.nn as nn
    model = timm.create_model("efficientnet_b4", pretrained=False)
    in_f = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 5),
    )
    model.load_state_dict(torch.load("best_efficientnet_b4.pth", map_location=DEVICE))
    return model.eval()


def preprocess(img_np):
    img = cv2.addWeighted(img_np, 4, cv2.GaussianBlur(img_np, (0, 0), 10), -4, 128)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform(image=img)["image"].unsqueeze(0)


st.set_page_config(page_title="DR Detection", page_icon="👁", layout="centered")
st.title("👁 Diabetic Retinopathy Detection")
st.caption("Upload a retinal fundus image — model will classify the DR severity grade.")

model = load_model()
uploaded = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB"))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analysing image..."):
        with torch.no_grad():
            probs = F.softmax(model(preprocess(img)), dim=1)[0].cpu().numpy()

    pred = int(probs.argmax())
    st.markdown(f"### Result: **{CLASS_NAMES[pred]}** (Grade {pred})")
    st.markdown(f"**Confidence:** {probs[pred] * 100:.1f}%")
    st.markdown("---")
    st.markdown("#### Probability per class")
    for name, p, color in zip(CLASS_NAMES, probs, CLASS_COLORS):
        st.markdown(f"**{name}**: {p * 100:.1f}%")
        st.progress(float(p))
