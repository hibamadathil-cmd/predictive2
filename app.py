import os

import streamlit as st
import torch
import torch.nn as nn
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILES = {
    "efficientnet_b4": "best_efficientnet_b4.pth",
    "resnet50": "best_resnet50.pth",
}


def ben_graham_preprocess(img_bgr, sigmaX=10):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    enhanced = cv2.addWeighted(
        img_rgb, 4,
        cv2.GaussianBlur(img_rgb, (0, 0), sigmaX), -4,
        128
    )
    return enhanced


def build_model(model_name: str, num_classes: int = 5) -> nn.Module:
    if model_name == "resnet50":
        model = timm.create_model("resnet50", pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif model_name == "efficientnet_b4":
        model = timm.create_model("efficientnet_b4", pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def guess_model_name_from_filename(filename: str) -> str:
    name = filename.lower()
    if "efficient" in name:
        return "efficientnet_b4"
    if "resnet" in name:
        return "resnet50"
    return "efficientnet_b4"


def find_model_file():
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            return name, path
    return None, None


def load_model_from_state_dict(state_dict, model_name: str):
    model = build_model(model_name)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, model_name


@st.cache_resource
def load_model(model_path=None, model_name=None):
    if model_path is None:
        model_name, model_path = find_model_file()
        if model_name is None:
            return None, None

    if hasattr(model_path, "read"):
        if model_name is None and hasattr(model_path, "name"):
            model_name = guess_model_name_from_filename(model_path.name)
        state = torch.load(model_path, map_location=DEVICE)
    else:
        if model_name is None and isinstance(model_path, str):
            model_name = guess_model_name_from_filename(os.path.basename(model_path))
        state = torch.load(model_path, map_location=DEVICE)

    return load_model_from_state_dict(state, model_name)


def preprocess(img_rgb: np.ndarray) -> torch.Tensor:
    img = ben_graham_preprocess(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform(image=img)["image"].unsqueeze(0).to(DEVICE)


def predict(image: np.ndarray, model: nn.Module):
    tensor = preprocess(image)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
    return pred, probs


st.set_page_config(
    page_title="DR Detection",
    page_icon="👁️",
    layout="centered",
)

st.title("👁️ Diabetic Retinopathy Detection")
st.write(
    "Upload a retinal fundus image and the model will predict the DR severity grade."
)

model, model_name = load_model()
checkpoint_upload = None
if model is None:
    st.warning(
        "No local checkpoint found. Upload a `.pth` or `.pt` model checkpoint below "
        "or place `best_efficientnet_b4.pth` / `best_resnet50.pth` in this folder."
    )
    checkpoint_upload = st.file_uploader(
        "Upload model checkpoint", type=["pth", "pt"]
    )

    if checkpoint_upload is not None:
        model_name = guess_model_name_from_filename(checkpoint_upload.name)
        model, model_name = load_model(model_path=checkpoint_upload, model_name=model_name)

if model is None:
    st.error(
        "No model checkpoint is available. Please upload a valid `.pth` or `.pt` file "
        "and reload the app."
    )
    st.stop()

st.success(f"Loaded model: {model_name}")

uploaded_file = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Uploaded fundus image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            pred_class, probs = predict(image_np, model)

        st.markdown(f"### Result: **{CLASS_NAMES[pred_class]}** (Grade {pred_class})")
        st.markdown(f"**Confidence:** {probs[pred_class] * 100:.1f}%")
        st.divider()
        st.markdown("#### Probability per class")
        for name, p, color in zip(CLASS_NAMES, probs, CLASS_COLORS):
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:12px;'>"
                f"<span style='width:100px'>{name}</span>"
                f"<progress value='{p}' max='1' style='flex:1;'></progress>"
                f"<span style='min-width:70px'>{p*100:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.write("")
        st.info(f"Model checkpoint: {MODEL_FILES[model_name]}")
