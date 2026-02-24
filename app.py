"""
Document Tampering Detection - Streamlit App
Detects whether an uploaded document image is original or tampered
using Error Level Analysis (ELA) and a trained CNN model.
"""

import os
import json
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt
import io

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Tampering Detection",
    # page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Model Definition (must match training architecture) ────────────────────────
class TamperingDetectionCNN(nn.Module):
    """Lightweight CNN designed for small datasets."""
    def __init__(self, image_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            dummy = self.features(dummy)
            flat_size = dummy.view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ─── Load Model (cached) ────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_tampering_model.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")


@st.cache_resource
def load_model():
    """Load the trained PyTorch model."""
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    image_size = config.get("image_size", 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TamperingDetectionCNN(image_size=image_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model, config, device


# ─── ELA Processing ─────────────────────────────────────────────────────────────
def convert_to_ela_image(image: Image.Image, quality: int = 90) -> Image.Image:
    """Convert a PIL image to its Error Level Analysis representation."""
    buffer = io.BytesIO()
    rgb_image = image.convert("RGB")
    rgb_image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    ela_im = ImageChops.difference(rgb_image, resaved)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    return ela_im


def predict_image(image: Image.Image, model, config, device):
    """Run tampering prediction on a single image."""
    image_size = config.get("image_size", 128)
    ela_quality = config.get("ela_quality", 90)

    # ELA transform
    ela_img = convert_to_ela_image(image, quality=ela_quality)
    ela_resized = ela_img.resize((image_size, image_size))

    # To tensor
    img_array = np.array(ela_resized) / 255.0
    tensor = torch.FloatTensor(img_array.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred_label = int(np.argmax(probs))
    class_names = {0: "Original", 1: "Tampered"}
    return {
        "label": class_names[pred_label],
        "confidence": float(probs[pred_label]),
        "prob_original": float(probs[0]),
        "prob_tampered": float(probs[1]),
        "ela_image": ela_resized,
    }


# ─── UI ──────────────────────────────────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        # st.image("https://img.icons8.com/color/96/000000/document.png", width=60)
        # st.title("Settings")
        # show_ela = st.checkbox("Show ELA Image", value=True)
        # show_probs = st.checkbox("Show Probability Details", value=True)
        # st.markdown("---")
        # st.markdown(
        #     """
        #     **How it works:**
        #     1. Upload a document image (JPG/PNG)
        #     2. The image is converted using **Error Level Analysis (ELA)**
        #     3. A trained **CNN model** classifies it as Original or Tampered
        #     4. Results are displayed with confidence scores
        #     """
        # )
        # st.markdown("---")
        st.caption("Built with PyTorch & Streamlit")

    # Header
    st.title("Document Tampering Detection")
    st.markdown(
        "Upload a document image to detect whether it has been **tampered** or is **original**. "
        # "The model uses *Error Level Analysis (ELA)* to identify compression inconsistencies."
    )

    # Load model
    try:
        model, config, device = load_model()
    except FileNotFoundError:
        st.error(
            "Model not found! Please train the model first by running "
            "`train_tampering_model.ipynb`, which saves it to `model/best_tampering_model.pth`."
        )
        return

    # File uploader
    st.markdown("### Upload Document")
    uploaded_files = st.file_uploader(
        "Choose document image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)

            st.markdown("---")
            st.subheader(f"{uploaded_file.name}")

            # Run prediction
            with st.spinner("Analyzing document..."):
                result = predict_image(image, model, config, device)

            # Layout: 3 columns
            # if show_ela:
            #     col1, col2, col3 = st.columns([1, 1, 1])
            # else:
                
                col1, col3 = st.columns([1, 1])
                col2 = None

            with col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)

            if col2 is not None:
                with col2:
                    st.markdown("**ELA Image**")
                    st.image(result["ela_image"], use_container_width=True)

            with col3:
                st.markdown("**Prediction Result**")

                # Big result display
                is_tampered = result["label"] == "Tampered"
                if is_tampered:
                    st.error(f"**{result['label']}**")
                else:
                    st.success(f"**{result['label']}**")

                st.metric(
                    label="Confidence",
                    value=f"{result['confidence']:.1%}",
                )

                # if show_probs:
                #     st.markdown("**Class Probabilities:**")
                #     prob_col1, prob_col2 = st.columns(2)
                #     with prob_col1:
                #         st.metric("Original", f"{result['prob_original']:.1%}")
                #     with prob_col2:
                #         st.metric("Tampered", f"{result['prob_tampered']:.1%}")

                #     # Progress bars
                #     st.markdown("**Original**")
                #     st.progress(result["prob_original"])
                #     st.markdown("**Tampered**")
                #     st.progress(result["prob_tampered"])

        # Summary for multiple files
        if len(uploaded_files) > 1:
            st.markdown("---")
            st.subheader("Batch Summary")
            results_summary = []
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file)
                res = predict_image(img, model, config, device)
                results_summary.append(
                    {
                        "File": uploaded_file.name,
                        "Prediction": res["label"],
                        "Confidence": f"{res['confidence']:.1%}",
                        "P(Original)": f"{res['prob_original']:.3f}",
                        "P(Tampered)": f"{res['prob_tampered']:.3f}",
                    }
                )
            st.table(results_summary)

    else:
        # Placeholder when no file is uploaded
        st.info("Upload one or more document images to get started.")

        st.markdown("### Model Information")
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Framework", "PyTorch")
        with info_col2:
            st.metric("Image Size", f"{config.get('image_size', 128)}×{config.get('image_size', 128)}")
        with info_col3:
            st.metric("Test Accuracy", f"{config.get('test_accuracy', 0):.1%}")
        with info_col4:
            st.metric("Training Images", config.get("total_images", "N/A"))


if __name__ == "__main__":
    main()
