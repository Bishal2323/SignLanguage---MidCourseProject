import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

# 1. Re-declare ASL_CNN architecture to load state_dict
class ASL_CNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 24),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

letter_labels = [chr(i + 65) for i in range(26) if i not in [9, 25]]
MODEL_DIR = Path(__file__).parent / "Models"

@st.cache_resource
def load_cnn_model():
    model = ASL_CNN()
    # Load weights onto CPU for deployment
    model.load_state_dict(
        torch.load(MODEL_DIR / "asl_cnn_best.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model

st.title("ASL Translation Engine")
st.write("Upload a static image of a hand sign (A–Y, omitting J & Z).")

uploaded_file = st.file_uploader("Choose an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display user image
    raw_img = Image.open(uploaded_file)
    st.image(raw_img, caption="Uploaded Image", width=160)

    # Preprocess: convert to Grayscale, resize to 28x28, normalize to [0, 1]
    gray_img = ImageOps.grayscale(raw_img).resize((28, 28))
    img_array = np.array(gray_img, dtype=np.float32) / 255.0

    # Shape expected by CNN: (batch_size=1, channels=1, height=28, width=28)
    tensor_input = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    model = load_cnn_model()
    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
    labels = letter_labels

    pred_idx = int(np.argmax(probabilities))
    pred_letter = labels[pred_idx]
    confidence = float(probabilities[pred_idx])

    st.subheader(f"Predicted Letter: **{pred_letter}**")
    st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")

    # Display Top-3 alternatives
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    st.write("**Top 3 Candidates:**")
    for idx in top_3_indices:
        st.write(f"- **{labels[idx]}**: {probabilities[idx] * 100:.2f}%")