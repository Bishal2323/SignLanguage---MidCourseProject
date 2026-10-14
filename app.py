import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pathlib import Path

st.set_page_config(
    page_title="ASL Recognition",
    page_icon="🤲",
    layout="wide",
)

# Light visual polish only
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { letter-spacing: -0.02em; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    .pred-letter {
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.4rem 0 0.2rem 0;
        color: #0f766e;
    }
    .pred-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Main Dashboard Header
st.title("ASL Recognition Dashboard")
st.caption("American Sign Language · CNN hand-sign classifier")

st.write(
    "Upload a static ASL hand sign image to get the predicted letter, "
    "confidence score, and top alternative guesses."
)

st.divider()

# -----------------------------
# Project Information Sidebar
# -----------------------------

with st.sidebar:
    st.header("About This Project")

    st.write("""
    This project explores how machine learning
    can recognize American Sign Language (ASL)
    hand gestures.

    Our team developed and compared two models:

    - Support Vector Machine (SVM)
    - Convolutional Neural Network (CNN)
    """)

    st.divider()

    st.header("How to Use")

    st.write("""
    1. Upload an image of an ASL hand sign.
    2. Wait for the CNN to analyze the image.
    3. View the predicted letter.
    4. Review the confidence score.
    """)

    st.divider()

    st.header("Model Information")

    st.info("""
    The CNN recognizes 24 static ASL letters.

    Letters J and Z are excluded because
    they require movement.
    """)

uploaded_file = st.file_uploader(
    "Choose an image (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    help="Clear, well-lit hand signs work best.",
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Uploaded Image")
        raw_img = Image.open(uploaded_file)
        st.image(raw_img, caption="Your ASL hand sign", use_container_width=True)

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

    with col2:
        st.subheader("Prediction")
        st.markdown(
            f'<p class="pred-label">Predicted letter</p>'
            f'<p class="pred-letter">{pred_letter}</p>',
            unsafe_allow_html=True,
        )
        st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")
        st.caption("Model output after softmax over 24 ASL classes.")

    st.divider()
    st.subheader("Top 3 Predictions")
    st.caption("Highest-confidence letter guesses from the CNN.")

    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_cols = st.columns(3)

    for rank, (col, idx) in enumerate(zip(top_cols, top_3_indices), start=1):
        letter = labels[idx]
        score = float(probabilities[idx])
        with col:
            st.metric(label=f"#{rank} · Letter {letter}", value=f"{score * 100:.1f}%")
            st.progress(score)

else:
    st.info("Upload a PNG or JPG hand-sign image to see predictions.")

# ---------------------------------
# SVM vs CNN Model Comparison
# ---------------------------------

st.divider()

st.header("Model Comparison")

st.write("""
Our project explores two different machine learning
approaches for recognizing American Sign Language gestures.
""")

comparison_data = {
    "Feature": [
        "Model Type",
        "Learning Approach",
        "Feature Extraction",
        "Image Processing",
        "Complexity",
    ],
    "SVM": [
        "Traditional Machine Learning",
        "Supervised Learning",
        "Uses provided image features",
        "Classifies flattened image data",
        "Generally less computationally intensive",
    ],
    "CNN": [
        "Deep Learning",
        "Supervised Learning",
        "Learns image features automatically",
        "Processes images using convolutional layers",
        "Generally more computationally intensive",
    ],
}

st.dataframe(comparison_data, hide_index=True, use_container_width=True)

st.info(
    "Both models classify ASL hand signs, but they recognize patterns differently."
)

# ---------------------------------
# Model Performance Comparison
# ---------------------------------

st.divider()

st.header("Model Performance")

st.write(
    "Test accuracy and Macro F1 from our group evaluation notebooks."
)

performance_df = pd.DataFrame(
    {
        "Model": ["SVM", "CNN"],
        "Accuracy": [84, 100],
        "Macro F1 Score": [82, 100],
    }
)

perf_col1, perf_col2 = st.columns([1, 1.2], gap="large")

with perf_col1:
    st.subheader("Evaluation Results")
    st.dataframe(performance_df, hide_index=True, use_container_width=True)

with perf_col2:
    st.subheader("Accuracy & F1")
    st.bar_chart(performance_df.set_index("Model"), stack=False)

st.caption(
    "Results are rounded percentages from the project notebooks. "
    "The CNN's perfect test scores should be interpreted cautiously "
    "until the evaluation setup is verified."
)