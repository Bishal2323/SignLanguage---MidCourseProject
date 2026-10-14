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

# Main Dashboard Header
st.title("🤟 ASL Recognition Dashboard")

st.markdown("""
### American Sign Language Recognition Using Machine Learning

Welcome to our ASL Recognition Dashboard!

This application uses a Convolutional Neural Network (CNN)
to recognize static American Sign Language hand gestures.

Upload an image of a hand sign to see the predicted letter
and the model's confidence score.
""")

st.divider()
# -----------------------------
# Project Information Sidebar
# -----------------------------

with st.sidebar:
    st.header("📌 About This Project")

    st.write("""
    This project explores how machine learning
    can recognize American Sign Language (ASL)
    hand gestures.

    Our team developed and compared two models:

    - Support Vector Machine (SVM)
    - Convolutional Neural Network (CNN)
    """)

    st.divider()

    st.header("📖 How to Use")

    st.write("""
    1. Upload an image of an ASL hand sign.
    2. Wait for the CNN to analyze the image.
    3. View the predicted letter.
    4. Review the confidence score.
    """)

    st.divider()

    st.header("🧠 Model Information")

    st.info("""
    The CNN recognizes 24 static ASL letters.

    Letters J and Z are excluded because
    they require movement.
    """)
uploaded_file = st.file_uploader("Choose an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Create two columns for the dashboard
    col1, col2 = st.columns(2)

    # Left column: Uploaded Image
    with col1:
        st.subheader("📷 Uploaded Image")

        raw_img = Image.open(uploaded_file)

        st.image(
            raw_img,
            caption="Your ASL Hand Sign",
            width=250
        )

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

        # Right column: Prediction Results
    with col2:
        st.subheader("🎯 Prediction Results")

        st.success(f"Predicted Letter: {pred_letter}")

        st.metric(
            label="Confidence Score",
            value=f"{confidence * 100:.2f}%"
        )

    # Display Top-3 alternatives
    # Display Top 3 Predictions
st.divider()

st.subheader("📊 Top 3 Predictions")

st.write(
    "These are the three letters with the highest "
    "prediction confidence scores."
)

top_3_indices = np.argsort(probabilities)[-3:][::-1]

for idx in top_3_indices:
    letter = labels[idx]
    score = float(probabilities[idx])

    st.write(f"**Letter {letter}: {score * 100:.2f}%**")
    st.progress(score)

# ---------------------------------
# SVM vs CNN Model Comparison
# ---------------------------------

st.divider()

st.header("🧠 Machine Learning Model Comparison")

st.write("""
Our project explores two different machine learning
approaches for recognizing American Sign Language gestures.

The table below highlights the main differences
between the Support Vector Machine (SVM) and
Convolutional Neural Network (CNN).
""")

comparison_data = {
    "Feature": [
        "Model Type",
        "Learning Approach",
        "Feature Extraction",
        "Image Processing",
        "Complexity"
    ],
    "SVM": [
        "Traditional Machine Learning",
        "Supervised Learning",
        "Uses provided image features",
        "Classifies flattened image data",
        "Generally less computationally intensive"
    ],
    "CNN": [
        "Deep Learning",
        "Supervised Learning",
        "Learns image features automatically",
        "Processes images using convolutional layers",
        "Generally more computationally intensive"
    ]
}

st.dataframe(
    comparison_data,
    hide_index=True,
    use_container_width=True
)

st.info("""
Both models are designed to classify ASL hand signs,
but they use different approaches to recognize patterns.
""")

# ---------------------------------
# Model Performance Comparison
# ---------------------------------

st.divider()

st.header("📈 Model Performance Comparison")

st.write("""
The chart below compares the test accuracy and
Macro F1 scores of our SVM and CNN models.

These results were obtained from our group's
model evaluation notebooks.
""")

performance_data = {
    "Model": ["SVM", "CNN"],
    "Accuracy": [84, 100],
    "Macro F1 Score": [82, 100]
}

import pandas as pd

performance_df = pd.DataFrame(performance_data)

# Display performance table
st.subheader("Model Evaluation Results")

st.dataframe(
    performance_df,
    hide_index=True,
    use_container_width=True
)

# Display comparison chart
st.subheader("Accuracy and F1 Score Comparison")

# Display grouped bar chart
st.bar_chart(
    performance_df.set_index("Model"),
    stack=False
)

st.caption(
    "Results are rounded percentages reported in the "
    "project notebooks. The CNN's perfect test scores "
    "should be interpreted cautiously until the "
    "evaluation setup is verified."
)