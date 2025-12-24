import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import pandas as pd

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Potato Disease Detection")

st.title("🥔 Potato Disease Detection")

uploaded_file = st.file_uploader(
    "Upload potato leaf image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = Image.fromarray((image * 255).astype(np.uint8))
    return image

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            image = preprocess_image(image)

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            files = {"file": ("image.png", img_bytes, "image/png")}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                st.success("Prediction Done ✅")
                st.write(f"### 🧪 Disease: **{result['class']}**")
                st.write(f"### 📊 Confidence: **{result['confidence']*100:.2f}%**")

                # ===== Probability Bar Chart =====
                st.subheader("🔎 Class Probabilities")

                prob_df = pd.DataFrame(
                    result["probabilities"].items(),
                    columns=["Class", "Probability"]
                )

                prob_df["Probability"] = prob_df["Probability"] * 100

                st.bar_chart(
                    prob_df.set_index("Class")
                )

            else:
                st.error("FastAPI not responding ❌")
