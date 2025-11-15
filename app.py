import os
import tempfile
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="centered")

# Use same image size as training script
IMAGE_SIZE = (350, 350)

# Default class names from training script
DEFAULT_CLASS_NAMES = [
    "normal",
    "adenocarcinoma",
    "large.cell.carcinoma",
    "squamous.cell.carcinoma"
]

@st.cache_resource
def load_trained_model():
    """
    Download .hdf5 from Hugging Face Hub once per app process and load the Keras model.
    Loads weights only (no architecture config in the file).
    """
    hf_repo_id = "sampujit/LungsCancerDetection"
    hf_model_filename = "best_model.hdf5"
    
    try:
        model_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_model_filename,
            cache_dir=tempfile.gettempdir()
        )
        
        # Build the model architecture (matches training script)
        pretrained_model = tf.keras.applications.Xception(
            weights='imagenet',
            include_top=False,
            input_shape=[*IMAGE_SIZE, 3]
        )
        pretrained_model.trainable = False
        
        # Add custom layers (matches training script)
        model = tf.keras.Sequential([
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        
        # Load weights
        model.load_weights(model_path)
        return model
    except Exception as e:
        raise

def get_class_names():
    raw = os.getenv("CLASS_NAMES", "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts
    return DEFAULT_CLASS_NAMES

def load_and_preprocess_image(pil_img, target_size=IMAGE_SIZE):
    """Load and preprocess image matching training script logic"""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    img_array = image.img_to_array(pil_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale like training images
    return img_array

# UI
st.title("🫁 Lung Cancer Detection")
st.markdown("Upload a chest X-ray image. Model is downloaded once.")


# Load model (display status)
model_state = st.empty()
model_state.text("Loading model...")

try:
    model = load_trained_model()
    model_state.text("Model loaded ✅")
except Exception as e:
    model_state.text("Failed to load model ❌")
    st.error(f"Model load error: {e}")
    st.stop()

class_names = get_class_names()
try:
    output_classes = int(model.output_shape[-1])
except Exception:
    output_classes = None

if output_classes and output_classes != len(class_names):
    st.warning(f"Model outputs {output_classes} classes but CLASS_NAMES has {len(class_names)} entries. Update CLASS_NAMES env var if needed.")

# File upload
uploaded_file = st.file_uploader("Upload chest X-ray image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file)
    st.image(pil_img, caption="Uploaded image", use_column_width=True)
    
    if st.button("Analyze"):
        with st.spinner("Running prediction..."):
            try:
                # Preprocess and predict
                img_array = load_and_preprocess_image(pil_img)
                predictions = model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])
                predicted_label = class_names[predicted_class] if predicted_class < len(class_names) else f"class_{predicted_class}"
                
                # Display results
                st.success(f"**Prediction:** {predicted_label}")
                st.metric("Confidence", f"{confidence*100:.2f}%")
                
                # Show all probabilities
                prob_df = pd.DataFrame({
                    "class": [class_names[i] if i < len(class_names) else f"class_{i}" for i in range(len(predictions[0]))],
                    "probability": predictions[0]
                }).sort_values("probability", ascending=False)
                st.bar_chart(prob_df.set_index("class")["probability"])
                
                # Alert based on prediction
                if "normal" not in predicted_label.lower():
                    st.warning("⚠️ Potential abnormality detected — consult a medical professional immediately.")
                else:
                    st.info("✅ No strong cancer indicator detected by the model.")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Model architecture: Xception (pre-trained on ImageNet) + GlobalAveragePooling2D + Dense(4, softmax). Image size: 350×350.")
