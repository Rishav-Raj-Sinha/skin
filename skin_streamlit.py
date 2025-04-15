import streamlit as st
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

st.title("Skin disease predictor")

repo_name = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
image_processor = AutoImageProcessor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)

class_names = [
    'Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa', 'Hailey-Hailey Disease',
    'Herpes Simplex', 'Impetigo', 'Larva Migrans', 'Leprosy Borderline', 'Leprosy Lepromatous',
    'Leprosy Tuberculoid', 'Lichen Planus', 'Lupus Erythematosus Chronicus Discoides', 'Melanoma',
    'Molluscum Contagiosum', 'Mycosis Fungoides', 'Neurofibromatosis',
    'Papilomatosis Confluentes And Reticulate', 'Pediculosis Capitis', 'Pityriasis Rosea',
    'Porokeratosis Actinic', 'Psoriasis', 'Tinea Corporis', 'Tinea Nigra', 'Tungiasis',
    'actinic keratosis', 'dermatofibroma', 'nevus', 'pigmented benign keratosis',
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
]

def predict_skin_disease(image):
    """Predict skin disease from an image file path."""
    try:
        # Load and preprocess image
        encoding = image_processor(image, return_tensors="pt")

        # Make prediction
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_name = class_names[predicted_class_idx]

        st.write(f"Predicted Disease: {predicted_class_name}")
    except Exception as e:
        st.write(f"Error during prediction: {e}")

# Example usage:
image = st.file_uploader("upload your image")
if image is not None:
    try:
            image_pil = Image.open(image).convert("RGB")
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            predict_skin_disease(image_pil)
    except Exception as e:
            st.write(f"Error loading image: {e}")
