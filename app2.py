import streamlit as st
import torch
from transformers import SwinForImageClassification, SwinConfig
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import gdown
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define class names
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Define the SwinWithDropout model
class SwinWithDropout(nn.Module):
    def __init__(self, model, dropout_rate=0.5):
        super(SwinWithDropout, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits
        logits = self.dropout(logits)
        return outputs, outputs.attentions

# Check if model file exists
model_path = 'swin_transformer_plant_village_finetuned_with_dropout.pth'
if not os.path.exists(model_path):
    st.error(f"Model file not found. Please download it from the link provided in the README and place it in the root directory of this repository.")

# Load the model
@st.cache_resource
def load_model():
    num_classes = len(class_names)
    config = SwinConfig(image_size=224, num_labels=num_classes, output_attentions=True)
    base_model = SwinForImageClassification(config)
    model = SwinWithDropout(base_model)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Display the prediction
def predict(model, image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs, attentions = model(image)
        logits = outputs.logits
        _, preds = torch.max(logits, 1)
    return class_names[preds.item()], attentions

# Streamlit app
st.title("Plant Disease Classification")
st.write("Upload an image of a plant leaf to classify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    model = load_model()
    image = preprocess_image(image)
    label, attentions = predict(model, image)
    st.write(f"Predicted Label: {label}")

    # Display attention map
    attention_map = attentions[-1][0].sum(dim=0).detach().cpu().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())  # Normalize to [0, 1]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image[0].permute(1, 2, 0).cpu().numpy())
    axs[0].axis('off')
    axs[0].set_title('Original Image')
    axs[1].imshow(attention_map, cmap='viridis')
    axs[1].axis('off')
    axs[1].set_title('Attention Map')
    st.pyplot(fig)
