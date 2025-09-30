#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os

# --- Re-define the HACT Model Architecture ---
# This class definition must be EXACTLY the same as the one you used for training.
# Python needs to know the model's structure to load the weights into it.
class HACTModel(nn.Module):
    def __init__(self,
                 cnn_model_name='tf_efficientnetv2_s_in21k',
                 transformer_layers=4,
                 transformer_heads=8,
                 embedding_dim=256,
                 pretrained=False): # Set pretrained=False for inference
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cnn_backbone = timm.create_model(cnn_model_name, pretrained=pretrained, num_classes=0)
        cnn_out_features = self.cnn_backbone.num_features
        self.projection = nn.Linear(cnn_out_features, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=transformer_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 1))

    def forward(self, x):
        cnn_features = self.cnn_backbone(x)
        projected_features = self.projection(cnn_features).unsqueeze(1)
        transformer_output = self.transformer_encoder(projected_features)
        output = self.classifier(transformer_output.squeeze(1))
        return output

# --- Prediction Function ---
def predict_image(image_path, model, device):
    """Loads an image, preprocesses it, and returns the model's prediction."""
    # Define the same transformations as the validation set during training
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Load and process the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = transform(image=image)['image']
    
    # Add a batch dimension and send to device
    input_tensor = processed_image.unsqueeze(0).to(device)

    # Make prediction
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item() # Get probability
        # Get binary prediction as an integer (1 or 0) to avoid boolean issues
        prediction = 1 if (probability > 0.5) else 0

    return prediction, probability

# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_WEIGHTS_PATH = 'hact_model_epoch_10.pth'
    IMAGE_FOLDER = 'test_images'

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print("Loading HACT model...")
    model = HACTModel()
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights file not found at '{MODEL_WEIGHTS_PATH}'")
        print("Please download the .pth file from Kaggle and place it in the same directory as this script.")
        exit()
        
    model.to(device)
    print("Model loaded successfully.")

    # --- Run Inference ---
    print(f"\n--- Making Predictions on all .tif images in '{IMAGE_FOLDER}' folder ---")
    
    # Check if the image folder exists
    if not os.path.isdir(IMAGE_FOLDER):
        print(f"Error: The folder '{IMAGE_FOLDER}' was not found.")
        print("Please create it and place your test images inside.")
        exit()
        
    # Automatically find all .tif files in the folder
    images_to_test = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.tif')]

    if not images_to_test:
        print(f"No .tif images found in the '{IMAGE_FOLDER}' folder.")
        exit()

    for img_name in images_to_test:
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        result = predict_image(img_path, model, device)
        if result:
            prediction, probability = result
            # Check against the integer 1 instead of a boolean
            label = "Cancer" if prediction == 1 else "No Cancer"
            print(f"Image: {img_name} -> Prediction: {label} (Probability: {probability:.4f})")


