from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import io
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Re-define the HACT Model Architecture ---
# This class definition must be EXACTLY the same as the one used for training.
class HACTModel(nn.Module):
    def __init__(self,
                 cnn_model_name='tf_efficientnetv2_s_in21k',
                 transformer_layers=4,
                 transformer_heads=8,
                 embedding_dim=256,
                 pretrained=False):
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

# --- Setup ---
MODEL_WEIGHTS_PATH = 'hact_model_epoch_10.pth'
device = torch.device("cpu")

# --- Load Model ---
print("Loading HACT model...")
model = HACTModel()
try:
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model weights file not found at '{MODEL_WEIGHTS_PATH}'")
    exit()

# --- NEW: Route to serve the frontend ---
@app.route('/')
def serve_frontend():
    """Serves the index.html file."""
    return send_from_directory('.', 'index.html')

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Read image file
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)

        # Preprocess the image
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        processed_image = transform(image=image_np)['image']
        input_tensor = processed_image.unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).item()

        # Format the response
        prediction = 1 if probability > 0.5 else 0
        response = {
            'prediction': 'Cancer' if prediction == 1 else 'No Cancer',
            'confidence': {
                'cancer': probability,
                'no_cancer': 1 - probability
            }
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error processing the image'}), 500

