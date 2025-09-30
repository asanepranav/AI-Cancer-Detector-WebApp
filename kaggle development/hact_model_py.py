# hact_model.py

import torch
import torch.nn as nn
import timm  # PyTorch Image Models library

print("Utility script 'hact_model.py' loaded.")

class HACTModel(nn.Module):
    """
    Hybrid Attention-CNN Transformer (HACT) Model for Histopathology.
 
    This model combines a powerful CNN backbone (EfficientNetV2) for local feature
    extraction with a Transformer Encoder for capturing global context and
    long-range dependencies between different parts of the image patch.
    """
    def __init__(self,
                 cnn_model_name='tf_efficientnetv2_s_in21k',
                 transformer_layers=4,
                 transformer_heads=8,
                 embedding_dim=256,
                 pretrained=True):
        """
        Initializes the HACT Model.

        Args:
            cnn_model_name (str): The name of the CNN backbone from the 'timm' library.
            transformer_layers (int): Number of layers in the Transformer Encoder.
            transformer_heads (int): Number of attention heads in the Transformer Encoder.
            embedding_dim (int): The dimension for the token embeddings fed to the Transformer.
            pretrained (bool): Whether to use a pre-trained CNN backbone.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # 1. CNN Backbone for local feature extraction
        self.cnn_backbone = timm.create_model(cnn_model_name, pretrained=pretrained, num_classes=0)
        
        # Get the number of output features from the CNN backbone
        cnn_out_features = self.cnn_backbone.num_features

        # 2. Projection Layer (Attention Bridge)
        # This linear layer projects the CNN's feature map into the embedding
        # dimension expected by the Transformer. It acts as a bridge.
        self.projection = nn.Linear(cnn_out_features, embedding_dim)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=transformer_heads,
            batch_first=True  # Crucial for (batch, sequence, feature) input shape
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # 4. Classification Head
        # Takes the output of the Transformer and produces the final binary prediction.
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim), # Normalize features before classification
            nn.Linear(embedding_dim, 1)   # Output one value for binary classification
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        # x shape: (batch_size, channels, height, width), e.g., (32, 3, 96, 96)

        # 1. Pass through CNN Backbone
        # The output of the backbone is a feature vector for each image.
        cnn_features = self.cnn_backbone(x)
        # cnn_features shape: (batch_size, cnn_out_features), e.g., (32, 1280)

        # 2. Project features to embedding dimension
        # We add a sequence dimension of 1, as the transformer expects a sequence.
        projected_features = self.projection(cnn_features).unsqueeze(1)
        # projected_features shape: (batch_size, 1, embedding_dim), e.g., (32, 1, 256)

        # 3. Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(projected_features)
        # transformer_output shape: (batch_size, 1, embedding_dim)

        # 4. Pass through Classification Head
        # We remove the sequence dimension before the classifier.
        output = self.classifier(transformer_output.squeeze(1))
        # output shape: (batch_size, 1)

        return output


