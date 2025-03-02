import torch
import torch.nn as nn
# Old way
#args.local_rank

# New way
import os
#local_rank = int(os.environ['LOCAL_RANK'])

# Modified sections of training.py

def build_model(cfg):
    """Build keypoint transformer model"""
    model = KeypointTransformer(
        input_dim=cfg['model']['RecognitionNetwork']['TransformerEncoder']['input_dim'],
        hidden_dim=cfg['model']['RecognitionNetwork']['TransformerEncoder']['hidden_dim'],
        nhead=cfg['model']['RecognitionNetwork']['TransformerEncoder']['nhead'],
        num_encoder_layers=cfg['model']['RecognitionNetwork']['TransformerEncoder']['num_encoder_layers'],
        dim_feedforward=cfg['model']['RecognitionNetwork']['TransformerEncoder']['dim_feedforward'],
        dropout=cfg['model']['RecognitionNetwork']['TransformerEncoder']['dropout'],
        activation=cfg['model']['RecognitionNetwork']['TransformerEncoder']['activation'],
        normalize_before=cfg['model']['RecognitionNetwork']['TransformerEncoder']['normalize_before'],
        feature_projection=cfg['model']['RecognitionNetwork']['TransformerEncoder']['feature_projection'],
        num_classes=len(gloss_tokenizer),  # Number of gloss classes
    )
    return model

class KeypointTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_encoder_layers, 
                 dim_feedforward, dropout, activation, normalize_before, 
                 feature_projection, num_classes):
        super().__init__()
        
        # Feature projection if needed
        self.feature_projection = nn.Linear(input_dim, hidden_dim) if feature_projection else None
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, keypoints, keypoint_masks=None):
        # keypoints shape: [batch_size, seq_len, num_keypoints, 2/3]
        B, T, K, C = keypoints.shape
        
        # Flatten keypoints and project if needed
        x = keypoints.view(B, T, -1)  # [B, T, K*C]
        if self.feature_projection:
            x = self.feature_projection(x)
            
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask from keypoint_masks if provided
        if keypoint_masks is not None:
            mask = ~keypoint_masks  # Convert to boolean mask
        else:
            mask = None
            
        # Transformer encoding
        x = x.transpose(0, 1)  # [T, B, D]
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)  # [B, T, D]
        
        # Classification
        logits = self.classifier(x)  # [B, T, num_classes]
        
        return logits

# Add cosine warmup scheduler
def build_scheduler(config, optimizer):
    """Build learning rate scheduler with warmup"""
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['t_max']
    )
    return scheduler, 'step'