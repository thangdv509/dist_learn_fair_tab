"""
Model components for fairness learning with disentangled representations.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

# Auto-detect device: use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============== GRADIENT REVERSAL LAYER ==============
class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for adversarial training.
    
    Forward: identity function (passes input through unchanged)
    Backward: reverses gradient (multiplies by -lambda_grl)
    
    This allows:
    - Adversarial classifier to minimize CE loss (normal training)
    - Encoder to maximize CE loss (via reversed gradient)
    """
    @staticmethod
    def forward(ctx, x, lambda_grl=1.0):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class GRL(nn.Module):
    """
    Wrapper module for Gradient Reversal Layer.
    """
    def __init__(self, lambda_grl=1.0):
        super().__init__()
        self.lambda_grl = lambda_grl
    
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_grl)


# ============== CLASSIFIERS ==============
class TaskClassifier(nn.Module):
    """
    Enhanced classifier to predict y from z_c (z_c should contain task-relevant content information).
    Works on latent_dim (can be 768 or smaller).
    """
    def __init__(self, latent_dim=768, num_classes=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Direct classification from 768-dim BERT embedding
        self.fc1 = nn.Linear(latent_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second block with residual connection
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.2)
        
        # Third block
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.15)
        
        # Output - supports any number of classes
        self.fc_out = nn.Linear(256, num_classes)
        
    def forward(self, z_c):
        # First block
        x1 = self.fc1(z_c)
        x1 = self.bn1(x1)
        x1 = torch.relu(x1)
        x1 = self.dropout1(x1)
        
        # Second block with residual
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = torch.relu(x2)
        x2 = x2 + x1  # Residual connection
        x2 = self.dropout2(x2)
        
        # Third block
        x3 = self.fc3(x2)
        x3 = self.bn3(x3)
        x3 = torch.relu(x3)
        x3 = self.dropout3(x3)
        
        # Output
        logits = self.fc_out(x3)
        return logits


class AdversarialClassifier(nn.Module):
    """
    Enhanced adversarial classifier to predict y from z_d.
    Stronger architecture to better detect any task information in z_d.
    We want z_d to NOT contain task information, so we maximize this loss.
    Works on latent_dim (can be 768 or smaller).
    """
    def __init__(self, latent_dim=768, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        # Enhanced architecture - stronger to find any information
        # Works directly on 768-dim BERT embedding
        self.fc1 = nn.Linear(latent_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output - supports any number of classes
        self.fc_out = nn.Linear(128, num_classes)
    
    def forward(self, z_d):
        x = self.fc1(z_d)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        logits = self.fc_out(x)
        return logits


class ContentDecoder(nn.Module):
    """Decoder for z_c: D_c(z_c) -> embed_dim"""
    def __init__(self, latent_dim=64, embed_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, embed_dim)
        )
    
    def forward(self, z_c):
        return self.fc(z_c)


class DemographicDecoder(nn.Module):
    """Decoder for z_d: D_d(z_d) -> embed_dim"""
    def __init__(self, latent_dim=64, embed_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, embed_dim)
        )
    
    def forward(self, z_d):
        return self.fc(z_d)


# ============== HEADS FOR z_c and z_d ==============
class ContentHead(nn.Module):
    """
    Head to extract z_c (Content) from shared BERT encoder.
    Uses attention pooling + VAE bottleneck to compress to smaller latent space.
    VAE is used for compression and regularization, not generation.
    """
    def __init__(self, d_model=768, latent_dim=64):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # Attention pooling: learn which tokens are important for content
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # VAE bottleneck: compress from d_model to latent_dim
        # Encoder: d_model -> hidden -> mu/logvar
        self.encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, bert_output, attention_mask=None, token_ids=None, mask_stop_words=False):
        """
        bert_output: [batch_size, seq_len, d_model]
        attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
        token_ids: [batch_size, seq_len] - BERT token IDs (for masking stop words)
        mask_stop_words: If True, mask common stop words in attention
        Returns: z_c [batch_size, latent_dim], attn_weights [batch_size, seq_len], mu, logvar
        """
        # Compute attention logits
        attn_logits = self.attention_pool(bert_output).squeeze(-1)  # [batch_size, seq_len]
        
        # Mask padding tokens: set logits to -inf so softmax gives them 0 probability
        if attention_mask is not None:
            attn_logits = attn_logits.masked_fill(attention_mask == 0, -1e9)
        
        # Optionally mask stop words (BERT tokenizer IDs for common stop words)
        # These are BERT's token IDs for: a, an, the, and, or, but, in, on, at, to, for, of, with, by, as, is, are, was, were
        if mask_stop_words and token_ids is not None:
            # Common stop word token IDs in BERT vocabulary (approximate, may need adjustment)
            # You can get these by: tokenizer.convert_tokens_to_ids(['a', 'an', 'the', ...])
            stop_word_ids = {1037, 1039, 1996, 1998, 2004, 2010, 2012, 2013, 2014, 2015, 2017, 2019, 2020, 2022, 2024, 2003, 2024, 2027, 2028}
            # Create mask for stop words
            stop_word_mask = torch.isin(token_ids, torch.tensor(list(stop_word_ids), device=token_ids.device))
            attn_logits = attn_logits.masked_fill(stop_word_mask, -1e9)
        
        # Apply softmax to get attention weights (padding tokens and stop words will have 0 weight)
        attn_weights = torch.softmax(attn_logits, dim=1)  # [batch_size, seq_len]
        
        # Weighted sum of token embeddings
        weighted = torch.sum(attn_weights.unsqueeze(-1) * bert_output, dim=1)  # [batch_size, d_model]
        
        # VAE encoder: compress to latent_dim
        h = self.encoder(weighted)  # [batch_size, 256]
        mu = self.fc_mu(h)  # [batch_size, latent_dim]
        logvar = self.fc_logvar(h)  # [batch_size, latent_dim]
        
        # Reparameterization
        z_c = self.reparameterize(mu, logvar)  # [batch_size, latent_dim]
        
        return z_c, attn_weights, mu, logvar


class DemographicHead(nn.Module):
    """
    Head to extract z_d (Demographic) from shared BERT encoder.
    Uses different attention pooling + VAE bottleneck to compress to smaller latent space.
    VAE is used for compression and regularization, not generation.
    """
    def __init__(self, d_model=768, latent_dim=64):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # Attention pooling: learn which tokens are important for demographic
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # VAE bottleneck: compress from d_model to latent_dim
        # Encoder: d_model -> hidden -> mu/logvar
        self.encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, bert_output, attention_mask=None, token_ids=None, mask_stop_words=False):
        """
        bert_output: [batch_size, seq_len, d_model]
        attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
        token_ids: [batch_size, seq_len] - BERT token IDs (for masking stop words)
        mask_stop_words: If True, mask common stop words in attention
        Returns: z_d [batch_size, latent_dim], attn_weights [batch_size, seq_len], mu, logvar
        """
        # Compute attention logits
        attn_logits = self.attention_pool(bert_output).squeeze(-1)  # [batch_size, seq_len]
        
        # Mask padding tokens: set logits to -inf so softmax gives them 0 probability
        if attention_mask is not None:
            attn_logits = attn_logits.masked_fill(attention_mask == 0, -1e9)
        
        # Optionally mask stop words (same as ContentHead)
        if mask_stop_words and token_ids is not None:
            stop_word_ids = {1037, 1039, 1996, 1998, 2004, 2010, 2012, 2013, 2014, 2015, 2017, 2019, 2020, 2022, 2024, 2003, 2024, 2027, 2028}
            stop_word_mask = torch.isin(token_ids, torch.tensor(list(stop_word_ids), device=token_ids.device))
            attn_logits = attn_logits.masked_fill(stop_word_mask, -1e9)
        
        # Apply softmax to get attention weights (padding tokens and stop words will have 0 weight)
        attn_weights = torch.softmax(attn_logits, dim=1)  # [batch_size, seq_len]
        
        # Weighted sum of token embeddings
        weighted = torch.sum(attn_weights.unsqueeze(-1) * bert_output, dim=1)  # [batch_size, d_model]
        
        # VAE encoder: compress to latent_dim
        h = self.encoder(weighted)  # [batch_size, 256]
        mu = self.fc_mu(h)  # [batch_size, latent_dim]
        logvar = self.fc_logvar(h)  # [batch_size, latent_dim]
        
        # Reparameterization
        z_d = self.reparameterize(mu, logvar)  # [batch_size, latent_dim]
        
        return z_d, attn_weights, mu, logvar


# ============== FULL MODEL ==============
class CD_Model(nn.Module):
    def __init__(self, num_classes=2, d_model=768, latent_dim=64, freeze_bert=False):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model  # 768 for BERT base
        self.latent_dim = latent_dim
        
        # One shared BERT encoder
        print("Loading pretrained BERT-base-uncased (shared encoder)...")
        self.bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
        self.use_bert = True  # Always use BERT (for visualization compatibility)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("BERT loaded and frozen")
        else:
            print("BERT loaded and trainable")
        
        # Two separate heads: Content head and Demographic head
        self.content_head = ContentHead(d_model=d_model, latent_dim=latent_dim)
        self.demographic_head = DemographicHead(d_model=d_model, latent_dim=latent_dim)
        
        # Classifier for task prediction from z_c
        self.classifier = TaskClassifier(latent_dim=latent_dim, num_classes=num_classes)  # Predicts y from z_c (Content - should work well)
        
        # Separate decoders: D_c(z_c) and D_d(z_d) for reconstruction
        # This prevents z_d from collapsing to noise/zero
        self.decoder_c = ContentDecoder(latent_dim=latent_dim, embed_dim=d_model)
        self.decoder_d = DemographicDecoder(latent_dim=latent_dim, embed_dim=d_model)
        
        # Adversarial classifier for monitoring only (not used in training loss)
        # Kept for backward compatibility and monitoring z_d → y accuracy
        self.adversarial_classifier = AdversarialClassifier(latent_dim=latent_dim, num_classes=num_classes)
    
    def forward(self, input_ids, attention_mask, bert_embeddings=None):
        """
        Forward pass with shared BERT encoder + 2 separate heads (LIRD framework).
        
        Uses HSIC for independence constraints (no GRL/adversarial training).
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            bert_embeddings: Not used (kept for compatibility)
        """
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Shared BERT encoder
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_output.last_hidden_state  # [batch_size, seq_len, 768]
        
        # Extract z_c and z_d using separate heads with VAE bottleneck
        # Heads will handle masking internally (mask attention logits before softmax)
        z_c, attn_c, mu_c, logvar_c = self.content_head(token_embeddings, attention_mask)  # [batch_size, latent_dim]
        z_d, attn_d, mu_d, logvar_d = self.demographic_head(token_embeddings, attention_mask)  # [batch_size, latent_dim]
        
        # Get CLS token for reconstruction target
        original_h = token_embeddings[:, 0, :]  # CLS token [batch_size, 768]
        
        # Task prediction from z_c (Content - should work well)
        logits_y_from_z_c = self.classifier(z_c)
        
        # Separate decoders: D_c(z_c) and D_d(z_d)
        # Reconstruction: D_c(z_c) + D_d(z_d) ≈ original_h
        reconstructed_c = self.decoder_c(z_c)  # [batch_size, embed_dim]
        reconstructed_d = self.decoder_d(z_d)  # [batch_size, embed_dim]
        reconstructed = reconstructed_c + reconstructed_d  # Additive reconstruction
        
        # Adversarial classifier for monitoring only (not used in training loss)
        # This is only for tracking z_d → y accuracy during training
        logits_y_from_z_d = self.adversarial_classifier(z_d)
        
        return {
            'z_c': z_c, 'z_d': z_d,
            'mu_c': mu_c, 'logvar_c': logvar_c,  # VAE parameters for KL loss
            'mu_d': mu_d, 'logvar_d': logvar_d,
            'logits_y_from_z_c': logits_y_from_z_c,  # Main task prediction from z_c
            'logits_y_from_z_d': logits_y_from_z_d,  # Monitoring only: z_d → y (should be random)
            'reconstructed': reconstructed,  # D_c(z_c) + D_d(z_d)
            'reconstructed_c': reconstructed_c,  # D_c(z_c) - for monitoring
            'reconstructed_d': reconstructed_d,  # D_d(z_d) - for monitoring and constraint
            'original_h': original_h,  # For reconstruction loss
            'attn_c': attn_c,  # Attention weights for z_c (for visualization)
            'attn_d': attn_d   # Attention weights for z_d (for visualization)
        }
