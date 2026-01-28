#!/usr/bin/env python3
"""
Fairness learning with disentangled representations from Natural Language:
1. Shared BERT encoder + 2 separate heads (Content head & Demographic head)
2. Attention pooling in heads to extract z_c and z_d from shared BERT output
3. Constraints applied on latent representations:
   - Orthogonality: z_c and z_d should be orthogonal (HSIC + Distance Correlation)
   - Adversarial: z_d should NOT predict y (random guess ~50/50)
   - Reconstruction: z_c + z_d should reconstruct original representation
   - Attention diversity: heads should attend to different tokens
   
Key Objectives:
- z_c (Content): Can predict y well (high accuracy)
- z_d (Demographic): Cannot predict y (accuracy ~50/50, random guess)
- z_c ⊥ z_d: Orthogonal/Independent (no overlap)
- z_c + z_d → x: Can reconstruct original representation
- Attention visualization: See which tokens each head focuses on
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
from data_loader import load_german_credit_data_balanced, load_dataset_generic
import math
from visualization import (
    visualize_reconstruction_and_attention, 
    visualize_batch_analysis, 
    visualize_attention_heatmap_batch, 
    analyze_attention_difference, 
    visualize_latent_space_pca,
    visualize_attention_on_text,
    visualize_attention_comparison
)
import datetime
import os
import numpy as np
import argparse
import hashlib
import pickle
import torch.nn.functional as F

# Auto-detect device: use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============== TOKENIZATION CACHE ==============
def get_cache_path(sentences, dataset_name, cache_dir="bert_cache"):
    """
    Generate cache file path based on dataset content.
    Cache is used to speed up tokenization - only token IDs are cached, not embeddings.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Create hash from sentences to ensure cache validity
    content = "".join(sentences)
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"{dataset_name}_{content_hash}.pkl")
    return cache_file

def compute_and_cache_tokenized_data(sentences, dataset_name, max_len=100, force_recompute=False):
    """
    Pre-compute tokenized data for all sentences and cache them for faster training.
    This significantly speeds up training by avoiding repeated tokenization.
    
    Caches:
        - input_ids: Token IDs [n_samples, seq_len]
        - attention_mask: Attention masks [n_samples, seq_len]
    
    Returns:
        cached_data: dict with 'input_ids', 'attention_mask'
    """
    cache_file = get_cache_path(sentences, dataset_name)
    
    # Check if cache exists
    if os.path.exists(cache_file) and not force_recompute:
        print(f"✓ Loading tokenized data from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            # Verify cache has required keys
            if 'input_ids' in cached_data and 'attention_mask' in cached_data:
                print(f"✓ Loaded {len(cached_data['input_ids'])} cached tokenizations")
                return cached_data
            else:
                print(f"⚠ Cache file exists but missing required keys. Recomputing...")
        except Exception as e:
            print(f"⚠ Error loading cache: {e}. Recomputing...")
    
    # Tokenize sentences (only if cache doesn't exist or force_recompute=True)
    print(f"Tokenizing {len(sentences)} sentences...")
    print("This may take a few minutes. Results will be cached for faster future runs.")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Use BERT tokenizer for vocab
    
    input_ids_list = []
    attention_mask_list = []
    
    batch_size = 32  # Process in batches for efficiency
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        tokens = tokenizer(
            batch_sentences, 
            padding="max_length", 
            truncation=True,
            max_length=max_len, 
            return_tensors="pt"
        )
        input_ids_list.append(tokens.input_ids.cpu())
        attention_mask_list.append(tokens.attention_mask.cpu())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(sentences))}/{len(sentences)} sentences...")
    
    # Concatenate all batches
    all_input_ids = torch.cat(input_ids_list, dim=0)
    all_attention_mask = torch.cat(attention_mask_list, dim=0)
    
    cached_data = {
        'input_ids': all_input_ids,    # [n_samples, seq_len]
        'attention_mask': all_attention_mask,  # [n_samples, seq_len]
    }
    
    # Save to cache for faster future runs
    print(f"Saving tokenized data to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"✓ Successfully cached {len(all_input_ids)} tokenizations")
        print(f"  Cache file: {cache_file}")
        print(f"  Next run will use cached data (much faster!)")
    except Exception as e:
        print(f"⚠ Warning: Could not save cache: {e}")
        print(f"  Training will continue but tokenization will be slower")
    
    return cached_data

# ============== DATASET ==============
class TextDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=256, cached_data=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cached_data = cached_data
        self.use_cache = cached_data is not None

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.use_cache:
            # Use cached tokenized data
            return {
                "input_ids": self.cached_data['input_ids'][idx],
                "attention_mask": self.cached_data['attention_mask'][idx],
                "label": torch.tensor(label, dtype=torch.long),
                "text": self.sentences[idx]
            }
        else:
            # Compute on-the-fly
            text = self.sentences[idx]
            tokens = self.tokenizer(
                text, padding="max_length", truncation=True,
                max_length=self.max_len, return_tensors="pt"
            )
            return {
                "input_ids": tokens.input_ids.squeeze(0),
                "attention_mask": tokens.attention_mask.squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
                "text": text
            }

# ============== TRANSFORMER COMPONENTS ==============
# ============== MODELS ==============
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


class CDDecoder(nn.Module):
    """Reconstructs BERT embedding from (z_c, z_d)"""
    def __init__(self, latent_dim=768, embed_dim=768):
        super().__init__()
        self.latent_dim = latent_dim
        # z_c and z_d are both 768-dim, so input is 2*768 = 1536
        self.fc = nn.Sequential(
            nn.Linear(2 * latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, embed_dim)
        )
    
    def forward(self, z_c, z_d):
        z = torch.cat([z_c, z_d], dim=1)
        reconstructed = self.fc(z)
        return reconstructed


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
        
        # Classifiers work on latent representations
        self.classifier = TaskClassifier(latent_dim=latent_dim, num_classes=num_classes)  # Predicts y from z_c (Content - should work well)
        self.adversarial_classifier = AdversarialClassifier(latent_dim=latent_dim, num_classes=num_classes)  # Tries to predict y from z_d (Demographic - we want this to fail)
        
        # Gradient Reversal Layer for adversarial training
        # lambda_grl controls the strength of gradient reversal (can be scheduled)
        self.grl = GRL(lambda_grl=1.0)
        
        # Decoder to reconstruct from z_c + z_d (optional, for reconstruction loss)
        self.decoder = CDDecoder(latent_dim=latent_dim, embed_dim=d_model)
    
    def forward(self, input_ids, attention_mask, bert_embeddings=None):
        """
        Forward pass with shared BERT encoder + 2 separate heads.
        
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
        
        # Predictions
        logits_y_from_z_c = self.classifier(z_c)  # Predict y from z_c (Content - should work well)
        
        # Apply GRL to z_d before adversarial classifier
        # Forward: z_d passes through unchanged
        # Backward: gradient is reversed, so encoder maximizes adversarial loss
        z_d_grl = self.grl(z_d)
        logits_y_from_z_d = self.adversarial_classifier(z_d_grl)  # Try to predict y from z_d (Demographic - should fail)
        
        # Reconstruction from z_c + z_d
        reconstructed = self.decoder(z_c, z_d)
        
        return {
            'z_c': z_c, 'z_d': z_d,
            'mu_c': mu_c, 'logvar_c': logvar_c,  # VAE parameters for KL loss
            'mu_d': mu_d, 'logvar_d': logvar_d,
            'logits_y_from_z_c': logits_y_from_z_c,  # Main task prediction from z_c (Content - should work well)
            'logits_y_from_z_d': logits_y_from_z_d,  # Adversarial prediction from z_d (Demographic - should fail)
            'reconstructed': reconstructed,
            'original_h': original_h,  # For reconstruction loss
            'attn_c': attn_c,  # Attention weights for z_c (for visualization)
            'attn_d': attn_d   # Attention weights for z_d (for visualization)
        }


# ============== LOSS FUNCTIONS ==============
def compute_kl_loss(mu, logvar):
    """
    Compute KL divergence loss: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    Normalized by batch size to match scale of other losses (CE, MSE, etc.)
    """
    # KL divergence per sample: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # Shape: [batch_size, latent_dim] -> [batch_size] -> scalar (mean)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(kl_per_sample)  # Normalize by batch size

def compute_reconstruction_loss(original_h, reconstructed_h):
    return nn.functional.mse_loss(reconstructed_h, original_h)

def compute_orthogonality_loss(z_c, z_d):
    """
    Direct orthogonal constraint: z_c^T * z_d should be close to zero.
    This enforces orthogonality directly in the latent space.
    """
    # Normalize z_c and z_d
    z_c_norm = torch.nn.functional.normalize(z_c, p=2, dim=1)
    z_d_norm = torch.nn.functional.normalize(z_d, p=2, dim=1)
    
    # Compute dot product (should be 0 for orthogonal vectors)
    dot_product = torch.sum(z_c_norm * z_d_norm, dim=1)  # [batch_size]
    
    # Loss is the squared dot product (want it to be 0)
    orthogonality_loss = torch.mean(dot_product ** 2)
    return orthogonality_loss

def compute_separation_loss(z, labels, margin=2.0):
    """
    Contrastive/separation loss to encourage better class separation in latent space.
    
    Objective:
    - Points with same label should be close together in latent space
    - Points with different labels should be far apart in latent space
    
    This helps the latent representation learn better discriminative features for task prediction.
    
    Args:
        z: Latent representation [batch_size, latent_dim] (can be z_c or z_d)
        labels: True labels [batch_size]
        margin: Margin for different-label pairs
    """
    batch_size = z.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=z.device)
    
    # Normalize z for cosine similarity
    z_norm = torch.nn.functional.normalize(z, p=2, dim=1)  # [batch_size, latent_dim]
    
    # Compute pairwise cosine similarities
    # z_norm @ z_norm.T gives [batch_size, batch_size] similarity matrix
    similarity_matrix = torch.matmul(z_norm, z_norm.T)  # [batch_size, batch_size]
    
    # Create label mask: 1 if same label, 0 if different
    labels_expanded = labels.unsqueeze(1)  # [batch_size, 1]
    label_mask = (labels_expanded == labels_expanded.T).float()  # [batch_size, batch_size]
    
    # Remove diagonal (self-similarity)
    mask = torch.eye(batch_size, device=z.device)
    label_mask = label_mask * (1 - mask)  # Remove self-pairs
    
    # Loss: 
    # - Maximize similarity for same-label pairs (minimize 1 - similarity)
    # - Minimize similarity for different-label pairs (maximize margin - similarity)
    same_label_sim = similarity_matrix * label_mask  # Only same-label pairs
    diff_label_sim = similarity_matrix * (1 - label_mask)  # Only different-label pairs
    
    # Count valid pairs
    num_same_pairs = label_mask.sum().clamp(min=1)
    num_diff_pairs = ((1 - label_mask) * (1 - mask)).sum().clamp(min=1)
    
    # Pull same-label pairs together (maximize similarity → minimize (1 - similarity))
    pull_loss = (1 - same_label_sim).sum() / num_same_pairs
    
    # Push different-label pairs apart (minimize similarity → maximize (margin - similarity), but only if similarity > margin)
    push_loss = torch.clamp(margin - diff_label_sim, min=0).sum() / num_diff_pairs
    
    separation_loss = pull_loss + push_loss
    
    return separation_loss


# ============== TRAINING ==============
def train_cd_model(sentences, labels, num_epochs=50, batch_size=8, lambda_task=3.0, lambda_adv=3.0, lambda_rec=0.5, lambda_ortho=1.0, lambda_kl=0.01, latent_dim=64, lr=2e-5, use_cache=True, dataset_name="default", force_recompute_cache=False, use_amp=False):
    """
    Train the model to learn disentangled representations from Natural Language.
    
    Architecture:
    - 1 shared BERT encoder (bert-base-uncased)
    - 2 separate heads: ContentHead (→ z_c) and DemographicHead (→ z_d)
    - Each head uses attention pooling + VAE bottleneck (for compression only)
    
    4 MAIN OBJECTIVES (4 losses):
    1. L_task: z_c → y (high accuracy) - z_c contains task-relevant info
    2. L_adv: z_d → y (via GRL, ~50%) - z_d does NOT contain task info
    3. L_ortho: z_c ⊥ z_d - z_c and z_d are independent/orthogonal
    4. L_rec: z_c + z_d → x - both can reconstruct original representation
    
    + L_kl: VAE regularization (compression + regularization only, small weight)
    
    NOTE: VAE is used for compression (768 → latent_dim), NOT for generation.
    """
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Latent dim: {latent_dim}")
    print(f"\n4 MAIN LOSSES:")
    print(f"  1. λ_task={lambda_task} : z_c → y (target: high accuracy ~1.0)")
    print(f"  2. λ_adv={lambda_adv} : z_d → y via GRL (target: random ~0.5)")
    print(f"  3. λ_ortho={lambda_ortho} : z_c ⊥ z_d (target: ~0)")
    print(f"  4. λ_rec={lambda_rec} : z_c + z_d → x (target: ~0)")
    print(f"\n  + λ_kl={lambda_kl} : VAE regularization (compression only)")
    print(f"{'='*70}\n")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Pre-compute and cache tokenized data if requested (speeds up training significantly)
    cached_data = None
    if use_cache:
        print(f"\n{'='*70}")
        print("CACHE SETUP - Tokenizing and caching data for faster training")
        print(f"{'='*70}")
        cached_data = compute_and_cache_tokenized_data(
            sentences, dataset_name, max_len=50, force_recompute=force_recompute_cache
        )
        print(f"{'='*70}\n")
    else:
        print("⚠ Cache disabled - tokenization will happen on-the-fly (slower)")
    
    dataset = TextDataset(sentences, labels, tokenizer, max_len=50, cached_data=cached_data)
    
    # Compute class weights for imbalanced dataset (generalized for any number of classes)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    unique_labels = torch.unique(labels_tensor)
    num_classes = len(unique_labels)
    
    # Compute class counts (handle any number of classes)
    max_label = labels_tensor.max().item()
    class_counts = torch.bincount(labels_tensor, minlength=max_label + 1)
    total_samples = len(labels)
    
    # Weight inversely proportional to class frequency (generalized formula)
    # Formula: weight_i = total_samples / (num_classes * count_i)
    class_weights = total_samples / (num_classes * class_counts.float())
    class_weights = class_weights.to(device)
    
    # Print class distribution and weights (generalized for any number of classes)
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution:")
    imbalance_ratio = class_counts.max().float() / class_counts[class_counts > 0].min().float()
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x (max/min class size)")
    for i, count in enumerate(class_counts):
        if count.item() > 0:  # Only print classes that exist
            print(f"  Class {i}: {count.item()} samples ({100*count.item()/total_samples:.1f}%) - weight: {class_weights[i]:.4f}")
    
    # Create weighted sampler for balanced batch sampling (helps with imbalanced datasets)
    # Assign weight to each sample based on its class
    sample_weights = class_weights[labels_tensor].cpu()
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow replacement to balance classes
    )
    
    # Use weighted sampler instead of shuffle=True for better handling of imbalanced data
    # Optimize DataLoader for GPU: use multiple workers and pin memory
    # RTX 5090 (31GB VRAM) can handle more workers and larger batches
    if torch.cuda.is_available():
        # Get GPU memory to optimize settings
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb >= 30:  # High-end GPU (RTX 5090, A100, etc.)
            num_workers = 8  # More workers for faster data loading
            prefetch_factor = 4  # Prefetch more batches
        else:
            num_workers = 4
            prefetch_factor = 2
        pin_memory = True
    else:
        num_workers = 0
        prefetch_factor = 2
        pin_memory = False
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=weighted_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=prefetch_factor if num_workers > 0 else None  # Prefetch batches
    )
    
    # Get vocab size from tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Use shared BERT + 2 heads with VAE bottleneck (trainable, not frozen)
    # VAE bottleneck compresses from 768 to latent_dim for better disentanglement
    model = CD_Model(num_classes=num_classes, d_model=768, latent_dim=latent_dim, freeze_bert=False).to(device)
    
    # Compile model for faster training (PyTorch 2.0+)
    # NOTE: torch.compile() can cause issues with gradient tracking when freezing/unfreezing parameters
    # Disable for now to avoid "does not require grad" errors
    # TODO: Re-enable with proper handling of requires_grad changes
    use_compile = False  # Set to True if you want to try compilation (may cause gradient issues)
    if use_compile:
        try:
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                print("✓ Compiling model with torch.compile() for faster training...")
                model = torch.compile(model, mode='reduce-overhead')  # Fastest mode for training
        except Exception as e:
            print(f"⚠ Could not compile model (PyTorch version may be < 2.0): {e}")
            print("  Training will continue without compilation (slightly slower)")
    else:
        print("ℹ Model compilation disabled (to avoid gradient tracking issues with freeze/unfreeze)")
    
    # Optimizers with different learning rates
    # BERT needs smaller LR, heads/decoder can use slightly higher
    # Separate BERT optimizer with smaller LR (BERT standard: 1e-5 to 5e-5)
    bert_lr = lr  # Use provided LR (should be ~2e-5 for BERT)
    head_lr = lr * 2.0  # Heads can learn faster
    
    optimizer = optim.AdamW([
        {'params': model.bert.parameters(), 'lr': bert_lr},
        {'params': list(model.content_head.parameters()) + 
                  list(model.demographic_head.parameters()) + 
                  list(model.decoder.parameters()), 'lr': head_lr}
    ], weight_decay=0.01)  # AdamW with weight decay for better BERT training
    
    # Separate optimizer for classifier with higher LR for faster learning
    optimizer_classifier = optim.Adam(
        model.classifier.parameters(),
        lr=lr * 10.0  # Much higher LR for classifier to learn task quickly
    )
    # Adversarial classifier with higher LR to learn faster
    optimizer_adv = optim.Adam(model.adversarial_classifier.parameters(), lr=lr * 10.0)
    
    # Learning rate schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    scheduler_classifier = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_classifier, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    scheduler_adv = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_adv, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Mixed Precision Training (AMP) for faster training on GPU
    scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None
    if use_amp and torch.cuda.is_available():
        print("✓ Mixed Precision Training (AMP) enabled - faster training on GPU")
        # Enable cuDNN benchmarking for consistent input sizes (faster on RTX 5090)
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN benchmarking enabled - faster convolutions on GPU")
    
    history = {
        'epoch': [], 'loss_enc': [], 
        'loss_task': [], 'loss_adv': [], 'loss_ortho': [], 'loss_rec': [],  # 4 main losses
        'loss_kl': [],  # VAE regularization (compression only)
        'acc_task_from_z_d': [], 'acc_task_from_z_c': [],
        'per_class_acc_z_d': [], 'per_class_acc_z_c': []
    }
    
    # Initialize per-class accuracy accumulators (for imbalanced task labels)
    class_correct_z_d = torch.zeros(num_classes, dtype=torch.long)
    class_total_z_d = torch.zeros(num_classes, dtype=torch.long)
    class_correct_z_c = torch.zeros(num_classes, dtype=torch.long)
    class_total_z_c = torch.zeros(num_classes, dtype=torch.long)
    
    best_loss = float('inf')
    patience_count = 0
    patience = 10
    best_model_state = None  # Store best model state
    best_model_epoch = 0
    
    print(f"Dataset: {len(sentences)} samples, {sum(labels)} good, {len(labels) - sum(labels)} bad")
    
    for epoch in range(num_epochs):
        # Initialize accumulators for this epoch
        loss_enc_total = 0.0
        loss_task_total = 0.0
        loss_rec_total = 0.0
        loss_kl_total = 0.0
        loss_ortho_total = 0.0
        loss_adv_total = 0.0
        correct_task_from_z_d = 0
        correct_task_from_z_c = 0
        total = 0
        attn_entropy_c_total = 0.0
        attn_entropy_d_total = 0.0
        attn_entropy_count = 0
        attn_diversity_total = 0.0
        attn_diversity_count = 0
        model.train()
        loss_enc_total = loss_task_total = loss_adv_total = loss_ortho_total = loss_rec_total = 0
        loss_kl_total = 0
        correct_task_from_z_d = correct_task_from_z_c = total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Use non_blocking=True for faster GPU transfer (overlaps with computation)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True)
                
                # Get output (includes original_h for reconstruction)
                # No need to get separately, it's in output['original_h']
                
                # === ADVERSARIAL CLASSIFIER STEP ===
                # Train adversarial classifier to predict y from z_d (Demographic)
                # This classifier tries to find ANY task information in z_d
                # We train it normally (minimize CE loss) so it becomes strong
                # The encoder will learn to hide this information via GRL (gradient reversal)
                
                # Ensure adversarial_classifier is trainable (unfreeze if needed)
                # This is critical - must be done before forward pass
                for param in model.adversarial_classifier.parameters():
                    param.requires_grad = True
                
                optimizer_adv.zero_grad()
                
                # Forward pass - get z_d from model
                with torch.set_grad_enabled(True):  # Ensure gradients are enabled
                    output_adv = model(input_ids, attention_mask)
                    z_d_detached = output_adv['z_d'].detach()  # Detach so encoder doesn't get gradient in this step
                    
                    # Forward through adversarial classifier
                    # This should create gradient graph for adversarial_classifier parameters
                    logits_y_from_z_d_adv = model.adversarial_classifier(z_d_detached)
                    
                    # Verify adversarial_classifier parameters require grad
                    has_grad = any(p.requires_grad for p in model.adversarial_classifier.parameters())
                    if not has_grad:
                        raise RuntimeError("adversarial_classifier parameters do not require grad! Check model setup.")
                    
                    # Use class weights for adversarial classifier too
                    loss_adv_classifier = nn.functional.cross_entropy(
                        logits_y_from_z_d_adv, 
                        labels_batch,
                        weight=class_weights
                    )
                    # Add entropy penalty to encourage confident predictions (if info exists)
                    # This helps encoder learn to remove that information
                    probs = torch.softmax(logits_y_from_z_d_adv, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                    loss_adv_classifier = loss_adv_classifier - 0.1 * entropy  # Encourage confident predictions
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss_adv_classifier).backward()
                    scaler.step(optimizer_adv)
                    scaler.update()
                else:
                    loss_adv_classifier.backward()
                    optimizer_adv.step()
                
                # === MAIN TRAINING STEP ===
                # NOTE: Model does NOT know sensitive attributes. It learns separation through:
                # 1. Adversarial training via GRL: encoder minimizes task loss on z_c, maximizes adv loss on z_d
                # 2. Orthogonality: forces z_c and z_d to be different
                # 3. Reconstruction: ensures both contain useful information
                #
                # GRL mechanism:
                # - Forward: z_d passes through GRL unchanged → adversarial classifier gets z_d
                # - Backward: gradient is REVERSED → encoder maximizes adversarial loss (hides task info in z_d)
                # - Adversarial classifier trains separately (minimizes CE) to become strong
                #
                # IMPORTANT: Freeze adversarial_classifier during encoder step to avoid unnecessary backprop
                # We only need gradient w.r.t. z_d (via GRL), not w.r.t. adversarial_classifier parameters
                for param in model.adversarial_classifier.parameters():
                    param.requires_grad = False
                
                optimizer.zero_grad()
                # Forward pass
                output = model(input_ids, attention_mask)
                
                # ============ 4 MAIN LOSSES ============
                # 1. L_task: z_c → y (MINIMIZE) - z_c should predict y well
                loss_task = nn.functional.cross_entropy(
                    output['logits_y_from_z_c'], 
                    labels_batch,
                    weight=class_weights
                )
                
                # 2. L_adv: z_d → y via GRL (MAXIMIZE via gradient reversal)
                # GRL reverses gradient → encoder learns to hide task info in z_d
                loss_adv = nn.functional.cross_entropy(
                    output['logits_y_from_z_d'], 
                    labels_batch,
                    weight=class_weights
                )
                
                # 3. L_ortho: z_c ⊥ z_d (MINIMIZE) - z_c and z_d should be orthogonal
                loss_ortho = compute_orthogonality_loss(output['z_c'], output['z_d'])
                
                # 4. L_rec: z_c + z_d → x (MINIMIZE) - reconstruction
                loss_rec = compute_reconstruction_loss(output['original_h'], output['reconstructed'])
                
                # + L_kl: VAE regularization (compression only, small weight)
                loss_kl_c = compute_kl_loss(output['mu_c'], output['logvar_c'])
                loss_kl_d = compute_kl_loss(output['mu_d'], output['logvar_d'])
                loss_kl = loss_kl_c + loss_kl_d
                
                # ============ TRAIN CLASSIFIER SEPARATELY (MULTIPLE STEPS) ============
                # Train classifier multiple times per batch to learn faster
                # This helps classifier learn task before encoder tries to optimize z_c
                num_classifier_steps = 3  # Train classifier 3 times per batch
                for _ in range(num_classifier_steps):
                    optimizer_classifier.zero_grad()
                    z_c_detached = output['z_c'].detach()  # Detach to avoid gradient conflict
                    logits_y_from_z_c_detached = model.classifier(z_c_detached)
                    loss_task_classifier = nn.functional.cross_entropy(
                        logits_y_from_z_c_detached, 
                        labels_batch,
                        weight=class_weights
                    )
                    if scaler is not None:
                        scaler.scale(loss_task_classifier).backward()
                        scaler.step(optimizer_classifier)
                        scaler.update()
                    else:
                        loss_task_classifier.backward()
                        optimizer_classifier.step()
                
                # Recompute loss_task with updated classifier (now well-trained)
                logits_y_from_z_c_updated = model.classifier(output['z_c'])
                loss_task_updated = nn.functional.cross_entropy(
                    logits_y_from_z_c_updated,
                    labels_batch,
                    weight=class_weights
                )
                
                # ============ COMBINED ENCODER LOSS (4 main + 1 regularization) ============
                # 1. L_task: z_c → y (MINIMIZE → Acc ~1.0)
                # 2. L_adv: z_d → y via GRL (MAXIMIZE → Acc ~0.5)
                # 3. L_ortho: z_c ⊥ z_d (MINIMIZE → 0)
                # 4. L_rec: z_c + z_d → x (MINIMIZE → 0)
                # + L_kl: VAE regularization (compression only)
                
                loss_enc = (
                    lambda_task * loss_task_updated +    # 1. z_c → y
                    lambda_adv * loss_adv +              # 2. z_d → y (via GRL)
                    lambda_ortho * loss_ortho +          # 3. z_c ⊥ z_d
                    lambda_rec * loss_rec +              # 4. reconstruction
                    lambda_kl * loss_kl                  # VAE regularization
                )
                
                # Train BERT_c, BERT_d and decoder with full loss (using AMP if enabled)
                if scaler is not None:
                    scaler.scale(loss_enc).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_enc.backward()
                    optimizer.step()
                
                # Restore adversarial_classifier gradients (for next adversarial step)
                for param in model.adversarial_classifier.parameters():
                    param.requires_grad = True
                
                # Accumulate losses (4 main + 1 regularization)
                loss_enc_total += loss_enc.item()
                loss_task_total += loss_task_updated.item()
                loss_adv_total += loss_adv.item()
                loss_ortho_total += loss_ortho.item()
                loss_rec_total += loss_rec.item()
                loss_kl_total += loss_kl.item()
                
                # Track attention entropy and diversity for monitoring
                if output.get('attn_c') is not None and output.get('attn_d') is not None:
                    attn_c = output['attn_c']
                    attn_d = output['attn_d']
                    entropy_c = -torch.sum(attn_c * torch.log(attn_c + 1e-8), dim=1).mean().item()
                    entropy_d = -torch.sum(attn_d * torch.log(attn_d + 1e-8), dim=1).mean().item()
                    attn_entropy_c_total += entropy_c
                    attn_entropy_d_total += entropy_d
                    attn_entropy_count += 1
                    
                    # Track attention diversity (similarity between z_c and z_d attention)
                    attn_c_norm = torch.nn.functional.normalize(attn_c, p=2, dim=1)
                    attn_d_norm = torch.nn.functional.normalize(attn_d, p=2, dim=1)
                    attn_similarity = torch.sum(attn_c_norm * attn_d_norm, dim=1).mean().item()
                    attn_diversity_total += attn_similarity
                    attn_diversity_count += 1
                
                with torch.no_grad():
                    # Use updated classifier predictions for accuracy (not the old output)
                    pred_z_d = output['logits_y_from_z_d'].argmax(1)
                    pred_z_c = logits_y_from_z_c_updated.argmax(1)  # Use updated classifier
                    correct_task_from_z_d += (pred_z_d == labels_batch).sum().item()
                    correct_task_from_z_c += (pred_z_c == labels_batch).sum().item()
                    total += labels_batch.size(0)
                    
                    # Compute per-class accuracy (average accuracy across task label classes)
                    # NOTE: This is NOT balanced accuracy w.r.t. sensitive attributes (we don't know them)
                    # This is balanced accuracy w.r.t. task labels (y) - useful for imbalanced task labels
                    # Reset accumulators at start of each epoch
                    if batch_idx == 0:
                        class_correct_z_d.zero_()
                        class_total_z_d.zero_()
                        class_correct_z_c.zero_()
                        class_total_z_c.zero_()
                    
                    # Accumulate per-class correct predictions (by task label y, not sensitive attributes)
                    for c in range(num_classes):
                        mask_d = (labels_batch == c)
                        mask_c = (labels_batch == c)
                        if mask_d.sum() > 0:
                            class_correct_z_d[c] += (pred_z_d[mask_d] == labels_batch[mask_d]).sum().item()
                            class_total_z_d[c] += mask_d.sum().item()
                        if mask_c.sum() > 0:
                            class_correct_z_c[c] += (pred_z_c[mask_c] == labels_batch[mask_c]).sum().item()
                            class_total_z_c[c] += mask_c.sum().item()
                    
                    # Debug: Check if model is just predicting majority class
                    if batch_idx == 0 and epoch == 0:
                        max_pred = max(pred_z_d.max().item(), pred_z_c.max().item(), labels_batch.max().item())
                        minlength = max_pred + 1
                        print(f"\nDEBUG - First batch predictions:")
                        print(f"  Labels: {labels_batch.cpu().numpy()[:10]}")
                        print(f"  Pred z_d: {pred_z_d.cpu().numpy()[:10]}")
                        print(f"  Pred z_c: {pred_z_c.cpu().numpy()[:10]}")
                        print(f"  Pred z_d distribution: {torch.bincount(pred_z_d, minlength=minlength).cpu().numpy()}")
                        print(f"  Pred z_c distribution: {torch.bincount(pred_z_c, minlength=minlength).cpu().numpy()}")
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Average and record (4 main losses + 1 regularization)
        num_batches = len(dataloader)
        history['epoch'].append(epoch)
        history['loss_enc'].append(loss_enc_total / num_batches)
        history['loss_task'].append(loss_task_total / num_batches)
        history['loss_adv'].append(loss_adv_total / num_batches)
        history['loss_ortho'].append(loss_ortho_total / num_batches)
        history['loss_rec'].append(loss_rec_total / num_batches)
        history['loss_kl'].append(loss_kl_total / num_batches)
        history['acc_task_from_z_d'].append(correct_task_from_z_d / total if total > 0 else 0)
        history['acc_task_from_z_c'].append(correct_task_from_z_c / total if total > 0 else 0)
        
        # Compute per-class accuracy (average accuracy across task label classes)
        # NOTE: This is balanced accuracy w.r.t. task labels (y), NOT sensitive attributes
        # We cannot compute balanced accuracy w.r.t. sensitive attributes because we don't know them
        per_class_acc_z_d = 0.0
        per_class_acc_z_c = 0.0
        if class_total_z_d.sum() > 0:
            per_class_acc_z_d_vals = class_correct_z_d.float() / (class_total_z_d.float() + 1e-8)
            per_class_acc_z_d = per_class_acc_z_d_vals.mean().item()
        if class_total_z_c.sum() > 0:
            per_class_acc_z_c_vals = class_correct_z_c.float() / (class_total_z_c.float() + 1e-8)
            per_class_acc_z_c = per_class_acc_z_c_vals.mean().item()
        
        history['per_class_acc_z_d'].append(per_class_acc_z_d)
        history['per_class_acc_z_c'].append(per_class_acc_z_c)
        
        avg_loss_enc = history['loss_enc'][-1]
        
        # Update learning rate schedulers
        scheduler.step(avg_loss_enc)
        scheduler_classifier.step(history['loss_task'][-1])  # Use task loss for classifier scheduler
        scheduler_adv.step(history['loss_adv'][-1])
        current_lr = optimizer.param_groups[0]['lr']
        classifier_lr = optimizer_classifier.param_groups[0]['lr']
        
        # Clear GPU cache periodically to prevent memory fragmentation
        if torch.cuda.is_available() and epoch % 5 == 0:
            torch.cuda.empty_cache()
        
        # Print every epoch (always print all epochs for monitoring)
        random_target_str = f"(→{1.0/num_classes:.3f})" if num_classes > 2 else "(→0.5)"
        per_class_acc_z_d_str = f"PerCls: {history['per_class_acc_z_d'][-1]:.3f}" if len(history['per_class_acc_z_d']) > 0 else ""
        per_class_acc_z_c_str = f"PerCls: {history['per_class_acc_z_c'][-1]:.3f}" if len(history['per_class_acc_z_c']) > 0 else ""
        
        # Print detailed epoch info (4 main losses)
        print(f"\nEpoch {epoch:3d}/{num_epochs}")
        print(f"  4 Main Losses: L_task={history['loss_task'][-1]:.4f} | L_adv={history['loss_adv'][-1]:.4f} | L_ortho={history['loss_ortho'][-1]:.4f} | L_rec={history['loss_rec'][-1]:.4f}")
        print(f"  Regularization: L_kl={history['loss_kl'][-1]:.4f} | Total L_enc={avg_loss_enc:.4f}")
        # Show weighted contributions for debugging
        weighted_task = lambda_task * history['loss_task'][-1]
        weighted_adv = lambda_adv * history['loss_adv'][-1]
        weighted_ortho = lambda_ortho * history['loss_ortho'][-1]
        weighted_rec = lambda_rec * history['loss_rec'][-1]
        weighted_kl = lambda_kl * history['loss_kl'][-1]
        print(f"  Weighted contributions: λ_task*L_task={weighted_task:.4f} | λ_adv*L_adv={weighted_adv:.4f} | λ_ortho*L_ortho={weighted_ortho:.4f} | λ_rec*L_rec={weighted_rec:.4f} | λ_kl*L_kl={weighted_kl:.4f}")
        print(f"  Accuracy: Acc(z_c→y)={history['acc_task_from_z_c'][-1]:.3f} {per_class_acc_z_c_str} (→1.0)")
        print(f"            Acc(z_d→y)={history['acc_task_from_z_d'][-1]:.3f} {per_class_acc_z_d_str} {random_target_str}")
        print(f"  LR: encoder={current_lr:.6f}, classifier={classifier_lr:.6f}")
        
        # Early stopping and save best model
        if avg_loss_enc < best_loss:
            best_loss = avg_loss_enc
            patience_count = 0
            # Save best model state
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss_enc,
                'acc_z_c': history['acc_task_from_z_c'][-1],
                'acc_z_d': history['acc_task_from_z_d'][-1],
                'num_classes': num_classes,
                'latent_dim': latent_dim,
                'd_model': 768
            }
            best_model_epoch = epoch
            print(f"  ✓ New best model! Loss: {best_loss:.4f}, Acc(z_c): {history['acc_task_from_z_c'][-1]:.3f}")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    random_target = 1.0 / num_classes
    random_target_str = f"~{random_target:.3f}" if num_classes > 2 else "~0.5"
    per_class_acc_z_d_final = history['per_class_acc_z_d'][-1] if len(history['per_class_acc_z_d']) > 0 else 0.0
    per_class_acc_z_c_final = history['per_class_acc_z_c'][-1] if len(history['per_class_acc_z_c']) > 0 else 0.0
    
    print(f"\n4 MAIN OBJECTIVES - Final Results:")
    print(f"  1. L_task (z_c → y): Acc={history['acc_task_from_z_c'][-1]:.3f} (target: ~1.0)")
    print(f"  2. L_adv (z_d → y): Acc={history['acc_task_from_z_d'][-1]:.3f} (target: {random_target_str})")
    print(f"  3. L_ortho (z_c ⊥ z_d): {history['loss_ortho'][-1]:.4f} (target: ~0)")
    print(f"  4. L_rec (reconstruction): {history['loss_rec'][-1]:.4f} (target: ~0)")
    print(f"\n  + L_kl (VAE regularization): {history['loss_kl'][-1]:.4f}")
    print(f"\nNOTE: Per-class accuracy is w.r.t. task labels (y), NOT sensitive attributes")
    print(f"{'='*70}")
    
    # Save best model
    if best_model_state is not None:
        model_dir = f"models/{dataset_name}"
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_path = os.path.join(model_dir, f"best_model_{timestamp}.pth")
        
        # Save best model
        torch.save(best_model_state, best_model_path)
        print(f"\n✓ Best model saved to: {best_model_path}")
        print(f"  Epoch: {best_model_epoch}, Loss: {best_loss:.4f}")
        print(f"  Acc(z_c→y): {best_model_state['acc_z_c']:.3f}, Acc(z_d→y): {best_model_state['acc_z_d']:.3f}")
        
        # Also save final model
        final_model_path = os.path.join(model_dir, f"final_model_{timestamp}.pth")
        final_model_state = {
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'loss': history['loss_enc'][-1],
            'acc_z_c': history['acc_task_from_z_c'][-1],
            'acc_z_d': history['acc_task_from_z_d'][-1],
            'num_classes': num_classes,
            'latent_dim': latent_dim,
            'd_model': 768,
            'history': history  # Include full training history
        }
        torch.save(final_model_state, final_model_path)
        print(f"✓ Final model saved to: {final_model_path}")
    else:
        print("\n⚠ No best model to save (training may have failed)")
    
    return model, history


def save_encodings(model, sentences, labels, dataset_name, tokenizer, max_len=50, cached_data=None):
    """
    Save:
      - z_c, z_d (latent reps) for all samples
      - attn_c, attn_d attention weights for all samples (per token position)
      - input_ids, attention_mask so you can decode tokens later
    into predicted/{dataset_name}_{timestamp}/
    """
    try:
        import pandas as pd
    except ImportError:
        print("⚠ pandas not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "pandas"])
        import pandas as pd

    from torch.utils.data import DataLoader
    import json

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_dir = os.path.join("predicted", f"{dataset_name}_{timestamp}")
    os.makedirs(pred_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SAVING ENCODINGS + ATTENTION - {dataset_name}")
    print(f"{'='*70}")
    print(f"Saving outputs to: {pred_dir}")

    # Create dataset and dataloader
    dataset = TextDataset(sentences, labels, tokenizer, max_len=max_len, cached_data=cached_data)
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    all_texts = []
    all_true_labels = []
    all_input_ids = []
    all_attention_mask = []

    all_z_c = []
    all_z_d = []
    all_attn_c = []
    all_attn_d = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].cpu().numpy()
            texts = batch['text']

            output = model(input_ids, attention_mask)

            # Latents
            z_c = output['z_c'].detach().cpu().numpy()       # [B, latent_dim]
            z_d = output['z_d'].detach().cpu().numpy()       # [B, latent_dim]

            # Attention weights (already masked-softmax; padding positions ~=0)
            attn_c = output['attn_c'].detach().cpu().numpy() # [B, seq_len]
            attn_d = output['attn_d'].detach().cpu().numpy() # [B, seq_len]

            # Save batch-level raw tokenization too (for decoding later)
            all_input_ids.append(input_ids.detach().cpu().numpy())         # [B, seq_len]
            all_attention_mask.append(attention_mask.detach().cpu().numpy())  # [B, seq_len]

            all_texts.extend(list(texts))
            all_true_labels.extend(labels_batch.tolist())

            all_z_c.append(z_c)
            all_z_d.append(z_d)
            all_attn_c.append(attn_c)
            all_attn_d.append(attn_d)

    # Stack arrays
    np_input_ids = np.concatenate(all_input_ids, axis=0)          # [N, seq_len]
    np_attention_mask = np.concatenate(all_attention_mask, axis=0)# [N, seq_len]
    np_z_c = np.concatenate(all_z_c, axis=0)                      # [N, latent_dim]
    np_z_d = np.concatenate(all_z_d, axis=0)                      # [N, latent_dim]
    np_attn_c = np.concatenate(all_attn_c, axis=0)                # [N, seq_len]
    np_attn_d = np.concatenate(all_attn_d, axis=0)                # [N, seq_len]
    np_labels = np.array(all_true_labels)

    N, seq_len = np_input_ids.shape
    latent_dim = np_z_c.shape[1]

    # ===== Save NPY files (fast to reload) =====
    np.save(os.path.join(pred_dir, "input_ids.npy"), np_input_ids.astype(np.int32))
    np.save(os.path.join(pred_dir, "attention_mask.npy"), np_attention_mask.astype(np.int8))
    np.save(os.path.join(pred_dir, "labels.npy"), np_labels.astype(np.int64))

    np.save(os.path.join(pred_dir, "z_c.npy"), np_z_c.astype(np.float32))
    np.save(os.path.join(pred_dir, "z_d.npy"), np_z_d.astype(np.float32))

    # Attention weights
    np.save(os.path.join(pred_dir, "attn_c.npy"), np_attn_c.astype(np.float32))
    np.save(os.path.join(pred_dir, "attn_d.npy"), np_attn_d.astype(np.float32))

    print("✓ Saved numpy arrays:")
    print(f"  - input_ids.npy:       {np_input_ids.shape}")
    print(f"  - attention_mask.npy:  {np_attention_mask.shape}")
    print(f"  - labels.npy:          {np_labels.shape}")
    print(f"  - z_c.npy:             {np_z_c.shape}")
    print(f"  - z_d.npy:             {np_z_d.shape}")
    print(f"  - attn_c.npy:          {np_attn_c.shape}")
    print(f"  - attn_d.npy:          {np_attn_d.shape}")

    # ===== Optional: lightweight CSV (text + label only) =====
    meta_df = pd.DataFrame({
        "text": all_texts,
        "true_label": all_true_labels
    })
    meta_csv_path = os.path.join(pred_dir, "meta.csv")
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"✓ Saved meta to: {meta_csv_path}")

    # ===== Optional: full encodings CSV (WARNING: huge if latent_dim large) =====
    # If you still want: uncomment this block. For latent_dim=64 it's okay.
    save_full_csv = True
    if save_full_csv:
        zc_cols = {f"z_c_dim_{i}": np_z_c[:, i] for i in range(latent_dim)}
        zd_cols = {f"z_d_dim_{i}": np_z_d[:, i] for i in range(latent_dim)}
        df = pd.DataFrame({"text": all_texts, "true_label": all_true_labels, **zc_cols, **zd_cols})
        csv_path = os.path.join(pred_dir, "encodings.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved encodings to: {csv_path} (latent_dim={latent_dim})")

    # ===== Summary JSON =====
    summary = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "total_samples": int(N),
        "seq_len": int(seq_len),
        "latent_dim": int(latent_dim),
        "files": {
            "meta_csv": "meta.csv",
            "encodings_csv": "encodings.csv" if save_full_csv else None,
            "input_ids": "input_ids.npy",
            "attention_mask": "attention_mask.npy",
            "labels": "labels.npy",
            "z_c": "z_c.npy",
            "z_d": "z_d.npy",
            "attn_c": "attn_c.npy",
            "attn_d": "attn_d.npy"
        },
        "note": "attn_* are attention pooling weights over token positions (length=seq_len). Use input_ids + tokenizer to decode tokens."
    }
    summary_path = os.path.join(pred_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {summary_path}")

    print(f"{'='*70}\n")
    return pred_dir

# ============== MAIN ==============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train improved fairness model with attention, MI minimization, and orthogonal constraints"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="german-credit-data",
        help="Dataset to train on"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,  # Auto-detect based on GPU memory
        help=f"Batch size for training (default: auto-detect based on GPU memory, ~64 for RTX 5090)"
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=torch.cuda.is_available(),
        help="Use Automatic Mixed Precision (AMP) for faster training on GPU (default: enabled on GPU)"
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=50,
        help="Number of samples to use for visualization"
    )
    parser.add_argument(
        "--lambda-task",
        type=float,
        default=3.0,
        help="Task loss weight (default: 3.0 to prioritize learning task from z_c)"
    )
    parser.add_argument(
        "--lambda-adv",
        type=float,
        default=3.0,
        help="Adversarial loss weight (default: 3.0, balanced with task loss)"
    )
    parser.add_argument(
        "--lambda-rec",
        type=float,
        default=0.5,
        help="Reconstruction loss weight"
    )
    parser.add_argument(
        "--lambda-ortho",
        type=float,
        default=0.5,
        help="Orthogonality constraint weight"
    )
    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=0.01,
        help="VAE KL loss weight (regularization for compression only, default: 0.01)"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Latent dimension for z_c and z_d (VAE bottleneck, default: 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for BERT (default: 2e-5, standard for BERT fine-tuning)"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use cached tokenization (input_ids/attention_mask) for faster training"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable tokenization cache (recompute tokenization every time)"
    )
    parser.add_argument(
        "--force-recompute-cache",
        action="store_true",
        help="Force recompute tokenization even if cache exists"
    )
    
    args = parser.parse_args()
    
    # Handle cache flags
    use_cache = args.use_cache and not args.no_cache
    
    print("\n" + "=" * 70)
    print(f"TRAINING ON DATASET: {args.dataset}")
    print("=" * 70)
    
    # Load dataset
    try:
        if args.dataset == "german-credit-data":
            sentences, labels, data, _ = load_german_credit_data_balanced(n_samples=None)
        else:
            sentences, labels, data, _ = load_dataset_generic(
                args.dataset,
            )
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        exit(1)
    
    print(f"\nLoaded {len(sentences)} samples")
    print(f"Label distribution: {sum(labels)} (1), {len(labels) - sum(labels)} (0)")
    
    # Save processed data
    from data_loader import save_processed_data
    save_processed_data(sentences, labels, args.dataset)
    
    # Create visualization directory
    vis_dir = f"visualizations/{args.dataset}"
    os.makedirs(vis_dir, exist_ok=True)
    print(f"\nVisualization directory: {vis_dir}")
    
    # Create tokenizer and full dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Pre-compute cache for full dataset if needed
    cached_data_full = None
    if use_cache:
        cached_data_full = compute_and_cache_tokenized_data(
            sentences, args.dataset, max_len=256, force_recompute=args.force_recompute_cache
        )
    
    full_dataset = TextDataset(sentences, labels, tokenizer, max_len=256, cached_data=cached_data_full)
    
    # Train
    print(f"\n{'='*70}")
    print("TRAINING PHASE - Using full dataset")
    if use_cache:
        print("✓ Using cached tokenized data (faster training - no repeated tokenization)")
    print(f"{'='*70}")
    
    # Auto-detect optimal batch size based on GPU memory if not specified
    if args.batch_size is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 30:  # RTX 5090, A100, etc. (30GB+)
                args.batch_size = 64
            elif gpu_memory_gb >= 20:  # RTX 4090, etc. (20-30GB)
                args.batch_size = 32
            elif gpu_memory_gb >= 10:  # RTX 3080, etc. (10-20GB)
                args.batch_size = 16
            else:  # Smaller GPUs (<10GB)
                args.batch_size = 8
            print(f"Auto-detected batch size: {args.batch_size} (GPU memory: {gpu_memory_gb:.1f}GB)")
        else:
            args.batch_size = 8
            print(f"Using CPU batch size: {args.batch_size}")
    
    model, history = train_cd_model(
        sentences, labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_task=args.lambda_task,
        lambda_adv=args.lambda_adv,
        lambda_rec=args.lambda_rec,
        lambda_ortho=args.lambda_ortho,
        lambda_kl=args.lambda_kl,
        latent_dim=args.latent_dim,
        lr=args.lr,
        use_cache=use_cache,
        dataset_name=args.dataset,
        force_recompute_cache=args.force_recompute_cache,
        use_amp=args.use_amp
    )
    
    # Visualization phase
    print(f"\n{'='*70}")
    print(f"VISUALIZATION PHASE - Using {args.visualize_samples} random samples")
    print(f"{'='*70}")
    
    np.random.seed(42)
    vis_indices = np.random.choice(len(sentences), size=min(args.visualize_samples, len(sentences)), replace=False)
    vis_sentences = [sentences[i] for i in vis_indices]
    vis_labels = [labels[i] for i in vis_indices]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vis_dataset = TextDataset(vis_sentences, vis_labels, tokenizer, max_len=256)
    vis_dataloader = DataLoader(vis_dataset, batch_size=1, shuffle=False)
    
    # Visualize
    print(f"\nGenerating individual visualizations...")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(vis_dataloader):
            if i >= 5:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentence = batch['text'][0]
            
            visualize_reconstruction_and_attention(
                model, input_ids, attention_mask, sentence,
                save_dir=vis_dir
            )
    
    print(f"\nGenerating batch analysis visualization...")
    visualize_batch_analysis(model, vis_dataloader, num_samples=args.visualize_samples, save_dir=vis_dir)
    
    print(f"\nGenerating unified attention heatmaps for z_c and z_d...")
    vis_dataloader_batch = DataLoader(vis_dataset, batch_size=8, shuffle=False)
    visualize_attention_heatmap_batch(model, vis_dataloader_batch, num_samples=args.visualize_samples, save_dir=vis_dir)
    
    print(f"\n📊 Analyzing attention separation quality...")
    analysis_dataloader = DataLoader(vis_dataset, batch_size=16, shuffle=False)
    stats = analyze_attention_difference(model, analysis_dataloader, num_samples=args.visualize_samples, save_dir=vis_dir)
    
    print(f"\n📊 Visualizing latent space with PCA...")
    full_dataloader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    latent_stats = visualize_latent_space_pca(model, full_dataloader, num_samples=None, save_dir=vis_dir)
    
    # NEW: Visualize attention on text with HTML highlighting
    print(f"\n📊 Generating attention visualization on text (HTML)...")
    vis_dataloader_text = DataLoader(vis_dataset, batch_size=8, shuffle=False)
    visualize_attention_on_text(model, vis_dataloader_text, num_samples=args.visualize_samples, save_dir=vis_dir)
    
    # NEW: Generate attention comparison bar charts
    print(f"\n📊 Generating attention comparison visualizations...")
    visualize_attention_comparison(model, vis_dataloader_text, num_samples=min(10, args.visualize_samples), save_dir=vis_dir)
    
    # Save encodings (z_c and z_d) to predicted/{dataset_name}_{timestamp}/
    print(f"\n{'='*70}")
    print(f"SAVING ENCODINGS")
    print(f"{'='*70}")
    pred_dir = save_encodings(
        model, sentences, labels, args.dataset, tokenizer, 
        max_len=256, cached_data=cached_data_full
    )
    
    print(f"\n{'='*70}")
    print(f"✓ TRAINING AND VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  📁 Visualizations: {vis_dir}/")
    print(f"     - attention_visualization.html (open in browser to see highlighted text)")
    print(f"     - attention_comparison_*.png (bar charts comparing z_c vs z_d attention)")
    print(f"     - sample_*_attention.png (heatmaps for each sample)")
    print(f"     - latent_space_pca_2d.png (PCA projection of z_c and z_d)")
    print(f"  📁 Predictions: {pred_dir}/")
    print(f"     - encodings.csv (z_c and z_d for all samples)")
    print(f"     - z_c.npy, z_d.npy (numpy arrays)")
    print(f"\nKey metrics to check:")
    print(f"  1. Acc(z_c→y) should be HIGH (~1.0) - z_c contains task info")
    print(f"  2. Acc(z_d→y) should be LOW (~0.5) - z_d does NOT contain task info")
    print(f"  3. Orthogonality loss should be LOW (~0) - z_c and z_d are independent")
    print(f"  4. Reconstruction loss should be LOW (~0) - z_c + z_d can reconstruct x")
    print(f"{'='*70}\n")
