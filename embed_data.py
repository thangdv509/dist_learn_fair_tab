#!/usr/bin/env python3
"""
Script to load trained model and embed new data into z_c and z_d representations.
Use this to:
1. Embed new data for fairness analysis
2. Extract disentangled representations for downstream tasks
3. Analyze what information is captured in z_c vs z_d
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import argparse
import os
from data_loader import load_dataset_generic
from main_attention import CD_Model, TextDataset, compute_and_cache_tokenized_data
from torch.utils.data import DataLoader

# Auto-detect device (use standard utility function)
from components.utils import get_device
device = get_device()
print(f"Using device: {device}")


def load_model(model_path, device=device):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to saved model (.pth file)
        device: Device to load model on
    
    Returns:
        model: Loaded CD_Model
        model_info: Dictionary with model metadata (epoch, loss, acc, etc.)
    """
    print(f"\nLoading model from: {model_path}")
    
    # Check if file exists and is readable
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        raise ValueError(f"Model file is empty: {model_path}")
    
    print(f"  File size: {file_size / (1024*1024):.2f} MB")
    
    # Try to load safetensors first (safer, faster), fallback to .pth
    safetensors_path = model_path.replace('.pth', '.safetensors')
    checkpoint = None
    use_safetensors = False
    
    # Check if safetensors file exists
    if os.path.exists(safetensors_path):
        try:
            import safetensors.torch
            print(f"  Loading from safetensors format: {safetensors_path}")
            # Load state_dict from safetensors to CPU first (handles GPU/CPU mismatch)
            # Safetensors supports device parameter: 'cpu', 'cuda', 'cuda:0', etc.
            state_dict = safetensors.torch.load_file(safetensors_path, device='cpu')
            
            # Load metadata from .pth file (metadata is always saved in .pth)
            # If .pth is corrupted, use default values
            if os.path.exists(model_path):
                try:
                    pth_checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    # Combine state_dict from safetensors with metadata from .pth
                    checkpoint = {
                        'model_state_dict': state_dict,
                        'epoch': pth_checkpoint.get('epoch', 'unknown'),
                        'loss': pth_checkpoint.get('loss', 'unknown'),
                        'acc_z_c': pth_checkpoint.get('acc_z_c', 'unknown'),
                        'acc_z_d': pth_checkpoint.get('acc_z_d', 'unknown'),
                        'num_classes': pth_checkpoint.get('num_classes', 2),
                        'latent_dim': pth_checkpoint.get('latent_dim', 64),
                        'd_model': pth_checkpoint.get('d_model', 768),
                    }
                except Exception as e:
                    # If .pth is corrupted, use state_dict from safetensors with default metadata
                    print(f"  ⚠ .pth file corrupted, using safetensors with default metadata")
                    checkpoint = {
                        'model_state_dict': state_dict,
                        'epoch': 'unknown',
                        'loss': 'unknown',
                        'acc_z_c': 'unknown',
                        'acc_z_d': 'unknown',
                        'num_classes': 2,
                        'latent_dim': 64,
                        'd_model': 768,
                    }
            else:
                # Only safetensors available, use default metadata
                checkpoint = {
                    'model_state_dict': state_dict,
                    'epoch': 'unknown',
                    'loss': 'unknown',
                    'acc_z_c': 'unknown',
                    'acc_z_d': 'unknown',
                    'num_classes': 2,
                    'latent_dim': 64,
                    'd_model': 768,
                }
            use_safetensors = True
            print(f"  ✓ Loaded state_dict from safetensors (faster, safer)")
        except ImportError:
            print("  ⚠ safetensors not available, falling back to .pth format")
        except Exception as e:
            print(f"  ⚠ Failed to load safetensors: {e}, falling back to .pth format")
    
    # Fallback to .pth format
    if checkpoint is None:
        print(f"  Loading from PyTorch format: {model_path}")
        try:
            # Always load to CPU first to handle GPU/CPU mismatch
            # This allows loading models trained on GPU on CPU (and vice versa)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except RuntimeError as e:
            if "failed finding central directory" in str(e) or "zip archive" in str(e):
                raise RuntimeError(
                    f"Model file appears to be corrupted: {model_path}\n"
                    f"Error: {e}\n"
                    f"Please check if the file was saved completely. "
                    f"You may need to retrain the model or use a different checkpoint."
                ) from e
            else:
                raise
    
    # Extract model parameters
    num_classes = checkpoint.get('num_classes', 2)
    latent_dim = checkpoint.get('latent_dim', 64)
    d_model = checkpoint.get('d_model', 768)
    
    # Create model with same architecture (on CPU first)
    model = CD_Model(
        num_classes=num_classes,
        d_model=d_model,
        latent_dim=latent_dim,
        freeze_bert=False
    )
    
    # Load state dict (on CPU)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to target device after loading
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Extract model info
    model_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown'),
        'acc_z_c': checkpoint.get('acc_z_c', 'unknown'),
        'acc_z_d': checkpoint.get('acc_z_d', 'unknown'),
        'num_classes': num_classes,
        'latent_dim': latent_dim,
        'd_model': d_model
    }
    
    print(f"✓ Model loaded successfully!")
    print(f"  Epoch: {model_info['epoch']}, Loss: {model_info['loss']}")
    print(f"  Acc(z_c→y): {model_info['acc_z_c']}, Acc(z_d→y): {model_info['acc_z_d']}")
    print(f"  Architecture: latent_dim={latent_dim}, num_classes={num_classes}")
    
    return model, model_info


def embed_data(model, sentences, tokenizer, max_len=50, batch_size=32, use_cache=True, dataset_name="embedding"):
    """
    Embed sentences into z_c and z_d representations.
    
    Args:
        model: Loaded CD_Model
        sentences: List of sentences to embed
        tokenizer: BERT tokenizer
        max_len: Maximum sequence length
        batch_size: Batch size for processing
        use_cache: Whether to use cached tokenization
        dataset_name: Name for cache file
    
    Returns:
        z_c: numpy array of shape [n_samples, latent_dim] - Content representations
        z_d: numpy array of shape [n_samples, latent_dim] - Demographic representations
        attn_c: numpy array of shape [n_samples, seq_len] - Attention weights for z_c
        attn_d: numpy array of shape [n_samples, seq_len] - Attention weights for z_d
    """
    print(f"\nEmbedding {len(sentences)} sentences...")
    
    # Tokenize and cache if needed
    cached_data = None
    if use_cache:
        try:
            cached_data = compute_and_cache_tokenized_data(
                sentences, dataset_name, max_len=max_len, force_recompute=False
            )
            print("✓ Using cached tokenization")
        except Exception as e:
            print(f"⚠ Could not load cache: {e}, will tokenize on-the-fly")
    
    # Create dataset
    dataset = TextDataset(sentences, [0] * len(sentences), tokenizer, max_len=max_len, cached_data=cached_data)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Embed all data
    z_c_list = []
    z_d_list = []
    attn_c_list = []
    attn_d_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            # Forward pass
            output = model(input_ids, attention_mask)
            
            # Extract representations
            z_c_list.append(output['z_c'].cpu().numpy())
            z_d_list.append(output['z_d'].cpu().numpy())
            attn_c_list.append(output['attn_c'].cpu().numpy())
            attn_d_list.append(output['attn_d'].cpu().numpy())
    
    # Concatenate all batches
    z_c = np.concatenate(z_c_list, axis=0)
    z_d = np.concatenate(z_d_list, axis=0)
    attn_c = np.concatenate(attn_c_list, axis=0)
    attn_d = np.concatenate(attn_d_list, axis=0)
    
    print(f"✓ Embedding complete!")
    print(f"  z_c shape: {z_c.shape}")
    print(f"  z_d shape: {z_d.shape}")
    print(f"  attn_c shape: {attn_c.shape}")
    print(f"  attn_d shape: {attn_d.shape}")
    
    return z_c, z_d, attn_c, attn_d


def save_embeddings(z_c, z_d, sentences, labels=None, output_dir="embeddings", filename="embeddings"):
    """
    Save embeddings to files.
    
    Args:
        z_c: Content representations [n_samples, latent_dim]
        z_d: Demographic representations [n_samples, latent_dim]
        sentences: Original sentences
        labels: Optional labels (for reference)
        output_dir: Directory to save embeddings
        filename: Base filename (without extension)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, f"{filename}_z_c.npy"), z_c)
    np.save(os.path.join(output_dir, f"{filename}_z_d.npy"), z_d)
    print(f"✓ Saved numpy arrays to {output_dir}/")
    
    # Save as CSV (with sentences and labels if available)
    df_data = {
        'sentence': sentences,
        'z_c': [z_c[i].tolist() for i in range(len(z_c))],
        'z_d': [z_d[i].tolist() for i in range(len(z_d))]
    }
    
    if labels is not None:
        df_data['label'] = labels
    
    df = pd.DataFrame(df_data)
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV to {csv_path}")
    
    # Save summary statistics
    summary = {
        'n_samples': len(z_c),
        'latent_dim': z_c.shape[1],
        'z_c_mean': z_c.mean(axis=0).tolist(),
        'z_c_std': z_c.std(axis=0).tolist(),
        'z_d_mean': z_d.mean(axis=0).tolist(),
        'z_d_std': z_d.std(axis=0).tolist(),
    }
    
    import json
    summary_path = os.path.join(output_dir, f"{filename}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Embed data using trained model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to input data file (CSV) or dataset name (e.g., 'german-credit-data')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="embeddings",
        help="Directory to save embeddings (default: embeddings/)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="embeddings",
        help="Base filename for output files (default: embeddings)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=50,
        help="Maximum sequence length (default: 50)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable tokenization cache"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, model_info = load_model(args.model_path, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load input data
    print(f"\nLoading input data from: {args.input_data}")
    
    # Check if it's a dataset name or file path
    if os.path.exists(args.input_data):
        # Load from CSV file
        df = pd.read_csv(args.input_data)
        if 'sentence' in df.columns:
            sentences = df['sentence'].tolist()
        elif 'text' in df.columns:
            sentences = df['text'].tolist()
        else:
            raise ValueError("CSV file must have 'sentence' or 'text' column")
        
        labels = df['label'].tolist() if 'label' in df.columns else None
        print(f"✓ Loaded {len(sentences)} sentences from CSV")
    else:
        # Try to load as dataset name
        try:
            sentences, labels, data, _ = load_dataset_generic(args.input_data)
            print(f"✓ Loaded {len(sentences)} sentences from dataset: {args.input_data}")
        except Exception as e:
            raise ValueError(f"Could not load data from '{args.input_data}': {e}")
    
    # Embed data
    z_c, z_d, attn_c, attn_d = embed_data(
        model=model,
        sentences=sentences,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
        dataset_name=args.output_name
    )
    
    # Save embeddings
    save_embeddings(
        z_c=z_c,
        z_d=z_d,
        sentences=sentences,
        labels=labels,
        output_dir=args.output_dir,
        filename=args.output_name
    )
    
    print(f"\n{'='*70}")
    print("EMBEDDING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput files saved to: {args.output_dir}/")
    print(f"  - {args.output_name}_z_c.npy (Content representations)")
    print(f"  - {args.output_name}_z_d.npy (Demographic representations)")
    print(f"  - {args.output_name}.csv (CSV with sentences and embeddings)")
    print(f"  - {args.output_name}_summary.json (Summary statistics)")
    print(f"\nYou can now use z_c and z_d for:")
    print(f"  - Fairness analysis")
    print(f"  - Downstream tasks")
    print(f"  - Analyzing what information each representation captures")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
