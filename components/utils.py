"""
Utility functions for saving encodings and other helper functions.
"""

import os
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
import json

from .dataset import TextDataset

# Auto-detect device: use CUDA if available, otherwise CPU
# This is the standard way to detect device across all files
def get_device():
    """
    Auto-detect and return the best available device.
    
    Returns:
        torch.device: 'cuda' if GPU is available, otherwise 'cpu'
    
    Example:
        device = get_device()
        model = model.to(device)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For backward compatibility, also export as 'device' variable
device = get_device()


def save_encodings(model, sentences, labels, dataset_name, tokenizer, max_len=256, cached_data=None):
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
    # Get device from model (model is already on the correct device)
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(model_device)
            attention_mask = batch['attention_mask'].to(model_device)
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
