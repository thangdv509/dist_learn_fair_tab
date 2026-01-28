#!/usr/bin/env python3
"""
Visualization module for analyzing reconstruction and attention.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
import os

# Auto-detect device from model (will be set in each function)
def get_device(model):
    """Get device from model parameters"""
    return next(model.parameters()).device


def is_meaningful_token(token):
    """
    Check if token is meaningful (not special char, punctuation, or connector words).
    We want to visualize only tokens that carry semantic information.
    """
    # Skip special tokens
    if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
        return False
    
    # Skip pure punctuation and special chars
    special_chars = {'-', ':', ',', '.', '(', ')', '/', '<', '>', '=', 
                    '[', ']', '{', '}', '!', '?', '"', "'", '&', '|', '^', '$', '#', '%'}
    
    if all(c in special_chars for c in token):
        return False
    
    # Skip connector words that don't add semantic value
    connector_words = {'is', 'a', 'an', 'the', 'of', 'to', 'for', 'in', 'on', 'at', 'by', 'with', 'and', 'or'}
    if token.lower() in connector_words:
        return False
    
    return True


def merge_subword_tokens(tokens, attention_weights=None):
    """
    Merge BERT subword tokens (with ## prefix) back into complete words.
    Also merges consecutive numbers into single tokens.
    
    Example:
        ["34", "##99", "d", "##m"] -> ["3499", "dm"]
    """
    merged_tokens = []
    merged_attention = [] if attention_weights is not None else None
    
    i = 0
    while i < len(tokens):
        current_token = tokens[i]
        current_attn = attention_weights[i] if attention_weights is not None else None
        
        # Check if this is a subword token (starts with ##)
        if current_token.startswith('##'):
            # Merge with previous token
            if merged_tokens:
                merged_tokens[-1] = merged_tokens[-1] + current_token[2:]  # Remove ##
                if merged_attention is not None:
                    # Average attention of merged tokens
                    merged_attention[-1] = (merged_attention[-1] + current_attn) / 2.0
            else:
                # First token is subword (shouldn't happen, but handle it)
                merged_tokens.append(current_token[2:])
                if merged_attention is not None:
                    merged_attention.append(current_attn)
        # Check if this and next are both numbers (merge consecutive numbers)
        elif (i + 1 < len(tokens) and 
              tokens[i].isdigit() and 
              (tokens[i+1].isdigit() or tokens[i+1].startswith('##') and tokens[i+1][2:].isdigit())):
            # Start building merged number
            merged_num = tokens[i]
            merged_attn_sum = current_attn if attention_weights is not None else None
            merged_attn_count = 1
            
            j = i + 1
            while j < len(tokens):
                if tokens[j].isdigit():
                    merged_num += tokens[j]
                    if merged_attn_sum is not None:
                        merged_attn_sum += attention_weights[j]
                        merged_attn_count += 1
                    j += 1
                elif tokens[j].startswith('##') and tokens[j][2:].isdigit():
                    merged_num += tokens[j][2:]
                    if merged_attn_sum is not None:
                        merged_attn_sum += attention_weights[j]
                        merged_attn_count += 1
                    j += 1
                else:
                    break
            
            merged_tokens.append(merged_num)
            if merged_attention is not None:
                merged_attention.append(merged_attn_sum / merged_attn_count)
            i = j
            continue
        # Check if this is a single letter followed by ##m or ##d (like "d" + "##m" -> "dm")
        elif (i + 1 < len(tokens) and 
              len(current_token) == 1 and 
              tokens[i+1].startswith('##') and 
              len(tokens[i+1][2:]) == 1):
            # Merge single letter + single letter subword
            merged_token = current_token + tokens[i+1][2:]
            merged_tokens.append(merged_token)
            if merged_attention is not None:
                merged_attention.append((current_attn + attention_weights[i+1]) / 2.0)
            i += 2
            continue
        else:
            # Regular token
            merged_tokens.append(current_token)
            if merged_attention is not None:
                merged_attention.append(current_attn)
        
        i += 1
    
    if merged_attention is not None:
        # Renormalize attention weights
        if sum(merged_attention) > 0:
            merged_attention = np.array(merged_attention) / sum(merged_attention)
        return merged_tokens, merged_attention
    
    return merged_tokens


def filter_tokens_for_display(tokens, attention_weights=None):
    """
    Filter tokens to keep only meaningful ones.
    Assumes tokens are already merged (no ## prefixes).
    Returns filtered tokens and corresponding attention weights.
    """
    meaningful_indices = [i for i, token in enumerate(tokens) if is_meaningful_token(token)]
    filtered_tokens = [tokens[i] for i in meaningful_indices]
    
    if attention_weights is not None:
        filtered_attention = attention_weights[meaningful_indices].copy()
        # Renormalize attention weights
        if filtered_attention.sum() > 0:
            filtered_attention = filtered_attention / filtered_attention.sum()
        return filtered_tokens, filtered_attention
    
    return filtered_tokens


def visualize_reconstruction_and_attention(
    model, input_ids, attention_mask, sentence, 
    save_dir="visualizations"
):
    """
    Visualize:
    1. Original tokens with z_c attention
    2. Original tokens with z_d attention  
    3. Reconstruction quality (cosine similarity)
    4. Comparison table
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Forward pass
    model.eval()
    device = get_device(model)
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        z_c = output['z_c']
        z_d = output['z_d']
        attn_c = output['attn_c']  # Shape: (batch_size, seq_len)
        attn_d = output['attn_d']
        reconstructed = output['reconstructed']
        
        # Get original embedding from BERT (pretrained)
        # Model always uses BERT now (no transformer option)
        with torch.no_grad():
            bert_output = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            original_h = bert_output.last_hidden_state[:, 0, :]  # CLS token
        
    # Get tokens
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    # Merge subword tokens before processing
    tokens_merged = merge_subword_tokens(tokens)
    # For backward compatibility, also create clean_tokens
    clean_tokens = [t.replace('##', '') for t in tokens]
    
    # Extract attention and reconstruction for first sample
    if attn_c is not None:
        attn_c_vals = attn_c[0].cpu().numpy()
        attn_d_vals = attn_d[0].cpu().numpy()
    else:
        attn_c_vals = np.zeros(len(tokens))
        attn_d_vals = np.zeros(len(tokens))
    
    # Merge subword tokens first
    tokens_merged_c, attn_c_merged = merge_subword_tokens(tokens, attn_c_vals)
    tokens_merged_d, attn_d_merged = merge_subword_tokens(tokens, attn_d_vals)
    
    # Filter out [PAD] tokens but keep all meaningful tokens (don't filter too aggressively)
    pad_mask_c = np.array([t != '[PAD]' and t != '[CLS]' and t != '[SEP]' for t in tokens_merged_c])
    pad_mask_d = np.array([t != '[PAD]' and t != '[CLS]' and t != '[SEP]' for t in tokens_merged_d])
    
    meaningful_tokens = [t for t, keep in zip(tokens_merged_c, pad_mask_c) if keep]
    meaningful_tokens_d = [t for t, keep in zip(tokens_merged_d, pad_mask_d) if keep]
    attn_c_vals_filtered = attn_c_merged[pad_mask_c]
    attn_d_vals_filtered = attn_d_merged[pad_mask_d]
    
    # Don't renormalize - keep original attention weights to see actual focus
    # Only normalize if sum is 0 (shouldn't happen)
    if attn_c_vals_filtered.sum() > 0:
        attn_c_vals_filtered = attn_c_vals_filtered / attn_c_vals_filtered.sum()  # Renormalize after filtering
    if attn_d_vals_filtered.sum() > 0:
        attn_d_vals_filtered = attn_d_vals_filtered / attn_d_vals_filtered.sum()
    
    # For compatibility: use meaningful_tokens as clean_tokens_filtered
    clean_tokens_filtered = meaningful_tokens
    
    # Compute reconstruction quality
    original_norm = torch.nn.functional.normalize(original_h, p=2, dim=1)
    reconstructed_norm = torch.nn.functional.normalize(reconstructed, p=2, dim=1)
    cosine_sim = torch.sum(original_norm * reconstructed_norm, dim=1).item()
    mse_recon = nn.functional.mse_loss(reconstructed, original_h).item()
    
    # Create figure with 2 heatmaps + 2 info panels
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    
    # 1. z_c Attention Heatmap (meaningful tokens only)
    ax = axes[0, 0]
    # Use better color scale and show actual values for clarity
    max_attn_c = max(attn_c_vals_filtered) if len(attn_c_vals_filtered) > 0 else 1.0
    sns.heatmap(
        [attn_c_vals_filtered], 
        annot=True, fmt='.3f', cmap='YlOrRd',  # Show values for clarity
        xticklabels=meaningful_tokens, yticklabels=['z_c'],
        ax=ax, cbar_kws={'label': 'Attention Weight'}, 
        cbar=True, vmin=0, vmax=max_attn_c,
        annot_kws={'size': 8}
    )
    ax.set_title(f'z_c (Content) Attention (Max: {max_attn_c:.3f})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tokens', fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 2. z_d Attention Heatmap (meaningful tokens only)
    ax = axes[0, 1]
    max_attn_d = max(attn_d_vals_filtered) if len(attn_d_vals_filtered) > 0 else 1.0
    sns.heatmap(
        [attn_d_vals_filtered], 
        annot=True, fmt='.3f', cmap='RdPu',  # Show values for clarity
        xticklabels=meaningful_tokens_d, yticklabels=['z_d'],
        ax=ax, cbar_kws={'label': 'Attention Weight'}, 
        cbar=True, vmin=0, vmax=max_attn_d,
        annot_kws={'size': 8}
    )
    ax.set_title(f'z_d (Demographic) Attention (Max: {max_attn_d:.3f})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tokens', fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_xlabel('Tokens (Filtered for Meaning)', fontsize=11)
    
    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Get top attention tokens
    top_c_idx = np.argmax(attn_c_vals) if len(attn_c_vals) > 0 else 0
    top_d_idx = np.argmax(attn_d_vals) if len(attn_d_vals) > 0 else 0
    top_c_token = clean_tokens[top_c_idx] if top_c_idx < len(clean_tokens) else "N/A"
    top_d_token = clean_tokens[top_d_idx] if top_d_idx < len(clean_tokens) else "N/A"
    
    summary_text = f"""
    ORIGINAL SENTENCE:
    {sentence}
    
    TOKENS: {len(tokens)}
    Clean tokens: {' | '.join(clean_tokens[:20])}... (showing first 20)
    
    RECONSTRUCTION METRICS:
    ✓ Cosine Similarity: {cosine_sim:.4f} (1.0 = perfect match)
    ✓ MSE Loss: {mse_recon:.4f} (lower = better)
    
    ATTENTION ANALYSIS:
    z_c (Content) focuses on: {top_c_token} (weight: {np.max(attn_c_vals):.3f})
    z_d (Demographic) focuses on: {top_d_token} (weight: {np.max(attn_d_vals):.3f})
    
    LATENT SPACE:
    z_c variance: {z_c.std().item():.4f}
    z_d variance: {z_d.std().item():.4f}
    z_c-z_d correlation: {torch.cosine_similarity(z_c, z_d, dim=1).mean().item():.4f}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    filename = f"{save_dir}/reconstruction_analysis_combined.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved combined visualization to {filename}")
    plt.close()
    
    # ============ SEPARATE HEATMAP FIGURES ============
    # Figure 1: z_c Attention Heatmap only (WITHOUT [PAD])
    fig_c, ax_c = plt.subplots(figsize=(14, 3))
    sns.heatmap(
        [attn_c_vals_filtered], 
        annot=False, cmap='YlOrRd',
        xticklabels=clean_tokens_filtered, yticklabels=['z_c Attention'],
        ax=ax_c, cbar_kws={'label': 'Attention Weight'}, 
        cbar=True, vmin=0, vmax=max(attn_c_vals_filtered) if len(attn_c_vals_filtered) > 0 else 1
    )
    ax_c.set_title(f'z_c (Content) Attention - Sentence: {sentence[:60]}...', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_c.set_xlabel('Tokens', fontsize=11)
    plt.tight_layout()
    filename_c = f"{save_dir}/z_c_attention_heatmap.png"
    plt.savefig(filename_c, dpi=150, bbox_inches='tight')
    print(f"✓ Saved z_c heatmap to {filename_c}")
    plt.close()
    
    # Figure 2: z_d Attention Heatmap only (WITHOUT [PAD])
    fig_d, ax_d = plt.subplots(figsize=(14, 3))
    sns.heatmap(
        [attn_d_vals_filtered], 
        annot=False, cmap='RdPu',
        xticklabels=clean_tokens_filtered, yticklabels=['z_d Attention'],
        ax=ax_d, cbar_kws={'label': 'Attention Weight'}, 
        cbar=True, vmin=0, vmax=max(attn_d_vals_filtered) if len(attn_d_vals_filtered) > 0 else 1
    )
    ax_d.set_title(f'z_d (Demographic) Attention - Sentence: {sentence[:60]}...', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_d.set_xlabel('Tokens', fontsize=11)
    plt.tight_layout()
    filename_d = f"{save_dir}/z_d_attention_heatmap.png"
    plt.savefig(filename_d, dpi=150, bbox_inches='tight')
    print(f"✓ Saved z_d heatmap to {filename_d}")
    plt.close()
    
    return {
        'cosine_sim': cosine_sim,
        'mse_recon': mse_recon,
        'tokens': clean_tokens_filtered,
        'attn_c': attn_c_vals_filtered,
        'attn_d': attn_d_vals_filtered,
    }


def visualize_batch_analysis(model, dataloader, num_samples=5, save_dir="visualizations"):
    """
    Visualize multiple samples to understand attention patterns
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = get_device(model)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    reconstruction_scores = []
    attn_c_focuses = []
    attn_d_focuses = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['text']
            
            output = model(input_ids, attention_mask)
            z_c = output['z_c']
            z_d = output['z_d']
            attn_c = output['attn_c']
            attn_d = output['attn_d']
            reconstructed = output['reconstructed']
            
            with torch.no_grad():
                bert_output = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                original_h = bert_output.last_hidden_state[:, 0, :]  # CLS token
            
            for i in range(input_ids.size(0)):
                if sample_count >= num_samples:
                    break
                
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                
                if attn_c is not None:
                    attn_c_vals = attn_c[i].cpu().numpy()
                    attn_d_vals = attn_d[i].cpu().numpy()
                else:
                    attn_c_vals = np.zeros(len(tokens))
                    attn_d_vals = np.zeros(len(tokens))
                
                # Merge subword tokens
                tokens_merged_c, attn_c_merged = merge_subword_tokens(tokens, attn_c_vals)
                tokens_merged_d, attn_d_merged = merge_subword_tokens(tokens, attn_d_vals)
                
                # Cosine similarity
                original_norm = torch.nn.functional.normalize(original_h[i:i+1], p=2, dim=1)
                reconstructed_norm = torch.nn.functional.normalize(reconstructed[i:i+1], p=2, dim=1)
                cosine_sim = torch.sum(original_norm * reconstructed_norm, dim=1).item()
                
                reconstruction_scores.append(cosine_sim)
                attn_c_focuses.append(tokens_merged_c[np.argmax(attn_c_merged)])
                attn_d_focuses.append(tokens_merged_d[np.argmax(attn_d_merged)])
                
                sample_count += 1
    
    # Plot summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Reconstruction scores
    ax = axes[0]
    ax.bar(range(len(reconstruction_scores)), reconstruction_scores, color='steelblue', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Reconstruction Quality across Samples')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Most focused tokens for z_c
    ax = axes[1]
    from collections import Counter
    c_counter = Counter(attn_c_focuses)
    tokens_c = list(c_counter.keys())
    counts_c = list(c_counter.values())
    ax.barh(tokens_c, counts_c, color='orange', alpha=0.7)
    ax.set_xlabel('Frequency')
    ax.set_title('z_c Most-Focused Tokens')
    ax.grid(axis='x', alpha=0.3)
    
    # Most focused tokens for z_d
    ax = axes[2]
    d_counter = Counter(attn_d_focuses)
    tokens_d = list(d_counter.keys())
    counts_d = list(d_counter.values())
    ax.barh(tokens_d, counts_d, color='purple', alpha=0.7)
    ax.set_xlabel('Frequency')
    ax.set_title('z_d Most-Focused Tokens')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    filename = f"{save_dir}/batch_analysis.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved batch analysis to {filename}")
    plt.close()
    
    return {
        'avg_reconstruction': np.mean(reconstruction_scores),
        'z_c_focuses': dict(c_counter),
        'z_d_focuses': dict(d_counter),
    }


def compare_original_vs_reconstructed_text(model, dataloader, num_samples=3, save_dir="visualizations"):
    """
    Create a detailed comparison table showing reconstruction quality
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = get_device(model)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    results = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['text']
            labels = batch['label']
            
            output = model(input_ids, attention_mask)
            reconstructed = output['reconstructed']
            
            with torch.no_grad():
                bert_output = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                original_h = bert_output.last_hidden_state[:, 0, :]  # CLS token
            
            for i in range(input_ids.size(0)):
                if sample_count >= num_samples:
                    break
                
                # Compute metrics
                mse = nn.functional.mse_loss(reconstructed[i], original_h[i]).item()
                original_norm = torch.nn.functional.normalize(original_h[i:i+1], p=2, dim=1)
                reconstructed_norm = torch.nn.functional.normalize(reconstructed[i:i+1], p=2, dim=1)
                cosine_sim = torch.sum(original_norm * reconstructed_norm, dim=1).item()
                
                results.append({
                    'Text': texts[i][:80] + '...' if len(texts[i]) > 80 else texts[i],
                    'Label': 'Good' if labels[i].item() == 1 else 'Bad',
                    'Cosine Sim': f'{cosine_sim:.4f}',
                    'MSE': f'{mse:.4f}',
                    'Quality': '✓ High' if cosine_sim > 0.95 else '⚠ Medium' if cosine_sim > 0.85 else '✗ Low'
                })
                
                sample_count += 1
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(16, max(4, len(results) * 0.8)))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Text', 'Label', 'Cosine Sim', 'MSE', 'Quality']
    ]
    for r in results:
        table_data.append([
            r['Text'],
            r['Label'],
            r['Cosine Sim'],
            r['MSE'],
            r['Quality']
        ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.1, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Reconstruction Quality Comparison', fontsize=14, fontweight='bold', pad=20)
    filename = f"{save_dir}/reconstruction_table.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison table to {filename}")
    plt.close()
    
    return results


def visualize_attention_heatmap_batch(model, dataloader, num_samples=50, save_dir="visualizations"):
    """
    Create individual heatmaps for each sample showing z_c and z_d attention vertically stacked.
    Saves num_samples separate PNG files, each with z_c (top) and z_d (bottom) heatmaps + original sentence.
    
    Args:
        model: Trained CD_Model
        dataloader: DataLoader for samples
        num_samples: Number of individual visualizations to create (default 50)
        save_dir: Directory to save visualizations
    """
    import os
    import textwrap
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = get_device(model)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    sample_count = 0
    
    print(f"\nGenerating {num_samples} individual attention heatmaps...")
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['text']
            
            output = model(input_ids, attention_mask)
            attn_c = output['attn_c']  # (batch_size, seq_len)
            attn_d = output['attn_d']
            
            for i in range(input_ids.size(0)):
                if sample_count >= num_samples:
                    break
                
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                
                # Filter out [PAD] tokens first
                pad_mask = np.array([t != '[PAD]' for t in tokens])
                tokens_no_pad = [t for t, keep in zip(tokens, pad_mask) if keep]
                
                if attn_c is not None:
                    attn_c_vals = attn_c[i].cpu().numpy()[pad_mask]
                    attn_d_vals = attn_d[i].cpu().numpy()[pad_mask]
                else:
                    attn_c_vals = np.zeros(len(tokens))[pad_mask]
                    attn_d_vals = np.zeros(len(tokens))[pad_mask]
                
                # Merge subword tokens
                tokens_merged_c, attn_c_merged = merge_subword_tokens(tokens_no_pad, attn_c_vals)
                tokens_merged_d, attn_d_merged = merge_subword_tokens(tokens_no_pad, attn_d_vals)
                
                clean_tokens_filtered = tokens_merged_c  # Use merged tokens
                original_sentence = texts[i]
                
                # Use merged attention values
                attn_c_vals = attn_c_merged
                attn_d_vals = attn_d_merged
                
                # Calculate figure size based on number of tokens
                num_tokens = len(clean_tokens_filtered)
                fig_width = max(14, num_tokens * 0.3)  # Scale width based on token count
                
                # Create figure with 3 rows: sentence info, z_c heatmap, z_d heatmap
                fig, axes = plt.subplots(3, 1, figsize=(fig_width, 10),
                                        gridspec_kw={'height_ratios': [1, 2, 2]})
                
                # Title with original sentence (with text wrapping)
                ax_title = axes[0]
                ax_title.axis('off')
                
                # Wrap text to fit width
                wrapped_sentence = textwrap.fill(f"Sample #{sample_count}: {original_sentence}", 
                                                width=100)
                ax_title.text(0.05, 0.5, wrapped_sentence, fontsize=11, 
                             verticalalignment='center', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
                
                # z_c Heatmap (top)
                ax_c = axes[1]
                sns.heatmap(
                    [attn_c_vals],
                    cmap='YlOrRd',
                    xticklabels=clean_tokens_filtered,
                    yticklabels=['z_c'],
                    ax=ax_c,
                    cbar_kws={'label': 'Attention Weight'},
                    vmin=0, vmax=max(attn_c_vals) if len(attn_c_vals) > 0 else 1,
                    annot=False,
                    cbar=True
                )
                ax_c.set_title('z_c (Content) Attention', fontsize=12, fontweight='bold', pad=8)
                ax_c.set_xlabel('Tokens', fontsize=10)
                plt.setp(ax_c.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                
                # z_d Heatmap (bottom)
                ax_d = axes[2]
                sns.heatmap(
                    [attn_d_vals],
                    cmap='RdPu',
                    xticklabels=clean_tokens_filtered,
                    yticklabels=['z_d'],
                    ax=ax_d,
                    cbar_kws={'label': 'Attention Weight'},
                    vmin=0, vmax=max(attn_d_vals) if len(attn_d_vals) > 0 else 1,
                    annot=False,
                    cbar=True
                )
                ax_d.set_title('z_d (Demographic) Attention', fontsize=12, fontweight='bold', pad=8)
                ax_d.set_xlabel('Tokens', fontsize=10)
                plt.setp(ax_d.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                
                plt.tight_layout()
                filename = f"{save_dir}/sample_{sample_count:03d}_attention.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                
                if (sample_count + 1) % 10 == 0:
                    print(f"  ✓ Generated {sample_count + 1}/{num_samples} visualizations...")
                
                sample_count += 1
    
    print(f"✓ Generated {sample_count} individual attention heatmaps")
    
    return {
        'num_samples': sample_count
    }


if __name__ == "__main__":
    print("Visualization module loaded. Use with your trained model.")


def visualize_latent_space_pca(model, dataloader, num_samples=None, save_dir="visualizations"):
    """
    Project z_c and z_d into 2D space using PCA and visualize by label.
    
    Args:
        model: Trained CD_Model
        dataloader: DataLoader for all data
        num_samples: Number of samples to use (None = all)
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = get_device(model)
    
    z_c_list = []
    z_d_list = []
    labels_list = []
    sample_count = 0
    
    print(f"\nExtracting latent representations...")
    
    with torch.no_grad():
        for batch in dataloader:
            if num_samples and sample_count >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].cpu().numpy()
            
            output = model(input_ids, attention_mask)
            z_c = output['z_c'].cpu().numpy()
            z_d = output['z_d'].cpu().numpy()
            
            z_c_list.append(z_c)
            z_d_list.append(z_d)
            labels_list.extend(labels_batch)
            sample_count += len(labels_batch)
    
    # Concatenate all batches
    z_c_all = np.vstack(z_c_list)
    z_d_all = np.vstack(z_d_list)
    labels_all = np.array(labels_list)
    
    print(f"✓ Extracted {len(z_c_all)} samples")
    print(f"  z_c shape: {z_c_all.shape}, z_d shape: {z_d_all.shape}")
    
    # Apply PCA
    print(f"\nApplying PCA(2) to project to 2D...")
    pca_c = PCA(n_components=2)
    pca_d = PCA(n_components=2)
    
    z_c_2d = pca_c.fit_transform(z_c_all)
    z_d_2d = pca_d.fit_transform(z_d_all)
    
    print(f"✓ z_c explained variance: {pca_c.explained_variance_ratio_.sum():.4f}")
    print(f"✓ z_d explained variance: {pca_d.explained_variance_ratio_.sum():.4f}")
    
    # Create side-by-side scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # z_c scatter plot
    ax = axes[0]
    scatter_c_0 = ax.scatter(z_c_2d[labels_all == 0, 0], z_c_2d[labels_all == 0, 1],
                            c='blue', alpha=0.6, s=50, label='Label 0')
    scatter_c_1 = ax.scatter(z_c_2d[labels_all == 1, 0], z_c_2d[labels_all == 1, 1],
                            c='red', alpha=0.6, s=50, label='Label 1')
    ax.set_xlabel(f'PC1 ({pca_c.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_c.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.set_title('z_c (Content) - PCA 2D Projection', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # z_d scatter plot
    ax = axes[1]
    scatter_d_0 = ax.scatter(z_d_2d[labels_all == 0, 0], z_d_2d[labels_all == 0, 1],
                            c='blue', alpha=0.6, s=50, label='Label 0')
    scatter_d_1 = ax.scatter(z_d_2d[labels_all == 1, 0], z_d_2d[labels_all == 1, 1],
                            c='red', alpha=0.6, s=50, label='Label 1')
    ax.set_xlabel(f'PC1 ({pca_d.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_d.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.set_title('z_d (Demographic) - PCA 2D Projection', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add note about why visual separation doesn't mean predictive power
    note_text = (
        "Note: Visual separation in 2D PCA ≠ Predictive power\n"
        "z_d may have demographic structure that doesn't\n"
        "correlate with task label (target: ~50% accuracy)"
    )
    ax.text(0.02, 0.98, note_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            family='monospace')
    
    plt.tight_layout()
    filename = f"{save_dir}/latent_space_pca_2d.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved PCA visualization to {filename}")
    plt.close()
    
    # Additional analysis: separation quality
    print(f"\n📊 Latent Space Separation Analysis:")
    
    # Calculate between-class and within-class distances
    def calculate_separation(z_2d, labels):
        # Between-class distance
        center_0 = z_2d[labels == 0].mean(axis=0)
        center_1 = z_2d[labels == 1].mean(axis=0)
        between_dist = np.linalg.norm(center_0 - center_1)
        
        # Within-class distances (average)
        within_0 = np.mean([np.linalg.norm(z - center_0) for z in z_2d[labels == 0]])
        within_1 = np.mean([np.linalg.norm(z - center_1) for z in z_2d[labels == 1]])
        within_dist = (within_0 + within_1) / 2
        
        # Separation ratio (higher is better)
        separation_ratio = between_dist / (within_dist + 1e-6)
        
        return between_dist, within_dist, separation_ratio
    
    b_c, w_c, s_c = calculate_separation(z_c_2d, labels_all)
    b_d, w_d, s_d = calculate_separation(z_d_2d, labels_all)
    
    print(f"  z_c - Between-class dist: {b_c:.4f}, Within-class dist: {w_c:.4f}")
    print(f"        Separation ratio: {s_c:.4f} (higher = better separation)")
    print(f"  z_d - Between-class dist: {b_d:.4f}, Within-class dist: {w_d:.4f}")
    print(f"        Separation ratio: {s_d:.4f}")
    
    if s_c > s_d:
        print(f"  ✓ z_c (Content) has better class separation")
    else:
        print(f"  ⚠️  z_d (Demographic) has better separation (should contain less task info)")
    
    # IMPORTANT: Calculate actual predictive power using full-dimensional z_d
    # This explains why 50% accuracy doesn't mean no structure
    print(f"\n🔍 Why z_d can show visual separation but still have 50% accuracy:")
    print(f"  (This is NORMAL and EXPECTED behavior)")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # Test predictive power on FULL z_d (not just 2D PCA)
    # Split data for testing
    n_samples = len(z_d_all)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    z_d_train = z_d_all[train_idx]
    z_d_test = z_d_all[test_idx]
    labels_train = labels_all[train_idx]
    labels_test = labels_all[test_idx]
    
    # Train classifier on full z_d
    clf_full = LogisticRegression(max_iter=1000, random_state=42)
    clf_full.fit(z_d_train, labels_train)
    acc_full = accuracy_score(labels_test, clf_full.predict(z_d_test))
    
    # Train classifier on 2D PCA projection of z_d
    z_d_2d_train = z_d_2d[train_idx]
    z_d_2d_test = z_d_2d[test_idx]
    clf_2d = LogisticRegression(max_iter=1000, random_state=42)
    clf_2d.fit(z_d_2d_train, labels_train)
    acc_2d = accuracy_score(labels_test, clf_2d.predict(z_d_2d_test))
    
    print(f"  ✓ Accuracy using FULL z_d ({z_d_all.shape[1]}D): {acc_full:.3f}")
    print(f"  ✓ Accuracy using 2D PCA projection: {acc_2d:.3f}")
    print(f"  ✓ PCA explains only {pca_d.explained_variance_ratio_.sum():.2%} of variance")
    
    print(f"\n  💡 Key Insights:")
    print(f"    1. Visual separation in 2D PCA ≠ Predictive power for task label")
    print(f"    2. z_d may have demographic structure (age, gender, etc.) that")
    print(f"       doesn't correlate with task label (credit risk)")
    print(f"    3. PCA only shows 2 dimensions - most information is in other dimensions")
    print(f"    4. 50% accuracy means z_d CANNOT predict task label (this is GOOD!)")
    print(f"    5. But z_d can still have structure for demographic features")
    
    if acc_full < 0.55:
        print(f"\n  ✅ z_d is working correctly: Cannot predict task label (acc={acc_full:.3f} ≈ 50%)")
    else:
        print(f"\n  ⚠️  Warning: z_d can predict task label (acc={acc_full:.3f} > 50%)")
        print(f"     This suggests adversarial training may need adjustment")
    
    return {
        'z_c_2d': z_c_2d,
        'z_d_2d': z_d_2d,
        'labels': labels_all,
        'z_c_explained_var': pca_c.explained_variance_ratio_.sum(),
        'z_d_explained_var': pca_d.explained_variance_ratio_.sum(),
        'z_c_separation': s_c,
        'z_d_separation': s_d
    }


def analyze_attention_difference(model, dataloader, num_samples=50, save_dir="visualizations"):
    """
    Analyze and visualize the difference between z_c and z_d attention.
    Creates comparison heatmaps to show which tokens are differentially attended.
    
    Args:
        model: Trained CD_Model
        dataloader: DataLoader for samples
        num_samples: Number of samples to analyze
        save_dir: Directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = get_device(model)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    attn_c_list = []
    attn_d_list = []
    attn_diff_list = []
    max_tokens = 0
    
    print(f"\nAnalyzing attention difference for {num_samples} samples...")
    
    with torch.no_grad():
        for batch in dataloader:
            if len(attn_c_list) >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            output = model(input_ids, attention_mask)
            attn_c = output['attn_c']
            attn_d = output['attn_d']
            
            for i in range(input_ids.size(0)):
                if len(attn_c_list) >= num_samples:
                    break
                
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                pad_mask = np.array([t != '[PAD]' for t in tokens])
                
                if attn_c is not None:
                    attn_c_vals = attn_c[i].cpu().numpy()[pad_mask]
                    attn_d_vals = attn_d[i].cpu().numpy()[pad_mask]
                else:
                    attn_c_vals = np.zeros(len(tokens))[pad_mask]
                    attn_d_vals = np.zeros(len(tokens))[pad_mask]
                
                # Calculate absolute difference
                attn_diff = np.abs(attn_c_vals - attn_d_vals)
                
                attn_c_list.append(attn_c_vals)
                attn_d_list.append(attn_d_vals)
                attn_diff_list.append(attn_diff)
                max_tokens = max(max_tokens, len(attn_c_vals))
    
    # Pad arrays
    attn_c_padded = np.zeros((len(attn_c_list), max_tokens))
    attn_d_padded = np.zeros((len(attn_d_list), max_tokens))
    attn_diff_padded = np.zeros((len(attn_diff_list), max_tokens))
    
    for i, (ac, ad, ad_diff) in enumerate(zip(attn_c_list, attn_d_list, attn_diff_list)):
        attn_c_padded[i, :len(ac)] = ac
        attn_d_padded[i, :len(ad)] = ad
        attn_diff_padded[i, :len(ad_diff)] = ad_diff
    
    # Create comparison figure
    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    
    # z_c
    ax = axes[0]
    sns.heatmap(attn_c_padded, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Attention Weight'})
    ax.set_title(f'z_c (Content) Attention - {len(attn_c_list)} samples', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Sample')
    
    # z_d
    ax = axes[1]
    sns.heatmap(attn_d_padded, cmap='RdPu', ax=ax, cbar_kws={'label': 'Attention Weight'})
    ax.set_title(f'z_d (Demographic) Attention - {len(attn_d_list)} samples', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Sample')
    
    # Difference (shows where they diverge)
    ax = axes[2]
    sns.heatmap(attn_diff_padded, cmap='viridis', ax=ax, cbar_kws={'label': 'Absolute Difference'})
    ax.set_title(f'|z_c - z_d| Attention Difference (higher = better separation)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Sample')
    
    plt.tight_layout()
    filename = f"{save_dir}/attention_analysis_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved attention comparison to {filename}")
    plt.close()
    
    # Statistics
    mean_c = np.mean(attn_c_padded)
    mean_d = np.mean(attn_d_padded)
    mean_diff = np.mean(attn_diff_padded)
    
    # Correlation between z_c and z_d attention
    correlation = np.corrcoef(attn_c_padded.flatten(), attn_d_padded.flatten())[0, 1]
    
    stats = {
        'z_c_mean': mean_c,
        'z_d_mean': mean_d,
        'difference_mean': mean_diff,
        'correlation': correlation,
        'num_samples': len(attn_c_list)
    }
    
    print(f"\n📊 Attention Analysis Statistics:")
    print(f"  z_c mean attention: {mean_c:.4f}")
    print(f"  z_d mean attention: {mean_d:.4f}")
    print(f"  Mean absolute difference: {mean_diff:.4f}")
    print(f"  z_c-z_d correlation: {correlation:.4f}")
    print(f"    ⚠️  High correlation (>0.8) = bad (not separated)")
    print(f"    ✓ Low correlation (<0.5) = good (well separated)")
    
    return stats


def visualize_attention_on_text(model, dataloader, num_samples=20, save_dir="visualizations"):
    """
    Visualize attention weights directly on original text with color highlighting.
    Creates HTML files showing which words z_c and z_d focus on.
    
    This helps understand:
    - What information z_c (Content) captures from text
    - What information z_d (Demographic) captures from text
    - Whether they focus on different parts of the text
    
    Args:
        model: Trained CD_Model
        dataloader: DataLoader for samples
        num_samples: Number of samples to visualize
        save_dir: Directory to save HTML files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = get_device(model)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attention Visualization - z_c vs z_d</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #333; text-align: center; }
            h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
            .sample { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .sample-header { font-weight: bold; color: #333; margin-bottom: 15px; }
            .attention-row { margin: 15px 0; padding: 10px; border-radius: 5px; }
            .z_c-row { background: #fff8e1; border-left: 4px solid #ff9800; }
            .z_d-row { background: #f3e5f5; border-left: 4px solid #9c27b0; }
            .label { font-weight: bold; display: inline-block; width: 120px; }
            .text-container { line-height: 2; }
            .token { padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block; }
            .legend { background: white; padding: 15px; margin: 20px 0; border-radius: 10px; }
            .legend-item { display: inline-block; margin-right: 20px; }
            .legend-color { display: inline-block; width: 20px; height: 20px; margin-right: 5px; vertical-align: middle; border-radius: 3px; }
            .stats { background: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 10px; }
            .top-tokens { font-size: 0.9em; color: #666; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Attention Visualization: z_c (Content) vs z_d (Demographic)</h1>
            
            <div class="legend">
                <h3>Legend</h3>
                <div class="legend-item">
                    <span class="legend-color" style="background: linear-gradient(to right, #fff3e0, #ff5722);"></span>
                    z_c (Content) - Higher attention = more orange/red
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: linear-gradient(to right, #f3e5f5, #9c27b0);"></span>
                    z_d (Demographic) - Higher attention = more purple
                </div>
            </div>
            
            <h2>Samples</h2>
    """
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['text']
            labels_batch = batch['label'].cpu().numpy()
            
            output = model(input_ids, attention_mask)
            attn_c = output['attn_c']
            attn_d = output['attn_d']
            
            for i in range(input_ids.size(0)):
                if sample_count >= num_samples:
                    break
                
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                
                # Get attention values
                attn_c_vals = attn_c[i].cpu().numpy()
                attn_d_vals = attn_d[i].cpu().numpy()
                
                # Merge subword tokens
                tokens_merged_c, attn_c_merged = merge_subword_tokens(tokens, attn_c_vals)
                tokens_merged_d, attn_d_merged = merge_subword_tokens(tokens, attn_d_vals)
                
                # Filter out special tokens
                valid_mask_c = np.array([t not in ['[CLS]', '[SEP]', '[PAD]'] for t in tokens_merged_c])
                valid_mask_d = np.array([t not in ['[CLS]', '[SEP]', '[PAD]'] for t in tokens_merged_d])
                
                tokens_filtered_c = [t for t, v in zip(tokens_merged_c, valid_mask_c) if v]
                tokens_filtered_d = [t for t, v in zip(tokens_merged_d, valid_mask_d) if v]
                attn_c_filtered = attn_c_merged[valid_mask_c]
                attn_d_filtered = attn_d_merged[valid_mask_d]
                
                # Normalize attention for visualization
                if attn_c_filtered.sum() > 0:
                    attn_c_norm = attn_c_filtered / attn_c_filtered.max()
                else:
                    attn_c_norm = attn_c_filtered
                    
                if attn_d_filtered.sum() > 0:
                    attn_d_norm = attn_d_filtered / attn_d_filtered.max()
                else:
                    attn_d_norm = attn_d_filtered
                
                # Get top tokens
                top_c_idx = np.argsort(attn_c_filtered)[-3:][::-1]
                top_d_idx = np.argsort(attn_d_filtered)[-3:][::-1]
                top_c_tokens = [tokens_filtered_c[j] for j in top_c_idx if j < len(tokens_filtered_c)]
                top_d_tokens = [tokens_filtered_d[j] for j in top_d_idx if j < len(tokens_filtered_d)]
                
                # Generate HTML for z_c attention
                z_c_html = ""
                for j, (token, attn) in enumerate(zip(tokens_filtered_c, attn_c_norm)):
                    # Orange color scale
                    r = 255
                    g = int(255 - attn * 100)
                    b = int(255 - attn * 200)
                    z_c_html += f'<span class="token" style="background: rgb({r},{g},{b});">{token}</span> '
                
                # Generate HTML for z_d attention
                z_d_html = ""
                for j, (token, attn) in enumerate(zip(tokens_filtered_d, attn_d_norm)):
                    # Purple color scale
                    r = int(255 - attn * 100)
                    g = int(255 - attn * 150)
                    b = 255
                    z_d_html += f'<span class="token" style="background: rgb({r},{g},{b});">{token}</span> '
                
                label_str = "Good (1)" if labels_batch[i] == 1 else "Bad (0)"
                
                html_content += f"""
                <div class="sample">
                    <div class="sample-header">Sample #{sample_count + 1} | Label: {label_str}</div>
                    <div class="attention-row z_c-row">
                        <span class="label">z_c (Content):</span>
                        <div class="text-container">{z_c_html}</div>
                        <div class="top-tokens">Top tokens: {', '.join(top_c_tokens)}</div>
                    </div>
                    <div class="attention-row z_d-row">
                        <span class="label">z_d (Demographic):</span>
                        <div class="text-container">{z_d_html}</div>
                        <div class="top-tokens">Top tokens: {', '.join(top_d_tokens)}</div>
                    </div>
                </div>
                """
                
                sample_count += 1
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    html_path = os.path.join(save_dir, "attention_visualization.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ Saved attention visualization HTML to: {html_path}")
    print(f"  Open this file in a browser to see highlighted attention on text")
    
    return {'html_path': html_path, 'num_samples': sample_count}


def visualize_attention_comparison(model, dataloader, num_samples=10, save_dir="visualizations"):
    """
    Create side-by-side comparison of z_c and z_d attention for each sample.
    Shows which tokens each representation focuses on.
    
    Args:
        model: Trained CD_Model
        dataloader: DataLoader for samples
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = get_device(model)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    sample_count = 0
    
    print(f"\nGenerating attention comparison visualizations...")
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['text']
            labels_batch = batch['label'].cpu().numpy()
            
            output = model(input_ids, attention_mask)
            attn_c = output['attn_c']
            attn_d = output['attn_d']
            
            for i in range(input_ids.size(0)):
                if sample_count >= num_samples:
                    break
                
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                
                # Get attention values
                attn_c_vals = attn_c[i].cpu().numpy()
                attn_d_vals = attn_d[i].cpu().numpy()
                
                # Merge subword tokens
                tokens_merged, attn_c_merged = merge_subword_tokens(tokens, attn_c_vals)
                _, attn_d_merged = merge_subword_tokens(tokens, attn_d_vals)
                
                # Filter out special tokens
                valid_mask = np.array([t not in ['[CLS]', '[SEP]', '[PAD]'] for t in tokens_merged])
                tokens_filtered = [t for t, v in zip(tokens_merged, valid_mask) if v]
                attn_c_filtered = attn_c_merged[valid_mask]
                attn_d_filtered = attn_d_merged[valid_mask]
                
                # Normalize
                if attn_c_filtered.sum() > 0:
                    attn_c_filtered = attn_c_filtered / attn_c_filtered.sum()
                if attn_d_filtered.sum() > 0:
                    attn_d_filtered = attn_d_filtered / attn_d_filtered.sum()
                
                # Create figure
                fig, axes = plt.subplots(2, 1, figsize=(max(12, len(tokens_filtered) * 0.4), 8))
                
                # z_c attention bar chart
                ax = axes[0]
                bars_c = ax.bar(range(len(tokens_filtered)), attn_c_filtered, color='#ff9800', alpha=0.8)
                ax.set_xticks(range(len(tokens_filtered)))
                ax.set_xticklabels(tokens_filtered, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Attention Weight', fontsize=11)
                ax.set_title(f'z_c (Content) Attention - Sample #{sample_count + 1}', fontsize=12, fontweight='bold')
                ax.set_ylim(0, max(attn_c_filtered) * 1.2 if len(attn_c_filtered) > 0 else 1)
                
                # Highlight top 3 tokens
                top_c_idx = np.argsort(attn_c_filtered)[-3:]
                for idx in top_c_idx:
                    bars_c[idx].set_color('#e65100')
                    bars_c[idx].set_edgecolor('black')
                    bars_c[idx].set_linewidth(2)
                
                # z_d attention bar chart
                ax = axes[1]
                bars_d = ax.bar(range(len(tokens_filtered)), attn_d_filtered, color='#9c27b0', alpha=0.8)
                ax.set_xticks(range(len(tokens_filtered)))
                ax.set_xticklabels(tokens_filtered, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Attention Weight', fontsize=11)
                ax.set_title(f'z_d (Demographic) Attention - Sample #{sample_count + 1}', fontsize=12, fontweight='bold')
                ax.set_ylim(0, max(attn_d_filtered) * 1.2 if len(attn_d_filtered) > 0 else 1)
                
                # Highlight top 3 tokens
                top_d_idx = np.argsort(attn_d_filtered)[-3:]
                for idx in top_d_idx:
                    bars_d[idx].set_color('#4a148c')
                    bars_d[idx].set_edgecolor('black')
                    bars_d[idx].set_linewidth(2)
                
                # Add original sentence as suptitle
                label_str = "Good (1)" if labels_batch[i] == 1 else "Bad (0)"
                fig.suptitle(f'Label: {label_str}\n"{texts[i][:100]}..."' if len(texts[i]) > 100 else f'Label: {label_str}\n"{texts[i]}"',
                            fontsize=10, y=1.02)
                
                plt.tight_layout()
                filename = os.path.join(save_dir, f"attention_comparison_{sample_count:03d}.png")
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                
                sample_count += 1
    
    print(f"✓ Generated {sample_count} attention comparison visualizations")
    
    return {'num_samples': sample_count}
