#!/usr/bin/env python3
"""
Fairness learning with disentangled representations from Natural Language (LIRD framework).

Architecture:
1. Shared BERT encoder + 2 separate heads (Content head & Demographic head)
2. Attention pooling in heads to extract z_c and z_d from shared BERT output
3. Separate decoders: D_c(z_c) and D_d(z_d) for additive reconstruction

LIRD Objectives (HSIC-based, no adversarial training):
- L_task: z_c → y (high accuracy) - z_c contains task-relevant info
- L_y: HSIC(z_d, y) (minimize) - z_d is independent of task label y
- L_ind: HSIC(z_c, z_d) (minimize) - z_c and z_d are statistically independent
- L_rec: ||D_c(z_c) + D_d(z_d) - h||² (minimize) - additive reconstruction
- L_var: Variance/energy constraint on z_d to prevent collapse to noise

Key Objectives:
- z_c (Content): Can predict y well (high accuracy)
- z_d (Demographic): Cannot predict y (accuracy ~50/50, random guess), but carries input information
- z_c ⊥ z_d: Statistically independent (HSIC ≈ 0)
- D_c(z_c) + D_d(z_d) → h: Can reconstruct original representation
- Attention visualization: See which tokens each head focuses on
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data_loader import load_german_credit_data_balanced, load_dataset_generic, save_processed_data
from visualization import (
    visualize_reconstruction_and_attention, 
    visualize_batch_analysis, 
    visualize_attention_heatmap_batch, 
    analyze_attention_difference, 
    visualize_latent_space_pca,
    visualize_attention_on_text,
    visualize_attention_comparison
)
import os
import numpy as np
import argparse

# Import from components module
from components import (
    CD_Model,
    TextDataset,
    train_cd_model,
    save_encodings,
    compute_and_cache_tokenized_data,
    device
)

# Auto-detect device: use CUDA if available, otherwise CPU
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

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
        default=1.0,
        help="HSIC(z_d, y) weight for label-independence (default: 1.0, LIRD framework)"
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
        default=1.0,
        help="HSIC(z_c, z_d) weight for disentanglement (default: 1.0, LIRD framework)"
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
                args.batch_size = 200
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
        lambda_var=args.lambda_var,
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
