"""
Training function for fairness learning with disentangled representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import datetime
import os

from .models import CD_Model, device
from .dataset import TextDataset
from .losses import (
    compute_kl_loss, 
    compute_reconstruction_loss, 
    compute_hsic,
    compute_variance_constraint,
    compute_energy_constraint
)
from .tokenization import compute_and_cache_tokenized_data


def train_cd_model(sentences, labels, num_epochs=50, batch_size=8, lambda_task=3.0, lambda_adv=8.0, lambda_rec=0.5, lambda_ortho=1.0, lambda_kl=0.01, lambda_var=0.1, latent_dim=64, lr=2e-5, use_cache=True, dataset_name="default", force_recompute_cache=False, use_amp=False):
    """
    Train the model to learn disentangled representations from Natural Language (LIRD framework).
    
    Architecture:
    - 1 shared BERT encoder (bert-base-uncased)
    - 2 separate heads: ContentHead (→ z_c) and DemographicHead (→ z_d)
    - Each head uses attention pooling + VAE bottleneck (for compression only)
    - 2 separate decoders: D_c(z_c) and D_d(z_d) for additive reconstruction
    
    LIRD OBJECTIVES (HSIC-based, no adversarial training):
    1. L_task: z_c → y (high accuracy) - z_c contains task-relevant info
    2. L_y: HSIC(z_d, y) (minimize) - z_d is independent of task label y
    3. L_ind: HSIC(z_c, z_d) (minimize) - z_c and z_d are statistically independent
    4. L_rec: ||D_c(z_c) + D_d(z_d) - h||² (minimize) - additive reconstruction
    
    + L_kl: VAE KL regularization KL(q(z_c|x)||N(0,I)) + KL(q(z_d|x)||N(0,I))
            Ensures z_c and z_d follow standard normal distribution for:
            - Smooth latent space (good interpolation)
            - Proper sampling (can sample z ~ N(0,I) and decode to generate data)
            - Normalized distribution (prevents sampling outside manifold)
    + L_var: Variance/energy constraint on z_d to prevent collapse to noise
    
    NOTE: VAE is used for compression (768 → latent_dim) AND generation (can sample z ~ N(0,I) and decode).
    NOTE: No GRL/adversarial training - pure HSIC-based LIRD framework.
    """
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Latent dim: {latent_dim}")
    print(f"\nLIRD OBJECTIVES (HSIC-based):")
    print(f"  1. λ_task={lambda_task} : z_c → y (target: high accuracy ~1.0)")
    print(f"  2. λ_adv={lambda_adv} : HSIC(z_d, y) (target: ~0, label-independence)")
    print(f"  3. λ_ortho={lambda_ortho} : HSIC(z_c, z_d) (target: ~0, disentangle)")
    print(f"  4. λ_rec={lambda_rec} : ||D_c(z_c) + D_d(z_d) - h||² (target: ~0, reconstruction)")
    print(f"\n  + λ_kl={lambda_kl} : VAE KL regularization KL(q(z_c|x)||N(0,I)) + KL(q(z_d|x)||N(0,I))")
    print(f"    → Ensures smooth latent space and proper sampling for generation")
    print(f"  + λ_var={lambda_var} : Variance/energy constraint on z_d (prevent collapse)")
    print(f"{'='*70}\n")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Pre-compute and cache tokenized data if requested (speeds up training significantly)
    cached_data = None
    if use_cache:
        print(f"\n{'='*70}")
        print("CACHE SETUP - Tokenizing and caching data for faster training")
        print(f"{'='*70}")
        cached_data = compute_and_cache_tokenized_data(
            sentences, dataset_name, max_len=256, force_recompute=force_recompute_cache
        )
        print(f"{'='*70}\n")
    else:
        print("⚠ Cache disabled - tokenization will happen on-the-fly (slower)")
    
    dataset = TextDataset(sentences, labels, tokenizer, max_len=256, cached_data=cached_data)
    
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
                  list(model.decoder_c.parameters()) + 
                  list(model.decoder_d.parameters()), 'lr': head_lr}
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
        'loss_var': [],  # Variance/energy constraint to prevent z_d collapse
        'acc_task_from_z_d': [], 'acc_task_from_z_c': [],
        'per_class_acc_z_d': [], 'per_class_acc_z_c': []
    }
    
    # Initialize per-class accuracy accumulators (for imbalanced task labels)
    class_correct_z_d = torch.zeros(num_classes, dtype=torch.long)
    class_total_z_d = torch.zeros(num_classes, dtype=torch.long)
    class_correct_z_c = torch.zeros(num_classes, dtype=torch.long)
    class_total_z_c = torch.zeros(num_classes, dtype=torch.long)
    
    best_loss = float('inf')
    best_model_state = None  # Store best model state
    best_model_epoch = 0
    
    print(f"Dataset: {len(sentences)} samples, {sum(labels)} good, {len(labels) - sum(labels)} bad")
    
    for epoch in range(num_epochs):
        # Initialize accumulators for this epoch
        model.train()
        loss_enc_total = loss_task_total = loss_adv_total = loss_ortho_total = loss_rec_total = 0
        loss_kl_total = 0
        loss_var_total = 0
        correct_task_from_z_d = correct_task_from_z_c = total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Use non_blocking=True for faster GPU transfer (overlaps with computation)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels_batch = batch['label'].to(device, non_blocking=True)
                
                # === MAIN TRAINING STEP (LIRD Framework) ===
                # LIRD uses HSIC instead of adversarial training:
                # - HSIC(z_c, z_d): Ensures z_c and z_d are statistically independent
                # - HSIC(z_d, y): Ensures z_d is independent of task label y
                # No need for adversarial classifier training with GRL
                
                optimizer.zero_grad()
                # Forward pass
                output = model(input_ids, attention_mask)
                
                # ============ LIRD LOSS COMPONENTS ============
                # 1. L_task: z_c → y (MINIMIZE) - sufficiency (z_c is minimal sufficient for y)
                loss_task = nn.functional.cross_entropy(
                    output['logits_y_from_z_c'], 
                    labels_batch,
                    weight=class_weights
                )
                
                # 2. L_ind: HSIC(z_c, z_d) (MINIMIZE) - disentangle
                # Ensures z_c and z_d are statistically independent
                loss_ind = compute_hsic(output['z_c'], output['z_d'])
                
                # 3. L_y: HSIC(z_d, y) (MINIMIZE) - label-independence
                # Ensures z_d is independent of task label y
                # Convert labels to one-hot for HSIC computation
                labels_onehot = torch.zeros(labels_batch.size(0), num_classes, device=labels_batch.device)
                labels_onehot.scatter_(1, labels_batch.unsqueeze(1), 1.0)
                loss_y = compute_hsic(output['z_d'], labels_onehot)
                
                # 4. L_rec: z_c + z_d → x (MINIMIZE) - reconstruction
                loss_rec = compute_reconstruction_loss(output['original_h'], output['reconstructed'])
                
                # + L_kl: VAE KL regularization to N(0,I) prior for both z_c and z_d
                # This ensures:
                # 1. Smooth latent space (good interpolation between points)
                # 2. Proper sampling (can sample z ~ N(0,I) and decode to generate realistic data)
                # 3. Normalized distribution (prevents sampling outside the learned manifold)
                loss_kl_c = compute_kl_loss(output['mu_c'], output['logvar_c'])  # KL(q(z_c|x) || N(0,I))
                loss_kl_d = compute_kl_loss(output['mu_d'], output['logvar_d'])  # KL(q(z_d|x) || N(0,I))
                loss_kl = loss_kl_c + loss_kl_d
                
                # + L_var: Variance/energy constraint to prevent z_d collapse
                # Encourage z_d to have sufficient variance and D_d(z_d) to have sufficient energy
                loss_var_z = compute_variance_constraint(output['z_d'], min_variance=0.1)
                loss_var_d = compute_energy_constraint(output['reconstructed_d'], min_energy=0.1)
                loss_var = loss_var_z + loss_var_d
                
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
                
                # ============ COMBINED ENCODER LOSS (LIRD Framework) ============
                # Following LIRD: min_θ L_task(y, C(z_c)) + λ_rec ||D_c(z_c) + D_d(z_d) - h||² 
                #                  + λ_ind HSIC(z_c, z_d) + λ_y HSIC(z_d, y)
                # 1. L_task: z_c → y (MINIMIZE → Acc ~1.0) - sufficiency
                # 2. L_rec: D_c(z_c) + D_d(z_d) → h (MINIMIZE → 0) - additive reconstruction
                # 3. L_ind: HSIC(z_c, z_d) (MINIMIZE → 0) - disentangle
                # 4. L_y: HSIC(z_d, y) (MINIMIZE → 0) - label-independence
                # + L_kl: VAE KL regularization to N(0,I) (smooth latent space + generation)
                # + L_var: Variance/energy constraint to prevent z_d collapse
                
                loss_enc = (
                    lambda_task * loss_task_updated +    # 1. L_task: sufficiency
                    lambda_rec * loss_rec +              # 2. L_rec: additive reconstruction
                    lambda_ortho * loss_ind +            # 3. L_ind: HSIC(z_c, z_d) - disentangle
                    lambda_adv * loss_y +                # 4. L_y: HSIC(z_d, y) - label-independence
                    lambda_kl * loss_kl +                # VAE regularization
                    lambda_var * loss_var                # Variance/energy constraint (prevent collapse)
                )
                
                # Train BERT_c, BERT_d and decoder with full loss (using AMP if enabled)
                if scaler is not None:
                    scaler.scale(loss_enc).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_enc.backward()
                    optimizer.step()
                
                # Accumulate losses for monitoring:
                # - loss_enc: Combined loss (used for training)
                # - Individual losses: Track each component separately for debugging
                loss_enc_total += loss_enc.item()              # Combined: λ_task*L_task + λ_rec*L_rec + λ_ind*HSIC(z_c,z_d) + λ_y*HSIC(z_d,y) + λ_kl*L_kl + λ_var*L_var
                loss_task_total += loss_task_updated.item()    # 1. L_task: z_c → y (sufficiency)
                loss_adv_total += loss_y.item()                # 2. L_y: HSIC(z_d, y) (label-independence)
                loss_ortho_total += loss_ind.item()            # 3. L_ind: HSIC(z_c, z_d) (disentangle)
                loss_rec_total += loss_rec.item()              # 4. L_rec: D_c(z_c) + D_d(z_d) → h (additive reconstruction)
                loss_kl_total += loss_kl.item()                # 5. L_kl: VAE regularization
                loss_var_total += loss_var.item()              # 6. L_var: Variance/energy constraint (prevent collapse)
                
                with torch.no_grad():
                    # Compute accuracy from z_d using adversarial classifier (for monitoring only, not trained)
                    # In LIRD, we use HSIC(z_d, y) instead of adversarial training
                    z_d_for_acc = output['z_d'].detach()
                    logits_y_from_z_d_monitor = model.adversarial_classifier(z_d_for_acc)
                    pred_z_d = logits_y_from_z_d_monitor.argmax(1)
                    
                    # Use updated classifier predictions for accuracy
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
        history['loss_var'].append(loss_var_total / num_batches)
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
        # scheduler_adv not used in LIRD (no adversarial training)
        # scheduler_adv.step(history['loss_adv'][-1])
        current_lr = optimizer.param_groups[0]['lr']
        classifier_lr = optimizer_classifier.param_groups[0]['lr']
        
        # Clear GPU cache periodically to prevent memory fragmentation
        if torch.cuda.is_available() and epoch % 5 == 0:
            torch.cuda.empty_cache()
        
        # Print every epoch (always print all epochs for monitoring)
        random_target_str = f"(→{1.0/num_classes:.3f})" if num_classes > 2 else "(→0.5)"
        per_class_acc_z_d_str = f"PerCls: {history['per_class_acc_z_d'][-1]:.3f}" if len(history['per_class_acc_z_d']) > 0 else ""
        per_class_acc_z_c_str = f"PerCls: {history['per_class_acc_z_c'][-1]:.3f}" if len(history['per_class_acc_z_c']) > 0 else ""
        
        # Print detailed epoch info (LIRD losses)
        print(f"\nEpoch {epoch:3d}/{num_epochs}")
        print(f"  LIRD Losses: L_task={history['loss_task'][-1]:.4f} | L_y={history['loss_adv'][-1]:.4f} | L_ind={history['loss_ortho'][-1]:.4f} | L_rec={history['loss_rec'][-1]:.4f}")
        print(f"  Regularization: L_kl={history['loss_kl'][-1]:.4f} | L_var={history['loss_var'][-1]:.4f} | Total L_enc={avg_loss_enc:.4f}")
        # Show weighted contributions for debugging
        weighted_task = lambda_task * history['loss_task'][-1]
        weighted_adv = lambda_adv * history['loss_adv'][-1]
        weighted_ortho = lambda_ortho * history['loss_ortho'][-1]
        weighted_rec = lambda_rec * history['loss_rec'][-1]
        weighted_kl = lambda_kl * history['loss_kl'][-1]
        weighted_var = lambda_var * history['loss_var'][-1]
        print(f"  Weighted: λ_task*L_task={weighted_task:.4f} | λ_adv*L_y={weighted_adv:.4f} | λ_ortho*L_ind={weighted_ortho:.4f} | λ_rec*L_rec={weighted_rec:.4f} | λ_kl*L_kl={weighted_kl:.4f} | λ_var*L_var={weighted_var:.4f}")
        print(f"  Accuracy: Acc(z_c→y)={history['acc_task_from_z_c'][-1]:.3f} {per_class_acc_z_c_str} (→1.0)")
        print(f"            Acc(z_d→y)={history['acc_task_from_z_d'][-1]:.3f} {per_class_acc_z_d_str} {random_target_str}")
        print(f"  LR: encoder={current_lr:.6f}, classifier={classifier_lr:.6f}")
        
        # Save best model based on lowest loss
        if avg_loss_enc < best_loss:
            best_loss = avg_loss_enc
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
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    random_target = 1.0 / num_classes
    random_target_str = f"~{random_target:.3f}" if num_classes > 2 else "~0.5"
    per_class_acc_z_d_final = history['per_class_acc_z_d'][-1] if len(history['per_class_acc_z_d']) > 0 else 0.0
    per_class_acc_z_c_final = history['per_class_acc_z_c'][-1] if len(history['per_class_acc_z_c']) > 0 else 0.0
    
    print(f"\nLIRD OBJECTIVES - Final Results:")
    print(f"  1. L_task (z_c → y): Acc={history['acc_task_from_z_c'][-1]:.3f} (target: ~1.0) - sufficiency")
    print(f"  2. L_y (HSIC(z_d, y)): {history['loss_adv'][-1]:.4f} (target: ~0) - label-independence")
    print(f"     Acc(z_d → y): {history['acc_task_from_z_d'][-1]:.3f} (target: {random_target_str})")
    print(f"  3. L_ind (HSIC(z_c, z_d)): {history['loss_ortho'][-1]:.4f} (target: ~0) - disentangle")
    print(f"  4. L_rec (reconstruction): {history['loss_rec'][-1]:.4f} (target: ~0) - additive reconstruction")
    print(f"\n  + L_kl (VAE KL to N(0,I)): {history['loss_kl'][-1]:.4f} (smooth latent space + generation)")
    print(f"  + L_var (variance/energy constraint): {history['loss_var'][-1]:.4f} (prevent z_d collapse)")
    print(f"\nNOTE: Per-class accuracy is w.r.t. task labels (y), NOT sensitive attributes")
    print(f"{'='*70}")
    
    # Save best model
    if best_model_state is not None:
        model_dir = f"models/{dataset_name}"
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_path = os.path.join(model_dir, f"best_model_{timestamp}.pth")
        
        # Save best model with atomic write (save to temp file first, then rename)
        # This prevents corruption if training is interrupted during save
        import tempfile
        import shutil
        
        # Save best model atomically
        temp_best_path = best_model_path + ".tmp"
        try:
            torch.save(best_model_state, temp_best_path)
            # Verify the file can be loaded before renaming
            test_checkpoint = torch.load(temp_best_path, map_location='cpu', weights_only=False)
            shutil.move(temp_best_path, best_model_path)
            print(f"\n✓ Best model saved to: {best_model_path}")
            print(f"  Epoch: {best_model_epoch}, Loss: {best_loss:.4f}")
            print(f"  Acc(z_c→y): {best_model_state['acc_z_c']:.3f}, Acc(z_d→y): {best_model_state['acc_z_d']:.3f}")
        except Exception as e:
            # Clean up temp file if save failed
            if os.path.exists(temp_best_path):
                os.remove(temp_best_path)
            print(f"⚠ Warning: Failed to save best model: {e}")
            raise
        
        # Also save final model atomically
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
        
        temp_final_path = final_model_path + ".tmp"
        try:
            torch.save(final_model_state, temp_final_path)
            # Verify the file can be loaded before renaming
            test_checkpoint = torch.load(temp_final_path, map_location='cpu', weights_only=False)
            shutil.move(temp_final_path, final_model_path)
            print(f"✓ Final model saved to: {final_model_path}")
        except Exception as e:
            # Clean up temp file if save failed
            if os.path.exists(temp_final_path):
                os.remove(temp_final_path)
            print(f"⚠ Warning: Failed to save final model: {e}")
            raise
    else:
        print("\n⚠ No best model to save (training may have failed)")
    
    return model, history
