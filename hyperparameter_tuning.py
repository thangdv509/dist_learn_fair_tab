#!/usr/bin/env python3
"""
Hyperparameter tuning script for finding the best parameters.
Uses grid search or random search to find optimal hyperparameters.
Updated to match new architecture (no MINE, adversarial classifier).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from data_loader import load_german_credit_data_balanced, load_dataset_generic
import numpy as np
import argparse
import json
import os
from datetime import datetime
from itertools import product
import random
# Import from main_attention
try:
    from main_attention import (
        TextDataset, CD_Model, 
        compute_kl_loss,
        compute_reconstruction_loss, compute_orthogonality_penalty,
        compute_and_cache_bert_embeddings,
        device
    )
except ImportError:
    # Fallback: import directly
    import main_attention
    TextDataset = main_attention.TextDataset
    CD_Model = main_attention.CD_Model
    compute_kl_loss = main_attention.compute_kl_loss
    compute_reconstruction_loss = main_attention.compute_reconstruction_loss
    compute_orthogonality_penalty = main_attention.compute_orthogonality_penalty
    compute_and_cache_bert_embeddings = main_attention.compute_and_cache_bert_embeddings
    device = main_attention.device

def train_and_evaluate(sentences, labels, 
                       lambda_adv=5.0, lambda_rec=0.5, lambda_ortho=0.5,
                       lr=0.001, batch_size=8, num_epochs=30, 
                       verbose=False, cached_embeddings=None):
    """
    Train model with given hyperparameters and return validation metrics.
    Uses a train/validation split.
    
    Args:
        cached_embeddings: Optional cached BERT embeddings dict
    """
    # Split data into train and validation (80/20)
    n_total = len(sentences)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_sentences = [sentences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_sentences = [sentences[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Split cached embeddings if available
    train_cached = val_cached = None
    if cached_embeddings is not None:
        train_cached = {
            'embeddings': cached_embeddings['embeddings'][train_indices],
            'input_ids': cached_embeddings['input_ids'][train_indices],
            'attention_mask': cached_embeddings['attention_mask'][train_indices]
        }
        val_cached = {
            'embeddings': cached_embeddings['embeddings'][val_indices],
            'input_ids': cached_embeddings['input_ids'][val_indices],
            'attention_mask': cached_embeddings['attention_mask'][val_indices]
        }
    
    train_dataset = TextDataset(train_sentences, train_labels, tokenizer, max_len=50, cached_embeddings=train_cached)
    val_dataset = TextDataset(val_sentences, val_labels, tokenizer, max_len=50, cached_embeddings=val_cached)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = CD_Model().to(device)
    
    # Optimizers
    optimizer = optim.Adam(
        list(model.encoder.parameters()) + 
        list(model.classifier.parameters()) +
        list(model.decoder.parameters()),
        lr=lr
    )
    optimizer_adv = optim.Adam(model.adversarial_classifier.parameters(), lr=lr)
    
    best_val_acc_z_d = 0.0
    best_val_acc_z_c = 1.0  # Lower is better (should be ~0.5)
    patience_count = 0
    patience = 5
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            # Get BERT embeddings (from cache if available)
            bert_embeddings = batch.get('bert_embeddings', None)
            if bert_embeddings is not None:
                bert_embeddings = bert_embeddings.to(device)
                original_h = bert_embeddings[:, 0, :]  # CLS token
            else:
                # Compute on-the-fly
                with torch.no_grad():
                    bert_output = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                    original_h = bert_output.last_hidden_state[:, 0, :]
                bert_embeddings = None
            
            # Adversarial classifier step
            optimizer_adv.zero_grad()
            output = model(input_ids, attention_mask, bert_embeddings=bert_embeddings)
            z_c_detached = output['z_c'].detach()
            logits_y_from_z_c = model.adversarial_classifier(z_c_detached)
            loss_adv_classifier = nn.functional.cross_entropy(logits_y_from_z_c, labels_batch)
            loss_adv_classifier.backward()
            optimizer_adv.step()
            
            # Main training step
            optimizer.zero_grad()
            output = model(input_ids, attention_mask, bert_embeddings=bert_embeddings)
            
            # Compute losses (matching main_attention.py)
            loss_task = nn.functional.cross_entropy(output['logits_y_from_z_d'], labels_batch)
            
            # Adversarial loss with penalty
            from main_attention import compute_random_guess_penalty
            loss_adv_ce = nn.functional.cross_entropy(output['logits_y_from_z_c'], labels_batch)
            loss_adv_penalty = compute_random_guess_penalty(output['logits_y_from_z_c'], target=0.5)
            loss_adv = loss_adv_ce + loss_adv_penalty
            
            loss_rec = compute_reconstruction_loss(original_h, output['reconstructed'])
            loss_kl_c = compute_kl_loss(output['mu_c'], output['logvar_c'])
            loss_kl_d = compute_kl_loss(output['mu_d'], output['logvar_d'])
            loss_kl = (loss_kl_c + loss_kl_d) / (output['mu_c'].size(0) * 2)
            loss_ortho = compute_orthogonality_penalty(output['z_c'], output['z_d'])
            
            # Combined loss
            loss_enc = (3.0 * loss_task + 
                       -lambda_adv * loss_adv + 
                       lambda_rec * loss_rec + 
                       0.5 * loss_kl + 
                       lambda_ortho * loss_ortho)
            
            loss_enc.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_correct_z_d = 0
        val_correct_z_c = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['label'].to(device)
                
                # Get cached embeddings if available
                bert_embeddings = batch.get('bert_embeddings', None)
                if bert_embeddings is not None:
                    bert_embeddings = bert_embeddings.to(device)
                
                output = model(input_ids, attention_mask, bert_embeddings=bert_embeddings)
                val_correct_z_d += (output['logits_y_from_z_d'].argmax(1) == labels_batch).sum().item()
                val_correct_z_c += (output['logits_y_from_z_c'].argmax(1) == labels_batch).sum().item()
                val_total += labels_batch.size(0)
        
        val_acc_z_d = val_correct_z_d / val_total if val_total > 0 else 0
        val_acc_z_c = val_correct_z_c / val_total if val_total > 0 else 0
        
        # Track best validation accuracy
        if val_acc_z_d > best_val_acc_z_d:
            best_val_acc_z_d = val_acc_z_d
            best_val_acc_z_c = val_acc_z_c
            patience_count = 0
        else:
            patience_count += 1
        
        if patience_count >= patience:
            break
    
    # Return metrics
    # Combined score: high z_d accuracy + z_c accuracy close to 0.5
    fairness_score = 1.0 - abs(val_acc_z_c - 0.5)  # Higher when z_c acc is close to 0.5
    combined_score = best_val_acc_z_d * fairness_score  # Balance between accuracy and fairness
    
    return {
        'val_acc_z_d': best_val_acc_z_d,
        'val_acc_z_c': best_val_acc_z_c,
        'fairness_score': fairness_score,
        'combined_score': combined_score
    }


def grid_search(sentences, labels, param_grid, num_epochs=30, verbose=True, cached_embeddings=None):
    """
    Perform grid search over hyperparameter space.
    
    param_grid: dict with lists of values to try
    cached_embeddings: Optional cached BERT embeddings
    """
    print("\n" + "="*70)
    print("GRID SEARCH HYPERPARAMETER TUNING")
    print("="*70)
    if cached_embeddings is not None:
        print("✓ Using cached BERT embeddings (faster training)")
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    print(f"\nTotal combinations to test: {len(combinations)}")
    print(f"Parameters to tune: {keys}")
    print(f"Will train for {num_epochs} epochs per combination\n")
    
    results = []
    best_score = -1.0
    best_params = None
    
    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        print(f"\n[{idx+1}/{len(combinations)}] Testing: {params}")
        
        try:
            metrics = train_and_evaluate(
                sentences, labels,
                lambda_adv=params.get('lambda_adv', 5.0),
                lambda_rec=params.get('lambda_rec', 0.5),
                lambda_ortho=params.get('lambda_ortho', 0.5),
                lr=params.get('lr', 0.001),
                batch_size=8,  # Fixed batch size (not tuned)
                num_epochs=num_epochs,
                verbose=verbose,
                cached_embeddings=cached_embeddings
            )
            
            result = {
                'params': params,
                'metrics': metrics
            }
            results.append(result)
            
            score = metrics['combined_score']
            print(f"  → Acc(z_d→y): {metrics['val_acc_z_d']:.4f}, "
                  f"Acc(z_c→y): {metrics['val_acc_z_c']:.4f}, "
                  f"Fairness: {metrics['fairness_score']:.4f}, "
                  f"Combined: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"  ★ NEW BEST! Combined Score: {score:.4f}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return results, best_params, best_score


def random_search(sentences, labels, param_distributions, n_iter=20, num_epochs=30, verbose=True, cached_embeddings=None):
    """
    Perform random search over hyperparameter space.
    
    param_distributions: dict with lists of values or (min, max) tuples for continuous
    cached_embeddings: Optional cached BERT embeddings
    """
    print("\n" + "="*70)
    print("RANDOM SEARCH HYPERPARAMETER TUNING")
    print("="*70)
    if cached_embeddings is not None:
        print("✓ Using cached BERT embeddings (faster training)")
    
    print(f"\nTotal iterations: {n_iter}")
    print(f"Parameters to tune: {list(param_distributions.keys())}")
    print(f"Will train for {num_epochs} epochs per iteration\n")
    
    results = []
    best_score = -1.0
    best_params = None
    
    for idx in range(n_iter):
        # Sample random parameters
        params = {}
        for key, values in param_distributions.items():
            if isinstance(values, tuple) and len(values) == 2:
                # Continuous range
                params[key] = random.uniform(values[0], values[1])
            else:
                # Discrete list
                params[key] = random.choice(values)
        
        print(f"\n[{idx+1}/{n_iter}] Testing: {params}")
        
        try:
            metrics = train_and_evaluate(
                sentences, labels,
                lambda_adv=params.get('lambda_adv', 5.0),
                lambda_rec=params.get('lambda_rec', 0.5),
                lambda_ortho=params.get('lambda_ortho', 0.5),
                lr=params.get('lr', 0.001),
                batch_size=8,  # Fixed batch size (not tuned)
                num_epochs=num_epochs,
                verbose=verbose,
                cached_embeddings=cached_embeddings
            )
            
            result = {
                'params': params,
                'metrics': metrics
            }
            results.append(result)
            
            score = metrics['combined_score']
            print(f"  → Acc(z_d→y): {metrics['val_acc_z_d']:.4f}, "
                  f"Acc(z_c→y): {metrics['val_acc_z_c']:.4f}, "
                  f"Fairness: {metrics['fairness_score']:.4f}, "
                  f"Combined: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"  ★ NEW BEST! Combined Score: {score:.4f}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return results, best_params, best_score


def save_results(results, best_params, best_score, output_dir="tuning_results"):
    """Save tuning results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"tuning_results_{timestamp}.json")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    output = {
        'timestamp': timestamp,
        'best_params': convert_to_serializable(best_params),
        'best_score': float(best_score),
        'all_results': convert_to_serializable(results)
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {filename}")
    return filename


def print_summary(results, best_params, best_score):
    """Print summary of tuning results"""
    print("\n" + "="*70)
    print("TUNING SUMMARY")
    print("="*70)
    
    if best_params is None:
        print("No successful runs!")
        return
    
    print(f"\n★ BEST PARAMETERS:")
    for key, value in sorted(best_params.items()):
        print(f"  {key}: {value}")
    
    print(f"\n★ BEST SCORES:")
    best_result = next(r for r in results if r['params'] == best_params)
    metrics = best_result['metrics']
    print(f"  Acc(z_d→y): {metrics['val_acc_z_d']:.4f} (target: ~1.0)")
    print(f"  Acc(z_c→y): {metrics['val_acc_z_c']:.4f} (target: ~0.5)")
    print(f"  Fairness Score: {metrics['fairness_score']:.4f}")
    print(f"  Combined Score: {best_score:.4f}")
    
    # Top 5 results
    sorted_results = sorted(results, key=lambda x: x['metrics']['combined_score'], reverse=True)
    print(f"\n★ TOP 5 CONFIGURATIONS:")
    for i, result in enumerate(sorted_results[:5], 1):
        params = result['params']
        metrics = result['metrics']
        print(f"\n  {i}. Combined Score: {metrics['combined_score']:.4f}")
        print(f"     Acc(z_d→y): {metrics['val_acc_z_d']:.4f}, "
              f"Acc(z_c→y): {metrics['val_acc_z_c']:.4f}")
        print(f"     Params: {params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for fairness model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="german-credit-data",
        help="Dataset to use"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "random"],
        default="random",
        help="Search method: grid or random"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of iterations for random search"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs per configuration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tuning_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use cached BERT embeddings (faster training)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable BERT cache (recompute embeddings every time)"
    )
    parser.add_argument(
        "--force-recompute-cache",
        action="store_true",
        help="Force recompute BERT embeddings even if cache exists"
    )
    
    args = parser.parse_args()
    
    # Handle cache flags
    use_cache = args.use_cache and not args.no_cache
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load dataset
    print("\n" + "="*70)
    print(f"LOADING DATASET: {args.dataset}")
    print("="*70)
    
    try:
        if args.dataset == "german-credit-data":
            sentences, labels, data, _ = load_german_credit_data_balanced(n_samples=None)
        else:
            sentences, labels, data, _ = load_dataset_generic(
                args.dataset,
                n_samples=500
            )
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        exit(1)
    
    print(f"\nLoaded {len(sentences)} samples")
    print(f"Label distribution: {sum(labels)} (1), {len(labels) - sum(labels)} (0)")
    
    # Pre-compute and cache BERT embeddings if requested
    cached_embeddings = None
    if use_cache:
        print(f"\n{'='*70}")
        print("PRE-COMPUTING BERT EMBEDDINGS")
        print(f"{'='*70}")
        cached_data = compute_and_cache_bert_embeddings(
            sentences, args.dataset, max_len=50, force_recompute=args.force_recompute_cache
        )
        cached_embeddings = cached_data
        print(f"✓ Cached embeddings ready for hyperparameter tuning\n")
    
    # Define parameter space
    # Note: batch_size is fixed at 8 (not tuned) because:
    # - With small dataset (~1000 samples), batch size has minimal impact
    # - BERT embeddings are cached, so batch size mainly affects training speed
    # - Focus tuning on more important hyperparameters (lambda_adv, lambda_rec, lambda_ortho, lr)
    fixed_batch_size = 8
    
    if args.method == "grid":
        param_grid = {
            'lambda_adv': [3.0, 5.0, 7.0],
            'lambda_rec': [0.3, 0.5, 0.7],
            'lambda_ortho': [0.3, 0.5, 0.7],
            'lr': [0.0005, 0.001, 0.002]
            # batch_size fixed at 8
        }
        
        results, best_params, best_score = grid_search(
            sentences, labels,
            param_grid,
            num_epochs=args.epochs,
            cached_embeddings=cached_embeddings
        )
    
    else:  # random search
        param_distributions = {
            'lambda_adv': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'lambda_rec': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'lambda_ortho': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'lr': [0.0003, 0.0005, 0.001, 0.0015, 0.002]
            # batch_size fixed at 8
        }
        
        results, best_params, best_score = random_search(
            sentences, labels,
            param_distributions,
            n_iter=args.n_iter,
            num_epochs=args.epochs,
            cached_embeddings=cached_embeddings
        )
    
    # Print summary
    print_summary(results, best_params, best_score)
    
    # Save results
    save_results(results, best_params, best_score, args.output_dir)
    
    print("\n" + "="*70)
    print("✓ Hyperparameter tuning complete!")
    print("="*70 + "\n")
