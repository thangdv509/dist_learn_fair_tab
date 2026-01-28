#!/usr/bin/env python3
"""
Script to perform K-means clustering on z_c embeddings from ORIGINAL data, 
find optimal K, and sample representative sentences from each cluster.

This is for clustering ORIGINAL training data to understand data distribution
and sample representative examples for synthetic data generation.

Usage (with pre-computed embeddings):
    python cluster_and_sample.py \
        --embeddings embeddings/original_z_c.npy \
        --sentences embeddings/original.csv \
        --output-dir sampled_data \
        --n-examples 10

Usage (auto-embed from dataset):
    python cluster_and_sample.py \
        --model-path models/dataset/best_model.pth \
        --dataset german-credit-data \
        --output-dir sampled_data \
        --n-examples 10
"""

import numpy as np
import pandas as pd
import argparse
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import json
from datetime import datetime
import sys

# Add parent directory to path to import embed_data functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from embed_data import load_model, embed_data
    EMBED_AVAILABLE = True
except ImportError:
    EMBED_AVAILABLE = False
    print("⚠ Warning: Cannot import embed_data. Will only work with pre-computed embeddings.")

try:
    from data_loader import load_german_credit_data_balanced, load_dataset_generic
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False


def load_embeddings(embeddings_path):
    """Load z_c embeddings from numpy file."""
    print(f"Loading embeddings from: {embeddings_path}")
    z_c = np.load(embeddings_path)
    print(f"✓ Loaded embeddings: shape {z_c.shape}")
    return z_c


def load_sentences(sentences_path):
    """Load sentences from CSV file."""
    print(f"Loading sentences from: {sentences_path}")
    df = pd.read_csv(sentences_path)
    
    # Try different column names
    if 'sentence' in df.columns:
        sentences = df['sentence'].tolist()
    elif 'text' in df.columns:
        sentences = df['text'].tolist()
    elif len(df.columns) == 1:
        sentences = df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Could not find 'sentence' or 'text' column in CSV. Available columns: {df.columns.tolist()}")
    
    # Also try to load labels if available
    labels = df['label'].tolist() if 'label' in df.columns else None
    
    print(f"✓ Loaded {len(sentences)} sentences")
    if labels is not None:
        print(f"✓ Loaded {len(labels)} labels")
    
    return sentences, labels, df


def find_optimal_k(z_c, max_k=20, min_k=2, method='silhouette'):
    """
    Find optimal number of clusters using elbow method or silhouette score.
    
    Args:
        z_c: Embeddings array [n_samples, latent_dim]
        max_k: Maximum K to test
        min_k: Minimum K to test
        method: 'silhouette', 'elbow', or 'davies_bouldin'
    
    Returns:
        optimal_k: Optimal number of clusters
        scores: Dictionary with scores for each K
    """
    print(f"\nFinding optimal K (testing K={min_k} to {max_k})...")
    
    n_samples = z_c.shape[0]
    max_k = min(max_k, n_samples - 1)  # Can't have more clusters than samples
    
    k_range = range(min_k, max_k + 1)
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    
    for k in k_range:
        print(f"  Testing K={k}...", end=' ', flush=True)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(z_c)
        
        inertia = kmeans.inertia_
        inertias.append(inertia)
        
        # Silhouette score (higher is better, range: -1 to 1)
        if k > 1:
            sil_score = silhouette_score(z_c, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(-1)
        
        # Davies-Bouldin score (lower is better)
        if k > 1:
            db_score = davies_bouldin_score(z_c, labels)
            davies_bouldin_scores.append(db_score)
        else:
            davies_bouldin_scores.append(float('inf'))
        
        print(f"✓ (silhouette={silhouette_scores[-1]:.3f}, DB={davies_bouldin_scores[-1]:.3f})")
    
    # Find optimal K based on method
    if method == 'silhouette':
        # Higher silhouette score is better
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_range)[optimal_idx]
        optimal_score = silhouette_scores[optimal_idx]
        print(f"\n✓ Optimal K={optimal_k} (silhouette score: {optimal_score:.3f})")
    elif method == 'davies_bouldin':
        # Lower Davies-Bouldin score is better
        optimal_idx = np.argmin(davies_bouldin_scores)
        optimal_k = list(k_range)[optimal_idx]
        optimal_score = davies_bouldin_scores[optimal_idx]
        print(f"\n✓ Optimal K={optimal_k} (Davies-Bouldin score: {optimal_score:.3f})")
    else:  # elbow method
        # Elbow method: Find point where inertia decrease slows down (diminishing returns)
        # Inertia = within-cluster sum of squares (LOWER is better)
        # As K increases, inertia decreases, but we want the "elbow" where
        # increasing K doesn't help much anymore (rate of decrease becomes small)
        
        if len(inertias) < 3:
            optimal_k = min_k
            optimal_inertia = inertias[0] if len(inertias) > 0 else 0
        else:
            # Calculate decrease in inertia for each step (how much inertia drops when K increases)
            # Inertia decreases, so decreases will be positive
            decreases = np.array(inertias[:-1]) - np.array(inertias[1:])  # How much inertia drops
            
            # Elbow point: where the decrease becomes small (diminishing returns)
            # Method: Find point where decrease drops below a threshold relative to max decrease
            # OR find point with maximum "distance" from line connecting first and last points
            
            # Use "kneedle" algorithm approach: find point with maximum distance from line
            # connecting (min_k, inertia[min_k]) to (max_k, inertia[max_k])
            first_point = np.array([k_range[0], inertias[0]])
            last_point = np.array([k_range[-1], inertias[-1]])
            
            # Calculate distance from each point to the line
            max_dist = -1
            optimal_idx = 0
            
            for i, (k, inertia) in enumerate(zip(k_range, inertias)):
                point = np.array([k, inertia])
                
                # Distance from point to line segment
                # Using formula: distance = |(y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
                numerator = abs((last_point[1] - first_point[1]) * point[0] - 
                               (last_point[0] - first_point[0]) * point[1] + 
                               last_point[0] * first_point[1] - last_point[1] * first_point[0])
                denominator = np.sqrt((last_point[1] - first_point[1])**2 + (last_point[0] - first_point[0])**2)
                
                if denominator > 0:
                    dist = numerator / denominator
                    if dist > max_dist:
                        max_dist = dist
                        optimal_idx = i
            
            optimal_k = list(k_range)[optimal_idx]
            optimal_inertia = inertias[optimal_idx]
        
        print(f"\n✓ Optimal K={optimal_k} (elbow method)")
        print(f"  Inertia at K={optimal_k}: {optimal_inertia:.2f} (lower is better)")
        print(f"  Note: Elbow is where inertia decrease slows down (diminishing returns)")
    
    scores = {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'optimal_k': optimal_k,
        'method': method
    }
    
    return optimal_k, scores


def plot_k_selection(scores, output_dir):
    """Plot K selection metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    k_range = scores['k_range']
    
    # Inertia (Elbow method)
    # Note: Lower inertia is better (tighter clusters)
    # Inertia = within-cluster sum of squares
    # Elbow point is where inertia decrease slows down (diminishing returns)
    axes[0].plot(k_range, scores['inertias'], 'bo-', label='Inertia (lower is better)')
    axes[0].axvline(x=scores['optimal_k'], color='r', linestyle='--', label=f'Optimal K={scores["optimal_k"]}')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia (Within-cluster Sum of Squares)')
    axes[0].set_title('Elbow Method\n(Lower Inertia = Tighter Clusters)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Don't invert - lower values are at bottom, which is correct
    
    # Silhouette score
    axes[1].plot(k_range, scores['silhouette_scores'], 'go-')
    axes[1].axvline(x=scores['optimal_k'], color='r', linestyle='--', label=f'Optimal K={scores["optimal_k"]}')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score (higher is better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Davies-Bouldin score
    axes[2].plot(k_range, scores['davies_bouldin_scores'], 'mo-')
    axes[2].axvline(x=scores['optimal_k'], color='r', linestyle='--', label=f'Optimal K={scores["optimal_k"]}')
    axes[2].set_xlabel('Number of Clusters (K)')
    axes[2].set_ylabel('Davies-Bouldin Score')
    axes[2].set_title('Davies-Bouldin Score (lower is better)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'k_selection_metrics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved K selection plot to: {plot_path}")
    plt.close()


def cluster_and_sample(z_c, sentences, labels=None, k=None, n_examples=10, random_state=42):
    """
    Perform K-means clustering and sample examples from each cluster.
    
    Args:
        z_c: Embeddings [n_samples, latent_dim]
        sentences: List of sentences
        labels: Optional labels
        k: Number of clusters (if None, will find optimal)
        n_examples: Number of examples to sample from each cluster
        random_state: Random seed
    
    Returns:
        clusters: Dictionary with cluster assignments and sampled data
    """
    if k is None:
        raise ValueError("k must be specified")
    
    print(f"\nPerforming K-means clustering with K={k}...")
    np.random.seed(random_state)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(z_c)
    
    # Get cluster statistics
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print(f"✓ Clustering complete!")
    print(f"  Cluster sizes: {dict(zip(unique_clusters, counts))}")
    
    # Sample from each cluster
    clusters = {}
    for cluster_id in unique_clusters:
        # Get indices of samples in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        
        # Sample n_examples (or all if cluster is smaller)
        n_sample = min(n_examples, cluster_size)
        sampled_indices = np.random.choice(cluster_indices, size=n_sample, replace=False)
        
        # Get sampled data
        sampled_sentences = [sentences[i] for i in sampled_indices]
        sampled_z_c = z_c[sampled_indices]
        
        cluster_data = {
            'cluster_id': int(cluster_id),
            'cluster_size': int(cluster_size),
            'n_sampled': int(n_sample),
            'indices': sampled_indices.tolist(),
            'sentences': sampled_sentences,
            'z_c': sampled_z_c.tolist(),
            'centroid': kmeans.cluster_centers_[cluster_id].tolist()
        }
        
        # Add labels if available
        if labels is not None:
            cluster_data['labels'] = [labels[i] for i in sampled_indices]
        
        clusters[cluster_id] = cluster_data
        
        print(f"  Cluster {cluster_id}: {cluster_size} samples, sampled {n_sample}")
    
    return clusters, cluster_labels, kmeans


def save_clustered_data(clusters, cluster_labels, sentences, labels=None, output_dir="sampled_data", filename="sampled_sentences"):
    """Save clustered and sampled data to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all sampled sentences to CSV
    all_sampled_sentences = []
    all_cluster_ids = []
    all_indices = []
    all_labels = []
    
    for cluster_id, cluster_data in clusters.items():
        all_sampled_sentences.extend(cluster_data['sentences'])
        all_cluster_ids.extend([cluster_id] * len(cluster_data['sentences']))
        all_indices.extend(cluster_data['indices'])
        if labels is not None:
            all_labels.extend(cluster_data.get('labels', []))
    
    df_sampled = pd.DataFrame({
        'cluster_id': all_cluster_ids,
        'original_index': all_indices,
        'sentence': all_sampled_sentences
    })
    
    if labels is not None:
        df_sampled['label'] = all_labels
    
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    df_sampled.to_csv(csv_path, index=False)
    print(f"✓ Saved sampled sentences to: {csv_path}")
    
    # Save cluster assignments for all data
    df_all = pd.DataFrame({
        'sentence': sentences,
        'cluster_id': cluster_labels
    })
    if labels is not None:
        df_all['label'] = labels
    
    csv_all_path = os.path.join(output_dir, f"{filename}_all_clusters.csv")
    df_all.to_csv(csv_all_path, index=False)
    print(f"✓ Saved all cluster assignments to: {csv_all_path}")
    
    # Save cluster metadata as JSON
    metadata = {
        'n_clusters': len(clusters),
        'total_samples': len(sentences),
        'sampled_samples': len(all_sampled_sentences),
        'clusters': {
            str(k): {
                'cluster_id': v['cluster_id'],
                'cluster_size': v['cluster_size'],
                'n_sampled': v['n_sampled']
            }
            for k, v in clusters.items()
        }
    }
    
    json_path = os.path.join(output_dir, f"{filename}_metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {json_path}")
    
    return csv_path, csv_all_path, json_path


def main():
    parser = argparse.ArgumentParser(
        description="K-means clustering on ORIGINAL data z_c embeddings and sample sentences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Option 1: Use pre-computed embeddings (from original data)
  python cluster_and_sample.py \\
      --embeddings embeddings/original_z_c.npy \\
      --sentences embeddings/original.csv \\
      --output-dir sampled_data \\
      --n-examples 10

  # Option 2: Auto-embed original dataset
  python cluster_and_sample.py \\
      --model-path models/german-credit-data/best_model.pth \\
      --dataset german-credit-data \\
      --output-dir sampled_data \\
      --n-examples 10

Note: This script is for clustering ORIGINAL training data.
      Use embed_data.py for embedding SYNTHETIC data after generation.
        """
    )
    
    # Two modes: pre-computed embeddings OR auto-embed from dataset
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--embeddings",
        type=str,
        help="Path to z_c embeddings numpy file (.npy) - use with --sentences"
    )
    group.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model (.pth) - use with --dataset to auto-embed original data"
    )
    
    parser.add_argument(
        "--sentences",
        type=str,
        default=None,
        help="Path to sentences CSV file (required if using --embeddings)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g., 'german-credit-data') - required if using --model-path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sampled_data",
        help="Output directory for sampled data (default: sampled_data/)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="sampled_sentences",
        help="Base filename for output files (default: sampled_sentences)"
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Number of examples to sample from each cluster (default: 10)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters (if not specified, will find optimal K)"
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=20,
        help="Maximum K to test when finding optimal K (default: 20)"
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
        help="Minimum K to test when finding optimal K (default: 2)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="silhouette",
        choices=['silhouette', 'elbow', 'davies_bouldin'],
        help="Method to find optimal K (default: silhouette)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("K-MEANS CLUSTERING AND SAMPLING (ORIGINAL DATA)")
    print("="*70)
    print("NOTE: This script clusters ORIGINAL training data.")
    print("      Use embed_data.py for SYNTHETIC data after generation.")
    print("="*70)
    
    # Two modes: pre-computed embeddings OR auto-embed
    if args.embeddings:
        # Mode 1: Use pre-computed embeddings
        if not args.sentences:
            parser.error("--sentences is required when using --embeddings")
        
        print("\n[Mode 1] Using pre-computed embeddings...")
        z_c = load_embeddings(args.embeddings)
        sentences, labels, df = load_sentences(args.sentences)
        
    elif args.model_path:
        # Mode 2: Auto-embed from dataset
        if not args.dataset:
            parser.error("--dataset is required when using --model-path")
        
        if not EMBED_AVAILABLE:
            raise ImportError("Cannot import embed_data. Please install dependencies or use --embeddings mode.")
        if not DATA_LOADER_AVAILABLE:
            raise ImportError("Cannot import data_loader. Please check data_loader.py exists.")
        
        print("\n[Mode 2] Auto-embedding original dataset...")
        print(f"  Model: {args.model_path}")
        print(f"  Dataset: {args.dataset}")
        
        # Load model
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_info = load_model(args.model_path, device)
        
        # Load original dataset
        print(f"\nLoading original dataset: {args.dataset}")
        if args.dataset == "german-credit-data":
            sentences, labels, data, _ = load_german_credit_data_balanced(n_samples=None)
        else:
            sentences, labels, data, _ = load_dataset_generic(args.dataset)
        
        print(f"✓ Loaded {len(sentences)} original samples")
        
        # Embed original data
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        z_c, z_d, attn_c, attn_d = embed_data(
            model=model,
            sentences=sentences,
            tokenizer=tokenizer,
            max_len=50,
            batch_size=32,
            use_cache=True,
            dataset_name=f"{args.dataset}_original"
        )
        
        print(f"✓ Embedded {len(sentences)} original samples")
        
        # Create DataFrame for compatibility
        labels_list = labels if labels is not None else [0] * len(sentences)
        df = pd.DataFrame({
            'sentence': sentences,
            'label': labels_list
        })
        labels = labels_list  # Set labels variable for later use
        
    else:
        parser.error("Must specify either --embeddings or --model-path")
    
    # Verify dimensions match
    if len(sentences) != z_c.shape[0]:
        raise ValueError(f"Mismatch: {len(sentences)} sentences but {z_c.shape[0]} embeddings")
    
    # Find optimal K if not specified
    if args.k is None:
        optimal_k, scores = find_optimal_k(
            z_c, 
            max_k=args.max_k, 
            min_k=args.min_k,
            method=args.method
        )
        k = optimal_k
        
        # Plot K selection metrics
        plot_k_selection(scores, args.output_dir)
        
        # Save scores
        scores_path = os.path.join(args.output_dir, f"{args.output_name}_k_scores.json")
        with open(scores_path, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"✓ Saved K selection scores to: {scores_path}")
    else:
        k = args.k
        print(f"\nUsing specified K={k}")
    
    # Perform clustering and sampling
    clusters, cluster_labels, kmeans = cluster_and_sample(
        z_c=z_c,
        sentences=sentences,
        labels=labels,
        k=k,
        n_examples=args.n_examples,
        random_state=args.random_state
    )
    
    # Save results
    csv_path, csv_all_path, json_path = save_clustered_data(
        clusters=clusters,
        cluster_labels=cluster_labels,
        sentences=sentences,
        labels=labels,
        output_dir=args.output_dir,
        filename=args.output_name
    )
    
    print(f"\n{'='*70}")
    print("CLUSTERING AND SAMPLING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput files saved to: {args.output_dir}/")
    print(f"  - {args.output_name}.csv (Sampled sentences from each cluster)")
    print(f"  - {args.output_name}_all_clusters.csv (All sentences with cluster assignments)")
    print(f"  - {args.output_name}_metadata.json (Cluster metadata)")
    if args.k is None:
        print(f"  - {args.output_name}_k_scores.json (K selection scores)")
        print(f"  - k_selection_metrics.png (K selection plots)")
    print(f"\nCluster summary:")
    for cluster_id, cluster_data in sorted(clusters.items()):
        print(f"  Cluster {cluster_id}: {cluster_data['cluster_size']} samples, {cluster_data['n_sampled']} sampled")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
