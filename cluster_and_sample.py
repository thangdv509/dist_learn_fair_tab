#!/usr/bin/env python3
"""
Script to perform K-means clustering on z_d embeddings from processed_data, 
find optimal K, and sample representative sentences from each cluster.

This script:
1. Reads data from processed_data/{dataset_name}.csv
2. Loads the corresponding best_model from models/{dataset_name}/
3. Generates z_d embeddings using the model
4. Clusters based on z_d
5. Finds optimal number of clusters
6. Samples n samples from each cluster

Usage:
    python cluster_and_sample.py \
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


def find_best_model_path(dataset_name, models_dir="models"):
    """
    Find the best_model file for a given dataset.
    Looks for best_model_*.pth in models/{dataset_name}/ directory.
    Falls back to final_model_*.pth if best_model is not found or corrupted.
    
    Args:
        dataset_name: Name of dataset (e.g., 'german-credit-data')
        models_dir: Directory containing model folders (default: 'models')
    
    Returns:
        Path to best_model file, or None if not found
    """
    import glob
    
    model_dir = os.path.join(models_dir, dataset_name)
    if not os.path.exists(model_dir):
        return None
    
    # First try to find best_model files
    pattern = os.path.join(model_dir, "best_model_*.pth")
    matches = glob.glob(pattern)
    
    if matches:
        # Try all best_model files, return the first one that is not corrupted
        best_model_paths = sorted(matches, key=os.path.getmtime, reverse=True)  # Most recent first
        
        for best_model_path in best_model_paths:
            # Verify the file is not corrupted by checking if it can be loaded
            try:
                import torch
                torch.load(best_model_path, map_location='cpu', weights_only=False)
                print(f"✓ Found valid best_model: {best_model_path}")
                return best_model_path
            except (RuntimeError, Exception) as e:
                print(f"⚠ Warning: best_model file appears corrupted: {best_model_path}")
                print(f"  Error: {e}")
                continue  # Try next best_model file
    
    # Fallback to final_model if best_model is not found or all are corrupted
    pattern = os.path.join(model_dir, "final_model_*.pth")
    matches = glob.glob(pattern)
    
    if matches:
        # Try all final_model files, return the first one that is not corrupted
        final_model_paths = sorted(matches, key=os.path.getmtime, reverse=True)  # Most recent first
        
        for final_model_path in final_model_paths:
            try:
                import torch
                torch.load(final_model_path, map_location='cpu', weights_only=False)
                print(f"✓ Found valid final_model: {final_model_path}")
                return final_model_path
            except (RuntimeError, Exception) as e:
                print(f"⚠ Warning: final_model file appears corrupted: {final_model_path}")
                print(f"  Error: {e}")
                continue  # Try next final_model file
    
    return None


def load_embeddings(embeddings_path):
    """Load z_d embeddings from numpy file."""
    print(f"Loading embeddings from: {embeddings_path}")
    z_d = np.load(embeddings_path)
    print(f"✓ Loaded embeddings: shape {z_d.shape}")
    return z_d


def load_processed_data(dataset_name, processed_data_dir="processed_data"):
    """
    Load processed data from processed_data/{dataset_name}.csv
    
    Args:
        dataset_name: Name of dataset (e.g., 'german-credit-data')
        processed_data_dir: Directory containing processed data (default: 'processed_data')
    
    Returns:
        sentences: List of sentences
        labels: List of labels (or None if not available)
        df: DataFrame with the data
    """
    csv_path = os.path.join(processed_data_dir, f"{dataset_name}.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Processed data file not found: {csv_path}\n"
            f"Please ensure the file exists in {processed_data_dir}/ directory."
        )
    
    print(f"Loading processed data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Try different column names for sentences
    if 'sentence' in df.columns:
        sentences = df['sentence'].tolist()
    elif 'text' in df.columns:
        sentences = df['text'].tolist()
    elif len(df.columns) == 1:
        sentences = df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Could not find 'sentence' or 'text' column in CSV. Available columns: {df.columns.tolist()}")
    
    # Load labels if available
    labels = df['label'].tolist() if 'label' in df.columns else None
    
    print(f"✓ Loaded {len(sentences)} sentences")
    if labels is not None:
        print(f"✓ Loaded {len(labels)} labels")
    
    return sentences, labels, df


def load_sentences(sentences_path):
    """Load sentences from CSV file (legacy function for backward compatibility)."""
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


def find_optimal_k(z_d, max_k=20, min_k=2, method='silhouette'):
    """
    Find optimal number of clusters using elbow method or silhouette score.
    
    Args:
        z_d: Embeddings array [n_samples, latent_dim] (z_d embeddings for clustering)
        max_k: Maximum K to test
        min_k: Minimum K to test
        method: 'silhouette', 'elbow', or 'davies_bouldin'
    
    Returns:
        optimal_k: Optimal number of clusters
        scores: Dictionary with scores for each K
    """
    print(f"\nFinding optimal K (testing K={min_k} to {max_k})...")
    
    n_samples = z_d.shape[0]
    max_k = min(max_k, n_samples - 1)  # Can't have more clusters than samples
    
    k_range = range(min_k, max_k + 1)
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    
    for k in k_range:
        print(f"  Testing K={k}...", end=' ', flush=True)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(z_d)
        
        inertia = kmeans.inertia_
        inertias.append(inertia)
        
        # Silhouette score (higher is better, range: -1 to 1)
        if k > 1:
            sil_score = silhouette_score(z_d, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(-1)
        
        # Davies-Bouldin score (lower is better)
        if k > 1:
            db_score = davies_bouldin_score(z_d, labels)
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


def cluster_and_sample(z_d, sentences, labels=None, k=None, n_examples=10, random_state=42):
    """
    Perform K-means clustering on z_d embeddings and sample examples from each cluster.
    
    Args:
        z_d: Embeddings [n_samples, latent_dim] (z_d embeddings for clustering)
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
    
    print(f"\nPerforming K-means clustering with K={k} on z_d embeddings...")
    np.random.seed(random_state)
    
    # K-means clustering on z_d
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(z_d)
    
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
        sampled_z_d = z_d[sampled_indices]
        
        cluster_data = {
            'cluster_id': int(cluster_id),
            'cluster_size': int(cluster_size),
            'n_sampled': int(n_sample),
            'indices': sampled_indices.tolist(),
            'sentences': sampled_sentences,
            'z_d': sampled_z_d.tolist(),
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
        description="K-means clustering on z_d embeddings from processed_data and sample sentences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Main usage: Read from processed_data and use best_model
  python cluster_and_sample.py \\
      --dataset german-credit-data \\
      --output-dir sampled_data \\
      --n-examples 10

  # With custom model path
  python cluster_and_sample.py \\
      --dataset german-credit-data \\
      --model-path models/german-credit-data/best_model_20260128_143044.pth \\
      --output-dir sampled_data \\
      --n-examples 10

Note: This script:
  1. Reads data from processed_data/{dataset_name}.csv
  2. Loads best_model from models/{dataset_name}/
  3. Generates z_d embeddings
  4. Clusters based on z_d
  5. Finds optimal K and samples n examples from each cluster
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'german-credit-data') - will read from processed_data/{dataset}.csv"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (.pth). If not specified, will auto-find best_model in models/{dataset}/"
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
    print("K-MEANS CLUSTERING AND SAMPLING (z_d embeddings)")
    print("="*70)
    print("This script:")
    print("  1. Reads data from processed_data/{dataset}.csv")
    print("  2. Loads best_model from models/{dataset}/")
    print("  3. Generates z_d embeddings")
    print("  4. Clusters based on z_d")
    print("  5. Finds optimal K and samples n examples from each cluster")
    print("="*70)
    
    if not EMBED_AVAILABLE:
        raise ImportError("Cannot import embed_data. Please install dependencies.")
    
    # Find or use provided model path
    if args.model_path is None:
        print(f"\nAuto-finding best_model for dataset: {args.dataset}")
        model_path = find_best_model_path(args.dataset)
        if model_path is None:
            raise FileNotFoundError(
                f"Could not find a valid model for dataset '{args.dataset}'.\n"
                f"Looked in: models/{args.dataset}/\n"
                f"Tried: best_model_*.pth and final_model_*.pth\n"
                f"Please specify --model-path or ensure a valid model exists.\n"
                f"If models are corrupted, you may need to retrain."
            )
        print(f"✓ Found model: {model_path}")
    else:
        model_path = args.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Verify the model file is not corrupted
        try:
            import torch
            torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"✓ Model file verified: {model_path}")
        except (RuntimeError, Exception) as e:
            if "failed finding central directory" in str(e) or "zip archive" in str(e):
                # Try to find alternative model files in the same directory
                model_dir = os.path.dirname(model_path)
                dataset_name = os.path.basename(model_dir)
                print(f"\n⚠ Provided model file is corrupted: {model_path}")
                print(f"  Error: {e}")
                print(f"  Attempting to find alternative model files...")
                
                # Try to find best_model or final_model in the same directory
                import glob
                alt_patterns = [
                    os.path.join(model_dir, "best_model_*.pth"),
                    os.path.join(model_dir, "final_model_*.pth")
                ]
                
                for pattern in alt_patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        # Try files in reverse chronological order
                        for alt_path in sorted(matches, key=os.path.getmtime, reverse=True):
                            try:
                                torch.load(alt_path, map_location='cpu', weights_only=False)
                                print(f"✓ Found alternative valid model: {alt_path}")
                                model_path = alt_path
                                break
                            except Exception:
                                continue
                        if model_path != args.model_path:  # Found alternative
                            break
                
                if model_path == args.model_path:  # No alternative found
                    raise RuntimeError(
                        f"Model file appears to be corrupted: {model_path}\n"
                        f"Error: {e}\n"
                        f"Please use a different model file or retrain the model."
                    ) from e
            else:
                raise
        
        print(f"\nUsing model: {model_path}")
    
    # Load processed data
    print(f"\nLoading processed data for dataset: {args.dataset}")
    sentences, labels, df = load_processed_data(args.dataset)
    
    # Load model
    import torch
    from embed_data import load_model
    from components.utils import get_device
    device = get_device()
    print(f"\nLoading model on device: {device}")
    model, model_info = load_model(model_path, device)
    
    # Generate z_d embeddings
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    print(f"\nGenerating z_d embeddings for {len(sentences)} sentences...")
    z_c, z_d, attn_c, attn_d = embed_data(
        model=model,
        sentences=sentences,
        tokenizer=tokenizer,
        max_len=256,
        batch_size=32,
        use_cache=True,
        dataset_name=f"{args.dataset}_clustering"
    )
    
    print(f"✓ Generated z_d embeddings: shape {z_d.shape}")
    
    # Verify dimensions match
    if len(sentences) != z_d.shape[0]:
        raise ValueError(f"Mismatch: {len(sentences)} sentences but {z_d.shape[0]} embeddings")
    
    # Find optimal K if not specified
    if args.k is None:
        optimal_k, scores = find_optimal_k(
            z_d, 
            max_k=args.max_k, 
            min_k=args.min_k,
            method=args.method
        )
        k = optimal_k
        
        # Plot K selection metrics
        output_dataset_dir = os.path.join(args.output_dir, args.dataset)
        os.makedirs(output_dataset_dir, exist_ok=True)
        plot_k_selection(scores, output_dataset_dir)
        
        # Save scores
        scores_path = os.path.join(output_dataset_dir, f"{args.output_name}_k_scores.json")
        with open(scores_path, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"✓ Saved K selection scores to: {scores_path}")
    else:
        k = args.k
        print(f"\nUsing specified K={k}")
    
    # Perform clustering and sampling on z_d
    clusters, cluster_labels, kmeans = cluster_and_sample(
        z_d=z_d,
        sentences=sentences,
        labels=labels,
        k=k,
        n_examples=args.n_examples,
        random_state=args.random_state
    )
    
    # Save results
    output_dataset_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    csv_path, csv_all_path, json_path = save_clustered_data(
        clusters=clusters,
        cluster_labels=cluster_labels,
        sentences=sentences,
        labels=labels,
        output_dir=output_dataset_dir,
        filename=args.output_name
    )
    
    print(f"\n{'='*70}")
    print("CLUSTERING AND SAMPLING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput files saved to: {output_dataset_dir}/")
    print(f"  - {args.output_name}.csv (Sampled sentences from each cluster)")
    print(f"  - {args.output_name}_all_clusters.csv (All sentences with cluster assignments)")
    print(f"  - {args.output_name}_metadata.json (Cluster metadata)")
    if args.k is None:
        print(f"  - {args.output_name}_k_scores.json (K selection scores)")
        print(f"  - k_selection_metrics.png (K selection plots)")
    print(f"\nCluster summary (clustered on z_d embeddings):")
    for cluster_id, cluster_data in sorted(clusters.items()):
        print(f"  Cluster {cluster_id}: {cluster_data['cluster_size']} samples, {cluster_data['n_sampled']} sampled")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
