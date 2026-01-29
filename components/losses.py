"""
Loss functions for fairness learning with disentangled representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kl_loss(mu, logvar):
    """
    Compute KL divergence loss: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    
    This regularizes the latent distribution to follow standard normal N(0,I), which:
    1. Ensures smooth latent space (good interpolation between points)
    2. Enables proper sampling (can sample z ~ N(0,I) and decode to generate realistic data)
    3. Prevents sampling outside the learned manifold (normalized distribution)
    
    Normalized by batch size to match scale of other losses (CE, MSE, etc.)
    
    Args:
        mu: Mean of latent distribution [batch_size, latent_dim]
        logvar: Log variance of latent distribution [batch_size, latent_dim]
    
    Returns:
        KL divergence loss (scalar tensor)
    """
    # KL divergence per sample: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # Formula: KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # Shape: [batch_size, latent_dim] -> [batch_size] -> scalar (mean)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(kl_per_sample)  # Normalize by batch size


def compute_reconstruction_loss(original_h, reconstructed_h):
    return nn.functional.mse_loss(reconstructed_h, original_h)


def compute_hsic(x, y, sigma=None):
    """
    Compute Hilbert-Schmidt Independence Criterion (HSIC) between x and y.
    
    HSIC measures statistical independence: HSIC = 0 if x and y are independent.
    We want to minimize HSIC to make variables independent.
    
    Args:
        x: [batch_size, dim_x]
        y: [batch_size, dim_y]
        sigma: Bandwidth for RBF kernel (if None, use median heuristic)
    
    Returns:
        HSIC value (scalar tensor)
    """
    batch_size = x.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=x.device)
    
    # Compute pairwise distances for median heuristic
    def median_heuristic(data):
        """Compute median of pairwise distances for RBF kernel bandwidth"""
        with torch.no_grad():
            # Compute pairwise squared distances
            data_norm = torch.sum(data ** 2, dim=1, keepdim=True)  # [batch_size, 1]
            dists = data_norm - 2 * torch.matmul(data, data.t()) + data_norm.t()  # [batch_size, batch_size]
            dists = dists.clamp(min=0)  # Ensure non-negative
            # Get upper triangular part (excluding diagonal)
            triu_indices = torch.triu_indices(batch_size, batch_size, offset=1, device=data.device)
            triu_dists = dists[triu_indices[0], triu_indices[1]]
            if len(triu_dists) > 0:
                median_dist = torch.median(triu_dists)
                # Use median as bandwidth (add small epsilon for stability)
                return median_dist.clamp(min=1e-5)
            else:
                return torch.tensor(1.0, device=data.device)
    
    # Compute RBF kernel matrices
    def rbf_kernel(data, sigma_val):
        """Compute RBF kernel matrix"""
        # Compute pairwise squared distances
        data_norm = torch.sum(data ** 2, dim=1, keepdim=True)  # [batch_size, 1]
        dists = data_norm - 2 * torch.matmul(data, data.t()) + data_norm.t()  # [batch_size, batch_size]
        dists = dists.clamp(min=0)  # Ensure non-negative
        # RBF kernel: K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2*sigma^2))
        K = torch.exp(-dists / (2 * sigma_val ** 2))
        return K
    
    # Get bandwidths
    if sigma is None:
        sigma_x = median_heuristic(x)
        sigma_y = median_heuristic(y)
    else:
        sigma_x = sigma_y = sigma
    
    # Compute kernel matrices
    K_x = rbf_kernel(x, sigma_x)  # [batch_size, batch_size]
    K_y = rbf_kernel(y, sigma_y)  # [batch_size, batch_size]
    
    # Center the kernel matrices: H = I - 1/n * 1*1^T
    H = torch.eye(batch_size, device=x.device) - 1.0 / batch_size
    
    # HSIC = trace(K_x @ H @ K_y @ H) / (batch_size - 1)^2
    # More numerically stable: compute trace directly
    HK_x = torch.matmul(H, K_x)
    HK_y = torch.matmul(H, K_y)
    HK_xH = torch.matmul(HK_x, H)
    
    # Trace of HK_xH @ K_y = sum of diagonal of (HK_xH @ K_y)
    hsic = torch.trace(torch.matmul(HK_xH, K_y)) / ((batch_size - 1) ** 2)
    
    return hsic


def compute_orthogonality_loss(z_c, z_d):
    """
    DEPRECATED: Use compute_hsic instead for better independence measure.
    Kept for backward compatibility.
    """
    # Use HSIC instead of simple orthogonality
    return compute_hsic(z_c, z_d)


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


def compute_random_guess_penalty(logits, target_acc=0.5, temperature=1.0):
    """
    Penalty loss to encourage predictions close to random guessing (50% accuracy).
    
    For z_d, we want the adversarial classifier to predict randomly (uniform distribution).
    This penalty encourages the logits to be close to uniform (equal probability for all classes).
    
    Args:
        logits: Classifier logits [batch_size, num_classes]
        target_acc: Target accuracy (0.5 for binary classification = random)
        temperature: Temperature for softmax (higher = more uniform)
    
    Returns:
        Penalty value (higher when predictions are far from uniform)
    """
    probs = torch.softmax(logits / temperature, dim=1)  # [batch_size, num_classes]
    
    # Target: uniform distribution (equal probability for all classes)
    num_classes = probs.size(1)
    target_uniform = torch.ones_like(probs) / num_classes  # [batch_size, num_classes]
    
    # KL divergence from uniform (higher when far from uniform)
    kl_div = torch.sum(probs * torch.log(probs / target_uniform + 1e-10), dim=1)
    penalty = torch.mean(kl_div)
    
    return penalty


def compute_variance_constraint(z, min_variance=0.1):
    """
    Constraint to prevent z from collapsing to zero/noise.
    
    Encourages z to have sufficient variance (energy) to carry information.
    This prevents z_d from collapsing when only minimizing HSIC(z_d, y) and HSIC(z_c, z_d).
    
    Args:
        z: Latent representation [batch_size, latent_dim]
        min_variance: Minimum expected variance per dimension
    
    Returns:
        Penalty (0 if variance >= min_variance, positive otherwise)
    """
    # Compute variance per dimension
    z_var = torch.var(z, dim=0)  # [latent_dim]
    # Penalty: encourage variance to be at least min_variance
    penalty = torch.clamp(min_variance - z_var, min=0.0).mean()
    return penalty


def compute_energy_constraint(reconstructed_d, min_energy=0.1):
    """
    Constraint to prevent D_d(z_d) from collapsing to zero.
    
    Encourages D_d(z_d) to have sufficient energy (L2 norm) to contribute to reconstruction.
    This ensures z_d carries meaningful information about the input.
    
    Args:
        reconstructed_d: D_d(z_d) output [batch_size, embed_dim]
        min_energy: Minimum expected L2 norm per sample
    
    Returns:
        Penalty (0 if energy >= min_energy, positive otherwise)
    """
    # Compute L2 norm per sample
    energy = torch.norm(reconstructed_d, p=2, dim=1)  # [batch_size]
    # Penalty: encourage energy to be at least min_energy
    penalty = torch.clamp(min_energy - energy, min=0.0).mean()
    return penalty
