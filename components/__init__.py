"""
Components module for fairness learning with disentangled representations.
"""

from .models import CD_Model, TaskClassifier, AdversarialClassifier, ContentDecoder, DemographicDecoder, ContentHead, DemographicHead, GRL
from .utils import device, get_device
from .dataset import TextDataset
from .losses import (
    compute_kl_loss,
    compute_reconstruction_loss,
    compute_hsic,
    compute_orthogonality_loss,
    compute_separation_loss,
    compute_random_guess_penalty,
    compute_variance_constraint,
    compute_energy_constraint
)
from .tokenization import compute_and_cache_tokenized_data, get_cache_path
from .training import train_cd_model
from .utils import save_encodings

__all__ = [
    'CD_Model',
    'TaskClassifier',
    'AdversarialClassifier',
    'ContentDecoder',
    'DemographicDecoder',
    'ContentHead',
    'DemographicHead',
    'GRL',
    'device',
    'get_device',
    'TextDataset',
    'compute_kl_loss',
    'compute_reconstruction_loss',
    'compute_hsic',
    'compute_orthogonality_loss',
    'compute_separation_loss',
    'compute_random_guess_penalty',
    'compute_variance_constraint',
    'compute_energy_constraint',
    'compute_and_cache_tokenized_data',
    'get_cache_path',
    'train_cd_model',
    'save_encodings',
]
