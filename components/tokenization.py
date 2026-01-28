"""
Tokenization and caching utilities for BERT tokenization.
"""

import os
import hashlib
import pickle
import torch
from transformers import AutoTokenizer


def get_cache_path(sentences, dataset_name, cache_dir="bert_cache"):
    """
    Generate cache file path based on dataset content.
    Cache is used to speed up tokenization - only token IDs are cached, not embeddings.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Create hash from sentences to ensure cache validity
    content = "".join(sentences)
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"{dataset_name}_{content_hash}.pkl")
    return cache_file


def compute_and_cache_tokenized_data(sentences, dataset_name, max_len=256, force_recompute=False):
    """
    Pre-compute tokenized data for all sentences and cache them for faster training.
    This significantly speeds up training by avoiding repeated tokenization.
    
    Caches:
        - input_ids: Token IDs [n_samples, seq_len]
        - attention_mask: Attention masks [n_samples, seq_len]
    
    Returns:
        cached_data: dict with 'input_ids', 'attention_mask'
    """
    cache_file = get_cache_path(sentences, dataset_name)
    
    # Check if cache exists
    if os.path.exists(cache_file) and not force_recompute:
        print(f"✓ Loading tokenized data from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            # Verify cache has required keys
            if 'input_ids' in cached_data and 'attention_mask' in cached_data:
                print(f"✓ Loaded {len(cached_data['input_ids'])} cached tokenizations")
                return cached_data
            else:
                print(f"⚠ Cache file exists but missing required keys. Recomputing...")
        except Exception as e:
            print(f"⚠ Error loading cache: {e}. Recomputing...")
    
    # Tokenize sentences (only if cache doesn't exist or force_recompute=True)
    print(f"Tokenizing {len(sentences)} sentences...")
    print("This may take a few minutes. Results will be cached for faster future runs.")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Use BERT tokenizer for vocab
    
    input_ids_list = []
    attention_mask_list = []
    
    batch_size = 32  # Process in batches for efficiency
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        tokens = tokenizer(
            batch_sentences, 
            padding="max_length", 
            truncation=True,
            max_length=max_len, 
            return_tensors="pt"
        )
        input_ids_list.append(tokens.input_ids.cpu())
        attention_mask_list.append(tokens.attention_mask.cpu())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(sentences))}/{len(sentences)} sentences...")
    
    # Concatenate all batches
    all_input_ids = torch.cat(input_ids_list, dim=0)
    all_attention_mask = torch.cat(attention_mask_list, dim=0)
    
    cached_data = {
        'input_ids': all_input_ids,    # [n_samples, seq_len]
        'attention_mask': all_attention_mask,  # [n_samples, seq_len]
    }
    
    # Save to cache for faster future runs
    print(f"Saving tokenized data to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"✓ Successfully cached {len(all_input_ids)} tokenizations")
        print(f"  Cache file: {cache_file}")
        print(f"  Next run will use cached data (much faster!)")
    except Exception as e:
        print(f"⚠ Warning: Could not save cache: {e}")
        print(f"  Training will continue but tokenization will be slower")
    
    return cached_data
