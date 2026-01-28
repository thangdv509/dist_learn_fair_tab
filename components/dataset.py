"""
Dataset class for text data with BERT tokenization.
"""

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=256, cached_data=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cached_data = cached_data
        self.use_cache = cached_data is not None

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.use_cache:
            # Use cached tokenized data
            return {
                "input_ids": self.cached_data['input_ids'][idx],
                "attention_mask": self.cached_data['attention_mask'][idx],
                "label": torch.tensor(label, dtype=torch.long),
                "text": self.sentences[idx]
            }
        else:
            # Compute on-the-fly
            text = self.sentences[idx]
            tokens = self.tokenizer(
                text, padding="max_length", truncation=True,
                max_length=self.max_len, return_tensors="pt"
            )
            return {
                "input_ids": tokens.input_ids.squeeze(0),
                "attention_mask": tokens.attention_mask.squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
                "text": text
            }
