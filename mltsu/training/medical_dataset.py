"""
Medical Dataset Processor for TinyBioBERT.
Handles medical NER datasets and tokenization for BioBERT training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import os
from collections import defaultdict
import random


@dataclass
class MedicalNERLabel:
    """Medical NER label configuration."""
    # Standard BIO tags for disease entities
    LABELS = [
        'O',           # Outside any entity
        'B-Disease',   # Beginning of disease
        'I-Disease',   # Inside disease
        'B-Chemical',  # Beginning of chemical/drug
        'I-Chemical',  # Inside chemical/drug
        'B-Gene',      # Beginning of gene
        'I-Gene',      # Inside gene
        'B-Species',   # Beginning of species
        'I-Species',   # Inside species
    ]

    @classmethod
    def get_label_map(cls) -> Dict[str, int]:
        """Get label to ID mapping."""
        return {label: i for i, label in enumerate(cls.LABELS)}

    @classmethod
    def get_id_map(cls) -> Dict[int, str]:
        """Get ID to label mapping."""
        return {i: label for i, label in enumerate(cls.LABELS)}

    @classmethod
    def num_labels(cls) -> int:
        """Get number of labels."""
        return len(cls.LABELS)


class MedicalTokenizer:
    """
    Simple tokenizer for medical text.
    In production, you'd use BioBERT tokenizer from HuggingFace.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        max_length: int = 128,
        pad_token: str = '[PAD]',
        cls_token: str = '[CLS]',
        sep_token: str = '[SEP]',
        unk_token: str = '[UNK]',
        mask_token: str = '[MASK]',
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Special tokens
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.unk_token = unk_token
        self.mask_token = mask_token

        # Build vocabulary (simplified for demo)
        self.vocab = self._build_vocabulary()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

        # Special token IDs
        self.pad_token_id = self.token_to_id[pad_token]
        self.cls_token_id = self.token_to_id[cls_token]
        self.sep_token_id = self.token_to_id[sep_token]
        self.unk_token_id = self.token_to_id[unk_token]
        self.mask_token_id = self.token_to_id[mask_token]

    def _build_vocabulary(self) -> List[str]:
        """Build a simple medical vocabulary."""
        # Special tokens
        special_tokens = [
            self.pad_token, self.cls_token, self.sep_token,
            self.unk_token, self.mask_token
        ]

        # Common medical terms (simplified)
        medical_terms = [
            # Diseases
            'cancer', 'diabetes', 'hypertension', 'pneumonia', 'influenza',
            'covid', 'tumor', 'carcinoma', 'syndrome', 'disease',
            'disorder', 'infection', 'inflammation', 'injury', 'trauma',

            # Chemicals/Drugs
            'aspirin', 'insulin', 'metformin', 'antibiotics', 'vaccine',
            'drug', 'medication', 'treatment', 'therapy', 'dose',

            # Genes
            'brca1', 'brca2', 'p53', 'egfr', 'kras', 'gene', 'protein',
            'mutation', 'expression', 'pathway',

            # Anatomy
            'heart', 'lung', 'liver', 'kidney', 'brain', 'blood',
            'cell', 'tissue', 'organ', 'system',

            # Clinical terms
            'patient', 'diagnosis', 'symptom', 'treatment', 'clinical',
            'study', 'trial', 'outcome', 'response', 'survival',

            # Common words
            'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'of',
            'to', 'for', 'with', 'and', 'or', 'not', 'that', 'this',
        ]

        # Generate additional tokens
        vocab = special_tokens + medical_terms

        # Add numbered tokens to reach vocab_size
        while len(vocab) < self.vocab_size:
            vocab.append(f'token_{len(vocab)}')

        return vocab[:self.vocab_size]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        # Simple whitespace tokenization (in production, use BioBERT tokenizer)
        tokens = text.lower().split()
        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add [CLS] and [SEP]
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            Dictionary with input_ids and attention_mask
        """
        max_length = max_length or self.max_length
        tokens = self.tokenize(text)

        # Add special tokens
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]

        # Truncate
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.sep_token]

        # Convert to IDs
        input_ids = [
            self.token_to_id.get(token, self.unk_token_id)
            for token in tokens
        ]

        # Pad
        if padding:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length
            attention_mask = [1] * len(tokens) + [0] * padding_length
        else:
            attention_mask = [1] * len(input_ids)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
        }

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = [
            self.id_to_token.get(id, self.unk_token)
            for id in token_ids
            if id != self.pad_token_id
        ]

        # Remove special tokens
        tokens = [
            t for t in tokens
            if t not in [self.cls_token, self.sep_token, self.pad_token]
        ]

        return ' '.join(tokens)


class MedicalNERDataset(Dataset):
    """
    Medical NER dataset for training TinyBioBERT.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: MedicalTokenizer,
        max_length: int = 128,
        label_map: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize medical NER dataset.

        Args:
            data: List of examples with 'text' and 'labels' fields
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            label_map: Mapping from label strings to IDs
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or MedicalNERLabel.get_label_map()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.data[idx]
        text = example['text']
        labels = example.get('labels', [])

        # Tokenize text
        encoded = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )

        # Process labels (simplified - assumes word-level alignment)
        if labels:
            # Convert string labels to IDs
            label_ids = [self.label_map.get(label, 0) for label in labels]

            # Add label for [CLS] and [SEP]
            label_ids = [0] + label_ids + [0]

            # Truncate/pad labels to match input length
            if len(label_ids) > self.max_length:
                label_ids = label_ids[:self.max_length]
            else:
                label_ids = label_ids + [0] * (self.max_length - len(label_ids))

            encoded['labels'] = torch.tensor(label_ids)
        else:
            # No labels (inference mode)
            encoded['labels'] = torch.zeros(self.max_length, dtype=torch.long)

        return encoded


def create_medical_datasets() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create synthetic medical NER datasets for demonstration.
    In production, load from NCBI-disease or BC5CDR datasets.
    """
    # Synthetic training examples
    train_data = [
        {
            'text': 'Patient diagnosed with type 2 diabetes and hypertension',
            'labels': ['O', 'O', 'O', 'B-Disease', 'I-Disease', 'I-Disease', 'O', 'B-Disease']
        },
        {
            'text': 'Treatment with metformin for diabetes management',
            'labels': ['O', 'O', 'B-Chemical', 'O', 'B-Disease', 'O']
        },
        {
            'text': 'BRCA1 mutation associated with breast cancer risk',
            'labels': ['B-Gene', 'O', 'O', 'O', 'B-Disease', 'I-Disease', 'O']
        },
        {
            'text': 'COVID-19 pneumonia treated with remdesivir',
            'labels': ['B-Disease', 'B-Disease', 'O', 'O', 'B-Chemical']
        },
        {
            'text': 'Elevated p53 expression in lung carcinoma cells',
            'labels': ['O', 'B-Gene', 'O', 'O', 'B-Disease', 'I-Disease', 'O']
        },
        # Add more examples...
    ] * 20  # Replicate for larger dataset

    # Validation data
    val_data = [
        {
            'text': 'Insulin therapy for type 1 diabetes patients',
            'labels': ['B-Chemical', 'O', 'O', 'B-Disease', 'I-Disease', 'I-Disease', 'O']
        },
        {
            'text': 'EGFR inhibitors in lung cancer treatment',
            'labels': ['B-Gene', 'O', 'O', 'B-Disease', 'I-Disease', 'O']
        },
    ] * 5

    # Test data
    test_data = [
        {
            'text': 'Aspirin reduces inflammation and fever',
            'labels': ['B-Chemical', 'O', 'B-Disease', 'O', 'B-Disease']
        },
        {
            'text': 'KRAS mutations found in pancreatic cancer',
            'labels': ['B-Gene', 'O', 'O', 'O', 'B-Disease', 'I-Disease']
        },
    ] * 5

    return train_data, val_data, test_data


class MedicalDataCollator:
    """
    Data collator for medical NER batches.
    """

    def __init__(self, tokenizer: MedicalTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples.

        Args:
            batch: List of encoded examples

        Returns:
            Batched tensors
        """
        # Stack all tensors
        input_ids = torch.stack([ex['input_ids'] for ex in batch])
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch])

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        # Add labels if present
        if 'labels' in batch[0]:
            labels = torch.stack([ex['labels'] for ex in batch])
            result['labels'] = labels

        # Add token type IDs (all zeros for single sentence)
        result['token_type_ids'] = torch.zeros_like(input_ids)

        return result


def create_medical_dataloaders(
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for medical NER training.

    Args:
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers

    Returns:
        Train, validation, and test data loaders
    """
    # Create tokenizer
    tokenizer = MedicalTokenizer(max_length=max_length)

    # Create datasets
    train_data, val_data, test_data = create_medical_datasets()

    # Create dataset objects
    train_dataset = MedicalNERDataset(train_data, tokenizer, max_length)
    val_dataset = MedicalNERDataset(val_data, tokenizer, max_length)
    test_dataset = MedicalNERDataset(test_data, tokenizer, max_length)

    # Create collator
    collator = MedicalDataCollator(tokenizer)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


class MaskedLanguageModelingDataset(Dataset):
    """
    Dataset for masked language modeling pretraining.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: MedicalTokenizer,
        max_length: int = 128,
        mlm_probability: float = 0.15,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get masked example for MLM training."""
        text = self.texts[idx]

        # Encode text
        encoded = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )

        # Create masked version
        input_ids = encoded['input_ids'].clone()
        labels = encoded['input_ids'].clone()

        # Mask random tokens (excluding special tokens)
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_tokens_mask[input_ids == self.tokenizer.pad_token_id] = True
        special_tokens_mask[input_ids == self.tokenizer.cls_token_id] = True
        special_tokens_mask[input_ids == self.tokenizer.sep_token_id] = True

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Replace masked tokens
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        # Only compute loss on masked tokens
        labels[~masked_indices] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        }