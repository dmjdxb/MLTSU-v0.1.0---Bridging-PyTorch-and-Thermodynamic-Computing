"""
TinyBioBERT: Medical BERT with Probabilistic Bit (P-bit) Training
Implements a lightweight BERT model for medical NER using thermodynamic computing.
This bridges mainstream PyTorch medical NLP with TSU hardware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass
import math

from ..tsu_core.interfaces import TSUBackend
from ..tsu_pytorch.attention import ThermodynamicAttention
from ..tsu_pytorch.binary_layer import TSUBinaryLayer
from ..tsu_pytorch.noise import TSUGaussianNoise


@dataclass
class TinyBioBERTConfig:
    """Configuration for TinyBioBERT model."""
    vocab_size: int = 10000  # Reduced from BioBERT's 30K
    hidden_size: int = 256  # Reduced from 768
    num_hidden_layers: int = 4  # Reduced from 12
    num_attention_heads: int = 4  # Reduced from 12
    intermediate_size: int = 1024  # Reduced from 3072
    max_position_embeddings: int = 128  # Reduced from 512
    type_vocab_size: int = 2  # For segment embeddings
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02

    # P-bit specific parameters
    pbit_temperature: float = 1.0  # Beta for TSU sampling
    num_pbit_samples: int = 32  # Number of samples for attention
    use_pbit_dropout: bool = True
    use_pbit_attention: bool = True
    pbit_noise_level: float = 0.1

    # Medical domain specific
    num_labels: int = 9  # NCBI disease NER labels (B-Disease, I-Disease, etc.)
    medical_uncertainty: bool = True
    calibration_temperature: float = 1.0


class BERTEmbedding(nn.Module):
    """
    BERT embedding layer with token, position, and segment embeddings.
    Enhanced with P-bit noise for regularization.
    """

    def __init__(self, config: TinyBioBERTConfig, tsu_backend: TSUBackend):
        super().__init__()
        self.config = config

        # Three types of embeddings for BERT
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # TSU noise generator for regularization
        self.tsu_noise = TSUGaussianNoise(tsu_backend, M=12)
        self.noise_level = config.pbit_noise_level

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with small random values."""
        for embedding in [self.token_embeddings, self.position_embeddings, self.token_type_embeddings]:
            nn.init.normal_(embedding.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for BERT embeddings.

        Args:
            input_ids: Token indices (batch_size, seq_length)
            token_type_ids: Segment token indices (batch_size, seq_length)
            position_ids: Position indices (batch_size, seq_length)

        Returns:
            Embedded representation (batch_size, seq_length, hidden_size)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Generate token type IDs if not provided (all zeros for single sentence)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Combine all embeddings
        embeddings = token_embeds + position_embeds + token_type_embeds

        # Add P-bit noise for regularization (training only)
        if self.training and self.noise_level > 0:
            noise = self.tsu_noise.sample_like(embeddings)
            embeddings = embeddings + self.noise_level * noise

        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class PbitDropout(nn.Module):
    """
    Dropout using P-bit sampling from TSU backend.
    More physically grounded than standard dropout.
    """

    def __init__(self, p: float, tsu_backend: TSUBackend):
        super().__init__()
        self.p = p
        self.tsu_backend = tsu_backend

        # Convert dropout probability to logit for TSU sampling
        if p > 0 and p < 1:
            self.logit = np.log((1 - p) / p)
        else:
            self.logit = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply P-bit dropout."""
        if not self.training or self.p == 0:
            return x

        # Sample binary mask using TSU
        shape = x.shape
        device = x.device

        # Create logits for TSU sampling (negative for keep probability)
        logits_np = np.full(shape, self.logit)

        # Sample binary mask
        mask = self.tsu_backend.sample_binary_layer(
            logits_np.reshape(1, -1),
            beta=1.0,
            num_steps=1
        ).reshape(shape)

        # Convert to torch and apply
        mask = torch.from_numpy(mask).float().to(device)

        # Scale by dropout probability
        return x * mask / (1 - self.p)


class PbitAttention(nn.Module):
    """
    BERT attention layer using P-bit sampling.
    Wraps ThermodynamicAttention for BERT-style bidirectional attention.
    """

    def __init__(self, config: TinyBioBERTConfig, tsu_backend: TSUBackend):
        super().__init__()
        self.config = config

        # Use existing ThermodynamicAttention
        self.attention = ThermodynamicAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            tsu_backend=tsu_backend,
            n_samples=config.num_pbit_samples,
            beta=config.pbit_temperature,
            dropout=config.attention_probs_dropout_prob,
        )

        # Additional dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for P-bit attention.

        Args:
            hidden_states: Input tensor (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask (batch_size, seq_length, seq_length)

        Returns:
            Attention output (batch_size, seq_length, hidden_size)
        """
        # ThermodynamicAttention handles everything
        attention_output = self.attention(hidden_states, mask=attention_mask)
        return self.dropout(attention_output)


class TinyBioBERTLayer(nn.Module):
    """
    Single BERT encoder layer with P-bit components.
    """

    def __init__(self, config: TinyBioBERTConfig, tsu_backend: TSUBackend):
        super().__init__()
        self.config = config

        # P-bit attention
        self.attention = PbitAttention(config, tsu_backend)

        # Feed-forward network
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)

        # Layer norms
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # P-bit dropout
        if config.use_pbit_dropout:
            self.dropout = PbitDropout(config.hidden_dropout_prob, tsu_backend)
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Activation
        self.activation = nn.GELU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through BERT layer.

        Args:
            hidden_states: Input tensor (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask

        Returns:
            Layer output (batch_size, seq_length, hidden_size)
        """
        # Self-attention with residual connection
        attention_output = self.attention(
            self.attention_layer_norm(hidden_states),
            attention_mask=attention_mask
        )
        hidden_states = hidden_states + attention_output

        # Feed-forward with residual connection
        layer_output = self.output_layer_norm(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.activation(layer_output)
        layer_output = self.output(layer_output)
        layer_output = self.dropout(layer_output)

        output = hidden_states + layer_output

        return output


class TinyBioBERTEncoder(nn.Module):
    """
    BERT encoder with multiple layers.
    """

    def __init__(self, config: TinyBioBERTConfig, tsu_backend: TSUBackend):
        super().__init__()
        self.config = config

        # Stack of BERT layers
        self.layers = nn.ModuleList([
            TinyBioBERTLayer(config, tsu_backend)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through encoder.

        Args:
            hidden_states: Input embeddings (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask
            output_hidden_states: Whether to return all hidden states

        Returns:
            Final hidden states or tuple of (final_states, all_hidden_states)
        """
        all_hidden_states = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states = layer(hidden_states, attention_mask)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            return hidden_states, all_hidden_states

        return hidden_states


class TinyBioBERTPooler(nn.Module):
    """
    Pooler for BERT's [CLS] token representation.
    """

    def __init__(self, config: TinyBioBERTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool by taking [CLS] token and applying dense layer."""
        # Take [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TinyBioBERT(nn.Module):
    """
    TinyBioBERT: Complete BERT model for medical NLP with P-bit training.

    This model demonstrates:
    1. P-bit attention mechanisms for energy-efficient attention
    2. P-bit dropout for regularization
    3. Medical domain specialization for NER tasks
    4. Uncertainty quantification for medical applications
    5. Hardware-ready architecture with TSU backend
    """

    def __init__(self, config: TinyBioBERTConfig, tsu_backend: TSUBackend):
        super().__init__()
        self.config = config
        self.tsu_backend = tsu_backend

        # BERT components
        self.embeddings = BERTEmbedding(config, tsu_backend)
        self.encoder = TinyBioBERTEncoder(config, tsu_backend)
        self.pooler = TinyBioBERTPooler(config)

        # Initialize weights
        self.apply(self._init_weights)

        # Statistics tracking
        self.register_buffer('total_energy', torch.tensor(0.0))
        self.register_buffer('total_samples', torch.tensor(0))

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TinyBioBERT.

        Args:
            input_ids: Token indices (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            token_type_ids: Segment token indices
            position_ids: Position indices
            output_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary containing:
                - last_hidden_state: Final layer hidden states
                - pooler_output: Pooled [CLS] representation
                - hidden_states: All hidden states (if requested)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Prepare attention mask (1 for tokens to attend to, 0 for padding)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        # Convert attention mask to extended format for multi-head attention
        # Shape: (batch_size, 1, 1, seq_length) -> broadcast for all heads
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since we're using bidirectional attention, we don't need causal masking
        # Just mask out padding tokens
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Get embeddings
        embedding_output = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        # Pass through encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=output_hidden_states
        )

        if output_hidden_states:
            sequence_output, all_hidden_states = encoder_outputs
        else:
            sequence_output = encoder_outputs
            all_hidden_states = None

        # Pool [CLS] token
        pooled_output = self.pooler(sequence_output)

        # Track energy consumption (simulated)
        if self.training:
            self._track_energy_consumption(batch_size)

        # Prepare outputs
        outputs = {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
        }

        if output_hidden_states:
            outputs['hidden_states'] = all_hidden_states

        return outputs

    def _track_energy_consumption(self, batch_size: int):
        """Track simulated energy consumption for P-bit operations."""
        # Estimate energy based on P-bit samples
        # Each P-bit operation ~ 10^-15 J (estimated)
        pbit_energy = 1e-15

        # Count P-bit operations
        num_attention_ops = (
            self.config.num_hidden_layers *
            self.config.num_attention_heads *
            self.config.num_pbit_samples *
            batch_size
        )

        # Update tracking
        self.total_energy += num_attention_ops * pbit_energy
        self.total_samples += batch_size

    def get_energy_statistics(self) -> Dict[str, float]:
        """Get energy consumption statistics."""
        return {
            'total_energy_joules': self.total_energy.item(),
            'total_samples': self.total_samples.item(),
            'energy_per_sample': (
                self.total_energy.item() / max(1, self.total_samples.item())
            ),
        }


class TinyBioBERTForTokenClassification(nn.Module):
    """
    TinyBioBERT for medical NER (Named Entity Recognition).
    Adds a token classification head on top of the base model.
    """

    def __init__(self, config: TinyBioBERTConfig, tsu_backend: TSUBackend):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        # Base BERT model
        self.bert = TinyBioBERT(config, tsu_backend)

        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Medical uncertainty quantification
        if config.medical_uncertainty:
            self.temperature = nn.Parameter(torch.ones(1) * config.calibration_temperature)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for token classification.

        Args:
            input_ids: Token indices (batch_size, seq_length)
            attention_mask: Attention mask
            token_type_ids: Segment token indices
            labels: Ground truth labels for NER (batch_size, seq_length)

        Returns:
            Dictionary containing:
                - logits: Classification logits
                - loss: Cross-entropy loss (if labels provided)
                - hidden_states: BERT hidden states
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs['last_hidden_state']

        # Apply dropout and classifier
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # Apply temperature scaling for uncertainty calibration
        if self.config.medical_uncertainty:
            logits = logits / self.temperature

        # Prepare outputs
        result = {
            'logits': logits,
            'hidden_states': sequence_output,
        }

        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            # Only compute loss on non-padding tokens
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            result['loss'] = loss

        return result


class TinyBioBERTForSequenceClassification(nn.Module):
    """
    TinyBioBERT for sequence classification tasks.
    Uses the pooled [CLS] representation for classification.
    """

    def __init__(self, config: TinyBioBERTConfig, tsu_backend: TSUBackend):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        # Base BERT model
        self.bert = TinyBioBERT(config, tsu_backend)

        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for sequence classification.

        Args:
            input_ids: Token indices
            attention_mask: Attention mask
            token_type_ids: Segment token indices
            labels: Ground truth labels

        Returns:
            Dictionary containing logits and optional loss
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use pooled [CLS] representation
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        result = {'logits': logits}

        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            result['loss'] = loss

        return result


def create_tiny_biobert(
    vocab_size: int = 10000,
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    tsu_backend: Optional[TSUBackend] = None,
    task: str = 'ner',
    **kwargs
) -> nn.Module:
    """
    Factory function to create a TinyBioBERT model.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_hidden_layers: Number of encoder layers
        num_attention_heads: Number of attention heads
        tsu_backend: TSU backend for P-bit operations
        task: Task type ('ner' or 'classification')
        **kwargs: Additional config parameters

    Returns:
        TinyBioBERT model for the specified task
    """
    # Create configuration
    config = TinyBioBERTConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        **kwargs
    )

    # Get TSU backend if not provided
    if tsu_backend is None:
        try:
            from ..tsu_jax_sim.backend import JAXTSUBackend
            tsu_backend = JAXTSUBackend(seed=42)
        except ImportError:
            raise ImportError("No TSU backend available. Please install JAX or provide a backend.")

    # Create model based on task
    if task == 'ner':
        model = TinyBioBERTForTokenClassification(config, tsu_backend)
    elif task == 'classification':
        model = TinyBioBERTForSequenceClassification(config, tsu_backend)
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'ner' or 'classification'.")

    return model