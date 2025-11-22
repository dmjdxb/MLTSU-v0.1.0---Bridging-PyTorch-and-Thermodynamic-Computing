"""
TinyThermoLM: A Minimal Language Model with Thermodynamic Computing
Demonstrates the full PyTorch → TSU bridge with a working language model
This is the crown jewel demonstration of the MLTSU framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import math

from ..tsu_core.interfaces import TSUBackend
from ..tsu_pytorch.attention import ThermodynamicAttention
from ..tsu_pytorch.binary_layer import TSUBinaryLayer
from ..tsu_pytorch.noise import TSUGaussianNoise
from ..tsu_pytorch.negatives import TSUNegativeSampler


class ThermodynamicEmbedding(nn.Module):
    """
    Embedding layer with TSU-powered noise injection for regularization.
    This helps with generalization and robustness.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        tsu_backend: TSUBackend,
        max_seq_len: int = 512,
        noise_level: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Standard embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # TSU noise generator for regularization
        self.tsu_noise = TSUGaussianNoise(tsu_backend, M=12)
        self.noise_level = noise_level

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with TSU noise regularization.

        Args:
            input_ids: Token indices of shape (batch, seq_len)

        Returns:
            Embedded tokens with position encoding
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        position_emb = self.position_embedding(positions)

        # Combine embeddings
        embeddings = token_emb + position_emb

        # Add TSU-generated noise for regularization (training only)
        if self.training and self.noise_level > 0:
            noise = self.tsu_noise.sample_like(embeddings)
            embeddings = embeddings + self.noise_level * noise

        # Layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ThermodynamicTransformerBlock(nn.Module):
    """
    Transformer block using thermodynamic attention and TSU components.
    This is where the magic happens - attention patterns sampled from physical distributions.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        tsu_backend: TSUBackend,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        n_samples: int = 32,
        beta: float = 1.0,
        use_tsu_gating: bool = True,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        # Thermodynamic attention
        self.attention = ThermodynamicAttention(
            d_model=d_model,
            n_heads=n_heads,
            tsu_backend=tsu_backend,
            n_samples=n_samples,
            beta=beta,
            dropout=dropout,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # TSU binary gating for sparse computation
        if use_tsu_gating:
            self.tsu_gate = TSUBinaryLayer(tsu_backend, beta=beta)
        else:
            self.tsu_gate = None

        # Layer norms and dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual
        attn_output = self.attention(self.ln1(x), mask=mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual
        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input)

        # Apply TSU gating for sparse computation
        if self.tsu_gate is not None:
            gate = self.tsu_gate(ffn_input)
            ffn_output = ffn_output * gate

        x = x + self.dropout(ffn_output)

        return x


class TinyThermoLM(nn.Module):
    """
    TinyThermoLM: A complete language model using thermodynamic computing.

    This model demonstrates:
    1. Thermodynamic attention replacing softmax
    2. TSU binary layers for sparse computation
    3. TSU noise generation for regularization
    4. Energy-based negative sampling for training
    5. Hardware-ready architecture with TSU backend

    This is the first-of-its-kind prototype bridging PyTorch to thermodynamic hardware!
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        tsu_backend: TSUBackend,
        max_seq_len: int = 512,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        n_samples: int = 32,
        beta: float = 1.0,
        use_tsu_gating: bool = True,
        use_negative_sampling: bool = True,
        n_negatives: int = 10,
    ):
        """
        Initialize TinyThermoLM.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            tsu_backend: TSU backend for thermodynamic operations
            max_seq_len: Maximum sequence length
            d_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout rate
            n_samples: Number of TSU samples for attention
            beta: Inverse temperature for TSU sampling
            use_tsu_gating: Whether to use TSU binary gating
            use_negative_sampling: Whether to use TSU negative sampling
            n_negatives: Number of negative samples
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tsu_backend = tsu_backend

        # Embedding layer with TSU noise
        self.embedding = ThermodynamicEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            tsu_backend=tsu_backend,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ThermodynamicTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                tsu_backend=tsu_backend,
                d_ff=d_ff,
                dropout=dropout,
                n_samples=n_samples,
                beta=beta,
                use_tsu_gating=use_tsu_gating,
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # TSU negative sampler for training
        if use_negative_sampling:
            self.negative_sampler = TSUNegativeSampler(
                tsu_backend=tsu_backend,
                n_negatives=n_negatives,
                beta=beta,
                sampling_strategy="mixed",
            )
        else:
            self.negative_sampler = None

        # Initialize weights
        self._init_weights()

        # Statistics tracking
        self.register_buffer('perplexity', torch.tensor(0.0))

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TinyThermoLM.

        Args:
            input_ids: Token indices of shape (batch, seq_len)
            labels: Target token indices for loss computation
            mask: Optional attention mask
            return_logits: Whether to return logits

        Returns:
            Dictionary containing:
                - logits: Output logits if return_logits=True
                - loss: Language modeling loss if labels provided
                - hidden_states: Final hidden states
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create causal mask if not provided
        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device),
                diagonal=1
            ).bool()
            mask = ~mask  # Flip to indicate allowed positions

        # Embed tokens
        x = self.embedding(input_ids)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        # Final layer norm
        x = self.ln_final(x)
        hidden_states = x

        outputs = {'hidden_states': hidden_states}

        # Compute logits if requested
        if return_logits:
            logits = self.lm_head(x)
            outputs['logits'] = logits

            # Compute loss if labels provided
            if labels is not None:
                loss = self.compute_loss(logits, labels, hidden_states)
                outputs['loss'] = loss

        return outputs

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute language modeling loss with optional TSU negative sampling.

        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            labels: Target tokens (batch, seq_len)
            hidden_states: Hidden states for energy computation

        Returns:
            Total loss
        """
        # Standard cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Reshape for loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Calculate cross-entropy loss
        ce_loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )

        # Add contrastive loss with TSU negative sampling
        if self.negative_sampler is not None and self.training:
            # Convert logits to energy for negative sampling
            energies = -shift_logits  # Energy is negative log-probability

            # Sample hard negatives
            neg_indices, neg_energies = self.negative_sampler(
                energies,
                target=shift_labels,
            )

            # Compute contrastive loss
            # Positive samples should have lower energy than negatives
            pos_energies = energies.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            margin = 1.0
            contrastive_loss = F.relu(
                neg_energies.mean(-1) - pos_energies + margin
            ).mean()

            total_loss = ce_loss + 0.1 * contrastive_loss
        else:
            total_loss = ce_loss

        # Update perplexity
        self.perplexity = torch.exp(ce_loss).detach()

        return total_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_tsu_sampling: bool = False,
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            input_ids: Starting token indices (batch, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            use_tsu_sampling: Whether to use TSU for token sampling

        Returns:
            Generated token indices
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Generate tokens one at a time
        generated = input_ids

        for _ in range(max_length - input_ids.shape[1]):
            # Get model predictions
            outputs = self(generated, return_logits=True)
            logits = outputs['logits']

            # Get logits for next token
            next_token_logits = logits[:, -1, :] / temperature

            if use_tsu_sampling and hasattr(self, 'tsu_backend'):
                # Use TSU for sampling
                # Convert logits to numpy for TSU
                logits_np = next_token_logits.cpu().numpy()

                # Sample using TSU
                samples = []
                for i in range(batch_size):
                    sample = self.tsu_backend.sample_binary_layer(
                        logits_np[i:i+1],
                        beta=1.0/temperature,
                        num_steps=1,
                    )
                    # Get token with highest probability
                    token = np.argmax(sample)
                    samples.append(token)

                next_tokens = torch.tensor(samples, device=device).unsqueeze(1)
            else:
                # Standard sampling
                if top_k is not None:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)

                if top_p is not None:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)

                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=1)

            # Stop if we hit end token (assuming 0 is padding/end)
            if (next_tokens == 0).all():
                break

        return generated

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
        return logits

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')
        return logits

    def get_statistics(self) -> Dict[str, float]:
        """
        Get model statistics for monitoring.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'perplexity': self.perplexity.item(),
        }

        # Add attention statistics
        for i, block in enumerate(self.transformer_blocks):
            block_stats = block.attention.get_statistics()
            for key, value in block_stats.items():
                stats[f'layer_{i}_{key}'] = value

        # Add negative sampling statistics if available
        if self.negative_sampler is not None:
            neg_stats = self.negative_sampler.get_statistics()
            for key, value in neg_stats.items():
                stats[f'neg_sampling_{key}'] = value

        return stats


def create_tiny_thermo_lm(
    vocab_size: int = 1000,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    tsu_backend: Optional[TSUBackend] = None,
    **kwargs
) -> TinyThermoLM:
    """
    Factory function to create a TinyThermoLM model.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        tsu_backend: TSU backend (will create default if None)
        **kwargs: Additional arguments for TinyThermoLM

    Returns:
        Initialized TinyThermoLM model
    """
    if tsu_backend is None:
        # Try to import JAX backend
        try:
            from ..tsu_jax_sim.backend import JAXTSUBackend
            tsu_backend = JAXTSUBackend(seed=42)
        except ImportError:
            raise ImportError(
                "No TSU backend available. Please install JAX or provide a backend."
            )

    model = TinyThermoLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        tsu_backend=tsu_backend,
        **kwargs
    )

    return model