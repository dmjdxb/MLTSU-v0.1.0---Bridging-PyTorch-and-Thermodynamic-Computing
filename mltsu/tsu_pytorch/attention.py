"""
Thermodynamic Attention for PyTorch
Implements attention using TSU-sampled patterns instead of softmax
This is the core innovation that bridges standard transformers to thermodynamic hardware
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
import math
from ..tsu_core.interfaces import TSUBackend
from .bridge import numpy_to_torch, torch_to_numpy
from .binary_layer import StraightThroughEstimator


class ThermodynamicAttention(nn.Module):
    """
    Multi-head attention using TSU-sampled patterns instead of softmax.

    Key differences from standard attention:
    1. Attention scores are converted to energies: E = -scores
    2. Binary attention patterns are sampled from Boltzmann distribution
    3. Multiple samples are averaged to approximate attention weights
    4. Straight-through estimator preserves gradient flow

    This allows attention computation to be offloaded to thermodynamic hardware
    while maintaining compatibility with PyTorch autograd.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        tsu_backend: TSUBackend,
        n_samples: int = 32,
        beta: float = 1.0,
        dropout: float = 0.1,
        use_tsu: bool = True,
        comparison_mode: bool = False,
    ):
        """
        Initialize Thermodynamic Attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            tsu_backend: TSU backend for sampling
            n_samples: Number of Monte Carlo samples per query
            beta: Inverse temperature (higher = sharper attention)
            dropout: Dropout probability
            use_tsu: Whether to use TSU sampling (False = standard softmax)
            comparison_mode: If True, compute both TSU and standard attention
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_samples = n_samples
        self.beta = beta
        self.use_tsu = use_tsu
        self.comparison_mode = comparison_mode
        self.tsu_backend = tsu_backend

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Statistics tracking
        self.register_buffer('attention_sparsity', torch.tensor(0.0))
        self.register_buffer('attention_entropy', torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of thermodynamic attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Attention mask of shape (batch, seq_len, seq_len) or (seq_len, seq_len)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
            Optionally, attention weights of shape (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Compute Q, K, V projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # Transpose for attention computation: (batch, n_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # Expand mask for all heads and batches
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = mask.expand(batch_size, self.n_heads, -1, -1)
            elif mask.dim() == 3:
                # Expand mask for all heads
                mask = mask.unsqueeze(1)
                mask = mask.expand(-1, self.n_heads, -1, -1)

            # Set masked positions to very negative value
            scores = scores.masked_fill(mask == 0, -1e9)

        # Choose attention mechanism
        if self.use_tsu and not self.comparison_mode:
            # Thermodynamic attention
            attn_weights = self._thermodynamic_attention(scores)
        elif not self.use_tsu and not self.comparison_mode:
            # Standard softmax attention
            attn_weights = F.softmax(scores, dim=-1)
        else:
            # Comparison mode: compute both
            tsu_weights = self._thermodynamic_attention(scores)
            softmax_weights = F.softmax(scores, dim=-1)
            # Use TSU weights but store both for analysis
            attn_weights = tsu_weights
            self._compare_attentions(tsu_weights, softmax_weights)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Final output projection
        output = self.W_o(attn_output)

        # Update statistics
        self._update_statistics(attn_weights)

        if return_attention:
            return output, attn_weights
        return output

    def _thermodynamic_attention(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights using TSU sampling.

        Args:
            scores: Attention scores of shape (batch, n_heads, seq_len, seq_len)

        Returns:
            Attention weights of shape (batch, n_heads, seq_len, seq_len)
        """
        batch_size, n_heads, seq_len_q, seq_len_k = scores.shape
        device = scores.device

        # Convert scores to energies (negative for TSU)
        energies = -scores * self.beta

        # Flatten for TSU processing
        energies_flat = energies.reshape(batch_size * n_heads * seq_len_q, seq_len_k)

        # Sample binary attention patterns using TSU
        all_samples = []

        for i in range(energies_flat.shape[0]):
            query_samples = []
            energy_row = energies_flat[i]

            # Convert to numpy for TSU backend
            energy_np = torch_to_numpy(energy_row)

            # Sample multiple binary patterns
            for _ in range(self.n_samples):
                # TSU samples from Boltzmann distribution
                # P(pattern) ∝ exp(-β * E)
                sample = self.tsu_backend.sample_binary_layer(
                    -energy_np.reshape(1, -1),  # Negate back for logits
                    beta=1.0,  # Beta already applied
                    num_steps=1
                )
                query_samples.append(sample.squeeze())

            # Average samples to get attention probabilities
            avg_attention = np.mean(query_samples, axis=0)

            # Normalize to sum to 1 (like softmax)
            if avg_attention.sum() > 0:
                avg_attention = avg_attention / avg_attention.sum()
            else:
                # Uniform attention if no samples
                avg_attention = np.ones(seq_len_k) / seq_len_k

            all_samples.append(avg_attention)

        # Convert back to torch
        weights_np = np.stack(all_samples)
        weights = numpy_to_torch(weights_np, device=device)

        # Reshape to original dimensions
        weights = weights.reshape(batch_size, n_heads, seq_len_q, seq_len_k)

        # Apply straight-through estimator for gradient flow
        if self.training:
            # Use STE to maintain gradient flow through sampling
            weights = StraightThroughEstimator.apply(scores, weights)

        return weights

    def _compare_attentions(self, tsu_weights: torch.Tensor, softmax_weights: torch.Tensor):
        """
        Compare TSU and softmax attention patterns for analysis.

        Args:
            tsu_weights: TSU-sampled attention weights
            softmax_weights: Standard softmax attention weights
        """
        # Compute KL divergence between distributions
        kl_div = F.kl_div(
            torch.log(tsu_weights + 1e-10),
            softmax_weights,
            reduction='batchmean'
        )

        # Compute cosine similarity
        tsu_flat = tsu_weights.flatten(2)
        softmax_flat = softmax_weights.flatten(2)
        cos_sim = F.cosine_similarity(tsu_flat, softmax_flat, dim=-1).mean()

        # Store for logging
        self.register_buffer('kl_divergence', kl_div.detach())
        self.register_buffer('cosine_similarity', cos_sim.detach())

    def _update_statistics(self, attn_weights: torch.Tensor):
        """
        Update attention statistics for monitoring.

        Args:
            attn_weights: Attention weights tensor
        """
        # Compute sparsity (percentage of near-zero weights)
        sparsity = (attn_weights < 0.01).float().mean()
        self.attention_sparsity = sparsity.detach()

        # Compute entropy
        entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean()
        self.attention_entropy = entropy.detach()

    def get_statistics(self) -> dict:
        """
        Get attention statistics for analysis.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'sparsity': self.attention_sparsity.item(),
            'entropy': self.attention_entropy.item(),
        }

        if self.comparison_mode:
            stats['kl_divergence'] = self.kl_divergence.item()
            stats['cosine_similarity'] = self.cosine_similarity.item()

        return stats


class ThermodynamicMultiHeadAttention(ThermodynamicAttention):
    """
    Alias for ThermodynamicAttention with clearer naming.
    """
    pass


class ThermodynamicSelfAttention(nn.Module):
    """
    Self-attention layer using thermodynamic attention.
    Includes layer normalization and residual connection.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        tsu_backend: TSUBackend,
        n_samples: int = 32,
        beta: float = 1.0,
        dropout: float = 0.1,
        use_tsu: bool = True,
    ):
        """
        Initialize thermodynamic self-attention layer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            tsu_backend: TSU backend for sampling
            n_samples: Number of Monte Carlo samples
            beta: Inverse temperature
            dropout: Dropout probability
            use_tsu: Whether to use TSU sampling
        """
        super().__init__()

        self.attention = ThermodynamicAttention(
            d_model=d_model,
            n_heads=n_heads,
            tsu_backend=tsu_backend,
            n_samples=n_samples,
            beta=beta,
            dropout=dropout,
            use_tsu=use_tsu,
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with residual connection and layer norm.

        Args:
            x: Input tensor
            mask: Optional attention mask

        Returns:
            Output tensor
        """
        # Apply attention with residual connection
        attn_output = self.attention(x, mask)
        x = x + self.dropout(attn_output)

        # Apply layer normalization
        x = self.layer_norm(x)

        return x


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal mask for autoregressive attention.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return (1 - mask).bool()