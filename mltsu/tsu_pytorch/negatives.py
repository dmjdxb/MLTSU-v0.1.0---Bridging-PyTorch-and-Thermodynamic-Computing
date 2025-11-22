"""
TSU Negative Sampling for Energy-Based Models
Implements hard negative sampling using thermodynamic sampling units
Critical for contrastive learning and energy-based language modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from ..tsu_core.interfaces import TSUBackend
from .bridge import numpy_to_torch, torch_to_numpy


class TSUNegativeSampler(nn.Module):
    """
    Sample hard negatives from energy-based distributions using TSU.

    This module samples negative examples from low-energy regions of the
    energy landscape, providing harder negatives than uniform sampling.
    This is critical for:
    - Contrastive learning objectives
    - Energy-based language models
    - Alignment training for LLMs
    """

    def __init__(
        self,
        tsu_backend: TSUBackend,
        n_negatives: int = 10,
        beta: float = 1.0,
        sampling_strategy: str = "energy_based",
        exclude_target: bool = True,
        hard_negative_ratio: float = 0.5,
    ):
        """
        Initialize TSU Negative Sampler.

        Args:
            tsu_backend: TSU backend for sampling
            n_negatives: Number of negative samples per positive
            beta: Inverse temperature (higher = focus on lower energy)
            sampling_strategy: Strategy for sampling ("energy_based", "top_k", "mixed")
            exclude_target: Whether to exclude the target from negative samples
            hard_negative_ratio: Ratio of hard negatives (low energy) to random
        """
        super().__init__()

        self.tsu_backend = tsu_backend
        self.n_negatives = n_negatives
        self.beta = beta
        self.sampling_strategy = sampling_strategy
        self.exclude_target = exclude_target
        self.hard_negative_ratio = hard_negative_ratio

        # Statistics tracking
        self.register_buffer('avg_negative_energy', torch.tensor(0.0))
        self.register_buffer('avg_positive_energy', torch.tensor(0.0))
        self.register_buffer('energy_gap', torch.tensor(0.0))

    def forward(
        self,
        energy: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negative examples from energy distribution.

        Args:
            energy: Energy tensor of shape (batch, vocab_size) or (batch, seq_len, vocab_size)
            target: Target indices of shape (batch,) or (batch, seq_len)
            mask: Optional mask for valid positions (batch, seq_len)

        Returns:
            negative_indices: Sampled negative indices (batch, n_negatives) or
                            (batch, seq_len, n_negatives)
            negative_energies: Energies of sampled negatives
        """
        device = energy.device
        original_shape = energy.shape

        # Handle different input dimensions
        if energy.dim() == 2:
            # (batch, vocab_size)
            batch_size, vocab_size = energy.shape
            seq_len = 1
            energy = energy.unsqueeze(1)  # (batch, 1, vocab_size)
            if target is not None and target.dim() == 1:
                target = target.unsqueeze(1)  # (batch, 1)
        else:
            # (batch, seq_len, vocab_size)
            batch_size, seq_len, vocab_size = energy.shape

        # Flatten for processing
        energy_flat = energy.reshape(batch_size * seq_len, vocab_size)

        if target is not None:
            target_flat = target.reshape(batch_size * seq_len)
        else:
            target_flat = None

        # Sample negatives for each position
        all_negative_indices = []
        all_negative_energies = []

        for i in range(batch_size * seq_len):
            # Skip masked positions if mask is provided
            if mask is not None:
                mask_idx = i // vocab_size
                if not mask.flatten()[mask_idx]:
                    # Return dummy negatives for masked positions
                    neg_idx = torch.zeros(self.n_negatives, dtype=torch.long, device=device)
                    neg_energy = torch.zeros(self.n_negatives, device=device)
                    all_negative_indices.append(neg_idx)
                    all_negative_energies.append(neg_energy)
                    continue

            # Get energy for this position
            pos_energy = energy_flat[i]

            # Exclude target if specified
            if self.exclude_target and target_flat is not None:
                target_idx = target_flat[i].item()
                # Set target energy to very high value to exclude it
                pos_energy = pos_energy.clone()
                pos_energy[target_idx] = float('inf')

            # Sample negatives based on strategy
            if self.sampling_strategy == "energy_based":
                neg_indices, neg_energies = self._energy_based_sampling(
                    pos_energy, self.n_negatives
                )
            elif self.sampling_strategy == "top_k":
                neg_indices, neg_energies = self._top_k_sampling(
                    pos_energy, self.n_negatives
                )
            elif self.sampling_strategy == "mixed":
                neg_indices, neg_energies = self._mixed_sampling(
                    pos_energy, self.n_negatives
                )
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

            all_negative_indices.append(neg_indices)
            all_negative_energies.append(neg_energies)

        # Stack results
        negative_indices = torch.stack(all_negative_indices)
        negative_energies = torch.stack(all_negative_energies)

        # Reshape to match input dimensions
        if original_shape[1] == 2:  # Was (batch, vocab_size)
            negative_indices = negative_indices.squeeze(1)
            negative_energies = negative_energies.squeeze(1)
        else:
            negative_indices = negative_indices.reshape(batch_size, seq_len, self.n_negatives)
            negative_energies = negative_energies.reshape(batch_size, seq_len, self.n_negatives)

        # Update statistics
        self._update_statistics(energy, target, negative_energies)

        return negative_indices, negative_energies

    def _energy_based_sampling(
        self,
        energy: torch.Tensor,
        n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from energy-based distribution using TSU.

        Args:
            energy: Energy values for all candidates (vocab_size,)
            n_samples: Number of samples

        Returns:
            indices: Sampled indices
            energies: Energies of sampled indices
        """
        device = energy.device

        # Convert to numpy for TSU backend
        energy_np = torch_to_numpy(energy)

        # TSU samples from Boltzmann distribution P(x) ∝ exp(-β * E(x))
        # We want to sample low-energy states (hard negatives)
        logits = -energy_np * self.beta  # Convert energy to logits

        # Sample multiple times to get diverse negatives
        samples = []
        for _ in range(n_samples):
            # Sample a binary vector indicating selection
            sample = self.tsu_backend.sample_binary_layer(
                logits.reshape(1, -1),
                beta=1.0,  # Beta already applied in logits
                num_steps=1
            ).squeeze()

            # Get the index of the selected item
            # If multiple items selected, pick one randomly
            selected = np.where(sample > 0.5)[0]
            if len(selected) > 0:
                idx = np.random.choice(selected)
            else:
                # Fallback to weighted sampling if TSU returns no selection
                probs = np.exp(-energy_np * self.beta)
                probs = probs / probs.sum()
                idx = np.random.choice(len(energy_np), p=probs)

            samples.append(idx)

        # Convert to tensor
        indices = torch.tensor(samples, dtype=torch.long, device=device)
        energies = energy[indices]

        return indices, energies

    def _top_k_sampling(
        self,
        energy: torch.Tensor,
        n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from top-k lowest energy states.

        Args:
            energy: Energy values (vocab_size,)
            n_samples: Number of samples

        Returns:
            indices: Sampled indices
            energies: Energies of sampled indices
        """
        # Get top-k lowest energy indices
        k = min(n_samples * 3, len(energy))  # Pool size
        top_k_energies, top_k_indices = torch.topk(energy, k, largest=False)

        # Sample from top-k with probability proportional to exp(-β * E)
        probs = F.softmax(-top_k_energies * self.beta, dim=0)
        sampled_idx = torch.multinomial(probs, n_samples, replacement=True)

        indices = top_k_indices[sampled_idx]
        energies = top_k_energies[sampled_idx]

        return indices, energies

    def _mixed_sampling(
        self,
        energy: torch.Tensor,
        n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixed sampling strategy: combination of hard and random negatives.

        Args:
            energy: Energy values (vocab_size,)
            n_samples: Number of samples

        Returns:
            indices: Sampled indices
            energies: Energies of sampled indices
        """
        n_hard = int(n_samples * self.hard_negative_ratio)
        n_random = n_samples - n_hard

        # Hard negatives from energy-based sampling
        if n_hard > 0:
            hard_indices, hard_energies = self._energy_based_sampling(energy, n_hard)
        else:
            hard_indices = torch.tensor([], dtype=torch.long, device=energy.device)
            hard_energies = torch.tensor([], device=energy.device)

        # Random negatives
        if n_random > 0:
            random_indices = torch.randint(0, len(energy), (n_random,), device=energy.device)
            random_energies = energy[random_indices]
        else:
            random_indices = torch.tensor([], dtype=torch.long, device=energy.device)
            random_energies = torch.tensor([], device=energy.device)

        # Combine
        indices = torch.cat([hard_indices, random_indices])
        energies = torch.cat([hard_energies, random_energies])

        return indices, energies

    def _update_statistics(
        self,
        energy: torch.Tensor,
        target: Optional[torch.Tensor],
        negative_energies: torch.Tensor
    ):
        """
        Update sampling statistics for monitoring.

        Args:
            energy: Full energy tensor
            target: Target indices
            negative_energies: Energies of sampled negatives
        """
        # Average negative energy
        self.avg_negative_energy = negative_energies.mean().detach()

        # Average positive energy (if target provided)
        if target is not None:
            # Get energies of target tokens
            if energy.dim() == 3:
                batch_size, seq_len, vocab_size = energy.shape
                target_flat = target.flatten()
                energy_flat = energy.reshape(-1, vocab_size)
                positive_energies = energy_flat[torch.arange(len(target_flat)), target_flat]
            else:
                positive_energies = energy[torch.arange(len(target)), target]

            self.avg_positive_energy = positive_energies.mean().detach()
            self.energy_gap = (self.avg_negative_energy - self.avg_positive_energy).detach()

    def get_statistics(self) -> dict:
        """
        Get sampling statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            'avg_negative_energy': self.avg_negative_energy.item(),
            'avg_positive_energy': self.avg_positive_energy.item(),
            'energy_gap': self.energy_gap.item(),
        }


class ContrastiveNegativeSampler(TSUNegativeSampler):
    """
    Specialized negative sampler for contrastive learning objectives.
    """

    def __init__(
        self,
        tsu_backend: TSUBackend,
        n_negatives: int = 10,
        beta: float = 1.0,
        temperature: float = 0.07,
    ):
        """
        Initialize contrastive negative sampler.

        Args:
            tsu_backend: TSU backend
            n_negatives: Number of negatives per positive
            beta: Inverse temperature for sampling
            temperature: Temperature for contrastive loss
        """
        super().__init__(
            tsu_backend=tsu_backend,
            n_negatives=n_negatives,
            beta=beta,
            sampling_strategy="energy_based",
            exclude_target=True,
        )
        self.temperature = temperature

    def compute_contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss with TSU-sampled negatives.

        Args:
            anchor: Anchor embeddings (batch, dim)
            positive: Positive embeddings (batch, dim)
            energy: Energy tensor for negative sampling (batch, vocab_size)

        Returns:
            Contrastive loss
        """
        batch_size = anchor.shape[0]
        device = anchor.device

        # Sample negatives
        neg_indices, _ = self.forward(energy)  # (batch, n_negatives)

        # Get negative embeddings (would need embedding layer in practice)
        # For now, we'll compute similarity based on energy

        # Compute similarities
        pos_sim = F.cosine_similarity(anchor, positive, dim=1) / self.temperature

        # For negatives, use energy as proxy for similarity
        neg_sim = -energy.gather(1, neg_indices) / self.temperature  # (batch, n_negatives)

        # Contrastive loss (InfoNCE style)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch, 1 + n_negatives)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # Positive is at index 0

        loss = F.cross_entropy(logits, labels)

        return loss


class HierarchicalNegativeSampler(nn.Module):
    """
    Hierarchical negative sampling using multiple TSU samplers at different temperatures.
    Useful for curriculum learning and progressive training.
    """

    def __init__(
        self,
        tsu_backend: TSUBackend,
        n_negatives_per_level: List[int] = [5, 10, 20],
        betas: List[float] = [0.5, 1.0, 2.0],
    ):
        """
        Initialize hierarchical sampler.

        Args:
            tsu_backend: TSU backend
            n_negatives_per_level: Number of negatives at each level
            betas: Inverse temperatures for each level (easy to hard)
        """
        super().__init__()

        assert len(n_negatives_per_level) == len(betas)

        self.levels = nn.ModuleList([
            TSUNegativeSampler(
                tsu_backend=tsu_backend,
                n_negatives=n_neg,
                beta=beta,
                sampling_strategy="energy_based",
            )
            for n_neg, beta in zip(n_negatives_per_level, betas)
        ])

    def forward(
        self,
        energy: torch.Tensor,
        level: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negatives at specified difficulty level.

        Args:
            energy: Energy tensor
            level: Difficulty level (None = all levels)

        Returns:
            negative_indices: Sampled indices
            negative_energies: Energies
        """
        if level is not None:
            return self.levels[level](energy)

        # Sample from all levels and concatenate
        all_indices = []
        all_energies = []

        for sampler in self.levels:
            indices, energies = sampler(energy)
            all_indices.append(indices)
            all_energies.append(energies)

        negative_indices = torch.cat(all_indices, dim=-1)
        negative_energies = torch.cat(all_energies, dim=-1)

        return negative_indices, negative_energies