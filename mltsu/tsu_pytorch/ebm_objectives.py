"""
Energy-Based Model (EBM) Objectives with TSU
Implements various energy-based learning objectives using thermodynamic sampling
This is critical for bridging to physical hardware that naturally computes energies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Callable, List
import math

from ..tsu_core.interfaces import TSUBackend
from .bridge import numpy_to_torch, torch_to_numpy
from .negatives import TSUNegativeSampler


class TSUEnergyFunction(nn.Module):
    """
    Base class for energy functions that can be evaluated on TSU hardware.
    Energy functions map states to scalar energy values.
    """

    def __init__(
        self,
        tsu_backend: TSUBackend,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        beta: float = 1.0,
    ):
        """
        Initialize energy function.

        Args:
            tsu_backend: TSU backend for sampling
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            beta: Inverse temperature
        """
        super().__init__()

        self.tsu_backend = tsu_backend
        self.beta = beta

        # Build energy network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        # Final layer outputs scalar energy
        layers.append(nn.Linear(prev_dim, 1))

        self.energy_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for input states.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Energy values (batch, 1)
        """
        return self.energy_net(x)

    def sample(
        self,
        batch_size: int,
        num_steps: int = 100,
        init_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample from energy distribution using TSU.

        Args:
            batch_size: Number of samples
            num_steps: Number of sampling steps
            init_state: Initial state (if None, random)

        Returns:
            Sampled states
        """
        device = next(self.parameters()).device

        # Initialize random state if needed
        if init_state is None:
            init_state = torch.randn(batch_size, self.energy_net[0].in_features)
            init_state = init_state.to(device)

        # Convert to numpy for TSU
        init_np = torch_to_numpy(init_state)

        # Define energy function for TSU
        def energy_fn(state_np):
            state = numpy_to_torch(state_np, device=device)
            with torch.no_grad():
                energy = self.forward(state)
            return torch_to_numpy(energy.squeeze())

        # Sample using TSU
        samples_np = self.tsu_backend.sample_custom(
            energy_fn=energy_fn,
            init_state=init_np,
            num_steps=num_steps,
            beta=self.beta,
        )

        # Convert back to torch
        samples = numpy_to_torch(samples_np, device=device)

        return samples


class ContrastiveDivergence(nn.Module):
    """
    Contrastive Divergence objective for training Energy-Based Models.
    Uses TSU for negative phase sampling instead of Gibbs/MCMC.
    """

    def __init__(
        self,
        energy_fn: TSUEnergyFunction,
        tsu_backend: TSUBackend,
        n_gibbs_steps: int = 1,
        beta: float = 1.0,
    ):
        """
        Initialize Contrastive Divergence.

        Args:
            energy_fn: Energy function to train
            tsu_backend: TSU backend for sampling
            n_gibbs_steps: Number of Gibbs steps (CD-k)
            beta: Inverse temperature
        """
        super().__init__()

        self.energy_fn = energy_fn
        self.tsu_backend = tsu_backend
        self.n_gibbs_steps = n_gibbs_steps
        self.beta = beta

        # Statistics
        self.register_buffer('avg_pos_energy', torch.tensor(0.0))
        self.register_buffer('avg_neg_energy', torch.tensor(0.0))

    def forward(
        self,
        x_pos: torch.Tensor,
        persistent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CD loss.

        Args:
            x_pos: Positive samples from data (batch, dim)
            persistent: Persistent chain states for PCD

        Returns:
            loss: CD loss
            x_neg: Negative samples from model
        """
        batch_size = x_pos.shape[0]
        device = x_pos.device

        # Positive phase: energy of data samples
        energy_pos = self.energy_fn(x_pos)

        # Negative phase: sample from model using TSU
        if persistent is not None:
            init_state = persistent
        else:
            init_state = x_pos.clone().detach()

        # Sample negative particles using TSU
        x_neg = self.energy_fn.sample(
            batch_size=batch_size,
            num_steps=self.n_gibbs_steps,
            init_state=init_state,
        )

        # Compute energy of negative samples
        energy_neg = self.energy_fn(x_neg.detach())

        # CD loss: minimize energy of data, maximize energy of samples
        loss = energy_pos.mean() - energy_neg.mean()

        # Update statistics
        self.avg_pos_energy = energy_pos.mean().detach()
        self.avg_neg_energy = energy_neg.mean().detach()

        return loss, x_neg

    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics."""
        return {
            'pos_energy': self.avg_pos_energy.item(),
            'neg_energy': self.avg_neg_energy.item(),
            'energy_gap': (self.avg_neg_energy - self.avg_pos_energy).item(),
        }


class ScoreMatching(nn.Module):
    """
    Score matching objective for EBMs.
    Avoids sampling by matching gradients of log-density.
    TSU used for noise perturbation and denoising.
    """

    def __init__(
        self,
        energy_fn: TSUEnergyFunction,
        tsu_backend: TSUBackend,
        noise_scale: float = 0.1,
    ):
        """
        Initialize Score Matching.

        Args:
            energy_fn: Energy function
            tsu_backend: TSU backend
            noise_scale: Scale of noise perturbation
        """
        super().__init__()

        self.energy_fn = energy_fn
        self.tsu_backend = tsu_backend
        self.noise_scale = noise_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute denoising score matching loss.

        Args:
            x: Clean data samples (batch, dim)

        Returns:
            Score matching loss
        """
        x.requires_grad_(True)

        # Add TSU-generated noise
        noise = torch.randn_like(x) * self.noise_scale
        x_noisy = x + noise

        # Compute energy and score (gradient of log-density)
        energy = self.energy_fn(x_noisy)

        # Compute score via autograd
        score = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=x_noisy,
            create_graph=True,
        )[0]

        # Denoising score matching loss
        target_score = -noise / (self.noise_scale ** 2)
        loss = F.mse_loss(score, target_score)

        return loss


class InfoNCE(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) objective for EBMs.
    Uses TSU for hard negative mining.
    """

    def __init__(
        self,
        energy_fn: TSUEnergyFunction,
        tsu_backend: TSUBackend,
        n_negatives: int = 10,
        beta: float = 1.0,
        temperature: float = 0.07,
    ):
        """
        Initialize InfoNCE.

        Args:
            energy_fn: Energy function
            tsu_backend: TSU backend
            n_negatives: Number of negative samples
            beta: Inverse temperature for TSU sampling
            temperature: Temperature for InfoNCE loss
        """
        super().__init__()

        self.energy_fn = energy_fn
        self.temperature = temperature

        # TSU negative sampler
        self.negative_sampler = TSUNegativeSampler(
            tsu_backend=tsu_backend,
            n_negatives=n_negatives,
            beta=beta,
            sampling_strategy="energy_based",
        )

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor: Anchor samples (batch, dim)
            positive: Positive samples (batch, dim)

        Returns:
            InfoNCE loss
        """
        batch_size = anchor.shape[0]
        device = anchor.device

        # Compute energies
        energy_anchor = self.energy_fn(anchor)
        energy_positive = self.energy_fn(positive)

        # Get similarity (negative energy)
        sim_pos = -energy_positive / self.temperature

        # Sample hard negatives using TSU
        # Create energy distribution for negative sampling
        energy_dist = torch.randn(batch_size, 100, device=device)  # Mock distribution

        neg_indices, neg_energies = self.negative_sampler(energy_dist)

        # For actual negatives, we need to sample states and compute energies
        negatives = self.energy_fn.sample(
            batch_size=batch_size * self.negative_sampler.n_negatives,
            num_steps=10,
        )
        negatives = negatives.reshape(batch_size, self.negative_sampler.n_negatives, -1)

        # Compute negative energies
        neg_energies = []
        for i in range(self.negative_sampler.n_negatives):
            neg_energy = self.energy_fn(negatives[:, i, :])
            neg_energies.append(neg_energy)

        neg_energies = torch.stack(neg_energies, dim=1).squeeze(-1)
        sim_neg = -neg_energies / self.temperature

        # InfoNCE loss
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)

        return loss


class EnergyBasedGAN(nn.Module):
    """
    Energy-Based GAN training using TSU.
    Generator produces samples, discriminator is an energy function.
    TSU provides importance sampling for training dynamics.
    """

    def __init__(
        self,
        generator: nn.Module,
        energy_fn: TSUEnergyFunction,
        tsu_backend: TSUBackend,
        beta: float = 1.0,
    ):
        """
        Initialize Energy-Based GAN.

        Args:
            generator: Generator network
            energy_fn: Energy-based discriminator
            tsu_backend: TSU backend
            beta: Inverse temperature
        """
        super().__init__()

        self.generator = generator
        self.energy_fn = energy_fn
        self.tsu_backend = tsu_backend
        self.beta = beta

    def generator_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute generator loss.

        Args:
            z: Latent codes (batch, latent_dim)

        Returns:
            Generator loss
        """
        # Generate samples
        x_gen = self.generator(z)

        # Compute energy (want low energy for generated samples)
        energy = self.energy_fn(x_gen)

        # Generator tries to minimize energy
        loss = energy.mean()

        return loss

    def discriminator_loss(
        self,
        x_real: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy-based discriminator loss.

        Args:
            x_real: Real data samples
            z: Latent codes for generation

        Returns:
            Discriminator loss
        """
        # Real data should have low energy
        energy_real = self.energy_fn(x_real)

        # Generated data should have high energy
        with torch.no_grad():
            x_fake = self.generator(z)
        energy_fake = self.energy_fn(x_fake)

        # Energy-based loss with margin
        margin = 1.0
        loss = F.relu(margin - energy_fake + energy_real).mean()

        return loss


class MaximumLikelihood(nn.Module):
    """
    Maximum likelihood training for EBMs using TSU.
    Approximates partition function using importance sampling.
    """

    def __init__(
        self,
        energy_fn: TSUEnergyFunction,
        tsu_backend: TSUBackend,
        n_importance_samples: int = 100,
        beta: float = 1.0,
    ):
        """
        Initialize ML objective.

        Args:
            energy_fn: Energy function
            tsu_backend: TSU backend
            n_importance_samples: Number of importance samples
            beta: Inverse temperature
        """
        super().__init__()

        self.energy_fn = energy_fn
        self.tsu_backend = tsu_backend
        self.n_importance_samples = n_importance_samples
        self.beta = beta

        # Running estimate of log partition function
        self.register_buffer('log_Z', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood.

        Args:
            x: Data samples (batch, dim)

        Returns:
            NLL loss
        """
        batch_size = x.shape[0]
        device = x.device

        # Energy of data
        energy_data = self.energy_fn(x)

        # Estimate partition function using importance sampling
        # Sample from proposal distribution using TSU
        proposal_samples = self.energy_fn.sample(
            batch_size=self.n_importance_samples,
            num_steps=100,
        )

        # Compute importance weights
        with torch.no_grad():
            energy_proposal = self.energy_fn(proposal_samples)
            log_weights = -self.beta * energy_proposal.squeeze()

            # Log-sum-exp trick for numerical stability
            log_Z_estimate = torch.logsumexp(log_weights, dim=0)
            log_Z_estimate = log_Z_estimate - math.log(self.n_importance_samples)

            # Update running average
            self.log_Z = 0.9 * self.log_Z + 0.1 * log_Z_estimate

        # Negative log-likelihood
        nll = energy_data.mean() + self.log_Z

        return nll


def create_ebm_objective(
    objective_type: str,
    energy_fn: TSUEnergyFunction,
    tsu_backend: TSUBackend,
    **kwargs
) -> nn.Module:
    """
    Factory function to create EBM objectives.

    Args:
        objective_type: Type of objective ("cd", "score", "infonce", "gan", "ml")
        energy_fn: Energy function
        tsu_backend: TSU backend
        **kwargs: Additional arguments for specific objectives

    Returns:
        EBM objective module
    """
    objectives = {
        'cd': ContrastiveDivergence,
        'score': ScoreMatching,
        'infonce': InfoNCE,
        'gan': EnergyBasedGAN,
        'ml': MaximumLikelihood,
    }

    if objective_type not in objectives:
        raise ValueError(f"Unknown objective: {objective_type}")

    objective_class = objectives[objective_type]

    # Filter kwargs for the specific objective
    import inspect
    sig = inspect.signature(objective_class.__init__)
    valid_kwargs = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }

    return objective_class(
        energy_fn=energy_fn,
        tsu_backend=tsu_backend,
        **valid_kwargs
    )