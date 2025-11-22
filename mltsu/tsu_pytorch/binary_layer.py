"""
TSU Binary Layer for PyTorch
Provides binary sampling layers that integrate with PyTorch autograd
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from ..tsu_core.interfaces import TSUBackend
from .bridge import numpy_to_torch, torch_to_numpy


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-through estimator for gradient flow through discrete sampling.
    Forward: returns discrete samples
    Backward: passes gradients straight through
    """

    @staticmethod
    def forward(ctx, input_logits, samples):
        """
        Forward pass: return samples while saving logits for backward.

        Args:
            input_logits: Continuous logits that produce the samples
            samples: Discrete samples from TSU

        Returns:
            samples (with gradient tracking enabled)
        """
        ctx.save_for_backward(input_logits)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: pass gradient straight through.

        Args:
            grad_output: Gradient w.r.t. samples

        Returns:
            Gradient w.r.t. logits (straight through), None for samples
        """
        input_logits, = ctx.saved_tensors

        # Optional: Apply sigmoid derivative for better gradient flow
        # sigmoid_deriv = torch.sigmoid(input_logits) * (1 - torch.sigmoid(input_logits))
        # grad_input = grad_output * sigmoid_deriv

        # Simple straight-through
        grad_input = grad_output

        return grad_input, None


class TSUBinaryLayer(nn.Module):
    """
    Binary sampling layer using TSU backend.
    Samples discrete binary states during forward pass while maintaining gradient flow.
    """

    def __init__(
        self,
        tsu_backend: TSUBackend,
        beta: float = 1.0,
        num_steps: int = 1,
        use_ste: bool = True,
        stochastic_eval: bool = False,
    ):
        """
        Initialize TSU binary layer.

        Args:
            tsu_backend: TSU backend for sampling
            beta: Inverse temperature (higher = more deterministic)
            num_steps: Number of sampling refinement steps
            use_ste: Use straight-through estimator for gradients
            stochastic_eval: Use stochastic sampling during evaluation
        """
        super().__init__()
        self.tsu_backend = tsu_backend
        self.beta = beta
        self.num_steps = num_steps
        self.use_ste = use_ste
        self.stochastic_eval = stochastic_eval

        # Optionally learnable temperature
        self.register_buffer("_beta", torch.tensor(beta))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample binary states from logits.

        Args:
            logits: Input logits of shape (batch, n_bits) or (batch, ..., n_bits)

        Returns:
            Binary samples of same shape as input, values in {0, 1}
        """
        # Store original shape and device
        original_shape = logits.shape
        device = logits.device
        batch_size = logits.shape[0]

        # Flatten to (batch, n_bits) for TSU
        if len(original_shape) > 2:
            logits_flat = logits.reshape(batch_size, -1)
        else:
            logits_flat = logits

        # Determine if we should sample stochastically
        if self.training or self.stochastic_eval:
            # Convert to numpy for TSU backend
            logits_np = torch_to_numpy(logits_flat)

            # Sample using TSU backend
            samples_np = self.tsu_backend.sample_binary_layer(
                logits_np, beta=self.beta, num_steps=self.num_steps
            )

            # Convert back to torch
            samples = numpy_to_torch(samples_np, device=device)

            # Apply straight-through estimator if enabled and training
            if self.training and self.use_ste:
                samples = StraightThroughEstimator.apply(logits_flat, samples)
        else:
            # Deterministic mode: use sigmoid threshold
            probs = torch.sigmoid(self.beta * logits_flat)
            samples = (probs > 0.5).float()

        # Reshape back to original shape
        if len(original_shape) > 2:
            samples = samples.reshape(original_shape)

        return samples

    def sample_multiple(
        self, logits: torch.Tensor, n_samples: int = 10
    ) -> torch.Tensor:
        """
        Generate multiple samples from the same logits.

        Args:
            logits: Input logits (batch, n_bits)
            n_samples: Number of samples per input

        Returns:
            Multiple samples (n_samples, batch, n_bits)
        """
        device = logits.device
        logits_np = torch_to_numpy(logits)

        all_samples = []
        for _ in range(n_samples):
            samples_np = self.tsu_backend.sample_binary_layer(
                logits_np, beta=self.beta, num_steps=self.num_steps
            )
            samples = numpy_to_torch(samples_np, device=device)
            all_samples.append(samples)

        return torch.stack(all_samples)

    def get_probability(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get probability of binary states being 1.

        Args:
            logits: Input logits

        Returns:
            Probabilities in [0, 1]
        """
        return torch.sigmoid(self.beta * logits)

    def extra_repr(self) -> str:
        """String representation of layer parameters."""
        return f"beta={self.beta}, num_steps={self.num_steps}, use_ste={self.use_ste}"


class TSUDropout(nn.Module):
    """
    Dropout using TSU binary sampling.
    More principled than standard dropout - samples from energy-based distribution.
    """

    def __init__(
        self,
        tsu_backend: TSUBackend,
        p: float = 0.5,
        beta: float = 1.0,
    ):
        """
        Initialize TSU dropout.

        Args:
            tsu_backend: TSU backend for sampling
            p: Dropout probability
            beta: Temperature parameter
        """
        super().__init__()
        self.tsu_backend = tsu_backend
        self.p = p
        self.beta = beta

        # Compute logit for desired dropout probability
        # P(drop) = p => P(keep) = 1-p = sigmoid(logit)
        # logit = log((1-p)/p)
        self.keep_logit = np.log((1 - p) / p) if p < 1.0 else -10.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TSU-based dropout.

        Args:
            x: Input tensor

        Returns:
            Dropped out tensor
        """
        if not self.training:
            return x

        # Create logits for dropout mask
        logits = torch.full(x.shape, self.keep_logit, device=x.device)
        logits_np = torch_to_numpy(logits)

        # Sample dropout mask
        mask_np = self.tsu_backend.sample_binary_layer(
            logits_np, beta=self.beta, num_steps=1
        )
        mask = numpy_to_torch(mask_np, device=x.device)

        # Scale and apply mask
        return x * mask / (1 - self.p)


class GumbelSoftmaxComparison(nn.Module):
    """
    Comparison layer: Gumbel-Softmax vs TSU sampling.
    Useful for benchmarking and analysis.
    """

    def __init__(
        self,
        tsu_backend: Optional[TSUBackend] = None,
        temperature: float = 1.0,
        hard: bool = True,
    ):
        """
        Initialize comparison layer.

        Args:
            tsu_backend: Optional TSU backend (if None, only Gumbel-Softmax)
            temperature: Temperature for both methods
            hard: Use hard (discrete) samples
        """
        super().__init__()
        self.tsu_backend = tsu_backend
        self.temperature = temperature
        self.hard = hard

    def forward(
        self, logits: torch.Tensor, method: str = "gumbel"
    ) -> torch.Tensor:
        """
        Sample using specified method.

        Args:
            logits: Input logits (batch, n_classes)
            method: 'gumbel' or 'tsu'

        Returns:
            Samples (one-hot if hard=True)
        """
        if method == "gumbel":
            return torch.nn.functional.gumbel_softmax(
                logits, tau=self.temperature, hard=self.hard
            )
        elif method == "tsu" and self.tsu_backend is not None:
            # Convert to binary sampling problem
            # For each class, sample if it's selected
            device = logits.device
            logits_np = torch_to_numpy(logits)

            samples_np = self.tsu_backend.sample_binary_layer(
                logits_np, beta=1.0 / self.temperature, num_steps=1
            )
            samples = numpy_to_torch(samples_np, device=device)

            # Normalize to ensure one-hot (select max if needed)
            if self.hard:
                # Make one-hot by selecting highest probability
                max_idx = samples.argmax(dim=-1, keepdim=True)
                samples = torch.zeros_like(samples)
                samples.scatter_(-1, max_idx, 1)

            return samples
        else:
            raise ValueError(f"Unknown method: {method}")