"""
P-bit Optimizer: Enhanced optimizer with probabilistic bit dynamics.
Adds thermodynamic noise perturbations to gradients for better exploration.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import math

from ..tsu_core.interfaces import TSUBackend
from ..tsu_pytorch.noise import TSUGaussianNoise


@dataclass
class PbitOptimizerConfig:
    """Configuration for P-bit optimizer."""
    base_lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    # P-bit specific parameters
    pbit_temperature: float = 1.0  # Controls noise level
    pbit_momentum: float = 0.9  # Momentum for P-bit dynamics
    use_gradient_noise: bool = True
    use_weight_noise: bool = False
    noise_schedule: str = 'constant'  # 'constant', 'linear', 'cosine'
    noise_warmup_steps: int = 100
    min_temperature: float = 0.1
    max_temperature: float = 2.0


class PbitOptimizer:
    """
    P-bit optimizer wrapper that adds thermodynamic perturbations to any PyTorch optimizer.

    This optimizer enhances standard optimization with P-bit dynamics:
    1. Gradient noise injection for better exploration
    2. Temperature-controlled perturbations
    3. Energy-aware parameter updates
    4. Simulated annealing schedules
    """

    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer,
        tsu_backend: TSUBackend,
        config: Optional[PbitOptimizerConfig] = None,
    ):
        """
        Initialize P-bit optimizer.

        Args:
            params: Model parameters to optimize
            base_optimizer: Base PyTorch optimizer (e.g., Adam, SGD)
            tsu_backend: TSU backend for P-bit noise generation
            config: P-bit optimizer configuration
        """
        self.base_optimizer = base_optimizer
        self.tsu_backend = tsu_backend
        self.config = config or PbitOptimizerConfig()

        # P-bit noise generator
        self.noise_generator = TSUGaussianNoise(tsu_backend, M=12)

        # State tracking
        self.step_count = 0
        self.current_temperature = self.config.pbit_temperature

        # Energy tracking
        self.energy_history = []

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable] = None):
        """
        Perform optimization step with P-bit dynamics.

        Args:
            closure: Optional closure for computing loss
        """
        # Update temperature based on schedule
        self._update_temperature()

        # Apply P-bit perturbations to gradients
        if self.config.use_gradient_noise:
            self._apply_gradient_noise()

        # Apply P-bit perturbations to weights
        if self.config.use_weight_noise:
            self._apply_weight_noise()

        # Perform base optimizer step
        loss = self.base_optimizer.step(closure)

        # Track energy (negative loss as energy)
        if loss is not None:
            self.energy_history.append(-loss.item())

        self.step_count += 1

        return loss

    def _update_temperature(self):
        """Update temperature based on schedule."""
        if self.config.noise_schedule == 'constant':
            self.current_temperature = self.config.pbit_temperature

        elif self.config.noise_schedule == 'linear':
            # Linear decay from max to min temperature
            if self.step_count < self.config.noise_warmup_steps:
                # Warmup phase
                progress = self.step_count / self.config.noise_warmup_steps
                self.current_temperature = (
                    self.config.min_temperature +
                    progress * (self.config.max_temperature - self.config.min_temperature)
                )
            else:
                # Decay phase
                decay_steps = 10000  # Total decay steps
                progress = min(1.0, (self.step_count - self.config.noise_warmup_steps) / decay_steps)
                self.current_temperature = (
                    self.config.max_temperature -
                    progress * (self.config.max_temperature - self.config.min_temperature)
                )

        elif self.config.noise_schedule == 'cosine':
            # Cosine annealing
            period = 1000  # Period of cosine wave
            self.current_temperature = (
                self.config.min_temperature +
                0.5 * (self.config.max_temperature - self.config.min_temperature) *
                (1 + math.cos(math.pi * self.step_count / period))
            )

    def _apply_gradient_noise(self):
        """Apply P-bit noise to gradients."""
        with torch.no_grad():
            for group in self.base_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue

                    # Generate P-bit noise
                    noise = self.noise_generator.sample_like(param.grad)

                    # Scale noise by gradient magnitude and temperature
                    noise_scale = self.current_temperature * torch.abs(param.grad).mean()

                    # Add noise to gradient
                    param.grad.add_(noise * noise_scale)

                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)

    def _apply_weight_noise(self):
        """Apply P-bit noise directly to weights (for exploration)."""
        with torch.no_grad():
            for group in self.base_optimizer.param_groups:
                for param in group['params']:
                    # Generate P-bit noise
                    noise = self.noise_generator.sample_like(param)

                    # Very small noise to weights
                    noise_scale = self.current_temperature * 1e-4

                    # Add noise to weights
                    param.add_(noise * noise_scale)

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict."""
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'step_count': self.step_count,
            'current_temperature': self.current_temperature,
            'energy_history': self.energy_history,
            'config': self.config.__dict__,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.step_count = state_dict['step_count']
        self.current_temperature = state_dict['current_temperature']
        self.energy_history = state_dict['energy_history']

    def get_statistics(self) -> Dict[str, float]:
        """Get optimization statistics."""
        stats = {
            'step_count': self.step_count,
            'current_temperature': self.current_temperature,
            'learning_rate': self.base_optimizer.param_groups[0]['lr'],
        }

        if self.energy_history:
            stats['current_energy'] = self.energy_history[-1]
            stats['avg_energy'] = np.mean(self.energy_history[-100:])
            stats['energy_variance'] = np.var(self.energy_history[-100:])

        return stats


class PbitAdamW(PbitOptimizer):
    """
    P-bit enhanced AdamW optimizer.
    Combines AdamW with P-bit dynamics for medical model training.
    """

    def __init__(
        self,
        params,
        tsu_backend: TSUBackend,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        pbit_temperature: float = 1.0,
        use_gradient_noise: bool = True,
        noise_schedule: str = 'cosine',
        **kwargs
    ):
        """
        Initialize P-bit AdamW optimizer.

        Args:
            params: Model parameters
            tsu_backend: TSU backend for P-bit operations
            lr: Learning rate
            betas: Adam beta parameters
            eps: Adam epsilon
            weight_decay: Weight decay coefficient
            pbit_temperature: P-bit noise temperature
            use_gradient_noise: Whether to apply gradient noise
            noise_schedule: Temperature schedule
            **kwargs: Additional config parameters
        """
        # Create base AdamW optimizer
        base_optimizer = optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )

        # Create P-bit config
        config = PbitOptimizerConfig(
            base_lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            pbit_temperature=pbit_temperature,
            use_gradient_noise=use_gradient_noise,
            noise_schedule=noise_schedule,
            **kwargs
        )

        super().__init__(params, base_optimizer, tsu_backend, config)


class PbitSGD(PbitOptimizer):
    """
    P-bit enhanced SGD optimizer.
    Adds Langevin dynamics to standard SGD.
    """

    def __init__(
        self,
        params,
        tsu_backend: TSUBackend,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        pbit_temperature: float = 1.0,
        use_gradient_noise: bool = True,
        **kwargs
    ):
        """
        Initialize P-bit SGD optimizer.

        Args:
            params: Model parameters
            tsu_backend: TSU backend
            lr: Learning rate
            momentum: SGD momentum
            weight_decay: Weight decay
            pbit_temperature: Noise temperature
            use_gradient_noise: Whether to use gradient noise
            **kwargs: Additional config parameters
        """
        # Create base SGD optimizer
        base_optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # Create P-bit config
        config = PbitOptimizerConfig(
            base_lr=lr,
            weight_decay=weight_decay,
            pbit_temperature=pbit_temperature,
            pbit_momentum=momentum,
            use_gradient_noise=use_gradient_noise,
            **kwargs
        )

        super().__init__(params, base_optimizer, tsu_backend, config)


def create_pbit_optimizer(
    model: torch.nn.Module,
    tsu_backend: TSUBackend,
    optimizer_type: str = 'adamw',
    **kwargs
) -> PbitOptimizer:
    """
    Factory function to create P-bit optimizer.

    Args:
        model: PyTorch model
        tsu_backend: TSU backend
        optimizer_type: Type of optimizer ('adamw', 'sgd')
        **kwargs: Optimizer parameters

    Returns:
        P-bit optimizer instance
    """
    params = model.parameters()

    if optimizer_type.lower() == 'adamw':
        return PbitAdamW(params, tsu_backend, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return PbitSGD(params, tsu_backend, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


class PbitLRScheduler:
    """
    Learning rate scheduler with P-bit temperature coupling.
    Coordinates learning rate and noise temperature schedules.
    """

    def __init__(
        self,
        optimizer: PbitOptimizer,
        schedule_type: str = 'cosine',
        num_warmup_steps: int = 100,
        num_training_steps: int = 10000,
    ):
        """
        Initialize P-bit LR scheduler.

        Args:
            optimizer: P-bit optimizer
            schedule_type: Schedule type ('cosine', 'linear', 'constant')
            num_warmup_steps: Number of warmup steps
            num_training_steps: Total training steps
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.current_step = 0

    def step(self):
        """Update learning rate and temperature."""
        self.current_step += 1

        # Calculate learning rate
        if self.current_step < self.num_warmup_steps:
            # Warmup phase
            lr_scale = self.current_step / self.num_warmup_steps
        else:
            # Main training phase
            progress = (self.current_step - self.num_warmup_steps) / (
                self.num_training_steps - self.num_warmup_steps
            )

            if self.schedule_type == 'cosine':
                lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            elif self.schedule_type == 'linear':
                lr_scale = 1.0 - progress
            else:
                lr_scale = 1.0

        # Update learning rate
        for group in self.optimizer.base_optimizer.param_groups:
            group['lr'] = self.optimizer.config.base_lr * lr_scale

        # Couple temperature to learning rate (inverse relationship)
        self.optimizer.current_temperature = (
            self.optimizer.config.min_temperature +
            (1.0 - lr_scale) * (
                self.optimizer.config.max_temperature -
                self.optimizer.config.min_temperature
            )
        )

    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.base_optimizer.param_groups]