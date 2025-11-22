"""
Progressive P-bit Training Scheduler
Gradually increases P-bit usage during training to improve convergence stability.

This module implements a key insight: starting with mostly deterministic
computation and progressively introducing P-bit stochasticity prevents
early training instability while allowing the model to benefit from
thermodynamic exploration in later stages.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass
import math
from enum import Enum


class ProgressiveScheduleType(Enum):
    """Types of progressive schedules."""
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    STEP = "step"
    WARMUP_CONSTANT = "warmup_constant"
    SIGMOID = "sigmoid"


@dataclass
class ProgressiveConfig:
    """Configuration for progressive P-bit scheduling."""

    # Schedule parameters
    schedule_type: ProgressiveScheduleType = ProgressiveScheduleType.LINEAR
    warmup_steps: int = 1000
    total_steps: int = 10000

    # P-bit usage bounds
    min_pbit_ratio: float = 0.1  # Start with 10% P-bit
    max_pbit_ratio: float = 0.9  # End with 90% P-bit

    # Layer-wise control
    layer_wise: bool = True
    layer_schedule_offset: int = 100  # Delay between layers

    # Component-specific settings
    attention_schedule: bool = True
    dropout_schedule: bool = True
    noise_schedule: bool = True

    # Advanced parameters
    sigmoid_beta: float = 10.0  # Steepness for sigmoid schedule
    step_milestones: List[int] = None  # Steps for step schedule
    step_values: List[float] = None  # Values for step schedule


class ProgressivePbitScheduler:
    """
    Progressive P-bit Training Scheduler.

    Gradually increases P-bit component usage during training to improve
    stability and convergence. Can control individual components and layers.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ProgressiveConfig] = None,
    ):
        """
        Initialize Progressive P-bit Scheduler.

        Args:
            model: TinyBioBERT or similar model
            config: Progressive scheduling configuration
        """
        self.model = model
        self.config = config or ProgressiveConfig()
        self.current_step = 0

        # Track components
        self.pbit_components = self._identify_pbit_components()
        self.component_ratios = {}

        # Statistics
        self.history = {
            'steps': [],
            'global_ratio': [],
            'component_ratios': {},
        }

        # Initialize component ratios
        self._initialize_ratios()

    def _identify_pbit_components(self) -> Dict[str, List[nn.Module]]:
        """Identify all P-bit components in the model."""
        components = {
            'attention': [],
            'dropout': [],
            'noise': [],
            'binary': [],
        }

        # Search for P-bit components
        for name, module in self.model.named_modules():
            # Thermodynamic attention
            if 'attention' in name.lower() and hasattr(module, 'n_samples'):
                components['attention'].append((name, module))

            # P-bit dropout
            elif 'dropout' in name.lower() and hasattr(module, 'tsu_backend'):
                components['dropout'].append((name, module))

            # Noise injection
            elif hasattr(module, 'noise_level'):
                components['noise'].append((name, module))

            # Binary layers
            elif hasattr(module, 'use_ste') and 'binary' in name.lower():
                components['binary'].append((name, module))

        return components

    def _initialize_ratios(self):
        """Initialize component ratios."""
        for component_type in self.pbit_components:
            self.component_ratios[component_type] = self.config.min_pbit_ratio

            # Initialize history tracking
            self.history['component_ratios'][component_type] = []

    def step(self, global_step: Optional[int] = None) -> Dict[str, float]:
        """
        Update P-bit ratios based on current training step.

        Args:
            global_step: Optional global training step (uses internal counter if None)

        Returns:
            Dictionary of current P-bit ratios
        """
        # Update step counter
        if global_step is not None:
            self.current_step = global_step
        else:
            self.current_step += 1

        # Calculate global P-bit ratio
        global_ratio = self._calculate_global_ratio(self.current_step)

        # Update component ratios
        ratios = {}

        if self.config.layer_wise:
            # Layer-wise progressive activation
            ratios = self._update_layer_wise(global_ratio)
        else:
            # Global update
            ratios = self._update_global(global_ratio)

        # Apply ratios to model
        self._apply_ratios(ratios)

        # Track history
        self._update_history(ratios, global_ratio)

        return ratios

    def _calculate_global_ratio(self, step: int) -> float:
        """Calculate global P-bit ratio based on schedule."""
        # Handle warmup phase
        if step < self.config.warmup_steps:
            progress = step / self.config.warmup_steps
        else:
            progress = 1.0

        # Apply schedule type
        if self.config.schedule_type == ProgressiveScheduleType.LINEAR:
            ratio = self._linear_schedule(progress)

        elif self.config.schedule_type == ProgressiveScheduleType.COSINE:
            ratio = self._cosine_schedule(progress)

        elif self.config.schedule_type == ProgressiveScheduleType.EXPONENTIAL:
            ratio = self._exponential_schedule(progress)

        elif self.config.schedule_type == ProgressiveScheduleType.SIGMOID:
            ratio = self._sigmoid_schedule(progress)

        elif self.config.schedule_type == ProgressiveScheduleType.STEP:
            ratio = self._step_schedule(step)

        elif self.config.schedule_type == ProgressiveScheduleType.WARMUP_CONSTANT:
            ratio = self.config.max_pbit_ratio if progress >= 1.0 else self.config.min_pbit_ratio

        else:
            ratio = self.config.min_pbit_ratio

        return ratio

    def _linear_schedule(self, progress: float) -> float:
        """Linear interpolation schedule."""
        return (self.config.min_pbit_ratio +
                (self.config.max_pbit_ratio - self.config.min_pbit_ratio) * progress)

    def _cosine_schedule(self, progress: float) -> float:
        """Cosine annealing schedule."""
        cosine_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return (self.config.min_pbit_ratio +
                (self.config.max_pbit_ratio - self.config.min_pbit_ratio) * cosine_factor)

    def _exponential_schedule(self, progress: float) -> float:
        """Exponential growth schedule."""
        exp_factor = (math.exp(progress) - 1) / (math.e - 1)
        return (self.config.min_pbit_ratio +
                (self.config.max_pbit_ratio - self.config.min_pbit_ratio) * exp_factor)

    def _sigmoid_schedule(self, progress: float) -> float:
        """Sigmoid (S-curve) schedule."""
        x = self.config.sigmoid_beta * (progress - 0.5)
        sigmoid_factor = 1 / (1 + math.exp(-x))
        return (self.config.min_pbit_ratio +
                (self.config.max_pbit_ratio - self.config.min_pbit_ratio) * sigmoid_factor)

    def _step_schedule(self, step: int) -> float:
        """Step-wise schedule with predefined milestones."""
        if not self.config.step_milestones or not self.config.step_values:
            return self.config.min_pbit_ratio

        for i, milestone in enumerate(self.config.step_milestones):
            if step < milestone:
                return self.config.step_values[i] if i < len(self.config.step_values) else self.config.min_pbit_ratio

        return self.config.max_pbit_ratio

    def _update_layer_wise(self, global_ratio: float) -> Dict[str, float]:
        """Update ratios with layer-wise progression."""
        ratios = {}

        # Attention layers (progressive by depth)
        if self.config.attention_schedule:
            attention_components = self.pbit_components.get('attention', [])
            num_layers = len(attention_components)

            for i, (name, module) in enumerate(attention_components):
                # Deeper layers get P-bit activation later
                layer_delay = i * self.config.layer_schedule_offset
                adjusted_step = max(0, self.current_step - layer_delay)
                layer_progress = min(1.0, adjusted_step / self.config.warmup_steps)

                layer_ratio = (self.config.min_pbit_ratio +
                              (global_ratio - self.config.min_pbit_ratio) * layer_progress)

                ratios[f'attention_{i}'] = layer_ratio

        # Dropout (uniform progression)
        if self.config.dropout_schedule:
            ratios['dropout'] = global_ratio

        # Noise (faster progression for regularization)
        if self.config.noise_schedule:
            noise_ratio = min(self.config.max_pbit_ratio, global_ratio * 1.2)
            ratios['noise'] = noise_ratio

        return ratios

    def _update_global(self, global_ratio: float) -> Dict[str, float]:
        """Update all components with the same ratio."""
        ratios = {}

        if self.config.attention_schedule:
            ratios['attention'] = global_ratio

        if self.config.dropout_schedule:
            ratios['dropout'] = global_ratio

        if self.config.noise_schedule:
            ratios['noise'] = global_ratio

        return ratios

    def _apply_ratios(self, ratios: Dict[str, float]):
        """Apply P-bit ratios to model components."""

        # Apply to attention layers
        if 'attention' in ratios:
            for name, module in self.pbit_components.get('attention', []):
                if hasattr(module, 'n_samples'):
                    # Scale number of samples based on ratio
                    base_samples = getattr(module, 'base_n_samples', 32)
                    module.n_samples = max(1, int(base_samples * ratios['attention']))

                # Enable/disable TSU based on ratio
                if hasattr(module, 'use_tsu'):
                    module.use_tsu = np.random.rand() < ratios['attention']

        # Apply layer-wise attention ratios
        for key in ratios:
            if key.startswith('attention_'):
                layer_idx = int(key.split('_')[1])
                if layer_idx < len(self.pbit_components.get('attention', [])):
                    name, module = self.pbit_components['attention'][layer_idx]

                    if hasattr(module, 'n_samples'):
                        base_samples = getattr(module, 'base_n_samples', 32)
                        module.n_samples = max(1, int(base_samples * ratios[key]))

                    if hasattr(module, 'use_tsu'):
                        module.use_tsu = np.random.rand() < ratios[key]

        # Apply to dropout
        if 'dropout' in ratios:
            for name, module in self.pbit_components.get('dropout', []):
                if hasattr(module, 'p'):
                    # Scale dropout rate based on ratio
                    base_p = getattr(module, 'base_p', 0.1)
                    module.p = base_p * ratios['dropout']

        # Apply to noise
        if 'noise' in ratios:
            for name, module in self.pbit_components.get('noise', []):
                if hasattr(module, 'noise_level'):
                    base_noise = getattr(module, 'base_noise_level', 0.1)
                    module.noise_level = base_noise * ratios['noise']

    def _update_history(self, ratios: Dict[str, float], global_ratio: float):
        """Track scheduling history."""
        self.history['steps'].append(self.current_step)
        self.history['global_ratio'].append(global_ratio)

        for key, value in ratios.items():
            if key not in self.history['component_ratios']:
                self.history['component_ratios'][key] = []
            self.history['component_ratios'][key].append(value)

    def get_current_ratios(self) -> Dict[str, float]:
        """Get current P-bit ratios."""
        return self.component_ratios

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            'current_step': self.current_step,
            'warmup_progress': min(1.0, self.current_step / self.config.warmup_steps),
            'total_progress': min(1.0, self.current_step / self.config.total_steps),
            'num_attention_layers': len(self.pbit_components.get('attention', [])),
            'num_dropout_layers': len(self.pbit_components.get('dropout', [])),
            'current_ratios': self.component_ratios,
        }

        # Add average ratios from history
        if self.history['global_ratio']:
            stats['avg_global_ratio'] = np.mean(self.history['global_ratio'][-100:])

        return stats

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0
        self._initialize_ratios()
        self.history = {
            'steps': [],
            'global_ratio': [],
            'component_ratios': {},
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'component_ratios': self.component_ratios,
            'history': self.history,
            'config': self.config.__dict__,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.component_ratios = state_dict['component_ratios']
        self.history = state_dict['history']


class AdaptivePbitScheduler(ProgressivePbitScheduler):
    """
    Adaptive P-bit scheduler that adjusts based on training dynamics.

    Monitors loss and gradient statistics to automatically adjust
    P-bit usage for optimal training stability.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ProgressiveConfig] = None,
        adaptation_rate: float = 0.01,
    ):
        super().__init__(model, config)
        self.adaptation_rate = adaptation_rate

        # Tracking for adaptation
        self.loss_history = []
        self.gradient_history = []
        self.stability_score = 1.0

    def adapt(
        self,
        loss: float,
        gradients: Optional[Dict[str, float]] = None
    ):
        """
        Adapt P-bit ratios based on training dynamics.

        Args:
            loss: Current training loss
            gradients: Optional gradient statistics
        """
        self.loss_history.append(loss)

        if gradients:
            self.gradient_history.append(gradients)

        # Calculate stability score
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            loss_variance = np.var(recent_losses)
            loss_trend = np.polyfit(range(10), recent_losses, 1)[0]

            # High variance or increasing loss indicates instability
            if loss_variance > 1.0 or loss_trend > 0:
                self.stability_score *= (1 - self.adaptation_rate)
            else:
                self.stability_score *= (1 + self.adaptation_rate)

            self.stability_score = np.clip(self.stability_score, 0.5, 1.5)

        # Adjust max P-bit ratio based on stability
        adjusted_max = self.config.max_pbit_ratio * self.stability_score
        self.config.max_pbit_ratio = np.clip(adjusted_max, 0.1, 0.95)


def create_progressive_scheduler(
    model: nn.Module,
    total_training_steps: int,
    warmup_ratio: float = 0.3,
    min_pbit: float = 0.1,
    max_pbit: float = 0.9,
    schedule_type: str = "linear",
    layer_wise: bool = True,
    adaptive: bool = False,
) -> Union[ProgressivePbitScheduler, AdaptivePbitScheduler]:
    """
    Factory function to create progressive P-bit scheduler.

    Args:
        model: Model with P-bit components
        total_training_steps: Total number of training steps
        warmup_ratio: Fraction of steps for warmup
        min_pbit: Minimum P-bit usage ratio
        max_pbit: Maximum P-bit usage ratio
        schedule_type: Type of schedule ('linear', 'cosine', etc.)
        layer_wise: Enable layer-wise progression
        adaptive: Use adaptive scheduler

    Returns:
        Configured scheduler instance
    """
    config = ProgressiveConfig(
        schedule_type=ProgressiveScheduleType(schedule_type),
        warmup_steps=int(total_training_steps * warmup_ratio),
        total_steps=total_training_steps,
        min_pbit_ratio=min_pbit,
        max_pbit_ratio=max_pbit,
        layer_wise=layer_wise,
    )

    if adaptive:
        return AdaptivePbitScheduler(model, config)
    else:
        return ProgressivePbitScheduler(model, config)