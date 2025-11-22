"""
P-bit Trainer: Training infrastructure for TinyBioBERT with thermodynamic computing.
Handles training loops, evaluation, and energy tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
import time
import os
from pathlib import Path
from tqdm import tqdm
import json

from ..tsu_core.interfaces import TSUBackend
from .pbit_optimizer import PbitOptimizer, PbitLRScheduler


@dataclass
class PbitTrainingConfig:
    """Configuration for P-bit training."""
    # Training parameters
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # P-bit specific
    pbit_temperature: float = 1.0
    pbit_noise_schedule: str = 'cosine'
    energy_tracking: bool = True
    uncertainty_sampling: bool = True
    num_uncertainty_samples: int = 5

    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_every_n_steps: int = 500
    evaluate_every_n_steps: int = 100

    # Logging
    log_every_n_steps: int = 10
    use_wandb: bool = False
    wandb_project: str = 'tiny-biobert-pbit'

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = 'f1'


class EnergyTracker:
    """
    Tracks energy consumption during training.
    Estimates both computational and P-bit energy.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.total_flops = 0
        self.total_pbit_ops = 0
        self.total_time = 0.0
        self.gpu_energy_j = 0.0
        self.pbit_energy_j = 0.0

    def update(
        self,
        model_flops: int = 0,
        pbit_ops: int = 0,
        elapsed_time: float = 0.0,
        gpu_power_w: float = 100.0,  # Estimated GPU power
    ):
        """Update energy counters."""
        self.total_flops += model_flops
        self.total_pbit_ops += pbit_ops
        self.total_time += elapsed_time

        # Estimate GPU energy (J = W * s)
        self.gpu_energy_j += gpu_power_w * elapsed_time

        # Estimate P-bit energy (1e-15 J per operation)
        self.pbit_energy_j += pbit_ops * 1e-15

    def get_stats(self) -> Dict[str, float]:
        """Get energy statistics."""
        return {
            'total_flops': self.total_flops,
            'total_pbit_ops': self.total_pbit_ops,
            'total_time_s': self.total_time,
            'gpu_energy_j': self.gpu_energy_j,
            'pbit_energy_j': self.pbit_energy_j,
            'energy_ratio': self.pbit_energy_j / max(self.gpu_energy_j, 1e-10),
        }


class PbitTrainer:
    """
    Trainer for models with P-bit components.
    Handles training, evaluation, and energy tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: PbitOptimizer,
        tsu_backend: TSUBackend,
        config: Optional[PbitTrainingConfig] = None,
    ):
        """
        Initialize P-bit trainer.

        Args:
            model: Model to train
            optimizer: P-bit optimizer
            tsu_backend: TSU backend for P-bit operations
            config: Training configuration
        """
        self.model = model
        self.optimizer = optimizer
        self.tsu_backend = tsu_backend
        self.config = config or PbitTrainingConfig()

        # Move model to device
        self.model = self.model.to(self.config.device)

        # Energy tracking
        self.energy_tracker = EnergyTracker()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('-inf')
        self.patience_counter = 0

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize scheduler
        self.scheduler = None

        # Metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'energy_stats': [],
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            # Track time
            start_time = time.time()

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss']

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            # Track energy
            elapsed_time = time.time() - start_time
            batch_size = batch['input_ids'].size(0)
            seq_length = batch['input_ids'].size(1)

            # Estimate operations
            model_flops = self._estimate_flops(batch_size, seq_length)
            pbit_ops = self._estimate_pbit_ops(batch_size, seq_length)

            self.energy_tracker.update(
                model_flops=model_flops,
                pbit_ops=pbit_ops,
                elapsed_time=elapsed_time,
            )

            # Update metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'lr': self.optimizer.base_optimizer.param_groups[0]['lr'],
                'temp': self.optimizer.current_temperature,
            })

            # Log metrics
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_metrics({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.base_optimizer.param_groups[0]['lr'],
                    'pbit_temperature': self.optimizer.current_temperature,
                })

            # Save checkpoint
            if self.global_step % self.config.save_every_n_steps == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

        return {
            'train_loss': total_loss / num_batches,
            'energy_stats': self.energy_tracker.get_stats(),
        }

    def evaluate(
        self,
        val_loader: DataLoader,
        compute_uncertainty: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            val_loader: Validation data loader
            compute_uncertainty: Whether to compute uncertainty estimates

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                # Move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                if compute_uncertainty and self.config.uncertainty_sampling:
                    # Multiple forward passes for uncertainty
                    logits_list = []
                    for _ in range(self.config.num_uncertainty_samples):
                        outputs = self.model(**batch)
                        logits_list.append(outputs['logits'])

                    # Average logits
                    logits = torch.stack(logits_list).mean(dim=0)

                    # Compute uncertainty (variance across samples)
                    uncertainty = torch.stack(logits_list).var(dim=0).mean()
                else:
                    # Single forward pass
                    outputs = self.model(**batch)
                    logits = outputs['logits']
                    uncertainty = None

                # Compute loss
                if 'labels' in batch:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, logits.size(-1)),
                        batch['labels'].view(-1)
                    )
                    total_loss += loss.item()

                    # Get predictions
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.append(predictions.cpu())
                    all_labels.append(batch['labels'].cpu())

                num_batches += 1

        # Compute metrics
        metrics = {
            'val_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        }

        if all_predictions:
            # Flatten predictions and labels
            all_predictions = torch.cat(all_predictions).numpy()
            all_labels = torch.cat(all_labels).numpy()

            # Compute F1 score (simplified)
            metrics['val_f1'] = self._compute_f1(all_predictions, all_labels)
            metrics['val_accuracy'] = np.mean(all_predictions == all_labels)

        if uncertainty is not None:
            metrics['val_uncertainty'] = uncertainty.item()

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train

        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.num_epochs

        # Initialize scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = PbitLRScheduler(
            self.optimizer,
            schedule_type=self.config.pbit_noise_schedule,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"Starting P-bit training for {num_epochs} epochs")
        print(f"Device: {self.config.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            self.metrics_history['train_loss'].append(train_metrics['train_loss'])

            print(f"Epoch {epoch} - Train Loss: {train_metrics['train_loss']:.4f}")

            # Evaluate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.metrics_history['val_loss'].append(val_metrics['val_loss'])

                print(f"Epoch {epoch} - Val Loss: {val_metrics['val_loss']:.4f}")

                if 'val_f1' in val_metrics:
                    self.metrics_history['val_f1'].append(val_metrics['val_f1'])
                    print(f"Epoch {epoch} - Val F1: {val_metrics['val_f1']:.4f}")

                    # Early stopping
                    if self._check_early_stopping(val_metrics):
                        print(f"Early stopping at epoch {epoch}")
                        break

            # Log energy stats
            energy_stats = self.energy_tracker.get_stats()
            self.metrics_history['energy_stats'].append(energy_stats)

            print(f"Energy stats: GPU={energy_stats['gpu_energy_j']:.2e}J, "
                  f"P-bit={energy_stats['pbit_energy_j']:.2e}J, "
                  f"Ratio={energy_stats['energy_ratio']:.4f}")

        # Save final checkpoint
        self.save_checkpoint('final_model.pt')

        return self.metrics_history

    def _estimate_flops(self, batch_size: int, seq_length: int) -> int:
        """Estimate FLOPs for standard computation."""
        # Simplified estimation for BERT
        hidden_size = 256  # From TinyBioBERTConfig
        num_layers = 4
        num_heads = 4

        # Attention FLOPs: O(batch * heads * seq^2 * dim)
        attention_flops = batch_size * num_heads * seq_length**2 * (hidden_size // num_heads)

        # FFN FLOPs: O(batch * seq * hidden * intermediate)
        ffn_flops = batch_size * seq_length * hidden_size * 1024  # intermediate_size

        total_flops = num_layers * (attention_flops + ffn_flops)
        return int(total_flops)

    def _estimate_pbit_ops(self, batch_size: int, seq_length: int) -> int:
        """Estimate P-bit operations."""
        # P-bit operations in attention
        num_layers = 4
        num_heads = 4
        num_samples = 32  # From config

        pbit_ops = batch_size * num_layers * num_heads * seq_length * num_samples
        return int(pbit_ops)

    def _compute_f1(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute F1 score (simplified)."""
        # Filter out padding tokens (label = 0 or -100)
        mask = (labels > 0) & (labels != -100)
        predictions = predictions[mask]
        labels = labels[mask]

        if len(labels) == 0:
            return 0.0

        # Compute precision and recall
        correct = (predictions == labels).sum()
        precision = correct / len(predictions) if len(predictions) > 0 else 0
        recall = correct / len(labels) if len(labels) > 0 else 0

        # Compute F1
        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)

    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check early stopping criteria."""
        metric = metrics.get(f'val_{self.config.early_stopping_metric}', 0.0)

        if metric > self.best_metric:
            self.best_metric = metric
            self.patience_counter = 0
            # Save best model
            self.save_checkpoint('best_model.pt')
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb or console."""
        if self.config.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=self.global_step)
            except ImportError:
                pass

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'metrics_history': self.metrics_history,
            'best_metric': self.best_metric,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.metrics_history = checkpoint['metrics_history']
        self.best_metric = checkpoint.get('best_metric', float('-inf'))

        print(f"Checkpoint loaded from {checkpoint_path}")


def create_pbit_trainer(
    model: nn.Module,
    tsu_backend: TSUBackend,
    config: Optional[PbitTrainingConfig] = None,
    **optimizer_kwargs
) -> PbitTrainer:
    """
    Factory function to create P-bit trainer.

    Args:
        model: Model to train
        tsu_backend: TSU backend
        config: Training configuration
        **optimizer_kwargs: Optimizer parameters

    Returns:
        P-bit trainer instance
    """
    from .pbit_optimizer import PbitAdamW

    config = config or PbitTrainingConfig()

    # Create optimizer
    optimizer = PbitAdamW(
        model.parameters(),
        tsu_backend,
        lr=config.learning_rate,
        pbit_temperature=config.pbit_temperature,
        noise_schedule=config.pbit_noise_schedule,
        **optimizer_kwargs
    )

    # Create trainer
    trainer = PbitTrainer(model, optimizer, tsu_backend, config)

    return trainer