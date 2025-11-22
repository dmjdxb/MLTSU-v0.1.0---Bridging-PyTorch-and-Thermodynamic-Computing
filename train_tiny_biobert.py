#!/usr/bin/env python
"""
Train TinyBioBERT: Medical BERT with P-bit Training
This script demonstrates end-to-end training of TinyBioBERT using thermodynamic computing.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mltsu.tsu_jax_sim.backend import JAXTSUBackend
from mltsu.models.tiny_biobert import (
    TinyBioBERTConfig,
    TinyBioBERTForTokenClassification,
    create_tiny_biobert,
)
from mltsu.training.pbit_trainer import (
    PbitTrainer,
    PbitTrainingConfig,
    create_pbit_trainer,
)
from mltsu.training.medical_dataset import (
    create_medical_dataloaders,
    MedicalTokenizer,
    MedicalNERLabel,
)
from mltsu.uncertainty.medical_uncertainty import (
    MedicalUncertaintyQuantifier,
    create_medical_uncertainty_quantifier,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TinyBioBERT with P-bit optimization")

    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of encoder layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length')

    # P-bit parameters
    parser.add_argument('--pbit_temperature', type=float, default=1.0,
                        help='P-bit temperature for sampling')
    parser.add_argument('--num_pbit_samples', type=int, default=32,
                        help='Number of P-bit samples for attention')
    parser.add_argument('--use_pbit_dropout', action='store_true',
                        help='Use P-bit dropout')
    parser.add_argument('--pbit_noise_schedule', type=str, default='cosine',
                        choices=['constant', 'linear', 'cosine'],
                        help='P-bit noise schedule')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')

    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Load checkpoint from file')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only evaluate, no training')
    parser.add_argument('--demo_mode', action='store_true',
                        help='Run in demo mode with small dataset')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_model_info(model: nn.Module):
    """Print model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    print("=" * 60 + "\n")


def evaluate_model(
    model: nn.Module,
    test_loader,
    uncertainty_quantifier: MedicalUncertaintyQuantifier,
    device: str,
):
    """
    Evaluate model performance and uncertainty calibration.
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Get predictions with uncertainty
            results = uncertainty_quantifier.predict_with_uncertainty(
                input_ids, attention_mask
            )

            predictions = results['predictions']
            confidences = results['confidences']
            uncertainties = results['uncertainties']

            # Store results
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_confidences.append(confidences.cpu())
            all_uncertainties.append(uncertainties.cpu())

    # Concatenate all results
    all_predictions = torch.cat(all_predictions).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    all_confidences = torch.cat(all_confidences).numpy().flatten()
    all_uncertainties = torch.cat(all_uncertainties).numpy().flatten()

    # Filter out padding
    mask = all_labels != -100
    all_predictions = all_predictions[mask]
    all_labels = all_labels[mask]
    all_confidences = all_confidences[mask]
    all_uncertainties = all_uncertainties[mask]

    # Compute metrics
    accuracy = np.mean(all_predictions == all_labels)

    # Compute F1 score for disease entities
    disease_label_ids = [1, 2]  # B-Disease, I-Disease
    disease_predictions = np.isin(all_predictions, disease_label_ids)
    disease_labels = np.isin(all_labels, disease_label_ids)

    tp = np.sum(disease_predictions & disease_labels)
    fp = np.sum(disease_predictions & ~disease_labels)
    fn = np.sum(~disease_predictions & disease_labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calibration metrics
    calibration_metrics = uncertainty_quantifier.evaluate_calibration(
        all_predictions, all_labels, all_confidences
    )

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Disease): {f1:.4f}")
    print(f"Precision (Disease): {precision:.4f}")
    print(f"Recall (Disease): {recall:.4f}")
    print(f"\nCalibration Metrics:")
    print(f"  ECE: {calibration_metrics['ece']:.4f}")
    print(f"  MCE: {calibration_metrics['mce']:.4f}")
    print(f"  Brier Score: {calibration_metrics['brier_score']:.4f}")
    print(f"\nUncertainty Statistics:")
    print(f"  Mean Confidence: {np.mean(all_confidences):.4f}")
    print(f"  Mean Uncertainty: {np.mean(all_uncertainties):.4f}")
    print(f"  Uncertainty Std: {np.std(all_uncertainties):.4f}")
    print("=" * 60 + "\n")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        **calibration_metrics,
    }


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("TINYBIOBERT P-BIT TRAINING")
    print("Thermodynamic Computing for Medical NLP")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device}")
    print(f"P-bit Temperature: {args.pbit_temperature}")
    print(f"P-bit Samples: {args.num_pbit_samples}")
    print("=" * 60 + "\n")

    # Initialize TSU backend
    print("Initializing TSU backend...")
    tsu_backend = JAXTSUBackend(seed=args.seed)
    print("✓ TSU backend ready (JAX simulation)")

    # Create model configuration
    config = TinyBioBERTConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        max_position_embeddings=args.max_seq_length,
        pbit_temperature=args.pbit_temperature,
        num_pbit_samples=args.num_pbit_samples,
        use_pbit_dropout=args.use_pbit_dropout,
        num_labels=MedicalNERLabel.num_labels(),
    )

    # Create model
    print("\nCreating TinyBioBERT model...")
    model = TinyBioBERTForTokenClassification(config, tsu_backend)
    model = model.to(args.device)
    print("✓ Model created")

    # Print model information
    print_model_info(model)

    # Create data loaders
    print("Loading medical NER datasets...")
    train_loader, val_loader, test_loader = create_medical_dataloaders(
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
    )
    print(f"✓ Dataset loaded: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches")

    # Create uncertainty quantifier
    print("\nCreating uncertainty quantifier...")
    uncertainty_quantifier = create_medical_uncertainty_quantifier(
        model,
        n_samples=10,
        calibrate=False,  # Will calibrate after training
    )
    print("✓ Uncertainty quantifier ready")

    # Load checkpoint if provided
    if args.load_checkpoint:
        print(f"\nLoading checkpoint from {args.load_checkpoint}...")
        checkpoint = torch.load(args.load_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Checkpoint loaded")

    # Evaluate only mode
    if args.evaluate_only:
        print("\nEvaluation mode - skipping training")
        evaluate_model(model, test_loader, uncertainty_quantifier, args.device)
        return

    # Create training configuration
    training_config = PbitTrainingConfig(
        num_epochs=args.num_epochs if not args.demo_mode else 2,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation,
        pbit_temperature=args.pbit_temperature,
        pbit_noise_schedule=args.pbit_noise_schedule,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        use_wandb=args.use_wandb,
        save_every_n_steps=500 if not args.demo_mode else 10,
        evaluate_every_n_steps=100 if not args.demo_mode else 5,
    )

    # Create trainer
    print("\nCreating P-bit trainer...")
    trainer = create_pbit_trainer(
        model,
        tsu_backend,
        training_config,
    )
    print("✓ Trainer initialized")

    # Training
    print("\n" + "=" * 60)
    print("STARTING P-BIT TRAINING")
    print("=" * 60)

    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=training_config.num_epochs,
    )

    print("\n✓ Training complete!")

    # Print energy statistics
    energy_stats = trainer.energy_tracker.get_stats()
    print("\n" + "=" * 60)
    print("ENERGY CONSUMPTION STATISTICS")
    print("=" * 60)
    print(f"Total GPU Energy: {energy_stats['gpu_energy_j']:.2e} J")
    print(f"Total P-bit Energy: {energy_stats['pbit_energy_j']:.2e} J")
    print(f"Energy Ratio (P-bit/GPU): {energy_stats['energy_ratio']:.6f}")
    print(f"Estimated Energy Savings: {(1 - energy_stats['energy_ratio']) * 100:.2f}%")
    print("=" * 60)

    # Calibrate uncertainty
    print("\nCalibrating uncertainty quantifier...")
    uncertainty_quantifier.calibrate_temperature(val_loader)

    # Final evaluation
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate_model(model, test_loader, uncertainty_quantifier, args.device)

    # Save final results
    results = {
        'config': config.__dict__,
        'training_args': vars(args),
        'test_metrics': test_metrics,
        'energy_stats': energy_stats,
        'training_history': {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_f1': history['val_f1'],
        }
    }

    results_path = os.path.join(args.checkpoint_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {results_path}")

    # Example predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    # Create tokenizer for demo
    tokenizer = MedicalTokenizer(max_length=args.max_seq_length)

    # Example sentences
    examples = [
        "Patient diagnosed with type 2 diabetes and hypertension",
        "BRCA1 mutation associated with breast cancer risk",
        "Treatment with metformin for diabetes management",
        "COVID-19 pneumonia treated with remdesivir",
    ]

    model.eval()
    for text in examples:
        # Tokenize
        encoded = tokenizer.encode(text)
        input_ids = encoded['input_ids'].unsqueeze(0).to(args.device)
        attention_mask = encoded['attention_mask'].unsqueeze(0).to(args.device)

        # Predict with uncertainty
        with torch.no_grad():
            results = uncertainty_quantifier.predict_with_uncertainty(
                input_ids, attention_mask
            )

        predictions = results['predictions'].squeeze()
        confidences = results['confidences'].squeeze()
        uncertainties = results['uncertainties'].squeeze()

        print(f"\nText: {text}")
        print("Predictions:")

        tokens = tokenizer.tokenize(text)
        for i, token in enumerate(tokens[:10]):  # Show first 10 tokens
            if i + 1 < len(predictions):  # Account for CLS token
                pred_label = MedicalNERLabel.LABELS[predictions[i+1].item()]
                conf = confidences[i+1].item()
                unc = uncertainties[i+1].item()
                print(f"  {token:15} -> {pred_label:12} (conf: {conf:.3f}, unc: {unc:.3f})")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - TINYBIOBERT WITH P-BIT OPTIMIZATION")
    print("=" * 60)
    print("This demonstrates the world's first PyTorch → TSU bridge for medical NLP!")
    print("P-bit computing enables energy-efficient medical AI with calibrated uncertainty.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()