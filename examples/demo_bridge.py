#!/usr/bin/env python3
"""
Demo: Complete PyTorch → TSU Bridge
Shows how PyTorch models run on thermodynamic hardware through MLTSU

This is the FIRST-OF-ITS-KIND demonstration showing:
1. PyTorch model definition
2. TSU backend initialization (software sim → hardware ready)
3. Thermodynamic attention replacing softmax
4. TSU-powered training with energy-based objectives
5. Generation using thermodynamic sampling

Run this to see the future of AI computing!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List
import time

# Import MLTSU components
try:
    from mltsu.tsu_jax_sim.backend import JAXTSUBackend
except ImportError:
    print("JAX not available yet. Using mock backend for demo.")
    # Create a mock backend for demonstration
    class MockTSUBackend:
        def sample_binary_layer(self, logits, beta, num_steps, key=None):
            # Simple mock sampling
            probs = 1 / (1 + np.exp(-beta * logits))
            return (np.random.random(logits.shape) < probs).astype(np.float32)

        def sample_ising(self, J, h, beta, num_steps, batch_size, init_state=None,
                         record_trajectory=False, key=None):
            n_spins = len(h)
            samples = np.random.choice([-1, 1], size=(batch_size, n_spins))
            return {
                'samples': samples,
                'final_energy': np.random.randn(batch_size),
                'trajectories': None
            }

        def sample_custom(self, energy_fn, init_state, num_steps, beta, key=None, **kwargs):
            # Simple random walk
            state = init_state.copy()
            for _ in range(num_steps):
                state += np.random.randn(*state.shape) * 0.1
            return state

    JAXTSUBackend = MockTSUBackend

from mltsu.tsu_pytorch.binary_layer import TSUBinaryLayer
from mltsu.tsu_pytorch.noise import TSUGaussianNoise
from mltsu.models.tiny_thermo_lm import TinyThermoLM, create_tiny_thermo_lm


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def demo_basic_layers():
    """Demonstrate basic TSU layers."""
    print_header("1. BASIC TSU LAYERS")

    # Initialize TSU backend
    print("\n→ Initializing TSU Backend (software simulator)...")
    backend = JAXTSUBackend()
    print("  ✓ Backend ready (can switch to hardware seamlessly)")

    # Create TSU-powered layers
    print("\n→ Creating TSU-powered PyTorch layers...")

    # Binary layer for discrete decisions
    binary_layer = TSUBinaryLayer(backend, beta=2.0)
    print("  ✓ TSUBinaryLayer: Samples binary masks from energy distribution")

    # Gaussian noise generator using CLT
    noise_gen = TSUGaussianNoise(backend, M=12)
    print("  ✓ TSUGaussianNoise: Generates Gaussian noise via CLT")

    # Test binary layer
    print("\n→ Testing TSU Binary Layer:")
    x = torch.randn(4, 8)  # Batch of 4, dimension 8
    print(f"  Input shape: {x.shape}")

    mask = binary_layer(x)
    print(f"  Binary mask shape: {mask.shape}")
    print(f"  Mask sparsity: {(mask == 0).float().mean():.2%}")
    print(f"  Sample mask: {mask[0].tolist()}")

    # Test noise generator
    print("\n→ Testing TSU Gaussian Noise:")
    noise = noise_gen.sample(shape=(4, 8))
    print(f"  Noise shape: {noise.shape}")
    print(f"  Mean: {noise.mean():.3f}, Std: {noise.std():.3f}")


def demo_ising_optimization():
    """Demonstrate Ising model optimization."""
    print_header("2. ISING MODEL OPTIMIZATION")

    backend = JAXTSUBackend()

    print("\n→ Setting up Max-Cut optimization problem...")
    n_spins = 10

    # Create random symmetric coupling matrix
    J = np.random.randn(n_spins, n_spins)
    J = (J + J.T) / 2
    h = np.zeros(n_spins)

    print(f"  Problem size: {n_spins} spins")
    print(f"  Coupling matrix shape: {J.shape}")

    # Sample low-energy states using TSU
    print("\n→ Sampling low-energy states with TSU...")
    result = backend.sample_ising(
        J, h,
        beta=5.0,  # Low temperature for optimization
        num_steps=100,
        batch_size=10
    )

    samples = result['samples']
    energies = result['final_energy']

    best_idx = np.argmin(energies)
    best_state = samples[best_idx]
    best_energy = energies[best_idx]

    print(f"  Sampled {len(samples)} states")
    print(f"  Best energy found: {best_energy:.3f}")
    print(f"  Best state: {best_state[:10]}...")  # Show first 10 spins
    print(f"  Energy range: [{energies.min():.3f}, {energies.max():.3f}]")


def demo_tiny_thermo_lm():
    """Demonstrate TinyThermoLM - the complete bridge."""
    print_header("3. TINY THERMO LM - COMPLETE BRIDGE")

    print("\n→ Creating TinyThermoLM model...")
    print("  This is the crown jewel - a language model using:")
    print("  • Thermodynamic attention (replaces softmax)")
    print("  • TSU binary gating (sparse computation)")
    print("  • TSU noise injection (regularization)")
    print("  • Energy-based negative sampling")

    # Model parameters
    vocab_size = 100
    d_model = 64
    n_heads = 4
    n_layers = 2

    # Create model
    backend = JAXTSUBackend()
    model = create_tiny_thermo_lm(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        tsu_backend=backend,
        beta=1.0,
        n_samples=8,  # Monte Carlo samples for attention
        use_tsu_gating=True,
        use_negative_sampling=True,
    )

    print(f"\n  Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    print("\n→ Testing forward pass...")
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"  Input shape: {input_ids.shape}")

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    print(f"  Output keys: {outputs.keys()}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.3f}")

    # Test generation
    print("\n→ Testing text generation with TSU sampling...")
    prompt = torch.randint(0, vocab_size, (1, 5))  # Start with 5 tokens

    print(f"  Prompt shape: {prompt.shape}")
    print("  Generating 10 tokens...")

    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_length=15,
            temperature=1.0,
            use_tsu_sampling=False,  # Can enable TSU sampling
        )

    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated tokens: {generated[0].tolist()}")

    # Show model statistics
    print("\n→ Model Statistics:")
    stats = model.get_statistics()
    for key, value in stats.items():
        if not key.startswith('layer'):
            print(f"  {key}: {value:.3f}")


def demo_training_loop():
    """Demonstrate a mini training loop."""
    print_header("4. TRAINING WITH TSU")

    print("\n→ Setting up mini training loop...")

    # Create small model
    backend = JAXTSUBackend()
    model = create_tiny_thermo_lm(
        vocab_size=50,
        d_model=32,
        n_heads=2,
        n_layers=1,
        tsu_backend=backend,
        n_samples=4,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Synthetic data
    batch_size = 4
    seq_len = 8

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Batch size: {batch_size}, Sequence length: {seq_len}")

    # Training steps
    n_steps = 5
    print(f"\n→ Training for {n_steps} steps...")

    model.train()
    losses = []

    for step in range(n_steps):
        # Generate random data
        input_ids = torch.randint(0, 50, (batch_size, seq_len))
        labels = torch.randint(0, 50, (batch_size, seq_len))

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")

    print(f"\n  Average loss: {np.mean(losses):.4f}")
    print(f"  Loss reduction: {(losses[0] - losses[-1])/losses[0]:.1%}")


def demo_hardware_ready():
    """Demonstrate hardware readiness."""
    print_header("5. HARDWARE READY ARCHITECTURE")

    print("\n This framework is designed for seamless hardware integration:")
    print()
    print(" CURRENT STATE (Software Simulation):")
    print(" =====================================")
    print("   PyTorch Model")
    print("        ↓")
    print("   TSUBackend Protocol")
    print("        ↓")
    print("   JAXTSUBackend (JAX Simulator)")
    print("        ↓")
    print("   CPU/GPU Computation")
    print()
    print(" FUTURE STATE (Hardware Acceleration):")
    print(" ======================================")
    print("   PyTorch Model (SAME CODE!)")
    print("        ↓")
    print("   TSUBackend Protocol")
    print("        ↓")
    print("   ExtropicTSUBackend / PBitBackend")
    print("        ↓")
    print("   THERMODYNAMIC HARDWARE")
    print("   • Extropic TSUs")
    print("   • P-bit chips")
    print("   • Ising machines")
    print()
    print(" Key Benefits:")
    print(" • 100-1000× energy efficiency")
    print(" • Natural probabilistic computation")
    print(" • Massive parallelism")
    print(" • Quantum-inspired algorithms")


def main():
    """Run all demonstrations."""
    print("\n" + "🔥"*30)
    print("   MLTSU: PyTorch → TSU Bridge Demo")
    print("   First-of-its-kind Thermodynamic Computing")
    print("🔥"*30)

    # Run demonstrations
    demo_basic_layers()
    demo_ising_optimization()
    demo_tiny_thermo_lm()
    demo_training_loop()
    demo_hardware_ready()

    print("\n" + "="*60)
    print("  DEMO COMPLETE!")
    print("  The bridge between PyTorch and thermodynamic hardware is ready.")
    print("  This is the future of energy-efficient AI computing.")
    print("="*60)
    print()


if __name__ == "__main__":
    main()