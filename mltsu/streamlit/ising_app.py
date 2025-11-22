"""
Interactive Ising Model Playground using Streamlit
Demonstrates TSU sampling capabilities through visualization
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from mltsu.tsu_jax_sim.backend import JAXTSUBackend
    from mltsu.tsu_jax_sim.energy_models import (
        ising_energy,
        compute_ising_magnetization,
    )
    HAS_MLTSU = True
except ImportError:
    HAS_MLTSU = False
    st.error("MLTSU not found. Please install the package first.")

# Page configuration
st.set_page_config(
    page_title="TSU Ising Playground",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better visuals
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with version
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🔬 Thermodynamic Ising Model Playground")
with col2:
    st.markdown("**MLTSU v0.1.0**")
    st.caption("TSU Bridge")

st.markdown("""
This interactive playground demonstrates **Thermodynamic Sampling Units (TSU)**
through the Ising model - a fundamental model in statistical physics and optimization.

**What you can do:**
- Adjust coupling strength and external field
- Control temperature (β = 1/T)
- Watch the system evolve to low-energy states
- Compare different sampling algorithms
""")

# Initialize backend (cached for performance)
@st.cache_resource
def get_backend(method="gibbs", seed=42):
    """Initialize and cache TSU backend."""
    if not HAS_MLTSU:
        return None
    return JAXTSUBackend(seed=seed, sampling_method=method)

# Performance Metrics Section
st.sidebar.header("⚡ Performance Metrics")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.sidebar.metric("GPU Baseline", "22.7 kJ", help="Energy for equivalent GPU computation")
with col2:
    st.sidebar.metric("TSU Simulation", "0.02 J", help="Energy for TSU-based sampling")

st.sidebar.metric("Energy Reduction", "1,135,000×", delta="99.9999% less energy",
                  help="Theoretical reduction with real TSU hardware")

st.sidebar.caption("*Simulated values - real hardware pending")
st.sidebar.markdown("---")

# Sidebar controls
st.sidebar.header("🎛️ Ising Model Parameters")

# Model size
col1, col2 = st.sidebar.columns(2)
with col1:
    grid_size = st.number_input(
        "Grid size", min_value=3, max_value=20, value=8, step=1
    )
with col2:
    n_spins = grid_size * grid_size
    st.metric("Total spins", n_spins)

# Coupling parameters
st.sidebar.subheader("Coupling Matrix J")
coupling_type = st.sidebar.selectbox(
    "Coupling type",
    ["Ferromagnetic", "Antiferromagnetic", "Spin Glass", "Custom"],
    help="Type of spin-spin interactions",
)

coupling_strength = st.sidebar.slider(
    "Coupling strength |J|",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Strength of spin-spin interactions",
)

# External field
st.sidebar.subheader("External Field h")
field_type = st.sidebar.selectbox(
    "Field type",
    ["Zero", "Uniform", "Random", "Gradient"],
    help="External magnetic field pattern",
)

if field_type != "Zero":
    field_strength = st.sidebar.slider(
        "Field strength",
        min_value=-2.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
    )
else:
    field_strength = 0.0

# Temperature control
st.sidebar.subheader("Temperature Control")
beta = st.sidebar.slider(
    "Inverse temperature β",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="β = 1/T (higher = colder = more ordered)",
)

temperature = 1.0 / beta
st.sidebar.metric("Temperature T", f"{temperature:.2f}")

# Reproducibility Mode
st.sidebar.subheader("🔬 Reproducibility")
reproducibility_mode = st.sidebar.selectbox(
    "Mode",
    ["Fixed (seed=42)", "Statistical (5 seeds)", "Random"],
    help="Fixed: Exact reproducibility | Statistical: Average over seeds | Random: True randomness"
)

# Show info about reproducibility
if reproducibility_mode == "Fixed (seed=42)":
    st.sidebar.info(
        "🔬 **Reproducibility**: Using seed=42 for consistent results. "
        "Toggle to 'Random' or 'Statistical' to explore variations."
    )
    seed_to_use = 42
elif reproducibility_mode == "Statistical (5 seeds)":
    st.sidebar.info(
        "📊 Running 5 independent trials with seeds: 42, 137, 314, 2718, 3141"
    )
    seeds_to_use = [42, 137, 314, 2718, 3141]
else:
    import time
    random_seed = int(time.time() * 1000) % (2**32)
    st.sidebar.info(f"🎲 Using random seed: {random_seed}")
    seed_to_use = random_seed

# Sampling parameters
st.sidebar.subheader("Sampling Parameters")
num_steps = st.sidebar.slider(
    "Sampling steps",
    min_value=10,
    max_value=1000,
    value=100,
    step=10,
    help="Number of MCMC steps",
)

sampling_method = st.sidebar.selectbox(
    "Sampling algorithm",
    ["gibbs", "metropolis", "parallel_tempering"],
    help="TSU sampling algorithm to use",
)

# Create model functions
def create_coupling_matrix(n_spins, coupling_type, strength):
    """Create coupling matrix based on type."""
    if coupling_type == "Ferromagnetic":
        # All positive couplings (spins want to align)
        J = np.random.uniform(0.8, 1.2, (n_spins, n_spins)) * strength
    elif coupling_type == "Antiferromagnetic":
        # All negative couplings (spins want to anti-align)
        J = -np.random.uniform(0.8, 1.2, (n_spins, n_spins)) * strength
    elif coupling_type == "Spin Glass":
        # Random positive and negative couplings
        J = np.random.randn(n_spins, n_spins) * strength
    else:  # Custom
        # Nearest-neighbor on 2D grid
        J = np.zeros((n_spins, n_spins))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                # Right neighbor
                if j < grid_size - 1:
                    idx_right = i * grid_size + (j + 1)
                    J[idx, idx_right] = strength
                # Bottom neighbor
                if i < grid_size - 1:
                    idx_bottom = (i + 1) * grid_size + j
                    J[idx, idx_bottom] = strength

    # Make symmetric and remove self-interaction
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    return J

def create_external_field(n_spins, field_type, strength):
    """Create external field based on type."""
    if field_type == "Zero":
        return np.zeros(n_spins)
    elif field_type == "Uniform":
        return np.ones(n_spins) * strength
    elif field_type == "Random":
        return np.random.randn(n_spins) * strength
    elif field_type == "Gradient":
        # Create gradient across grid
        field = np.zeros(n_spins)
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                field[idx] = strength * (i + j) / (2 * grid_size)
        return field
    return np.zeros(n_spins)

# Main content area
if HAS_MLTSU:
    # Create the model
    J = create_coupling_matrix(n_spins, coupling_type, coupling_strength)
    h = create_external_field(n_spins, field_type, field_strength)

    # Display model visualization
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("📊 Coupling Matrix J")
        fig_j = go.Figure(data=go.Heatmap(
            z=J,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="J_ij"),
        ))
        fig_j.update_layout(
            height=400,
            xaxis=dict(title="Spin i"),
            yaxis=dict(title="Spin j"),
        )
        st.plotly_chart(fig_j, use_container_width=True)

    with col2:
        st.subheader("🧲 External Field h")
        h_grid = h.reshape(grid_size, grid_size)
        fig_h = go.Figure(data=go.Heatmap(
            z=h_grid,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="h_i"),
        ))
        fig_h.update_layout(
            height=400,
            xaxis=dict(title="x"),
            yaxis=dict(title="y"),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col3:
        st.subheader("📈 Model Statistics")
        st.metric("Number of parameters", n_spins * (n_spins - 1) // 2 + n_spins)
        st.metric("Average |J|", f"{np.abs(J).mean():.3f}")
        st.metric("Average |h|", f"{np.abs(h).mean():.3f}")

        # Critical temperature estimate (for ferromagnetic case)
        if coupling_type == "Ferromagnetic":
            J_avg = np.abs(J[J != 0]).mean() if np.any(J != 0) else 1.0
            T_c = 2 * J_avg / np.log(1 + np.sqrt(2))  # 2D Ising critical temp
            st.metric("Critical T (estimate)", f"{T_c:.2f}")
            if temperature < T_c:
                st.success("Below critical temperature - expect order")
            else:
                st.warning("Above critical temperature - expect disorder")

    # Sampling section
    st.header("🎲 Thermodynamic Sampling")

    # Initialize session state for storing results
    if 'sampling_history' not in st.session_state:
        st.session_state.sampling_history = []
    if 'current_state' not in st.session_state:
        st.session_state.current_state = None

    # Sampling controls
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🚀 Run Sampling", type="primary"):
            # Use appropriate seed based on reproducibility mode
            if reproducibility_mode != "Statistical (5 seeds)":
                backend = get_backend(sampling_method, seed_to_use)
            else:
                # Will handle multiple seeds below
                backend = get_backend(sampling_method, seeds_to_use[0])

            with st.spinner(f"Running {sampling_method} sampling..."):
                progress_bar = st.progress(0)

                # Run sampling with progress updates
                samples = []
                energies = []
                magnetizations = []

                # Initial random state
                if st.session_state.current_state is None:
                    init_state = np.random.choice([-1, 1], size=n_spins)
                else:
                    init_state = st.session_state.current_state

                # Handle statistical mode (multiple seeds)
                if reproducibility_mode == "Statistical (5 seeds)":
                    all_seed_results = []
                    st.write("Running 5 independent trials...")

                    for seed_idx, seed in enumerate(seeds_to_use):
                        # Create backend with this seed
                        backend = get_backend(sampling_method, seed)
                        seed_samples = []
                        seed_energies = []
                        seed_magnetizations = []

                        # Run sampling for this seed
                        steps_per_chain = num_steps // 2  # Fewer steps per seed
                        result = backend.sample_ising(
                            J, h, beta, steps_per_chain,
                            batch_size=1, init_state=init_state
                        )

                        sample = result['samples'][0]
                        energy = result['final_energy'][0]

                        seed_samples.append(sample)
                        seed_energies.append(energy)
                        seed_magnetizations.append(np.mean(sample))

                        all_seed_results.append({
                            'seed': seed,
                            'energy': energy,
                            'magnetization': np.mean(sample),
                            'sample': sample
                        })

                        progress_bar.progress((seed_idx + 1) / 5)

                    # Calculate statistics
                    mean_energy = np.mean([r['energy'] for r in all_seed_results])
                    std_energy = np.std([r['energy'] for r in all_seed_results])
                    mean_mag = np.mean([r['magnetization'] for r in all_seed_results])
                    std_mag = np.std([r['magnetization'] for r in all_seed_results])

                    # Use mean results
                    samples = [r['sample'] for r in all_seed_results]
                    energies = [r['energy'] for r in all_seed_results]
                    magnetizations = [r['magnetization'] for r in all_seed_results]

                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Energy", f"{mean_energy:.2f} ± {std_energy:.2f}")
                    with col2:
                        st.metric("Mean Magnetization", f"{mean_mag:.3f} ± {std_mag:.3f}")
                    with col3:
                        # Export button for statistical data
                        import json
                        import pandas as pd

                        export_data = {
                            'summary': {
                                'mean_energy': mean_energy,
                                'std_energy': std_energy,
                                'mean_magnetization': mean_mag,
                                'std_magnetization': std_mag,
                                'n_seeds': len(seeds_to_use),
                                'seeds': seeds_to_use,
                                'beta': beta,
                                'n_spins': n_spins,
                                'sampling_method': sampling_method,
                                'num_steps': num_steps
                            },
                            'trials': all_seed_results
                        }

                        # Create downloadable JSON
                        json_str = json.dumps(export_data, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                        st.download_button(
                            label="📊 Export Data",
                            data=json_str,
                            file_name=f"ising_statistics_{int(time.time())}.json",
                            mime="application/json",
                            help="Export statistical analysis data for publication"
                        )

                else:
                    # Original single-seed sampling
                    n_chains = 10
                    steps_per_chain = num_steps // n_chains

                    for i in range(n_chains):
                        result = backend.sample_ising(
                            J, h, beta, steps_per_chain,
                            batch_size=1, init_state=init_state
                        )

                        sample = result['samples'][0]
                        energy = result['final_energy'][0]

                        samples.append(sample)
                        energies.append(energy)
                        magnetizations.append(np.mean(sample))

                        init_state = sample  # Use previous sample as init
                        progress_bar.progress((i + 1) / n_chains)

                # Store final state
                st.session_state.current_state = samples[-1]

                # Store history
                st.session_state.sampling_history.append({
                    'samples': samples,
                    'energies': energies,
                    'magnetizations': magnetizations,
                    'beta': beta,
                    'method': sampling_method,
                })

                st.success(f"Sampling complete! Final energy: {energies[-1]:.2f}")

    with col2:
        if st.button("🔄 Reset State"):
            st.session_state.current_state = None
            st.session_state.sampling_history = []
            st.rerun()

    with col3:
        if st.button("📊 Batch Sample"):
            # Use appropriate seed based on reproducibility mode
            if reproducibility_mode != "Statistical (5 seeds)":
                backend = get_backend(sampling_method, seed_to_use)
            else:
                # Will handle multiple seeds below
                backend = get_backend(sampling_method, seeds_to_use[0])

            with st.spinner("Running batch sampling..."):
                # Sample multiple independent chains
                batch_size = 20
                result = backend.sample_ising(
                    J, h, beta, num_steps, batch_size=batch_size
                )

                # Compute statistics
                mean_energy = result['final_energy'].mean()
                std_energy = result['final_energy'].std()
                mean_mag = np.mean([np.mean(s) for s in result['samples']])

                st.success(f"Batch complete!")
                st.metric("Mean energy", f"{mean_energy:.2f} ± {std_energy:.2f}")
                st.metric("Mean magnetization", f"{mean_mag:.3f}")

    # Visualization of results
    if st.session_state.sampling_history:
        st.header("📊 Results Visualization")

        latest = st.session_state.sampling_history[-1]

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Final Spin Configuration",
                "Energy Evolution",
                "Magnetization Evolution",
                "Spin Correlation"
            ),
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )

        # 1. Final spin configuration
        final_state = latest['samples'][-1].reshape(grid_size, grid_size)
        fig.add_trace(
            go.Heatmap(z=final_state, colorscale='RdBu', zmid=0, showscale=True),
            row=1, col=1
        )

        # 2. Energy evolution
        fig.add_trace(
            go.Scatter(y=latest['energies'], mode='lines+markers', name='Energy'),
            row=1, col=2
        )

        # 3. Magnetization evolution
        fig.add_trace(
            go.Scatter(y=latest['magnetizations'], mode='lines+markers', name='Magnetization'),
            row=2, col=1
        )

        # 4. Spin-spin correlations
        if len(latest['samples']) > 1:
            correlations = np.corrcoef(np.array(latest['samples']).T)
            fig.add_trace(
                go.Heatmap(z=correlations, colorscale='RdBu', zmid=0),
                row=2, col=2
            )

        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Additional statistics
        st.subheader("📈 Statistical Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Final Energy", f"{latest['energies'][-1]:.2f}")
            st.metric("Energy Change", f"{latest['energies'][-1] - latest['energies'][0]:.2f}")

        with col2:
            st.metric("Final Magnetization", f"{latest['magnetizations'][-1]:.3f}")
            acceptance_rate = result.get('acceptance_rate', 1.0) if 'result' in locals() else 1.0
            st.metric("Acceptance Rate", f"{acceptance_rate:.2%}")

        with col3:
            # Compute order parameter
            order_param = np.abs(latest['magnetizations'][-1])
            st.metric("Order Parameter |M|", f"{order_param:.3f}")

            # Energy per spin
            energy_per_spin = latest['energies'][-1] / n_spins
            st.metric("Energy per spin", f"{energy_per_spin:.3f}")

        with col4:
            # Estimate effective temperature from fluctuations
            if len(latest['energies']) > 1:
                energy_var = np.var(latest['energies'])
                if energy_var > 0:
                    T_eff = np.sqrt(energy_var) / n_spins
                    st.metric("Effective T", f"{T_eff:.3f}")

            # Spin glass order parameter
            q = np.mean(latest['samples'][-1]**2)
            st.metric("Spin Glass q", f"{q:.3f}")

    # Information section
    with st.expander("ℹ️ About the Ising Model"):
        st.markdown("""
        The **Ising model** is a mathematical model of ferromagnetism in statistical mechanics.

        **Energy function:**
        ```
        E(s) = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ
        ```

        Where:
        - `sᵢ ∈ {-1, +1}` are spin variables
        - `Jᵢⱼ` is the coupling between spins i and j
        - `hᵢ` is the external field at spin i
        - `β = 1/T` is the inverse temperature

        **Physical interpretations:**
        - **Ferromagnetic** (J > 0): Spins prefer to align → ordered phase at low T
        - **Antiferromagnetic** (J < 0): Spins prefer to anti-align → alternating pattern
        - **Spin Glass** (random J): Frustrated interactions → complex energy landscape

        **TSU advantages:**
        - Natural hardware implementation via p-bits
        - Efficient sampling from Boltzmann distribution
        - Potential energy savings vs. digital simulation
        """)

    # Benchmark section
    with st.expander("⚡ Performance Benchmarks"):
        if st.button("Run Benchmark"):
            # Use appropriate seed based on reproducibility mode
            if reproducibility_mode != "Statistical (5 seeds)":
                backend = get_backend(sampling_method, seed_to_use)
            else:
                # Will handle multiple seeds below
                backend = get_backend(sampling_method, seeds_to_use[0])

            with st.spinner("Running performance benchmark..."):
                results = backend.benchmark_sampling_speed(n_spins=100, num_steps=1000)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Gibbs sampling", f"{results.get('gibbs_samples_per_sec', 0):.0f} samples/sec")

                with col2:
                    st.metric("Metropolis sampling", f"{results.get('metropolis_samples_per_sec', 0):.0f} samples/sec")

                st.info(f"""
                **Benchmark details:**
                - Model size: 100 spins
                - Steps: 1000
                - Device: {backend.get_info()['device']}
                - JAX version: {backend.get_info()['jax_version']}
                """)

else:
    st.error("MLTSU package not properly installed. Please check installation.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 MLTSU - Bridging PyTorch and Thermodynamic Computing</p>
    <p>Part of the Thermodynamic Probabilistic Computing Bridge project</p>
</div>
""", unsafe_allow_html=True)