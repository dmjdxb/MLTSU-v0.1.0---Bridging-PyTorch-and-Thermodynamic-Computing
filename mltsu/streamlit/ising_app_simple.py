"""
Simple Ising Model Playground - Works without JAX
Demonstrates the concept while JAX installs
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="TSU Ising Playground (Demo)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("🔬 Thermodynamic Ising Model Playground (Demo)")
st.markdown("""
This is a simplified demonstration of the **Thermodynamic Sampling Units (TSU)** concept
through the Ising model - a fundamental model in statistical physics and optimization.

**Note**: This is a demo version that works without JAX. The full version with JAX-accelerated
sampling will provide much better performance.
""")

# Sidebar controls
st.sidebar.header("🎛️ Ising Model Parameters")

# Model size
col1, col2 = st.sidebar.columns(2)
with col1:
    grid_size = st.number_input(
        "Grid size", min_value=3, max_value=10, value=5, step=1
    )
with col2:
    n_spins = grid_size * grid_size
    st.metric("Total spins", n_spins)

# Coupling parameters
st.sidebar.subheader("Coupling Matrix J")
coupling_type = st.sidebar.selectbox(
    "Coupling type",
    ["Ferromagnetic", "Antiferromagnetic", "Spin Glass"],
    help="Type of spin-spin interactions",
)

coupling_strength = st.sidebar.slider(
    "Coupling strength |J|",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.1,
)

# External field
st.sidebar.subheader("External Field")
field_strength = st.sidebar.slider(
    "Field strength",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
)

# Temperature
st.sidebar.subheader("Temperature")
beta = st.sidebar.slider(
    "Inverse temperature β",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
)
temperature = 1.0 / beta
st.sidebar.metric("Temperature T", f"{temperature:.2f}")

# Simple Python-based Ising simulation
def create_coupling_matrix(n_spins, grid_size, coupling_type, strength):
    """Create coupling matrix for 2D lattice"""
    J = np.zeros((n_spins, n_spins))

    # Create nearest-neighbor connections on 2D grid
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j

            # Right neighbor
            if j < grid_size - 1:
                idx_right = i * grid_size + (j + 1)
                if coupling_type == "Ferromagnetic":
                    J[idx, idx_right] = strength
                elif coupling_type == "Antiferromagnetic":
                    J[idx, idx_right] = -strength
                else:  # Spin Glass
                    J[idx, idx_right] = np.random.choice([-1, 1]) * strength

            # Bottom neighbor
            if i < grid_size - 1:
                idx_bottom = (i + 1) * grid_size + j
                if coupling_type == "Ferromagnetic":
                    J[idx, idx_bottom] = strength
                elif coupling_type == "Antiferromagnetic":
                    J[idx, idx_bottom] = -strength
                else:  # Spin Glass
                    J[idx, idx_bottom] = np.random.choice([-1, 1]) * strength

    # Make symmetric
    J = J + J.T
    return J

def simple_metropolis(J, h, beta, n_steps, initial_state=None):
    """Simple Metropolis algorithm for Ising model"""
    n_spins = len(h)

    # Initialize random state
    if initial_state is None:
        state = np.random.choice([-1, 1], size=n_spins).astype(float)
    else:
        state = initial_state.copy()

    energies = []
    states = []

    for step in range(n_steps):
        # Pick random spin to flip
        i = np.random.randint(0, n_spins)

        # Calculate energy change
        delta_E = 2 * state[i] * (np.dot(J[i], state) + h[i])

        # Metropolis acceptance
        if delta_E < 0 or np.random.random() < np.exp(-beta * delta_E):
            state[i] *= -1

        # Record state periodically
        if step % 10 == 0:
            states.append(state.copy())
            energy = -0.5 * np.dot(state, np.dot(J, state)) - np.dot(h, state)
            energies.append(energy)

    return states, energies

# Create the model
J = create_coupling_matrix(n_spins, grid_size, coupling_type, coupling_strength)
h = np.ones(n_spins) * field_strength

# Display model
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Coupling Matrix")
    fig_j = go.Figure(data=go.Heatmap(
        z=J,
        colorscale='RdBu',
        zmid=0,
    ))
    fig_j.update_layout(height=400)
    st.plotly_chart(fig_j, use_container_width=True)

with col2:
    st.subheader("🎯 Initial Configuration")
    initial_state = np.random.choice([-1, 1], size=n_spins).astype(float)
    initial_grid = initial_state.reshape(grid_size, grid_size)

    fig_init = go.Figure(data=go.Heatmap(
        z=initial_grid,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
    ))
    fig_init.update_layout(height=400)
    st.plotly_chart(fig_init, use_container_width=True)

# Sampling section
st.header("🎲 Run Simulation")

if st.button("🚀 Run Metropolis Sampling", type="primary"):
    with st.spinner("Running simulation..."):
        # Run simple Metropolis
        states, energies = simple_metropolis(
            J, h, beta, n_steps=1000, initial_state=initial_state
        )

        # Display results
        st.success(f"Simulation complete! Final energy: {energies[-1]:.2f}")

        # Create visualization
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Final Configuration")
            final_grid = states[-1].reshape(grid_size, grid_size)
            fig_final = go.Figure(data=go.Heatmap(
                z=final_grid,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
            ))
            fig_final.update_layout(height=400)
            st.plotly_chart(fig_final, use_container_width=True)

        with col2:
            st.subheader("Energy Evolution")
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(
                y=energies,
                mode='lines',
                name='Energy'
            ))
            fig_energy.update_layout(
                xaxis_title="Step",
                yaxis_title="Energy",
                height=400
            )
            st.plotly_chart(fig_energy, use_container_width=True)

        # Statistics
        st.subheader("📈 Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Final Energy", f"{energies[-1]:.2f}")
            st.metric("Energy per spin", f"{energies[-1]/n_spins:.3f}")

        with col2:
            magnetization = np.mean(states[-1])
            st.metric("Magnetization", f"{magnetization:.3f}")
            st.metric("Absolute Magnetization", f"{abs(magnetization):.3f}")

        with col3:
            energy_std = np.std(energies)
            st.metric("Energy Std Dev", f"{energy_std:.2f}")

            if len(energies) > 1:
                energy_change = energies[-1] - energies[0]
                st.metric("Total Energy Change", f"{energy_change:.2f}")

# Information
with st.expander("ℹ️ About This Demo"):
    st.markdown("""
    This is a simplified demonstration of the Ising model concept.

    **What's happening:**
    - We create a 2D grid of spins that can be +1 or -1
    - Spins interact with their neighbors according to the coupling matrix J
    - The Metropolis algorithm randomly flips spins, accepting or rejecting based on energy
    - Lower energy states are preferred at lower temperatures (high β)

    **Full Version Features (with JAX):**
    - 100× faster sampling with JAX acceleration
    - Multiple sampling algorithms (Gibbs, Parallel Tempering)
    - Larger grid sizes (up to 20×20)
    - Real-time trajectory visualization
    - Hardware backend support

    The full MLTSU framework bridges PyTorch deep learning with thermodynamic computing hardware.
    """)

st.markdown("---")
st.info("⏳ JAX is still installing. Once complete, you can run the full version with `streamlit run ising_app.py`")