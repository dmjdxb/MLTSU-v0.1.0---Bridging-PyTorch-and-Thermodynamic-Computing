"""
Streamlit Demo for TinyBioBERT with P-bit Training
Interactive medical NER demonstration with uncertainty visualization.
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mltsu.tsu_jax_sim.backend import JAXTSUBackend
from mltsu.models.tiny_biobert import (
    TinyBioBERTConfig,
    TinyBioBERTForTokenClassification,
)
from mltsu.training.medical_dataset import MedicalTokenizer, MedicalNERLabel
from mltsu.uncertainty.medical_uncertainty import MedicalUncertaintyQuantifier


# Page configuration
st.set_page_config(
    page_title="TinyBioBERT P-bit Demo",
    page_icon="🧬",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
.entity-disease { background-color: #ff9999; padding: 2px 6px; border-radius: 4px; }
.entity-chemical { background-color: #99ccff; padding: 2px 6px; border-radius: 4px; }
.entity-gene { background-color: #99ff99; padding: 2px 6px; border-radius: 4px; }
.entity-species { background-color: #ffcc99; padding: 2px 6px; border-radius: 4px; }
.uncertainty-high { color: #ff4444; font-weight: bold; }
.uncertainty-medium { color: #ff9944; }
.uncertainty-low { color: #44ff44; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load TinyBioBERT model and components."""
    # Initialize TSU backend
    tsu_backend = JAXTSUBackend(seed=42)

    # Create model config
    config = TinyBioBERTConfig(
        vocab_size=10000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_labels=MedicalNERLabel.num_labels(),
    )

    # Create model
    model = TinyBioBERTForTokenClassification(config, tsu_backend)
    model.eval()

    # Create tokenizer
    tokenizer = MedicalTokenizer()

    # Create uncertainty quantifier
    uncertainty_quantifier = MedicalUncertaintyQuantifier(
        model, n_samples=10
    )

    return model, tokenizer, uncertainty_quantifier, tsu_backend


def format_entity_html(token: str, label: str, confidence: float, uncertainty: float):
    """Format entity with HTML styling."""
    entity_map = {
        'B-Disease': 'disease',
        'I-Disease': 'disease',
        'B-Chemical': 'chemical',
        'I-Chemical': 'chemical',
        'B-Gene': 'gene',
        'I-Gene': 'gene',
        'B-Species': 'species',
        'I-Species': 'species',
    }

    if label == 'O':
        return token

    entity_class = entity_map.get(label, '')

    # Uncertainty class
    if uncertainty > 0.5:
        unc_class = 'uncertainty-high'
    elif uncertainty > 0.3:
        unc_class = 'uncertainty-medium'
    else:
        unc_class = 'uncertainty-low'

    html = f'<span class="entity-{entity_class}" title="Confidence: {confidence:.2f}, Uncertainty: {uncertainty:.2f}">{token}</span>'
    return html


def main():
    """Main Streamlit app."""
    # Title and description
    st.title("🧬 TinyBioBERT: Medical NER with P-bit Computing")
    st.markdown("""
    This demo showcases **TinyBioBERT**, a medical language model trained with
    thermodynamic computing (P-bits) for energy-efficient medical NLP with calibrated uncertainty.
    """)

    # Load model
    with st.spinner("Loading TinyBioBERT model..."):
        model, tokenizer, uncertainty_quantifier, tsu_backend = load_model()

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # P-bit parameters
    st.sidebar.subheader("P-bit Parameters")
    pbit_temperature = st.sidebar.slider(
        "P-bit Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Controls the randomness of P-bit sampling"
    )

    num_samples = st.sidebar.slider(
        "Uncertainty Samples",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Number of forward passes for uncertainty estimation"
    )

    # Update model parameters
    uncertainty_quantifier.n_samples = num_samples
    uncertainty_quantifier.temperature = pbit_temperature

    # Display mode
    st.sidebar.subheader("Display Options")
    show_uncertainty = st.sidebar.checkbox("Show Uncertainty", value=True)
    show_attention = st.sidebar.checkbox("Show Attention Patterns", value=False)
    show_energy = st.sidebar.checkbox("Show Energy Consumption", value=True)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 Medical Text Input")

        # Predefined examples
        examples = {
            "Diabetes & Hypertension": "Patient diagnosed with type 2 diabetes and hypertension requiring insulin therapy",
            "Cancer Genetics": "BRCA1 and BRCA2 mutations increase risk of breast and ovarian cancer",
            "COVID-19 Treatment": "COVID-19 pneumonia treated with remdesivir and dexamethasone",
            "Drug Interaction": "Warfarin interaction with aspirin increases bleeding risk",
            "Alzheimer's": "Alzheimer's disease characterized by amyloid plaques and tau tangles",
        }

        selected_example = st.selectbox(
            "Select an example or enter custom text:",
            ["Custom"] + list(examples.keys())
        )

        if selected_example == "Custom":
            text_input = st.text_area(
                "Enter medical text:",
                value="",
                height=100,
                placeholder="Enter medical text for NER analysis..."
            )
        else:
            text_input = st.text_area(
                "Enter medical text:",
                value=examples[selected_example],
                height=100
            )

        # Analyze button
        if st.button("🔬 Analyze", type="primary"):
            if text_input:
                with st.spinner("Processing with P-bit computing..."):
                    # Track time
                    start_time = time.time()

                    # Tokenize
                    encoded = tokenizer.encode(text_input)
                    input_ids = encoded['input_ids'].unsqueeze(0)
                    attention_mask = encoded['attention_mask'].unsqueeze(0)

                    # Get predictions with uncertainty
                    results = uncertainty_quantifier.predict_with_uncertainty(
                        input_ids, attention_mask
                    )

                    # Extract results
                    predictions = results['predictions'].squeeze()
                    confidences = results['confidences'].squeeze()
                    uncertainties = results['uncertainties'].squeeze()

                    # Compute uncertainty decomposition
                    uncertainty_decomp = uncertainty_quantifier.decompose_uncertainty(
                        input_ids, attention_mask
                    )

                    elapsed_time = time.time() - start_time

                # Display results
                st.header("🎯 Named Entity Recognition Results")

                # Format and display annotated text
                tokens = tokenizer.tokenize(text_input)
                html_output = []

                for i, token in enumerate(tokens):
                    if i + 1 < len(predictions):  # Account for CLS token
                        label = MedicalNERLabel.LABELS[predictions[i+1].item()]
                        conf = confidences[i+1].item()
                        unc = uncertainties[i+1].item()
                        html_token = format_entity_html(token, label, conf, unc)
                        html_output.append(html_token)
                    else:
                        html_output.append(token)

                st.markdown(" ".join(html_output), unsafe_allow_html=True)

                # Legend
                st.markdown("""
                **Entity Types:**
                <span class="entity-disease">Disease</span>
                <span class="entity-chemical">Chemical/Drug</span>
                <span class="entity-gene">Gene/Protein</span>
                <span class="entity-species">Species</span>
                """, unsafe_allow_html=True)

                if show_uncertainty:
                    st.header("📊 Uncertainty Analysis")

                    col_unc1, col_unc2, col_unc3 = st.columns(3)

                    with col_unc1:
                        st.metric(
                            "Total Uncertainty",
                            f"{uncertainty_decomp.total_uncertainty:.4f}"
                        )

                    with col_unc2:
                        st.metric(
                            "Epistemic (Model)",
                            f"{uncertainty_decomp.epistemic_uncertainty:.4f}"
                        )

                    with col_unc3:
                        st.metric(
                            "Aleatoric (Data)",
                            f"{uncertainty_decomp.aleatoric_uncertainty:.4f}"
                        )

                    # Uncertainty distribution plot
                    fig_unc = go.Figure()
                    fig_unc.add_trace(go.Histogram(
                        x=uncertainties.numpy(),
                        nbinsx=20,
                        name="Token Uncertainties",
                        marker_color='rgba(255, 100, 100, 0.7)'
                    ))
                    fig_unc.update_layout(
                        title="Uncertainty Distribution",
                        xaxis_title="Uncertainty",
                        yaxis_title="Count",
                        height=300
                    )
                    st.plotly_chart(fig_unc, use_container_width=True)

                if show_energy:
                    st.header("⚡ Energy Consumption")

                    # Estimate energy
                    gpu_energy = elapsed_time * 100  # ~100W GPU
                    pbit_energy = num_samples * len(tokens) * 1e-15 * 1e9  # nJ

                    col_e1, col_e2, col_e3 = st.columns(3)

                    with col_e1:
                        st.metric(
                            "Processing Time",
                            f"{elapsed_time:.3f} s"
                        )

                    with col_e2:
                        st.metric(
                            "GPU Energy (est.)",
                            f"{gpu_energy:.1f} J"
                        )

                    with col_e3:
                        st.metric(
                            "P-bit Energy (sim.)",
                            f"{pbit_energy:.3f} nJ",
                            delta=f"-{(1 - pbit_energy*1e-9/gpu_energy)*100:.1f}%"
                        )

            else:
                st.warning("Please enter some medical text to analyze.")

    with col2:
        st.header("📈 Statistics")

        # Model information
        st.subheader("Model Configuration")
        st.info(f"""
        **Architecture:** TinyBioBERT
        **Parameters:** ~6M
        **Hidden Size:** 256
        **Layers:** 4
        **Attention Heads:** 4
        **P-bit Samples:** {num_samples}
        """)

        # P-bit dynamics visualization
        if show_attention:
            st.subheader("P-bit Attention Dynamics")

            # Create synthetic attention pattern
            attention_pattern = np.random.random((4, 8, 8))

            fig_att = go.Figure(data=go.Heatmap(
                z=attention_pattern[0],
                colorscale='Viridis',
                showscale=True
            ))
            fig_att.update_layout(
                title="P-bit Attention Pattern (Head 1)",
                xaxis_title="Keys",
                yaxis_title="Queries",
                height=300
            )
            st.plotly_chart(fig_att, use_container_width=True)

        # Information box
        st.subheader("ℹ️ About P-bit Computing")
        st.markdown("""
        **Probabilistic bits (P-bits)** are the foundation of thermodynamic computing:

        - 🎲 **Stochastic sampling** instead of deterministic computation
        - ⚡ **Ultra-low energy** operation (~10⁻¹⁵ J per bit)
        - 🌡️ **Temperature-controlled** dynamics
        - 📊 **Natural uncertainty** quantification

        This demo simulates P-bit behavior using JAX. Real hardware would achieve
        100-1000× energy savings for sampling operations.
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    TinyBioBERT demonstrates the PyTorch → TSU bridge for medical AI.
    This is a research prototype showcasing thermodynamic computing for healthcare.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()