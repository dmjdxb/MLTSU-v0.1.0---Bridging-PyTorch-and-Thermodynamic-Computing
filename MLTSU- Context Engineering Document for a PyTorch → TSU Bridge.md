MLTSU: Context Engineering Document for a PyTorch → TSU Bridge

Thermodynamic Sampling for Large Language Models

0. Preamble

This document defines the conceptual, mathematical, and software architecture for MLTSU: a PyTorch-first stack where:

GPUs/CPUs handle deterministic linear algebra (matmuls, layer norms, etc.), and

Thermodynamic Sampling Units (TSUs) / p-bit hardware handle probabilistic sampling (attention patterns, noise, negative examples, memory retrieval, etc.).

The endgame: create the canonical bridge from mainstream deep learning (PyTorch) to emerging thermodynamic hardware (Extropic-style TSUs), starting with simulators and becoming hardware-ready the moment real TSU devices are available.

This is not “we magically train GPT-4 at 1000× lower wall-plug energy.” It is:

“We design the APIs, model patterns, and reference implementations that let thermodynamic hardware actually slot into modern LLM training and inference.”

1. Modern history of probabilistic computing & p-bits
1.1 From Ising models to Ising machines

The Ising model began as a toy for ferromagnets: a set of spins 
𝑠
𝑖
∈
{
−
1
,
+
1
}
s
i
	​

∈{−1,+1} with energy

𝐸
(
𝑠
)
=
−
∑
𝑖
<
𝑗
𝐽
𝑖
𝑗
𝑠
𝑖
𝑠
𝑗
−
∑
𝑖
ℎ
𝑖
𝑠
𝑖
,
E(s)=−
i<j
∑
	​

J
ij
	​

s
i
	​

s
j
	​

−
i
∑
	​

h
i
	​

s
i
	​

,

whose low-energy configurations correspond to ordered phases. Over time, it became the Swiss army knife of combinatorial optimization: many NP-hard problems (Max-Cut, SAT, TSP) can be mapped into Ising Hamiltonians whose ground states encode optimal solutions.

Because of that, an entire ecosystem of Ising machines emerged:

Coherent Ising machines (optical/quantum-inspired).

Hardware Boltzmann / Ising machines in CMOS and spintronics.

“Probabilistic Ising Machines” (PIMs), which explicitly embrace stochastic dynamics to sample from Boltzmann distributions.

These machines effectively outsource sampling from complicated distributions to physics itself, instead of simulating everything numerically.

1.2 Probabilistic bits (p-bits)

Classical bits are stable: 0 or 1 until flipped. Quantum bits (qubits) are coherent superpositions, with all the fun and pain that entails.

Probabilistic bits, or p-bits, sit in between: they are classical entities that fluctuate randomly between 0 and 1, with a tunable bias. A simple model:

𝑚
(
𝑡
)
∈
{
−
1
,
+
1
}
,
𝑃
(
𝑚
(
𝑡
)
=
+
1
)
=
𝜎
(
𝛽
𝐼
)
,
m(t)∈{−1,+1},P(m(t)=+1)=σ(βI),

where 
𝐼
I is an effective “input current” and 
𝜎
σ is a logistic function.

Key ideas from Camsari, Datta and others:

p-bits can be realized in low-barrier nanomagnets, CMOS circuits, or other noisy devices.

Networks of coupled p-bits can implement Boltzmann machines, logical circuits, and optimization solvers.

Because they run at room temperature and leverage natural fluctuations, they are good candidates for probabilistic computers (p-computers).

Recent work shows p-bit based probabilistic Ising machines can be built in integrated CMOS+MTJ platforms, with nanosecond update times and milliwatt-scale power budgets.

1.3 Probabilistic computing as “p-computers”

The emerging narrative:

p-computers are architectures where the primitive is sampling from a distribution, not evaluating a deterministic logic function.

They are well-suited to:

Combinatorial optimization (finding low-energy states).

Probabilistic inference (Boltzmann machines, Markov Random Fields).

Stochastic machine learning tasks (e.g. generative models, energy-based models).

This is exactly the regime where LLMs also spend a lot of time: sampling from high-dimensional distributions, either during training (noise, negatives) or inference (token generation).

2. Thermodynamic computing & TSUs
2.1 Extropic-style thermodynamic sampling units

Extropic is pushing a specific vision: Thermodynamic Sampling Units (TSUs) – chips that exploit thermal noise in transistor networks to sample from parameterized probability distributions.

Roughly:

You encode an energy-based model (EBM) or related structure into a TSU’s parameters (weights, couplings, biases).

The TSU physically relaxes toward low-energy states, emitting samples from the corresponding distribution.

Extropic claims ~10,000× energy savings on certain Denoising Thermodynamic Model (DTM) generative benchmarks compared with GPU baselines.

Important caveats:

These gains are currently simulation- and benchmark-specific, not a universal constant.

TSUs excel at sampling tasks, not dense linear algebra per se.

This aligns almost perfectly with p-bit / Ising-machine literature: use physics as a Bayesian sampler, not as a deterministic matmul engine.

2.2 Why LLMs care about sampling

LLM training/inference involves:

Deterministic modules:

Matmuls (QKV projections, MLPs),

Layernorm, residual adds.

Stochastic or combinatorial modules:

Attention pattern selection (implicitly via softmax),

Dropout / stochastic depth,

Negative sampling (contrastive / EBM / alignment),

Token sampling at inference,

Routing (Mixture-of-Experts, sparsity patterns),

Retrieval and memory access decisions.

TSUs map naturally to the second category. That’s where MLTSU lives.

3. Design goals for MLTSU
3.1 High-level goals

MLTSU is a PyTorch-native framework that:

Treats TSUs as a sampling co-processor:

PyTorch keeps doing forward/backward passes and matmuls.

TSUs generate stochastic and combinatorial structures.

Exposes a clean, minimal interface that:

Runs today with a JAX-based p-bit simulator,

Can be swapped for real TSU hardware later without changing model code.

Provides reference model patterns:

Thermodynamic attention layers.

TSU-backed Gaussian noise.

TSU-based hard negative sampling for EBM/contrastive losses.

(Later) TSU-based memory/retrieval modules.

Is scientifically respectable:

Tied to Ising/p-bit literature and EBM theory.

Designed for experiments comparing TSU-based vs conventional sampling.

3.2 Non-goals (for now)

Not replacing all matmuls with analog/Ising operations.

Not replacing backpropagation with fully thermodynamic gradient descent.

Not promising end-to-end “1000× cheaper GPT-4 training” today.

4. Core architectural idea
4.1 Two-plane architecture

Think of MLTSU as two planes:

Deterministic plane (GPU/CPU + PyTorch)

Handles all continuous differentiable computation:

Embeddings

Linear layers / MLPs

LayerNorm, residuals

Loss computation (CE, MSE, etc.)

Backprop is standard PyTorch autograd.

Probabilistic plane (TSU / p-bit backend)

Handles discrete or stochastic steps via sampling:

Binary masks (attention, dropout, sparsity).

Approximate Gaussian noise (via CLT on p-bits).

Discrete choices (token candidates, routing).

Energy-based negative sampling.

These planes talk through a well-defined interface (TSUBackend). The core conceptual contract:

Give TSU a description of an energy landscape or logits → get back samples from the corresponding distribution.

4.2 The TSUBackend interface

In code terms (conceptually):

class TSUBackend(Protocol):
    def sample_ising(J, h, beta, num_steps, batch_size, init_state, record_trajectory, key) -> dict:
        """
        Sample from an Ising model with couplings J and fields h.
        """

    def sample_binary_layer(logits, beta, num_steps, key) -> Array:
        """
        Given input logits (for each bit), return samples in {0,1} or {-1,+1}.
        """

    def sample_custom(energy_fn, init_state, num_steps, beta, key, **kwargs) -> Array:
        """
        Generic energy-based sampling hook.
        """


Different backends implement this:

JAXTSUBackend – JAX-based p-bit simulator (today).

ExtropicTSUBackend – real TSU hardware (future).

Others – e.g. software Gibbs samplers, FPGA prototypes, etc.

4.3 PyTorch integration pattern

PyTorch modules never talk directly to JAX, PCIe, or hardware. They only depend on TSUBackend. Example patterns:

TSU binary sampling layer

class TSUBinaryLayer(nn.Module):
    def __init__(self, tsu_backend, beta=1.0, num_steps=1):
        ...

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Forward: use TSU to sample binary states.
        Backward: use straight-through estimator so gradients flow through logits.
        """


TSU Gaussian noise generator

class TSUGaussianNoise:
    def __init__(self, tsu_backend, M=12, beta=1.0, num_steps=1):
        ...

    def sample(self, shape, device) -> torch.Tensor:
        """
        Use M p-bits per scalar, map {0,1}-> {-1,+1}, sum and normalize to approximate N(0,1).
        """


Thermodynamic attention layer

class ThermodynamicAttention(nn.Module):
    def __init__(self, d_model, n_heads, tsu_backend, n_samples=32, beta=1.0):
        ...

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Q,K,V as usual, but attention weights are sampled via TSU-backed binary patterns
        instead of softmax.
        """


TSU negative sampler

class TSUNegativeSampler(nn.Module):
    def __init__(self, tsu_backend, n_negatives: int):
        ...

    def forward(self, energy: torch.Tensor, target_onehot: torch.Tensor):
        """
        Use TSU to sample “hard” negative tokens from low-energy competitors.
        """


This modularity is the core of the “bridge.”

5. Mathematical grounding for TSU primitives
5.1 p-bits as noisy logistic units

A typical idealized p-bit follows something like:

𝑚
𝑖
(
𝑡
)
∈
{
−
1
,
+
1
}
,
𝑃
(
𝑚
𝑖
(
𝑡
)
=
+
1
∣
𝐼
𝑖
)
=
𝜎
(
𝛽
𝐼
𝑖
)
,
m
i
	​

(t)∈{−1,+1},P(m
i
	​

(t)=+1
	​

I
i
	​

)=σ(βI
i
	​

),

with 
𝐼
𝑖
=
∑
𝑗
𝑊
𝑖
𝑗
𝑚
𝑗
+
𝑏
𝑖
I
i
	​

=∑
j
	​

W
ij
	​

m
j
	​

+b
i
	​

. Under asynchronous updating and symmetric weights, such a network converges to a Boltzmann distribution

𝑃
(
𝑚
)
∝
𝑒
−
𝛽
𝐸
(
𝑚
)
,
𝐸
(
𝑚
)
=
−
1
2
∑
𝑖
,
𝑗
𝑊
𝑖
𝑗
𝑚
𝑖
𝑚
𝑗
−
∑
𝑖
𝑏
𝑖
𝑚
𝑖
.
P(m)∝e
−βE(m)
,E(m)=−
2
1
	​

i,j
∑
	​

W
ij
	​

m
i
	​

m
j
	​

−
i
∑
	​

b
i
	​

m
i
	​

.

Our TSU backend abstracts away how this sampling is implemented (spintronics, thermodynamic silicon, JAX simulation) and exposes only the resulting sampling oracle.

5.2 TSU binary layer

Given logits 
ℓ
𝑖
ℓ
i
	​

, you can interpret them as:

Either direct inputs 
𝐼
𝑖
=
ℓ
𝑖
I
i
	​

=ℓ
i
	​

 with

𝑃
(
𝑥
𝑖
=
1
)
=
𝜎
(
𝛽
ℓ
𝑖
)
,
P(x
i
	​

=1)=σ(βℓ
i
	​

),

Or parameters of an energy function 
𝐸
(
𝑥
)
E(x) where 
𝑥
𝑖
∈
{
0
,
1
}
x
i
	​

∈{0,1} and

𝑃
(
𝑥
)
∝
exp
⁡
(
−
𝛽
𝐸
(
𝑥
)
)
.
P(x)∝exp(−βE(x)).

The TSU binary layer gives you samples 
𝑥
x according to such distributions. That’s enough for:

Stochastic attention masks: sample which keys a given query attends to.

Dropout masks: drop neurons according to thermodynamic distribution.

Sparsity patterns: sample which weights/neurons are “on.”

5.3 TSU Gaussian via central limit theorem

To get approximate Gaussian noise from p-bits:

For each scalar we want, sample 
𝑀
M p-bits 
𝑏
𝑗
∈
{
0
,
1
}
b
j
	​

∈{0,1}.

Map to spins 
𝑠
𝑗
=
2
𝑏
𝑗
−
1
∈
{
−
1
,
+
1
}
s
j
	​

=2b
j
	​

−1∈{−1,+1}.

Compute

𝑧
=
1
𝑀
∑
𝑗
=
1
𝑀
𝑠
𝑗
.
z=
M
	​

1
	​

j=1
∑
M
	​

s
j
	​

.

By the central limit theorem, as 
𝑀
M grows, 
𝑧
≈
𝑁
(
0
,
1
)
z≈N(0,1) if the p-bits are weakly correlated. This is enough to replace torch.randn_like with TSU-backed noise in:

Diffusion models.

Bayesian weight sampling.

Gradient noise injection.

5.4 Thermodynamic attention

Standard attention:

Attn
(
𝑄
,
𝐾
,
𝑉
)
=
softmax
(
𝑄
𝐾
⊤
𝑑
𝑘
)
𝑉
.
Attn(Q,K,V)=softmax(
d
k
	​

	​

QK
⊤
	​

)V.

We reinterpret this as:

Scores: 
𝑆
=
𝑄
𝐾
⊤
𝑑
𝑘
S=
d
k
	​

	​

QK
⊤
	​

 (higher = more preferred).

Energy: 
𝐸
=
−
𝑆
E=−S (lower = more preferred).

Instead of computing a softmax distribution, we sample binary attention patterns 
𝑎
∈
{
0
,
1
}
𝑇
a∈{0,1}
T
 for each query using TSU:

𝑃
(
𝑎
∣
𝐸
)
∝
exp
⁡
(
−
𝛽
 
𝑎
⊤
𝐸
)
,
P(a∣E)∝exp(−βa
⊤
E),

then approximate attention weights via Monte Carlo:

𝑤
^
=
1
𝑁
∑
𝑛
=
1
𝑁
𝑎
(
𝑛
)
.
w
^
=
N
1
	​

n=1
∑
N
	​

a
(n)
.

These approximate probabilities of attending to each key. You can then:

Normalize 
𝑤
^
w
^
 to sum to 1.

Compute output as 
𝑤
^
𝑉
w
^
V.

Gradients flow through the pre-sampling logits using STE.

5.5 TSU negative sampling

In energy-based LM objectives, one often wants:

Low energy for the correct token,

High energy for plausible but incorrect tokens (“hard negatives”).

Given an energy tensor 
𝐸
∈
𝑅
𝐵
×
𝑇
×
𝑉
E∈R
B×T×V
, rather than:

Sampling negatives uniformly or via top-k on GPU,

we use TSUs to sample low-energy alternatives:

𝑃
(
neg token
=
𝑣
)
∝
exp
⁡
(
−
𝛽
𝐸
𝑏
,
𝑡
,
𝑣
)
,
P(neg token=v)∝exp(−βE
b,t,v
	​

),

conditioned on excluding the true target. TSUs give us efficient ways to explore the tail of this distribution, which may be energy-costly on a GPU.

6. Software architecture of MLTSU
6.1 Repository layout (conceptual)
mltsu/
  tsu_core/
    interfaces.py        # TSUBackend, common API
    utils.py

  tsu_jax_sim/
    state.py             # p-bit / Ising state (JAX)
    energy_models.py     # Ising, simple EBMs
    sampler.py           # Gibbs / Langevin / binary layer sampling
    backend.py           # JAXTSUBackend(TSUBackend)

  tsu_pytorch/
    bridge.py            # torch <-> JAX interop via DLPack
    binary_layer.py      # TSUBinaryLayer
    noise.py             # TSUGaussianNoise
    attention.py         # ThermodynamicAttention
    negatives.py         # TSUNegativeSampler
    memory.py            # (future) ThermodynamicMemory

  models/
    tiny_thermo_lm.py    # small transformer using thermodynamic attention
    mnist_tsu_diffusion.py

  streamlit/
    ising_app.py         # scientist playground
    attention_viz.py
    lm_playground.py

  docs/
    context_engineering.md  # this document
    math_notes.md           # detailed derivations and experiments

6.2 TSUBackend variants

JAXTSUBackend

Implemented using JAX for fast vectorized Gibbs / Langevin sampling.

Provides sample_ising, sample_binary_layer.

Used for development and reproducible experiments.

Software reference backend

Pure Python/NumPy implementation for debugging or platforms without JAX.

ExtropicTSUBackend (future)

Wraps Extropic’s XTR-0 TSU SDK.

Likely communicates over PCIe, Ethernet or similar.

Same interface, different latency/throughput and, crucially, energy behavior.

7. Model-level design patterns and use cases
7.1 Thermodynamic attention in LLM blocks

A typical transformer block with MLTSU:

class ThermoBlock(nn.Module):
    def __init__(self, d_model, n_heads, tsu_backend):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ThermodynamicAttention(d_model, n_heads, tsu_backend)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp  = FeedForward(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


Experiments to run:

Compare standard softmax vs thermodynamic attention in:

Perplexity,

Attention sparsity,

Calibration/uncertainty,

Robustness to noise.

How many TSU samples per head per token are needed to approximate softmax behavior?

7.2 TSU-backed noise in diffusion models

Use TSUGaussianNoise in a DDPM / diffusion model:

Forward process: use TSU noise for 
𝜖
ϵ in 
𝑥
𝑡
=
𝛼
ˉ
𝑡
𝑥
0
+
1
−
𝛼
ˉ
𝑡
𝜖
x
t
	​

=
α
ˉ
t
	​

	​

x
0
	​

+
1−
α
ˉ
t
	​

	​

ϵ.

Reverse process: use TSU noise for the stochastic part of sampling.

Compare with standard Gaussian RNG in:

Sample quality (FID),

Diversity,

Robustness to under/over-fitting.

This is tightly aligned with Extropic’s claimed Denoising Thermodynamic Models (DTM) story.

7.3 TSU negative sampling for energy-based LM objectives

Use TSUNegativeSampler as part of an auxiliary loss:

Compute energies / logits for all vocab items.

Sample TSU-based hard negatives.

Use a margin-based or contrastive energy loss that:

Lowers energy of true targets.

Raises energy of TSU-sampled negatives.

This emphasizes TSUs as probabilistic EBM accelerators for language.

7.4 Thermodynamic memory (future work)

Long-context LLMs struggle with:

Quadratic attention cost.

Deciding which past tokens/segments to attend to.

A thermodynamic memory module could:

Compress segments into energy representations.

Use TSU sampling to perform probabilistic retrieval of relevant segments given a query energy pattern.

Act as a learned, physics-backed retrieval mechanism.

This aligns with p-bit and Ising work on associative memory and pattern retrieval.

8. Evaluation plan & scientific questions

To make this credible to Extropic and the broader community, MLTSU should be evaluated along several axes.

8.1 Functional correctness

Does thermodynamic attention produce sensible attention maps?

Do diffusion models with TSU noise match baseline quality?

Are energy-based LM losses with TSU negatives stable to train?

8.2 Statistical properties

How does TSU sampling affect:

Calibration (expected calibration error),

Uncertainty estimation (entropy of logits, variance across samples),

Sparsity patterns (e.g., how many keys a query attends to)?

Can we shape attention via energy priors more directly than via softmax?

8.3 Algorithmic efficiency (simulator phase)

Even before real hardware, we can ask:

For a fixed budget of samples, does TSU-style sampling converge faster to good structures (negatives, masks) than e.g. Gumbel-softmax, top-k, or dropout-style schemes?

Are there regimes where TSU-based samplers discover “harder” negatives that standard heuristics miss?

8.4 Hardware-in-the-loop energy/performance (future)

Once real TSUs are accessible:

Measure energy per:

Sampled attention mask,

Gaussian noise vector,

Hard negative batch,

Memory retrieval query.

Compare:

Energy per operation vs GPU/CPU RNG and sampling.

End-to-end training energy fraction attributable to TSU-accelerated parts.

This is where claim ranges like “up to 10,000× per sampling workload” can be empirically tested in ML contexts.

9. Roadmap
Phase 1 – Core bridge & simulators

Implement TSUBackend and JAXTSUBackend.

Implement TSUBinaryLayer, TSUGaussianNoise.

Build:

Ising playground (Streamlit).

Diffusion demo with TSU noise.

Phase 2 – Thermodynamic attention + tiny LM

Implement ThermodynamicAttention.

Build a small decoder-only LM using this attention.

Train on a modest dataset; compare against softmax baseline.

Phase 3 – TSU negative sampling & EB auxiliary loss

Implement TSUNegativeSampler and SimpleEnergyBasedLMObjective.

Integrate as a side loss in tiny LM.

Measure impact on calibration, robustness, representation quality.

Phase 4 – Documentation & scientist UX

Finalize this context document and a companion math_notes.md.

Ship Streamlit apps for:

Ising dynamics,

Thermodynamic attention visualization,

LM sampling with TSU vs softmax.

Phase 5 – Hardware integration

Implement ExtropicTSUBackend once APIs/SDKs are available.

Run side-by-side experiments:

Same model, same training loop,

Backend = simulator vs real TSU,

Measure latency, throughput, energy, and statistical behavior.

10. Summary

MLTSU is not a marketing line about “1000× cheaper GPT tomorrow.” It’s a concrete stack to:

Bring p-bit / thermodynamic sampling into mainstream PyTorch models,

Align with the physics-driven literature on probabilistic computing and Ising machines,

Provide clean APIs that Extropic-style TSUs can implement,

Demonstrate real, end-to-end models (LLMs, diffusion) where:

Deterministic math runs on GPUs/CPUs,

Stochastic structure is delegated to thermodynamic hardware.

If we get this right, then when TSUs mature, you’re not scrambling to invent a new stack—you’re already holding the reference PyTorch → TSU bridge that people plug new probabilistic hardware into.

That’s the game: design the bridge, prove it works on simulators, then let physics do the energy-saving flex when the silicon arrives.