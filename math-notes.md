1. docs/math-notes.md
# Thermodynamic Probabilistic Computing – Mathematical Notes

Version: 0.1  
Audience: ML / physics / hardware researchers

---

## 1. Ising and Boltzmann Distributions

### 1.1 Binary Spin Representation

We consider a system of \(N\) binary variables (“spins” or p-bits):

\[
\mathbf{s} = (s_1, \dots, s_N), \quad s_i \in \{-1, +1\}.
\]

This representation is convenient for Ising-type models and p-bit networks.

### 1.2 Energy Function

The standard Ising / Boltzmann energy function over spins is:

\[
E(\mathbf{s}) \;=\; - \sum_{i=1}^N h_i s_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N J_{ij} s_i s_j,
\]

where:

- \(h_i\) is an external bias on spin \(i\),
- \(J_{ij}\) is the coupling between spins \(i\) and \(j\) (often \(J_{ij} = J_{ji}\), and \(J_{ii}=0\)),
- the factor \(1/2\) avoids double counting symmetric couplings.

In vector notation:

\[
E(\mathbf{s}) \;=\; - \mathbf{h}^\top \mathbf{s} - \frac{1}{2} \mathbf{s}^\top J \mathbf{s}.
\]

### 1.3 Boltzmann Distribution

At inverse temperature \(\beta = \frac{1}{k_B T}\), the equilibrium distribution is the Boltzmann distribution:

\[
p(\mathbf{s}) \;=\; \frac{1}{Z} \exp\left(- \beta E(\mathbf{s})\right),
\]

where the partition function \(Z\) is:

\[
Z \;=\; \sum_{\mathbf{s} \in \{-1,+1\}^N} \exp\left(- \beta E(\mathbf{s})\right).
\]

For small \(N\), \(Z\) can be computed exactly (summing over \(2^N\) states). For realistic \(N\), we approximate expectations using sampling.

---

## 2. Conditional Probabilities and Gibbs Sampling

### 2.1 Local Field

Define the effective local field at spin \(i\):

\[
u_i(\mathbf{s}) \;=\; h_i + \sum_{j} J_{ij} s_j.
\]

This is the contribution to the energy gradient w.r.t. spin \(i\).

### 2.2 Conditional Spin Flip Probability

Consider the probability of \(s_i = +1\) given all other spins \(\mathbf{s}_{\neg i}\):

\[
p(s_i \mid \mathbf{s}_{\neg i})
\;\propto\;
\exp\left( \beta s_i \, u_i(\mathbf{s}) \right).
\]

Since \(s_i \in \{-1,+1\}\), we can write:

\[
p(s_i = +1 \mid \mathbf{s}_{\neg i})
=
\frac{\exp\left( \beta u_i \right)}{\exp\left( \beta u_i \right) + \exp\left(- \beta u_i \right)}
=
\frac{1}{1 + e^{-2 \beta u_i}}
=
\sigma\!\left( 2 \beta u_i \right),
\]

where \(\sigma(x) = \frac{1}{1 + e^{-x}}\) is the logistic sigmoid.

Similarly:

\[
p(s_i = -1 \mid \mathbf{s}_{\neg i}) = 1 - p(s_i = +1 \mid \mathbf{s}_{\neg i}).
\]

This logistic form is exactly the update rule of a p-bit with input \(u_i\) at effective temperature \(1/\beta\).

### 2.3 Gibbs Sampling Scheme

Gibbs sampling iteratively updates one (or a subset) of spins by drawing from these conditionals:

1. Start from an initial configuration \(\mathbf{s}^{(0)}\).
2. For each step \(t = 0, 1, \dots\):
   - Choose an index \(i_t\) (sequential, random, or block-structured).
   - Compute the local field \(u_{i_t}(\mathbf{s}^{(t)})\).
   - Sample \(s_{i_t}^{(t+1)} \sim p(s_{i_t} \mid \mathbf{s}^{(t)}_{\neg i_t})\).
   - Keep all other spins unchanged: \(s_j^{(t+1)} = s_j^{(t)}\) for \(j \neq i_t\).

Under mild conditions (ergodicity, aperiodicity), this Markov chain converges to the Boltzmann distribution \(p(\mathbf{s})\).

---

## 3. p-Bits as Probabilistic Neurons

### 3.1 p-Bit Definition

A p-bit is a noisy binary element whose state fluctuates with a tunable bias. One common definition is:

\[
m_i(t) \in \{-1, +1\},
\]

with dynamics:

\[
m_i(t + 1) = \mathrm{sgn}\left( \mathrm{rand}(-1, 1) + \tanh(\gamma u_i(t)) \right),
\]

where:

- \(\mathrm{rand}(-1,1)\) is a random number in \([-1,1]\),
- \(u_i(t)\) is an input signal (local field),
- \(\gamma\) controls “gain” / inverse temperature.

In probabilistic terms:

\[
\Pr(m_i = +1 \mid u_i) = \frac{1}{2}\left[ 1 + \tanh(\gamma u_i) \right] 
= \sigma(2\gamma u_i).
\]

Thus p-bits are **binary stochastic neurons** with sigmoid activation.

### 3.2 Neuron–Spin Mapping

Mapping between \(\{0,1\}\) and \(\{-1,+1\}\):

- From spin to bit: \(b_i = \frac{1 + s_i}{2} \in \{0,1\}\),
- From bit to spin: \(s_i = 2 b_i - 1 \in \{-1,1\}\).

Using this mapping, any p-bit network can be expressed as an Ising model, and conversely, any Ising model can be implemented by a p-bit network with appropriate biases and couplings.

---

## 4. Diffusion Models and Thermodynamic Denoising

### 4.1 Forward Diffusion SDE (Continuous-Time View)

Score-based diffusion models often use a forward stochastic differential equation (SDE):

\[
\mathrm{d} \mathbf{x}_t = f(\mathbf{x}_t, t)\, \mathrm{d}t + g(t) \, \mathrm{d}\mathbf{w}_t,
\quad t \in [0, T],
\]

where:

- \(\mathbf{x}_t \in \mathbb{R}^d\) is the state,
- \(f(\cdot)\) is the drift function,
- \(g(t)\) is the noise scale,
- \(\mathbf{w}_t\) is standard Brownian motion.

The forward process progressively destroys information, driving the data distribution towards a simple prior (e.g., Gaussian).

### 4.2 Reverse-Time SDE and Score Function

The reverse-time SDE (under mild conditions) has the form:

\[
\mathrm{d} \mathbf{x}_t
=
\Bigl[f(\mathbf{x}_t, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t) \Bigr] \,\mathrm{d}t + g(t)\, \mathrm{d}\bar{\mathbf{w}}_t,
\]

where:

- \(p_t(\mathbf{x})\) is the marginal density at time \(t\),
- \(\nabla_{\mathbf{x}} \log p_t(\mathbf{x})\) is the *score* function,
- \(\bar{\mathbf{w}}_t\) is a Brownian motion in reverse time.

In practice:

- A neural network \(s_\theta(\mathbf{x}, t)\) is trained to approximate the score \(\nabla_{\mathbf{x}} \log p_t(\mathbf{x})\).
- Sampling uses numerical SDE or ODE solvers that alternate between deterministic score-based updates and Gaussian noise.

### 4.3 Discrete-Time Diffusion Step

A simple Euler–Maruyama discretization of the reverse SDE:

\[
\mathbf{x}_{k-1}
=
\mathbf{x}_k
+
\Delta t\, \Bigl[f(\mathbf{x}_k, t_k) - g(t_k)^2 s_\theta(\mathbf{x}_k, t_k)\Bigr]
+
g(t_k)\sqrt{\Delta t} \,\boldsymbol{\xi}_k,
\]

with \(\boldsymbol{\xi}_k \sim \mathcal{N}(0, I)\).

In many diffusion models, this is specialized to simpler update rules where the score network directly predicts noise or gradients.

---

## 5. Connecting Diffusion and Thermodynamic Sampling

### 5.1 Energy-Based Perspective

Suppose we define an energy function \(E_\theta(\mathbf{x}, t)\) such that:

\[
\nabla_{\mathbf{x}} E_\theta(\mathbf{x}, t) \approx - \nabla_{\mathbf{x}} \log p_t(\mathbf{x}).
\]

Then the score can be approximated as:

\[
\nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \approx - \nabla_{\mathbf{x}} E_\theta(\mathbf{x}, t).
\]

Substituting into the reverse SDE, the drift becomes:

\[
f(\mathbf{x}_t, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)
\approx
f(\mathbf{x}_t, t) + g(t)^2 \nabla_{\mathbf{x}} E_\theta(\mathbf{x}_t, t).
\]

So diffusion sampling can be interpreted as **time-dependent sampling in an energy landscape** whose gradient is learned.

### 5.2 Discrete Thermodynamic Updates

A thermodynamic sampler such as a TSU can implement discrete-time updates that approximate a gradient-descent-plus-noise step on \(E_\theta\):

\[
\mathbf{x}_{k+1}
=
\mathbf{x}_k - \eta_k \nabla_{\mathbf{x}} E_\theta(\mathbf{x}_k, t_k) + \sqrt{2 \eta_k / \beta_k} \,\boldsymbol{\xi}_k,
\]

where:

- \(\eta_k\) is a step-size parameter,
- \(\beta_k\) is an effective inverse temperature,
- \(\boldsymbol{\xi}_k\) is standard Gaussian noise.

This resembles **Langevin dynamics** and can be realized thermodynamically.

In a binary / p-bit setting, the analog is:

1. Define an energy \(E_\theta(\mathbf{s}, t)\) over discrete configurations.
2. Update spins by Gibbs sampling at a scheduled temperature \(\beta(t)\).
3. The annealing of \(\beta(t)\) and the time-dependence of \(E_\theta\) together implement a denoising / generative process analogous to diffusion.

---

## 6. TSU-Like Sampler Abstraction

### 6.1 Abstract Operations

We define a TSU-like sampler as providing:

1. **Ising sampling:**
   \[
   \mathbf{s}_{\text{final}} \sim p(\mathbf{s}) \propto \exp\left(-\beta E(\mathbf{s})\right)
   \]
   via iterated Gibbs or Metropolis updates.

2. **Binary layer sampling:**
   Given logits \(\mathbf{z}\), produce stochastic binary outputs \(\mathbf{s}\):
   \[
   s_i \sim \mathrm{Bernoulli}\left(\sigma(\beta z_i)\right).
   \]

3. **Custom energy sampling:**
   Given a generic energy function \(E(\mathbf{x})\), evolve a configuration \(\mathbf{x}\) approximately according to Langevin / thermodynamic dynamics.

### 6.2 Interface Mapping

The abstract `TSUBackend` interface is the software manifestation of this sampler. JAX-based implementations simulate the dynamics; physical TSUs would implement the same high-level behavior in hardware.

---