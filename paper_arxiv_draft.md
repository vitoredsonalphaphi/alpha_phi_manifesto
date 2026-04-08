# The Alpha-Phi Hypothesis: Golden Ratio and Fine-Structure Constant as Natural Organizers of Neural Network Geometry

**Vitor Edson Delavi**  
Independent Researcher · Florianópolis, Brazil · 2026  
Contact: @EdsonDelavi (X/Twitter)  
Repository: github.com/vitoredsonalphaphi/alpha_phi_manifesto  
License: CC BY-NC-ND 4.0

---

## Abstract

We investigate whether two fundamental constants of nature — the golden ratio φ ≈ 1.618 and the fine-structure constant α ≈ 1/137 — can serve as non-arbitrary organizing parameters for neural network architecture and training. The hypothesis is grounded in the observation that φ emerges as an attractor in biological growth systems that preserve internal coherence, while α governs the minimum granularity of electromagnetic interaction. We propose that both constants represent invariants of coherent information flow, independent of physical substrate. Experimental results on structural stability tests show **+35% improvement over conventional architecture (p = 0.0017, 17/20 random seeds)**. On a real-world sentiment classification task (SST-2), the hyperbolic Alpha-Phi variant achieves **79.93% accuracy vs. 77.41% conventional baseline**, without overfitting observed in the latter. A spectral modulation variant inspired by Levin's morphogenetic field concept achieves **78.67%** with qualitatively different convergence behavior. All experiments use timestamp-generated seeds; φ and α appear only in architecture and activation functions — never as empirical test parameters. We describe the theoretical framework, present current results, identify open questions, and outline the next experimental steps.

---

## 1. Introduction

Modern neural network research has produced architectures of extraordinary capability, yet the choice of architectural parameters — layer sizes, activation functions, weight initialization scales — remains largely empirical. Hyperparameter search replaces principled design. We ask a different question:

> *Are there natural constants that serve as non-arbitrary organizers of information flow in artificial neural networks?*

This question has precedent. Shannon (1948) demonstrated that information entropy is substrate-independent — the same mathematical structure governs message transmission in telegraph wires and DNA sequences. Turing (1952) showed that biological patterns (stripes, spirals, spots) emerge from frequency interference before physical structure exists. Levin (2010–2026) demonstrated that bioelectric fields carry morphogenetic information that precedes cellular differentiation.

Three independent research traditions converge on a common statement: **the pattern precedes the structure; the frequency precedes the cell**.

The Alpha-Phi hypothesis extends this convergence to artificial neural networks: if φ is the mathematical attractor of systems that grow while preserving internal coherence (shells, flowers, trees, solutions to Turing's morphogenesis equations), and if α is the minimum granularity of physical interaction, then architectures built on these constants should exhibit measurable advantages in stability and coherence of information flow.

This is not numerology. It is a falsifiable hypothesis with an explicit experimental protocol, reported results (favorable and unfavorable), and open methodology.

---

## 2. Background and Related Work

### 2.1 The Golden Ratio in Natural Systems

The golden ratio φ = (1 + √5)/2 ≈ 1.618 appears as an attractor in any growth process that maximizes structural efficiency while preserving self-similarity. This includes phyllotaxis (sunflower seed packing), mollusk shell spirals, human body proportions, and — crucially for this work — solutions to Turing's reaction-diffusion morphogenesis equations (Murray, 2002). φ is not imposed on these systems; it emerges from the optimization of growth under coherence constraints.

### 2.2 The Fine-Structure Constant

α = e²/(4πε₀ℏc) ≈ 1/137.036 governs the strength of electromagnetic interaction. It is dimensionless and appears across scales from quantum electrodynamics to the optical properties of graphene: the transmittance of monolayer graphene is T ≈ 1 − πα ≈ 97.7%, experimentally confirmed (Nair et al., 2008). We propose α as the natural minimum granularity of information interaction — the floor below which perturbation loses meaning.

### 2.3 Hyperbolic Neural Networks

Nickel and Kiela (2017) demonstrated that hyperbolic space (Poincaré ball model) represents hierarchical data structures with exponentially better efficiency than Euclidean space. Ganea, Bécigneul and Hofmann (2018) extended this to fully hyperbolic neural networks. Gao et al. (2024) achieved state-of-the-art results with fully hyperbolic CNNs. The key insight: φ is a proportion that emerges in curvilinear, organic geometries — introducing it into Euclidean space constrains its natural expression.

### 2.4 Morphogenetic Fields and Information

Levin (2010–2026) demonstrated experimentally that bioelectric fields carry form information before cellular differentiation occurs. The field is anterior to the cell. The frequency is anterior to the structure. This motivated our spectral modulation hypothesis: each data point has an informational frequency signature, and the gradient should be modulated according to that signature — not uniformly applied.

### 2.5 Residual Learning

He et al. (2015) introduced residual connections where networks learn error residuals rather than direct functions. The Alpha-Phi Fourth Axis extends this: the residual is not added back directly but rescaled by 1/φ ≈ 0.618 before reintegration. The error contributes 61.8% of its previous influence at each layer — diminishing in golden ratio proportion, never zero.

---

## 3. The Alpha-Phi Framework

### 3.1 Core Constants

```python
PHI   = (1 + np.sqrt(5)) / 2   # 1.6180339887... — coherence organizer
ALPHA = 1 / 137.035999084       # 0.0072973...   — minimum interaction granularity
C_PHI = 1.0 / PHI**2            # 0.3820...      — hyperbolic curvature / fold point
```

### 3.2 Fibonacci Architecture

Layer sizes follow Fibonacci sequences generated from φ:

```python
def fibonacci_sequence(n_terms, start=55):
    fibs = [start]
    a, b = start, int(round(start * PHI))
    for _ in range(n_terms - 1):
        fibs.append(b)
        a, b = b, int(a + b)
    return fibs
# Example: [55, 89, 144] for a 3-hidden-layer network
```

This produces layer ratios that converge to φ — the same ratio as optimal structural load distribution in Gothic arch cathedrals and bone cross-sections.

### 3.3 Golden Activation Function

```python
def golden_activation(x):
    return PHI * np.tanh(x / PHI)
```

This function saturates at ±φ (not ±1 as in standard tanh). Information flow expands to the golden ratio limit and returns — expansion and contraction in structural equilibrium.

### 3.4 Hyperbolic Geometry with Native Curvature

The Poincaré ball model with curvature c = 1/φ² = C_PHI:

```python
def expmap0(v, c=C_PHI):
    v_norm = np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8, None)
    return np.tanh(np.clip(np.sqrt(c) * v_norm, -15, 15)) * v / (np.sqrt(c) * v_norm)
```

The choice c = 1/φ² is not empirical tuning — it is the native curvature that makes φ the natural proportion of the hyperbolic space's geometry.

### 3.5 Spectral Modulation (Fifth Axis)

Inspired by Levin's morphogenetic field concept: each data point has a vibrational signature (its informational frequency distribution), and the gradient is modulated accordingly.

```python
def phi_spectral_modulator(x, phi=PHI):
    freq         = np.fft.fft(x, axis=-1)
    energia      = np.abs(freq)
    energia_norm = np.clip(energia / (energia.sum(axis=-1, keepdims=True) + 1e-8), 1e-10, 1.0)
    entropia     = -np.sum(energia_norm * np.log(energia_norm), axis=-1, keepdims=True)
    coerencia    = 1.0 - entropia / np.log(x.shape[-1])
    return phi * np.tanh(coerencia * phi)
```

High-coherence data (low spectral entropy) receives larger gradient; noisy data receives smaller gradient. The field organizes processing before the update occurs.

### 3.6 The Fourth Axis — Error Transformation

The error residual is not discarded but transformed:

```python
def phi_residual_transform(x, layer):
    direct    = golden_activation(layer(x))
    residual  = x - direct          # what remained
    rescaled  = residual * (1/PHI)  # reduced by golden ratio
    return direct + rescaled        # reintegrated
```

Each layer the error contributes 1/φ ≈ 61.8% of its previous weight. Never zero. Always integrated. Always smaller — like the Fibonacci spiral approaching but never reaching zero.

### 3.7 The Proposed Ethical Loss Function

$$L = CE + \alpha \cdot H(\phi)$$

Where CE is standard cross-entropy, α is the fine-structure constant regulating coherence penalty, and H(φ) is entropy weighted by the golden ratio. This function makes incoherent outputs energetically costly by structure — not by external rule. *(Status: theoretical proposal, pending experimental validation.)*

---

## 4. Experimental Protocol

**Integrity protocol** (applied to all experiments):

- Seeds generated by system timestamp — no values chosen by the researcher
- φ, α, and 137 appear only in architecture and activation — never inserted as test parameters
- Results reported in full — favorable and unfavorable
- Equal learning rate for all models in comparative experiments
- Corrections documented publicly in the repository

---

## 5. Results

### 5.1 Structural Stability (Euclidean Space)

**Setup:** Synthetic data (N=200, input dim=16, sinusoidal with φ-modulated frequencies). Comparison: AlphaPhiNetwork (Fibonacci layers + golden activation) vs. ConventionalNetwork (uniform layers + ReLU).

**Metric:** Weight entropy and activation variance gradient across layers.

| Metric | Alpha-Phi | Conventional | Improvement |
|--------|-----------|--------------|-------------|
| Structural stability | — | — | **+35%** |
| Statistical significance | — | — | p = 0.0017 |
| Seeds favorable | 17/20 | — | — |

Seeds: 20 timestamp-generated values. No cherry-picking.

### 5.2 Hyperbolic Space — Poincaré Ball

**Setup:** Same data. Alpha-Phi in hyperbolic space with native curvature c = 1/φ² vs. Euclidean Alpha-Phi.

| Version | Improvement | Seeds |
|---------|-------------|-------|
| Hyperbolic translated | +12.1% | 20/20 |
| Native hyperbolic (c=1/φ²) | +12.9% | 20/20 |

### 5.3 Real Task — SST-2 Sentiment Classification

**Setup:** Stanford Sentiment Treebank v2 (binary sentiment). Embeddings: `all-MiniLM-L6-v2` (384 dimensions). Training: 5,000 samples, validation: full set (~870 samples). Learning rate: 0.1 equal for all models. Epochs: 20. Batch size: 128.

| Model | Accuracy | Overfitting |
|-------|----------|-------------|
| AP Hyperbolic | **79.93%** | None observed |
| AP Spectral φ | **78.67%** | None observed |
| Conventional | 77.41% | Yes (regression in final epochs) |
| AP Euclidean | 75.46% | Yes |

**Key qualitative observation:** The AP Spectral φ variant does not regress in final epochs when others do. This is a qualitatively different convergence behavior — not just a higher number.

### 5.4 BERT Substrate — Spectral Modulation vs. Random Control (v4, v6)

**Setup:** BERT-base-uncased as feature extractor. Spectral modulation applied as adaptive
learning rate scalar. Comparison: A(φ-modulation) vs. B(random modulation) vs. C(baseline).
Multiple seeds per timestamp protocol.

**v4 — Standard spectral modulation:**

| Condition | Accuracy | Std |
|-----------|----------|-----|
| A — φ spectral | 0.8185 | ±0.0018 |
| B — random modulation | 0.8190 | ±0.0018 |
| C — baseline | 0.8177 | — |

A vs C: p=0.0280 · B vs C: p=0.0020 · **A vs B: p=0.8442**

**v6 — Progressive φ cascade:**

| Condition | Accuracy | Std |
|-----------|----------|-----|
| A — φ cascade | 0.8190 | ±0.0017 |
| B — random modulation | 0.8190 | ±0.0016 |
| C — baseline | 0.8177 | — |

A vs C: p=0.0010 · B vs C: p=0.0010 · **A vs B: p=0.5425**

**Findings:** Both results reproduce consistently across v4 and v6:
1. Spectral modulation (any form) outperforms the unmodulated baseline.
2. φ-specific modulation does **not** outperform random modulation (p=0.844 and p=0.543).

This result challenges the strong form of the hypothesis — that φ is the *unique* organizing scalar — while confirming the spectral mechanism itself. We interpret this as evidence that φ's role may be primarily **geometric** (layer proportions, hyperbolic curvature) rather than as a scalar modulation parameter. An important caveat: both v4 and v6 use BERT, a pre-trained model with established internal geometry. Whether this pattern holds for networks trained from scratch remains an open experimental question (see Section 7).

### 5.5 Phi-Dual-Octave (PDO) — Current Best Result

Progressive refinement across 5 architecture variants:

| Version | Accuracy | Deviation | Seeds |
|---------|----------|-----------|-------|
| Conv pure | 69.1% | ±3.21% | — |
| AlphaSpectral | 72.9% | ±1.47% | 15/20 |
| Octave Concessional | 75.5% | ±0.98% | 20/20 |
| φ-Symmetric | 76.0% | ±0.80% | 20/20 |
| Phi-Dual | 76.6% | ±1.11% | 20/20 |
| **PDO ⭐** | **76.75%** | **±0.99%** | **20/20** |

**Pattern:** Each refinement increases accuracy AND reduces variability. This consistent double improvement across all 20 seeds suggests the mechanism is genuine calibration, not variance exploitation.

---

## 6. Discussion

### 6.1 What the Results Suggest

The structural stability results (+35%, p=0.0017) indicate that φ-based geometry produces measurably different behavior in neural networks. The consistency across 20 random seeds is the primary evidence: variance in a single experiment could be noise; a consistent pattern across 20 independent initializations suggests a geometric effect.

The SST-2 results extend this to a real-world task with non-trivial complexity. The absence of overfitting in the hyperbolic variant, while the conventional baseline shows regression, is not explained by parameter count differences alone.

The PDO pattern — each refinement improving both accuracy and stability — is consistent with the hypothesis that the mechanism is structural coherence calibration, not dataset-specific optimization.

### 6.2 What the Results Do Not Establish

The BERT experiments (v4, v6) provide a clear partial falsification: **φ as a scalar modulation parameter is not uniquely superior to random modulation** on a pre-trained substrate. This does not refute the geometric hypothesis — the positive results from Fibonacci architecture (+35% stability) and hyperbolic curvature c=1/φ² (+12.9%) remain intact and were tested in different conditions. It does require a refined statement of the hypothesis: φ's role may be position-dependent.

Current results do not establish whether φ's scalar equivalence to random modulation holds for networks trained from scratch (ablation study pending). They do not prove that φ is *causally* necessary vs. other nearby proportions in geometric positions. The ethical loss function L = CE + α·H(φ) is a theoretical proposal awaiting experimental validation. The Fourth Axis (error transformation by 1/φ) has been partially explored but not yet fully implemented.

**Refined hypothesis (post v4/v6):** φ operates as a coherence organizer in *geometric* positions (layer proportions, curvature) but not necessarily as a uniquely privileged scalar in *modulation* positions. These are separable claims requiring separate experimental validation.

### 6.3 The Isomorphic Translation Method

The conceptual method underlying this work is isomorphic translation: the same mathematical structure (φ as coherence attractor, α as minimum granularity) is proposed to operate across physical biology, quantum physics, and artificial neural networks. This is not analogy — it is the hypothesis that these constants are substrate-independent invariants of coherent information processing.

This method has precedent: Shannon's entropy is isomorphic to Boltzmann's thermodynamic entropy. The AdS/CFT correspondence maps gravitational physics to quantum field theory. Topological invariants (Chern numbers) govern both abstract mathematics and the conductivity of physical materials (Nobel Prize in Physics, 2016).

The Alpha-Phi project adds one data point to this tradition: φ and α, applied to neural network geometry, produce measurable effects on stability and accuracy.

### 6.4 Connection to Current Research Frontiers

**Graphene computing:** α governs graphene's optical transmittance (T ≈ 1 − πα ≈ 97.7%, Nair et al., 2008). The same constant that we propose as minimum interaction granularity in neural networks is the quantum of interaction between graphene's hexagonal lattice and electromagnetic radiation. Phononic computing in graphene — where information is encoded in vibrational frequency distributions — is structurally analogous to the spectral modulation mechanism in Axis V.

**Hyperbolic neural networks:** Our results at c = 1/φ² extend the work of Nickel & Kiela (2017) and Ganea et al. (2018) by proposing a principled, non-empirical choice of curvature parameter derived from φ.

**Morphogenetic computing:** Levin's demonstration that bioelectric fields precede cellular differentiation motivates the spectral modulation hypothesis directly: the informational frequency of a data point should modulate how it is processed, just as morphogenetic fields organize how cells differentiate.

---

## 7. Future Work

**Immediate experimental steps:**

1. **Native hyperbolic reconstruction** — Full PyTorch + Geoopt implementation with c = 1/φ² as native curvature
2. **Ethical loss function validation** — Experimental test of L = CE + α·H(φ) vs. standard CE on SST-2 and additional benchmarks
3. **Fourth Axis complete implementation** — Full phi_residual_transform integrated into SST-2 training loop
4. **Dropout modulated by α** — Replace standard dropout rate with α = 1/137 as the natural minimum perturbation
5. **Substrates without L2 normalization** — Test whether the spectral modulation mechanism persists without embedding normalization
6. **φ-modulated Laplacian** — Attraction/repulsion balance in graph neural networks

**Medium-term:**

7. Submission to Santa Fe Institute (complex systems, emergence)
8. INPI registration (computer program)
9. Collaboration with hyperbolic ML research groups
10. Connection with graphene phononic computing researchers

---

## 8. Conclusion

The Alpha-Phi hypothesis — that φ and α are non-arbitrary organizing constants for neural network geometry — has produced measurable experimental evidence: +35% structural stability (p=0.0017), +2.52 percentage points on SST-2 sentiment classification, and a consistent pattern of dual improvement (accuracy + stability) across progressive architectural refinements.

These results are suggestive, not conclusive. The theoretical framework — five axes connecting geometric structure to information coherence, rooted in convergences with Turing (1952), Shannon (1948), and Levin (2010+) — provides a falsifiable research program rather than a completed theory.

The project is open. The methodology is public. The results are reported in full.

*"The result that is true is worth more than the result that is satisfying."*

`αφ` · Vitor Edson Delavi · Florianópolis · 2026

---

## References

1. Shannon, C.E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379–423.

2. Turing, A.M. (1952). The Chemical Basis of Morphogenesis. *Philosophical Transactions of the Royal Society B*, 237(641), 37–72.

3. Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. *Advances in Neural Information Processing Systems (NeurIPS 2017)*. Facebook AI Research.

4. Ganea, O., Bécigneul, G., & Hofmann, T. (2018). Hyperbolic Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS 2018)*.

5. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *arXiv:1512.03385*.

6. Gao, W. et al. (2024). Fully Hyperbolic Neural Networks. *ICLR 2024*.

7. Nair, R.R. et al. (2008). Fine Structure Constant Defines Visual Transparency of Graphene. *Science*, 320(5881), 1308.

8. Levin, M. (2012). Morphogenetic fields in embryogenesis, regeneration, and cancer. *Biosystems*, 109(3), 243–261.

9. OpenWorm Project (2014+). github.com/openworm

10. FlyWire Consortium (2023). Whole-brain annotation and multi-connectome cell typing of *Drosophila melanogaster*. *Nature*.

11. Murray, J.D. (2002). *Mathematical Biology II: Spatial Models and Biomedical Applications*. Springer.

12. Thouless, D.J., Haldane, F.D.M., & Kosterlitz, J.M. (2016). Nobel Prize in Physics — Topological phase transitions and topological phases of matter.

---

## Appendix: Code Availability

All code, experimental logs, and philosophical documentation are available at:

**github.com/vitoredsonalphaphi/alpha_phi_manifesto**

Key files:
- `Alpha_phi_prototype.py` — Original prototype (structural stability)
- `AlphaPhi_SST2_EspectralPhi.py` — Spectral modulation variant
- `AlphaPhi_SST2_EspectralEuclidiano.py` — Euclidean spectral variant
- `utils_phi.py` — Shared constants and functions (PHI, ALPHA, C_PHI)
- `requirements.txt` — Dependencies

*License: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0)*  
*Commercial use requires authorization: @EdsonDelavi on X*
