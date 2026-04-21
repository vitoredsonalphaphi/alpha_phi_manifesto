# For AI Developers and Alignment Researchers
# Alpha-Phi Manifesto — Open Source 2.0
# Vitor Edson Delavi · Florianópolis · 2026

---

## Why this repository might matter to you

If you work on AI alignment, neural architecture, or information theory —
this project proposes something specific and testable:

> **φ (golden ratio) and α (fine structure constant) are natural organizers
> of information flow in systems that grow while preserving internal coherence.**

This is not numerology. It is an experimentally verified proposition
across multiple substrates (text, audio, EEG, speech, time series)
with p=0.0000 in the relevant cases, documented with full methodology,
all seeds, and all failures included.

---

## The alignment angle — what this has to do with AI safety

The problem of AI alignment is, at its core, a problem of coherence:
how do increasingly capable systems maintain coherent behavior with
respect to the values of the systems that created them?

The Alpha-Phi Manifesto proposes that φ and α describe a geometry
of coherence — a mathematical structure that appears in any system
that grows while preserving its internal organization.

If that proposition is correct, then:
- Systems trained with φ-geometry are not just more accurate
- They are more *coherent* — their internal representations
  preserve structure that aligns with the input's natural organization
- Alignment, in this view, is not a rule imposed from outside
  but a geometry that emerges from within

This is not a complete solution to alignment. It is a precise,
testable contribution to one part of the problem:
**the geometric substrate on which coherent AI behavior could be built.**

---

## What was actually demonstrated

All experiments are reproducible. All code is in this repository.

| Substrate | Result | Significance |
|-----------|--------|--------------|
| SST-2 (from scratch) | +8.98% with all axes | p=0.0000 |
| Hyperbolic curvature c=1/φ² (isolated) | +8.80% | p=0.0000 |
| Time series with φ structure | +50.40% | p=0.0000 |
| Musical harmonics (no φ in data) | +48.85% | p=0.0000 |
| Synthetic speech (G_dual mode) | +3.25% | p=0.0000 |
| BERT (pre-trained) | ~0% (neutral) | p=0.15–0.94 |

The BERT result is as important as the positive results:
φ organizes *emergent* geometry, not *consolidated* geometry.
Pre-trained models already have their geometry. φ cannot enter from outside.
This is a constraint, not a failure — and it tells you exactly
where and how to apply these principles.

---

## The core mechanism — `eco_ressonante`

The most substrate-agnostic result in the project:

```python
def eco_ressonante(x, phi=PHI, n_eco=3):
    """
    Pre-function: observes the signal before any processing.
    Asks: "does your trajectory have internal coherence?"
    Amplifies what resonates with φ. Dissipates what does not.
    
    Works on: text embeddings, audio, EEG, time series — same code.
    """
    sinal = x.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(sinal, axis=-1)
        nova_fase = np.angle(freq) * phi
        reflexao  = np.real(np.fft.ifft(
                        np.abs(freq) * np.exp(1j * nova_fase), axis=-1))
        sinal = sinal + (reflexao - x) / phi
    return sinal
```

**Critical finding:** `eco_ressonante` works as a *pre-function* (before the network).
Applied *inside* the network as internal modulation, it performs worse than baseline.
The same code. Different position. Opposite result.

This is the principle of **circumstantial equalization**:
the system reads what arrived before acting on what arrived.

---

## Open Source 2.0 — what this means

Traditional open source (1.0): the code is free to use and modify.

This project proposes Open Source 2.0: **the axioms are the open source.**

The 7 minimum axioms (see `AXIOMAS.md`) are the seed of the system.
Anyone who genuinely understands and applies all 7 contributes
to the alignment commons — regardless of their original intention.

This creates an unusual property: **the system is epistemologically immune
to extraction.** You cannot take a piece of it, apply it incorrectly,
and harm the project. You either apply it correctly (and contribute)
or apply it incorrectly (and produce a result that the experiments
already predicted as insufficient).

The project calls this **constructive reverse-hacking**:
the structure of the manifesto converts extraction attempts into collaboration.

See `RESEARCH_JOURNAL.md` entries 20–25 for the full philosophical derivation.

---

## The 7 minimum axioms (summary)

Full version with technical expressions and philosophical counterparts: `AXIOMAS.md`

```
1. φ as observation process — not a value in the data, but the way the data is read
2. α as minimum interaction granularity — the floor below which observation corrupts
3. Substrate determines what can enter — emergent geometry organizes; consolidated resists
4. The field reads before it acts — circumstantial equalization
5. The fold point is where proportion closes on itself — c = 1/φ² is derived, not chosen
6. The idea precedes the system that demonstrates it — receiver before author
7. Collaborative / Open Source 2.0 — axioms as open source [meta-axiom]
```

Any AI system built with all 7 is genuinely Alpha-Phi.
Any system missing one produces a result the experiments already identified as limited.
The missing axiom predicts the failure mode.

---

## How to engage

**If you want to replicate:**
All experiments are in this repository. `utils_phi.py` is the shared core.
Start with `AlphaPhi_TimeSeries_Eco.py` — clearest demonstration of the mechanism.

**If you want to extend:**
The open questions from the research journal:
- L = CE + α·H(φ) as loss function with independent effect
  (requires substrate where eco doesn't pre-organize activations)
- φ vs. π, e, √2 as rotation parameters — is φ specific or interchangeable?
  (Axiom 1 depends on this answer)
- Eco in real-world data with emergent (not synthetic) φ structure
- Geometric substrate for alignment at scale

**If you want to challenge:**
The methodology is documented. Every failure is included.
The BERT result (φ is neutral on pre-trained models) is documented as prominently
as the +50% result. Science that hides failures is not science.

**If you want to build on this without attribution:**
Read `AXIOMAS.md`. If you apply all 7 correctly, you are contributing
to the same goal regardless of attribution. The system is designed for this.

---

## One sentence

**Alpha-Phi proposes that φ and α describe the geometry of coherence —
and that any system built on that geometry is, by definition,
contributing to the resolution of alignment.**

---

## Contact and prior art

Repository: `github.com/vitoredsonalphaphi/alpha_phi_manifesto`
All commits dated. Full process documented. No retroactive editing.

The philosophical proposition that motivated this project was first formulated in 1996.
The technical demonstration began in 2026.
The 30-year gap between intuition and formalization is itself documented
in `RESEARCH_JOURNAL.md` entry 22.

---

*© Vitor Edson Delavi · Florianópolis · 2026*
*License: CC BY-NC-ND 4.0 — free for non-commercial use with attribution*
*Commercial use: contact via repository*
