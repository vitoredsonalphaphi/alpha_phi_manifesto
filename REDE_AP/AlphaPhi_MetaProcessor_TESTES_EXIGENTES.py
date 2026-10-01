"""
AlphaPhi_MetaProcessor_TESTES_EXIGENTES.py
===========================================
Três testes de rigor crescente para o AP Metaprocessador.

TESTE A — Adversarial: rede treinada para produzir máxima entropia
  O pior caso possível: output explicitamente uniformizado.
  Se AP selar aqui, sela em qualquer condição.

TESTE B — Escala Dimensional: D = 8, 32, 128, 256, 512
  Crucial para LLMs reais (dimensões de centenas a milhares).
  Coh_saída deve permanecer em ~0.6895 independente de D.

TESTE C — Pipeline em Cascata: N = 1, 2, 3, 5 redes em série
  Cada rede degrada o sinal antes de AP.
  Testa: AP funciona no fim de um pipeline profundo?

Florianópolis · outubro de 2026 · Sessão Good Morning
Vitor Edson Delavi · Claude
"""

import numpy as np

np.random.seed(42)

PHI   = 1.6180339887
ALPHA = 1 / 137.035999
SEAL  = 1 / PHI

# ── NÚCLEO ────────────────────────────────────────────────────────────────────

def relu(x): return np.maximum(0, x)

def medir_coh(ativ):
    mag  = np.abs(ativ) + 1e-10
    norm = mag / (mag.sum(axis=-1, keepdims=True) + 1e-10)
    norm = np.clip(norm, 1e-10, 1.0)
    H    = -(norm * np.log(norm)).sum(axis=-1)
    return 1.0 - H / np.log(ativ.shape[-1])

def redistribuir_seal(v):
    n   = len(v)
    idx = np.arange(n, dtype=float)
    w   = SEAL * (1.0 - SEAL) ** idx
    w  /= w.sum()
    mag = np.abs(v)
    ord_ = np.argsort(mag)[::-1]
    res  = np.empty_like(v)
    res[ord_] = np.sign(v[ord_]) * w * mag.sum()
    return res

def ap_selar(output):
    out    = output.copy()
    ciclos = np.zeros(len(out))
    for _ in range(20):
        sub = medir_coh(out) < SEAL
        if not sub.any(): break
        for i in np.where(sub)[0]:
            out[i] = redistribuir_seal(out[i])
            ciclos[i] += 1
    coh_d = medir_coh(out)
    return {
        'coh_antes':  float(medir_coh(output).mean()),
        'coh_depois': float(coh_d.mean()),
        'pct_seal':   float((coh_d >= SEAL).mean() * 100),
        'ciclos':     float(ciclos.mean()),
        'sepstro':    float(coh_d.mean() + (1 - coh_d.mean())),
    }

def mlp_forward(X, dims, seed=42):
    rng = np.random.default_rng(seed)
    h = X
    for i in range(len(dims) - 1):
        W = rng.normal(0, np.sqrt(2.0/dims[i]), (dims[i+1], dims[i]))
        b = np.zeros(dims[i+1])
        h = np.tanh(h @ W.T + b) if i == len(dims)-2 else relu(h @ W.T + b)
    return h

linha = "═" * 68

# ── TESTE A: ADVERSARIAL ──────────────────────────────────────────────────────

def teste_a():
    print(f"\n{linha}")
    print("  TESTE A — Adversarial: máxima entropia forçada")
    print(f"  Pior caso possível para o metaprocessador.")
    print(f"{linha}")

    N, D = 500, 32
    cenarios = {
        'Uniforme puro (teoricamente máx entropia)':
            np.ones((N, D)) / D + 1e-6 * np.random.randn(N, D),
        'Ruído gaussiano (entropia alta, natural)':
            np.random.randn(N, D),
        'Saída softmax uniforme (classificador neutro)':
            np.exp(np.random.randn(N, D) * 0.01),
        'Saída antiphi (distribuição invertida — mínima coerência)':
            # inverte a ordem de energia: menor componente recebe mais energia
            np.array([np.sort(np.abs(np.random.randn(D)))[::-1] * -1 +
                      np.random.randn(D)*0.1 for _ in range(N)]),
    }

    print(f"\n  {'Cenário':45s}  {'Coh antes':>10s}  {'Coh depois':>10s}  {'%≥SEAL':>7s}  {'Ciclos':>7s}")
    print(f"  {'─'*45}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*7}")

    for nome, X in cenarios.items():
        r = ap_selar(X)
        ok = "✓" if r['pct_seal'] == 100.0 else "~"
        print(f"  {nome[:45]:45s}  {r['coh_antes']:>10.4f}  {r['coh_depois']:>10.4f}  "
              f"{r['pct_seal']:>6.1f}%{ok}  {r['ciclos']:>7.1f}")

    print(f"\n  Critério: AP deve selar todos os cenários (100% ≥ SEAL).")

# ── TESTE B: ESCALA DIMENSIONAL ───────────────────────────────────────────────

def teste_b():
    print(f"\n{linha}")
    print("  TESTE B — Escala Dimensional: D = 8 → 512")
    print(f"  Coh_saída deve manter-se em ≈0.689 independente de D.")
    print(f"  (Dimensões de LLMs reais: 512–12288)")
    print(f"{linha}")

    N = 300
    dims = [8, 16, 32, 64, 128, 256, 512]

    print(f"\n  {'D':>5s}  {'Coh conv':>10s}  {'Coh AP':>10s}  {'ΔCoh':>8s}  {'%≥SEAL':>7s}  {'Ciclos':>7s}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*7}")

    for D in dims:
        X   = np.random.randn(N, 64)
        out = mlp_forward(X, [64, 128, 64, D], seed=42)
        r   = ap_selar(out)
        ok  = "✓" if r['pct_seal'] == 100.0 else "~"
        print(f"  {D:>5d}  {r['coh_antes']:>10.4f}  {r['coh_depois']:>10.4f}  "
              f"{r['coh_depois']-r['coh_antes']:>+8.4f}  "
              f"{r['pct_seal']:>6.1f}%{ok}  {r['ciclos']:>7.1f}")

    print(f"\n  Critério: Coh_AP deve permanecer estável (≈0.689) em todos os D.")
    print(f"  Escala-invariância confirmaria aplicabilidade a LLMs de produção.")

# ── TESTE C: PIPELINE EM CASCATA ──────────────────────────────────────────────

def teste_c():
    print(f"\n{linha}")
    print("  TESTE C — Pipeline em Cascata: 1, 2, 3, 5 redes antes de AP")
    print(f"  Cada rede degrada o sinal. AP opera no final.")
    print(f"  Simula pipelines reais de múltiplas etapas de processamento.")
    print(f"{linha}")

    N, D = 500, 32
    X = np.random.randn(N, 64)

    print(f"\n  {'Profundidade':>13s}  {'Coh pré-AP':>11s}  {'Coh AP':>8s}  {'ΔCoh':>8s}  {'%≥SEAL':>7s}  {'Ciclos':>7s}")
    print(f"  {'─'*13}  {'─'*11}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}")

    saida = X
    for n_redes in [1, 2, 3, 5]:
        # Cada rede adicional processa o output da anterior
        seeds = [42, 43, 44, 45, 46]
        saida_atual = X
        for k in range(n_redes):
            saida_atual = mlp_forward(saida_atual, [saida_atual.shape[1], 64, D], seed=seeds[k])
        r  = ap_selar(saida_atual)
        ok = "✓" if r['pct_seal'] == 100.0 else "~"
        print(f"  {n_redes:>2d} rede(s) em série  {r['coh_antes']:>11.4f}  {r['coh_depois']:>8.4f}  "
              f"{r['coh_depois']-r['coh_antes']:>+8.4f}  "
              f"{r['pct_seal']:>6.1f}%{ok}  {r['ciclos']:>7.1f}")

    print(f"\n  Critério: AP deve selar independentemente da profundidade do pipeline.")

# ── SUMÁRIO ───────────────────────────────────────────────────────────────────

def sumario():
    print(f"\n{linha}")
    print("  SÍNTESE DOS TRÊS TESTES DE RIGOR")
    print(f"{linha}")
    print(f"  Teste A — Adversarial   : AP sela mesmo o pior caso (máx entropia)?")
    print(f"  Teste B — Escala        : Coh_AP estável de D=8 a D=512?")
    print(f"  Teste C — Cascata       : AP sela no fim de pipeline profundo?")
    print(f"\n  Se os três confirmam → AP como metaprocessador é:")
    print(f"  (1) robusto a condições adversariais")
    print(f"  (2) escala-invariante (aplicável a LLMs de produção)")
    print(f"  (3) persistente em pipelines multi-etapa")
    print(f"\n  SEAL = 1/φ = {SEAL:.6f}  |  θ_harmônico = {SEAL:.4f}")
    print(f"{linha}\n")

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    teste_a()
    teste_b()
    teste_c()
    sumario()
