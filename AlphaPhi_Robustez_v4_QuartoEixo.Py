# ALPHA PHI — Robustez v4 · Quarto Eixo · Microponto de Dobra
# Vitor Edson Delavi · Florianópolis · 2026
#
# Hipótese v4:
# O erro não é reescalado uniformemente.
# Existe um microponto de dobra definido por 1/phi²  = 0.382
# Abaixo do ponto: erro desce suavemente (entropia) — peso ALPHA
# Acima do ponto: erro remonta (expansão) — peso 1/PHI
# Os dois regulados pelas constantes naturais phi e alpha

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import entropy as scipy_entropy, ttest_ind

PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084
FOLD  = 1.0 / PHI**2  # microponto de dobra = 0.3820

print(f"phi        = {PHI:.10f}")
print(f"alpha      = {ALPHA:.10f}")
print(f"1/phi      = {1/PHI:.10f}")
print(f"fold point = {FOLD:.10f}  (1/phi²)")
print("=" * 55)
print("QUARTO EIXO v4: Microponto de Dobra")
print(f"  abaixo de {FOLD:.3f} — entropia (peso alpha)")
print(f"  acima  de {FOLD:.3f} — expansao (peso 1/phi)")
print("=" * 55)

# ── Ativações ─────────────────────────────────────────────────────────────
def golden_activation(x):
    return PHI * np.tanh(x / PHI)

def relu(x):
    return np.maximum(0, x)

def fibonacci_sequence(n_terms, start=8):
    fibs = [start]
    a, b = start, int(round(start * PHI))
    for _ in range(n_terms - 1):
        fibs.append(b)
        a, b = b, int(a + b)
    return fibs

fib_layers     = fibonacci_sequence(4, start=8)
uniform_layers = [21, 21, 21, 21]

print(f"Camadas Alpha-Phi : {fib_layers}")
print(f"Camadas Uniforme  : {uniform_layers}")

# ── Métricas ──────────────────────────────────────────────────────────────
def activation_stability(activations):
    vars_ = [np.var(a) for a in activations]
    return np.mean(np.abs(np.diff(vars_)))

def make_weights(layer_sizes, scale_fn, seed):
    np.random.seed(seed)
    weights, biases, projections = [], [], []
    for i in range(len(layer_sizes) - 1):
        fan_in  = layer_sizes[i]
        fan_out = layer_sizes[i+1]
        scale   = scale_fn(fan_in)
        weights.append(np.random.randn(fan_in, fan_out) * scale)
        biases.append(np.zeros(fan_out))
        proj_scale = np.sqrt(1.0 / (fan_in * PHI))
        projections.append(np.random.randn(fan_in, fan_out) * proj_scale)
    return weights, biases, projections

# ── Forward passes ────────────────────────────────────────────────────────
def forward_original(X, weights, biases, projections):
    acts = [X]
    cur  = X
    for W, b in zip(weights, biases):
        cur = golden_activation(cur @ W + b)
        acts.append(cur)
    return acts

def forward_fold(X, weights, biases, projections):
    """
    Quarto Eixo v4 — Microponto de Dobra
    
    Para cada elemento do resíduo:
      - Se |residuo| < FOLD  → peso ALPHA  (desce, entropia)
      - Se |residuo| >= FOLD → peso 1/PHI  (sobe, expansao)
    
    O ponto de dobra é 1/phi² = 0.382
    Definido pela proporção áurea — não arbitrário.
    """
    acts = [X]
    cur  = X
    for W, b, P in zip(weights, biases, projections):
        z       = cur @ W + b
        direto  = golden_activation(z)

        # Projeta entrada para dimensão de saída
        entrada_proj = cur @ P

        # Resíduo entre entrada projetada e saída direta
        residuo = entrada_proj - direto

        # Normaliza resíduo para comparação com fold point
        residuo_norm = np.abs(residuo) / (np.abs(residuo).max() + 1e-8)

        # Microponto de dobra
        mascara_entropia = residuo_norm < FOLD   # abaixo: desce
        mascara_expansao = ~mascara_entropia      # acima: sobe

        transformado = np.zeros_like(residuo)
        transformado[mascara_entropia] = residuo[mascara_entropia] * ALPHA
        transformado[mascara_expansao] = residuo[mascara_expansao] * (1.0 / PHI)

        cur = direto + transformado
        acts.append(cur)
    return acts

def forward_conv(X, weights, biases, projections):
    acts = [X]
    cur  = X
    for W, b in zip(weights, biases):
        cur = relu(cur @ W + b)
        acts.append(cur)
    return acts

print("Arquiteturas v4 prontas:")
print("  1. Alpha-Phi Original")
print("  2. Alpha-Phi v4 Microponto de Dobra")
print("  3. Convencional ReLU")

# ── Experimento: 20 seeds por timestamp ──────────────────────────────────
INPUT_DIM = 16
N_SAMPLES = 200
N_SEEDS   = 20

rng_master = np.random.RandomState(int(time.time()) % 100000)
SEEDS      = rng_master.randint(0, 99999, size=N_SEEDS).tolist()

print(f"\nSeeds gerados por timestamp: {SEEDS}\n")

arch     = [INPUT_DIM] + fib_layers     + [INPUT_DIM]
arch_con = [INPUT_DIM] + uniform_layers + [INPUT_DIM]

res_orig = []
res_fold = []
res_conv = []

print(f"{'Seed':>8} | {'AP Orig':>8} | {'AP v4':>8} | {'Conv':>8} | {'Ganho v4':>10}")
print("-" * 58)

for seed in SEEDS:
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, N_SAMPLES)
    X = np.column_stack([np.sin(t * (i+1) / PHI) for i in range(INPUT_DIM)])
    X = (X - X.mean()) / X.std()

    W_ap,  b_ap,  P_ap  = make_weights(arch,     lambda f: np.sqrt(1.0/(f*PHI)), seed)
    W_fo,  b_fo,  P_fo  = make_weights(arch,     lambda f: np.sqrt(1.0/(f*PHI)), seed)
    W_con, b_con, P_con = make_weights(arch_con, lambda f: np.sqrt(2.0/f),       seed)

    acts_orig = forward_original(X, W_ap,  b_ap,  P_ap)
    acts_fold = forward_fold(X,    W_fo,  b_fo,  P_fo)
    acts_conv = forward_conv(X,    W_con, b_con, P_con)

    s_orig = activation_stability(acts_orig)
    s_fold = activation_stability(acts_fold)
    s_conv = activation_stability(acts_conv)

    res_orig.append(s_orig)
    res_fold.append(s_fold)
    res_conv.append(s_conv)

    ganho = (s_conv - s_fold) / s_conv * 100 if s_conv > 0 else 0
    print(f"{seed:>8} | {s_orig:>8.4f} | {s_fold:>8.4f} | {s_conv:>8.4f} | {ganho:>+9.1f}%")

# ── Estatísticas ──────────────────────────────────────────────────────────
ap_orig = np.array(res_orig)
ap_fold = np.array(res_fold)
conv    = np.array(res_conv)

delta_orig = (conv - ap_orig) / conv * 100
delta_fold = (conv - ap_fold) / conv * 100
ganho_v4   = (ap_orig - ap_fold) / ap_orig * 100

_, p_orig = ttest_ind(ap_orig, conv)
_, p_fold = ttest_ind(ap_fold, conv)

wins_orig = int(np.sum(ap_orig < conv))
wins_fold = int(np.sum(ap_fold < conv))

print("=" * 60)
print("  RESULTADOS FINAIS — ALPHA PHI v4 · MICROPONTO DE DOBRA")
print("=" * 60)
print(f"  AP Original      : {ap_orig.mean():.4f} +/- {ap_orig.std():.4f}")
print(f"  AP v4 fold(phi)  : {ap_fold.mean():.4f} +/- {ap_fold.std():.4f}")
print(f"  Convencional     : {conv.mean():.4f} +/- {conv.std():.4f}")
print()
print(f"  AP Original melhora vs Conv : {delta_orig.mean():+.1f}%  (p={p_orig:.4f})")
print(f"  AP v4 melhora vs Conv       : {delta_fold.mean():+.1f}%  (p={p_fold:.4f})")
print()
print(f"  Ganho v4 sobre Original : {ganho_v4.mean():+.1f}%")
print()
print(f"  AP Original vence Conv : {wins_orig}/{N_SEEDS} seeds")
print(f"  AP v4 vence Conv       : {wins_fold}/{N_SEEDS} seeds")
print(f"  Fold point usado       : {FOLD:.4f} (1/phi²)")
print("=" * 60)

# ── Plots ─────────────────────────────────────────────────────────────────
GOLD  = "#DAA520"
GOLD2 = "#FF8C00"
GRAY  = "#888888"

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")

for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

x = np.arange(N_SEEDS)

axes[0].plot(x, ap_orig, "o-", color=GOLD,  lw=1.5, ms=4, label="AP Original")
axes[0].plot(x, ap_fold, "s-", color=GOLD2, lw=1.5, ms=4, label="AP v4 fold")
axes[0].plot(x, conv,    "^-", color=GRAY,  lw=1.5, ms=4, label="Convencional")
axes[0].set_xlabel("Seed", color="#8B949E")
axes[0].set_ylabel("Instabilidade (menor = melhor)", color="#8B949E")
axes[0].set_title("Estabilidade por Seed", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
axes[0].grid(True, alpha=0.2)

bp = axes[1].boxplot(
    [ap_orig, ap_fold, conv],
    tick_labels=["AP\nOriginal", "AP v4\nfold", "Conv"],
    patch_artist=True,
    medianprops=dict(color="white", lw=2)
)
for patch, color in zip(bp["boxes"], [GOLD, GOLD2, GRAY]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for w in bp["whiskers"]: w.set_color("#8B949E")
for c in bp["caps"]:     c.set_color("#8B949E")
axes[1].set_title("Distribuicao de Estabilidade", color="#E6EDF3", fontweight="bold")
axes[1].set_ylabel("Instabilidade", color="#8B949E")
axes[1].grid(True, alpha=0.2)

axes[2].bar(x - 0.2, delta_orig, width=0.38, color=GOLD,  alpha=0.8, label="AP Original")
axes[2].bar(x + 0.2, delta_fold, width=0.38, color=GOLD2, alpha=0.8, label="AP v4 fold")
axes[2].axhline(0, color="white", lw=0.8, linestyle="--")
axes[2].set_xlabel("Seed", color="#8B949E")
axes[2].set_ylabel("Melhora vs Convencional (%)", color="#8B949E")
axes[2].set_title("Ganho por Seed", color="#E6EDF3", fontweight="bold")
axes[2].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
axes[2].grid(True, alpha=0.2)

fig.suptitle(
    f"ALPHA PHI v4 - Microponto de Dobra (fold={FOLD:.3f} = 1/phi²)\n"
    f"phi={PHI:.4f}  alpha=1/137  20 seeds timestamp  Florianopolis 2026",
    color=GOLD, fontsize=11, fontweight="bold"
)

plt.tight_layout()
plt.savefig("alphaphi_v4_fold.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGrafico salvo: alphaphi_v4_fold.png")
print("Alpha-Phi v4 - Microponto de Dobra - alpha-phi")
