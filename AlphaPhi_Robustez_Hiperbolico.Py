# ALPHA PHI — Robustez no Espaço Hiperbólico Puro
# Vitor Edson Delavi · Florianópolis · 2026
#
# Pergunta central:
# O Robustez (+35% euclidiano) no espaço hiperbólico PURO — melhora?
#
# Três comparações:
# 1. Alpha-Phi euclidiano    — resultado consolidado +35%
# 2. Alpha-Phi hiperbólico   — todas as camadas em Poincaré
# 3. Convencional euclidiano — baseline

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import entropy as scipy_entropy, ttest_ind

PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084

print(f"phi   = {PHI:.10f}")
print(f"alpha = {ALPHA:.10f}")
print("=" * 60)
print("ROBUSTEZ — EUCLIDIANO vs HIPERBOLICO PURO")
print("Todas as camadas no espaco de Poincare")
print("=" * 60)

# ── Espaço Hiperbólico — Poincaré Ball ───────────────────────────────────
def expmap0(v, c=1.0):
    """Euclidiano → Poincaré"""
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.clip(v_norm, 1e-8, None)
    tanh_v = np.tanh(np.sqrt(c) * v_norm)
    return tanh_v * v / (np.sqrt(c) * v_norm)

def logmap0(y, c=1.0):
    """Poincaré → Euclidiano"""
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    y_norm = np.clip(y_norm, 1e-8, 1.0 - 1e-5)
    return np.arctanh(np.sqrt(c) * y_norm) * y / (np.sqrt(c) * y_norm)

def hyperbolic_linear(x, W, b, c=1.0):
    """Camada linear no espaço de Poincaré"""
    x_e   = logmap0(x, c)
    out_e = x_e @ W + b
    return expmap0(out_e, c)

def golden_activation_hyp(x, c=1.0):
    """φ·tanh(x/φ) no espaço de Poincaré"""
    x_e   = logmap0(x, c)
    act_e = PHI * np.tanh(x_e / PHI)
    norm  = np.linalg.norm(act_e, axis=-1, keepdims=True)
    act_e = act_e / np.maximum(norm, 1.0) * np.minimum(norm, 0.95)
    return expmap0(act_e, c)

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

# ── Pesos ─────────────────────────────────────────────────────────────────
def make_weights_phi(layer_sizes, seed):
    np.random.seed(seed)
    W, b = [], []
    for i in range(len(layer_sizes)-1):
        fan_in = layer_sizes[i]
        scale  = np.sqrt(1.0 / (fan_in * PHI))
        W.append(np.random.randn(fan_in, layer_sizes[i+1]) * scale)
        b.append(np.zeros(layer_sizes[i+1]))
    return W, b

def make_weights_conv(layer_sizes, seed):
    np.random.seed(seed)
    W, b = [], []
    for i in range(len(layer_sizes)-1):
        fan_in = layer_sizes[i]
        scale  = np.sqrt(2.0 / fan_in)
        W.append(np.random.randn(fan_in, layer_sizes[i+1]) * scale)
        b.append(np.zeros(layer_sizes[i+1]))
    return W, b

# ── Forward passes ────────────────────────────────────────────────────────
def forward_euclidean(X, weights, biases):
    """Alpha-Phi Robustez Original — euclidiano puro"""
    acts = [X]
    cur  = X
    for W, b in zip(weights, biases):
        cur = golden_activation(cur @ W + b)
        acts.append(cur)
    return acts

def forward_hyperbolic(X, weights, biases, c=1.0):
    """
    Alpha-Phi Robustez — espaço hiperbólico PURO
    Todas as camadas na bola de Poincaré
    Entrada projetada para Poincaré no início
    Saída projetada de volta ao euclidiano no final
    """
    acts = [X]
    # Projeta entrada para espaço hiperbólico
    cur = expmap0(X, c)
    acts.append(cur)

    for i, (W, b) in enumerate(zip(weights, biases)):
        cur = hyperbolic_linear(cur, W, b, c)
        cur = golden_activation_hyp(cur, c)
        acts.append(cur)

    # Projeta saída de volta para euclidiano
    cur = logmap0(cur, c)
    acts[-1] = cur
    return acts

def forward_conv(X, weights, biases):
    acts = [X]
    cur  = X
    for W, b in zip(weights, biases):
        cur = relu(cur @ W + b)
        acts.append(cur)
    return acts

def activation_stability(activations):
    vars_ = [np.var(a) for a in activations]
    return np.mean(np.abs(np.diff(vars_)))

def weight_entropy(weights):
    all_w = np.concatenate([W.flatten() for W in weights])
    hist, _ = np.histogram(all_w, bins=50, density=True)
    hist = hist + 1e-10
    hist /= hist.sum()
    return scipy_entropy(hist)

print("Forward passes:")
print("  1. Alpha-Phi Euclidiano puro")
print("  2. Alpha-Phi Hiperbolico puro (Poincare Ball)")
print("  3. Convencional ReLU")

# ── Experimento 20 seeds ──────────────────────────────────────────────────
INPUT_DIM = 16
N_SAMPLES = 200
N_SEEDS   = 20

rng_master = np.random.RandomState(int(time.time()) % 100000)
SEEDS      = rng_master.randint(0, 99999, size=N_SEEDS).tolist()

print(f"\nSeeds por timestamp: {SEEDS}\n")

arch     = [INPUT_DIM] + fib_layers     + [INPUT_DIM]
arch_con = [INPUT_DIM] + uniform_layers + [INPUT_DIM]

res_euc  = []
res_hyp  = []
res_conv = []

print(f"{'Seed':>8} | {'AP Eucl':>8} | {'AP Hyp':>8} | {'Conv':>8} | {'Δ Hyp>Eucl':>11}")
print("-" * 62)

for seed in SEEDS:
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, N_SAMPLES)
    X = np.column_stack([np.sin(t*(i+1)/PHI) for i in range(INPUT_DIM)])
    X = (X - X.mean()) / X.std()

    W_e, b_e = make_weights_phi(arch,      seed)
    W_h, b_h = make_weights_phi(arch,      seed)
    W_c, b_c = make_weights_conv(arch_con, seed)

    try:
        acts_euc  = forward_euclidean(X,   W_e, b_e)
        acts_hyp  = forward_hyperbolic(X,  W_h, b_h)
        acts_conv = forward_conv(X,        W_c, b_c)

        s_euc  = activation_stability(acts_euc)
        s_hyp  = activation_stability(acts_hyp)
        s_conv = activation_stability(acts_conv)

        res_euc.append(s_euc)
        res_hyp.append(s_hyp)
        res_conv.append(s_conv)

        delta = (s_euc - s_hyp) / s_euc * 100
        print(f"{seed:>8} | {s_euc:>8.4f} | {s_hyp:>8.4f} | {s_conv:>8.4f} | {delta:>+10.1f}%")

    except Exception as e:
        print(f"{seed:>8} | ERRO: {e}")

# ── Estatísticas ──────────────────────────────────────────────────────────
ap_euc  = np.array(res_euc)
ap_hyp  = np.array(res_hyp)
conv    = np.array(res_conv)

delta_euc_conv = (conv - ap_euc) / conv * 100
delta_hyp_conv = (conv - ap_hyp) / conv * 100
delta_hyp_euc  = (ap_euc - ap_hyp) / ap_euc * 100

_, p_euc  = ttest_ind(ap_euc, conv)
_, p_hyp  = ttest_ind(ap_hyp, conv)
_, p_hvse = ttest_ind(ap_hyp, ap_euc)

wins_euc     = int(np.sum(ap_euc < conv))
wins_hyp     = int(np.sum(ap_hyp < conv))
wins_hyp_euc = int(np.sum(ap_hyp < ap_euc))

print("=" * 65)
print("  RESULTADOS — ROBUSTEZ EUCLIDIANO vs HIPERBOLICO PURO")
print("=" * 65)
print(f"  AP Euclidiano  : {ap_euc.mean():.4f} +/- {ap_euc.std():.4f}")
print(f"  AP Hiperbolico : {ap_hyp.mean():.4f} +/- {ap_hyp.std():.4f}")
print(f"  Convencional   : {conv.mean():.4f} +/- {conv.std():.4f}")
print()
print(f"  AP Eucl vs Conv  : {delta_euc_conv.mean():+.1f}%  (p={p_euc:.4f})")
print(f"  AP Hyp  vs Conv  : {delta_hyp_conv.mean():+.1f}%  (p={p_hyp:.4f})")
print()
print(f"  Hiperbolico vs Euclidiano : {delta_hyp_euc.mean():+.1f}%  (p={p_hvse:.4f})")
print()
print(f"  AP Eucl vence Conv       : {wins_euc}/{N_SEEDS} seeds")
print(f"  AP Hyp  vence Conv       : {wins_hyp}/{N_SEEDS} seeds")
print(f"  AP Hyp  vence AP Eucl    : {wins_hyp_euc}/{N_SEEDS} seeds")
print("=" * 65)

if delta_hyp_euc.mean() > 2 and p_hvse < 0.05:
    print("\n  HIPOTESE CONFIRMADA: hiperbolico puro > euclidiano")
    print(f"  Ganho: {delta_hyp_euc.mean():+.1f}%")
elif abs(delta_hyp_euc.mean()) <= 2:
    print("\n  RESULTADO NEUTRO: diferenca nao significativa")
else:
    print("\n  HIPOTESE NEGADA: euclidiano ainda superior")

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

x = np.arange(len(res_euc))

axes[0].plot(x, ap_euc, "o-", color=GOLD,  lw=1.5, ms=4, label=f"AP Eucl ({delta_euc_conv.mean():+.0f}% vs Conv)")
axes[0].plot(x, ap_hyp, "s-", color=GOLD2, lw=1.5, ms=4, label=f"AP Hyp ({delta_hyp_conv.mean():+.0f}% vs Conv)")
axes[0].plot(x, conv,   "^-", color=GRAY,  lw=1.5, ms=4, label="Convencional")
axes[0].set_xlabel("Seed", color="#8B949E")
axes[0].set_ylabel("Instabilidade (menor = melhor)", color="#8B949E")
axes[0].set_title("Estabilidade por Seed", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7)
axes[0].grid(True, alpha=0.2)

bp = axes[1].boxplot(
    [ap_euc, ap_hyp, conv],
    tick_labels=["AP\nEucl", "AP\nHyp", "Conv"],
    patch_artist=True,
    medianprops=dict(color="white", lw=2)
)
for patch, color in zip(bp["boxes"], [GOLD, GOLD2, GRAY]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for w in bp["whiskers"]: w.set_color("#8B949E")
for c in bp["caps"]:     c.set_color("#8B949E")
axes[1].set_title("Distribuicao", color="#E6EDF3", fontweight="bold")
axes[1].set_ylabel("Instabilidade", color="#8B949E")
axes[1].grid(True, alpha=0.2)

axes[2].bar(x-0.2, delta_euc_conv, width=0.38, color=GOLD,  alpha=0.8, label="AP Eucl vs Conv")
axes[2].bar(x+0.2, delta_hyp_conv, width=0.38, color=GOLD2, alpha=0.8, label="AP Hyp vs Conv")
axes[2].axhline(0, color="white", lw=0.8, linestyle="--")
axes[2].set_xlabel("Seed", color="#8B949E")
axes[2].set_ylabel("Melhora vs Conv (%)", color="#8B949E")
axes[2].set_title("Ganho vs Convencional", color="#E6EDF3", fontweight="bold")
axes[2].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7)
axes[2].grid(True, alpha=0.2)

fig.suptitle(
    f"ALPHA PHI — Euclidiano vs Hiperbolico Puro (Poincare Ball)\n"
    f"phi={PHI:.4f}  Fibonacci{fib_layers}  20 seeds timestamp  Florianopolis 2026",
    color=GOLD, fontsize=11, fontweight="bold"
)

plt.tight_layout()
plt.savefig("alphaphi_robustez_hiperbolico.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGrafico salvo: alphaphi_robustez_hiperbolico.png")
print("alpha-phi")
