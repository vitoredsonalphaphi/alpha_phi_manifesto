# ALPHA PHI — Robustez v2 · Quarto Eixo
# Vitor Edson Delavi · Florianópolis · 2026
#
# Execute no Google Colab célula por célula
# copiando cada seção marcada com # ── CÉLULA N

# ── CÉLULA 1: Imports e Constantes ───────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import entropy as scipy_entropy, ttest_ind

PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084

print(f"phi = {PHI:.10f}")
print(f"alpha = {ALPHA:.10f}")
print(f"1/phi = {1/PHI:.10f}")
print("=" * 55)
print("QUARTO EIXO: Transformacao do Erro")
print("erro x (1/phi) -> reintegrado ao fluxo")
print("=" * 55)

# ── CÉLULA 2: Ativações e Fibonacci ──────────────────────────────────────
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

# ── CÉLULA 3: Arquiteturas ────────────────────────────────────────────────
def weight_entropy(weights):
    all_w = np.concatenate([W.flatten() for W in weights])
    hist, _ = np.histogram(all_w, bins=50, density=True)
    hist = hist + 1e-10
    hist /= hist.sum()
    return scipy_entropy(hist)

def activation_stability(activations):
    vars_ = [np.var(a) for a in activations]
    return np.mean(np.abs(np.diff(vars_)))

def make_weights(layer_sizes, scale_fn, seed):
    np.random.seed(seed)
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        scale  = scale_fn(fan_in)
        weights.append(np.random.randn(fan_in, layer_sizes[i+1]) * scale)
        biases.append(np.zeros(layer_sizes[i+1]))
    return weights, biases

def forward_original(X, weights, biases):
    acts = [X]
    cur  = X
    for W, b in zip(weights, biases):
        cur = golden_activation(cur @ W + b)
        acts.append(cur)
    return acts

def forward_residual(X, weights, biases):
    """Quarto Eixo: phi_residual_transform"""
    acts = [X]
    cur  = X
    for W, b in zip(weights, biases):
        entrada = cur
        z       = cur @ W + b
        direto  = golden_activation(z)
        if entrada.shape == direto.shape:
            residuo = entrada - direto
            cur     = direto + residuo * (1.0 / PHI)
        else:
            cur = direto
        acts.append(cur)
    return acts

def forward_conv(X, weights, biases):
    acts = [X]
    cur  = X
    for W, b in zip(weights, biases):
        cur = relu(cur @ W + b)
        acts.append(cur)
    return acts

print("Arquiteturas prontas:")
print("  1. Alpha-Phi Original")
print("  2. Alpha-Phi v2 Quarto Eixo (residual phi)")
print("  3. Convencional ReLU")

# ── CÉLULA 4: Experimento 20 seeds ───────────────────────────────────────
INPUT_DIM = 16
N_SAMPLES = 200
N_SEEDS   = 20

rng_master = np.random.RandomState(int(time.time()) % 100000)
SEEDS      = rng_master.randint(0, 99999, size=N_SEEDS).tolist()

print(f"Seeds gerados por timestamp: {SEEDS}")
print()

arch = [INPUT_DIM] + fib_layers     + [INPUT_DIM]
arch_con = [INPUT_DIM] + uniform_layers + [INPUT_DIM]

res_orig  = []
res_resid = []
res_conv  = []

print(f"{'Seed':>8} | {'AP Orig':>8} | {'AP v2':>8} | {'Conv':>8} | {'Ganho v2':>10}")
print("-" * 58)

for seed in SEEDS:
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, N_SAMPLES)
    X = np.column_stack([np.sin(t * (i+1) / PHI) for i in range(INPUT_DIM)])
    X = (X - X.mean()) / X.std()

    W_ap,  b_ap  = make_weights(arch,     lambda f: np.sqrt(1.0/(f*PHI)), seed)
    W_res, b_res = make_weights(arch,     lambda f: np.sqrt(1.0/(f*PHI)), seed)
    W_con, b_con = make_weights(arch_con, lambda f: np.sqrt(2.0/f),       seed)

    acts_orig  = forward_original(X, W_ap,  b_ap)
    acts_resid = forward_residual(X, W_res, b_res)
    acts_conv  = forward_conv(X,     W_con, b_con)

    s_orig  = activation_stability(acts_orig)
    s_resid = activation_stability(acts_resid)
    s_conv  = activation_stability(acts_conv)

    res_orig.append(s_orig)
    res_resid.append(s_resid)
    res_conv.append(s_conv)

    ganho = (s_conv - s_resid) / s_conv * 100
    print(f"{seed:>8} | {s_orig:>8.4f} | {s_resid:>8.4f} | {s_conv:>8.4f} | {ganho:>+9.1f}%")

# ── CÉLULA 5: Estatísticas ────────────────────────────────────────────────
ap_orig  = np.array(res_orig)
ap_resid = np.array(res_resid)
conv     = np.array(res_conv)

delta_orig  = (conv - ap_orig)  / conv * 100
delta_resid = (conv - ap_resid) / conv * 100
ganho_v2    = (ap_orig - ap_resid) / ap_orig * 100

_, p_orig  = ttest_ind(ap_orig,  conv)
_, p_resid = ttest_ind(ap_resid, conv)

wins_orig  = np.sum(ap_orig  < conv)
wins_resid = np.sum(ap_resid < conv)

print("=" * 60)
print("  RESULTADOS FINAIS — ALPHA PHI v2 · QUARTO EIXO")
print("=" * 60)
print(f"  AP Original  : {ap_orig.mean():.4f} +/- {ap_orig.std():.4f}")
print(f"  AP v2 (phi)  : {ap_resid.mean():.4f} +/- {ap_resid.std():.4f}")
print(f"  Convencional : {conv.mean():.4f} +/- {conv.std():.4f}")
print()
print(f"  AP Original melhora vs Conv : {delta_orig.mean():+.1f}%  (p={p_orig:.4f})")
print(f"  AP v2 melhora vs Conv       : {delta_resid.mean():+.1f}%  (p={p_resid:.4f})")
print()
print(f"  Ganho Quarto Eixo sobre Original : {ganho_v2.mean():+.1f}%")
print()
print(f"  AP Original vence Conv : {wins_orig}/{N_SEEDS} seeds")
print(f"  AP v2 vence Conv       : {wins_resid}/{N_SEEDS} seeds")
print("=" * 60)

# ── CÉLULA 6: Plots ───────────────────────────────────────────────────────
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

# Plot 1 — Estabilidade por seed
axes[0].plot(x, ap_orig,  "o-", color=GOLD,  lw=1.5, ms=4, label="AP Original")
axes[0].plot(x, ap_resid, "s-", color=GOLD2, lw=1.5, ms=4, label="AP v2 phi")
axes[0].plot(x, conv,     "^-", color=GRAY,  lw=1.5, ms=4, label="Convencional")
axes[0].set_xlabel("Seed", color="#8B949E")
axes[0].set_ylabel("Instabilidade (menor = melhor)", color="#8B949E")
axes[0].set_title("Estabilidade por Seed", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
axes[0].grid(True, alpha=0.2)

# Plot 2 — Boxplot
bp = axes[1].boxplot(
    [ap_orig, ap_resid, conv],
    tick_labels=["AP\nOriginal", "AP v2\nphi", "Conv"],
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

# Plot 3 — Melhora percentual
axes[2].bar(x - 0.2, delta_orig,  width=0.38, color=GOLD,  alpha=0.8, label="AP Original")
axes[2].bar(x + 0.2, delta_resid, width=0.38, color=GOLD2, alpha=0.8, label="AP v2 phi")
axes[2].axhline(0, color="white", lw=0.8, linestyle="--")
axes[2].set_xlabel("Seed", color="#8B949E")
axes[2].set_ylabel("Melhora vs Convencional (%)", color="#8B949E")
axes[2].set_title("Ganho por Seed", color="#E6EDF3", fontweight="bold")
axes[2].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
axes[2].grid(True, alpha=0.2)

fig.suptitle(
    f"ALPHA PHI v2 - Quarto Eixo: Transformacao do Erro\n"
    f"phi={PHI:.4f}  alpha=1/137  20 seeds por timestamp  Florianopolis 2026",
    color=GOLD, fontsize=11, fontweight="bold"
)

plt.tight_layout()
plt.savefig("alphaphi_quarto_eixo.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGrafico salvo: alphaphi_quarto_eixo.png")
print("Proximo passo: subir no GitHub - alpha-phi")
