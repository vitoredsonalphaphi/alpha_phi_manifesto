# ALPHA PHI — Robustez v3 · Quarto Eixo Corrigido
# Vitor Edson Delavi · Florianópolis · 2026
#
# Correção v3: projeção linear do resíduo quando dimensões diferem
# O erro é TRADUZIDO para a nova dimensão e reintegrado por 1/phi
# Em todas as camadas — sem exceção

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import entropy as scipy_entropy, ttest_ind

PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084

print(f"phi   = {PHI:.10f}")
print(f"alpha = {ALPHA:.10f}")
print(f"1/phi = {1/PHI:.10f}")
print("=" * 55)
print("QUARTO EIXO v3: Transformacao do Erro com Projecao")
print("erro traduzido para nova dimensao x (1/phi) -> reintegrado")
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

fib_layers     = fibonacci_sequence(4, start=8)  # [8, 13, 21, 34]
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
        # Projeção do resíduo: traduz erro de fan_in para fan_out
        # Escalonada por 1/phi para suavidade
        proj_scale = np.sqrt(1.0 / (fan_in * PHI))
        projections.append(np.random.randn(fan_in, fan_out) * proj_scale)
    return weights, biases, projections

# ── Forward passes ────────────────────────────────────────────────────────
def forward_original(X, weights, biases, projections):
    """Alpha-Phi original: phi·tanh(x/phi)"""
    acts = [X]
    cur  = X
    for W, b in zip(weights, biases):
        cur = golden_activation(cur @ W + b)
        acts.append(cur)
    return acts

def forward_residual(X, weights, biases, projections):
    """
    Quarto Eixo v3 — Transformação Residual com Projeção
    
    Para cada camada:
      direto     = phi·tanh(entrada @ W + b)
      residuo    = entrada - direto
                   SE dimensões iguais: direto
                   SE dimensões diferentes: projeta entrada para nova dim
      reescalado = residuo × (1/phi)
      saida      = direto + reescalado
    
    O erro nunca é descartado.
    É TRADUZIDO para a nova linguagem (dimensão)
    e reintegrado com peso 1/phi.
    """
    acts = [X]
    cur  = X
    for W, b, P in zip(weights, biases, projections):
        entrada = cur
        z       = cur @ W + b
        direto  = golden_activation(z)

        # Projeção do resíduo para a dimensão de saída
        entrada_projetada = entrada @ P  # traduz para nova dimensão
        residuo           = entrada_projetada - direto
        reescalado        = residuo * (1.0 / PHI)

        cur = direto + reescalado
        acts.append(cur)
    return acts

def forward_conv(X, weights, biases, projections):
    """Convencional: ReLU + He init"""
    acts = [X]
    cur  = X
    for W, b in zip(weights, biases):
        cur = relu(cur @ W + b)
        acts.append(cur)
    return acts

print("Arquiteturas v3 prontas:")
print("  1. Alpha-Phi Original     — phi·tanh(x/phi)")
print("  2. Alpha-Phi v3 Quarto Eixo — residuo projetado × (1/phi)")
print("  3. Convencional ReLU      — baseline")

# ── Experimento: 20 seeds por timestamp ──────────────────────────────────
INPUT_DIM = 16
N_SAMPLES = 200
N_SEEDS   = 20

rng_master = np.random.RandomState(int(time.time()) % 100000)
SEEDS      = rng_master.randint(0, 99999, size=N_SEEDS).tolist()

print(f"\nSeeds gerados por timestamp: {SEEDS}\n")

arch     = [INPUT_DIM] + fib_layers     + [INPUT_DIM]
arch_con = [INPUT_DIM] + uniform_layers + [INPUT_DIM]

res_orig  = []
res_resid = []
res_conv  = []

print(f"{'Seed':>8} | {'AP Orig':>8} | {'AP v3':>8} | {'Conv':>8} | {'Ganho v3':>10}")
print("-" * 58)

for seed in SEEDS:
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, N_SAMPLES)
    X = np.column_stack([np.sin(t * (i+1) / PHI) for i in range(INPUT_DIM)])
    X = (X - X.mean()) / X.std()

    W_ap,  b_ap,  P_ap  = make_weights(arch,     lambda f: np.sqrt(1.0/(f*PHI)), seed)
    W_res, b_res, P_res = make_weights(arch,     lambda f: np.sqrt(1.0/(f*PHI)), seed)
    W_con, b_con, P_con = make_weights(arch_con, lambda f: np.sqrt(2.0/f),       seed)

    acts_orig  = forward_original(X,  W_ap,  b_ap,  P_ap)
    acts_resid = forward_residual(X,  W_res, b_res, P_res)
    acts_conv  = forward_conv(X,      W_con, b_con, P_con)

    s_orig  = activation_stability(acts_orig)
    s_resid = activation_stability(acts_resid)
    s_conv  = activation_stability(acts_conv)

    res_orig.append(s_orig)
    res_resid.append(s_resid)
    res_conv.append(s_conv)

    ganho = (s_conv - s_resid) / s_conv * 100 if s_conv > 0 else 0
    print(f"{seed:>8} | {s_orig:>8.4f} | {s_resid:>8.4f} | {s_conv:>8.4f} | {ganho:>+9.1f}%")

# ── Estatísticas ──────────────────────────────────────────────────────────
ap_orig  = np.array(res_orig)
ap_resid = np.array(res_resid)
conv     = np.array(res_conv)

delta_orig  = (conv - ap_orig)  / conv * 100
delta_resid = (conv - ap_resid) / conv * 100
ganho_v3    = (ap_orig - ap_resid) / ap_orig * 100

_, p_orig  = ttest_ind(ap_orig,  conv)
_, p_resid = ttest_ind(ap_resid, conv)

wins_orig  = int(np.sum(ap_orig  < conv))
wins_resid = int(np.sum(ap_resid < conv))

print("=" * 60)
print("  RESULTADOS FINAIS — ALPHA PHI v3 · QUARTO EIXO")
print("=" * 60)
print(f"  AP Original   : {ap_orig.mean():.4f} +/- {ap_orig.std():.4f}")
print(f"  AP v3 (phi)   : {ap_resid.mean():.4f} +/- {ap_resid.std():.4f}")
print(f"  Convencional  : {conv.mean():.4f} +/- {conv.std():.4f}")
print()
print(f"  AP Original melhora vs Conv : {delta_orig.mean():+.1f}%  (p={p_orig:.4f})")
print(f"  AP v3 melhora vs Conv       : {delta_resid.mean():+.1f}%  (p={p_resid:.4f})")
print()
print(f"  Ganho Quarto Eixo v3 sobre Original : {ganho_v3.mean():+.1f}%")
print()
print(f"  AP Original vence Conv : {wins_orig}/{N_SEEDS} seeds")
print(f"  AP v3 vence Conv       : {wins_resid}/{N_SEEDS} seeds")
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

# Plot 1 — Estabilidade por seed
axes[0].plot(x, ap_orig,  "o-", color=GOLD,  lw=1.5, ms=4, label="AP Original")
axes[0].plot(x, ap_resid, "s-", color=GOLD2, lw=1.5, ms=4, label="AP v3 phi-residual")
axes[0].plot(x, conv,     "^-", color=GRAY,  lw=1.5, ms=4, label="Convencional")
axes[0].set_xlabel("Seed", color="#8B949E")
axes[0].set_ylabel("Instabilidade (menor = melhor)", color="#8B949E")
axes[0].set_title("Estabilidade por Seed", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
axes[0].grid(True, alpha=0.2)

# Plot 2 — Boxplot
bp = axes[1].boxplot(
    [ap_orig, ap_resid, conv],
    tick_labels=["AP\nOriginal", "AP v3\nphi", "Conv"],
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
axes[2].bar(x + 0.2, delta_resid, width=0.38, color=GOLD2, alpha=0.8, label="AP v3 phi")
axes[2].axhline(0, color="white", lw=0.8, linestyle="--")
axes[2].set_xlabel("Seed", color="#8B949E")
axes[2].set_ylabel("Melhora vs Convencional (%)", color="#8B949E")
axes[2].set_title("Ganho por Seed", color="#E6EDF3", fontweight="bold")
axes[2].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
axes[2].grid(True, alpha=0.2)

fig.suptitle(
    f"ALPHA PHI v3 - Quarto Eixo: Transformacao do Erro com Projecao\n"
    f"phi={PHI:.4f}  1/phi={1/PHI:.4f}  20 seeds timestamp  Florianopolis 2026",
    color=GOLD, fontsize=11, fontweight="bold"
)

plt.tight_layout()
plt.savefig("alphaphi_quarto_eixo_v3.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGrafico salvo: alphaphi_quarto_eixo_v3.png")
print("Alpha-Phi v3 - Quarto Eixo operacional - alpha-phi")
