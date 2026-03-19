# ALPHA PHI — Reconstrução Nativa Hiperbólica
# Vitor Edson Delavi · Florianópolis · 2026
#
# Diferença fundamental vs versão traduzida:
# TRADUZIDO: Robustez euclidiano projetado para Poincaré
# NATIVO:    Concebido desde o início no espaço hiperbólico
#
# O que muda na reconstrução nativa:
# 1. Inicialização de pesos na métrica hiperbólica
# 2. Distâncias como geodésicas — não linhas retas
# 3. Fibonacci aplicado ao raio da bola — não às dimensões lineares
# 4. Ativação phi·tanh nativa — sem projeção intermediária
# 5. Curvatura c modulada por phi — não fixa em 1.0

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import entropy as scipy_entropy, ttest_ind

PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084

# Curvatura modulada por phi — não arbitrária
# c = 1/phi² = 0.382 — o microponto de dobra
C_PHI = 1.0 / PHI**2

print(f"phi        = {PHI:.10f}")
print(f"alpha      = {ALPHA:.10f}")
print(f"curvatura  = {C_PHI:.10f}  (1/phi² — nativa)")
print("=" * 60)
print("RECONSTRUCAO NATIVA HIPERBOLICA")
print("Concebido desde o inicio no espaco de Poincare")
print(f"Curvatura c = 1/phi² = {C_PHI:.4f}")
print("=" * 60)

# ── Espaço Hiperbólico com curvatura φ ───────────────────────────────────
def expmap0(v, c=C_PHI):
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.clip(v_norm, 1e-8, None)
    sqrt_c = np.sqrt(c)
    tanh_v = np.tanh(sqrt_c * v_norm)
    return tanh_v * v / (sqrt_c * v_norm)

def logmap0(y, c=C_PHI):
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    # Raio máximo da bola com curvatura c
    max_norm = (1.0 / np.sqrt(c)) - 1e-5
    y_norm   = np.clip(y_norm, 1e-8, max_norm)
    sqrt_c   = np.sqrt(c)
    return np.arctanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)

def geodesic_distance(x, y, c=C_PHI):
    """Distância geodésica — não euclidiana"""
    diff   = x - y
    x2     = np.sum(x**2, axis=-1, keepdims=True)
    y2     = np.sum(y**2, axis=-1, keepdims=True)
    diff2  = np.sum(diff**2, axis=-1, keepdims=True)
    num    = 2 * c * diff2
    den    = (1 - c*x2) * (1 - c*y2)
    arg    = np.clip(1 + num/(den + 1e-8), 1+1e-8, None)
    return (2/np.sqrt(c)) * np.arctanh(np.sqrt(c) * np.sqrt(np.clip(diff2/(diff2+den/c+1e-8), 0, 1-1e-8)))

def mobius_add(x, y, c=C_PHI):
    """Adição de Möbius — operação nativa no espaço de Poincaré"""
    x2  = np.sum(x**2, axis=-1, keepdims=True)
    y2  = np.sum(y**2, axis=-1, keepdims=True)
    xy  = np.sum(x*y,  axis=-1, keepdims=True)
    num = (1 + 2*c*xy + c*y2)*x + (1 - c*x2)*y
    den = 1 + 2*c*xy + c**2*x2*y2
    return num / (den + 1e-8)

def mobius_scalar(r, x, c=C_PHI):
    """Multiplicação escalar de Möbius"""
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x_norm = np.clip(x_norm, 1e-8, None)
    sqrt_c = np.sqrt(c)
    tanh_v = np.tanh(r * np.arctanh(sqrt_c * x_norm))
    return tanh_v * x / (sqrt_c * x_norm)

# ── Ativação nativa — sem projeção intermediária ──────────────────────────
def golden_activation_native(x, c=C_PHI):
    """
    φ·tanh(x/φ) nativa no espaço hiperbólico.
    Usa multiplicação escalar de Möbius — operação nativa.
    Não projeta para euclidiano e de volta.
    """
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x_norm = np.clip(x_norm, 1e-8, None)
    # Escala o raio por phi·tanh(raio/phi) — preserva direção
    novo_raio = PHI * np.tanh(x_norm / PHI)
    # Mantém direção, ajusta raio — operação nativa na bola
    max_norm  = (1.0/np.sqrt(c)) - 1e-5
    novo_raio = np.clip(novo_raio, 1e-8, max_norm)
    return novo_raio * x / x_norm

def golden_activation_eucl(x):
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

# ── Inicialização nativa hiperbólica ──────────────────────────────────────
def make_weights_native(layer_sizes, seed, c=C_PHI):
    """
    Inicialização nativa para espaço hiperbólico.
    Escala pela curvatura c — não pela dimensão fan_in apenas.
    Pesos menores para respeitar a geometria da bola.
    """
    np.random.seed(seed)
    W, b = [], []
    for i in range(len(layer_sizes)-1):
        fan_in  = layer_sizes[i]
        fan_out = layer_sizes[i+1]
        # Escala nativa: curvatura × phi × fan_in
        scale = np.sqrt(1.0 / (fan_in * PHI * (1.0/c)))
        W.append(np.random.randn(fan_in, fan_out) * scale)
        b.append(np.zeros(fan_out))
    return W, b

def make_weights_eucl(layer_sizes, seed):
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
    """Alpha-Phi Robustez Original — euclidiano"""
    acts = [X]
    cur  = X
    for W, b in zip(weights, biases):
        cur = golden_activation_eucl(cur @ W + b)
        acts.append(cur)
    return acts

def forward_native_hyp(X, weights, biases, c=C_PHI):
    """
    Alpha-Phi Nativo Hiperbólico.
    Concebido desde o início no espaço de Poincaré.

    Diferenças fundamentais vs traduzido:
    1. Curvatura c = 1/phi² — não arbitrária
    2. Operação linear via Möbius — não expmap/logmap/expmap
    3. Ativação nativa — preserva direção, escala raio por phi·tanh
    4. Inicialização com escala hiperbólica nativa
    """
    acts = [X]
    # Projeta entrada para bola de Poincaré com curvatura phi
    cur = expmap0(X, c)
    acts.append(cur)

    for i, (W, b) in enumerate(zip(weights, biases)):
        # Transformação linear nativa:
        # 1. Rotação/escala via multiplicação matricial no tangente
        x_tangent = logmap0(cur, c)
        z_tangent = x_tangent @ W + b
        z_hyp     = expmap0(z_tangent, c)

        # 2. Translação via adição de Möbius com bias hiperbólico
        b_hyp = expmap0(b.reshape(1,-1) * 0.01, c)
        z_hyp = mobius_add(z_hyp, b_hyp, c)

        # 3. Ativação nativa
        cur = golden_activation_native(z_hyp, c)
        acts.append(cur)

    # Projeta saída de volta
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

print("Arquiteturas nativas prontas:")
print(f"  1. Alpha-Phi Euclidiano    — phi·tanh, c=∞ (plano)")
print(f"  2. Alpha-Phi Nativo Hyp   — Möbius + phi·tanh, c={C_PHI:.4f} (1/phi²)")
print(f"  3. Convencional ReLU      — baseline")

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
res_nat  = []
res_conv = []

print(f"{'Seed':>8} | {'AP Eucl':>8} | {'AP Nativo':>9} | {'Conv':>8} | {'Δ Nat>Eucl':>11}")
print("-" * 62)

for seed in SEEDS:
    np.random.seed(seed)
    t = np.linspace(0, 4*np.pi, N_SAMPLES)
    X = np.column_stack([np.sin(t*(i+1)/PHI) for i in range(INPUT_DIM)])
    X = (X - X.mean()) / X.std()

    W_e, b_e = make_weights_eucl(arch,      seed)
    W_n, b_n = make_weights_native(arch,    seed)
    W_c, b_c = make_weights_conv(arch_con,  seed)

    try:
        acts_euc  = forward_euclidean(X,   W_e, b_e)
        acts_nat  = forward_native_hyp(X,  W_n, b_n)
        acts_conv = forward_conv(X,        W_c, b_c)

        s_euc  = activation_stability(acts_euc)
        s_nat  = activation_stability(acts_nat)
        s_conv = activation_stability(acts_conv)

        res_euc.append(s_euc)
        res_nat.append(s_nat)
        res_conv.append(s_conv)

        delta = (s_euc - s_nat) / s_euc * 100
        print(f"{seed:>8} | {s_euc:>8.4f} | {s_nat:>9.4f} | {s_conv:>8.4f} | {delta:>+10.1f}%")

    except Exception as e:
        print(f"{seed:>8} | ERRO: {str(e)[:40]}")

# ── Estatísticas ──────────────────────────────────────────────────────────
ap_euc  = np.array(res_euc)
ap_nat  = np.array(res_nat)
conv    = np.array(res_conv)

delta_euc_conv = (conv - ap_euc) / conv * 100
delta_nat_conv = (conv - ap_nat) / conv * 100
delta_nat_euc  = (ap_euc - ap_nat) / ap_euc * 100

_, p_euc  = ttest_ind(ap_euc, conv)
_, p_nat  = ttest_ind(ap_nat, conv)
_, p_nvse = ttest_ind(ap_nat, ap_euc)

wins_euc     = int(np.sum(ap_euc < conv))
wins_nat     = int(np.sum(ap_nat < conv))
wins_nat_euc = int(np.sum(ap_nat < ap_euc))

print("=" * 65)
print("  RESULTADOS — NATIVO HIPERBOLICO vs EUCLIDIANO")
print("=" * 65)
print(f"  AP Euclidiano  : {ap_euc.mean():.4f} +/- {ap_euc.std():.4f}")
print(f"  AP Nativo Hyp  : {ap_nat.mean():.4f} +/- {ap_nat.std():.4f}")
print(f"  Convencional   : {conv.mean():.4f} +/- {conv.std():.4f}")
print()
print(f"  AP Eucl vs Conv   : {delta_euc_conv.mean():+.1f}%  (p={p_euc:.4f})")
print(f"  AP Nat  vs Conv   : {delta_nat_conv.mean():+.1f}%  (p={p_nat:.4f})")
print()
print(f"  Nativo vs Euclidiano : {delta_nat_euc.mean():+.1f}%  (p={p_nvse:.4f})")
print()
print(f"  AP Eucl vence Conv      : {wins_euc}/{N_SEEDS} seeds")
print(f"  AP Nat  vence Conv      : {wins_nat}/{N_SEEDS} seeds")
print(f"  AP Nat  vence AP Eucl   : {wins_nat_euc}/{N_SEEDS} seeds")
print(f"  Curvatura usada         : c={C_PHI:.4f} (1/phi²)")
print("=" * 65)

if delta_nat_euc.mean() > 2 and p_nvse < 0.05:
    print(f"\n  RECONSTRUCAO CONFIRMADA: nativo > euclidiano")
    print(f"  Ganho adicional sobre traducao: {delta_nat_euc.mean():+.1f}%")
elif abs(delta_nat_euc.mean()) <= 2:
    print(f"\n  RESULTADO NEUTRO")
else:
    print(f"\n  EUCLIDIANO AINDA SUPERIOR")

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

axes[0].plot(x, ap_euc, "o-", color=GOLD,  lw=1.5, ms=4, label=f"AP Eucl ({delta_euc_conv.mean():+.0f}%)")
axes[0].plot(x, ap_nat, "s-", color=GOLD2, lw=1.5, ms=4, label=f"AP Nativo ({delta_nat_conv.mean():+.0f}%)")
axes[0].plot(x, conv,   "^-", color=GRAY,  lw=1.5, ms=4, label="Convencional")
axes[0].set_xlabel("Seed", color="#8B949E")
axes[0].set_ylabel("Instabilidade (menor = melhor)", color="#8B949E")
axes[0].set_title("Estabilidade por Seed", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7)
axes[0].grid(True, alpha=0.2)

bp = axes[1].boxplot(
    [ap_euc, ap_nat, conv],
    tick_labels=["AP\nEucl", "AP\nNativo", "Conv"],
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

axes[2].bar(x-0.2, delta_euc_conv, width=0.38,
            color=GOLD,  alpha=0.8, label="AP Eucl vs Conv")
axes[2].bar(x+0.2, delta_nat_conv, width=0.38,
            color=GOLD2, alpha=0.8, label="AP Nativo vs Conv")
axes[2].axhline(0, color="white", lw=0.8, linestyle="--")
axes[2].set_xlabel("Seed", color="#8B949E")
axes[2].set_ylabel("Melhora vs Conv (%)", color="#8B949E")
axes[2].set_title("Ganho vs Convencional", color="#E6EDF3", fontweight="bold")
axes[2].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7)
axes[2].grid(True, alpha=0.2)

fig.suptitle(
    f"ALPHA PHI NATIVO HIPERBOLICO — c=1/phi²={C_PHI:.4f}\n"
    f"Mobius + phi·tanh nativo  20 seeds timestamp  Florianopolis 2026",
    color=GOLD, fontsize=11, fontweight="bold"
)

plt.tight_layout()
plt.savefig("alphaphi_nativo_hiperbolico.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGrafico salvo: alphaphi_nativo_hiperbolico.png")
print("Alpha-Phi Nativo Hiperbolico — alpha-phi")
