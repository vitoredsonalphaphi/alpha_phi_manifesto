# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Eco_Phyllotaxis_V3.py
Vitor Edson Delavi · Florianópolis · 2026

Hipótese (origem: análise de formações geométricas, abril 2026):

    A formação "girassol" de Avebury Trusloe usa φ não como escalar,
    mas como gerador de ângulo — o ângulo áureo α = 2π/φ² ≈ 137.508°.
    Cada semente n está a n·α graus da anterior, garantindo que nenhuma
    direção se repete (α/2π é irracional).

    O eco fonônico V2 aplica φ como escalar uniforme a todos os bins:
        reflexao = ifft(|freq| * exp(i * angle(freq) * k))
        s = s + (reflexao - X) * PHI

    V3 aplica φ como progressão angular por bin — filotaxia espectral:
        phase_offset[n] = n * golden_angle
        reflexao = ifft(|freq| * exp(i * (angle(freq) * k + phase_offset)))
        s = s + (reflexao - X) * PHI

    Se V3 > V2 com p < 0.05: hipótese tem suporte empírico.
    Se V3 ≤ V2: correlação era conveniente. Registrar como negativo honesto.

Protocolo: 20 seeds × timestamp, mesmo dataset que V2 (série φ vs ruído).
"""

import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils_phi import PHI, golden_activation, clip_grad, sigmoid

# ── Parâmetros ────────────────────────────────────────────────────────────────
N_SEEDS  = 20
N_EPOCHS = 60
N_TRAIN  = 400
N_TEST   = 100
DIM      = 128
HIDDEN   = 89
LR       = 0.01
N_ECO    = 3
K_MIN    = np.sqrt(2)

GOLDEN_ANGLE = 2 * np.pi / PHI**2   # ≈ 2.3999... rad ≈ 137.508°

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print("Experimento: Eco Filotáxico V3 — ângulo áureo por bin de frequência")
print(f"V2: coupling φ uniforme  |  V3: coupling φ + phase_offset[n] = n·α")
print(f"Ângulo áureo: {np.degrees(GOLDEN_ANGLE):.4f}° = 2π/φ²")
print(f"Timestamp: {TIMESTAMP}\n")

# ── Dados (idêntico ao V2) ─────────────────────────────────────────────────

def gerar_serie_phi(n, dim, rng):
    t = np.linspace(0, 4 * np.pi, dim)
    X = np.zeros((n, dim))
    for i in range(n):
        freq_base = rng.uniform(0.5, 2.0)
        sinal = np.zeros(dim)
        for k in range(5):
            freq_k = freq_base * (PHI ** k)
            amp_k  = rng.uniform(0.3, 1.0) / (k + 1)
            fase_k = rng.uniform(0, 2 * np.pi)
            sinal += amp_k * np.sin(freq_k * t + fase_k)
        ruido = rng.normal(0, 0.1, dim)
        sinal = (sinal + ruido) / (np.std(sinal + ruido) + 1e-8)
        X[i] = sinal
    return X

def gerar_ruido(n, dim, rng):
    X = rng.normal(0, 1, (n, dim))
    return X / (np.std(X, axis=1, keepdims=True) + 1e-8)

def gerar_dados(seed):
    rng = np.random.default_rng(seed)
    n_tr, n_te = N_TRAIN // 2, N_TEST // 2
    X_tr = np.vstack([gerar_serie_phi(n_tr, DIM, rng), gerar_ruido(n_tr, DIM, rng)])
    y_tr = np.array([1]*n_tr + [0]*n_tr, dtype=float)
    X_te = np.vstack([gerar_serie_phi(n_te, DIM, rng), gerar_ruido(n_te, DIM, rng)])
    y_te = np.array([1]*n_te + [0]*n_te, dtype=float)
    idx_tr = rng.permutation(N_TRAIN)
    idx_te = rng.permutation(N_TEST)
    return X_tr[idx_tr], y_tr[idx_tr], X_te[idx_te], y_te[idx_te]

# ── Campo coletivo ─────────────────────────────────────────────────────────

def medir_campo(X):
    freq_batch = np.fft.fft(X, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)
    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia   = -np.sum(amp_norm * np.log(amp_norm))
    coh_campo  = float(1.0 - entropia / np.log(X.shape[-1]))
    return K_MIN + (PHI - K_MIN) * coh_campo, coh_campo

# ── Funções de eco ─────────────────────────────────────────────────────────

def eco_v2(X, n_eco=N_ECO):
    """V2: coupling = φ escalar uniforme."""
    X = np.asarray(X, dtype=float)
    k, _ = medir_campo(X)
    s = X.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) * PHI
    return s, k

def eco_v3_phyllotaxis(X, n_eco=N_ECO):
    """V3: coupling = φ + deslocamento de fase por bin seguindo ângulo áureo.

    Hipótese filotáxica: cada bin n recebe fase adicional n·α, onde
    α = 2π/φ² é o ângulo áureo. Isso distribui as rotações de fase
    de forma que nenhum bin repete direção — exatamente o princípio
    do girassol de Avebury Trusloe.
    """
    X = np.asarray(X, dtype=float)
    k, _ = medir_campo(X)
    N = X.shape[-1]
    phase_offset = np.arange(N) * GOLDEN_ANGLE   # progressão áurea por bin
    s = X.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(s, axis=-1)
        # fase = rotação coletiva k + progressão filotáxica por bin
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * (np.angle(freq) * k + phase_offset)),
            axis=-1))
        s = s + (reflexao - X) * PHI
    return s, k

# ── Rede neural ────────────────────────────────────────────────────────────

def treinar(X_tr, y_tr, X_te, y_te, seed):
    dim_in = X_tr.shape[1]
    rng = np.random.default_rng(seed + 999)
    W1 = rng.normal(0, np.sqrt(2.0 / dim_in),  (dim_in, HIDDEN))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, np.sqrt(2.0 / HIDDEN), (HIDDEN, 1))
    b2 = np.zeros(1)
    for _ in range(N_EPOCHS):
        idx = rng.permutation(len(X_tr))
        for i in range(0, len(X_tr) - 31, 32):
            Xb, yb = X_tr[idx[i:i+32]], y_tr[idx[i:i+32]]
            z1 = Xb @ W1 + b1
            a1 = golden_activation(z1)
            yh = sigmoid(a1 @ W2 + b2).squeeze()
            dL = (yh - yb) / len(yb)
            dW2 = a1.T @ dL.reshape(-1, 1)
            db2 = dL.sum(keepdims=True)
            dz1 = (dL.reshape(-1, 1) * W2.T) * (1 - np.tanh(z1 / PHI)**2)
            W1 -= LR * clip_grad(Xb.T @ dz1)
            b1 -= LR * np.clip(dz1.sum(axis=0), -1, 1)
            W2 -= LR * clip_grad(dW2)
            b2 -= LR * np.clip(db2, -1, 1)
    yh_te = sigmoid(golden_activation(X_te @ W1 + b1) @ W2 + b2).squeeze()
    return float(np.mean((yh_te >= 0.5) == y_te))

# ── Execução ───────────────────────────────────────────────────────────────

res_G, res_v2, res_v3 = [], [], []
k_log = []

print(f"{'Seed':>14} {'G':>8} {'V2(φ)':>8} {'V3(α)':>8} {'Δ v3-v2':>9}")
print("-" * 55)

for seed in SEEDS:
    X_tr, y_tr, X_te, y_te = gerar_dados(seed)

    Xv2_tr, k_tr = eco_v2(X_tr)
    Xv2_te, _    = eco_v2(X_te)
    Xv3_tr, _    = eco_v3_phyllotaxis(X_tr)
    Xv3_te, _    = eco_v3_phyllotaxis(X_te)

    acc_G  = treinar(X_tr,    y_tr, X_te,    y_te, seed)
    acc_v2 = treinar(Xv2_tr,  y_tr, Xv2_te,  y_te, seed)
    acc_v3 = treinar(Xv3_tr,  y_tr, Xv3_te,  y_te, seed)

    res_G.append(acc_G)
    res_v2.append(acc_v2)
    res_v3.append(acc_v3)
    k_log.append(k_tr)

    print(f"{seed:>14} {acc_G:>8.3f} {acc_v2:>8.3f} {acc_v3:>8.3f} {acc_v3-acc_v2:>+9.3f}")

# ── Estatísticas ───────────────────────────────────────────────────────────

G   = np.array(res_G)
V2  = np.array(res_v2)
V3  = np.array(res_v3)
k_m = float(np.mean(k_log))

_, p_v2_g  = stats.wilcoxon(V2, G)
_, p_v3_g  = stats.wilcoxon(V3, G)
_, p_v3_v2 = stats.wilcoxon(V3, V2)

print(f"\n{'='*55}")
print(f"  G  (baseline):   {G.mean():.4f}")
print(f"  V2 (φ escalar):  {V2.mean():.4f}  Δ={V2.mean()-G.mean():+.4f}  p={p_v2_g:.6f}")
print(f"  V3 (α por bin):  {V3.mean():.4f}  Δ={V3.mean()-G.mean():+.4f}  p={p_v3_g:.6f}")
print(f"\n  V3 vs V2: Δ={V3.mean()-V2.mean():+.4f}  p={p_v3_v2:.6f}")
print(f"  k_campo médio: {k_m:.5f}  (√2={np.sqrt(2):.5f}, φ={PHI:.5f})")

if V3.mean() > V2.mean() and p_v3_v2 < 0.05:
    print(f"\n  ✅ V3 supera V2 — hipótese filotáxica tem suporte empírico")
elif V3.mean() > V2.mean():
    print(f"\n  ⚠️  V3 > V2 mas p={p_v3_v2:.4f} — diferença não significativa")
else:
    print(f"\n  ❌ V3 ≤ V2 — correlação geométrica era conveniente, não funcional")
    print(f"     Registrado como negativo honesto.")

# ── Visualização ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#0d1117")

ax1, ax2 = axes

# Boxplot comparativo
ax1.set_facecolor("#161b22")
bp = ax1.boxplot([G, V2, V3],
                 labels=["G\n(baseline)", "V2\n(φ escalar)", "V3\n(ângulo áureo)"],
                 patch_artist=True,
                 medianprops=dict(color="#E6EDF3", linewidth=2))
colors = ["#888888", "#FF8C00", "#DAA520"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_title("Distribuição de acurácia por modo", color="#E6EDF3", fontsize=11)
ax1.set_ylabel("Acurácia", color="#E6EDF3")
ax1.tick_params(colors="#8B949E")
for spine in ax1.spines.values():
    spine.set_edgecolor("#30363d")

# Evolução por seed
ax2.set_facecolor("#161b22")
seeds_idx = range(N_SEEDS)
ax2.plot(seeds_idx, G,  color="#888888", alpha=0.6, label=f"G  {G.mean():.4f}", linewidth=1.5)
ax2.plot(seeds_idx, V2, color="#FF8C00", alpha=0.8, label=f"V2 {V2.mean():.4f}", linewidth=2)
ax2.plot(seeds_idx, V3, color="#DAA520", alpha=0.9, label=f"V3 {V3.mean():.4f}", linewidth=2, linestyle="--")
ax2.axhline(V2.mean(), color="#FF8C00", linestyle=":", alpha=0.4)
ax2.axhline(V3.mean(), color="#DAA520", linestyle=":", alpha=0.4)
ax2.set_title(f"V3 vs V2: Δ={V3.mean()-V2.mean():+.4f}  p={p_v3_v2:.4f}", color="#E6EDF3", fontsize=11)
ax2.set_xlabel("Seed (por ordem)", color="#8B949E")
ax2.set_ylabel("Acurácia", color="#E6EDF3")
ax2.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=9)
ax2.tick_params(colors="#8B949E")
for spine in ax2.spines.values():
    spine.set_edgecolor("#30363d")

plt.suptitle("Eco Filotáxico V3 — ângulo áureo α = 2π/φ² por bin FFT",
             color="#E6EDF3", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("phyllotaxis_v3_results.png", dpi=140, bbox_inches="tight",
            facecolor="#0d1117")
print("\nGráfico salvo: phyllotaxis_v3_results.png")

# ── Salvar resultados ──────────────────────────────────────────────────────

resultados = {
    "timestamp": TIMESTAMP,
    "golden_angle_rad": float(GOLDEN_ANGLE),
    "golden_angle_deg": float(np.degrees(GOLDEN_ANGLE)),
    "G":  {"mean": float(G.mean()),  "std": float(G.std())},
    "V2": {"mean": float(V2.mean()), "std": float(V2.std()),
           "delta_G": float(V2.mean()-G.mean()), "p_vs_G": float(p_v2_g)},
    "V3": {"mean": float(V3.mean()), "std": float(V3.std()),
           "delta_G": float(V3.mean()-G.mean()), "p_vs_G": float(p_v3_g),
           "delta_V2": float(V3.mean()-V2.mean()), "p_vs_V2": float(p_v3_v2)},
    "k_campo_medio": k_m,
    "hipotese_confirmada": bool(V3.mean() > V2.mean() and p_v3_v2 < 0.05),
}

with open("phyllotaxis_v3_results.json", "w") as f:
    json.dump(resultados, f, indent=2)
print("Resultados salvos: phyllotaxis_v3_results.json")
