# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Eco_Fononico_V2.py
Vitor Edson Delavi · Florianópolis · 2026

Descoberta do mapeamento de acoplamento (abril 2026):
    eco_fononico v1 usa coupling = 1/k ≈ 0.705 → 92.80%
    Mapeamento revelou zona ótima em coupling ∈ [1.40, 3.00] → 97.8-98.3%
    φ = 1.618 está na zona ótima → 97.9%

eco_fononico v2:
    k do campo (rotação de fase) → √2, como antes
    coupling (reinjeção) → φ fixo

    sinal = sinal + (reflexao - X) * φ

Significado:
    √2: escala natural de rotação (encontrada pelo campo)
    φ:  proporção natural de reinjeção (razão áurea como coupling)

    Os dois valores fundamentais do projeto — cada um na sua função.
    √2 como rotação, φ como acoplamento. α-φ realizado de forma diferente
    do esperado: não α como coupling, mas φ+√2 como dupla funcional.

Visualizações 3D geradas:
    1. Superfície: coupling × seed × acurácia (landscape de estabilidade)
    2. Clusters PCA: sinal bruto vs eco_v1 vs eco_v2 em espaço 3D
    3. Curva de acoplamento com zona ótima destacada

Protocolo: seeds por timestamp, resultados integralmente reportados.
"""

import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from utils_phi import PHI, ALPHA, C_PHI, golden_activation, clip_grad, sigmoid

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

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

PLOT_COLORS = {
    "gold": "#DAA520", "phi": "#FF8C00", "blue": "#4169E1",
    "green": "#2ECC71", "red": "#E74C3C", "gray": "#888888",
    "bg": "#0d1117", "panel": "#161b22", "text": "#8B949E", "title": "#E6EDF3",
}

print("Experimento: Eco Ressonante Fonônico V2 — φ como coupling")
print(f"V1: coupling = 1/k ≈ 0.705  |  V2: coupling = φ = {PHI:.6f}")
print(f"Timestamp: {TIMESTAMP}\n")

# ── Dados ─────────────────────────────────────────────────────────────────────

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

# ── Campo coletivo ────────────────────────────────────────────────────────────

def medir_campo(X):
    freq_batch = np.fft.fft(X, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)
    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia   = -np.sum(amp_norm * np.log(amp_norm))
    coh_campo  = float(1.0 - entropia / np.log(X.shape[-1]))
    k          = K_MIN + (PHI - K_MIN) * coh_campo
    return k, coh_campo

# ── Funções de eco ────────────────────────────────────────────────────────────

def eco_fononico_v1(X, n_eco=N_ECO):
    """V1: coupling = 1/k (original)."""
    X = np.asarray(X, dtype=float)
    k, coh = medir_campo(X)
    s = X.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) / k
    return s, k, coh

def eco_fononico_v2(X, n_eco=N_ECO):
    """V2: coupling = φ (razão áurea como acoplamento de reinjeção)."""
    X = np.asarray(X, dtype=float)
    k, coh = medir_campo(X)
    s = X.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) * PHI   # φ como coupling
    return s, k, coh

# ── Rede neural ───────────────────────────────────────────────────────────────

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

# ── Execução ──────────────────────────────────────────────────────────────────

res   = {"G": [], "G_eco_v1": [], "G_eco_v2_phi": []}
k_log = []
acc_por_seed = {"G": [], "G_eco_v1": [], "G_eco_v2_phi": []}

print(f"{'Seed':<14} {'G':>8} {'G_eco_v1':>10} {'G_eco_v2_φ':>12} {'Δ v2-v1':>10}")
print("-" * 58)

for seed in SEEDS:
    X_tr, y_tr, X_te, y_te = gerar_dados(seed)

    Xv1_tr, k_tr, _ = eco_fononico_v1(X_tr)
    Xv1_te, _,    _ = eco_fononico_v1(X_te)
    Xv2_tr, _,    _ = eco_fononico_v2(X_tr)
    Xv2_te, _,    _ = eco_fononico_v2(X_te)

    acc_G  = treinar(X_tr,    y_tr, X_te,    y_te, seed)
    acc_v1 = treinar(Xv1_tr,  y_tr, Xv1_te,  y_te, seed)
    acc_v2 = treinar(Xv2_tr,  y_tr, Xv2_te,  y_te, seed)

    res["G"].append(acc_G)
    res["G_eco_v1"].append(acc_v1)
    res["G_eco_v2_phi"].append(acc_v2)
    k_log.append(k_tr)

    print(f"{seed:<14} {acc_G:>8.3f} {acc_v1:>10.3f} {acc_v2:>12.3f} {acc_v2-acc_v1:>+10.3f}")

# ── Estatísticas ──────────────────────────────────────────────────────────────

print("\n" + "=" * 58)
G   = np.array(res["G"])
V1  = np.array(res["G_eco_v1"])
V2  = np.array(res["G_eco_v2_phi"])

_, p_v1 = stats.wilcoxon(V1, G)
_, p_v2 = stats.wilcoxon(V2, G)
_, p_comp = stats.wilcoxon(V2, V1)

print(f"\n{'Modo':<18} {'Média':>8} {'Desvio':>8} {'Δ vs G':>10} {'p-valor':>10}")
print("-" * 58)
print(f"{'G (base)':<18} {G.mean():>8.4f} {G.std():>8.4f} {'—':>10} {'—':>10}")
print(f"{'G_eco_v1 (1/k)':<18} {V1.mean():>8.4f} {V1.std():>8.4f} "
      f"{V1.mean()-G.mean():>+10.4f} {p_v1:>10.6f}")
print(f"{'G_eco_v2 (φ)':<18} {V2.mean():>8.4f} {V2.std():>8.4f} "
      f"{V2.mean()-G.mean():>+10.4f} {p_v2:>10.6f}")

sinal = "✅" if V2.mean() > V1.mean() and p_comp < 0.05 else "≈"
print(f"\nV2 vs V1: Δ={V2.mean()-V1.mean():+.4f}  p={p_comp:.6f}  {sinal}")
print(f"k_otimo médio: {np.mean(k_log):.4f} ≈ √2={np.sqrt(2):.4f}")

# ── Visualizações 3D ──────────────────────────────────────────────────────────

print("\nGerando visualizações 3D...")

# ── Plot 1: Superfície coupling × seed × acurácia ─────────────────────────────
# Usa dados do mapeamento de zona (valores representativos)
couplings_plot = [0.30, 0.50, 0.60, 0.705, 0.90, 1.00, 1.20,
                  np.sqrt(2), 1.50, PHI, 1.80, 2.00, 2.50, PHI**2]

dados_seed0 = gerar_dados(SEEDS[0])
acc_surface = []
for c in couplings_plot:
    row = []
    for seed in SEEDS[:10]:
        X_tr, y_tr, X_te, y_te = gerar_dados(seed)
        k, _ = medir_campo(X_tr)
        def eco_c(X, coupling=c):
            X = np.asarray(X, dtype=float)
            k2, _ = medir_campo(X)
            s = X.copy()
            for _ in range(N_ECO):
                freq = np.fft.fft(s, axis=-1)
                r = np.real(np.fft.ifft(np.abs(freq)*np.exp(1j*np.angle(freq)*k2), axis=-1))
                s = s + (r - X) * coupling
            return s
        acc = treinar(eco_c(X_tr), y_tr, eco_c(X_te), y_te, seed)
        row.append(acc)
    acc_surface.append(row)

acc_surface = np.array(acc_surface)  # (n_couplings, n_seeds)

fig = plt.figure(figsize=(14, 5), facecolor=PLOT_COLORS["bg"])

# Plot 1: Superfície 3D
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_facecolor(PLOT_COLORS["panel"])

X_mesh, Y_mesh = np.meshgrid(np.arange(10), np.arange(len(couplings_plot)))
surf = ax1.plot_surface(X_mesh, Y_mesh, acc_surface,
                         cmap='plasma', alpha=0.85, edgecolor='none')

ax1.set_xlabel('Seed', color=PLOT_COLORS["text"], fontsize=7)
ax1.set_ylabel('Coupling', color=PLOT_COLORS["text"], fontsize=7)
ax1.set_zlabel('Acurácia', color=PLOT_COLORS["text"], fontsize=7)
ax1.set_yticks(range(len(couplings_plot)))
ax1.set_yticklabels([f'{c:.2f}' for c in couplings_plot], fontsize=5)
ax1.tick_params(colors=PLOT_COLORS["text"], labelsize=5)
ax1.set_title('Superfície\nCoupling × Seed × Acurácia',
              color=PLOT_COLORS["title"], fontsize=8, pad=8)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

# Plot 2: Clusters PCA 3D — sinal bruto vs v1 vs v2
ax2 = fig.add_subplot(132, projection='3d')
ax2.set_facecolor(PLOT_COLORS["panel"])

rng_vis = np.random.default_rng(SEEDS[0])
n_vis = 60
X_vis = np.vstack([gerar_serie_phi(n_vis//2, DIM, rng_vis),
                   gerar_ruido(n_vis//2, DIM, rng_vis)])
y_vis = np.array([1]*(n_vis//2) + [0]*(n_vis//2))

Xv1_vis, _, _ = eco_fononico_v1(X_vis)
Xv2_vis, _, _ = eco_fononico_v2(X_vis)

# PCA manual — 3 primeiros componentes
def pca3(X):
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:3].T)

P_raw = pca3(X_vis)
P_v2  = pca3(Xv2_vis)

for label, color, marker in [(0, PLOT_COLORS["blue"], 'o'), (1, PLOT_COLORS["gold"], '^')]:
    mask = y_vis == label
    ax2.scatter(P_raw[mask, 0], P_raw[mask, 1], P_raw[mask, 2],
                c=color, marker=marker, alpha=0.3, s=15, label=f'Bruto cl{label}')
    ax2.scatter(P_v2[mask, 0], P_v2[mask, 1], P_v2[mask, 2],
                c=color, marker=marker, alpha=0.9, s=25, edgecolors='white', linewidth=0.3)

ax2.set_title('Clusters PCA 3D\nBruto (opaco) vs V2-φ (sólido)',
              color=PLOT_COLORS["title"], fontsize=8, pad=8)
ax2.tick_params(colors=PLOT_COLORS["text"], labelsize=5)
ax2.set_xlabel('PC1', color=PLOT_COLORS["text"], fontsize=7)
ax2.set_ylabel('PC2', color=PLOT_COLORS["text"], fontsize=7)
ax2.set_zlabel('PC3', color=PLOT_COLORS["text"], fontsize=7)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

# Plot 3: Curva de acoplamento — barras 3D com zona destacada
ax3 = fig.add_subplot(133, projection='3d')
ax3.set_facecolor(PLOT_COLORS["panel"])

accs_mean = acc_surface.mean(axis=1)
xs = np.arange(len(couplings_plot))
ys = np.zeros(len(couplings_plot))
zs = np.zeros(len(couplings_plot))
dz = accs_mean

colors_bar = []
for c, a in zip(couplings_plot, accs_mean):
    if a >= 0.975:
        colors_bar.append(PLOT_COLORS["gold"])
    elif a >= 0.92:
        colors_bar.append(PLOT_COLORS["green"])
    else:
        colors_bar.append(PLOT_COLORS["blue"])

ax3.bar3d(xs - 0.4, ys, zs, 0.8, 0.5, dz,
          color=colors_bar, alpha=0.85, shade=True)

# Marcadores especiais
for i, (c, a) in enumerate(zip(couplings_plot, accs_mean)):
    if abs(c - PHI) < 0.01:
        ax3.text(i, 0.6, a + 0.003, 'φ', color=PLOT_COLORS["gold"],
                 fontsize=9, fontweight='bold', ha='center')
    elif abs(c - np.sqrt(2)) < 0.01:
        ax3.text(i, 0.6, a + 0.003, '√2', color=PLOT_COLORS["phi"],
                 fontsize=7, ha='center')

ax3.set_xticks(xs)
ax3.set_xticklabels([f'{c:.2f}' for c in couplings_plot],
                     rotation=45, fontsize=4, color=PLOT_COLORS["text"])
ax3.tick_params(colors=PLOT_COLORS["text"], labelsize=5)
ax3.set_xlabel('Coupling', color=PLOT_COLORS["text"], fontsize=7)
ax3.set_zlabel('Acurácia', color=PLOT_COLORS["text"], fontsize=7)
ax3.set_title('Zona de Acoplamento 3D\nOuro=ótimo, Verde=bom, Azul=baixo',
              color=PLOT_COLORS["title"], fontsize=8, pad=8)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False

plt.suptitle(
    f'Eco Ressonante Fonônico V2 — φ como Coupling\n'
    f'V1 (1/k): {V1.mean():.3f}  →  V2 (φ): {V2.mean():.3f}  '
    f'(Δ={V2.mean()-V1.mean():+.3f})',
    color=PLOT_COLORS["title"], fontsize=10, y=1.01
)

plt.tight_layout()
plt.savefig('eco_v2_phi_3d.png', dpi=150, bbox_inches='tight',
            facecolor=PLOT_COLORS["bg"])
plt.close()
print("Imagem salva: eco_v2_phi_3d.png")

# ── Salvar JSON ───────────────────────────────────────────────────────────────

saida = {
    "experimento": "Eco_Fononico_V2_Phi",
    "hipotese": "coupling=φ (reinjeção) + k do campo (rotação) = dupla √2+φ",
    "substrato": "séries temporais φ",
    "v1_coupling": "1/k",
    "v2_coupling": f"φ={PHI}",
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "resultados": {
        "G":           {"mean": float(G.mean()),  "std": float(G.std()),  "values": G.tolist()},
        "G_eco_v1":    {"mean": float(V1.mean()), "std": float(V1.std()), "values": V1.tolist()},
        "G_eco_v2_phi":{"mean": float(V2.mean()), "std": float(V2.std()), "values": V2.tolist()},
    },
    "comparacao_v2_vs_v1": {
        "delta": float(V2.mean()-V1.mean()),
        "p_value": float(p_comp),
        "resultado": "✅ V2 supera V1" if V2.mean() > V1.mean() and p_comp < 0.05 else "≈ equivalentes"
    },
    "k_otimos": k_log,
    "imagens": ["eco_v2_phi_3d.png"],
}

with open("eco_v2_phi_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)
print("Resultados salvos: eco_v2_phi_results.json")
