# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Curvatura_V2.py
Vitor Edson Delavi · Florianópolis · 2026

Pergunta:
    As representações geradas pelo eco_fononico_v2 (coupling=φ)
    têm estrutura métrica hiperbólica com curvatura c ≈ 1/φ²?

    Se sim: eco_fononico e campo_transmorfo chegam ao mesmo espaço
    geométrico por caminhos independentes — a curva de acoplamento
    traça a transição Euclidiana→Hiperbólica, e o arabesco é
    o diagrama do processo, não apenas metáfora.

Método: Gromov δ-hiperbolidade

    Para um espaço métrico com curvatura -c (CAT(-c)):
        δ = log(3) / √c

    Estimativa por quádruplas aleatórias:
        Para 4 pontos x,y,z,w: s1 = d(x,y)+d(z,w), s2 = d(x,z)+d(y,w), s3 = d(x,w)+d(y,z)
        δ_quadrupla = (max(s1,s2,s3) - segundo_maior) / 2
        δ_gromov = máximo sobre todas as quádruplas

    Curvatura estimada: c_est = (log(3) / δ_max)²

    Referência: C_PHI = 1/φ² ≈ 0.382  →  δ_esperado ≈ log(3)/√(1/φ²) = log(3)·φ ≈ 1.777

Condições comparadas:
    X_raw:  representação sem eco
    X_v1:   eco_fononico V1 (coupling = 1/k ≈ 0.705)
    X_v2:   eco_fononico V2 (coupling = φ = 1.618)

Hipótese:
    c_est(V2) ≈ C_PHI = 1/φ²
    δ(V2) < δ(V1) < δ(raw)  — V2 mais hiperbólico

Protocolo: seeds por timestamp, resultados integralmente reportados.
"""

import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from utils_phi import PHI, ALPHA, C_PHI, golden_activation, clip_grad, sigmoid

# ── Parâmetros ────────────────────────────────────────────────────────────────
N_SEEDS      = 10
N_TEST       = 200   # mais amostras = melhor estimativa geométrica
N_TRAIN      = 400
DIM          = 128
N_ECO        = 3
K_MIN        = np.sqrt(2)
N_QUADRUPLES = 5000   # quádruplas por estimativa de δ

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

# Referências teóricas
DELTA_ESPERADO_CPHI = np.log(3) / np.sqrt(C_PHI)   # ≈ 1.777
DELTA_ESPERADO_C1   = np.log(3)                      # ≈ 1.099 (c=1)

print("Experimento: Curvatura das representações eco_fononico_v2")
print(f"C_PHI = 1/φ² = {C_PHI:.5f}")
print(f"δ esperado para c=C_PHI: {DELTA_ESPERADO_CPHI:.4f}")
print(f"δ esperado para c=1:     {DELTA_ESPERADO_C1:.4f}")
print(f"(Euclidiano: δ→∞ | mais hiperbólico: δ→0)")
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

# ── Campo coletivo + Eco ───────────────────────────────────────────────────────

def medir_campo(X):
    freq_batch = np.fft.fft(X, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)
    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia   = -np.sum(amp_norm * np.log(amp_norm))
    coh_campo  = float(1.0 - entropia / np.log(X.shape[-1]))
    k          = K_MIN + (PHI - K_MIN) * coh_campo
    return k, coh_campo

def eco_v1(X):
    X = np.asarray(X, dtype=float)
    k, _ = medir_campo(X)
    s = X.copy()
    for _ in range(N_ECO):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) / k
    return s

def eco_v2(X):
    X = np.asarray(X, dtype=float)
    k, _ = medir_campo(X)
    s = X.copy()
    for _ in range(N_ECO):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) * PHI
    return s

# ── Gromov δ ──────────────────────────────────────────────────────────────────

def normalizar_norma(X):
    """Projeta cada vetor para norma unitária — remove efeito de escala."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-8)

def gromov_delta(X, n_quadruples=N_QUADRUPLES, seed=42):
    """
    Estima Gromov δ-hiperbolidade via amostragem de quádruplas.

    Normaliza para norma unitária antes de medir — remove o efeito
    de amplificação de escala (crítico para comparar eco V1 vs V2).

    Fórmula: para (x,y,z,w), δ = (max_soma - segundo_maior) / 2
    onde as somas são as 3 partições de pares do quarteto.

    Retorna: δ_mean, δ_max, δ_norm (= δ_max / diâmetro), diâmetro
    """
    rng  = np.random.default_rng(seed)
    Xn   = normalizar_norma(X)          # ← normalização de norma
    n    = len(Xn)
    D    = squareform(pdist(Xn, 'euclidean'))
    diam = D.max()

    deltas  = np.zeros(n_quadruples)
    idx_all = np.arange(n)

    for q in range(n_quadruples):
        i, j, k, l = rng.choice(idx_all, 4, replace=False)
        s1 = D[i,j] + D[k,l]
        s2 = D[i,k] + D[j,l]
        s3 = D[i,l] + D[j,k]
        somas = sorted([s1, s2, s3], reverse=True)
        deltas[q] = (somas[0] - somas[1]) / 2.0

    delta_mean = float(np.mean(deltas))
    delta_max  = float(np.max(deltas))
    delta_norm = delta_max / (diam + 1e-8)   # adimensional

    return delta_mean, delta_max, delta_norm, float(diam)

def c_estimado(delta_max):
    """c_est = (log(3) / δ_max)²  — válido para CAT(-c) perfeito."""
    if delta_max < 1e-8:
        return float('inf')
    return float((np.log(3) / delta_max) ** 2)

def separacao_classes(X, y):
    """Razão inter/intra distância por classe."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    D = squareform(pdist(X, 'euclidean'))

    intra0 = D[np.ix_(idx0, idx0)]
    intra1 = D[np.ix_(idx1, idx1)]
    inter  = D[np.ix_(idx0, idx1)]

    d_intra = (intra0[intra0 > 0].mean() + intra1[intra1 > 0].mean()) / 2
    d_inter = inter.mean()

    return float(d_inter / (d_intra + 1e-8))

# ── Execução ──────────────────────────────────────────────────────────────────

print(f"{'Seed':<14} {'cond':<8} {'δ_mean':>8} {'δ_max':>8} {'δ_norm':>8} {'c_est':>8} {'sep':>8}")
print("-" * 66)

resultados = {"raw": [], "v1": [], "v2": []}

for seed in SEEDS:
    _, _, X_te, y_te = gerar_dados(seed)

    X_raw = X_te
    X_v1  = eco_v1(X_te)
    X_v2  = eco_v2(X_te)

    for nome, X in [("raw", X_raw), ("v1", X_v1), ("v2", X_v2)]:
        dm, dmax, dnorm, diam = gromov_delta(X, seed=seed)
        cest  = c_estimado(dmax)
        sep   = separacao_classes(X, y_te)
        prefixo = f"{seed:<14}" if nome == "raw" else f"{'':14}"
        print(f"{prefixo} {nome:<8} {dm:>8.4f} {dmax:>8.4f} {dnorm:>8.4f} {cest:>8.4f} {sep:>8.4f}")
        resultados[nome].append({
            "seed": seed, "delta_mean": dm, "delta_max": dmax,
            "delta_norm": dnorm, "c_est": cest, "separacao": sep, "diametro": diam
        })

    print()

# ── Síntese ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 66)
print(f"\n{'Condição':<12} {'δ_max médio':>12} {'δ_norm médio':>13} {'c_est médio':>12} {'sep média':>10}")
print("-" * 62)

sint = {}
for nome in ["raw", "v1", "v2"]:
    r = resultados[nome]
    dm_m   = np.mean([x["delta_max"]  for x in r])
    dn_m   = np.mean([x["delta_norm"] for x in r])
    ce_m   = np.mean([x["c_est"]      for x in r])
    sep_m  = np.mean([x["separacao"]  for x in r])
    sint[nome] = {"delta_max": dm_m, "delta_norm": dn_m, "c_est": ce_m, "sep": sep_m}
    print(f"  {nome:<10} {dm_m:>12.4f} {dn_m:>13.4f} {ce_m:>12.4f} {sep_m:>10.4f}")

print(f"\n  C_PHI = 1/φ²       {'—':>12} {'—':>13} {C_PHI:>12.4f}")
print(f"  δ_esp(C_PHI)   {DELTA_ESPERADO_CPHI:>12.4f}")

print("\n── Interpretação ────────────────────────────────────────────────────")
c_raw = sint["raw"]["c_est"]
c_v1  = sint["v1"]["c_est"]
c_v2  = sint["v2"]["c_est"]

print(f"  c_est(raw) = {c_raw:.4f}  |  c_est(V1) = {c_v1:.4f}  |  c_est(V2) = {c_v2:.4f}")
print(f"  C_PHI      = {C_PHI:.4f}")
print(f"  |c_est(V2) - C_PHI| = {abs(c_v2 - C_PHI):.4f}")
print(f"  |c_est(V1) - C_PHI| = {abs(c_v1 - C_PHI):.4f}")
print(f"  |c_est(raw)- C_PHI| = {abs(c_raw- C_PHI):.4f}")

dist_v2  = abs(c_v2  - C_PHI)
dist_v1  = abs(c_v1  - C_PHI)
dist_raw = abs(c_raw - C_PHI)

if dist_v2 < dist_v1 < dist_raw:
    print("\n  ✅ V2 é o mais próximo de C_PHI — progressão raw→V1→V2 em direção a 1/φ²")
    print("     A curva de acoplamento traça a transição geométrica.")
elif dist_v2 < dist_raw:
    print("\n  ◐  V2 mais próximo de C_PHI que raw, mas V1 interfere na progressão.")
else:
    print("\n  ⚠️  Progressão não confirma hipótese — ver dados brutos.")

# ── Visualização ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("#0d1117")

cores = {"raw": "#888888", "v1": "#4169E1", "v2": "#DAA520"}
labels = {"raw": "X_raw", "v1": "V1 (1/k)", "v2": "V2 (φ)"}

# Plot 1: δ_max por seed e condição
ax = axes[0]
ax.set_facecolor("#161b22")
for nome in ["raw", "v1", "v2"]:
    vals = [x["delta_max"] for x in resultados[nome]]
    ax.plot(range(N_SEEDS), vals, 'o-', color=cores[nome], label=labels[nome], linewidth=1.5)
ax.axhline(DELTA_ESPERADO_CPHI, color="#FF8C00", linestyle='--', linewidth=1.2,
           label=f'δ esp(C_PHI)={DELTA_ESPERADO_CPHI:.3f}')
ax.set_title("δ_max por seed", color="#E6EDF3")
ax.set_xlabel("seed index", color="#8B949E")
ax.set_ylabel("δ_max (Gromov)", color="#8B949E")
ax.tick_params(colors="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
for spine in ax.spines.values(): spine.set_color("#30363d")

# Plot 2: c_est por seed
ax = axes[1]
ax.set_facecolor("#161b22")
for nome in ["raw", "v1", "v2"]:
    vals = [x["c_est"] for x in resultados[nome]]
    ax.plot(range(N_SEEDS), vals, 'o-', color=cores[nome], label=labels[nome], linewidth=1.5)
ax.axhline(C_PHI, color="#FF8C00", linestyle='--', linewidth=1.5,
           label=f'C_PHI=1/φ²={C_PHI:.3f}')
ax.set_title("c_est por seed", color="#E6EDF3")
ax.set_xlabel("seed index", color="#8B949E")
ax.set_ylabel("curvatura estimada", color="#8B949E")
ax.tick_params(colors="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
for spine in ax.spines.values(): spine.set_color("#30363d")

# Plot 3: separação de classes
ax = axes[2]
ax.set_facecolor("#161b22")
for nome in ["raw", "v1", "v2"]:
    vals = [x["separacao"] for x in resultados[nome]]
    ax.plot(range(N_SEEDS), vals, 'o-', color=cores[nome], label=labels[nome], linewidth=1.5)
ax.set_title("separação inter/intra classes", color="#E6EDF3")
ax.set_xlabel("seed index", color="#8B949E")
ax.set_ylabel("razão inter/intra", color="#8B949E")
ax.tick_params(colors="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
for spine in ax.spines.values(): spine.set_color("#30363d")

plt.suptitle(
    f"Curvatura das representações: raw vs V1(1/k) vs V2(φ)\n"
    f"C_PHI = 1/φ² = {C_PHI:.4f}  |  c_est(V2) = {c_v2:.4f}",
    color="#E6EDF3", fontsize=11
)
plt.tight_layout()
plt.savefig("curvatura_v2_results.png", dpi=150, bbox_inches='tight',
            facecolor="#0d1117")
plt.close()
print("\nVisualização salva: curvatura_v2_results.png")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "Curvatura_V2",
    "pergunta": "c_est(V2) ≈ C_PHI = 1/φ²?",
    "C_PHI": C_PHI,
    "delta_esperado_CPHI": DELTA_ESPERADO_CPHI,
    "n_seeds": N_SEEDS, "n_quadruples": N_QUADRUPLES,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "sintese": {
        "raw": sint["raw"], "v1": sint["v1"], "v2": sint["v2"],
        "dist_v2_CPHI":  abs(c_v2  - C_PHI),
        "dist_v1_CPHI":  abs(c_v1  - C_PHI),
        "dist_raw_CPHI": abs(c_raw - C_PHI),
    },
    "resultados": resultados,
}

with open("curvatura_v2_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print("Dados salvos: curvatura_v2_results.json")
