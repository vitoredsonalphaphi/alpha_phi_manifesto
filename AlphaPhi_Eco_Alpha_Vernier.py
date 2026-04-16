# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Eco_Alpha_Vernier.py
Vitor Edson Delavi · Florianópolis · 2026

Contexto:
    eco_alpha falhou: α=0.007 como rotação de fase — escala errada.
    eco_dual falhou:  α=0.007 como reinjeção direta — ~100× pequeno demais.

    O campo fonônico encontrou a escala natural de acoplamento: 1/k ≈ 0.70.
    α é invariável — mas sua função (razão entre escalas) pode operar
    sobre o valor que o campo encontrou.

Hipótese — α como vernier:
    Escala grossa: 1/k (encontrada pelo campo coletivo)
    Ajuste fino:   α como operador sobre 1/k

    α não substitui 1/k — afina 1/k.
    Como um paquímetro: escala principal + nônio de precisão.

Variantes testadas:
    v1: (1/k) × (1 - α)       — α reduz levemente o acoplamento
    v2: (1/k) × (1 + α)       — α aumenta levemente o acoplamento
    v3: (1/k) ^ (1 + α)       — α como expoente de ajuste fino
    v4: (1/k) ^ (1 - α)       — α como expoente redutor
    v5: (1/k) + α             — α como deslocamento aditivo

Referência: eco_fononico usa exatamente 1/k — sem ajuste α.
Se alguma variante superar eco_fononico: α tem papel de afinamento.
Se nenhuma superar: 1/k é o acoplamento exato, sem ajuste necessário.

Protocolo: seeds por timestamp, resultados integralmente reportados.
"""

import numpy as np
import json
import time
from scipy import stats
from utils_phi import (
    PHI, ALPHA, C_PHI,
    golden_activation, clip_grad, sigmoid
)

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

print("Experimento: Eco α-Vernier — α como ajuste fino sobre escala do campo")
print(f"α = {ALPHA:.8f}  |  1-α = {1-ALPHA:.6f}  |  1+α = {1+ALPHA:.6f}")
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

def eco_com_acoplamento(X, acoplamento_fn, n_eco=N_ECO):
    """Eco genérico — acoplamento definido por função do k."""
    X = np.asarray(X, dtype=float)
    k, coh = medir_campo(X)
    coupling = acoplamento_fn(k)
    s = X.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) * coupling
    return s, k, coh, coupling

# Definições de acoplamento
acopl = {
    "G_eco_fononico":       lambda k: 1.0 / k,                    # referência
    "G_vernier_v1_(1-α)/k": lambda k: (1 - ALPHA) / k,            # α reduz
    "G_vernier_v2_(1+α)/k": lambda k: (1 + ALPHA) / k,            # α aumenta
    "G_vernier_v3_(1/k)^1+α":lambda k: (1/k) ** (1 + ALPHA),     # α expoente+
    "G_vernier_v4_(1/k)^1-α":lambda k: (1/k) ** (1 - ALPHA),     # α expoente-
    "G_vernier_v5_1/k+α":   lambda k: (1/k) + ALPHA,              # α aditivo
}

modos = ["G"] + list(acopl.keys())
res   = {m: [] for m in modos}
k_log = []

# ── Execução ──────────────────────────────────────────────────────────────────

print(f"{'Seed':<14}", end="")
for m in modos:
    label = m[:16]
    print(f"{label:<17}", end="")
print()
print("-" * (14 + 17 * len(modos)))

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

for seed in SEEDS:
    X_tr, y_tr, X_te, y_te = gerar_dados(seed)

    acc_G = treinar(X_tr, y_tr, X_te, y_te, seed)
    res["G"].append(acc_G)

    linha = f"{seed:<14}{acc_G:<17.3f}"
    k_seed = None

    for m, fn in acopl.items():
        Xtr_eco, k_tr, _, _ = eco_com_acoplamento(X_tr, fn)
        Xte_eco, _,    _, _ = eco_com_acoplamento(X_te, fn)
        acc = treinar(Xtr_eco, y_tr, Xte_eco, y_te, seed)
        res[m].append(acc)
        linha += f"{acc:<17.3f}"
        if k_seed is None:
            k_seed = k_tr

    k_log.append(k_seed)
    print(linha)

# ── Estatísticas ──────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
G  = np.array(res["G"])
Gf = np.array(res["G_eco_fononico"])

print(f"\n{'Modo':<28} {'Média':>8} {'Δ vs G':>8} {'Δ vs fonônico':>14} {'p vs fonônico':>14}")
print("-" * 76)
print(f"{'G (base)':<28} {G.mean():>8.4f} {'—':>8} {'—':>14} {'—':>14}")
print(f"{'G_eco_fononico [ref]':<28} {Gf.mean():>8.4f} "
      f"{Gf.mean()-G.mean():>+8.4f} {'[referência]':>14} {'—':>14}")

resultados = {}
k_medio = float(np.mean(k_log))
for m, fn in acopl.items():
    if m == "G_eco_fononico":
        continue
    v = np.array(res[m])
    _, p_fn = stats.wilcoxon(v, Gf)
    sinal = "✅" if v.mean() > Gf.mean() and p_fn < 0.05 else (
            "≈" if abs(v.mean() - Gf.mean()) < 0.003 else "⚠️")
    coupling_val = fn(k_medio)
    print(f"{m:<28} {v.mean():>8.4f} {v.mean()-G.mean():>+8.4f} "
          f"{v.mean()-Gf.mean():>+14.4f} {p_fn:>14.6f} {sinal}  "
          f"[coupling={coupling_val:.5f}]")
    resultados[m] = {
        "mean": float(v.mean()), "delta_G": float(v.mean()-G.mean()),
        "delta_fononico": float(v.mean()-Gf.mean()),
        "p_vs_fononico": float(p_fn),
        "coupling_value": float(coupling_val),
        "values": v.tolist()
    }

print(f"\nk_otimo médio: {k_medio:.4f}  →  1/k = {1/k_medio:.5f}")
print(f"Acoplamentos testados:")
for m, fn in acopl.items():
    print(f"  {m:<32} = {fn(k_medio):.6f}")

# Ranking
print("\n── Ranking ──────────────────────────────────────────────────────────")
ranking = sorted([(m, float(np.array(res[m]).mean())) for m in modos], key=lambda x: -x[1])
for pos, (m, mean) in enumerate(ranking, 1):
    print(f"  {pos}. {m:<32} {mean:.4f}")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "Eco_Alpha_Vernier",
    "hipotese": "α como ajuste fino (vernier) sobre escala de acoplamento 1/k do campo",
    "substrato": "séries temporais φ",
    "referencia": {"G_eco_fononico": float(Gf.mean()), "acoplamento": float(1/k_medio)},
    "constantes": {"PHI": PHI, "ALPHA": ALPHA, "K_MIN": float(K_MIN)},
    "k_medio": k_medio,
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "resultados": {
        "G": {"mean": float(G.mean()), "values": G.tolist()},
        "G_eco_fononico": {"mean": float(Gf.mean()), "values": Gf.tolist()},
        **resultados
    },
    "k_otimos": k_log,
    "ranking": [(m, v) for m, v in ranking],
}

with open("eco_alpha_vernier_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print("\nResultados salvos: eco_alpha_vernier_results.json")
