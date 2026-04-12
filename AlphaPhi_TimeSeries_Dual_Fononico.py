# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_TimeSeries_Dual_Fononico.py
Vitor Edson Delavi · Florianópolis · 2026

Pergunta:
    eco_fononico + modo informando (G_dual) produz sinergia?

    Histórico do substrato séries temporais φ:
        G             → ~52%   (baseline)
        G_eco_phi     → ~90%   (eco substituindo, k=φ fixo)
        G_eco_fononico → 92.80% (eco substituindo, k calibrado pelo campo)

    Modos testados aqui:
        G_dual_phi      → [x_original ‖ eco_phi(x)]      → 256 dims
        G_dual_fononico → [x_original ‖ eco_fononico(x)] → 256 dims

    Hipótese: eco fonônico informa melhor que eco fixo.
    Rede recebe sinal original + sinal calibrado pelo campo.
    Gradiente decide o peso de cada canal.

Protocolo: seeds por timestamp, resultados integralmente reportados.
"""

import numpy as np
import json
import time
from scipy import stats
from utils_phi import (
    PHI, C_PHI,
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

print("Experimento: TimeSeries — G_dual_phi vs G_dual_fononico")
print(f"Substrato: séries temporais φ  |  modo: informando [x ‖ eco(x)]")
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

# ── Eco ───────────────────────────────────────────────────────────────────────

def eco_phi(X, n_eco=N_ECO):
    X = np.asarray(X, dtype=float)
    s = X.copy()
    for _ in range(n_eco):
        freq = np.fft.fft(s, axis=-1)
        r    = np.real(np.fft.ifft(np.abs(freq) * np.exp(1j * np.angle(freq) * PHI), axis=-1))
        s    = s + (r - X) / PHI
    return s

def eco_fononico(X, n_eco=N_ECO):
    X = np.asarray(X, dtype=float)
    freq_batch = np.fft.fft(X, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)
    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia   = -np.sum(amp_norm * np.log(amp_norm))
    coh_campo  = float(1.0 - entropia / np.log(X.shape[-1]))
    k          = K_MIN + (PHI - K_MIN) * coh_campo

    s = X.copy()
    for _ in range(n_eco):
        freq = np.fft.fft(s, axis=-1)
        r    = np.real(np.fft.ifft(np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s    = s + (r - X) / k
    return s, k, coh_campo

# ── Rede neural (aceita dim_in variável para suportar 128 e 256) ─────────────

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

modos = ["G", "G_eco_phi", "G_eco_fononico", "G_dual_phi", "G_dual_fononico"]
res   = {m: [] for m in modos}
k_log = []

header = f"{'Seed':<14}" + "".join(f"{m:<18}" for m in modos)
print(header)
print("-" * len(header))

for seed in SEEDS:
    X_tr, y_tr, X_te, y_te = gerar_dados(seed)

    X_eco_tr = eco_phi(X_tr);          X_eco_te = eco_phi(X_te)
    X_fn_tr, k_tr, _ = eco_fononico(X_tr); X_fn_te, _, _ = eco_fononico(X_te)

    configs = {
        "G":              (X_tr,                                    X_te),
        "G_eco_phi":      (X_eco_tr,                               X_eco_te),
        "G_eco_fononico": (X_fn_tr,                                X_fn_te),
        "G_dual_phi":     (np.hstack([X_tr, X_eco_tr]),            np.hstack([X_te, X_eco_te])),
        "G_dual_fononico":(np.hstack([X_tr, X_fn_tr]),             np.hstack([X_te, X_fn_te])),
    }

    linha = f"{seed:<14}"
    for m, (Xtr_m, Xte_m) in configs.items():
        acc = treinar(Xtr_m, y_tr, Xte_m, y_te, seed)
        res[m].append(acc)
        linha += f"{acc:<18.3f}"
    k_log.append(k_tr)
    print(linha)

# ── Estatísticas ──────────────────────────────────────────────────────────────

print("\n" + "=" * len(header))
G = np.array(res["G"])

print(f"\n{'Modo':<20} {'Média':>8} {'Desvio':>8} {'Δ vs G':>10} {'p-valor':>10}")
print("-" * 60)
print(f"{'G (base)':<20} {G.mean():>8.4f} {G.std():>8.4f} {'—':>10} {'—':>10}")

testes = {}
for m in modos[1:]:
    v = np.array(res[m])
    delta = v.mean() - G.mean()
    _, p  = stats.wilcoxon(v, G)
    sinal = "✅" if delta > 0 and p < 0.05 else ("≈" if abs(delta) < 0.005 else "⚠️")
    print(f"{m:<20} {v.mean():>8.4f} {v.std():>8.4f} {delta:>+10.4f} {p:>10.6f} {sinal}")
    testes[m] = {"mean": float(v.mean()), "std": float(v.std()),
                 "delta": float(delta), "p_value": float(p), "values": v.tolist()}

# Comparação direta G_dual_fononico vs G_dual_phi
v_df = np.array(res["G_dual_fononico"])
v_dp = np.array(res["G_dual_phi"])
_, p_comp = stats.wilcoxon(v_df, v_dp)
print(f"\nComparação direta G_dual_fononico vs G_dual_phi:")
sinal = "✅" if v_df.mean() > v_dp.mean() and p_comp < 0.05 else "≈"
print(f"  Δ = {v_df.mean()-v_dp.mean():+.4f}  p = {p_comp:.6f}  {sinal}")

# Comparação G_dual_fononico vs G_eco_fononico
v_ef = np.array(res["G_eco_fononico"])
_, p_ef = stats.wilcoxon(v_df, v_ef)
print(f"\nComparação G_dual_fononico vs G_eco_fononico:")
sinal2 = "✅" if v_df.mean() > v_ef.mean() and p_ef < 0.05 else "≈"
print(f"  Δ = {v_df.mean()-v_ef.mean():+.4f}  p = {p_ef:.6f}  {sinal2}")

print(f"\nk_otimo médio (campo fonônico): {np.mean(k_log):.4f}")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "TimeSeries_Dual_Fononico",
    "pergunta": "eco_fononico + modo informando → sinergia?",
    "substrato": "séries temporais φ",
    "referencia_historica": {
        "G": "~52%", "G_eco_phi": "~90%", "G_eco_fononico": "92.80%"
    },
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "n_train": N_TRAIN, "n_test": N_TEST,
    "dim": DIM, "dim_dual": DIM * 2, "hidden": HIDDEN,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "resultados": {
        "G": {"mean": float(G.mean()), "std": float(G.std()), "values": G.tolist()},
        **testes
    },
    "comparacoes": {
        "dual_fononico_vs_dual_phi": {
            "delta": float(v_df.mean()-v_dp.mean()), "p_value": float(p_comp)},
        "dual_fononico_vs_eco_fononico": {
            "delta": float(v_df.mean()-v_ef.mean()), "p_value": float(p_ef)},
    },
    "k_otimos": k_log,
}

with open("timeseries_dual_fononico_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print("\nResultados salvos: timeseries_dual_fononico_results.json")

# Ranking
print("\n── Ranking ─────────────────────────────────────────────────────")
ranking = sorted([(m, np.array(res[m]).mean()) for m in modos], key=lambda x: -x[1])
for pos, (m, mean) in enumerate(ranking, 1):
    print(f"  {pos}. {m:<22} {mean:.4f}")
