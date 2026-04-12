# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Audio_Fononico.py
Vitor Edson Delavi · Florianópolis · 2026

Pergunta — generalização do eco fonônico:
    O eco_fononico melhora também em substratos fora das séries φ?

Substrato: harmônicos musicais naturais (dó-mi-sol).
    φ NÃO está inserido nos dados — as frequências são razões simples
    (1, 5/4, 3/2, 2, 5/2, 3 × freq_base).
    Referência: G=48.53% → G_eco_phi=97.38% (+48.85%, p=0.0000).

Modos testados:
    G              → baseline sem eco
    G_eco_phi      → eco substituindo, k=φ fixo  (referência: 97.38%)
    G_eco_fononico → eco substituindo, k calibrado pelo campo
    G_dual_phi     → [x ‖ eco_phi(x)]   — modo informando
    G_dual_fononico → [x ‖ eco_fononico(x)] — modo informando fonônico

Questão central: o campo fonônico de áudio (harmônicos naturais)
    produz k diferente do campo fonônico de séries φ?
    O k do áudio converge para onde — √2, φ, ou outra zona?

Protocolo: seeds por timestamp, resultados integralmente reportados.
"""

import numpy as np
import json
import time
from scipy import stats
from utils_phi import PHI, golden_activation, clip_grad, sigmoid

# ── Parâmetros ────────────────────────────────────────────────────────────────
N_SEEDS  = 20
N_EPOCHS = 60
N_PC     = 400          # amostras por classe no treino
N_PC_TE  = 100          # amostras por classe no teste
DIM      = 128
HIDDEN   = 89
LR       = 0.01
N_ECO    = 3
K_MIN    = np.sqrt(2)

# Razões harmônicas de dó-mi-sol (sem φ)
HARMONICOS = [1.0, 5/4, 3/2, 2.0, 5/2, 3.0]

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print("Experimento: Audio — eco_fononico em harmônicos naturais")
print(f"Substrato: dó-mi-sol  |  φ não está nos dados")
print(f"Referência G_eco_phi: 97.38%  |  Timestamp: {TIMESTAMP}\n")

# ── Dados ─────────────────────────────────────────────────────────────────────

def gerar_tom(n, dim, rng):
    """Classe 1: tom musical — harmônicos dó-mi-sol (razões simples, sem φ)."""
    t = np.linspace(0, 4 * np.pi, dim)
    X = np.zeros((n, dim))
    for i in range(n):
        freq_base = rng.uniform(1.0, 4.0)
        sinal = np.zeros(dim)
        for h in HARMONICOS:
            amp  = rng.uniform(0.3, 1.0)
            fase = rng.uniform(0, 2 * np.pi)
            sinal += amp * np.sin(freq_base * h * t + fase)
        ruido = rng.normal(0, 0.05, dim)
        sinal = (sinal + ruido) / (np.std(sinal + ruido) + 1e-8)
        X[i] = sinal
    return X

def gerar_ruido(n, dim, rng):
    """Classe 0: ruído gaussiano puro."""
    X = rng.normal(0, 1, (n, dim))
    return X / (np.std(X, axis=1, keepdims=True) + 1e-8)

def gerar_dados(seed):
    rng  = np.random.default_rng(seed)
    X_tr = np.vstack([gerar_tom(N_PC, DIM, rng),    gerar_ruido(N_PC,    DIM, rng)])
    y_tr = np.array([1]*N_PC + [0]*N_PC, dtype=float)
    X_te = np.vstack([gerar_tom(N_PC_TE, DIM, rng), gerar_ruido(N_PC_TE, DIM, rng)])
    y_te = np.array([1]*N_PC_TE + [0]*N_PC_TE, dtype=float)
    N_TR, N_TE = len(y_tr), len(y_te)
    idx_tr = rng.permutation(N_TR)
    idx_te = rng.permutation(N_TE)
    return X_tr[idx_tr], y_tr[idx_tr], X_te[idx_te], y_te[idx_te]

# ── Eco ───────────────────────────────────────────────────────────────────────

def eco_phi(X, n_eco=N_ECO):
    X = np.asarray(X, dtype=float); s = X.copy()
    for _ in range(n_eco):
        freq = np.fft.fft(s, axis=-1)
        r    = np.real(np.fft.ifft(np.abs(freq) * np.exp(1j * np.angle(freq) * PHI), axis=-1))
        s    = s + (r - X) / PHI
    return s

def eco_fononico(X, n_eco=N_ECO):
    X = np.asarray(X, dtype=float)
    amp_media = np.abs(np.fft.fft(X, axis=-1)).mean(axis=0)
    amp_norm  = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia  = -np.sum(amp_norm * np.log(amp_norm))
    coh       = float(1.0 - entropia / np.log(X.shape[-1]))
    k         = K_MIN + (PHI - K_MIN) * coh

    s = X.copy()
    for _ in range(n_eco):
        freq = np.fft.fft(s, axis=-1)
        r    = np.real(np.fft.ifft(np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s    = s + (r - X) / k
    return s, k, coh

# ── Rede neural ───────────────────────────────────────────────────────────────

def treinar(X_tr, y_tr, X_te, y_te, seed):
    dim_in = X_tr.shape[1]
    rng    = np.random.default_rng(seed + 999)
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
k_log = []; coh_log = []

header = f"{'Seed':<14}" + "".join(f"{m:<18}" for m in modos) + f"{'k_otimo':>9}"
print(header)
print("-" * len(header))

for seed in SEEDS:
    X_tr, y_tr, X_te, y_te = gerar_dados(seed)

    X_ep_tr = eco_phi(X_tr);             X_ep_te = eco_phi(X_te)
    X_fn_tr, k_tr, coh_tr = eco_fononico(X_tr)
    X_fn_te, _,    _      = eco_fononico(X_te)

    configs = {
        "G":              (X_tr,                         X_te),
        "G_eco_phi":      (X_ep_tr,                      X_ep_te),
        "G_eco_fononico": (X_fn_tr,                      X_fn_te),
        "G_dual_phi":     (np.hstack([X_tr, X_ep_tr]),   np.hstack([X_te, X_ep_te])),
        "G_dual_fononico":(np.hstack([X_tr, X_fn_tr]),   np.hstack([X_te, X_fn_te])),
    }

    linha = f"{seed:<14}"
    for m, (Xtr_m, Xte_m) in configs.items():
        acc = treinar(Xtr_m, y_tr, Xte_m, y_te, seed)
        res[m].append(acc)
        linha += f"{acc:<18.3f}"
    linha += f"{k_tr:>9.4f}"
    k_log.append(k_tr); coh_log.append(coh_tr)
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
    _, p = stats.wilcoxon(v, G)
    delta = v.mean() - G.mean()
    sinal = "✅" if delta > 0 and p < 0.05 else ("≈" if abs(delta) < 0.005 else "⚠️")
    print(f"{m:<20} {v.mean():>8.4f} {v.std():>8.4f} {delta:>+10.4f} {p:>10.6f} {sinal}")
    testes[m] = {"mean": float(v.mean()), "std": float(v.std()),
                 "delta": float(delta), "p_value": float(p), "values": v.tolist()}

v_df = np.array(res["G_dual_fononico"])
v_dp = np.array(res["G_dual_phi"])
v_ef = np.array(res["G_eco_fononico"])
v_ep = np.array(res["G_eco_phi"])
_, p1 = stats.wilcoxon(v_df, v_dp)
_, p2 = stats.wilcoxon(v_df, v_ef)
_, p3 = stats.wilcoxon(v_df, v_ep)

print(f"\nComparações diretas:")
print(f"  dual_fononico vs dual_phi:     Δ={v_df.mean()-v_dp.mean():+.4f}  p={p1:.6f}")
print(f"  dual_fononico vs eco_fononico: Δ={v_df.mean()-v_ef.mean():+.4f}  p={p2:.6f}")
print(f"  dual_fononico vs eco_phi:      Δ={v_df.mean()-v_ep.mean():+.4f}  p={p3:.6f}")

print(f"\nEstado fonônico — campo de áudio:")
print(f"  k_otimo médio:   {np.mean(k_log):.4f}  (√2={K_MIN:.4f} … φ={PHI:.4f})")
print(f"  coh_campo médio: {np.mean(coh_log):.4f}")
print(f"  k_otimo min/max: {min(k_log):.4f} / {max(k_log):.4f}")

# Ranking
print("\n── Ranking ─────────────────────────────────────────────────────")
ranking = sorted([(m, np.array(res[m]).mean()) for m in modos], key=lambda x: -x[1])
for pos, (m, mean) in enumerate(ranking, 1):
    print(f"  {pos}. {m:<22} {mean:.4f}")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "Audio_Fononico",
    "substrato": "harmônicos musicais dó-mi-sol (φ não está nos dados)",
    "referencia_audio_eco": {"G": 0.4853, "G_eco_phi": 0.9738, "delta": 0.4885},
    "pergunta": "eco_fononico generaliza para áudio? qual k o campo de áudio produz?",
    "harmonicos": HARMONICOS,
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "n_pc_train": N_PC, "n_pc_test": N_PC_TE,
    "dim": DIM, "dim_dual": DIM * 2, "hidden": HIDDEN,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "resultados": {
        "G": {"mean": float(G.mean()), "std": float(G.std()), "values": G.tolist()},
        **testes
    },
    "comparacoes": {
        "dual_fononico_vs_dual_phi":     {"delta": float(v_df.mean()-v_dp.mean()), "p_value": float(p1)},
        "dual_fononico_vs_eco_fononico": {"delta": float(v_df.mean()-v_ef.mean()), "p_value": float(p2)},
        "dual_fononico_vs_eco_phi":      {"delta": float(v_df.mean()-v_ep.mean()), "p_value": float(p3)},
    },
    "estado_fononico_audio": {
        "k_otimos":  k_log,
        "coh_campos": coh_log,
        "k_medio":   float(np.mean(k_log)),
        "coh_media": float(np.mean(coh_log)),
    },
    "ranking": [{"pos": i+1, "modo": m, "mean": mean} for i, (m, mean) in enumerate(ranking)],
}

with open("audio_fononico_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print("\nResultados salvos: audio_fononico_results.json")
