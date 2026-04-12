# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Eco_Fononico.py
Vitor Edson Delavi · Florianópolis · 2026

Hipótese — meta-calibração coletiva (estado fonônico):

    O eco atual observa cada dado individualmente e aplica k=φ fixo.
    O parâmetro k não sabe nada sobre o campo que os dados formam juntos.

    Proposta: dois estágios antes da rede neural.

    Estágio 0 — estado fonônico (meta-pré-função):
        Observa o BATCH COMPLETO como rede de átomos vibrantes.
        Mede o padrão coletivo de fases — não a coerência de cada amostra,
        mas a coerência do campo que as amostras formam juntas.
        Retorna k_otimo: parâmetro informado pelo campo.

    Estágio 1 — eco calibrado (pré-função):
        Aplica eco com k_otimo (não φ fixo).
        Cada amostra é transformada por um eco que já conhece o terreno.

    Diferença crítica:
        eco_phi   → k=φ fixo para todo dado, todo batch
        eco_alpha → k adaptado por coerência LOCAL de cada amostra
        eco_fractal → k sobe oitava por Δcoerência LOCAL
        eco_fononico → k determinado pelo campo COLETIVO do batch inteiro

    O fonon, por definição, é vibração coletiva.
    Usar coerência individual é medir a temperatura de um único átomo.
    O estado fonônico é a temperatura da REDE.

    Calibração de k:
        k_otimo = √2 + (φ - √2) * coerencia_campo
        Zona ótima descoberta no experimento de intercambiabilidade: [√2, φ]
        k=√2 → 92.90% (melhor); k=φ → 90.60% (segundo)
        Campo incoerente → k tende a √2; campo coerente → k tende a φ.

Protocolo de idoneidade:
    Seeds por timestamp. Resultados integralmente reportados.
"""

import numpy as np
import json
import time
from scipy import stats
from utils_phi import (
    PHI, ALPHA, C_PHI,
    golden_activation, golden_activation_deriv,
    clip_grad, sigmoid
)

# ── Parâmetros ────────────────────────────────────────────────────────────────
N_SEEDS   = 20
N_EPOCHS  = 60
N_TRAIN   = 400
N_TEST    = 100
DIM       = 128
HIDDEN    = 89
LR        = 0.01
N_ECO     = 3

K_MIN = np.sqrt(2)   # limite inferior da zona ótima (intercambiabilidade)
K_MAX = PHI          # limite superior da zona ótima

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print(f"Experimento: eco_fononico — meta-calibração coletiva")
print(f"Hipótese:    campo espectral do batch informa k_otimo antes do eco")
print(f"Zona ótima:  [√2={K_MIN:.4f}, φ={K_MAX:.4f}] (intercambiabilidade)")
print(f"Timestamp:   {TIMESTAMP}")
print()

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
    X_train = np.vstack([gerar_serie_phi(n_tr, DIM, rng), gerar_ruido(n_tr, DIM, rng)])
    y_train = np.array([1]*n_tr + [0]*n_tr, dtype=float)
    X_test  = np.vstack([gerar_serie_phi(n_te, DIM, rng), gerar_ruido(n_te, DIM, rng)])
    y_test  = np.array([1]*n_te + [0]*n_te, dtype=float)
    idx_tr  = rng.permutation(N_TRAIN)
    idx_te  = rng.permutation(N_TEST)
    return X_train[idx_tr], y_train[idx_tr], X_test[idx_te], y_test[idx_te]

# ── Funções de eco ────────────────────────────────────────────────────────────

def eco_ressonante(x, phi=PHI, n_eco=N_ECO):
    """Eco original — k=φ fixo, referência."""
    x = np.asarray(x, dtype=float)
    sinal = x.copy()
    for _ in range(n_eco):
        freq      = np.fft.fft(sinal, axis=-1)
        nova_fase = np.angle(freq) * phi
        reflexao  = np.real(np.fft.ifft(np.abs(freq) * np.exp(1j * nova_fase), axis=-1))
        sinal     = sinal + (reflexao - x) / phi
    return sinal

def medir_estado_fononico(X_batch):
    """
    Estágio 0 — estado fonônico do batch.

    Observa o campo espectral coletivo: média das amplitudes FFT do batch.
    Isso é a 'temperatura da rede' — não de um átomo individual.

    Retorna:
        k_otimo     — parâmetro de rotação informado pelo campo
        coh_campo   — coerência do campo coletivo ∈ [0, 1]
    """
    freq_batch = np.fft.fft(X_batch, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)          # perfil coletivo (dim,)

    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia   = -np.sum(amp_norm * np.log(amp_norm))
    coh_campo  = float(1.0 - entropia / np.log(len(amp_media)))

    # k interpolado na zona ótima: campo incoerente → √2; coerente → φ
    k_otimo = K_MIN + (K_MAX - K_MIN) * coh_campo
    return k_otimo, coh_campo

def eco_fononico(X, n_eco=N_ECO):
    """
    Eco com meta-calibração coletiva — dois estágios:

    Estágio 0: medir estado fonônico do batch → k_otimo
    Estágio 1: aplicar eco com k_otimo (não φ fixo)

    O campo diz ao eco qual parâmetro usar antes de qualquer transformação.
    """
    X = np.asarray(X, dtype=float)
    k_otimo, coh_campo = medir_estado_fononico(X)

    sinal = X.copy()
    for _ in range(n_eco):
        freq      = np.fft.fft(sinal, axis=-1)
        nova_fase = np.angle(freq) * k_otimo
        reflexao  = np.real(np.fft.ifft(np.abs(freq) * np.exp(1j * nova_fase), axis=-1))
        sinal     = sinal + (reflexao - X) / k_otimo

    return sinal, k_otimo, coh_campo

# ── Rede neural ───────────────────────────────────────────────────────────────

def treinar(X_tr, y_tr, X_te, y_te, seed):
    rng = np.random.default_rng(seed + 999)
    W1 = rng.normal(0, np.sqrt(2.0 / X_tr.shape[1]), (X_tr.shape[1], HIDDEN))
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

res = {"G": [], "G_eco_phi": [], "G_eco_fononico": []}
k_otimos   = []
coh_campos = []

print(f"{'Seed':<14} {'G':>7} {'G_eco_φ':>9} {'G_fonon':>9} {'k_otimo':>9} {'coh_campo':>11}")
print("-" * 63)

for seed in SEEDS:
    X_tr, y_tr, X_te, y_te = gerar_dados(seed)

    # Baseline
    acc_G = treinar(X_tr, y_tr, X_te, y_te, seed)
    res["G"].append(acc_G)

    # Eco φ fixo
    acc_phi = treinar(eco_ressonante(X_tr), y_tr, eco_ressonante(X_te), y_te, seed)
    res["G_eco_phi"].append(acc_phi)

    # Eco fonônico — k informado pelo campo
    Xtr_fn, k_tr, coh_tr = eco_fononico(X_tr)
    Xte_fn, _,    _      = eco_fononico(X_te)
    acc_fn = treinar(Xtr_fn, y_tr, Xte_fn, y_te, seed)
    res["G_eco_fononico"].append(acc_fn)

    k_otimos.append(k_tr)
    coh_campos.append(coh_tr)

    print(f"{seed:<14} {acc_G:>7.3f} {acc_phi:>9.3f} {acc_fn:>9.3f} "
          f"{k_tr:>9.4f} {coh_tr:>11.4f}")

# ── Estatísticas ──────────────────────────────────────────────────────────────

print()
print("=" * 63)

G     = np.array(res["G"])
E_phi = np.array(res["G_eco_phi"])
E_fn  = np.array(res["G_eco_fononico"])

_, p_phi  = stats.wilcoxon(E_phi, G)
_, p_fn   = stats.wilcoxon(E_fn,  G)
_, p_comp = stats.wilcoxon(E_fn, E_phi)

print(f"\n{'Modo':<18} {'Média':>8} {'Desvio':>8} {'Δ vs G':>10} {'p-valor':>10}")
print("-" * 58)
print(f"{'G (base)':<18} {G.mean():>8.4f} {G.std():>8.4f} {'—':>10} {'—':>10}")
print(f"{'G_eco_phi':<18} {E_phi.mean():>8.4f} {E_phi.std():>8.4f} "
      f"{E_phi.mean()-G.mean():>+10.4f} {p_phi:>10.6f}")
print(f"{'G_eco_fononico':<18} {E_fn.mean():>8.4f} {E_fn.std():>8.4f} "
      f"{E_fn.mean()-G.mean():>+10.4f} {p_fn:>10.6f}")

print(f"\nComparação direta fonônico vs eco_phi:")
sinal = "✅" if E_fn.mean() > E_phi.mean() and p_comp < 0.05 else (
        "≈" if abs(E_fn.mean() - E_phi.mean()) < 0.01 else "⚠️")
print(f"  Δ = {E_fn.mean()-E_phi.mean():+.4f}  p = {p_comp:.6f}  {sinal}")

print(f"\nEstado fonônico — campo coletivo:")
print(f"  k_otimo médio:    {np.mean(k_otimos):.4f}  "
      f"(√2={K_MIN:.4f} … φ={K_MAX:.4f})")
print(f"  coerência campo:  {np.mean(coh_campos):.4f}  "
      f"(0=ruído puro, 1=coerência total)")
print(f"  k_otimo min/max:  {min(k_otimos):.4f} / {max(k_otimos):.4f}")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "Eco_Fononico",
    "hipotese": "meta-calibração coletiva — campo espectral do batch informa k_otimo",
    "mecanismo": {
        "estagio_0": "medir estado fonônico do batch → k_otimo ∈ [√2, φ]",
        "estagio_1": "eco com k_otimo (não φ fixo)",
        "calibracao": f"k = √2 + (φ - √2) * coerencia_campo",
        "zona_otima": f"[{K_MIN:.4f}, {K_MAX:.4f}] — descoberta no experimento de intercambiabilidade",
    },
    "referencia_intercambiabilidade": {
        "k_sqrt2": 0.9290, "k_phi": 0.9060,
        "observacao": "k=√2 superou k=φ — campo dos dados tinha estrutura √2"
    },
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "n_train": N_TRAIN, "n_test": N_TEST,
    "dim": DIM, "hidden": HIDDEN, "n_eco": N_ECO,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "resultados": {
        "G":              {"mean": float(G.mean()),     "std": float(G.std()),     "values": G.tolist()},
        "G_eco_phi":      {"mean": float(E_phi.mean()), "std": float(E_phi.std()), "values": E_phi.tolist()},
        "G_eco_fononico": {"mean": float(E_fn.mean()),  "std": float(E_fn.std()),  "values": E_fn.tolist()},
    },
    "comparacoes": {
        "eco_phi_vs_G":      {"delta": float(E_phi.mean()-G.mean()),  "p_value": float(p_phi)},
        "eco_fononico_vs_G": {"delta": float(E_fn.mean()-G.mean()),   "p_value": float(p_fn)},
        "fononico_vs_phi":   {"delta": float(E_fn.mean()-E_phi.mean()), "p_value": float(p_comp)},
    },
    "estado_fononico": {
        "k_otimos":    k_otimos,
        "coh_campos":  coh_campos,
        "k_medio":     float(np.mean(k_otimos)),
        "coh_media":   float(np.mean(coh_campos)),
    }
}

with open("eco_fononico_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print(f"\nResultados salvos: eco_fononico_results.json")
print(f"\nInterpretação:")
if E_fn.mean() > E_phi.mean() and p_comp < 0.05:
    print(f"  ✅ eco_fononico supera eco_phi — campo coletivo informa melhor que k=φ fixo.")
    print(f"  O fonon do batch carrega informação que a amostra individual não vê.")
elif abs(E_fn.mean() - E_phi.mean()) < 0.01:
    print(f"  ≈  eco_fononico ≈ eco_phi — k_otimo converge para φ (campo já coerente).")
    print(f"  Coerência do campo: {np.mean(coh_campos):.3f} → k_otimo: {np.mean(k_otimos):.4f}")
else:
    print(f"  ⚠️  eco_phi ainda superior — investigar faixa de k_otimo.")
