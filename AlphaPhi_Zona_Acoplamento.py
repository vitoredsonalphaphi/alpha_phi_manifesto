# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Zona_Acoplamento.py
Vitor Edson Delavi · Florianópolis · 2026

Pergunta:
    O eco_fononico encontrou 1/k ≈ 0.705 como acoplamento natural.
    O teste vernier mostrou que [0.700, 0.713] é uma zona estável.

    Qual a largura real dessa zona?
    Existe um pico dentro dela ou é plana (equipotencial)?
    Onde os limites caem (ascendente e descendente)?
    α é a unidade natural de granularidade dentro dela?

Método:
    Mapear acurácia vs acoplamento em duas progressões a partir do centro:

    Centro natural: 1/k_otimo ≈ 0.705 ≈ 1/√2

    Progressão grossa (explorar zona ampla):
        {0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00}

    Progressão fina α-modulada (explorar zona estreita):
        centro - 100α, - 50α, - 20α, - 10α, - 5α, - 2α, - α,
        centro,
        centro + α, + 2α, + 5α, + 10α, + 20α, + 50α, + 100α

    Se o pico coincide com 1/√2 → atrator matemático real.
    Se a zona é plana → equipotencial (qualquer valor na faixa serve).
    Se α define a granularidade → α é a unidade natural neste domínio.

Nota sobre os 4 eventos:
    Se a zona tem centro em 1/√2 e o campo encontrou exatamente esse valor,
    isso é evidência estrutural de que a emergência não foi acidental —
    o campo convergiu para um atrator real, verificável geometricamente.

Protocolo: seeds por timestamp, resultados integralmente reportados.
"""

import numpy as np
import json
import time
from scipy import stats
from utils_phi import PHI, ALPHA, golden_activation, clip_grad, sigmoid

# ── Parâmetros ────────────────────────────────────────────────────────────────
N_SEEDS  = 10   # 10 seeds por ponto — 26 pontos no total
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

# Centro natural
K_OTIMO   = K_MIN + (PHI - K_MIN) * 0.018  # coh_campo típico ≈ 0.018
CENTRO    = 1.0 / K_OTIMO                   # ≈ 0.705 ≈ 1/√2

print("Experimento: Zona de Acoplamento — mapeamento completo")
print(f"Centro natural: 1/k_otimo = {CENTRO:.5f}  (k_otimo={K_OTIMO:.5f})")
print(f"1/√2 = {1/np.sqrt(2):.5f}  |  α = {ALPHA:.6f}")
print(f"Timestamp: {TIMESTAMP}\n")

# ── Progressões de acoplamento ────────────────────────────────────────────────

# Grossa: explorar zona ampla
grossa = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]

# Fina α-modulada: explorar em torno do centro
multiplos = [-100, -50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 100]
fina = [CENTRO + m * ALPHA for m in multiplos]

# Combinado (único, ordenado, sem repetição)
todos = sorted(set([round(v, 6) for v in grossa + fina]))
todos = [v for v in todos if 0.05 < v < 1.5]   # limites práticos

print(f"Pontos a mapear: {len(todos)}")
print(f"Faixa: [{min(todos):.4f}, {max(todos):.4f}]")
print(f"Zona fina: [{min(fina):.5f}, {max(fina):.5f}]  (±100α em torno do centro)\n")

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

# ── Eco com acoplamento fixo ──────────────────────────────────────────────────

def eco_acoplamento_fixo(X, coupling, n_eco=N_ECO):
    """Usa k_otimo do campo para rotação, coupling fixo para reinjeção."""
    X = np.asarray(X, dtype=float)
    # Mede campo para k (rotação) — mesmo processo do eco_fononico
    freq_batch = np.fft.fft(X, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)
    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia   = -np.sum(amp_norm * np.log(amp_norm))
    coh_campo  = float(1.0 - entropia / np.log(X.shape[-1]))
    k          = K_MIN + (PHI - K_MIN) * coh_campo

    s = X.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) * coupling   # coupling fixo, não 1/k
    return s

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

# ── Execução: mapeamento completo ─────────────────────────────────────────────

print(f"{'coupling':>10} {'acc_media':>10} {'desvio':>8} {'zona':>8}")
print("-" * 42)

mapa = []
dados_cache = {seed: gerar_dados(seed) for seed in SEEDS}

for coupling in todos:
    accs = []
    for seed in SEEDS:
        X_tr, y_tr, X_te, y_te = dados_cache[seed]
        Xtr_eco = eco_acoplamento_fixo(X_tr, coupling)
        Xte_eco = eco_acoplamento_fixo(X_te, coupling)
        acc = treinar(Xtr_eco, y_tr, Xte_eco, y_te, seed)
        accs.append(acc)

    media  = float(np.mean(accs))
    desvio = float(np.std(accs))

    # Identificar zona
    dist_centro = abs(coupling - CENTRO)
    if dist_centro <= 5 * ALPHA:
        zona = "●CENTRO"
    elif dist_centro <= 20 * ALPHA:
        zona = "◐ fina"
    elif coupling < CENTRO:
        zona = "↓ abaixo"
    else:
        zona = "↑ acima"

    print(f"{coupling:>10.5f} {media:>10.4f} {desvio:>8.4f} {zona:>8}")
    mapa.append({
        "coupling": coupling,
        "acc_media": media,
        "desvio": desvio,
        "valores": accs,
        "dist_centro_alpha": float(dist_centro / ALPHA),
    })

# ── Análise da zona ───────────────────────────────────────────────────────────

print("\n── Análise da zona estável ──────────────────────────────────────────")
accs_map = [(p["coupling"], p["acc_media"]) for p in mapa]
acc_max  = max(p["acc_media"] for p in mapa)
tol      = 0.005  # dentro de 0.5pp do máximo = zona estável

zona_estavel = [(c, a) for c, a in accs_map if a >= acc_max - tol]
if zona_estavel:
    c_min = min(c for c, _ in zona_estavel)
    c_max = max(c for c, _ in zona_estavel)
    c_pico = max(zona_estavel, key=lambda x: x[1])[0]
    largura_alpha = (c_max - c_min) / ALPHA

    print(f"Acurácia máxima:     {acc_max:.4f}")
    print(f"Zona estável (±0.5pp): [{c_min:.5f}, {c_max:.5f}]")
    print(f"Largura da zona:     {c_max-c_min:.5f}  ≈ {largura_alpha:.1f}α")
    print(f"Pico observado em:   {c_pico:.5f}")
    print(f"Centro natural 1/k:  {CENTRO:.5f}")
    print(f"1/√2:                {1/np.sqrt(2):.5f}")
    print(f"Distância pico→centro: {abs(c_pico-CENTRO):.5f}  = {abs(c_pico-CENTRO)/ALPHA:.1f}α")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "Zona_Acoplamento",
    "pergunta": "qual a largura e estrutura da zona estável de acoplamento?",
    "centro_natural": CENTRO,
    "inv_sqrt2": float(1/np.sqrt(2)),
    "alpha": ALPHA,
    "k_otimo_tipico": K_OTIMO,
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "mapa": mapa,
    "zona_estavel": {
        "acc_max": acc_max,
        "c_min": c_min if zona_estavel else None,
        "c_max": c_max if zona_estavel else None,
        "largura": float(c_max - c_min) if zona_estavel else None,
        "largura_em_alpha": float(largura_alpha) if zona_estavel else None,
        "pico": float(c_pico) if zona_estavel else None,
    },
}

with open("zona_acoplamento_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print("\nResultados salvos: zona_acoplamento_results.json")
