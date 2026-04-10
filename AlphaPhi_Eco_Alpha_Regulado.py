# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Eco_Alpha_Regulado.py
Vitor Edson Delavi · Florianópolis · 2026

Hipótese:
    α (1/137) pode atuar como constante de acoplamento dentro do eco,
    regulando a força do feedback proporcional à coerência do sinal.
    Analogia com QED: α regula quanto o elétron se acopla ao campo fotônico —
    sem α, a série diverge. Com α, cada ciclo contribui com fator estável.

    Os três valores do eco:
        φ → rotação de fase (estrutura da transformação)
        α → granularidade mínima do acoplamento (piso de segurança)
        coerência do dado → adapta o blend ao que o sinal permite absorver

    Blend atual (eco original): 1/φ ≈ 0.618 — fixo, independente do sinal.
    Blend proposto (eco_alpha): interpola entre α e 1/φ por coerência:
        sinal incoerente  → blend ≈ α ≈ 0.0073  (eco quase não alimenta)
        sinal coerente    → blend ≈ 1/φ ≈ 0.618 (eco alimenta plenamente)

    Resultado esperado:
        Para dados estruturados (coerentes): comportamento próximo ao eco original.
        Para ruído (incoerente): α amorte o feedback — não "estoura".
        Para k=π (rotação grande): α pode salvar o resultado que antes colapsava.

Experimentos:
    1. TimeSeries φ — mesmo substrato dos experimentos anteriores
       G vs G_eco_phi (original) vs G_eco_alpha (regulado) vs G_eco_alpha_pi (k=π + α)

    2. Pergunta experimental adicional:
       α-regulação com k=π consegue recuperar o resultado que k=π sem α perdeu?
       (Experimento intercambiabilidade: k=π → 59.95%, k=φ → 90.60%)

Protocolo de idoneidade:
    Seeds por timestamp — ninguém escolhe os valores.
    Resultados reportados integralmente — favoráveis ou não.
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
N_SEEDS  = 20
N_EPOCHS = 60
N_TRAIN  = 400
N_TEST   = 100
DIM      = 128
HIDDEN   = 89
LR       = 0.01
N_ECO    = 3

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print(f"Experimento: eco_alpha_regulado")
print(f"Hipótese:    α como constante de acoplamento adaptativo no eco")
print(f"Timestamp:   {TIMESTAMP}")
print(f"φ = {PHI:.6f}  |  α = {ALPHA:.8f}  |  1/φ = {1/PHI:.6f}")
print(f"Blend mínimo (α):  {ALPHA:.6f}")
print(f"Blend máximo (1/φ): {1/PHI:.6f}")
print()

# ── Dados (idêntico ao TimeSeries_Eco) ───────────────────────────────────────

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
    """Eco original — blend fixo em 1/φ."""
    sinal = x.copy()
    for _ in range(n_eco):
        freq      = np.fft.fft(sinal, axis=-1)
        nova_fase = np.angle(freq) * phi
        reflexao  = np.real(np.fft.ifft(np.abs(freq) * np.exp(1j * nova_fase), axis=-1))
        sinal     = sinal + (reflexao - x) / phi
    return sinal

def eco_alpha_regulado(x, phi=PHI, alpha=ALPHA, k_rot=None, n_eco=N_ECO):
    """
    Eco com regulação α — três valores:
        k_rot (default=φ): parâmetro de rotação de fase
        φ: teto do blend (sinal plenamente coerente)
        α: piso do blend (sinal incoerente — acoplamento mínimo)

    Blend adaptativo:
        coerência = 0 → blend = α      (echo quase não retroalimenta)
        coerência = 1 → blend = 1/φ   (echo retroalimenta plenamente)
        blend(c) = α + (1/φ - α) * c   (interpolação linear por coerência)

    Isso replica o papel de α em QED:
        alta coerência  → forte acoplamento (sinal absorve o eco)
        baixa coerência → fraco acoplamento (sinal não "estoura")
    """
    if k_rot is None:
        k_rot = phi

    x = np.asarray(x, dtype=float)
    sinal = x.copy()

    for _ in range(n_eco):
        freq      = np.fft.fft(sinal, axis=-1)
        amplitude = np.abs(freq)
        fase      = np.angle(freq)

        # Rotação de fase por k_rot
        nova_fase = fase * k_rot
        reflexao  = np.real(np.fft.ifft(amplitude * np.exp(1j * nova_fase), axis=-1))

        # Medir coerência espectral do sinal atual
        e_norm    = np.clip(amplitude / (amplitude.sum(axis=-1, keepdims=True) + 1e-8),
                            1e-10, 1.0)
        entropia  = -np.sum(e_norm * np.log(e_norm), axis=-1, keepdims=True)
        coerencia = 1.0 - entropia / np.log(x.shape[-1])   # (batch, 1) ∈ [0, 1]

        # Blend adaptativo: α (piso) → 1/φ (teto)
        blend = alpha + (1.0 / phi - alpha) * coerencia

        # Retroalimentação regulada
        sinal = sinal + (reflexao - x) * blend

    return sinal

# ── Rede neural ───────────────────────────────────────────────────────────────

def treinar(X_tr, y_tr, X_te, y_te, seed):
    rng = np.random.default_rng(seed + 999)
    e1 = np.sqrt(2.0 / X_tr.shape[1])
    e2 = np.sqrt(2.0 / HIDDEN)
    W1 = rng.normal(0, e1, (X_tr.shape[1], HIDDEN))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, e2, (HIDDEN, 1))
    b2 = np.zeros(1)

    for _ in range(N_EPOCHS):
        idx = rng.permutation(len(X_tr))
        for i in range(0, len(X_tr) - 31, 32):
            Xb = X_tr[idx[i:i+32]]
            yb = y_tr[idx[i:i+32]]
            z1 = Xb @ W1 + b1
            a1 = golden_activation(z1)
            z2 = a1 @ W2 + b2
            yh = sigmoid(z2).squeeze()
            dL = (yh - yb) / len(yb)
            dW2 = a1.T @ dL.reshape(-1, 1)
            db2 = dL.sum(keepdims=True)
            da1 = dL.reshape(-1, 1) * W2.T
            dz1 = da1 * (1 - np.tanh(z1 / PHI)**2)
            dW1 = Xb.T @ dz1
            db1 = dz1.sum(axis=0)
            W1 -= LR * clip_grad(dW1)
            b1 -= LR * np.clip(db1, -1, 1)
            W2 -= LR * clip_grad(dW2)
            b2 -= LR * np.clip(db2, -1, 1)

    yh_te = sigmoid(golden_activation(X_te @ W1 + b1) @ W2 + b2).squeeze()
    return float(np.mean((yh_te >= 0.5) == y_te))

# ── Execução ──────────────────────────────────────────────────────────────────

modos = {
    "G":              lambda Xtr, Xte: (Xtr, Xte),
    "G_eco_phi":      lambda Xtr, Xte: (eco_ressonante(Xtr), eco_ressonante(Xte)),
    "G_eco_alpha":    lambda Xtr, Xte: (eco_alpha_regulado(Xtr), eco_alpha_regulado(Xte)),
    "G_eco_alpha_pi": lambda Xtr, Xte: (eco_alpha_regulado(Xtr, k_rot=np.pi),
                                         eco_alpha_regulado(Xte, k_rot=np.pi)),
}

resultados = {m: [] for m in modos}

print(f"{'Seed':<14}", end="")
for m in modos:
    print(f"{m:<18}", end="")
print()
print("-" * (14 + 18 * len(modos)))

for seed in SEEDS:
    X_tr, y_tr, X_te, y_te = gerar_dados(seed)
    linha = f"{seed:<14}"
    for nome, prepara in modos.items():
        Xtr_mod, Xte_mod = prepara(X_tr, X_te)
        acc = treinar(Xtr_mod, y_tr, Xte_mod, y_te, seed)
        resultados[nome].append(acc)
        linha += f"{acc:<18.3f}"
    print(linha)

# ── Estatísticas ──────────────────────────────────────────────────────────────

print()
print("=" * (14 + 18 * len(modos)))

G = np.array(resultados["G"])
print(f"\n{'Modo':<20} {'Média':>8} {'Desvio':>8} {'Delta vs G':>12} {'p-valor':>10}")
print("-" * 62)

testes = {}
for nome in modos:
    v = np.array(resultados[nome])
    delta = v.mean() - G.mean()
    if nome == "G":
        print(f"{nome:<20} {v.mean():>8.4f} {v.std():>8.4f} {'—':>12} {'—':>10}")
        testes[nome] = {"mean": float(v.mean()), "std": float(v.std()),
                        "delta": 0.0, "p_value": None, "values": v.tolist()}
    else:
        _, p = stats.wilcoxon(v, G)
        sinal = "✅" if delta > 0 and p < 0.05 else "❌"
        print(f"{nome:<20} {v.mean():>8.4f} {v.std():>8.4f} {delta:>+12.4f} {p:>10.6f} {sinal}")
        testes[nome] = {"mean": float(v.mean()), "std": float(v.std()),
                        "delta": float(delta), "p_value": float(p), "values": v.tolist()}

# Comparação direta eco_alpha vs eco_phi
v_alpha = np.array(resultados["G_eco_alpha"])
v_phi   = np.array(resultados["G_eco_phi"])
_, p_comp = stats.wilcoxon(v_alpha, v_phi)
delta_comp = v_alpha.mean() - v_phi.mean()
print(f"\nComparação direta eco_alpha vs eco_phi:")
print(f"  Δ = {delta_comp:+.4f}  p = {p_comp:.6f}")

# eco_alpha_pi vs eco original com phi
v_api = np.array(resultados["G_eco_alpha_pi"])
_, p_pi = stats.wilcoxon(v_api, v_phi)
delta_pi = v_api.mean() - v_phi.mean()
print(f"eco_alpha(k=π) vs eco_phi:")
print(f"  Δ = {delta_pi:+.4f}  p = {p_pi:.6f}")
print(f"  Referência intercambiabilidade: k=π sem α → 59.95%")
print(f"  eco_alpha_pi → {v_api.mean():.4f}")
recuperacao = v_api.mean() - 0.5995
print(f"  Recuperação por α: {recuperacao:+.4f}")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "Eco_Alpha_Regulado",
    "hipotese": "α como constante de acoplamento adaptativo — três valores no eco",
    "mecanismo": {
        "phi": "rotação de fase — estrutura da transformação",
        "alpha": "piso do blend — acoplamento mínimo (não estoura)",
        "coerencia": "adapta o blend entre α e 1/φ proporcional à estrutura do dado",
        "blend_formula": "blend = alpha + (1/phi - alpha) * coerencia"
    },
    "referencia_intercambiabilidade": {
        "k_phi_sem_alpha": 0.9060,
        "k_pi_sem_alpha":  0.5995,
    },
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "n_train": N_TRAIN, "n_test": N_TEST,
    "dim": DIM, "hidden": HIDDEN, "n_eco": N_ECO,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "resultados": testes,
    "comparacoes": {
        "eco_alpha_vs_eco_phi": {"delta": float(delta_comp), "p_value": float(p_comp)},
        "eco_alpha_pi_vs_eco_phi": {"delta": float(delta_pi), "p_value": float(p_pi)},
    }
}

with open("eco_alpha_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print(f"\nResultados salvos: eco_alpha_results.json")
print(f"\nInterpretação:")
if v_alpha.mean() > v_phi.mean() and p_comp < 0.05:
    print("  ✅ eco_alpha supera eco_phi — α melhora a regulação.")
elif abs(delta_comp) < 0.01:
    print("  ≈  eco_alpha ≈ eco_phi — regulação não prejudica, mantém resultado.")
else:
    print("  ⚠️  eco_phi ainda superior — α pode estar sub-regulando.")

if v_api.mean() > 0.80:
    print(f"  ✅ α recuperou k=π: {v_api.mean():.4f} vs 0.5995 sem α.")
else:
    print(f"  ⚠️  α não recuperou completamente k=π: {v_api.mean():.4f}")
