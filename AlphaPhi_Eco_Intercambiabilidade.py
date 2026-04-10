# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Eco_Intercambiabilidade.py
Vitor Edson Delavi · Florianópolis · 2026

Pergunta experimental:
    φ como parâmetro de rotação do eco é específico ou intercambiável?
    Trocar φ por π, e, √2, 2.0 — o resultado se mantém?

    Se sim → eco é detector genérico de coerência; φ é substituível.
    Se não → φ tem propriedade geométrica específica nesta função.

Substrato: séries temporais sintéticas com estrutura φ (mesma do TimeSeries_Eco).
Baseline já estabelecido: G=46.52%, G_eco(φ)=96.92% (+50.40%, p=0.0000).

Parâmetros testados (k):
    k=1.0    → controle: rotação identidade (sem rotação real)
    k=√2     → irracional algébrico
    k=φ      → razão áurea (referência)
    k=e      → número de Euler
    k=π      → pi
    k=2.0    → inteiro simples

Protocolo de idoneidade:
    Seeds geradas por timestamp — ninguém escolhe os valores.
    Resultados reportados integralmente — favoráveis ou não.
    Mesma arquitetura, mesma LR, mesmos dados para todos os k.
"""

import numpy as np
import json
import time
from scipy import stats
from utils_phi import (
    PHI, ALPHA, golden_activation, golden_activation_deriv,
    clip_grad, sigmoid
)

# ── Constantes testadas ───────────────────────────────────────────────────────
CONSTANTES = {
    "k=1.0":  1.0,
    "k=√2":   np.sqrt(2),
    "k=φ":    PHI,
    "k=e":    np.e,
    "k=π":    np.pi,
    "k=2.0":  2.0,
}

# ── Parâmetros do experimento ─────────────────────────────────────────────────
N_SEEDS   = 20
N_EPOCHS  = 60
N_TRAIN   = 400          # 200 por classe
N_TEST    = 100          # 50 por classe
DIM       = 128
HIDDEN    = 89           # Fibonacci
LR        = 0.01
N_ECO     = 3

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print(f"Experimento: Intercambiabilidade do parâmetro de rotação do eco")
print(f"Timestamp:   {TIMESTAMP}")
print(f"Seeds:       {SEEDS[0]} … {SEEDS[-1]}")
print(f"Constantes:  {list(CONSTANTES.keys())}")
print()

# ── Geração de dados (idêntica ao TimeSeries_Eco) ─────────────────────────────

def gerar_serie_phi(n, dim, rng):
    """Classe 1: série com frequências em proporção φ."""
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
        sinal = sinal + ruido
        sinal = sinal / (np.std(sinal) + 1e-8)
        X[i] = sinal
    return X

def gerar_ruido(n, dim, rng):
    """Classe 0: ruído gaussiano puro."""
    X = rng.normal(0, 1, (n, dim))
    X = X / (np.std(X, axis=1, keepdims=True) + 1e-8)
    return X

def gerar_dados(seed):
    rng = np.random.default_rng(seed)
    n_por_classe_train = N_TRAIN // 2
    n_por_classe_test  = N_TEST  // 2

    X_train = np.vstack([
        gerar_serie_phi(n_por_classe_train, DIM, rng),
        gerar_ruido(n_por_classe_train, DIM, rng)
    ])
    y_train = np.array([1]*n_por_classe_train + [0]*n_por_classe_train, dtype=float)

    X_test = np.vstack([
        gerar_serie_phi(n_por_classe_test, DIM, rng),
        gerar_ruido(n_por_classe_test, DIM, rng)
    ])
    y_test = np.array([1]*n_por_classe_test + [0]*n_por_classe_test, dtype=float)

    idx_train = rng.permutation(N_TRAIN)
    idx_test  = rng.permutation(N_TEST)
    return X_train[idx_train], y_train[idx_train], X_test[idx_test], y_test[idx_test]

# ── Eco com parâmetro k arbitrário ────────────────────────────────────────────

def eco_k(x, k, n_eco=N_ECO):
    """
    Eco ressonante com constante k (substitui φ).
    Varia tanto a rotação de fase (k) quanto o fator de blend (1/k).

    k=1.0 → controle: rotação identidade → eco ≈ zero → sinal inalterado.
    """
    x = np.asarray(x, dtype=float)
    sinal = x.copy()
    for _ in range(n_eco):
        freq      = np.fft.fft(sinal, axis=-1)
        nova_fase = np.angle(freq) * k
        reflexao  = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * nova_fase), axis=-1
        ))
        sinal = sinal + (reflexao - x) / k
    return sinal

# ── Rede neural (idêntica ao TimeSeries_Eco) ──────────────────────────────────

def inicializar_pesos(dim_in, hidden, rng):
    escala1 = np.sqrt(2.0 / dim_in)
    escala2 = np.sqrt(2.0 / hidden)
    W1 = rng.normal(0, escala1, (dim_in, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, escala2, (hidden, 1))
    b2 = np.zeros(1)
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = golden_activation(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2).squeeze()
    return y_hat, a1, z1

def backward(X, y, y_hat, a1, z1, W2):
    n = len(y)
    dL = (y_hat - y) / n

    dW2 = a1.T @ dL.reshape(-1, 1)
    db2 = dL.sum(keepdims=True)

    da1 = dL.reshape(-1, 1) * W2.T
    dz1 = da1 * golden_activation_deriv(z1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return dW1, db1, dW2, db2

def treinar(X_train, y_train, X_test, y_test, seed):
    rng = np.random.default_rng(seed + 999)
    W1, b1, W2, b2 = inicializar_pesos(X_train.shape[1], HIDDEN, rng)

    batch_size = 32
    n = len(X_train)

    for epoch in range(N_EPOCHS):
        idx = rng.permutation(n)
        batches = [idx[i:i+batch_size] for i in range(0, n - batch_size + 1, batch_size)]
        for batch in batches:
            Xb, yb = X_train[batch], y_train[batch]
            y_hat, a1, z1 = forward(Xb, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward(Xb, yb, y_hat, a1, z1, W2)
            W1 -= LR * clip_grad(dW1)
            b1 -= LR * np.clip(db1, -1, 1)
            W2 -= LR * clip_grad(dW2)
            b2 -= LR * np.clip(db2, -1, 1)

    y_hat_test, _, _ = forward(X_test, W1, b1, W2, b2)
    acc = np.mean((y_hat_test >= 0.5) == y_test)
    return float(acc)

# ── Execução ──────────────────────────────────────────────────────────────────

resultados = {nome: [] for nome in CONSTANTES}
resultados["G"] = []   # baseline sem eco

print(f"{'Seed':<14}", end="")
print(f"{'G (base)':<12}", end="")
for nome in CONSTANTES:
    print(f"{nome:<12}", end="")
print()
print("-" * (14 + 12 + 12 * len(CONSTANTES)))

for i, seed in enumerate(SEEDS):
    X_train, y_train, X_test, y_test = gerar_dados(seed)

    # Baseline sem eco
    acc_G = treinar(X_train, y_train, X_test, y_test, seed)
    resultados["G"].append(acc_G)

    linha = f"{seed:<14}{acc_G:<12.3f}"

    # Eco com cada constante k
    for nome, k in CONSTANTES.items():
        X_eco_train = eco_k(X_train, k)
        X_eco_test  = eco_k(X_test,  k)
        acc = treinar(X_eco_train, y_train, X_eco_test, y_test, seed)
        resultados[nome].append(acc)
        linha += f"{acc:<12.3f}"

    print(linha)

# ── Estatísticas ──────────────────────────────────────────────────────────────

print()
print("=" * (14 + 12 + 12 * len(CONSTANTES)))
print(f"\n{'Configuração':<14} {'Média':>8} {'Desvio':>8} {'Delta vs G':>12} {'p-valor':>10}")
print("-" * 56)

G_vals = np.array(resultados["G"])
print(f"{'G (base)':<14} {G_vals.mean():>8.4f} {G_vals.std():>8.4f} {'—':>12} {'—':>10}")

testes = {}
for nome, k in CONSTANTES.items():
    vals = np.array(resultados[nome])
    delta = vals.mean() - G_vals.mean()
    _, p = stats.wilcoxon(vals, G_vals)
    testes[nome] = {"k": k, "mean": float(vals.mean()), "std": float(vals.std()),
                    "delta": float(delta), "p_value": float(p), "values": vals.tolist()}
    print(f"{nome:<14} {vals.mean():>8.4f} {vals.std():>8.4f} {delta:>+12.4f} {p:>10.6f}")

# ── Ranking ───────────────────────────────────────────────────────────────────

print("\n── Ranking por acurácia média ───────────────────────────────────────────")
ranking = sorted(testes.items(), key=lambda x: x[1]["mean"], reverse=True)
for pos, (nome, r) in enumerate(ranking, 1):
    sinal = "✅" if r["delta"] > 0 and r["p_value"] < 0.05 else "❌"
    print(f"  {pos}. {nome:<10} {r['mean']:.4f}  Δ{r['delta']:+.4f}  p={r['p_value']:.6f} {sinal}")

# ── Salvar resultados ─────────────────────────────────────────────────────────

saida = {
    "experimento": "Eco_Intercambiabilidade",
    "substrato": "séries temporais φ — mesmo do TimeSeries_Eco",
    "pergunta": "φ como parâmetro de rotação é específico ou intercambiável?",
    "referencia_TimeSeries_Eco": {
        "G": 0.4653, "G_eco_phi": 0.9693, "delta": 0.504, "p_value": 0.0
    },
    "n_seeds": N_SEEDS,
    "n_epochs": N_EPOCHS,
    "n_train": N_TRAIN,
    "n_test": N_TEST,
    "dim": DIM,
    "hidden": HIDDEN,
    "n_eco": N_ECO,
    "timestamp": TIMESTAMP,
    "seeds": SEEDS,
    "G_baseline": {
        "mean": float(G_vals.mean()),
        "std":  float(G_vals.std()),
        "values": G_vals.tolist()
    },
    "constantes_testadas": {
        nome: {
            "valor_k": float(k),
            **testes[nome]
        }
        for nome, k in CONSTANTES.items()
    },
    "ranking": [
        {"posicao": i+1, "nome": nome, "mean": r["mean"], "delta": r["delta"]}
        for i, (nome, r) in enumerate(ranking)
    ],
}

with open("intercambiabilidade_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print(f"\nResultados salvos: intercambiabilidade_results.json")
print(f"\nResposta à pergunta:")
melhor = ranking[0]
phi_pos = next(i for i, (n, _) in enumerate(ranking) if n == "k=φ") + 1
print(f"  Melhor resultado: {melhor[0]} ({melhor[1]['mean']:.4f})")
print(f"  φ ficou em posição: {phi_pos}/{len(CONSTANTES)}")
if phi_pos == 1:
    print("  → φ É o parâmetro mais eficaz. Propriedade específica confirmada.")
elif phi_pos <= 2:
    print("  → φ entre os melhores. Possível propriedade específica — investigar.")
else:
    print("  → φ não se destaca. Eco é detector genérico; φ intercambiável nesta função.")
