# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Eco_Fractal.py
Vitor Edson Delavi · Florianópolis · 2026

Hipótese — isomorfismo semente/dado:

    A semente na terra experimenta quatro tensões em uníssono:
    energia ascendente interna, resistência descendente da terra,
    ressonância do ambiente externo, e ancoragem pós-nascimento.
    O ponto de dobra não é da semente — é do sistema inteiro.
    Depois do limiar, a mesma força que resistia passa a ancorar.

    Isomorfismo com o dado:
        Tensão interna ascendente  → estrutura coerente do dado
        Terra / resistência         → ruído, fases aleatórias
        Ambiente ressonante         → rede neural que recebe o dado
        Ponto de dobra              → threshold de coerência (1/φ² = 0.382)
        Raiz pós-nascimento         → gradiente que ancora o aprendizado

    eco_fractal: eco com detecção de limiar e expansão por oitavas.
        Estado 1 (pré-limiar): eco opera com k=φ, observa coerência.
        Nascimento: se Δcoerência ≥ 1/φ² → sobe oitava: k → k*φ.
        Estado 2 (pós-limiar): eco opera com k=φ² (oitava acima).
        Fractal: cada nascimento habilita o próximo nível.
        φ → φ² → φ³ — auto-similar, como raiz e galho.

    Analogia fonon: eco não age sobre ponto individual do dado —
    age sobre o padrão coletivo de fases. Vibração da rede, não do átomo.

Experimento:
    Substrato: séries temporais φ (mesmo do TimeSeries_Eco).
    Comparação: G vs G_eco_phi vs G_eco_fractal (1, 2, 3 oitavas máx).
    Métricas adicionais: oitavas alcançadas por seed, k final médio.

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
N_SEEDS    = 20
N_EPOCHS   = 60
N_TRAIN    = 400
N_TEST     = 100
DIM        = 128
HIDDEN     = 89
LR         = 0.01
N_ECO      = 3
MAX_OITAVAS = 3

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print(f"Experimento: eco_fractal — limiar de nascimento + expansão por oitavas")
print(f"Isomorfismo: semente/dado — ponto de dobra = 1/φ² = {C_PHI:.4f}")
print(f"Oitavas: φ={PHI:.4f} → φ²={PHI**2:.4f} → φ³={PHI**3:.4f}")
print(f"Timestamp: {TIMESTAMP}")
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

def medir_coerencia(x):
    """Coerência espectral média do batch — ∈ [0, 1]."""
    freq  = np.fft.fft(x, axis=-1)
    e     = np.abs(freq)
    e_norm = np.clip(e / (e.sum(axis=-1, keepdims=True) + 1e-8), 1e-10, 1.0)
    entropia  = -np.sum(e_norm * np.log(e_norm), axis=-1)
    coerencia = 1.0 - entropia / np.log(x.shape[-1])
    return float(np.mean(coerencia))

def ciclo_eco(sinal, k, x_orig, n_eco=N_ECO):
    """Bloco de n_eco ciclos com parâmetro k."""
    for _ in range(n_eco):
        freq     = np.fft.fft(sinal, axis=-1)
        nova_fase = np.angle(freq) * k
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * nova_fase), axis=-1))
        sinal = sinal + (reflexao - x_orig) / k
    return sinal

def eco_ressonante(x, phi=PHI, n_eco=N_ECO):
    """Eco original — referência."""
    x = np.asarray(x, dtype=float)
    sinal = x.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(sinal, axis=-1)
        nova_fase = np.angle(freq) * phi
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * nova_fase), axis=-1))
        sinal = sinal + (reflexao - x) / phi
    return sinal

def eco_fractal(x, phi=PHI, limiar=C_PHI, n_eco=N_ECO, max_oitavas=MAX_OITAVAS):
    """
    Eco com detecção de limiar e expansão por oitavas.

    Pré-limiar:  k = φ  — observa, rotaciona, mede coerência.
    Nascimento:  Δcoerência ≥ limiar (1/φ² = 0.382) → k *= φ (sobe oitava).
    Pós-limiar:  opera na nova oitava — mesma força, novo regime.
    Fractal:     até max_oitavas nascimentos possíveis.

    Retorna: (sinal_transformado, oitavas_alcancadas, k_final)
    """
    x = np.asarray(x, dtype=float)
    sinal = x.copy()
    k = phi
    oitavas = 0

    for _ in range(max_oitavas):
        coh_antes  = medir_coerencia(sinal)
        sinal      = ciclo_eco(sinal, k, x)
        coh_depois = medir_coerencia(sinal)

        delta = coh_depois - coh_antes

        if delta >= limiar:
            # Nascimento — sobe oitava
            oitavas += 1
            k = k * phi
        else:
            # Sem nascimento — permanece na oitava atual
            break

    return sinal, oitavas, k

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

resultados = {"G": [], "G_eco_phi": [], "G_eco_fractal": []}
oitavas_por_seed = []
k_final_por_seed = []

print(f"{'Seed':<14} {'G':>8} {'G_eco_φ':>10} {'G_fractal':>11} {'Oitavas':>9} {'k_final':>9}")
print("-" * 65)

for seed in SEEDS:
    X_tr, y_tr, X_te, y_te = gerar_dados(seed)

    # Baseline
    acc_G = treinar(X_tr, y_tr, X_te, y_te, seed)
    resultados["G"].append(acc_G)

    # Eco original φ
    Xtr_eco = eco_ressonante(X_tr)
    Xte_eco = eco_ressonante(X_te)
    acc_phi = treinar(Xtr_eco, y_tr, Xte_eco, y_te, seed)
    resultados["G_eco_phi"].append(acc_phi)

    # Eco fractal
    Xtr_fr, oit_tr, k_tr = eco_fractal(X_tr)
    Xte_fr, oit_te, k_te = eco_fractal(X_te)
    acc_fr = treinar(Xtr_fr, y_tr, Xte_fr, y_te, seed)
    resultados["G_eco_fractal"].append(acc_fr)

    oitavas_por_seed.append(oit_tr)
    k_final_por_seed.append(k_tr)

    print(f"{seed:<14} {acc_G:>8.3f} {acc_phi:>10.3f} {acc_fr:>11.3f} "
          f"{oit_tr:>9d} {k_tr:>9.4f}")

# ── Estatísticas ──────────────────────────────────────────────────────────────

print()
print("=" * 65)
G     = np.array(resultados["G"])
E_phi = np.array(resultados["G_eco_phi"])
E_fr  = np.array(resultados["G_eco_fractal"])

_, p_phi = stats.wilcoxon(E_phi, G)
_, p_fr  = stats.wilcoxon(E_fr,  G)
_, p_comp = stats.wilcoxon(E_fr, E_phi)

print(f"\n{'Modo':<18} {'Média':>8} {'Desvio':>8} {'Δ vs G':>10} {'p-valor':>10}")
print("-" * 58)
print(f"{'G (base)':<18} {G.mean():>8.4f} {G.std():>8.4f} {'—':>10} {'—':>10}")
print(f"{'G_eco_phi':<18} {E_phi.mean():>8.4f} {E_phi.std():>8.4f} "
      f"{E_phi.mean()-G.mean():>+10.4f} {p_phi:>10.6f}")
print(f"{'G_eco_fractal':<18} {E_fr.mean():>8.4f} {E_fr.std():>8.4f} "
      f"{E_fr.mean()-G.mean():>+10.4f} {p_fr:>10.6f}")

print(f"\nComparação direta fractal vs eco_phi:")
print(f"  Δ = {E_fr.mean()-E_phi.mean():+.4f}  p = {p_comp:.6f}")

print(f"\nOitavas alcançadas por seed:")
print(f"  Média: {np.mean(oitavas_por_seed):.2f}  "
      f"Máx: {max(oitavas_por_seed)}  "
      f"Distribuição: {sorted(set(oitavas_por_seed))}")
print(f"  k_final médio: {np.mean(k_final_por_seed):.4f}  "
      f"(φ={PHI:.4f}, φ²={PHI**2:.4f}, φ³={PHI**3:.4f})")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "Eco_Fractal",
    "hipotese": "isomorfismo semente/dado — limiar de nascimento + oitavas φ",
    "mecanismo": {
        "pre_limiar": "eco com k=φ, mede Δcoerência a cada bloco",
        "nascimento": f"Δcoerência ≥ {C_PHI:.4f} (1/φ²) → k *= φ (sobe oitava)",
        "pos_limiar": "opera na nova oitava — mesma força, novo regime",
        "fractal": "φ → φ² → φ³, auto-similar, até max_oitavas"
    },
    "limiar_nascimento": C_PHI,
    "oitavas_phi": [PHI, PHI**2, PHI**3],
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "n_train": N_TRAIN, "n_test": N_TEST,
    "dim": DIM, "hidden": HIDDEN, "n_eco": N_ECO,
    "max_oitavas": MAX_OITAVAS,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "resultados": {
        "G":             {"mean": float(G.mean()),     "std": float(G.std()),     "values": G.tolist()},
        "G_eco_phi":     {"mean": float(E_phi.mean()), "std": float(E_phi.std()), "values": E_phi.tolist()},
        "G_eco_fractal": {"mean": float(E_fr.mean()),  "std": float(E_fr.std()),  "values": E_fr.tolist()},
    },
    "comparacoes": {
        "eco_phi_vs_G":     {"delta": float(E_phi.mean()-G.mean()), "p_value": float(p_phi)},
        "eco_fractal_vs_G": {"delta": float(E_fr.mean()-G.mean()),  "p_value": float(p_fr)},
        "fractal_vs_phi":   {"delta": float(E_fr.mean()-E_phi.mean()), "p_value": float(p_comp)},
    },
    "oitavas_por_seed": oitavas_por_seed,
    "k_final_por_seed": k_final_por_seed,
}

with open("eco_fractal_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print(f"\nResultados salvos: eco_fractal_results.json")

print(f"\nInterpretação:")
if E_fr.mean() > E_phi.mean() and p_comp < 0.05:
    print(f"  ✅ eco_fractal supera eco_phi — oitavas amplificam o resultado.")
elif abs(E_fr.mean() - E_phi.mean()) < 0.01:
    print(f"  ≈  eco_fractal ≈ eco_phi — limiar não ativado ou efeito neutro.")
else:
    print(f"  ⚠️  eco_phi ainda superior — limiar pode precisar de ajuste.")

if np.mean(oitavas_por_seed) > 0.5:
    print(f"  ✅ Nascimentos detectados — limiar 1/φ² ativo em {sum(o>0 for o in oitavas_por_seed)}/20 seeds.")
else:
    print(f"  ⚠️  Poucos nascimentos — Δcoerência raramente ≥ {C_PHI:.3f}. Testar limiar menor.")
