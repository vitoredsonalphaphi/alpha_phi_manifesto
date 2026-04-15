# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Eco_Fractal_Coerente.py
Vitor Edson Delavi · Florianópolis · 2026

Hipótese — o limiar de nascimento exige campo preparado:

    No experimento anterior (eco_fractal), o batch era 50% φ + 50% ruído.
    A coerência coletiva do campo: 0.018 — quase zero.
    O limiar 1/φ² = 0.382 nunca foi atingido — 0/20 seeds.

    Analogia com neurônio: o potencial de ação exige gradiente suficiente.
    Com 50% de ruído, a temperatura da rede é baixa demais para o disparo.

    Este experimento testa o eco_fractal com campo 100% coerente:
        Classe 1: séries φ em registro grave  (freq_base 0.5–1.5)
        Classe 0: séries φ em registro agudo  (freq_base 2.5–4.0)
        Ambas com estrutura φ — sem ruído gaussiano puro.

    Se o batch é todo coerente, a coerência coletiva sobe.
    Se sobe o suficiente, Δcoerência pode cruzar 1/φ² = 0.382.
    O nascimento fractal pode finalmente ocorrer.

Diagnóstico incluído:
    Mede Δcoerência real por seed — em batch misto e em batch coerente.
    Identifica onde o threshold está em relação ao que os dados produzem.

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
N_SEEDS    = 20
N_EPOCHS   = 60
N_TRAIN    = 400
N_TEST     = 100
DIM        = 128
HIDDEN     = 89
LR         = 0.01
N_ECO      = 3
MAX_OITAVAS = 3
LIMIAR     = C_PHI   # 1/φ² = 0.382

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print("Experimento: eco_fractal com campo 100% coerente")
print(f"Hipótese: batch sem ruído → coerência coletiva alta → nascimento ativado")
print(f"Limiar de nascimento: 1/φ² = {LIMIAR:.4f}")
print(f"Timestamp: {TIMESTAMP}\n")

# ── Dados — duas classes de séries φ, sem ruído puro ─────────────────────────

def gerar_serie_phi(n, dim, rng, freq_min, freq_max):
    """Série com estrutura φ em faixa de frequência específica."""
    t = np.linspace(0, 4 * np.pi, dim)
    X = np.zeros((n, dim))
    for i in range(n):
        freq_base = rng.uniform(freq_min, freq_max)
        sinal = np.zeros(dim)
        for k in range(5):
            freq_k = freq_base * (PHI ** k)
            amp_k  = rng.uniform(0.3, 1.0) / (k + 1)
            fase_k = rng.uniform(0, 2 * np.pi)
            sinal += amp_k * np.sin(freq_k * t + fase_k)
        ruido = rng.normal(0, 0.05, dim)   # ruído mínimo (5%), não dominante
        sinal = (sinal + ruido) / (np.std(sinal + ruido) + 1e-8)
        X[i] = sinal
    return X

def gerar_dados(seed):
    rng = np.random.default_rng(seed)
    n_tr, n_te = N_TRAIN // 2, N_TEST // 2
    # Classe 1: registro grave (0.5–1.5) — φ-coerente
    # Classe 0: registro agudo (2.5–4.0) — φ-coerente, frequência diferente
    X_tr = np.vstack([
        gerar_serie_phi(n_tr, DIM, rng, 0.5, 1.5),
        gerar_serie_phi(n_tr, DIM, rng, 2.5, 4.0),
    ])
    y_tr = np.array([1]*n_tr + [0]*n_tr, dtype=float)
    X_te = np.vstack([
        gerar_serie_phi(n_te, DIM, rng, 0.5, 1.5),
        gerar_serie_phi(n_te, DIM, rng, 2.5, 4.0),
    ])
    y_te = np.array([1]*n_te + [0]*n_te, dtype=float)
    idx_tr = rng.permutation(N_TRAIN)
    idx_te = rng.permutation(N_TEST)
    return X_tr[idx_tr], y_tr[idx_tr], X_te[idx_te], y_te[idx_te]

# ── Funções de eco ────────────────────────────────────────────────────────────

def medir_coerencia(x):
    """Coerência espectral média do batch ∈ [0, 1]."""
    freq   = np.fft.fft(x, axis=-1)
    e      = np.abs(freq)
    e_norm = np.clip(e / (e.sum(axis=-1, keepdims=True) + 1e-8), 1e-10, 1.0)
    entropia  = -np.sum(e_norm * np.log(e_norm), axis=-1)
    coerencia = 1.0 - entropia / np.log(x.shape[-1])
    return float(np.mean(coerencia))

def ciclo_eco(sinal, k, x_orig, n_eco=N_ECO):
    for _ in range(n_eco):
        freq      = np.fft.fft(sinal, axis=-1)
        nova_fase = np.angle(freq) * k
        reflexao  = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * nova_fase), axis=-1))
        sinal = sinal + (reflexao - x_orig) / k
    return sinal

def eco_ressonante(x, phi=PHI, n_eco=N_ECO):
    x = np.asarray(x, dtype=float)
    s = x.copy()
    for _ in range(n_eco):
        freq      = np.fft.fft(s, axis=-1)
        nova_fase = np.angle(freq) * phi
        reflexao  = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * nova_fase), axis=-1))
        s = s + (reflexao - x) / phi
    return s

def eco_fractal(x, phi=PHI, limiar=LIMIAR, n_eco=N_ECO, max_oitavas=MAX_OITAVAS):
    x = np.asarray(x, dtype=float)
    sinal   = x.copy()
    k       = phi
    oitavas = 0
    deltas  = []

    for _ in range(max_oitavas):
        coh_antes  = medir_coerencia(sinal)
        sinal      = ciclo_eco(sinal, k, x)
        coh_depois = medir_coerencia(sinal)
        delta      = coh_depois - coh_antes
        deltas.append(delta)

        if delta >= limiar:
            oitavas += 1
            k = k * phi   # sobe oitava: φ → φ² → φ³
        else:
            break

    return sinal, oitavas, k, deltas

# ── Diagnóstico — Δcoerência real ─────────────────────────────────────────────

print("── Diagnóstico: Δcoerência real por seed ────────────────────────────")
print(f"{'Seed':<14} {'coh_campo':>11} {'Δcoh_eco1':>11} {'Δcoh_eco2':>11} {'Δcoh_eco3':>11} {'nasceu?':>9}")
print("-" * 72)

diag_deltas = []
for seed in SEEDS[:10]:   # amostra de 10 seeds para diagnóstico
    X_tr, y_tr, _, _ = gerar_dados(seed)
    coh_inicial = medir_coerencia(X_tr)
    _, oitavas, k_final, deltas = eco_fractal(X_tr)
    diag_deltas.append(deltas)
    nasceu = "✅" if oitavas > 0 else "❌"
    d_str = "   ".join(f"{d:+.4f}" for d in deltas)
    print(f"{seed:<14} {coh_inicial:>11.4f}   {d_str}   {nasceu:>9}")

print()
todos_deltas = [d for ds in diag_deltas for d in ds]
print(f"Δcoerência médio (todos os ciclos): {np.mean(todos_deltas):+.4f}")
print(f"Δcoerência máximo observado:        {max(todos_deltas):+.4f}")
print(f"Limiar necessário:                   {LIMIAR:.4f}")
print()

# ── Execução completa ─────────────────────────────────────────────────────────

res = {"G": [], "G_eco_phi": [], "G_eco_fractal": []}
oitavas_log = []
k_final_log = []

print(f"{'Seed':<14} {'G':>8} {'G_eco_φ':>10} {'G_fractal':>11} {'Oitavas':>9} {'k_final':>9}")
print("-" * 65)

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

    acc_G   = treinar(X_tr, y_tr, X_te, y_te, seed)
    acc_phi = treinar(eco_ressonante(X_tr), y_tr, eco_ressonante(X_te), y_te, seed)

    Xtr_fr, oit_tr, k_tr, _ = eco_fractal(X_tr)
    Xte_fr, oit_te, k_te, _ = eco_fractal(X_te)
    acc_fr = treinar(Xtr_fr, y_tr, Xte_fr, y_te, seed)

    res["G"].append(acc_G)
    res["G_eco_phi"].append(acc_phi)
    res["G_eco_fractal"].append(acc_fr)
    oitavas_log.append(oit_tr)
    k_final_log.append(k_tr)

    print(f"{seed:<14} {acc_G:>8.3f} {acc_phi:>10.3f} {acc_fr:>11.3f} "
          f"{oit_tr:>9d} {k_tr:>9.4f}")

# ── Estatísticas ──────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
G     = np.array(res["G"])
E_phi = np.array(res["G_eco_phi"])
E_fr  = np.array(res["G_eco_fractal"])

_, p_phi  = stats.wilcoxon(E_phi, G)
_, p_fr   = stats.wilcoxon(E_fr,  G)

print(f"\n{'Modo':<18} {'Média':>8} {'Desvio':>8} {'Δ vs G':>10} {'p-valor':>10}")
print("-" * 58)
print(f"{'G (base)':<18} {G.mean():>8.4f} {G.std():>8.4f} {'—':>10} {'—':>10}")
print(f"{'G_eco_phi':<18} {E_phi.mean():>8.4f} {E_phi.std():>8.4f} "
      f"{E_phi.mean()-G.mean():>+10.4f} {p_phi:>10.6f}")
print(f"{'G_eco_fractal':<18} {E_fr.mean():>8.4f} {E_fr.std():>8.4f} "
      f"{E_fr.mean()-G.mean():>+10.4f} {p_fr:>10.6f}")

if not all(a == b for a, b in zip(res["G_eco_fractal"], res["G_eco_phi"])):
    _, p_comp = stats.wilcoxon(E_fr, E_phi)
    print(f"\nfractal vs eco_phi: Δ={E_fr.mean()-E_phi.mean():+.4f}  p={p_comp:.6f}")

print(f"\nOitavas ativadas:")
print(f"  Seeds com nascimento: {sum(o > 0 for o in oitavas_log)}/20")
print(f"  Oitavas médias: {np.mean(oitavas_log):.2f}")
print(f"  Distribuição: {sorted(set(oitavas_log))}")
print(f"  k_final médio: {np.mean(k_final_log):.4f}  "
      f"(φ={PHI:.4f}, φ²={PHI**2:.4f}, φ³={PHI**3:.4f})")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "Eco_Fractal_Coerente",
    "hipotese": "campo 100% coerente → coerência coletiva alta → nascimento fratal ativado",
    "substrato": "duas classes de séries φ (grave 0.5-1.5 / agudo 2.5-4.0) — sem ruído puro",
    "referencia_fractal_misto": {"oitavas": 0, "comentario": "50% ruído → coh=0.018 → limiar nunca atingido"},
    "limiar_nascimento": LIMIAR,
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "n_train": N_TRAIN, "n_test": N_TEST,
    "dim": DIM, "hidden": HIDDEN, "n_eco": N_ECO, "max_oitavas": MAX_OITAVAS,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "diagnostico": {
        "delta_medio": float(np.mean(todos_deltas)),
        "delta_maximo": float(max(todos_deltas)),
        "limiar": LIMIAR,
        "nascimento_possivel": bool(max(todos_deltas) >= LIMIAR),
    },
    "resultados": {
        "G":             {"mean": float(G.mean()),     "std": float(G.std()),     "values": G.tolist()},
        "G_eco_phi":     {"mean": float(E_phi.mean()), "std": float(E_phi.std()), "values": E_phi.tolist()},
        "G_eco_fractal": {"mean": float(E_fr.mean()),  "std": float(E_fr.std()),  "values": E_fr.tolist()},
    },
    "oitavas_por_seed": oitavas_log,
    "k_final_por_seed": k_final_log,
    "seeds_com_nascimento": sum(o > 0 for o in oitavas_log),
}

with open("eco_fractal_coerente_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print("\nResultados salvos: eco_fractal_coerente_results.json")

print("\nInterpretação:")
if sum(o > 0 for o in oitavas_log) > 0:
    print(f"  ✅ Nascimento ativado em {sum(o>0 for o in oitavas_log)}/20 seeds.")
    print(f"  O campo 100% coerente cruzou o limiar — análogo ao potencial de ação.")
    if E_fr.mean() > E_phi.mean():
        print(f"  ✅ eco_fractal supera eco_phi: {E_fr.mean():.4f} vs {E_phi.mean():.4f}")
    else:
        print(f"  ≈  eco_fractal ≈ eco_phi apesar do nascimento — oitava não ajuda na classificação.")
else:
    print(f"  ⚠️  Nascimento ainda não ativado — Δcoerência máximo: {max(todos_deltas):+.4f}")
    print(f"  Limiar 1/φ² = {LIMIAR:.4f} ainda acima do alcançável neste substrato.")
    print(f"  Próximo passo: testar limiar adaptativo = Δcoerência_máximo × φ")
