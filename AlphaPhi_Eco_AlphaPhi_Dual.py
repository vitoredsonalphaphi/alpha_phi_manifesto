# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Eco_AlphaPhi_Dual.py
Vitor Edson Delavi · Florianópolis · 2026

Pergunta central:
    Desde o início, o projeto se chama Alpha-Phi.
    φ recebeu todas as progressões experimentais:
        rotação de fase, ativação, curvatura hiperbólica, calibração coletiva.
    α apareceu no nome, nas constantes, na arquitetura —
    mas nunca como parâmetro de acoplamento na função de reinjeção.

    Hipótese:
        φ calibra o campo — proporção, estrutura, expansão.
        α controla o acoplamento — intensidade com que o eco retorna ao sinal.
        Juntos, como funções distintas e complementares no mesmo método.

    α como constante de acoplamento eletromagnético:
        Na física, α mede a força de interação entre campo e matéria.
        No eco: (reflexao - x) é o resíduo — a "força" entre eco e original.
        α como coeficiente de reinjeção = α como acoplamento eco↔sinal.

Modos testados:
    G                → baseline
    G_eco_fononico   → φ calibra campo (k do campo), /k na reinjeção  [melhor atual]
    G_eco_dual_v1    → φ calibra campo (k do campo), ×α na reinjeção  [α puro]
    G_eco_dual_v2    → φ calibra campo (k do campo), ×α/k na reinjeção [α+φ juntos]
    G_eco_dual_v3    → φ calibra campo (k do campo), ×(α·137) na reinjeção [α em escala natural]

Nota sobre α·137:
    1/α ≈ 137 — em unidades naturais, α·137 ≈ 1.
    Testar α·137 como acoplamento verifica se a escala natural de α
    (não o valor bruto 0.0073) é o parâmetro relevante.

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
ALPHA_NATURAL = ALPHA * (1.0 / ALPHA)  # = 1.0 — α em escala natural (α·137≈1)

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print("Experimento: Eco α-φ Dual — α como acoplamento, φ como campo")
print(f"Substrato: séries temporais φ")
print(f"α = {ALPHA:.8f}  |  1/α ≈ {1/ALPHA:.2f}  |  φ = {PHI:.6f}")
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

# ── Medição do campo coletivo (fonônico) ──────────────────────────────────────

def medir_campo(X):
    freq_batch = np.fft.fft(X, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)
    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia   = -np.sum(amp_norm * np.log(amp_norm))
    coh_campo  = float(1.0 - entropia / np.log(X.shape[-1]))
    k          = K_MIN + (PHI - K_MIN) * coh_campo
    return k, coh_campo

# ── Funções de eco ─────────────────────────────────────────────────────────────

def eco_fononico(X, n_eco=N_ECO):
    """Melhor resultado atual: φ calibra campo, /k na reinjeção."""
    X = np.asarray(X, dtype=float)
    k, coh = medir_campo(X)
    s = X.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) / k          # reinjeção: /k
    return s, k, coh

def eco_dual_v1(X, n_eco=N_ECO):
    """φ calibra campo (k), α como acoplamento puro na reinjeção."""
    X = np.asarray(X, dtype=float)
    k, coh = medir_campo(X)
    s = X.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) * ALPHA      # reinjeção: ×α
    return s, k, coh

def eco_dual_v2(X, n_eco=N_ECO):
    """φ calibra campo (k), α/k como acoplamento na reinjeção."""
    X = np.asarray(X, dtype=float)
    k, coh = medir_campo(X)
    s = X.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) * ALPHA / k  # reinjeção: ×α/k
    return s, k, coh

def eco_dual_v3(X, n_eco=N_ECO):
    """φ calibra campo (k), α em escala natural (α·1/α=1 → testar α·10 como meio-termo)."""
    X = np.asarray(X, dtype=float)
    k, coh = medir_campo(X)
    # α·(1/α) = 1.0 é idêntico a /1 — sem sentido como teste
    # Testamos α como fração de 1/k: acoplamento no meio entre α puro e /k
    acoplamento = np.sqrt(ALPHA / k)        # média geométrica entre α e 1/k
    s = X.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) * acoplamento  # reinjeção: ×√(α/k)
    return s, k, coh

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

# ── Execução ──────────────────────────────────────────────────────────────────

modos = ["G", "G_eco_fononico", "G_dual_v1_α", "G_dual_v2_α/k", "G_dual_v3_√(α/k)"]
res   = {m: [] for m in modos}
k_log = []

header = f"{'Seed':<14}" + "".join(f"{m:<18}" for m in modos)
print(header)
print("-" * len(header))

for seed in SEEDS:
    X_tr, y_tr, X_te, y_te = gerar_dados(seed)

    Xfn_tr, k_tr, _ = eco_fononico(X_tr)
    Xfn_te, _,    _ = eco_fononico(X_te)
    Xd1_tr, _,    _ = eco_dual_v1(X_tr)
    Xd1_te, _,    _ = eco_dual_v1(X_te)
    Xd2_tr, _,    _ = eco_dual_v2(X_tr)
    Xd2_te, _,    _ = eco_dual_v2(X_te)
    Xd3_tr, _,    _ = eco_dual_v3(X_tr)
    Xd3_te, _,    _ = eco_dual_v3(X_te)

    configs = {
        "G":               (X_tr,    X_te),
        "G_eco_fononico":  (Xfn_tr,  Xfn_te),
        "G_dual_v1_α":     (Xd1_tr,  Xd1_te),
        "G_dual_v2_α/k":   (Xd2_tr,  Xd2_te),
        "G_dual_v3_√(α/k)":(Xd3_tr,  Xd3_te),
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
G  = np.array(res["G"])
Gf = np.array(res["G_eco_fononico"])

print(f"\n{'Modo':<22} {'Média':>8} {'Desvio':>8} {'Δ vs G':>10} {'Δ vs fonônico':>14} {'p vs G':>10}")
print("-" * 76)
print(f"{'G (base)':<22} {G.mean():>8.4f} {G.std():>8.4f} {'—':>10} {'—':>14} {'—':>10}")
print(f"{'G_eco_fononico':<22} {Gf.mean():>8.4f} {Gf.std():>8.4f} "
      f"{Gf.mean()-G.mean():>+10.4f} {'[referência]':>14} ", end="")
_, p = stats.wilcoxon(Gf, G)
print(f"{p:>10.6f}")

resultados = {}
for m in modos[2:]:
    v = np.array(res[m])
    _, p_g  = stats.wilcoxon(v, G)
    _, p_fn = stats.wilcoxon(v, Gf)
    sinal = "✅" if v.mean() > Gf.mean() and p_fn < 0.05 else (
            "≈" if abs(v.mean() - Gf.mean()) < 0.005 else "⚠️")
    print(f"{m:<22} {v.mean():>8.4f} {v.std():>8.4f} "
          f"{v.mean()-G.mean():>+10.4f} {v.mean()-Gf.mean():>+14.4f} "
          f"{p_g:>10.6f} {sinal}")
    resultados[m] = {
        "mean": float(v.mean()), "std": float(v.std()),
        "delta_G": float(v.mean()-G.mean()),
        "delta_fononico": float(v.mean()-Gf.mean()),
        "p_vs_G": float(p_g), "p_vs_fononico": float(p_fn),
        "values": v.tolist()
    }

print(f"\nk_otimo médio (campo fonônico): {np.mean(k_log):.4f}")
print(f"α = {ALPHA:.8f}  |  acoplamento v1: α={ALPHA:.6f}  "
      f"v2: α/k={ALPHA/np.mean(k_log):.6f}  "
      f"v3: √(α/k)={np.sqrt(ALPHA/np.mean(k_log)):.6f}")

# ── Ranking ───────────────────────────────────────────────────────────────────

print("\n── Ranking ──────────────────────────────────────────────────────────")
ranking = sorted([(m, np.array(res[m]).mean()) for m in modos], key=lambda x: -x[1])
for pos, (m, mean) in enumerate(ranking, 1):
    print(f"  {pos}. {m:<26} {mean:.4f}")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "Eco_AlphaPhi_Dual",
    "pergunta": "α como acoplamento de reinjeção + φ como calibrador de campo",
    "hipotese": "φ define proporção/estrutura; α define intensidade de acoplamento eco↔sinal",
    "substrato": "séries temporais φ",
    "referencia": {"G_eco_fononico": float(Gf.mean()), "comentario": "melhor resultado anterior"},
    "constantes": {"PHI": PHI, "ALPHA": ALPHA, "K_MIN": float(K_MIN)},
    "acoplamentos_testados": {
        "v1": "×α (α puro como coupling)",
        "v2": "×α/k (α escalado por k do campo)",
        "v3": "×√(α/k) (média geométrica entre α e 1/k)"
    },
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "n_train": N_TRAIN, "n_test": N_TEST,
    "dim": DIM, "hidden": HIDDEN, "n_eco": N_ECO,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "resultados": {
        "G":             {"mean": float(G.mean()),  "std": float(G.std()),  "values": G.tolist()},
        "G_eco_fononico":{"mean": float(Gf.mean()), "std": float(Gf.std()), "values": Gf.tolist()},
        **resultados
    },
    "k_otimos": k_log,
    "ranking": [(m, float(np.array(res[m]).mean())) for m, _ in ranking],
}

with open("eco_alphaphi_dual_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print("\nResultados salvos: eco_alphaphi_dual_results.json")
