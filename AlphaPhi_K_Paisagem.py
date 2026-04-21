# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_K_Paisagem.py
Vitor Edson Delavi · Florianópolis · 2026

Pergunta:
    O eco_fononico usa interpolação contínua: k = √2 + (φ - √2) × coh.
    Isso assume que a relação k_ótimo × coerência é monotônica e suave.

    Mas existe estrutura discreta? Picos? Platôs?
    A zona ótima é uma faixa ou um ponto?

Método:
    Gerar batches com coerência controlada em 7 níveis:
        c ∈ {0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90}

    Para cada nível de coerência, testar 9 valores de k fixo:
        k ∈ {1.00, 1.20, √2≈1.41, 1.60, φ≈1.62, 1.80, 2.00, φ²≈2.62, φ³≈4.24}

    Medir acurácia média (10 seeds) em cada célula (coerência, k).
    Mapear a paisagem resultante.

Predição do modelo contínuo (eco_fononico):
    Diagonal ascendente — cada coerência tem um k_ótimo crescente.
    Sem picos discretos.

Predição do modelo de oitavas:
    Faixas diagonais com plateaus — como camadas eletrônicas.
    k_ótimo salta entre valores em vez de crescer continuamente.

Protocolo: seeds por timestamp, resultados integralmente reportados.
"""

import numpy as np
import json
import time
from utils_phi import PHI, golden_activation, clip_grad, sigmoid

# ── Parâmetros ────────────────────────────────────────────────────────────────
N_SEEDS  = 10
N_EPOCHS = 60
N_TRAIN  = 400
N_TEST   = 100
DIM      = 128
HIDDEN   = 89
LR       = 0.01
N_ECO    = 3

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

# Níveis de coerência controlada
NIVEIS_COH = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]

# Valores de k a testar
K_VALS = [1.00, 1.20, np.sqrt(2), 1.60, PHI, 1.80, 2.00, PHI**2, PHI**3]
K_NOMES = ["1.00", "1.20", "√2", "1.60", "φ", "1.80", "2.00", "φ²", "φ³"]

print("Experimento: Paisagem k × Coerência")
print(f"Coerências: {NIVEIS_COH}")
print(f"k valores:  {[f'{k:.4f}' for k in K_VALS]}")
print(f"Seeds: {N_SEEDS}  |  Timestamp: {TIMESTAMP}\n")

# ── Dados com coerência controlada ────────────────────────────────────────────

def gerar_batch_coerente(n, dim, nivel_coh, rng):
    """
    Gera batch com coerência espectral aproximada ao nível solicitado.
    nivel_coh ≈ 0 → ruído puro (alta entropia)
    nivel_coh ≈ 1 → sinal harmônico puro (baixa entropia)
    Mistura proporcional: fração (1-nivel_coh) de ruído + fração nivel_coh de φ-sinal.
    """
    t = np.linspace(0, 4 * np.pi, dim)
    X = np.zeros((n, dim))
    for i in range(n):
        # Componente φ-estruturada
        freq_base = rng.uniform(0.5, 2.0)
        sinal_phi = np.zeros(dim)
        for k in range(5):
            freq_k = freq_base * (PHI ** k)
            amp_k  = rng.uniform(0.3, 1.0) / (k + 1)
            fase_k = rng.uniform(0, 2 * np.pi)
            sinal_phi += amp_k * np.sin(freq_k * t + fase_k)
        # Componente ruído
        ruido = rng.normal(0, 1.0, dim)
        # Mistura controlada
        sinal = nivel_coh * sinal_phi + (1 - nivel_coh) * ruido
        X[i] = sinal / (np.std(sinal) + 1e-8)
    return X

def gerar_dados(seed, nivel_coh):
    rng = np.random.default_rng(seed)
    n_tr, n_te = N_TRAIN // 2, N_TEST // 2
    # Classe 1: batch coerente no nível solicitado
    # Classe 0: ruído puro (coerência ≈ 0)
    X1_tr = gerar_batch_coerente(n_tr, DIM, nivel_coh, rng)
    X0_tr = gerar_batch_coerente(n_tr, DIM, 0.0, rng)
    X_tr  = np.vstack([X1_tr, X0_tr])
    y_tr  = np.array([1]*n_tr + [0]*n_tr, dtype=float)
    X1_te = gerar_batch_coerente(n_te, DIM, nivel_coh, rng)
    X0_te = gerar_batch_coerente(n_te, DIM, 0.0, rng)
    X_te  = np.vstack([X1_te, X0_te])
    y_te  = np.array([1]*n_te + [0]*n_te, dtype=float)
    idx_tr = rng.permutation(N_TRAIN)
    idx_te = rng.permutation(N_TEST)
    return X_tr[idx_tr], y_tr[idx_tr], X_te[idx_te], y_te[idx_te]

# ── Medir coerência real do batch ─────────────────────────────────────────────

def medir_coerencia(X):
    freq   = np.fft.fft(X, axis=-1)
    e      = np.abs(freq)
    e_norm = np.clip(e / (e.sum(axis=-1, keepdims=True) + 1e-8), 1e-10, 1.0)
    entropia  = -np.sum(e_norm * np.log(e_norm), axis=-1)
    return float(np.mean(1.0 - entropia / np.log(X.shape[-1])))

# ── Eco com k fixo ────────────────────────────────────────────────────────────

def eco_fixo(X, k, n_eco=N_ECO):
    X = np.asarray(X, dtype=float)
    s = X.copy()
    for _ in range(n_eco):
        freq      = np.fft.fft(s, axis=-1)
        nova_fase = np.angle(freq) * k
        reflexao  = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * nova_fase), axis=-1))
        s = s + (reflexao - X) / k
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

# ── Execução: paisagem completa ───────────────────────────────────────────────

paisagem = {}   # paisagem[nivel_coh][k_nome] = acc_media
coh_reais = {}  # coerência medida real por nível

print(f"{'Nível coh':>10} {'coh_real':>10} ", end="")
for kn in K_NOMES:
    print(f"{kn:>8}", end="")
print(f"  {'k_ótimo':>8}")
print("-" * (22 + 8 * len(K_VALS) + 10))

for nivel_coh in NIVEIS_COH:
    paisagem[nivel_coh] = {}
    accs_por_k = []
    coh_medidas = []

    for ki, k in enumerate(K_VALS):
        accs = []
        for seed in SEEDS:
            X_tr, y_tr, X_te, y_te = gerar_dados(seed, nivel_coh)
            if ki == 0:
                coh_medidas.append(medir_coerencia(X_tr))
            Xtr_eco = eco_fixo(X_tr, k)
            Xte_eco = eco_fixo(X_te, k)
            acc = treinar(Xtr_eco, y_tr, Xte_eco, y_te, seed)
            accs.append(acc)
        media = float(np.mean(accs))
        paisagem[nivel_coh][K_NOMES[ki]] = media
        accs_por_k.append(media)

    coh_real = float(np.mean(coh_medidas))
    coh_reais[nivel_coh] = coh_real
    k_otimo_idx = int(np.argmax(accs_por_k))
    k_otimo_nome = K_NOMES[k_otimo_idx]

    print(f"{nivel_coh:>10.2f} {coh_real:>10.4f} ", end="")
    for acc in accs_por_k:
        print(f"{acc:>8.3f}", end="")
    print(f"  {k_otimo_nome:>8}")

# ── Resumo: k_ótimo por nível de coerência ────────────────────────────────────

print("\n── Resumo: k_ótimo × coerência ─────────────────────────────────────────")
print(f"{'nivel_coh':>10} {'coh_real':>10} {'k_ótimo':>10} {'k_val':>10} {'acc':>8}")
print("-" * 52)

resumo = []
for nivel_coh in NIVEIS_COH:
    accs = [paisagem[nivel_coh][kn] for kn in K_NOMES]
    k_idx = int(np.argmax(accs))
    resumo.append({
        "nivel_coh": nivel_coh,
        "coh_real": coh_reais[nivel_coh],
        "k_otimo_nome": K_NOMES[k_idx],
        "k_otimo_val": float(K_VALS[k_idx]),
        "acc_otima": float(accs[k_idx]),
    })
    print(f"{nivel_coh:>10.2f} {coh_reais[nivel_coh]:>10.4f} "
          f"{K_NOMES[k_idx]:>10} {K_VALS[k_idx]:>10.4f} {accs[k_idx]:>8.3f}")

# Verifica se k_ótimo cresce monotonicamente com coerência
k_otimos_vals = [r["k_otimo_val"] for r in resumo]
monotonica = all(k_otimos_vals[i] <= k_otimos_vals[i+1]
                 for i in range(len(k_otimos_vals)-1))
print(f"\nProgressão de k_ótimo monotônica: {'✅ Sim' if monotonica else '⚠️  Não'}")
print(f"k_ótimos: {[r['k_otimo_nome'] for r in resumo]}")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "K_Paisagem",
    "pergunta": "a relação k_ótimo × coerência é contínua ou discreta?",
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "n_train": N_TRAIN, "n_test": N_TEST,
    "dim": DIM, "hidden": HIDDEN, "n_eco": N_ECO,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "niveis_coerencia": NIVEIS_COH,
    "k_vals": K_VALS,
    "k_nomes": K_NOMES,
    "coerencias_reais": coh_reais,
    "paisagem": {str(c): paisagem[c] for c in NIVEIS_COH},
    "resumo": resumo,
    "k_otimo_monotonica": monotonica,
}

with open("k_paisagem_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print("\nResultados salvos: k_paisagem_results.json")

print("\n── Interpretação ───────────────────────────────────────────────────────")
if monotonica:
    print("  Progressão monotônica confirmada.")
    print("  eco_fononico (interpolação contínua) é o modelo correto.")
    print("  Não há evidência de oitavas discretas na paisagem k × coerência.")
else:
    print("  Progressão não-monotônica detectada.")
    print("  Pode existir estrutura discreta — investigar platôs ou saltos.")
    for i in range(len(resumo)-1):
        if resumo[i]["k_otimo_val"] > resumo[i+1]["k_otimo_val"]:
            print(f"  Inversão: coh={resumo[i]['nivel_coh']} → k={resumo[i]['k_otimo_nome']}, "
                  f"coh={resumo[i+1]['nivel_coh']} → k={resumo[i+1]['k_otimo_nome']}")
