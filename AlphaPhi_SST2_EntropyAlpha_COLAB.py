# ALPHA PHI — Validação de α como Fator de Entropia
# Vitor Edson Delavi · Florianópolis · 2026
#
# PERGUNTA CENTRAL:
#   α = 1/137 como escala da penalidade de entropia produz resultado
#   distinto de outros valores próximos? Ou qualquer número pequeno funciona?
#
# FUNÇÃO DE PERDA:
#   L = CE + λ · H(φ)
#   onde H(φ) = φ · H_shannon_normalizada
#
# ABLATION — varia apenas λ:
#   G  : λ = 0          (CE pura — baseline)
#   H  : λ = α = 1/137  (hipótese Alpha-Phi)
#   I  : λ = 0.010      (comparativo próximo)
#   J  : λ = 0.050      (comparativo maior)
#   K  : λ = 0.001      (comparativo menor)
#
# Arquitetura fixa (melhor resultado do ablation anterior):
#   Fibonacci + φ·tanh + curvatura hiperbólica c=1/φ²
#
# Protocolo de integridade:
#   Seeds por timestamp — nenhuma escolhida manualmente
#   Resultados reportados na íntegra

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers", "datasets", "scipy"], check=True)

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ── Constantes ────────────────────────────────────────────────────────────────
PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084
C_PHI = 1.0 / PHI**2

print(f"φ         = {PHI:.10f}")
print(f"α = 1/137 = {ALPHA:.10f}")
print(f"c = 1/φ²  = {C_PHI:.10f}")
print("=" * 65)
print("ALPHA PHI — Validação de α como Fator de Entropia Shannon")
print("=" * 65)

# ── Espaço Hiperbólico ────────────────────────────────────────────────────────
def expmap0(v, c=C_PHI):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return np.tanh(np.sqrt(c) * norm) * v / (np.sqrt(c) * norm)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def fibonacci_layers(n, start=55):
    layers = [start]
    a, b = start, int(round(start * PHI))
    for _ in range(n - 1):
        layers.append(b); a, b = b, a + b
    return layers

FIB = fibonacci_layers(3)   # [55, 89, 144]
print(f"Camadas Fibonacci: {FIB}")

# ── Entropia φ de Shannon ─────────────────────────────────────────────────────
def phi_entropy_gradient(yhat):
    """
    Gradiente de α·H(φ) em relação à pré-ativação z (sigmoid).

    H(φ) = φ · H_shannon_normalizada
    H_shannon = -(ŷ·log(ŷ) + (1-ŷ)·log(1-ŷ))
    H_norm    = H_shannon / log(2)      ← normaliza para [0,1]

    ∂H(φ)/∂z = φ/log(2) · log((1-ŷ)/ŷ) · ŷ·(1-ŷ)

    Interpretação:
      Outputs próximos de 0.5 (incoerentes) têm H alta → penalizados.
      Outputs próximos de 0 ou 1 (coerentes) têm H baixa → favorecidos.
      Escalar por α garante que a correção seja perturbação mínima —
      análogo ao papel de α em QED como parâmetro de expansão perturbativa.
    """
    eps = 1e-10
    y  = np.clip(yhat, eps, 1.0 - eps)
    # gradiente de H_phi em relação à pré-ativação z
    grad_H = (PHI / np.log(2)) * np.log((1.0 - y) / y) * y * (1.0 - y)
    return grad_H

def phi_entropy_value(yhat):
    """Valor escalar de H(φ) — para monitoramento."""
    eps = 1e-10
    y   = np.clip(yhat, eps, 1.0 - eps)
    H   = -(y * np.log(y) + (1.0 - y) * np.log(1.0 - y))
    return float((PHI * H / np.log(2)).mean())

# ── Rede Alpha-Phi com Penalidade de Entropia ─────────────────────────────────
class RedeEntropyAlpha:
    """
    Arquitetura: Fibonacci + φ·tanh (camadas ocultas) + curvatura c=1/φ²
    Perda:       L = CE + λ · H(φ)
    λ é o único parâmetro que varia entre as configs G, H, I, J, K.
    """
    def __init__(self, arch, seed, lambda_entropy=0.0):
        np.random.seed(seed)
        self.arch   = arch
        self.lambda_e = lambda_entropy
        self.W, self.b = [], []
        for i in range(len(arch) - 1):
            s = np.sqrt(2.0 / arch[i])   # He init
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))

    def _activation(self, z, last=False):
        if last:
            return sigmoid(z)
        # Ativação hiperbólica: expmap0 com curvatura c=1/φ²
        return expmap0(PHI * np.tanh(z / PHI))

    def forward(self, X):
        self.acts = [X]
        cur = X
        for i in range(len(self.W)):
            z   = cur @ self.W[i] + self.b[i]
            cur = self._activation(z, last=(i == len(self.W) - 1))
            self.acts.append(cur)
        return cur

    def backward(self, X, y, lr):
        m    = X.shape[0]
        yhat = self.acts[-1]

        # Gradiente da CE: δ = ŷ - y  (simplificação sigmoid + BCE)
        delta = yhat - y.reshape(-1, 1)

        # Gradiente da penalidade de entropia φ (se λ > 0)
        if self.lambda_e > 0:
            delta = delta + self.lambda_e * phi_entropy_gradient(yhat)

        for i in reversed(range(len(self.W))):
            self.W[i] -= lr * (self.acts[i].T @ delta / m)
            self.b[i]  -= lr * delta.mean(0)
            if i > 0:
                delta = delta @ self.W[i].T

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

# ── Dados SST-2 ───────────────────────────────────────────────────────────────
print("\nCarregando SST-2 e embeddings MiniLM...")
dataset = load_dataset("nyu-mll/glue", "sst2")
modelo_emb = SentenceTransformer("all-MiniLM-L6-v2")

train_data = dataset["train"].shuffle(seed=42).select(range(5000))
val_data   = dataset["validation"]

X_train = modelo_emb.encode(train_data["sentence"], batch_size=256,
                             show_progress_bar=False)
y_train = np.array(train_data["label"])
X_val   = modelo_emb.encode(val_data["sentence"], batch_size=256,
                             show_progress_bar=False)
y_val   = np.array(val_data["label"])

print(f"Treino: {X_train.shape}  Val: {X_val.shape}")

# ── Configurações do Ablation ─────────────────────────────────────────────────
CONFIGS = {
    "G — CE pura (baseline)":     0.0,
    "H — λ=α=1/137  (AlphaPhi)":  ALPHA,
    "I — λ=0.010    (controle)":   0.010,
    "J — λ=0.050    (controle)":   0.050,
    "K — λ=0.001    (controle)":   0.001,
}

# ── Seeds por timestamp ───────────────────────────────────────────────────────
N_SEEDS  = 10
EPOCHS   = 20
LR       = 0.1
BATCH    = 128
arch_in  = [X_train.shape[1]] + FIB + [1]   # 384 → 55 → 89 → 144 → 1

base_seed = int(time.time())
seeds = [base_seed + i * 7 for i in range(N_SEEDS)]
print(f"\nSeeds ({N_SEEDS}): {seeds[:3]}...{seeds[-1]}")
print(f"Arquitetura: {arch_in}")

# ── Treinamento ───────────────────────────────────────────────────────────────
resultados = {cfg: [] for cfg in CONFIGS}
historico  = {cfg: [] for cfg in CONFIGS}

print("\n" + "=" * 65)
for cfg_nome, lambda_val in CONFIGS.items():
    print(f"\n{cfg_nome}  (λ={lambda_val:.6f})")
    accs_seed = []
    hist_seed = []
    for seed in seeds:
        rede = RedeEntropyAlpha(arch_in, seed, lambda_entropy=lambda_val)
        hist = []
        for ep in range(EPOCHS):
            idx = np.random.permutation(len(X_train))
            for b in range(0, len(X_train), BATCH):
                xb = X_train[idx[b:b+BATCH]]
                yb = y_train[idx[b:b+BATCH]]
                rede.forward(xb)
                rede.backward(xb, yb, LR)
            if ep % 5 == 4:
                acc = rede.accuracy(X_val, y_val)
                hist.append(acc)
        acc_final = rede.accuracy(X_val, y_val)
        accs_seed.append(acc_final)
        hist_seed.append(hist)
        print(f"  seed {seed} → {acc_final:.4f}")
    resultados[cfg_nome] = accs_seed
    historico[cfg_nome]  = np.mean(hist_seed, axis=0)
    print(f"  Média: {np.mean(accs_seed):.4f} ± {np.std(accs_seed):.4f}")

# ── Análise Estatística ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTADOS — Ablation λ (escala da penalidade de entropia)")
print("=" * 65)

baseline = np.array(resultados["G — CE pura (baseline)"])
print(f"\n{'Config':<35} {'Média':>7} {'Std':>7} {'Δ vs G':>8} {'p-valor':>10} {'Conclusão'}")
print("-" * 85)

for cfg_nome, accs in resultados.items():
    arr  = np.array(accs)
    mean = arr.mean()
    std  = arr.std()
    delta = mean - baseline.mean()
    if cfg_nome == "G — CE pura (baseline)":
        print(f"{cfg_nome:<35} {mean:>7.4f} {std:>7.4f} {'—':>8} {'—':>10} baseline")
    else:
        t, p = stats.ttest_ind(arr, baseline)
        sig = "✓ significativo" if p < 0.05 else "ns"
        print(f"{cfg_nome:<35} {mean:>7.4f} {std:>7.4f} {delta:>+8.4f} {p:>10.4f} {sig}")

# Comparação direta H vs I, J, K
print("\n" + "─" * 65)
print("Comparação H (α=1/137) vs controles:")
h_vals = np.array(resultados["H — λ=α=1/137  (AlphaPhi)"])
for cfg_nome, accs in resultados.items():
    if cfg_nome in ["G — CE pura (baseline)", "H — λ=α=1/137  (AlphaPhi)"]:
        continue
    arr = np.array(accs)
    t, p = stats.ttest_ind(h_vals, arr)
    delta = h_vals.mean() - arr.mean()
    sig = "✓ α é único" if p < 0.05 else "ns — valor específico não importa"
    print(f"  H vs {cfg_nome:<30} Δ={delta:+.4f}  p={p:.4f}  {sig}")

# ── Entropia média final por config ──────────────────────────────────────────
print("\n" + "─" * 65)
print("Entropia H(φ) média nas predições finais (val set):")
for cfg_nome, lambda_val in CONFIGS.items():
    rede_test = RedeEntropyAlpha(arch_in, seeds[0], lambda_entropy=lambda_val)
    # treina 1 seed rápido
    for ep in range(EPOCHS):
        idx = np.random.permutation(len(X_train))
        for b in range(0, len(X_train), BATCH):
            xb = X_train[idx[b:b+BATCH]]
            yb = y_train[idx[b:b+BATCH]]
            rede_test.forward(xb)
            rede_test.backward(xb, yb, LR)
    yhat_val = rede_test.forward(X_val)
    H_val = phi_entropy_value(yhat_val)
    print(f"  {cfg_nome:<35} H(φ)={H_val:.6f}")

# ── Visualização ──────────────────────────────────────────────────────────────
CORES = {
    "G — CE pura (baseline)":     "#8B949E",
    "H — λ=α=1/137  (AlphaPhi)":  "#00FF88",
    "I — λ=0.010    (controle)":   "#00BFFF",
    "J — λ=0.050    (controle)":   "#FF9944",
    "K — λ=0.001    (controle)":   "#FF4466",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#0d1117")
epochs_check = [5, 10, 15, 20]

# Gráfico 1 — Distribuição de acurácias por config
ax = axes[0]
ax.set_facecolor("#161b22")
ax.set_title("Distribuição de Acurácias (10 seeds)\nAblation λ — penalidade de entropia φ",
             color="#DAA520", fontsize=11, fontweight='bold')
posicoes = list(range(len(CONFIGS)))
for pos, (cfg_nome, accs) in zip(posicoes, resultados.items()):
    cor = CORES[cfg_nome]
    arr = np.array(accs)
    ax.boxplot(arr, positions=[pos], widths=0.5,
               patch_artist=True,
               boxprops=dict(facecolor=cor, alpha=0.7),
               medianprops=dict(color='white', linewidth=2),
               whiskerprops=dict(color=cor),
               capprops=dict(color=cor),
               flierprops=dict(marker='o', color=cor, alpha=0.5))
    ax.scatter([pos]*len(arr), arr, color=cor, alpha=0.6, s=20, zorder=5)

labels_curtos = ["G\nbaseline", "H\nα=1/137", "I\n0.010", "J\n0.050", "K\n0.001"]
ax.set_xticks(posicoes)
ax.set_xticklabels(labels_curtos, color="#8B949E", fontsize=9)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_ylabel("Acurácia (val)", color="#8B949E")
ax.grid(True, alpha=0.15)
ax.set_ylim(0.6, 0.9)

# Linha de baseline
ax.axhline(np.mean(resultados["G — CE pura (baseline)"]),
           color="#8B949E", linestyle='--', alpha=0.5, label="baseline")

# Gráfico 2 — Curvas de convergência
ax = axes[1]
ax.set_facecolor("#161b22")
ax.set_title("Convergência média (10 seeds)\nAcurácia por época",
             color="#DAA520", fontsize=11, fontweight='bold')

epocas_plot = [5, 10, 15, 20]
for cfg_nome, hist in historico.items():
    cor = CORES[cfg_nome]
    lw  = 2.5 if "AlphaPhi" in cfg_nome else 1.2
    ax.plot(epocas_plot, hist, color=cor, linewidth=lw,
            label=cfg_nome.split("(")[0].strip(),
            marker='o', markersize=4)

ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Época", color="#8B949E")
ax.set_ylabel("Acurácia (val)", color="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8,
          loc='lower right')
ax.grid(True, alpha=0.15)

fig.suptitle(
    f"ALPHA PHI — Validação de α=1/137 como Escala de Entropia Shannon\n"
    f"L = CE + λ·H(φ)   |   φ={PHI:.4f}   |   α=1/{round(1/ALPHA)}   |   Florianópolis 2026",
    color="#DAA520", fontsize=11, fontweight='bold'
)

plt.tight_layout()
plt.savefig("alphaphi_entropy_alpha_validation.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.show()

print("\nGráfico salvo: alphaphi_entropy_alpha_validation.png")
print("\nO que o experimento responde:")
print("  H vs G: a penalidade de entropia φ ajuda? (qualquer λ)")
print("  H vs I/J/K: α=1/137 especificamente importa, ou qualquer λ pequeno funciona?")
print("\nSe H supera I, J e K significativamente → α validado como constante.")
print("Se H ≈ I ≈ J ≈ K → a abordagem funciona, mas o valor específico não é o que importa.")
print("\nalpha-phi")
