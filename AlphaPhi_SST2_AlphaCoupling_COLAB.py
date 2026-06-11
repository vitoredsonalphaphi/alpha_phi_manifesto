# ALPHA PHI — α como Constante de Acoplamento Geométrico
# Vitor Edson Delavi · Florianópolis · 2026
#
# REFORMULAÇÃO:
#   Experimento anterior: α como perturbação aditiva no output → nulo
#   Este experimento:     α como constante de acoplamento geométrico
#
# ANALOGIA COM QED:
#   Em QED, α não adiciona ao campo — governa COM QUE FORÇA campo e matéria
#   se acoplam. Aqui: α governa com que força a geometria φ (espaço hiperbólico)
#   acopla com o gradiente de aprendizado (CE).
#
# MECANISMO:
#   δ = (ŷ - y) · exp(-|norm(h) - φ|² / (α · φ²))
#
#   onde h = última camada oculta (representação no espaço hiperbólico)
#         norm(h) = norma da representação
#         φ = raio natural do espaço (1/√c = φ)
#         α = largura do acoplamento
#
#   Amostras com norm(h) ≈ φ  →  fator ≈ 1  →  gradiente completo
#   Amostras com norm(h) ≠ φ  →  fator < 1  →  gradiente reduzido
#   → o modelo aprende a organizar representações no atrator φ
#     antes de corrigir o erro de classificação
#
# ABLATION — varia apenas a largura de acoplamento:
#   G  : sem acoplamento (CE pura — baseline)
#   H  : largura = α = 1/137  (hipótese Alpha-Phi)
#   I  : largura = 0.010
#   J  : largura = 0.050
#   K  : largura = 0.001
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
PHI_R = 1.0 / np.sqrt(C_PHI)   # raio natural = φ

print(f"φ           = {PHI:.10f}")
print(f"α = 1/137   = {ALPHA:.10f}")
print(f"c = 1/φ²    = {C_PHI:.10f}")
print(f"raio φ      = {PHI_R:.10f}  (1/√c = φ)")
print(f"largura α·φ²= {ALPHA * PHI_R**2:.8f}  (bandwidth do acoplamento)")
print("=" * 65)
print("ALPHA PHI — α como Constante de Acoplamento Geométrico")
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

# ── Acoplamento Geométrico φ ──────────────────────────────────────────────────
def phi_geometric_coupling(h_hidden, bandwidth):
    """
    Fator de acoplamento geométrico.

    Gaussiana centrada no raio φ, largura = bandwidth · φ².
    α é a largura — quanto menor, mais estrita a exigência geométrica.

    Em QED: α aparece no expoente do vértice de interação.
    Aqui:   α aparece no expoente do kernel de acoplamento geométrico.

    coupling = exp( -|norm(h) - φ|² / (bandwidth · φ²) )

    Retorna shape (m, 1) para multiplicar o delta diretamente.
    """
    norms     = np.linalg.norm(h_hidden, axis=-1)        # (m,)
    deviation = (norms - PHI_R) ** 2                      # desvio quadrático do raio φ
    coupling  = np.exp(-deviation / (bandwidth * PHI_R**2 + 1e-10))
    return coupling.reshape(-1, 1)                         # (m, 1)

def coupling_stats(h_hidden, bandwidth):
    """Estatísticas do acoplamento para monitoramento."""
    c = phi_geometric_coupling(h_hidden, bandwidth).flatten()
    return float(c.mean()), float(c.std())

# ── Rede Alpha-Phi com Acoplamento Geométrico ─────────────────────────────────
class RedeGeometricCoupling:
    """
    Arquitetura: Fibonacci + expmap0(φ·tanh) + c=1/φ²
    Backward:    δ = (ŷ - y) · coupling(h_last_hidden, bandwidth)

    O acoplamento geométrico modula o gradiente CE
    pela posição da representação no espaço hiperbólico.
    bandwidth = 0 → CE pura (sem acoplamento)
    bandwidth = α → acoplamento com largura 1/137
    """
    def __init__(self, arch, seed, bandwidth=0.0):
        np.random.seed(seed)
        self.arch      = arch
        self.bandwidth = bandwidth
        self.W, self.b = [], []
        for i in range(len(arch) - 1):
            s = np.sqrt(2.0 / arch[i])   # He init
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))
        self.coupling_hist = []

    def _activation(self, z, last=False):
        if last:
            return sigmoid(z)
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

        # Gradiente base: CE
        delta = yhat - y.reshape(-1, 1)

        # Acoplamento geométrico: modula pelo raio φ da última camada oculta
        if self.bandwidth > 0:
            h_hidden = self.acts[-2]   # última camada oculta (antes do output)
            coupling = phi_geometric_coupling(h_hidden, self.bandwidth)
            delta    = delta * coupling
            # monitora acoplamento médio
            self.coupling_hist.append(float(coupling.mean()))

        for i in reversed(range(len(self.W))):
            self.W[i] -= lr * (self.acts[i].T @ delta / m)
            self.b[i]  -= lr * delta.mean(0)
            if i > 0:
                delta = delta @ self.W[i].T

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def coupling_medio(self):
        if not self.coupling_hist:
            return 1.0
        return float(np.mean(self.coupling_hist[-100:]))

# ── Dados SST-2 ───────────────────────────────────────────────────────────────
print("\nCarregando SST-2 e embeddings MiniLM...")
dataset    = load_dataset("nyu-mll/glue", "sst2")
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

# ── Configurações ─────────────────────────────────────────────────────────────
CONFIGS = {
    "G — sem acoplamento (baseline)": 0.0,
    "H — bw=α=1/137   (AlphaPhi)":   ALPHA,
    "I — bw=0.010     (controle)":    0.010,
    "J — bw=0.050     (controle)":    0.050,
    "K — bw=0.001     (controle)":    0.001,
}

# ── Seeds por timestamp ───────────────────────────────────────────────────────
N_SEEDS   = 10
EPOCHS    = 20
LR        = 0.1
BATCH     = 128
arch_in   = [X_train.shape[1]] + FIB + [1]

base_seed = int(time.time())
seeds     = [base_seed + i * 7 for i in range(N_SEEDS)]
print(f"\nSeeds ({N_SEEDS}): {seeds[:3]}...{seeds[-1]}")
print(f"Arquitetura: {arch_in}")
print(f"\nBandwidth α·φ² = {ALPHA * PHI_R**2:.6f}  (raio de acoplamento em norma²)")

# ── Treinamento ───────────────────────────────────────────────────────────────
resultados  = {cfg: [] for cfg in CONFIGS}
historico   = {cfg: [] for cfg in CONFIGS}
acoplamentos = {cfg: [] for cfg in CONFIGS}

print("\n" + "=" * 65)
for cfg_nome, bw in CONFIGS.items():
    print(f"\n{cfg_nome}  (bw={bw:.6f})")
    accs_seed = []
    hist_seed = []
    coup_seed = []
    for seed in seeds:
        rede = RedeGeometricCoupling(arch_in, seed, bandwidth=bw)
        hist = []
        for ep in range(EPOCHS):
            idx = np.random.permutation(len(X_train))
            for b in range(0, len(X_train), BATCH):
                xb = X_train[idx[b:b+BATCH]]
                yb = y_train[idx[b:b+BATCH]]
                rede.forward(xb)
                rede.backward(xb, yb, LR)
            if ep % 5 == 4:
                hist.append(rede.accuracy(X_val, y_val))
        acc_final = rede.accuracy(X_val, y_val)
        accs_seed.append(acc_final)
        hist_seed.append(hist)
        coup_seed.append(rede.coupling_medio())
        print(f"  seed {seed} → {acc_final:.4f}  (acoplamento médio: {rede.coupling_medio():.4f})")
    resultados[cfg_nome]   = accs_seed
    historico[cfg_nome]    = np.mean(hist_seed, axis=0)
    acoplamentos[cfg_nome] = np.mean(coup_seed)
    print(f"  Média: {np.mean(accs_seed):.4f} ± {np.std(accs_seed):.4f}")

# ── Análise Estatística ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTADOS — α como Constante de Acoplamento Geométrico")
print("=" * 65)

baseline = np.array(resultados["G — sem acoplamento (baseline)"])
print(f"\n{'Config':<38} {'Média':>7} {'Std':>7} {'Δ vs G':>8} {'p-valor':>10} {'Acopl.':>8}")
print("-" * 90)

for cfg_nome, accs in resultados.items():
    arr  = np.array(accs)
    coup = acoplamentos[cfg_nome]
    if cfg_nome == "G — sem acoplamento (baseline)":
        print(f"{cfg_nome:<38} {arr.mean():>7.4f} {arr.std():>7.4f} {'—':>8} {'—':>10} {'—':>8}")
    else:
        t, p = stats.ttest_ind(arr, baseline)
        sig = "✓" if p < 0.05 else "ns"
        print(f"{cfg_nome:<38} {arr.mean():>7.4f} {arr.std():>7.4f} "
              f"{arr.mean()-baseline.mean():>+8.4f} {p:>10.4f} {coup:>8.4f} {sig}")

print("\nH (α=1/137) vs controles:")
h_vals = np.array(resultados["H — bw=α=1/137   (AlphaPhi)"])
for cfg_nome, accs in resultados.items():
    if "baseline" in cfg_nome or "AlphaPhi" in cfg_nome:
        continue
    arr = np.array(accs)
    t, p = stats.ttest_ind(h_vals, arr)
    delta = h_vals.mean() - arr.mean()
    sig = "✓ α é único" if p < 0.05 else "ns — largura não importa"
    print(f"  H vs {cfg_nome:<32} Δ={delta:+.4f}  p={p:.4f}  {sig}")

# ── Diagnóstico de acoplamento ────────────────────────────────────────────────
print("\n" + "─" * 65)
print("Diagnóstico — acoplamento médio por configuração:")
print("(1.0 = gradiente completo | <1.0 = representações fora do atrator φ)")
for cfg_nome, coup in acoplamentos.items():
    bw = CONFIGS[cfg_nome]
    bw_str = f"bw={bw:.4f}" if bw > 0 else "sem acopl."
    print(f"  {cfg_nome:<38} coupling={coup:.6f}")

# ── Visualização ──────────────────────────────────────────────────────────────
CORES = {
    "G — sem acoplamento (baseline)": "#8B949E",
    "H — bw=α=1/137   (AlphaPhi)":   "#00FF88",
    "I — bw=0.010     (controle)":    "#00BFFF",
    "J — bw=0.050     (controle)":    "#FF9944",
    "K — bw=0.001     (controle)":    "#FF4466",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("#0d1117")

# Gráfico 1 — Distribuição acurácias
ax = axes[0]
ax.set_facecolor("#161b22")
ax.set_title("Distribuição de Acurácias (10 seeds)\nAblation — largura de acoplamento",
             color="#DAA520", fontsize=10, fontweight='bold')
for pos, (cfg_nome, accs) in enumerate(resultados.items()):
    cor = CORES[cfg_nome]
    arr = np.array(accs)
    ax.boxplot(arr, positions=[pos], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=cor, alpha=0.7),
               medianprops=dict(color='white', linewidth=2),
               whiskerprops=dict(color=cor), capprops=dict(color=cor),
               flierprops=dict(marker='o', color=cor, alpha=0.5))
    ax.scatter([pos]*len(arr), arr, color=cor, alpha=0.6, s=20, zorder=5)
ax.set_xticks(range(5))
ax.set_xticklabels(["G\nbaseline","H\nα=1/137","I\n0.010","J\n0.050","K\n0.001"],
                   color="#8B949E", fontsize=9)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_ylabel("Acurácia (val)", color="#8B949E")
ax.grid(True, alpha=0.15)
ax.set_ylim(0.6, 0.9)
ax.axhline(baseline.mean(), color="#8B949E", linestyle='--', alpha=0.5)

# Gráfico 2 — Convergência
ax = axes[1]
ax.set_facecolor("#161b22")
ax.set_title("Convergência média (10 seeds)", color="#DAA520",
             fontsize=10, fontweight='bold')
epocas_plot = [5, 10, 15, 20]
for cfg_nome, hist in historico.items():
    lw = 2.5 if "AlphaPhi" in cfg_nome else 1.2
    ax.plot(epocas_plot, hist, color=CORES[cfg_nome], linewidth=lw,
            label=cfg_nome.split("(")[0].strip(), marker='o', markersize=4)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Época", color="#8B949E")
ax.set_ylabel("Acurácia (val)", color="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7, loc='lower right')
ax.grid(True, alpha=0.15)

# Gráfico 3 — Acoplamento médio por largura
ax = axes[2]
ax.set_facecolor("#161b22")
ax.set_title("Acoplamento médio vs largura\n(força do vínculo geométrico φ)",
             color="#DAA520", fontsize=10, fontweight='bold')
bws   = [CONFIGS[c] for c in CONFIGS if CONFIGS[c] > 0]
coups = [acoplamentos[c] for c in CONFIGS if CONFIGS[c] > 0]
cores = [CORES[c] for c in CORES if CONFIGS.get(c, 0) > 0]
ax.scatter(bws, coups, c=cores, s=120, zorder=5)
for bw, coup, cor, lbl in zip(bws, coups, cores,
                                ["H α=1/137", "I 0.010", "J 0.050", "K 0.001"]):
    ax.annotate(lbl, (bw, coup), textcoords="offset points",
                xytext=(5, 5), color=cor, fontsize=8)
ax.axvline(ALPHA, color="#00FF88", linestyle='--', alpha=0.7, label=f"α=1/137={ALPHA:.4f}")
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Largura de acoplamento (bandwidth)", color="#8B949E")
ax.set_ylabel("Coupling médio", color="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.grid(True, alpha=0.15)

fig.suptitle(
    f"ALPHA PHI — α como Acoplamento Geométrico\n"
    f"δ = (ŷ−y)·exp(−|h−φ|²/(bw·φ²))  |  φ={PHI:.4f}  |  α=1/{round(1/ALPHA)}  |  Florianópolis 2026",
    color="#DAA520", fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig("alphaphi_alpha_coupling.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.show()

print("\nGráfico salvo: alphaphi_alpha_coupling.png")
print("\nO que observar:")
print("  Acoplamento médio → quanto as representações estão no atrator φ")
print("  Se H > G com p<0.05 → acoplamento geométrico valida α")
print("  Se H ≠ I,J,K com p<0.05 → α=1/137 é a largura específica que importa")
print("\nalpha-phi")
