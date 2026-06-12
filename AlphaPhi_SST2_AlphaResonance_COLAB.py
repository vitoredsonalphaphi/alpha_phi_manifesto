# ALPHA PHI — α como Equilíbrio Ressonante Interno (Eco Ressonante)
# Vitor Edson Delavi · Florianópolis · 2026
#
# DIAGNÓSTICO CUMULATIVO DOS CINCO EXPERIMENTOS:
#   Em todos os casos, α foi INJETADO no sistema como fator externo.
#   Resultado consistente: efeito abaixo do limiar detectável (p≈1.0).
#
# OBSERVAÇÃO DOS DADOS (emergência, não imposição):
#   norm_h_natural ≈ 1.567 em todos os experimentos
#   gap = φ − norm_h = 1.618 − 1.567 = 0.051 ≈ 7α
#   A casca natural da rede orbita φ em exatamente ~7 unidades de α.
#
#   E_φ  = média(|norm(h) − φ|²) ≈ 0.0026   (energia geométrica)
#   E_CE = perda CE               ≈ 0.55      (energia do erro)
#   R = E_φ / E_CE ≈ 0.005 ≈ 0.7α             (razão natural ≈ α)
#
# NOVO PRINCÍPIO — ECO RESSONANTE INTERNO:
#   Em vez de INJETAR α, criar um ATRATOR para α:
#   o sistema mede R continuamente e é atraído para o equilíbrio R = α.
#
# MECANISMO (Exp 6):
#   R(t) = E_φ(t) / E_CE(t)                        ← razão medida por batch
#   R_ema(t) = α·R(t) + (1-α)·R_ema(t-1)           ← memória EMA (τ=1/α)
#   erro = R_ema − target                            ← desvio do equilíbrio
#   gain = 1/α = 137                                 ← escala natural de amplificação
#   lr_eff = lr · clip(1 − gain·erro, 0.3, 3.0)     ← realimentação proporcional
#
#   R_ema < target → erro negativo → lr_eff > lr → CE cai mais rápido → R sobe
#   R_ema > target → erro positivo → lr_eff < lr → CE cai mais devagar → R desce
#   R_ema = target → equilíbrio → lr_eff = lr
#
# ABLATION — varia o TARGET do equilíbrio:
#   AA : sem realimentação (baseline — só mede R)
#   BB : target = α = 1/137     (hipótese Alpha-Phi)
#   CC : target = 2α = 2/137    (controle — dobro)
#   DD : target = α/2 = 0.5/137 (controle — metade)
#   EE : target = 0.010         (controle — escala diferente)
#
# Diagnóstico principal: trajetória de R_ema por época.
# Hipótese: target=α produz convergência mais estável porque
#           α é o equilíbrio natural do sistema (R_natural ≈ α).
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
PHI_R = 1.0 / np.sqrt(C_PHI)
GAIN  = 1.0 / ALPHA   # = 137 — escala natural de amplificação

print(f"φ             = {PHI:.10f}")
print(f"α = 1/137     = {ALPHA:.10f}")
print(f"raio φ        = {PHI_R:.10f}")
print(f"gain = 1/α    = {GAIN:.2f}")
print(f"R_natural ≈   {0.051**2 / 0.55:.5f}  (≈ {(0.051**2/0.55)/ALPHA:.2f}α)")
print(f"gap natural ≈ {0.051:.4f}  ≈ {0.051/ALPHA:.1f}α")
print("=" * 65)
print("ALPHA PHI — Exp 6: α como Equilíbrio Ressonante (Eco Interno)")
print("R = E_φ/E_CE → α  |  gain=1/α=137  |  EMA τ=1/α")
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

FIB = fibonacci_layers(3)
print(f"Camadas Fibonacci: {FIB}")

# ── Razão Ressonante R = E_φ / E_CE ──────────────────────────────────────────
def compute_R(h_hidden, y, yhat):
    """
    R = E_φ / E_CE

    E_φ  = energia geométrica = média(|norm(h) − φ|²)
    E_CE = cross-entropy binária do batch

    R ≈ α quando o sistema está no equilíbrio ressonante natural.
    """
    eps = 1e-10
    # Energia geométrica
    norms = np.linalg.norm(h_hidden, axis=-1)
    E_phi = float(np.mean((norms - PHI_R) ** 2))

    # Cross-entropy do batch
    y_     = y.reshape(-1)
    yhat_  = yhat.flatten()
    yhat_  = np.clip(yhat_, eps, 1 - eps)
    E_CE   = float(-np.mean(y_ * np.log(yhat_) + (1 - y_) * np.log(1 - yhat_)))

    R = E_phi / (E_CE + eps)
    return R, E_phi, E_CE

# ── Rede Alpha-Phi com Eco Ressonante Interno ─────────────────────────────────
class RedeRessonante:
    """
    Canal φ:  expmap0(φ·tanh) com c=1/φ²                (forward — geometria)
    Canal CE: δ = (ŷ-y)                                  (backward — intacto)
    Canal α:  R_ema → target via realimentação lr        (eco ressonante)

    gain = 1/α = 137: quando R desvia α de target, lr muda 13.7%.
    EMA usa α como coeficiente: τ = 1/α = 137 batches.
    """
    def __init__(self, arch, seed, target=0.0):
        np.random.seed(seed)
        self.arch   = arch
        self.target = target
        self.W, self.b = [], []
        for i in range(len(arch) - 1):
            s = np.sqrt(2.0 / arch[i])
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))
        self.R_ema    = ALPHA       # inicia no equilíbrio α
        self.R_hist   = []          # R_ema por batch
        self.lr_hist  = []
        self.E_phi_hist = []
        self.E_CE_hist  = []
        self.norm_hist  = []

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

    def backward(self, X, y, lr_base):
        m    = X.shape[0]
        yhat = self.acts[-1]

        # Canal CE — gradiente puro, intocado
        delta = yhat - y.reshape(-1, 1)

        # Canal α — eco ressonante interno
        h_hidden = self.acts[-2]
        norms    = np.linalg.norm(h_hidden, axis=-1)
        self.norm_hist.append(float(norms.mean()))

        R, E_phi, E_CE = compute_R(h_hidden, y, yhat)

        # EMA de R com α como coeficiente (τ = 1/α = 137 batches)
        self.R_ema = ALPHA * R + (1.0 - ALPHA) * self.R_ema
        self.R_hist.append(self.R_ema)
        self.E_phi_hist.append(E_phi)
        self.E_CE_hist.append(E_CE)

        # Realimentação proporcional: puxar R_ema → target
        if self.target > 0:
            erro  = self.R_ema - self.target
            lr_eff = lr_base * float(np.clip(1.0 - GAIN * erro, 0.3, 3.0))
        else:
            lr_eff = lr_base

        self.lr_hist.append(lr_eff)

        for i in reversed(range(len(self.W))):
            self.W[i] -= lr_eff * (self.acts[i].T @ delta / m)
            self.b[i]  -= lr_eff * delta.mean(0)
            if i > 0:
                delta = delta @ self.W[i].T

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def R_final(self):
        return float(self.R_ema)

    def R_medio(self):
        return float(np.mean(self.R_hist[-100:])) if self.R_hist else 0.0

    def lr_medio(self):
        return float(np.mean(self.lr_hist)) if self.lr_hist else 0.0

    def norm_media(self):
        return float(np.mean(self.norm_hist)) if self.norm_hist else 0.0

    def R_por_epoca(self, n_batches_por_epoca):
        if not self.R_hist:
            return [0.0] * 4
        resultado = []
        for ep in range(4):
            inicio = ep * n_batches_por_epoca * 5
            fim    = inicio + n_batches_por_epoca * 5
            trecho = self.R_hist[inicio:fim]
            resultado.append(float(np.mean(trecho)) if trecho else 0.0)
        return resultado

    def lr_por_epoca(self, n_batches_por_epoca):
        if not self.lr_hist:
            return [0.0] * 4
        resultado = []
        for ep in range(4):
            inicio = ep * n_batches_por_epoca * 5
            fim    = inicio + n_batches_por_epoca * 5
            trecho = self.lr_hist[inicio:fim]
            resultado.append(float(np.mean(trecho)) if trecho else 0.0)
        return resultado

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

n_batches = len(X_train) // 128
print(f"\nBatches por época: {n_batches}")
print(f"Previsão de R_natural ≈ {0.051**2/0.55:.5f}  (≈ {(0.051**2/0.55)/ALPHA:.2f}α)")
print(f"Se R ≈ 0.7α → erro = −0.3α → gain×erro = −137×0.3×0.007 = −{137*0.3*ALPHA:.3f}")
print(f"→ lr_eff ≈ lr × {1+137*0.3*ALPHA:.3f} (+{137*0.3*ALPHA*100:.1f}%)")

# ── Configurações ─────────────────────────────────────────────────────────────
CONFIGS = {
    "AA — sem realimentação (baseline)": 0.0,
    "BB — target=α=1/137   (AlphaPhi)":  ALPHA,
    "CC — target=2α=2/137  (controle)":  2 * ALPHA,
    "DD — target=α/2       (controle)":  ALPHA / 2,
    "EE — target=0.010     (controle)":  0.010,
}

N_SEEDS = 10
EPOCHS  = 20
LR      = 0.1
BATCH   = 128
arch_in = [X_train.shape[1]] + FIB + [1]

base_seed = int(time.time())
seeds     = [base_seed + i * 7 for i in range(N_SEEDS)]
print(f"\nSeeds ({N_SEEDS}): {seeds[:3]}...{seeds[-1]}")
print(f"Arquitetura: {arch_in}")

# ── Treinamento ───────────────────────────────────────────────────────────────
resultados  = {cfg: [] for cfg in CONFIGS}
historico   = {cfg: [] for cfg in CONFIGS}
R_finais    = {cfg: [] for cfg in CONFIGS}
lrs_medios  = {cfg: [] for cfg in CONFIGS}
normas      = {cfg: [] for cfg in CONFIGS}
traj_R      = {cfg: [] for cfg in CONFIGS}
traj_lr     = {cfg: [] for cfg in CONFIGS}

print("\n" + "=" * 65)
for cfg_nome, target in CONFIGS.items():
    target_str = f"target={target:.5f}" if target > 0 else "sem feedback"
    print(f"\n{cfg_nome}  ({target_str})")
    accs_seed, hist_seed, R_seed, lr_seed, norm_seed = [], [], [], [], []
    traj_R_seed, traj_lr_seed = [], []

    for seed in seeds:
        rede = RedeRessonante(arch_in, seed, target=target)
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
        R_seed.append(rede.R_final())
        lr_seed.append(rede.lr_medio())
        norm_seed.append(rede.norm_media())
        traj_R_seed.append(rede.R_por_epoca(n_batches))
        traj_lr_seed.append(rede.lr_por_epoca(n_batches))

        target_label = f"{target/ALPHA:.1f}α" if target > 0 else "—"
        print(f"  seed {seed} → acc={acc_final:.4f}  "
              f"R_ema={rede.R_final():.5f}  "
              f"lr={rede.lr_medio():.5f}  "
              f"norm={rede.norm_media():.4f}")

    resultados[cfg_nome] = accs_seed
    historico[cfg_nome]  = np.mean(hist_seed, axis=0)
    R_finais[cfg_nome]   = float(np.mean(R_seed))
    lrs_medios[cfg_nome] = float(np.mean(lr_seed))
    normas[cfg_nome]     = float(np.mean(norm_seed))
    traj_R[cfg_nome]     = np.mean(traj_R_seed, axis=0).tolist()
    traj_lr[cfg_nome]    = np.mean(traj_lr_seed, axis=0).tolist()
    print(f"  → Média: {np.mean(accs_seed):.4f} ± {np.std(accs_seed):.4f}  "
          f"R_ema={np.mean(R_seed):.5f}  lr={np.mean(lr_seed):.5f}")

# ── Análise Estatística ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTADOS — α como Equilíbrio Ressonante Interno")
print("R=E_φ/E_CE → target  |  gain=1/α=137  |  EMA τ=1/α batches")
print("=" * 65)

baseline = np.array(resultados["AA — sem realimentação (baseline)"])
print(f"\n{'Config':<38} {'Média':>7} {'Std':>7} {'Δ vs AA':>8} {'p-valor':>10} "
      f"{'R_ema':>8} {'R/α':>6} {'lr_med':>8}")
print("-" * 105)

for cfg_nome, accs in resultados.items():
    arr    = np.array(accs)
    R_f    = R_finais[cfg_nome]
    lr_m   = lrs_medios[cfg_nome]
    R_over_alpha = R_f / ALPHA
    if "baseline" in cfg_nome:
        print(f"{cfg_nome:<38} {arr.mean():>7.4f} {arr.std():>7.4f} "
              f"{'—':>8} {'—':>10} {R_f:>8.5f} {R_over_alpha:>6.2f} {'—':>8}")
    else:
        t, p = stats.ttest_ind(arr, baseline)
        sig  = "✓" if p < 0.05 else "ns"
        print(f"{cfg_nome:<38} {arr.mean():>7.4f} {arr.std():>7.4f} "
              f"{arr.mean()-baseline.mean():>+8.4f} {p:>10.4f} "
              f"{R_f:>8.5f} {R_over_alpha:>6.2f} {lr_m:>8.5f} {sig}")

print("\nBB (target=α) vs controles:")
bb_vals = np.array(resultados["BB — target=α=1/137   (AlphaPhi)"])
for cfg_nome, accs in resultados.items():
    if "baseline" in cfg_nome or "AlphaPhi" in cfg_nome:
        continue
    arr  = np.array(accs)
    t, p = stats.ttest_ind(bb_vals, arr)
    delta = bb_vals.mean() - arr.mean()
    sig   = "✓ target α específico" if p < 0.05 else "ns — target não importa"
    print(f"  BB vs {cfg_nome:<30} Δ={delta:+.4f}  p={p:.4f}  {sig}")

# ── Diagnóstico de equilíbrio ─────────────────────────────────────────────────
print("\n" + "─" * 65)
print("Diagnóstico — equilíbrio ressonante R_ema vs target:")
print(f"  R natural sem feedback ≈ {R_finais['AA — sem realimentação (baseline)']:.5f} "
      f"= {R_finais['AA — sem realimentação (baseline)']/ALPHA:.2f}α")
print()
for cfg_nome in CONFIGS:
    target = CONFIGS[cfg_nome]
    R_f    = R_finais[cfg_nome]
    lr_m   = lrs_medios[cfg_nome]
    if target > 0:
        convergiu = abs(R_f - target) / target * 100
        print(f"  {cfg_nome:<38} R={R_f:.5f}  target={target:.5f}  "
              f"dist={convergiu:.1f}%  lr={lr_m:.5f}")
    else:
        print(f"  {cfg_nome:<38} R={R_f:.5f} (natural)  lr=0.10000")

# ── Visualização ──────────────────────────────────────────────────────────────
CORES = {
    "AA — sem realimentação (baseline)": "#8B949E",
    "BB — target=α=1/137   (AlphaPhi)":  "#00FF88",
    "CC — target=2α=2/137  (controle)":  "#00BFFF",
    "DD — target=α/2       (controle)":  "#FF9944",
    "EE — target=0.010     (controle)":  "#FF4466",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("#0d1117")

# Gráfico 1 — Distribuição acurácias
ax = axes[0]
ax.set_facecolor("#161b22")
ax.set_title("Distribuição de Acurácias (10 seeds)\nAblation — target do equilíbrio R",
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
ax.set_xticklabels(["AA\nbaseline","BB\nα=1/137","CC\n2α","DD\nα/2","EE\n0.010"],
                   color="#8B949E", fontsize=9)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_ylabel("Acurácia (val)", color="#8B949E")
ax.grid(True, alpha=0.15)
ax.set_ylim(0.4, 0.9)
ax.axhline(baseline.mean(), color="#8B949E", linestyle='--', alpha=0.5)

# Gráfico 2 — Trajetória de R_ema
ax = axes[1]
ax.set_facecolor("#161b22")
ax.set_title("Trajetória R_ema por época\n(eco ressonante interno — R → target)",
             color="#DAA520", fontsize=10, fontweight='bold')
epocas_plot = [5, 10, 15, 20]
for cfg_nome, traj in traj_R.items():
    if not traj or not any(traj):
        continue
    target = CONFIGS[cfg_nome]
    lw = 2.5 if "AlphaPhi" in cfg_nome else 1.5 if "baseline" in cfg_nome else 1.0
    label = cfg_nome.split("(")[0].strip()
    ax.plot(epocas_plot, traj, color=CORES[cfg_nome], linewidth=lw,
            label=label, marker='o', markersize=4)
    if target > 0:
        ax.axhline(target, color=CORES[cfg_nome], linestyle=':', alpha=0.4)
ax.axhline(ALPHA, color="#00FF88", linestyle='--', alpha=0.6,
           label=f"α={ALPHA:.4f}")
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Época", color="#8B949E")
ax.set_ylabel("R_ema = E_φ/E_CE", color="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7)
ax.grid(True, alpha=0.15)

# Gráfico 3 — Convergência de acurácia
ax = axes[2]
ax.set_facecolor("#161b22")
ax.set_title("Convergência média (10 seeds)\nlr_eff = lr·clip(1−gain·erro, 0.3, 3.0)",
             color="#DAA520", fontsize=10, fontweight='bold')
for cfg_nome, hist in historico.items():
    lw = 2.5 if "AlphaPhi" in cfg_nome else 1.5 if "baseline" in cfg_nome else 1.0
    ax.plot(epocas_plot, hist, color=CORES[cfg_nome], linewidth=lw,
            label=cfg_nome.split("(")[0].strip(), marker='o', markersize=4)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Época", color="#8B949E")
ax.set_ylabel("Acurácia (val)", color="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7, loc='lower right')
ax.grid(True, alpha=0.15)

fig.suptitle(
    f"ALPHA PHI — Exp 6: α como Equilíbrio Ressonante Interno\n"
    f"R=E_φ/E_CE → α  |  gain=1/α={round(GAIN)}  |  EMA τ=1/α  |  "
    f"φ={PHI:.4f}  α=1/{round(1/ALPHA)}  |  Florianópolis 2026",
    color="#DAA520", fontsize=10, fontweight='bold'
)
plt.tight_layout()
plt.savefig("alphaphi_alpha_resonance.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.show()

print("\nGráfico salvo: alphaphi_alpha_resonance.png")

print("\n" + "=" * 65)
print("SÍNTESE — Seis experimentos")
print("=" * 65)
print(f"""
Exp 1–5: α INJETADO como fator externo → p≈1.0 em todos

Exp 6 — Princípio novo: α como ATRATOR interno
  R = E_φ/E_CE mede a razão natural entre as duas energias
  R_natural ≈ 0.005 ≈ 0.7α  (emergência, não imposição)
  gap = φ − norm_h ≈ 7α     (casca natural em 7 unidades de α)

  Se BB (target=α) produz melhor ou mais estável convergência:
    → α é o equilíbrio natural do sistema AlphaPhi
  Se R_ema converge para α sem feedback (AA):
    → α emerge espontaneamente — resultado independente de qualquer imposição
""")
print("alpha-phi")
