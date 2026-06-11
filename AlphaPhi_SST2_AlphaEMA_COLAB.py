# ALPHA PHI — α como Taxa de Memória Temporal (EMA Geométrica)
# Vitor Edson Delavi · Florianópolis · 2026
#
# TRAJETÓRIA DOS QUATRO EXPERIMENTOS ANTERIORES:
#   Exp 1 (EntropyAlpha):    α aditivo na saída           → nulo (0.017 << CE)
#   Exp 2 (AlphaCoupling):   α como largura gaussiana      → travado (coupling≈0)
#   Exp 3 (AlphaAmplitude):  α como amplitude de δ         → nulo (0.002%)
#   Exp 4 (AlphaEntropy):    α como amplitude de lr        → nulo (0.35%)
#
# DIAGNÓSTICO CONSOLIDADO:
#   Em todas as formulações anteriores, α operou no ESPAÇO (posição do kernel,
#   amplitude da modulação, escala do lr). O efeito ficou abaixo do limiar
#   detectável (10 seeds, 20 épocas) porque α=0.007 como fator multiplicativo
#   produz correções de 0.002% a 2.4% — abaixo do ruído experimental.
#
# NOVA DIMENSÃO — α no TEMPO:
#   Em QED, α não é apenas um fator de escala espacial — ele governa a escala
#   de tempo das interações eletromagnéticas (e.g. tempo de vida de estados
#   excitados ∝ 1/α³ em emissão espontânea).
#
#   Aqui: α como coeficiente de média móvel exponencial (EMA) define
#   a janela temporal com que a rede "recorda" o estado geométrico:
#
#     τ_α = 1/α = 137 batches  (memória de 1/α batches → ~3.4 épocas)
#
#   Isso é exclusivo de α: nenhum outro valor de fator produz SIMULTANEAMENTE
#   a mesma memória temporal E a mesma amplitude de modulação.
#
# MECANISMO (Exp 5):
#   H_ema(t) = α · H(t) + (1-α) · H_ema(t-1)   ← α como taxa de memória
#   lr_eff   = lr · (1 + α · H_ema(t))           ← α como amplitude
#
#   α = 1/137: τ ≈ 137 batches, memória ~3.4 épocas → suavização longa
#   0.050:     τ ≈  20 batches, memória ~0.5 época  → resposta rápida
#   0.010:     τ ≈ 100 batches, memória ~2.5 épocas → intermediário
#   0.001:     τ ≈ 1000 batches, memória ~25 épocas → memória muito longa
#
#   α governa dois papéis distintos simultaneamente:
#     - QUANDO a rede "percebe" mudanças na geometria φ (τ = 1/fator)
#     - QUANTO essa percepção modula o passo (amplitude = fator × H_ema)
#
# ABLATION — varia o coeficiente EMA (fator):
#   V  : sem modulação (CE pura — baseline)
#   W  : EMA com fator = α = 1/137  (hipótese Alpha-Phi — τ=137 batches)
#   X  : EMA com fator = 0.010      (controle — τ=100 batches)
#   Y  : EMA com fator = 0.050      (controle — τ=20 batches)
#   Z  : EMA com fator = 0.001      (controle — τ=1000 batches)
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

BATCHES_PER_EPOCH = 5000 // 128   # ~39

print(f"φ             = {PHI:.10f}")
print(f"α = 1/137     = {ALPHA:.10f}")
print(f"raio φ        = {PHI_R:.10f}")
print(f"τ_α = 1/α     = {1/ALPHA:.1f} batches  ≈ {(1/ALPHA)/BATCHES_PER_EPOCH:.1f} épocas")
print(f"τ_0.010       = {1/0.010:.1f} batches  ≈ {(1/0.010)/BATCHES_PER_EPOCH:.1f} épocas")
print(f"τ_0.050       = {1/0.050:.1f} batches  ≈ {(1/0.050)/BATCHES_PER_EPOCH:.1f} épocas")
print(f"τ_0.001       = {1/0.001:.1f} batches  ≈ {(1/0.001)/BATCHES_PER_EPOCH:.1f} épocas")
print("=" * 65)
print("ALPHA PHI — Exp 5: α como Taxa de Memória Temporal (EMA)")
print("α governa QUANDO e QUANTO — dois papéis simultâneos no tempo")
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

# ── Entropia Geométrica (batch) ───────────────────────────────────────────────
def phi_shannon_entropy_batch(h_hidden, n_bins=10):
    """
    Shannon entropy de |norm(h) - φ| para um batch.
    Normalização correta: counts/total (sem density=True).
    """
    norms   = np.linalg.norm(h_hidden, axis=-1)
    desvios = np.abs(norms - PHI_R)
    counts, _ = np.histogram(desvios, bins=n_bins, range=(0.0, PHI_R))
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

# ── Rede Alpha-Phi com Memória Temporal α ─────────────────────────────────────
class RedeMemoriaTemporal:
    """
    Canal φ:  expmap0(φ·tanh) com c=1/φ²           (forward — geometria)
    Canal CE: δ = (ŷ-y)                             (backward — gradiente intacto)
    Canal α:  H_ema(t) = α·H(t) + (1-α)·H_ema(t-1) (EMA com janela τ=1/α)
              lr_eff   = lr · (1 + α · H_ema)

    α governa dois papéis simultâneos:
      τ = 1/fator : escala de tempo da memória geométrica
      amp = fator : amplitude da modulação de lr
    """
    def __init__(self, arch, seed, fator=0.0):
        np.random.seed(seed)
        self.arch  = arch
        self.fator = fator
        self.W, self.b = [], []
        for i in range(len(arch) - 1):
            s = np.sqrt(2.0 / arch[i])
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))
        self.H_ema        = 0.0          # estado inicial da EMA
        self.ema_hist     = []           # trajetória de H_ema
        self.H_batch_hist = []           # entropia instantânea por batch
        self.lr_hist      = []
        self.norm_hist    = []
        self.batch_count  = 0

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

        # Canal α — memória temporal da entropia geométrica
        h_hidden = self.acts[-2]
        norms    = np.linalg.norm(h_hidden, axis=-1)
        self.norm_hist.append(float(norms.mean()))

        if self.fator > 0:
            H_batch  = phi_shannon_entropy_batch(h_hidden)
            # EMA: α como coeficiente de suavização temporal
            self.H_ema = self.fator * H_batch + (1.0 - self.fator) * self.H_ema
            lr_eff     = lr_base * (1.0 + self.fator * self.H_ema)
            self.H_batch_hist.append(H_batch)
            self.ema_hist.append(self.H_ema)
            self.lr_hist.append(lr_eff)
        else:
            lr_eff = lr_base

        self.batch_count += 1

        for i in reversed(range(len(self.W))):
            self.W[i] -= lr_eff * (self.acts[i].T @ delta / m)
            self.b[i]  -= lr_eff * delta.mean(0)
            if i > 0:
                delta = delta @ self.W[i].T

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def ema_final(self):
        return float(self.H_ema)

    def lr_medio(self):
        if not self.lr_hist:
            return 0.0
        return float(np.mean(self.lr_hist))

    def norm_media(self):
        return float(np.mean(self.norm_hist)) if self.norm_hist else 0.0

    def ema_por_epoca(self, n_batches_por_epoca):
        """Média de H_ema por época para o gráfico de trajetória."""
        if not self.ema_hist:
            return []
        resultado = []
        for ep in range(4):   # épocas 5, 10, 15, 20
            inicio = ep * n_batches_por_epoca * 5
            fim    = inicio + n_batches_por_epoca * 5
            trecho = self.ema_hist[inicio:fim]
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
print(f"Total batches (20 épocas): {n_batches * 20}")
print(f"τ_α = 137 batches ≈ {137/n_batches:.1f} épocas")
print(f"τ_0.010 = 100 batches ≈ {100/n_batches:.1f} épocas")
print(f"τ_0.050 = 20 batches ≈ {20/n_batches:.1f} épocas")
print(f"τ_0.001 = 1000 batches ≈ {1000/n_batches:.1f} épocas")

# ── Configurações ─────────────────────────────────────────────────────────────
CONFIGS = {
    "V — sem modulação (baseline)":   0.0,
    "W — EMA α=1/137   (AlphaPhi)":   ALPHA,
    "X — EMA 0.010     (controle)":   0.010,
    "Y — EMA 0.050     (controle)":   0.050,
    "Z — EMA 0.001     (controle)":   0.001,
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
emas_finais = {cfg: [] for cfg in CONFIGS}
lrs_medios  = {cfg: [] for cfg in CONFIGS}
normas      = {cfg: [] for cfg in CONFIGS}
traj_ema    = {cfg: [] for cfg in CONFIGS}   # trajetória de H_ema por época

print("\n" + "=" * 65)
for cfg_nome, fator in CONFIGS.items():
    tau_str = f"τ={1/fator:.0f}b" if fator > 0 else "sem EMA"
    print(f"\n{cfg_nome}  ({tau_str})")
    accs_seed, hist_seed, ema_seed, lr_seed, norm_seed, traj_seed = [], [], [], [], [], []
    for seed in seeds:
        rede = RedeMemoriaTemporal(arch_in, seed, fator=fator)
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
        ema_seed.append(rede.ema_final())
        lr_seed.append(rede.lr_medio())
        norm_seed.append(rede.norm_media())
        traj = rede.ema_por_epoca(n_batches)
        traj_seed.append(traj)
        print(f"  seed {seed} → acc={acc_final:.4f}  "
              f"H_ema_final={rede.ema_final():.4f}  "
              f"lr_med={rede.lr_medio():.6f}  "
              f"norm={rede.norm_media():.4f}")
    resultados[cfg_nome]  = accs_seed
    historico[cfg_nome]   = np.mean(hist_seed, axis=0)
    emas_finais[cfg_nome] = float(np.mean(ema_seed))
    lrs_medios[cfg_nome]  = float(np.mean(lr_seed))
    normas[cfg_nome]      = float(np.mean(norm_seed))
    traj_ema[cfg_nome]    = np.mean(traj_seed, axis=0).tolist() if traj_seed[0] else [0]*4
    print(f"  → Média: {np.mean(accs_seed):.4f} ± {np.std(accs_seed):.4f}  "
          f"H_ema_final={np.mean(ema_seed):.4f}  τ={1/fator:.0f}b" if fator > 0 else
          f"  → Média: {np.mean(accs_seed):.4f} ± {np.std(accs_seed):.4f}  (baseline)")

# ── Análise Estatística ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTADOS — α como Taxa de Memória Temporal (EMA)")
print("H_ema(t) = α·H(t) + (1-α)·H_ema(t-1)  |  lr_eff = lr·(1+α·H_ema)")
print("=" * 65)

baseline = np.array(resultados["V — sem modulação (baseline)"])
print(f"\n{'Config':<38} {'Média':>7} {'Std':>7} {'Δ vs V':>8} {'p-valor':>10} {'H_ema':>7} {'Δlr%':>8} {'τ(b)':>7}")
print("-" * 105)

for cfg_nome, accs in resultados.items():
    arr    = np.array(accs)
    fator  = CONFIGS[cfg_nome]
    h_ema  = emas_finais[cfg_nome]
    lr_med = lrs_medios[cfg_nome]
    tau    = round(1/fator) if fator > 0 else 0
    if "baseline" in cfg_nome:
        print(f"{cfg_nome:<38} {arr.mean():>7.4f} {arr.std():>7.4f} "
              f"{'—':>8} {'—':>10} {'—':>7} {'—':>8} {'—':>7}")
    else:
        t, p   = stats.ttest_ind(arr, baseline)
        dlr    = (lr_med / LR - 1) * 100 if lr_med > 0 else 0
        sig    = "✓" if p < 0.05 else "ns"
        print(f"{cfg_nome:<38} {arr.mean():>7.4f} {arr.std():>7.4f} "
              f"{arr.mean()-baseline.mean():>+8.4f} {p:>10.4f} "
              f"{h_ema:>7.4f} {dlr:>+8.4f}% {tau:>7d} {sig}")

print("\nW (α=1/137, τ=137b) vs controles:")
w_vals = np.array(resultados["W — EMA α=1/137   (AlphaPhi)"])
for cfg_nome, accs in resultados.items():
    if "baseline" in cfg_nome or "AlphaPhi" in cfg_nome:
        continue
    arr  = np.array(accs)
    fator = CONFIGS[cfg_nome]
    tau   = round(1/fator)
    t, p  = stats.ttest_ind(w_vals, arr)
    delta = w_vals.mean() - arr.mean()
    sig   = "✓ τ específico" if p < 0.05 else "ns — τ não importa"
    print(f"  W vs {cfg_nome:<32} Δ={delta:+.4f}  p={p:.4f}  τ_ctrl={tau}b  {sig}")

# ── Diagnóstico de escala temporal ───────────────────────────────────────────
print("\n" + "─" * 65)
print("Diagnóstico — escala de memória temporal por configuração:")
print(f"  Total batches treinados: {N_SEEDS * EPOCHS * n_batches}")
for cfg_nome in CONFIGS:
    fator = CONFIGS[cfg_nome]
    if fator > 0:
        tau      = 1/fator
        frac_tau = tau / (EPOCHS * n_batches) * 100
        dlr      = (lrs_medios[cfg_nome] / LR - 1) * 100
        print(f"  {cfg_nome:<38} τ={tau:6.0f}b  ({frac_tau:.1f}% do treino)  "
              f"Δlr={dlr:+.4f}%  H_ema={emas_finais[cfg_nome]:.4f}")
    else:
        print(f"  {cfg_nome:<38} sem EMA")

# ── Visualização ──────────────────────────────────────────────────────────────
CORES = {
    "V — sem modulação (baseline)":   "#8B949E",
    "W — EMA α=1/137   (AlphaPhi)":   "#00FF88",
    "X — EMA 0.010     (controle)":   "#00BFFF",
    "Y — EMA 0.050     (controle)":   "#FF9944",
    "Z — EMA 0.001     (controle)":   "#FF4466",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("#0d1117")

# Gráfico 1 — Distribuição acurácias
ax = axes[0]
ax.set_facecolor("#161b22")
ax.set_title("Distribuição de Acurácias (10 seeds)\nAblation — τ = 1/fator EMA",
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
ax.set_xticklabels(["V\nbaseline","W\nα=1/137\nτ=137b","X\n0.010\nτ=100b",
                    "Y\n0.050\nτ=20b","Z\n0.001\nτ=1000b"], color="#8B949E", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_ylabel("Acurácia (val)", color="#8B949E")
ax.grid(True, alpha=0.15)
ax.set_ylim(0.5, 0.9)
ax.axhline(baseline.mean(), color="#8B949E", linestyle='--', alpha=0.5)

# Gráfico 2 — Convergência
ax = axes[1]
ax.set_facecolor("#161b22")
ax.set_title("Convergência média (10 seeds)\nlr_eff = lr·(1 + fator·H_ema)",
             color="#DAA520", fontsize=10, fontweight='bold')
epocas_plot = [5, 10, 15, 20]
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

# Gráfico 3 — Trajetória de H_ema por época
ax = axes[2]
ax.set_facecolor("#161b22")
ax.set_title("Trajetória H_ema por época\n(α governa τ=1/fator batches de memória)",
             color="#DAA520", fontsize=10, fontweight='bold')
for cfg_nome, traj in traj_ema.items():
    if CONFIGS[cfg_nome] == 0 or not traj or not any(traj):
        continue
    fator = CONFIGS[cfg_nome]
    lw = 2.5 if "AlphaPhi" in cfg_nome else 1.0
    ax.plot(epocas_plot, traj, color=CORES[cfg_nome], linewidth=lw,
            label=f"{cfg_nome.split('(')[0].strip()} τ={round(1/fator)}b",
            marker='o', markersize=4)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Época", color="#8B949E")
ax.set_ylabel("H_ema (entropia geom. suavizada)", color="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7, loc='upper right')
ax.grid(True, alpha=0.15)

fig.suptitle(
    f"ALPHA PHI — Exp 5: α como Taxa de Memória Temporal\n"
    f"H_ema(t)=α·H(t)+(1-α)·H_ema(t-1)  |  lr_eff=lr·(1+α·H_ema)  |  τ_α=1/α={round(1/ALPHA)}b  |  "
    f"φ={PHI:.4f}  α=1/{round(1/ALPHA)}  |  Florianópolis 2026",
    color="#DAA520", fontsize=10, fontweight='bold'
)
plt.tight_layout()
plt.savefig("alphaphi_alpha_ema.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.show()

print("\nGráfico salvo: alphaphi_alpha_ema.png")

# ── Síntese dos cinco experimentos ───────────────────────────────────────────
print("\n" + "=" * 65)
print("SÍNTESE — Cinco experimentos, cinco posições de α")
print("=" * 65)
print(f"""
Exp 1 — EntropyAlpha:    α na saída aditiva    → nulo (α·H << CE)
Exp 2 — AlphaCoupling:   α como largura kernel → travado (coupling=0)
Exp 3 — AlphaAmplitude:  α como amplitude δ    → nulo (0.002% — coupling 0.997)
Exp 4 — AlphaEntropy:    α como amplitude lr   → nulo (0.35% — H_campo baixo)
Exp 5 — AlphaEMA (este): α como taxa EMA       → τ=1/α={round(1/ALPHA)} batches

Diferencial do Exp 5:
  α governa DOIS PAPÉIS SIMULTÂNEOS no tempo:
    τ = 1/α = {round(1/ALPHA)} batches: escala de memória da geometria
    amp = α = 1/{round(1/ALPHA)}: amplitude da modulação de lr
  Nenhum outro valor de fator produz a mesma combinação (τ, amp).
""")
print("alpha-phi")
