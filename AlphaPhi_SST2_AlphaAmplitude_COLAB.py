# ALPHA PHI — α como Amplitude de Acoplamento Geométrico
# Vitor Edson Delavi · Florianópolis · 2026
#
# DIAGNÓSTICO DOS EXPERIMENTOS ANTERIORES:
#   Exp 1 (EntropyAlpha):   α como penalidade aditiva de entropia → nulo
#                           α·H(φ) ≈ 0.017 — pequeno demais para competir com CE
#
#   Exp 2 (AlphaCoupling):  α como largura de acoplamento gaussiano → rede travada
#                           bandwidth α=0.007 em normas 0.1–0.5 → coupling≈0.0005
#                           α não é parâmetro espacial — é constante de acoplamento
#
# REFORMULAÇÃO FUNDAMENTAL:
#   Em QED: α não define a escala do campo — define com que AMPLITUDE
#           o campo eletromagnético acopla com a carga.
#   Analogia: α não define a largura da gaussiana — define o QUANTO
#             a geometria φ modula o gradiente de aprendizado.
#
# MECANISMO CORRIGIDO:
#   coupling_natural = exp(-|norm(h) - φ|² / φ²)    ← escala natural φ² (não α)
#   δ = (ŷ - y) · [1 - α · (1 - coupling_natural)]   ← α como amplitude
#
#   Quando coupling_natural = 1 (norm(h) ≈ φ):
#     δ = (ŷ-y) · [1 - α·0] = (ŷ-y) · 1.0           → gradiente completo
#
#   Quando coupling_natural = 0 (norm(h) muito longe de φ):
#     δ = (ŷ-y) · [1 - α·1] = (ŷ-y) · (1 - 1/137)   → gradiente 99.27% do normal
#                                                        (α determina o quanto)
#
#   → α governa a AMPLITUDE da influência geométrica, não o raio de ação
#   → φ² governa a ESCALA em que a influência atua
#   → os dois papéis são distintos e específicos de cada constante
#
# ABLATION — varia apenas a amplitude de acoplamento:
#   L  : sem acoplamento (CE pura — baseline)
#   M  : amplitude = α = 1/137  (hipótese Alpha-Phi)
#   N  : amplitude = 0.010      (controle)
#   O  : amplitude = 0.050      (controle)
#   P  : amplitude = 0.001      (controle)
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

print(f"φ             = {PHI:.10f}")
print(f"α = 1/137     = {ALPHA:.10f}")
print(f"c = 1/φ²      = {C_PHI:.10f}")
print(f"raio φ        = {PHI_R:.10f}  (1/√c = φ)")
print(f"escala φ²     = {PHI_R**2:.8f}  (largura natural do acoplamento)")
print(f"amplitude α   = {ALPHA:.8f}  (fator de modulação do gradiente)")
print(f"efeito máx.   = {ALPHA*100:.4f}% de redução (1 - 1/137)")
print("=" * 65)
print("ALPHA PHI — α como Amplitude de Acoplamento Geométrico")
print("Exp 3 — α na posição correta: amplitude, não escala")
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

# ── Acoplamento Geométrico — Amplitude α ──────────────────────────────────────
def phi_coupling_natural(h_hidden):
    """
    Acoplamento geométrico com escala natural φ².

    coupling = exp( -|norm(h) - φ|² / φ² )

    A largura é φ² — a escala natural do espaço hiperbólico.
    α não entra aqui. α entra no uso desta grandeza.

    Retorna shape (m, 1).
    """
    norms     = np.linalg.norm(h_hidden, axis=-1)        # (m,)
    deviation = (norms - PHI_R) ** 2                      # desvio quadrático
    coupling  = np.exp(-deviation / (PHI_R**2 + 1e-10))  # escala natural φ²
    return coupling.reshape(-1, 1)                         # (m, 1)

def amplitude_delta(yhat, y, h_hidden, amplitude):
    """
    Gradiente modulado por amplitude de acoplamento geométrico.

    δ = (ŷ - y) · [1 - amplitude · (1 - coupling_natural)]

    amplitude = 0   → CE pura: δ = (ŷ-y)·1
    amplitude = α   → AlphaPhi: modulação 0.73% nos pontos fora de φ
    amplitude = 0.1 → modulação 10% nos pontos fora de φ

    Retorna (delta, coupling_mean).
    """
    delta    = yhat - y.reshape(-1, 1)                    # gradiente CE base
    coupling = phi_coupling_natural(h_hidden)             # [0, 1] — posição no atrator
    modulator = 1.0 - amplitude * (1.0 - coupling)       # [1-amp, 1]
    return delta * modulator, float(coupling.mean())

# ── Rede Alpha-Phi com Amplitude α ────────────────────────────────────────────
class RedeAmplitudeCoupling:
    """
    Arquitetura: Fibonacci + expmap0(φ·tanh) + c=1/φ²
    Backward:    δ = (ŷ - y) · [1 - amplitude · (1 - coupling_natural(h))]

    amplitude = 0       → CE pura (baseline)
    amplitude = α=1/137 → hipótese AlphaPhi
    amplitude = outros  → controles ablation
    """
    def __init__(self, arch, seed, amplitude=0.0):
        np.random.seed(seed)
        self.arch      = arch
        self.amplitude = amplitude
        self.W, self.b = [], []
        for i in range(len(arch) - 1):
            s = np.sqrt(2.0 / arch[i])   # He init
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))
        self.coupling_hist = []
        self.norm_hist     = []

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

        if self.amplitude > 0:
            h_hidden = self.acts[-2]   # última camada oculta (espaço hiperbólico)
            delta, coup_mean = amplitude_delta(yhat, y, h_hidden, self.amplitude)
            self.coupling_hist.append(coup_mean)
            # monitorar normas da camada oculta
            norms = np.linalg.norm(h_hidden, axis=-1)
            self.norm_hist.append(float(norms.mean()))
        else:
            delta = yhat - y.reshape(-1, 1)   # CE pura

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

    def norm_media(self):
        if not self.norm_hist:
            return 0.0
        return float(np.mean(self.norm_hist[-100:]))

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

# Diagnóstico: normas dos embeddings vs raio φ
emb_norms = np.linalg.norm(X_train, axis=-1)
print(f"\nNormas dos embeddings brutos:")
print(f"  média={emb_norms.mean():.4f}  std={emb_norms.std():.4f}  "
      f"min={emb_norms.min():.4f}  max={emb_norms.max():.4f}")
print(f"  raio φ = {PHI_R:.4f}")
print(f"  → normas {emb_norms.mean()/PHI_R:.1f}× o raio φ no input")
print(f"  (a rede aprende a reposicionar via expmap0 + tanh)")

# ── Configurações ─────────────────────────────────────────────────────────────
CONFIGS = {
    "L — sem acoplamento (baseline)":  0.0,
    "M — amp=α=1/137    (AlphaPhi)":   ALPHA,
    "N — amp=0.010      (controle)":   0.010,
    "O — amp=0.050      (controle)":   0.050,
    "P — amp=0.001      (controle)":   0.001,
}

print(f"\nAmplitudes testadas:")
for nome, amp in CONFIGS.items():
    if amp > 0:
        print(f"  {nome}  →  modulação máx={amp*100:.4f}%")

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

# ── Treinamento ───────────────────────────────────────────────────────────────
resultados   = {cfg: [] for cfg in CONFIGS}
historico    = {cfg: [] for cfg in CONFIGS}
acoplamentos = {cfg: [] for cfg in CONFIGS}
normas       = {cfg: [] for cfg in CONFIGS}

print("\n" + "=" * 65)
for cfg_nome, amp in CONFIGS.items():
    print(f"\n{cfg_nome}  (amplitude={amp:.6f})")
    accs_seed  = []
    hist_seed  = []
    coup_seed  = []
    norm_seed  = []
    for seed in seeds:
        rede = RedeAmplitudeCoupling(arch_in, seed, amplitude=amp)
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
        norm_seed.append(rede.norm_media())
        print(f"  seed {seed} → acc={acc_final:.4f}  "
              f"coupling={rede.coupling_medio():.4f}  "
              f"norm_h={rede.norm_media():.4f}")
    resultados[cfg_nome]   = accs_seed
    historico[cfg_nome]    = np.mean(hist_seed, axis=0)
    acoplamentos[cfg_nome] = np.mean(coup_seed)
    normas[cfg_nome]       = np.mean(norm_seed)
    print(f"  → Média: {np.mean(accs_seed):.4f} ± {np.std(accs_seed):.4f}  "
          f"coupling médio: {np.mean(coup_seed):.4f}  "
          f"norm_h média: {np.mean(norm_seed):.4f}")

# ── Análise Estatística ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTADOS — α como Amplitude de Acoplamento Geométrico")
print("Exp 3 — escala=φ² (natural), amplitude=α (hipótese)")
print("=" * 65)

baseline = np.array(resultados["L — sem acoplamento (baseline)"])
print(f"\n{'Config':<38} {'Média':>7} {'Std':>7} {'Δ vs L':>8} {'p-valor':>10} {'Acopl.':>8} {'Norm_h':>7}")
print("-" * 100)

for cfg_nome, accs in resultados.items():
    arr    = np.array(accs)
    coup   = acoplamentos[cfg_nome]
    norm_h = normas[cfg_nome]
    if "baseline" in cfg_nome:
        print(f"{cfg_nome:<38} {arr.mean():>7.4f} {arr.std():>7.4f} "
              f"{'—':>8} {'—':>10} {'—':>8} {norm_h:>7.4f}")
    else:
        t, p = stats.ttest_ind(arr, baseline)
        sig = "✓" if p < 0.05 else "ns"
        print(f"{cfg_nome:<38} {arr.mean():>7.4f} {arr.std():>7.4f} "
              f"{arr.mean()-baseline.mean():>+8.4f} {p:>10.4f} {coup:>8.4f} {norm_h:>7.4f} {sig}")

print("\nM (α=1/137) vs controles de amplitude:")
m_vals = np.array(resultados["M — amp=α=1/137    (AlphaPhi)"])
for cfg_nome, accs in resultados.items():
    if "baseline" in cfg_nome or "AlphaPhi" in cfg_nome:
        continue
    arr = np.array(accs)
    t, p = stats.ttest_ind(m_vals, arr)
    delta = m_vals.mean() - arr.mean()
    sig = "✓ α é específico" if p < 0.05 else "ns — amplitude não importa"
    print(f"  M vs {cfg_nome:<32} Δ={delta:+.4f}  p={p:.4f}  {sig}")

# ── Diagnóstico de posição no atrator ────────────────────────────────────────
print("\n" + "─" * 65)
print(f"Diagnóstico — posição das representações vs raio φ={PHI_R:.4f}:")
print(f"(coupling=1.0 → norm(h)≈φ | coupling→0 → representação distante de φ)")
for cfg_nome in CONFIGS:
    amp    = CONFIGS[cfg_nome]
    coup   = acoplamentos[cfg_nome]
    norm_h = normas[cfg_nome]
    dist   = abs(norm_h - PHI_R)
    if amp > 0:
        efeito = amp * (1.0 - coup) * 100
        print(f"  {cfg_nome:<38} norm={norm_h:.4f}  dist_φ={dist:.4f}  "
              f"coupling={coup:.4f}  efeito={efeito:.4f}%")
    else:
        print(f"  {cfg_nome:<38} norm={norm_h:.4f}  (sem modulação)")

print(f"\nNota interpretativa:")
print(f"  coupling_natural usa escala φ²={PHI_R**2:.4f} (natural ao espaço)")
print(f"  α={ALPHA:.6f} governa a AMPLITUDE, não a escala")
print(f"  modulação máxima (coupling=0): δ reduzido em {ALPHA*100:.4f}%")
print(f"  modulação mínima (coupling=1): δ intacto (gradiente completo)")

# ── Visualização ──────────────────────────────────────────────────────────────
CORES = {
    "L — sem acoplamento (baseline)":  "#8B949E",
    "M — amp=α=1/137    (AlphaPhi)":   "#00FF88",
    "N — amp=0.010      (controle)":   "#00BFFF",
    "O — amp=0.050      (controle)":   "#FF9944",
    "P — amp=0.001      (controle)":   "#FF4466",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("#0d1117")

# Gráfico 1 — Distribuição acurácias
ax = axes[0]
ax.set_facecolor("#161b22")
ax.set_title("Distribuição de Acurácias (10 seeds)\nAblation — amplitude de acoplamento α",
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
ax.set_xticklabels(["L\nbaseline","M\nα=1/137","N\n0.010","O\n0.050","P\n0.001"],
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
ax.set_title("Convergência média (10 seeds)\nEscala=φ²  Amplitude=α,0.001,0.010,0.050",
             color="#DAA520", fontsize=10, fontweight='bold')
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

# Gráfico 3 — Coupling médio por amplitude
ax = axes[2]
ax.set_facecolor("#161b22")
ax.set_title("Coupling natural por amplitude\n(φ² = escala natural — invariante)",
             color="#DAA520", fontsize=10, fontweight='bold')
amps  = [CONFIGS[c] for c in CONFIGS if CONFIGS[c] > 0]
coups = [acoplamentos[c] for c in CONFIGS if CONFIGS[c] > 0]
cores = [CORES[c] for c in CORES if CONFIGS.get(c, 0) > 0]
ax.scatter(amps, coups, c=cores, s=120, zorder=5)
for amp, coup, cor, lbl in zip(amps, coups, cores,
                                ["M α=1/137", "N 0.010", "O 0.050", "P 0.001"]):
    ax.annotate(lbl, (amp, coup), textcoords="offset points",
                xytext=(5, 5), color=cor, fontsize=8)
ax.axvline(ALPHA, color="#00FF88", linestyle='--', alpha=0.7,
           label=f"α=1/137={ALPHA:.4f}")
ax.axhline(1.0, color="#8B949E", linestyle=':', alpha=0.4, label="coupling=1 (≡φ)")
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Amplitude de acoplamento", color="#8B949E")
ax.set_ylabel("Coupling médio (escala φ²)", color="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.grid(True, alpha=0.15)

fig.suptitle(
    f"ALPHA PHI — Exp 3: α como Amplitude de Acoplamento Geométrico\n"
    f"δ=(ŷ−y)·[1−α·(1−coupling_φ)]  |  coupling_φ=exp(−|h−φ|²/φ²)  |  φ={PHI:.4f}  α=1/{round(1/ALPHA)}"
    f"  |  Florianópolis 2026",
    color="#DAA520", fontsize=10, fontweight='bold'
)
plt.tight_layout()
plt.savefig("alphaphi_alpha_amplitude.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.show()

print("\nGráfico salvo: alphaphi_alpha_amplitude.png")

# ── Síntese dos três experimentos ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("SÍNTESE — Trajetória dos três experimentos")
print("=" * 65)
print("""
Exp 1 — EntropyAlpha:
  α como penalidade aditiva de entropia na saída
  L = CE + α·H(φ)·yhat·(1-yhat)
  Resultado: NULO (p≈1.0 todos configs)
  Diagnóstico: gradiente α·H ≈ 0.017 << CE — α·H não chega à arquitetura

Exp 2 — AlphaCoupling:
  α como largura da gaussiana de acoplamento geométrico
  δ = (ŷ-y) · exp(-|norm-φ|²/(α·φ²))
  Resultado: TRAVADO (coupling≈0.0005, rede freeze em ~50%)
  Diagnóstico: normas ativação 0.1–0.5, raio φ=1.618 → distância enorme
               α=0.007 como largura ≡ exigir norma exatamente em 1.618 ± 0.007
               α não é parâmetro espacial — é constante de acoplamento

Exp 3 — AlphaAmplitude (este):
  α como amplitude da modulação de acoplamento geométrico
  coupling_natural = exp(-|norm-φ|²/φ²)   ← escala natural φ²
  δ = (ŷ-y) · [1 - α·(1 - coupling_natural)]
  α governa o QUANTO a geometria influencia — não ONDE ela atua
  φ² governa a escala — papel específico de φ
  α governa a amplitude — papel específico de α
""")
print("alpha-phi")
