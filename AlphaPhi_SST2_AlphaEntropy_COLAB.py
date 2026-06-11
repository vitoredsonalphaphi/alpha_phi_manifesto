# ALPHA PHI — α como Regulador Termodinâmico via Entropia Geométrica
# Vitor Edson Delavi · Florianópolis · 2026
#
# TRAJETÓRIA DOS EXPERIMENTOS:
#   Exp 1 (EntropyAlpha):    α aditivo na saída       → nulo (0.017 << CE)
#   Exp 2 (AlphaCoupling):   α como largura gaussiana  → travado (coupling≈0)
#   Exp 3 (AlphaAmplitude):  α como amplitude do δ     → nulo (modulação 0.002%)
#
# DIAGNÓSTICO CONSOLIDADO:
#   φ age na geometria das ativações via expmap0 — as normas convergem para
#   97% de φ no forward pass. α tentou agir no backward pass, mas o gradiente
#   CE absorveu o efeito antes de chegar a nível mensurável.
#
# PRINCÍPIO DO ENVELOPAMENTO HERMÉTICO:
#   Nos experimentos ECO-BEEP, a terceira paralela foi envelopada de forma
#   hermética para que o processamento angular dos cinco pontos de dobra não
#   interferisse nem no sinal de base nem no campo harmônico.
#   O mesmo princípio aqui: α não entra no canal do gradiente (CE).
#   α opera em canal próprio — a taxa de aprendizado — de forma que
#   CE, φ e α atuem em domínios separados sem absorção mútua.
#
# MECANISMO (Exp 4):
#   Canal φ:  forward — expmap0(φ·tanh) com c=1/φ²   [geometria — intacto]
#   Canal CE: backward — δ = (ŷ-y)                   [gradiente — hermético]
#   Canal α:  lr — lr_dinâmico = lr · (1 + α · H_campo)  [termodinâmica — separado]
#
#   H_campo = Shannon entropy de |norm(h) - φ| sobre o batch
#             → alto quando representações dispersas (longe de φ) → α acelera
#             → baixo quando representações concentradas (perto de φ) → α estabiliza
#
#   Interpretação termodinâmica:
#     Alta entropia geométrica → sistema desordenado → α amplifica passo de correção
#     Baixa entropia geométrica → sistema harmônico → α freia, preserva estrutura
#
# ABLATION — varia apenas o fator de modulação de lr:
#   Q  : sem modulação (CE pura — baseline)
#   R  : fator = α = 1/137  (hipótese Alpha-Phi)
#   S  : fator = 0.010      (controle)
#   T  : fator = 0.050      (controle)
#   U  : fator = 0.001      (controle)
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
print(f"φ²            = {PHI_R**2:.8f}  (escala natural)")
print(f"α·log2(10)    = {ALPHA * np.log2(10):.6f}  (efeito máx. de α sobre lr)")
print(f"efeito lr máx = {ALPHA * np.log2(10) * 100:.4f}%  (entropia máx 10 bins)")
print("=" * 65)
print("ALPHA PHI — Exp 4: α como Regulador Termodinâmico")
print("Envelopamento hermético: α no canal lr, CE no canal δ, φ no canal geom.")
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

# ── Entropia Geométrica de Shannon ────────────────────────────────────────────
def phi_shannon_entropy(h_hidden, n_bins=10):
    """
    Entropia de Shannon da distribuição de |norm(h) - φ| no batch.

    Mede o grau de desordem das representações em relação ao atrator φ.
    Alta entropia  → representações dispersas (longe de φ) → sistema desordenado
    Baixa entropia → representações concentradas (perto de φ) → sistema harmônico

    Correção em relação à versão Gemini:
      density=False + normalização manual → distribuição de probabilidade correta.
      density=True fornece densidade (integral=1), não probabilidade por bin.
    """
    norms   = np.linalg.norm(h_hidden, axis=-1)
    desvios = np.abs(norms - PHI_R)

    # range cobre até PHI_R para incluir desvios desde 0 até o raio inteiro
    counts, _ = np.histogram(desvios, bins=n_bins, range=(0.0, PHI_R))
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total              # probabilidade por bin (soma = 1)
    p = p[p > 0]                    # remove zeros para evitar log(0)
    return float(-np.sum(p * np.log2(p)))  # entropia de Shannon

def lr_termodinamico(lr_base, fator, h_campo):
    """
    Taxa de aprendizado modulada por α e entropia do campo.

    lr_din = lr_base · (1 + fator · H_campo)

    fator = 0     → lr constante (baseline)
    fator = α     → α como regulador termodinâmico (hipótese)
    H_campo alto  → lr maior (sistema desordenado, α acelera correção)
    H_campo baixo → lr menor (sistema harmônico, α preserva estrutura)
    """
    return lr_base * (1.0 + fator * h_campo)

# ── Rede Alpha-Phi com Regulação Termodinâmica ────────────────────────────────
class RedeTermodinamica:
    """
    Três canais separados — envelopamento hermético:
      Canal φ:  expmap0(φ·tanh) com c=1/φ²           (forward — geometria)
      Canal CE: δ = (ŷ-y)                             (backward — gradiente, intacto)
      Canal α:  lr_din = lr·(1 + fator·H_campo)       (lr — termodinâmica)

    α não entra no gradiente. α opera no envelope externo (lr).
    CE e α não competem no mesmo canal.
    """
    def __init__(self, arch, seed, fator=0.0):
        np.random.seed(seed)
        self.arch  = arch
        self.fator = fator
        self.W, self.b = [], []
        for i in range(len(arch) - 1):
            s = np.sqrt(2.0 / arch[i])   # He init
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))
        self.entropia_hist = []
        self.lr_hist       = []
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

    def backward(self, X, y, lr_base):
        m    = X.shape[0]
        yhat = self.acts[-1]

        # Canal CE — gradiente puro, intocado
        delta = yhat - y.reshape(-1, 1)

        # Canal α — calcula lr pelo estado termodinâmico do campo
        h_hidden = self.acts[-2]
        norms    = np.linalg.norm(h_hidden, axis=-1)
        self.norm_hist.append(float(norms.mean()))

        if self.fator > 0:
            h_campo = phi_shannon_entropy(h_hidden)
            lr_din  = lr_termodinamico(lr_base, self.fator, h_campo)
            self.entropia_hist.append(h_campo)
            self.lr_hist.append(lr_din)
        else:
            lr_din  = lr_base   # baseline: lr constante

        # Backpropagation com lr termodinâmico
        for i in reversed(range(len(self.W))):
            self.W[i] -= lr_din * (self.acts[i].T @ delta / m)
            self.b[i]  -= lr_din * delta.mean(0)
            if i > 0:
                delta = delta @ self.W[i].T

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def entropia_media(self):
        if not self.entropia_hist:
            return 0.0
        return float(np.mean(self.entropia_hist[-100:]))

    def lr_medio(self):
        if not self.lr_hist:
            return 0.0
        return float(np.mean(self.lr_hist[-100:]))

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

# Diagnóstico de escala de entropia esperada
print(f"\nEscala de entropia geométrica:")
print(f"  H mínima (representações em 1 bin) = 0.0")
print(f"  H máxima (10 bins uniformes)       = {np.log2(10):.4f}")
print(f"  Efeito de α no lr: [{ALPHA*0.0*100:.4f}%, {ALPHA*np.log2(10)*100:.4f}%]")
print(f"  Efeito de 0.010  : [{0.010*0.0*100:.4f}%, {0.010*np.log2(10)*100:.4f}%]")
print(f"  Efeito de 0.050  : [{0.050*0.0*100:.4f}%, {0.050*np.log2(10)*100:.4f}%]")

# ── Configurações ─────────────────────────────────────────────────────────────
CONFIGS = {
    "Q — sem modulação (baseline)":  0.0,
    "R — fator=α=1/137  (AlphaPhi)": ALPHA,
    "S — fator=0.010    (controle)": 0.010,
    "T — fator=0.050    (controle)": 0.050,
    "U — fator=0.001    (controle)": 0.001,
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

# ── Treinamento ───────────────────────────────────────────────────────────────
resultados  = {cfg: [] for cfg in CONFIGS}
historico   = {cfg: [] for cfg in CONFIGS}
entropias   = {cfg: [] for cfg in CONFIGS}
lrs_medios  = {cfg: [] for cfg in CONFIGS}
normas      = {cfg: [] for cfg in CONFIGS}
hist_entropia = {cfg: [] for cfg in CONFIGS}   # trajetória de entropia por época

print("\n" + "=" * 65)
for cfg_nome, fator in CONFIGS.items():
    print(f"\n{cfg_nome}  (fator={fator:.6f})")
    accs_seed  = []
    hist_seed  = []
    entr_seed  = []
    lr_seed    = []
    norm_seed  = []
    hist_e_seed = []
    for seed in seeds:
        rede = RedeTermodinamica(arch_in, seed, fator=fator)
        hist  = []
        hist_e = []
        for ep in range(EPOCHS):
            idx = np.random.permutation(len(X_train))
            for b in range(0, len(X_train), BATCH):
                xb = X_train[idx[b:b+BATCH]]
                yb = y_train[idx[b:b+BATCH]]
                rede.forward(xb)
                rede.backward(xb, yb, LR)
            if ep % 5 == 4:
                hist.append(rede.accuracy(X_val, y_val))
                # entropia média nesta época (últimos batches)
                if rede.entropia_hist:
                    n_ep = len(X_train) // BATCH
                    ep_entropias = rede.entropia_hist[-n_ep:]
                    hist_e.append(float(np.mean(ep_entropias)))
                else:
                    hist_e.append(0.0)
        acc_final = rede.accuracy(X_val, y_val)
        accs_seed.append(acc_final)
        hist_seed.append(hist)
        entr_seed.append(rede.entropia_media())
        lr_seed.append(rede.lr_medio())
        norm_seed.append(rede.norm_media())
        hist_e_seed.append(hist_e)
        print(f"  seed {seed} → acc={acc_final:.4f}  "
              f"H_campo={rede.entropia_media():.4f}  "
              f"lr_din={rede.lr_medio():.6f}  "
              f"norm_h={rede.norm_media():.4f}")
    resultados[cfg_nome]    = accs_seed
    historico[cfg_nome]     = np.mean(hist_seed, axis=0)
    entropias[cfg_nome]     = float(np.mean(entr_seed))
    lrs_medios[cfg_nome]    = float(np.mean(lr_seed))
    normas[cfg_nome]        = float(np.mean(norm_seed))
    hist_entropia[cfg_nome] = np.mean(hist_e_seed, axis=0) if hist_e_seed[0] else np.zeros(4)
    print(f"  → Média: {np.mean(accs_seed):.4f} ± {np.std(accs_seed):.4f}  "
          f"H_campo={np.mean(entr_seed):.4f}  "
          f"lr_din={np.mean(lr_seed):.6f}")

# ── Análise Estatística ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTADOS — α como Regulador Termodinâmico (Envelopamento Hermético)")
print("Canal α: lr_din = lr·(1+fator·H_campo)  |  Canal CE: δ=ŷ-y (intacto)")
print("=" * 65)

baseline = np.array(resultados["Q — sem modulação (baseline)"])
print(f"\n{'Config':<38} {'Média':>7} {'Std':>7} {'Δ vs Q':>8} {'p-valor':>10} {'H_campo':>8} {'lr_din':>9}")
print("-" * 105)

for cfg_nome, accs in resultados.items():
    arr    = np.array(accs)
    h_c    = entropias[cfg_nome]
    lr_d   = lrs_medios[cfg_nome]
    if "baseline" in cfg_nome:
        print(f"{cfg_nome:<38} {arr.mean():>7.4f} {arr.std():>7.4f} "
              f"{'—':>8} {'—':>10} {'—':>8} {'—':>9}")
    else:
        t, p = stats.ttest_ind(arr, baseline)
        sig = "✓" if p < 0.05 else "ns"
        print(f"{cfg_nome:<38} {arr.mean():>7.4f} {arr.std():>7.4f} "
              f"{arr.mean()-baseline.mean():>+8.4f} {p:>10.4f} {h_c:>8.4f} {lr_d:>9.6f} {sig}")

print("\nR (α=1/137) vs controles:")
r_vals = np.array(resultados["R — fator=α=1/137  (AlphaPhi)"])
for cfg_nome, accs in resultados.items():
    if "baseline" in cfg_nome or "AlphaPhi" in cfg_nome:
        continue
    arr = np.array(accs)
    t, p = stats.ttest_ind(r_vals, arr)
    delta = r_vals.mean() - arr.mean()
    sig = "✓ α é específico" if p < 0.05 else "ns — fator não importa"
    print(f"  R vs {cfg_nome:<32} Δ={delta:+.4f}  p={p:.4f}  {sig}")

# ── Diagnóstico termodinâmico ─────────────────────────────────────────────────
print("\n" + "─" * 65)
print("Diagnóstico — entropia geométrica e efeito termodinâmico de α:")
print(f"(H_campo mede desordem das normas em relação ao atrator φ={PHI_R:.4f})")
for cfg_nome in CONFIGS:
    fator  = CONFIGS[cfg_nome]
    h_c    = entropias[cfg_nome]
    lr_d   = lrs_medios[cfg_nome]
    norm_h = normas[cfg_nome]
    if fator > 0:
        delta_lr = (lr_d / LR - 1) * 100
        print(f"  {cfg_nome:<38} H={h_c:.4f}  "
              f"lr_din={lr_d:.6f}  Δlr={delta_lr:+.4f}%  norm_h={norm_h:.4f}")
    else:
        print(f"  {cfg_nome:<38} H=—       lr_base=0.100000  norm_h={norm_h:.4f}")

# ── Visualização ──────────────────────────────────────────────────────────────
CORES = {
    "Q — sem modulação (baseline)":  "#8B949E",
    "R — fator=α=1/137  (AlphaPhi)": "#00FF88",
    "S — fator=0.010    (controle)": "#00BFFF",
    "T — fator=0.050    (controle)": "#FF9944",
    "U — fator=0.001    (controle)": "#FF4466",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("#0d1117")

# Gráfico 1 — Distribuição acurácias
ax = axes[0]
ax.set_facecolor("#161b22")
ax.set_title("Distribuição de Acurácias (10 seeds)\nAblation — fator termodinâmico α",
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
ax.set_xticklabels(["Q\nbaseline","R\nα=1/137","S\n0.010","T\n0.050","U\n0.001"],
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
ax.set_title("Convergência média (10 seeds)\nlr_din = lr·(1 + fator·H_campo)",
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

# Gráfico 3 — Trajetória de entropia geométrica
ax = axes[2]
ax.set_facecolor("#161b22")
ax.set_title("Trajetória da Entropia Geométrica H(campo)\n(desordem das normas em relação a φ)",
             color="#DAA520", fontsize=10, fontweight='bold')
for cfg_nome, hist_e in hist_entropia.items():
    if CONFIGS[cfg_nome] == 0 or not any(hist_e):
        continue
    lw = 2.5 if "AlphaPhi" in cfg_nome else 1.2
    ax.plot(epocas_plot, hist_e, color=CORES[cfg_nome], linewidth=lw,
            label=cfg_nome.split("(")[0].strip(), marker='o', markersize=4)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Época", color="#8B949E")
ax.set_ylabel("H_campo (entropia Shannon)", color="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7, loc='upper right')
ax.grid(True, alpha=0.15)

fig.suptitle(
    f"ALPHA PHI — Exp 4: α como Regulador Termodinâmico · Envelopamento Hermético\n"
    f"lr_din=lr·(1+α·H_φ)  |  H_φ=Shannon(|norm−φ|)  |  Canal CE intacto  |  "
    f"φ={PHI:.4f}  α=1/{round(1/ALPHA)}  |  Florianópolis 2026",
    color="#DAA520", fontsize=10, fontweight='bold'
)
plt.tight_layout()
plt.savefig("alphaphi_alpha_entropy.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.show()

print("\nGráfico salvo: alphaphi_alpha_entropy.png")

# ── Síntese dos quatro experimentos ──────────────────────────────────────────
print("\n" + "=" * 65)
print("SÍNTESE — Quatro experimentos, quatro posições de α")
print("=" * 65)
print("""
Exp 1 — EntropyAlpha:
  α aditivo na saída — L = CE + α·H(φ)·yhat·(1-yhat)
  Resultado: NULO — gradiente 0.017 << CE

Exp 2 — AlphaCoupling:
  α como largura do kernel gaussiano
  δ = (ŷ-y)·exp(-|norm-φ|²/(α·φ²))
  Resultado: TRAVADO — coupling≈0, rede freeze em 50%

Exp 3 — AlphaAmplitude:
  α como amplitude da modulação do gradiente
  δ = (ŷ-y)·[1-α·(1-coupling_natural)]
  Resultado: NULO — coupling≈0.997, modulação 0.002%
  Observação: φ já posicionou representações em norm≈1.57 no forward

Exp 4 — AlphaEntropy (este):
  α como regulador termodinâmico no canal lr — envelopamento hermético
  lr_din = lr·(1 + α·H_campo)
  Canal CE intacto — α não compete com o gradiente
  Canal φ intacto — expmap0 age no forward sem interferência
  Canal α separado — opera no envelope da taxa de aprendizado
""")
print("alpha-phi")
