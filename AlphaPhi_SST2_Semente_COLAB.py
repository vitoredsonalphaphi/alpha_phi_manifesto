# AlphaPhi_SST2_Semente_COLAB.py
# Vitor Edson Delavi · Florianopolis · 2026
#
# SST2 — REDE NEURAL COM SEMENTE ALPHA
#
# Semente aplicada as ativacoes ocultas da rede:
#
#   SEMENTE:   an = clip(|h_i| / sum(|h_i|), alpha, 1.0)   piso alpha
#              H_alpha_i = H_i / log(137)                   regua irrevogavel
#              coh_alpha_i = 1 - H_alpha_i                  coerencia em alpha-nats
#
#   BETA TEXT: beta_text = phi^(3 * mean(coh_alpha))        campo harmonico textual
#              CAMPO: beta_text >= phi^3 = 4.236
#
#   ECO-ENTROPICO (CC):
#              delta_H = H_alpha - H_ema
#              periferica (H>H_ema): lr *= clip(1 + delta*phi, 1, phi^2)  ancora
#              nuclear   (H<H_ema): lr *= clip(1 - |delta|/phi, 0.1, 1)   expande
#
# ABLACAO:
#   AA: baseline Exp6 (sem semente, sem eco)         regressao
#   BB: semente medindo (H_alpha, beta_text — sem lr feedback)
#   CC: semente + eco-entropico (H_alpha drive lr)
#
# PERGUNTAS:
#   (1) Campo harmonico textual forma? (beta_text >= phi^3)
#   (2) BB/CC diferencia de AA em acuracia?
#   (3) H_alpha_equilibrio do substrato texto (alpha* textual)
#   (4) R_natural texto com semente -> numero de Lucas?

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers", "datasets", "scipy"], check=True)

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI       = (1 + np.sqrt(5)) / 2
ALPHA     = 1 / 137.035999084
LOG_ALPHA = np.log(1.0 / ALPHA)   # log(137) ~= 4.920
C_PHI     = 1.0 / PHI**2

print(f"phi          = {PHI:.10f}")
print(f"alpha        = {ALPHA:.10f}  (1/{int(round(1/ALPHA))})")
print(f"log(137)     = {LOG_ALPHA:.6f}  (regua de entropia)")
print(f"phi^3        = {PHI**3:.6f}  (limiar campo harmonico)")
print("=" * 65)
print("SST2 — Semente Alpha na Rede Neural")
print("alpha na entropia das ativacoes ocultas")
print("=" * 65)

# ─── Funcoes base ─────────────────────────────────────────────────────────────
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

# ─── M1: Semente nas ativacoes ────────────────────────────────────────────────
def semente_ativacoes(h):
    """
    Aplica semente alpha as ativacoes ocultas.
    h: (batch, features)
    Retorna: H_alpha por amostra, coh_alpha por amostra
    """
    mag  = np.abs(h)                                          # (batch, features)
    s    = mag.sum(axis=1, keepdims=True)
    an   = np.clip(mag / (s + 1e-8), ALPHA, 1.0)             # piso alpha
    an   = an / an.sum(axis=1, keepdims=True)                 # normaliza
    H    = -np.sum(an * np.log(an + 1e-15), axis=1)          # (batch,)
    H_a  = np.clip(H / LOG_ALPHA, 0.0, 1.0)                  # alpha-nats
    return H_a, 1.0 - H_a                                     # H_alpha, coh_alpha

# ─── M2: Eco-entropico ────────────────────────────────────────────────────────
def eco_entropico_lr(H_alpha_med, H_ema, lr_base):
    """
    Modula lr com base na posicao da entropia no triangulo.
    """
    delta = H_alpha_med - H_ema
    if delta > 0:   # periferica: ancora (E)
        scale = float(np.clip(1.0 + delta * PHI, 1.0, PHI**2))
    else:           # nuclear: expande (X)
        scale = float(np.clip(1.0 - abs(delta) / PHI, 0.1, 1.0))
    return lr_base * scale

# ─── Rede Neural com Semente ──────────────────────────────────────────────────
class RedeSemente:
    """
    Rede com semente alpha nas ativacoes ocultas.
    usar_semente: mede H_alpha + beta_text por batch
    usar_eco:     eco-entropico modula lr
    """
    def __init__(self, arch, seed, usar_semente=False, usar_eco=False):
        np.random.seed(seed)
        self.arch         = arch
        self.usar_semente = usar_semente
        self.usar_eco     = usar_eco
        self.W, self.b    = [], []
        for i in range(len(arch) - 1):
            s = np.sqrt(2.0 / arch[i])
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))

        # Estado semente
        self.H_ema      = 0.5          # inicia no meio do triangulo
        self.beta_mem   = 1.0          # memoria do beta textual
        self.R_ema      = ALPHA        # mesmo inicio do Exp6

        # Historico
        self.R_hist      = []
        self.H_alpha_hist = []
        self.beta_hist   = []
        self.coh_hist    = []
        self.lr_hist     = []
        self.campo_ciclo = None

    def _activation(self, z, last=False):
        if last: return sigmoid(z)
        return expmap0(PHI * np.tanh(z / PHI))

    def forward(self, X):
        self.acts = [X]
        cur = X
        for i in range(len(self.W)):
            z   = cur @ self.W[i] + self.b[i]
            cur = self._activation(z, last=(i == len(self.W) - 1))
            self.acts.append(cur)
        return cur

    def backward(self, X, y, lr_base, batch_idx=0):
        m    = X.shape[0]
        yhat = self.acts[-1]
        h    = self.acts[-2]   # ultima camada oculta

        # Gradiente CE — intocado
        delta = yhat - y.reshape(-1, 1)

        # Medicao semente
        if self.usar_semente:
            H_alphas, coh_alphas = semente_ativacoes(h)
            H_alpha_med  = float(H_alphas.mean())
            coh_alpha_med = float(coh_alphas.mean())

            # Beta textual: campo harmonico em alpha-nats
            ba         = PHI ** (3.0 * coh_alpha_med)
            beta_text  = 0.9 * ba + 0.1 * self.beta_mem   # memoria leve
            self.beta_mem = beta_text

            # EMA de H_alpha (tau = 1/alpha = 137 batches)
            self.H_ema = ALPHA * H_alpha_med + (1.0 - ALPHA) * self.H_ema

            self.H_alpha_hist.append(H_alpha_med)
            self.beta_hist.append(beta_text)
            self.coh_hist.append(coh_alpha_med)

            # Campo harmonico textual
            if beta_text >= PHI**3 and self.campo_ciclo is None:
                self.campo_ciclo = batch_idx

            # Eco-entropico
            lr_eff = eco_entropico_lr(H_alpha_med, self.H_ema, lr_base) \
                     if self.usar_eco else lr_base
        else:
            # AA baseline: sem semente
            H_alpha_med  = 0.0
            coh_alpha_med = 0.0
            beta_text    = 1.0
            lr_eff       = lr_base

        # R = E_phi / E_CE (mantido para comparacao com Exp6)
        norms  = np.linalg.norm(h, axis=-1)
        E_phi  = float(np.mean((norms - 1.0/np.sqrt(C_PHI))**2))
        eps    = 1e-10
        yhat_  = np.clip(yhat.flatten(), eps, 1-eps)
        E_CE   = float(-np.mean(y.reshape(-1)*np.log(yhat_) +
                                (1-y.reshape(-1))*np.log(1-yhat_)))
        R      = E_phi / (E_CE + eps)
        self.R_ema = ALPHA * R + (1.0 - ALPHA) * self.R_ema
        self.R_hist.append(self.R_ema)
        self.lr_hist.append(lr_eff)

        # Backprop
        for i in reversed(range(len(self.W))):
            self.W[i] -= lr_eff * (self.acts[i].T @ delta / m)
            self.b[i]  -= lr_eff * delta.mean(0)
            if i > 0:
                delta = delta @ self.W[i].T

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def H_alpha_equilibrio(self):
        if not self.H_alpha_hist: return 0.0
        return float(np.mean(self.H_alpha_hist[-50:]))

    def beta_max(self):
        return float(max(self.beta_hist)) if self.beta_hist else 0.0

    def coh_media(self):
        return float(np.mean(self.coh_hist[-50:])) if self.coh_hist else 0.0

# ─── Dados SST-2 ──────────────────────────────────────────────────────────────
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

N_SEEDS = 10
EPOCHS  = 20
LR      = 0.1
BATCH   = 128
arch_in = [X_train.shape[1]] + FIB + [1]
n_batches = len(X_train) // BATCH
print(f"Arquitetura: {arch_in}")
print(f"Batches/epoca: {n_batches}")

base_seed = int(time.time())
seeds = [base_seed + i * 7 for i in range(N_SEEDS)]
print(f"Seeds ({N_SEEDS}): {seeds[:3]}...{seeds[-1]}")

# ─── Configs ──────────────────────────────────────────────────────────────────
CONFIGS = {
    "AA — baseline   (sem semente, sem eco)": {"usar_semente": False, "usar_eco": False, "color": "#8B949E"},
    "BB — semente    (H_alpha medindo)      ": {"usar_semente": True,  "usar_eco": False, "color": "#00BFFF"},
    "CC — eco-entrop (semente + lr feedback)": {"usar_semente": True,  "usar_eco": True,  "color": "#00FF88"},
}

# ─── Treinamento ──────────────────────────────────────────────────────────────
resultados = {}
print("\n" + "=" * 65)

for cfg_nome, cfg in CONFIGS.items():
    print(f"\n{cfg_nome}")
    accs, R_vals, H_eq_vals, beta_max_vals, campo_ciclos, coh_vals = \
        [], [], [], [], [], []

    for seed in seeds:
        rede = RedeSemente(arch_in, seed,
                           usar_semente=cfg["usar_semente"],
                           usar_eco=cfg["usar_eco"])
        batch_global = 0
        for ep in range(EPOCHS):
            idx = np.random.permutation(len(X_train))
            for b in range(0, len(X_train), BATCH):
                xb = X_train[idx[b:b+BATCH]]
                yb = y_train[idx[b:b+BATCH]]
                rede.forward(xb)
                rede.backward(xb, yb, LR, batch_global)
                batch_global += 1

        acc = rede.accuracy(X_val, y_val)
        accs.append(acc)
        R_vals.append(rede.R_ema)
        H_eq_vals.append(rede.H_alpha_equilibrio())
        beta_max_vals.append(rede.beta_max())
        coh_vals.append(rede.coh_media())
        campo_ciclos.append(rede.campo_ciclo)

        campo_str = f"b{rede.campo_ciclo}" if rede.campo_ciclo else "—"
        print(f"  seed {seed} → acc={acc:.4f}"
              f"  H_eq={rede.H_alpha_equilibrio():.4f}"
              f"  beta_max={rede.beta_max():.4f}"
              f"  campo={campo_str}"
              f"  R={rede.R_ema:.5f}")

    campos_formados = [c for c in campo_ciclos if c is not None]
    resultados[cfg_nome] = {
        "accs": accs, "R_vals": R_vals,
        "H_eq": H_eq_vals, "beta_max": beta_max_vals,
        "campo_ciclos": campo_ciclos,
        "campos_formados": campos_formados,
        "coh": coh_vals,
        "color": cfg["color"]
    }
    print(f"  → acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}"
          f"  H_eq: {np.mean(H_eq_vals):.4f}"
          f"  beta_max: {np.mean(beta_max_vals):.4f}"
          f"  campo: {len(campos_formados)}/{N_SEEDS} sementes"
          f"  R: {np.mean(R_vals):.5f}")

# ─── Sintese ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SINTESE — SST2 Semente Alpha")
print("=" * 65)

baseline_accs = np.array(resultados["AA — baseline   (sem semente, sem eco)"]["accs"])
print(f"\n{'Config':<42} {'Acc':>8} {'Std':>7} {'Delta':>8}"
      f" {'H_eq':>7} {'beta_max':>9} {'campo':>8}")
print("-" * 95)

for nome, res in resultados.items():
    accs = np.array(res["accs"])
    delta = np.mean(accs) - np.mean(baseline_accs)
    n_campo = len(res["campos_formados"])
    print(f"{nome:<42} {np.mean(accs):.4f} {np.std(accs):.4f}"
          f"  {delta:+.4f}"
          f"  {np.mean(res['H_eq']):.4f}"
          f"  {np.mean(res['beta_max']):.4f}"
          f"  {n_campo}/{N_SEEDS}")

# Estatistica BB vs AA
bb_accs = np.array(resultados["BB — semente    (H_alpha medindo)      "]["accs"])
cc_accs = np.array(resultados["CC — eco-entrop (semente + lr feedback)"]["accs"])
_, p_bb = stats.ttest_ind(baseline_accs, bb_accs)
_, p_cc = stats.ttest_ind(baseline_accs, cc_accs)
print(f"\nTeste t: BB vs AA p={p_bb:.4f}  |  CC vs AA p={p_cc:.4f}")

print(f"\nReferencia campo harmonico: phi^3 = {PHI**3:.4f}")
print(f"beta_max esperado se campo forma: >= {PHI**3:.4f}")

phi4_lucas = PHI**4 + PHI**(-4)
print(f"\nphi^4 + phi^-4 = {phi4_lucas:.6f}  (Lucas 4 — referencia audio)")
print(f"R_natural texto (Exp6 baseline): ~5*alpha = {5*ALPHA:.5f}")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("SST2 — Semente Alpha na Rede Neural\n"
             "α na entropia das ativacoes | φ como atrator",
             fontsize=12, fontweight="bold")

configs_list = list(resultados.keys())

ax = axes[0]
for nome, res in resultados.items():
    accs = res["accs"]
    ax.scatter([nome[:10]] * len(accs), accs,
               color=res["color"], alpha=0.6, s=30)
    ax.plot([nome[:10]], [np.mean(accs)],
            color=res["color"], marker="D", markersize=8)
ax.set_title("Acuracia por config"); ax.set_ylabel("Acc")

ax = axes[1]
for nome, res in resultados.items():
    betas = res["beta_max"]
    ax.scatter([nome[:10]] * len(betas), betas,
               color=res["color"], alpha=0.6, s=30)
    ax.plot([nome[:10]], [np.mean(betas)],
            color=res["color"], marker="D", markersize=8)
ax.axhline(PHI**3, color="gold", linestyle="--", linewidth=1.5,
           label=f"phi^3={PHI**3:.3f}")
ax.set_title("beta_max (campo harmonico textual)"); ax.legend(fontsize=8)

ax = axes[2]
for nome, res in resultados.items():
    h_vals = res["H_eq"]
    ax.scatter([nome[:10]] * len(h_vals), h_vals,
               color=res["color"], alpha=0.6, s=30)
    ax.plot([nome[:10]], [np.mean(h_vals)],
            color=res["color"], marker="D", markersize=8)
ax.set_title("H_alpha equilibrio (eixo entropico)")

for a in axes:
    a.set_facecolor("#0d1117"); a.grid(alpha=0.2)
    for sp in a.spines.values(): sp.set_edgecolor("#444")
    a.tick_params(colors="#ccc"); a.title.set_color("#eee")
fig.patch.set_facecolor("#0d1117")

plt.tight_layout()
plt.savefig("alphaphi_sst2_semente.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("\nalpha-phi | sst2 semente | alphaphi_sst2_semente.png")
