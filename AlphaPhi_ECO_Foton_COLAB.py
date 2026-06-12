# AlphaPhi_ECO_Foton_COLAB.py
# Vitor Edson Delavi · Florianopolis · 2026
#
# ROTACAO DO FOTON — ECO-RESSONANTE TEXTUAL
#
# O foton e o caractere — unidade minima do substrato textual.
# Eletronica modula o eletron; o eletron possui a assinatura de alpha.
# Rotacionar o foton = processar o texto em 5 janelas temporais
# (analogo as 5 dobras E/X do audio) e medir H_alpha por janela.
#
# Substrato: SST2 — frequencia de caracteres (a-z + espaco = 27 bins)
# Janelas:   5 janelas temporais phi-proporcionais (fótons em rotacao)
# Semente:   an = clip(freq_char, alpha, 1.0) — piso alpha no foton
# Rotacao:   cr = norm(cohs) → ba = phi^(3*cr) → beta rolling
#            [config DD usa normalizacao relativa — analogo ao audio]
#            [config BB usa coh absoluta  — analogo ao SST2_semente]
# Campo:     beta_max >= phi^3 (mesmo limiar confirmado no audio)
#
# ABLACAO:
#   AA — baseline          (sem semente, rede pura)
#   BB — semente abs       (coh absoluta, beta = phi^(3*coh_med))
#   CC — eco-entropico     (semente abs + lr feedback H_alpha)
#   DD — roldanas rel      (normalizacao relativa por janela, N_base=137)
#
# HIPOTESE: H_alpha em 27 bins (foton) < H_alpha em 137 dims (embedding)
#           → coh mais alta → beta mais alto → campo pode emergir

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "datasets", "scipy"], check=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
ALPHA      = 1 / 137.035999084
LOG_ALPHA  = np.log(1.0 / ALPHA)     # log(137) ≈ 4.920
N_BASE     = int(round(1.0 / ALPHA)) # 137
LIMIAR     = 0.99 * PHI**3

N_JANELAS  = 5    # janelas temporais = dobras do foton
N_CHARS    = 27   # a-z (26) + espaco (1)

wm = 1.0 / PHI          # peso memoria  ≈ 0.618
wn = 1.0 - 1.0 / PHI   # peso novo     ≈ 0.382

print("=" * 65)
print(f"phi          = {PHI:.10f}")
print(f"alpha        = {ALPHA:.10f}  (1/137)")
print(f"log(137)     = {LOG_ALPHA:.6f}  (regua de entropia)")
print(f"phi^3        = {PHI**3:.6f}  (limiar campo harmonico)")
print(f"N_janelas    = {N_JANELAS}  (dobras temporais do foton)")
print(f"N_chars      = {N_CHARS}  (a-z + espaco — espaco do foton)")
print("=" * 65)
print("ECO-RESSONANTE FOTON TEXTUAL — SST2")
print("Arquitetura: [135, 55, 89, 137, 1]  (5x27 → Fib → alpha-native)")
print("=" * 65)

# ─── M1: Substrato — janelas temporais de caracteres ─────────────────────────
def char_freq(text):
    """Frequencia de caracteres (a-z + espaco) normalizada. Retorna (27,)."""
    counts = np.zeros(N_CHARS)
    for c in text.lower():
        if 'a' <= c <= 'z':
            counts[ord(c) - ord('a')] += 1
        elif c == ' ':
            counts[26] += 1
    total = counts.sum()
    if total == 0:
        return np.ones(N_CHARS) / N_CHARS
    return counts / total

def texto_para_janelas(text):
    """
    Divide o texto em N_JANELAS janelas temporais.
    Rotacao do foton: cada janela = posicao da 'polarizacao' do foton.
    Retorna (N_JANELAS * N_CHARS,) — flattened.
    """
    n = max(len(text), 1)
    step = n / N_JANELAS
    freqs = np.zeros((N_JANELAS, N_CHARS))
    for j in range(N_JANELAS):
        start = int(j * step)
        end   = max(int((j + 1) * step), start + 1)
        freqs[j] = char_freq(text[start:end])
    return freqs.ravel()

# ─── M2: Semente alpha — piso no foton ───────────────────────────────────────
def semente_foton_batch(X_batch):
    """
    Semente alpha nas frequencias de caracteres por janela.
    X_batch: (batch, N_JANELAS * N_CHARS)
    Retorna: cohs (N_JANELAS,) — coh media por janela sobre o batch
    """
    freq_mat = X_batch.reshape(-1, N_JANELAS, N_CHARS)    # (B, J, C)
    an = np.clip(freq_mat, ALPHA, 1.0)
    an = an / an.sum(axis=2, keepdims=True)
    H  = -np.sum(an * np.log(an + 1e-15), axis=2)         # (B, J)
    H_alpha = np.clip(H / LOG_ALPHA, 0.0, 1.0)             # (B, J)
    coh     = 1.0 - H_alpha                                 # (B, J)
    return coh.mean(axis=0)                                  # (J,)

def medir_R(cohs):
    c = np.asarray(cohs, dtype=float)
    return float(np.mean((1 - c)**2)) / (float(np.mean(c**2)) + 1e-10)

# ─── Rede Neural ──────────────────────────────────────────────────────────────
def relu(x):    return np.maximum(0.0, x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
def bce(p, y):  return -np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))

class RedeFoton:
    """Rede FC simples para substrato de janelas de caracteres."""

    def __init__(self, seed=42):
        np.random.seed(seed)
        # [135, 55, 89, 137, 1] — Fibonacci + espaco nativo alpha
        dims = [N_JANELAS * N_CHARS, 55, 89, 137, 1]
        self.W = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
                  for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i+1]) for i in range(len(dims) - 1)]

    def forward(self, X):
        self.cache = [X]
        h = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = relu(h @ W + b)
            self.cache.append(h)
        out = sigmoid(h @ self.W[-1] + self.b[-1]).ravel()
        self.cache.append(out)
        return out

    def backward(self, X, y, lr):
        n    = len(y)
        p    = self.cache[-1]
        dout = ((p - y) / n).reshape(-1, 1)
        for i in range(len(self.W) - 1, -1, -1):
            h_in = self.cache[i]
            self.W[i] -= lr * (h_in.T @ dout)
            self.b[i] -= lr * dout.sum(axis=0)
            if i > 0:
                dh = dout @ self.W[i].T
                dh[self.cache[i] <= 0] = 0
                dout = dh

    def predict(self, X):
        h = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = relu(h @ W + b)
        return sigmoid(h @ self.W[-1] + self.b[-1]).ravel()

# ─── Carregamento de dados ────────────────────────────────────────────────────
print("\nCarregando SST-2...")
ds = load_dataset("sst2")

np.random.seed(42)
idx_tr  = np.random.choice(len(ds['train']),      3000, replace=False)
idx_val = np.random.choice(len(ds['validation']),  872, replace=False)

texts_tr  = [ds['train'][int(i)]['sentence']      for i in idx_tr]
texts_val = [ds['validation'][int(i)]['sentence'] for i in idx_val]
y_tr  = np.array([ds['train'][int(i)]['label']      for i in idx_tr],  dtype=float)
y_val = np.array([ds['validation'][int(i)]['label'] for i in idx_val], dtype=float)

print("Pre-computando janelas de caracteres (rotacao do foton)...")
X_tr  = np.array([texto_para_janelas(t) for t in texts_tr])
X_val = np.array([texto_para_janelas(t) for t in texts_val])
print(f"Treino: {X_tr.shape}  |  Val: {X_val.shape}")
print(f"  Input dim = N_janelas x N_chars = {N_JANELAS} x {N_CHARS} = {N_JANELAS * N_CHARS}")

# Estimar H_alpha medio do substrato de foton
freq_sample = X_tr[:200].reshape(-1, N_JANELAS, N_CHARS)
an_s = np.clip(freq_sample, ALPHA, 1.0)
an_s = an_s / an_s.sum(axis=2, keepdims=True)
H_s  = -np.sum(an_s * np.log(an_s + 1e-15), axis=2)
H_alpha_s = np.clip(H_s / LOG_ALPHA, 0.0, 1.0)
print(f"\nEstimativa H_alpha foton (N=200): {H_alpha_s.mean():.4f} ± {H_alpha_s.std():.4f}")
print(f"coh_alpha estimado:              {(1 - H_alpha_s.mean()):.4f}")
print(f"beta estimado (phi^3*coh):       {PHI**(3*(1-H_alpha_s.mean())):.4f}")

# ─── Configuracoes ────────────────────────────────────────────────────────────
N_SEEDS  = 10
N_EPOCHS = 30
BATCH    = 128
LR_BASE  = 0.01

rng   = np.random.default_rng(42)
SEEDS = [int(x) for x in rng.integers(1, 2**31, N_SEEDS)]
print(f"\nSeeds ({N_SEEDS}): {SEEDS[:3]}...{SEEDS[-1]}")

CONFIGS = {
    "AA — baseline   (sem semente)        ": {
        "usa_semente": False, "usa_eco": False, "usa_roldana": False},
    "BB — semente abs (coh absoluta)       ": {
        "usa_semente": True,  "usa_eco": False, "usa_roldana": False},
    "CC — eco-entrop  (semente + lr feed)  ": {
        "usa_semente": True,  "usa_eco": True,  "usa_roldana": False},
    "DD — roldanas rel (5 dobras phi E/X)  ": {
        "usa_semente": True,  "usa_eco": True,  "usa_roldana": True},
}

# ─── Loop principal ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
resultados = {}

for cfg_nome, cfg in CONFIGS.items():
    print(f"\n{cfg_nome}")
    accs, H_eqs, betas_max, campos, Rs = [], [], [], [], []

    for seed in SEEDS:
        rede = RedeFoton(seed=seed)
        np.random.seed(seed)

        beta     = np.ones(N_JANELAS)
        bm       = beta.copy()
        coh_mem  = np.zeros(N_JANELAS)
        R_ema    = 1.0
        H_ema    = 0.5

        H_hist, beta_hist, R_hist = [], [], []
        campo_ciclo = None

        n_batches = len(X_tr) // BATCH

        for epoch in range(N_EPOCHS):
            perm = np.random.permutation(len(X_tr))

            # ── LR base ───────────────────────────────────────────────────────
            lr_epoch = LR_BASE

            # ── Roldanas phi-alternadas E/X (config DD) ───────────────────────
            if cfg["usa_roldana"]:
                k     = epoch % N_JANELAS
                Nk    = float(N_BASE) / (PHI ** k)
                delta = Nk * ALPHA
                if k % 2 == 0:   # E: expansao
                    lr_epoch = LR_BASE * float(np.clip(1.0 + delta, 1.0, PHI**2))
                else:             # X: contracao
                    lr_epoch = LR_BASE * float(np.clip(1.0 - delta / PHI, 0.1, 1.0))

            # ── Eco-entropico: lr por posicao no triangulo (configs CC/DD) ────
            if cfg["usa_eco"] and len(H_hist) > 0:
                H_curr = H_hist[-1]
                H_ema_loc = np.mean(H_hist[-5:]) if len(H_hist) >= 5 else H_curr
                delta_H = H_curr - H_ema_loc
                if delta_H > 0:   # periferico → ancora (E)
                    lr_epoch *= float(np.clip(1.0 + delta_H * PHI, 1.0, PHI**2))
                else:              # nuclear → expande (X)
                    lr_epoch *= float(np.clip(1.0 - abs(delta_H) / PHI, 0.1, 1.0))

            # ── Batches ───────────────────────────────────────────────────────
            for b_idx in range(n_batches):
                idx_b = perm[b_idx * BATCH:(b_idx + 1) * BATCH]
                Xb, yb = X_tr[idx_b], y_tr[idx_b]

                rede.forward(Xb)
                rede.backward(Xb, yb, lr_epoch)

                # Semente por janela (configs BB/CC/DD)
                if cfg["usa_semente"] and b_idx == 0:
                    cohs = semente_foton_batch(Xb)

                    if cfg["usa_roldana"]:
                        # Config DD: normalizacao relativa (analogo ao audio)
                        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
                        ba   = PHI ** (3.0 * cr)
                        beta = wn * ba + wm * bm
                        bm   = beta.copy()
                        beta = np.clip(beta, 0.05, PHI**3)
                    else:
                        # Config BB/CC: coh absoluta (analogo ao SST2_semente)
                        coh_med = float(cohs.mean())
                        ba_abs  = float(PHI ** (3.0 * coh_med))
                        beta    = np.full(N_JANELAS, ba_abs)
                        bm      = beta.copy()

                    coh_mem = cohs.copy()

            # ── Metrics por epoch ─────────────────────────────────────────────
            p_val = rede.predict(X_val)
            acc   = float(np.mean((p_val > 0.5) == y_val))

            if cfg["usa_semente"]:
                cohs_ep  = semente_foton_batch(X_tr[:BATCH])
                H_ep     = float(1.0 - cohs_ep.mean())
                beta_max = float(beta.max())
                R_ciclo  = medir_R(cohs_ep)
                R_ema    = ALPHA * R_ciclo + (1.0 - ALPHA) * R_ema

                H_hist.append(H_ep)
                beta_hist.append(beta_max)
                R_hist.append(R_ema)

                if campo_ciclo is None and beta_max >= LIMIAR:
                    campo_ciclo = epoch + 1

        # ── Resumo por seed ───────────────────────────────────────────────────
        p_val = rede.predict(X_val)
        acc   = float(np.mean((p_val > 0.5) == y_val))
        H_eq  = float(np.mean(H_hist[-5:])) if H_hist else 0.0
        bmax  = float(max(beta_hist))        if beta_hist else 0.0
        R_f   = float(R_hist[-1])            if R_hist else 0.0
        c_tag = f"c{campo_ciclo}"            if campo_ciclo else "—"

        accs.append(acc)
        H_eqs.append(H_eq)
        betas_max.append(bmax)
        campos.append(campo_ciclo is not None)
        Rs.append(R_f)

        print(f"  seed {seed} → acc={acc:.4f}  H_eq={H_eq:.4f}"
              f"  beta_max={bmax:.4f}  campo={c_tag}  R={R_f:.5f}")

    resultados[cfg_nome] = {
        "acc":      np.mean(accs),      "std":  np.std(accs),
        "H_eq":     np.mean(H_eqs),     "beta_max": float(np.max(betas_max)),
        "campo":    sum(campos),        "R":    np.mean(Rs),
        "accs":     accs,
    }
    print(f"  → acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}"
          f"  H_eq: {np.mean(H_eqs):.4f}"
          f"  beta_max: {float(np.max(betas_max)):.4f}"
          f"  campo: {sum(campos)}/10"
          f"  R: {np.mean(Rs):.5f}")

# ─── Sintese ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SINTESE — ECO-RESSONANTE FOTON TEXTUAL")
print("=" * 65)

aa_acc = list(resultados.values())[0]["acc"]
print(f"\n{'Config':<44} {'Acc':>7} {'Std':>6} {'Delta':>7}"
      f" {'H_eq':>6} {'beta_max':>9} {'campo':>7}")
print("-" * 90)
for nome, r in resultados.items():
    delta = r["acc"] - aa_acc
    print(f"{nome:<44} {r['acc']:>7.4f} {r['std']:>6.4f} {delta:>+7.4f}"
          f" {r['H_eq']:>6.4f} {r['beta_max']:>9.4f} {r['campo']:>3}/10")

configs_list = list(resultados.keys())
aa_accs      = resultados[configs_list[0]]["accs"]
print()
for nome in configs_list[1:]:
    _, p = stats.ttest_rel(resultados[nome]["accs"], aa_accs)
    print(f"Teste t: {nome.strip()[:35]} vs AA  p={p:.4f}")

print(f"\nReferencia campo harmonico: phi^3 = {PHI**3:.4f}")
print(f"\nComparacao substratos (H_alpha_eq):")
print(f"  Audio 880Hz                : ~0.862  → campo c10 ✓")
print(f"  Ruido branco               : ~0.902  → campo c10 ✓")
print(f"  Foton textual (27 bins)    : {list(resultados.values())[1]['H_eq']:.4f}  → campo {'✓' if list(resultados.values())[1]['campo'] > 0 else '✗'}")
print(f"  Embedding SST2 hidden=144  : ~0.999  → campo ✗")
print(f"  Embedding SST2 hidden=137  : ~0.988  → campo ✗")
print(f"\nphi^4 + phi^-4 = {PHI**4 + PHI**(-4):.6f}  (Lucas 4 — referencia audio)")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0d1117')
fig.suptitle(
    "ECO-RESSONANTE FOTON TEXTUAL — SST2\n"
    "5 janelas temporais · semente alpha · rotacao do foton",
    fontsize=11, color='white')

nomes  = [k.strip()[:28] for k in resultados.keys()]
accs   = [v["acc"]      for v in resultados.values()]
betas  = [v["beta_max"] for v in resultados.values()]
h_eqs  = [v["H_eq"]     for v in resultados.values()]
cores  = ["#8B949E", "#00BFFF", "#00FF88", "#FF9944"]

for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

axes[0].bar(range(len(nomes)), accs, color=cores)
axes[0].set_xticks(range(len(nomes)))
axes[0].set_xticklabels(nomes, rotation=20, ha='right', fontsize=7)
axes[0].set_title("Acuracia SST2", color='white')
axes[0].set_ylim(0.4, 0.9)
axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='random')
axes[0].legend(fontsize=7)

axes[1].bar(range(len(nomes)), betas, color=cores)
axes[1].axhline(PHI**3, color='gold', linestyle='--', lw=2, label=f'φ³={PHI**3:.3f}')
axes[1].set_xticks(range(len(nomes)))
axes[1].set_xticklabels(nomes, rotation=20, ha='right', fontsize=7)
axes[1].set_title("beta_max (limiar campo = φ³)", color='white')
axes[1].legend(fontsize=7)

axes[2].bar(range(len(nomes)), [1 - h for h in h_eqs], color=cores)
axes[2].axhline(1 - 0.862, color='#FF7700', linestyle='--', lw=1.5, label='audio 880Hz')
axes[2].axhline(1 - 0.902, color='#8888FF', linestyle='--', lw=1.5, label='ruido branco')
axes[2].set_xticks(range(len(nomes)))
axes[2].set_xticklabels(nomes, rotation=20, ha='right', fontsize=7)
axes[2].set_title("coh_alpha_eq (1 − H_eq)", color='white')
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig("alphaphi_eco_foton.png", dpi=120, bbox_inches="tight",
            facecolor='#0d1117')
plt.close()

print("\nalpha-phi | eco foton textual | alphaphi_eco_foton.png")
print("=" * 65)
