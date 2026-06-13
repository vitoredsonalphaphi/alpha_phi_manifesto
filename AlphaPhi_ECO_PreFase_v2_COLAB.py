# AlphaPhi_ECO_PreFase_v2_COLAB.py
# Vitor Edson Delavi · Florianópolis · 2026
#
# ECO-RESSONANTE MULTI-ESPECTRAL — PRÉ-FUNÇÃO ADAPTATIVA v2
#
# Extensão do eco-ressonante estendido (PreFase v1) com:
#
# 1. OBSERVAÇÃO MULTI-ESPECTRAL: cada fase sondada em 3 espectros:
#    - Espectro A (energia)  : H_alpha da distribuição de |ativações|
#    - Espectro B (ativação) : fração de neurônios ativos (pós-ReLU)
#    - Espectro C (variância): H_alpha da variância inter-amostras
#    Coerência composta: wm × coh_A + wn × mean(coh_B, coh_C)  [φ-ponderada]
#
# 2. META-COERÊNCIA: eco mede a coerência da própria observação
#    H_alpha dos scores entre fases → concentrado = em fase → age
#                                    disperso    = incerto → expande
#
# 3. SONDAGEM ADAPTATIVA: N_SONDA_MIN=3 .. N_SONDA_MAX=10
#    para quando meta_coh >= META_COH_LIMIAR OU esgota ciclos
#
# 4. LIMIAR CORRIGIDO: LIMIAR_SUBSTRATO = 1e-3
#
# CONFIGS:
#   AA — baseline (sem semente)
#   BB — semente Fase 3 fixo (espaço 137 — v1)
#   CC — pré-função multi-espectral com meta-coerência adaptativa
#
# HIPÓTESE: meta_coh baixa sustentada = substrato inadequado (diagnóstico)
#           meta_coh alta rápida      = substrato adequado, fase localizada

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "--upgrade", "huggingface_hub", "datasets", "scipy"],
               check=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI       = (1 + np.sqrt(5)) / 2
ALPHA     = 1 / 137.035999084
LOG_ALPHA = np.log(1.0 / ALPHA)
N_BASE    = int(round(1.0 / ALPHA))

N_JANELAS = 5
N_CHARS   = 27

wm = 1.0 / PHI        # ≈ 0.618 — peso memória / espectro primário
wn = 1.0 - 1.0 / PHI  # ≈ 0.382

N_SONDA_MIN     = 3
N_SONDA_MAX     = 10
META_COH_LIMIAR = 0.70
LIMIAR_SUBSTRATO = 1e-3

N_EPOCHS = 30
BATCH    = 128
LR_BASE  = 0.01

print("=" * 65)
print(f"phi              = {PHI:.10f}")
print(f"alpha            = {ALPHA:.10f}  (1/137)")
print(f"log(137)         = {LOG_ALPHA:.6f}")
print(f"phi^3            = {PHI**3:.6f}  (limiar campo)")
print(f"N_sonda_min/max  = {N_SONDA_MIN}/{N_SONDA_MAX}")
print(f"meta_coh_limiar  = {META_COH_LIMIAR}")
print(f"limiar_substrato = {LIMIAR_SUBSTRATO}")
print("=" * 65)
print("ECO MULTI-ESPECTRAL — PRÉ-FUNÇÃO ADAPTATIVA v2")
print("3 espectros por fase · meta-coerência · sondagem adaptativa")
print("=" * 65)

# ─── Substrato ────────────────────────────────────────────────────────────────
def char_freq(text):
    counts = np.zeros(N_CHARS)
    for c in text.lower():
        if 'a' <= c <= 'z':
            counts[ord(c) - ord('a')] += 1
        elif c == ' ':
            counts[26] += 1
    total = counts.sum()
    return counts / total if total > 0 else np.ones(N_CHARS) / N_CHARS

def texto_para_janelas(text):
    n = max(len(text), 1)
    step = n / N_JANELAS
    freqs = np.zeros((N_JANELAS, N_CHARS))
    for j in range(N_JANELAS):
        start = int(j * step)
        end   = max(int((j + 1) * step), start + 1)
        freqs[j] = char_freq(text[start:end])
    return freqs.ravel()

# ─── Multi-espectral ─────────────────────────────────────────────────────────
def espectro_energia(h):
    """Espectro A: H_alpha da distribuição de |ativações| por amostra."""
    h_abs  = np.abs(h) + 1e-10
    h_norm = h_abs / h_abs.sum(axis=1, keepdims=True)
    H      = -np.sum(h_norm * np.log(h_norm + 1e-15), axis=1)
    H_max  = max(np.log(h.shape[1]), LOG_ALPHA)
    H_alpha = np.clip(H / H_max, 0.0, 1.0)
    return float(H_alpha.mean()), float(1.0 - H_alpha.mean())  # H, coh

def espectro_ativacao(h):
    """Espectro B: fração de neurônios ativos (pós-ReLU)."""
    frac_ativa = float((h > 0).mean())
    return 1.0 - frac_ativa, frac_ativa  # H, coh

def espectro_variancia(h):
    """Espectro C: H_alpha da variância inter-amostras por neurônio."""
    var_inter = h.var(axis=0) + 1e-10          # (n_units,)
    v_norm    = var_inter / var_inter.sum()
    H_v       = float(-np.sum(v_norm * np.log(v_norm + 1e-15)))
    H_max_v   = max(np.log(h.shape[1]), LOG_ALPHA)
    H_alpha_v = np.clip(H_v / H_max_v, 0.0, 1.0)
    return H_alpha_v, float(1.0 - H_alpha_v)  # H, coh

def disc_linear(h, y):
    mask0, mask1 = y < 0.5, y >= 0.5
    if mask0.sum() == 0 or mask1.sum() == 0:
        return 0.0
    mu0, mu1 = h[mask0].mean(axis=0), h[mask1].mean(axis=0)
    return float(np.mean((mu1 - mu0) ** 2))

def disc_angular(h, y):
    mask0, mask1 = y < 0.5, y >= 0.5
    if mask0.sum() == 0 or mask1.sum() == 0:
        return 0.0
    mu0 = h[mask0].mean(axis=0)
    mu1 = h[mask1].mean(axis=0)
    n0  = np.linalg.norm(mu0)
    n1  = np.linalg.norm(mu1)
    if n0 < 1e-10 or n1 < 1e-10:
        return 0.0
    cos = np.dot(mu0, mu1) / (n0 * n1)
    return float(1.0 - abs(cos))   # 0=idênticos, 1=perpendiculares

def sonda_fase(h, y):
    """Sondagem multi-espectral de uma fase. Retorna dict completo."""
    H_A, coh_A = espectro_energia(h)
    H_B, coh_B = espectro_ativacao(h)
    H_C, coh_C = espectro_variancia(h)

    coh_composta  = wm * coh_A + wn * (coh_B + coh_C) / 2.0
    disc_lin      = disc_linear(h, y)
    disc_ang      = disc_angular(h, y)
    disc_composta = (disc_lin + disc_ang) / 2.0
    score         = coh_composta * disc_composta

    return {
        'H_A': H_A, 'coh_A': coh_A,
        'H_B': H_B, 'coh_B': coh_B,
        'H_C': H_C, 'coh_C': coh_C,
        'coh_composta':  coh_composta,
        'disc_linear':   disc_lin,
        'disc_angular':  disc_ang,
        'disc_composta': disc_composta,
        'score':         score,
    }

def meta_coerencia(scores):
    """H_alpha da distribuição de scores entre fases — eco mede a si mesmo."""
    s = np.array(scores, dtype=float) + 1e-10
    s_norm = s / s.sum()
    H_meta = float(-np.sum(s_norm * np.log(s_norm + 1e-15)))
    H_max  = np.log(len(s))
    H_meta_alpha = float(np.clip(H_meta / H_max if H_max > 0 else 0.0, 0.0, 1.0))
    return H_meta_alpha, float(1.0 - H_meta_alpha)  # H, coh

# ─── Rede ─────────────────────────────────────────────────────────────────────
def relu(x):    return np.maximum(0.0, x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

class Rede:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.dims = [N_JANELAS * N_CHARS, 55, 89, 137, 1]
        self.W = [np.random.randn(self.dims[i], self.dims[i+1])
                  * np.sqrt(2.0 / self.dims[i])
                  for i in range(len(self.dims)-1)]
        self.b = [np.zeros(self.dims[i+1]) for i in range(len(self.dims)-1)]

    def forward_sonda(self, X, y):
        """Forward com sondagem multi-espectral de cada fase."""
        h, perfil = X, []
        for i, (W, b) in enumerate(zip(self.W[:-1], self.b[:-1])):
            h = relu(h @ W + b)
            p = sonda_fase(h, y)
            p['fase'] = i + 1
            p['dim']  = self.dims[i + 1]
            perfil.append(p)
        return perfil

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
        for i in range(len(self.W)-1, -1, -1):
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

    def coh_na_fase(self, fase_idx, X_batch):
        h = X_batch
        for i, (W, b) in enumerate(zip(self.W[:-1], self.b[:-1])):
            h = relu(h @ W + b)
            if (i + 1) == fase_idx:
                h_abs = np.abs(h)
                piso  = ALPHA * h_abs.max(axis=1, keepdims=True)
                h_cp  = np.where(h > 0, np.maximum(h, piso), h)
                _, coh = espectro_energia(h_cp)
                return coh
        return 0.0

# ─── Dataset ──────────────────────────────────────────────────────────────────
print("\nCarregando SST-2...")
_ds = None
for _name, _cfg, _field in [
    ("stanfordnlp/sst2", None,   "sentence"),
    ("SetFit/sst2",      None,   "text"),
    ("nyu-mll/glue",     "sst2", "sentence"),
    ("glue",             "sst2", "sentence"),
]:
    try:
        _ds = load_dataset(_name) if _cfg is None else load_dataset(_name, _cfg)
        _TEXT_FIELD = _field
        print(f"  Dataset: {_name}  campo='{_field}'")
        break
    except Exception as _e:
        print(f"  {_name}: falhou ({type(_e).__name__})")
if _ds is None:
    raise RuntimeError("Não foi possível carregar SST-2")

np.random.seed(42)
idx_tr  = np.random.choice(len(_ds['train']),      3000, replace=False)
idx_val = np.random.choice(len(_ds['validation']),  872, replace=False)

texts_tr  = [_ds['train'][int(i)][_TEXT_FIELD]      for i in idx_tr]
texts_val = [_ds['validation'][int(i)][_TEXT_FIELD] for i in idx_val]
y_tr  = np.array([_ds['train'][int(i)]['label']      for i in idx_tr],  dtype=float)
y_val = np.array([_ds['validation'][int(i)]['label'] for i in idx_val], dtype=float)

print("Pré-computando janelas de caracteres...")
X_tr  = np.array([texto_para_janelas(t) for t in texts_tr])
X_val = np.array([texto_para_janelas(t) for t in texts_val])
print(f"Treino: {X_tr.shape}  Val: {X_val.shape}")

# ─── Experimento ──────────────────────────────────────────────────────────────
N_SEEDS = 10
rng     = np.random.default_rng(42)
SEEDS   = [int(x) for x in rng.integers(1, 2**31, N_SEEDS)]
print(f"\nSeeds ({N_SEEDS}): {SEEDS[:3]}...{SEEDS[-1]}")

CONFIGS = {
    "AA — baseline       (sem semente)         ": {"modo": "aa"},
    "BB — semente Fase 3 (espaço 137 fixo)     ": {"modo": "bb"},
    "CC — multi-espectral (meta-coh adaptativo)": {"modo": "cc"},
}

resultados = {}

for cfg_nome, cfg in CONFIGS.items():
    print(f"\n{'='*65}\n{cfg_nome}")
    accs, fases_otimas, perfis_finais = [], [], []
    meta_cohs_finais, ciclos_sonda = [], []

    for seed in SEEDS:
        rede = Rede(seed=seed)
        np.random.seed(seed)

        fase_otima    = 3
        sondagem_ok   = False
        scores_ema    = np.zeros(rede.dims[1:-1].__len__() if False else 3)
        beta, bm      = 1.0, 1.0
        meta_coh_hist = []
        n_ciclos_sonda = 0
        perfil_final  = None

        for epoch in range(N_EPOCHS):
            perm      = np.random.permutation(len(X_tr))
            n_batches = len(X_tr) // BATCH

            # ── Sondagem multi-espectral (CC) ─────────────────────────────
            if cfg["modo"] == "cc" and not sondagem_ok and epoch < N_SONDA_MAX:
                Xb = X_tr[perm[:BATCH]]
                yb = y_tr[perm[:BATCH]]
                perfil = rede.forward_sonda(Xb, yb)
                scores = np.array([p['score'] for p in perfil])
                scores_ema = ALPHA * scores + (1.0 - ALPHA) * scores_ema

                H_meta, meta_coh = meta_coerencia(scores_ema)
                meta_coh_hist.append(meta_coh)
                n_ciclos_sonda = epoch + 1

                parar = (epoch >= N_SONDA_MIN - 1 and meta_coh >= META_COH_LIMIAR)
                ultimo = (epoch == N_SONDA_MAX - 1)

                if parar or ultimo:
                    fase_otima   = int(np.argmax(scores_ema)) + 1
                    sondagem_ok  = True
                    perfil_final = perfil

            # ── LR ────────────────────────────────────────────────────────
            lr_epoch = LR_BASE * max(beta, 0.1)

            # ── Batches ───────────────────────────────────────────────────
            for b_idx in range(n_batches):
                idx_b = perm[b_idx * BATCH:(b_idx + 1) * BATCH]
                rede.forward(X_tr[idx_b])
                rede.backward(X_tr[idx_b], y_tr[idx_b], lr_epoch)

            # ── Semente ───────────────────────────────────────────────────
            if cfg["modo"] == "bb":
                coh  = rede.coh_na_fase(3, X_tr[:BATCH])
                ba   = float(PHI ** (3.0 * coh))
                beta = wn * ba + wm * bm; bm = beta
            elif cfg["modo"] == "cc":
                f    = fase_otima
                coh  = rede.coh_na_fase(f, X_tr[:BATCH])
                ba   = float(PHI ** (3.0 * coh))
                beta = wn * ba + wm * bm; bm = beta

        # ── Resumo por seed ───────────────────────────────────────────────
        acc = float(np.mean((rede.predict(X_val) > 0.5) == y_val))
        accs.append(acc)
        fases_otimas.append(fase_otima)

        if perfil_final:
            perfis_finais.append(perfil_final)
            max_score = max(p['score'] for p in perfil_final)
            adequado  = max_score > LIMIAR_SUBSTRATO
            meta_coh_final = meta_coh_hist[-1] if meta_coh_hist else 0.0
            meta_cohs_finais.append(meta_coh_final)
            ciclos_sonda.append(n_ciclos_sonda)
            tag = f"F{fase_otima}({rede.dims[fase_otima]})  meta_coh={meta_coh_final:.3f}"
            tag += f"  ciclos={n_ciclos_sonda}  {'OK' if adequado else 'INADEQUADO'}"
        else:
            tag = ""
        print(f"  seed {seed} → acc={acc:.4f}  {tag}")

    from collections import Counter
    resultados[cfg_nome] = {
        "acc":    np.mean(accs), "std": np.std(accs),
        "accs":   accs, "fases": fases_otimas,
        "perfis": perfis_finais, "meta_cohs": meta_cohs_finais,
        "ciclos": ciclos_sonda,
    }
    dist = Counter(fases_otimas)
    print(f"  → acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}"
          f"  fases: {dict(dist)}"
          f"  meta_coh_med: {np.mean(meta_cohs_finais):.3f}" if meta_cohs_finais else "")

# ─── Síntese ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SÍNTESE — ECO MULTI-ESPECTRAL v2")
print("=" * 65)

configs_list = list(resultados.keys())
aa_accs = resultados[configs_list[0]]["accs"]
aa_acc  = resultados[configs_list[0]]["acc"]

print(f"\n{'Config':<50} {'Acc':>7} {'Std':>6} {'Delta':>7}")
print("-" * 75)
for nome, r in resultados.items():
    print(f"{nome:<50} {r['acc']:>7.4f} {r['std']:>6.4f} {r['acc']-aa_acc:>+7.4f}")

print()
for nome in configs_list[1:]:
    _, p = stats.ttest_rel(resultados[nome]["accs"], aa_accs)
    print(f"Teste t: {nome.strip()[:45]} vs AA  p={p:.4f}")

# Perfil multi-espectral (CC)
cc_perfis = resultados[configs_list[2]]["perfis"]
cc_meta   = resultados[configs_list[2]]["meta_cohs"]
cc_ciclos = resultados[configs_list[2]]["ciclos"]
if cc_perfis:
    print(f"\nMeta-coerência média (CC): {np.mean(cc_meta):.4f}"
          f"  ±{np.std(cc_meta):.4f}")
    print(f"Ciclos de sondagem médio:  {np.mean(cc_ciclos):.1f}"
          f"  (min={min(cc_ciclos)}, max={max(cc_ciclos)})")

    print(f"\nPerfil multi-espectral médio (CC):")
    print(f"  {'F':<3} {'dim':<5} {'coh_A':>7} {'coh_B':>7} {'coh_C':>7}"
          f" {'coh_comp':>9} {'disc_lin':>10} {'disc_ang':>10} {'score':>10}")
    n_fases = len(cc_perfis[0])
    scores_med = []
    for fi in range(n_fases):
        cA   = np.mean([p[fi]['coh_A']        for p in cc_perfis])
        cB   = np.mean([p[fi]['coh_B']        for p in cc_perfis])
        cC   = np.mean([p[fi]['coh_C']        for p in cc_perfis])
        cc   = np.mean([p[fi]['coh_composta'] for p in cc_perfis])
        dl   = np.mean([p[fi]['disc_linear']  for p in cc_perfis])
        da   = np.mean([p[fi]['disc_angular'] for p in cc_perfis])
        sc   = np.mean([p[fi]['score']        for p in cc_perfis])
        dim  = cc_perfis[0][fi]['dim']
        scores_med.append(sc)
        print(f"  F{fi+1:<2} {dim:<5} {cA:>7.4f} {cB:>7.4f} {cC:>7.4f}"
              f" {cc:>9.4f} {dl:>10.6f} {da:>10.6f} {sc:>10.6f}")

    fase_otima_med = int(np.argmax(scores_med)) + 1
    max_score_med  = max(scores_med)
    print(f"\n  → Fase ótima média: Fase {fase_otima_med}"
          f"  max_score={max_score_med:.6f}")
    if max_score_med > LIMIAR_SUBSTRATO:
        print(f"  → Diagnóstico: substrato ADEQUADO")
    else:
        print(f"  → Diagnóstico: substrato INADEQUADO (max_score < {LIMIAR_SUBSTRATO})")
        print(f"     α não encontra residência — sinal semântico ausente")
    if np.mean(cc_meta) < META_COH_LIMIAR:
        print(f"  → Meta-coerência baixa ({np.mean(cc_meta):.3f} < {META_COH_LIMIAR})")
        print(f"     confirma: observação do eco não convergiu para nenhuma fase")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0d1117')
fig.suptitle(
    "ECO MULTI-ESPECTRAL v2 — PRÉ-FUNÇÃO ADAPTATIVA\n"
    "3 espectros por fase · meta-coerência · sondagem adaptativa",
    fontsize=11, color='white')

for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

nomes  = [k.strip()[:32] for k in resultados.keys()]
accs_v = [v["acc"] for v in resultados.values()]
cores  = ["#8B949E", "#00BFFF", "#FF9944"]

axes[0].bar(range(len(nomes)), accs_v, color=cores)
axes[0].set_xticks(range(len(nomes)))
axes[0].set_xticklabels(nomes, rotation=15, ha='right', fontsize=7)
axes[0].set_title("Acurácia SST-2", color='white')
axes[0].set_ylim(0.4, 0.9)
axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='random')
axes[0].legend(fontsize=8)

if cc_perfis:
    fases_labels = [f"F{p['fase']}\n({p['dim']})" for p in cc_perfis[0]]
    cores_fase = ["#FF9944" if i == int(np.argmax(scores_med))
                  else "#30363d" for i in range(len(scores_med))]
    axes[1].bar(range(len(fases_labels)), scores_med, color=cores_fase)
    axes[1].set_xticks(range(len(fases_labels)))
    axes[1].set_xticklabels(fases_labels, fontsize=9, color='white')
    axes[1].set_title("Score por fase (CC v2)", color='white')
    axes[1].axhline(LIMIAR_SUBSTRATO, color='red', linestyle='--',
                    alpha=0.7, label=f'limiar={LIMIAR_SUBSTRATO}')
    axes[1].legend(fontsize=8)

    # Espectros sobrepostos por fase
    x = np.arange(len(cc_perfis[0]))
    cA_v = [np.mean([p[fi]['coh_A'] for p in cc_perfis]) for fi in range(n_fases)]
    cB_v = [np.mean([p[fi]['coh_B'] for p in cc_perfis]) for fi in range(n_fases)]
    cC_v = [np.mean([p[fi]['coh_C'] for p in cc_perfis]) for fi in range(n_fases)]
    w = 0.25
    axes[2].bar(x - w, cA_v, w, color='#00BFFF', label='coh_A (energia)', alpha=0.9)
    axes[2].bar(x,     cB_v, w, color='#00FF88', label='coh_B (ativação)', alpha=0.9)
    axes[2].bar(x + w, cC_v, w, color='#FF9944', label='coh_C (variância)', alpha=0.9)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(fases_labels, fontsize=9, color='white')
    axes[2].set_title("3 Espectros por fase (CC v2)", color='white')
    axes[2].set_ylabel("coerência", color='white')
    axes[2].legend(fontsize=7)
    meta_coh_med = np.mean(cc_meta) if cc_meta else 0.0
    axes[2].set_xlabel(f"meta_coh = {meta_coh_med:.3f}", color='white', fontsize=9)

plt.tight_layout()
plt.savefig("alphaphi_eco_multiespectral_v2.png", dpi=120, bbox_inches="tight",
            facecolor='#0d1117')
plt.close()

print("\nalpha-phi | eco multi-espectral v2 | alphaphi_eco_multiespectral_v2.png")
print("=" * 65)
