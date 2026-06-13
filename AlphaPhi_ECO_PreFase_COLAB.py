# AlphaPhi_ECO_PreFase_COLAB.py
# Vitor Edson Delavi · Florianópolis · 2026
#
# PRÉ-FUNÇÃO DE SONDAGEM DE FASE — ECO-RESSONANTE ADAPTATIVO
#
# A pré-função do eco-ressonante já observa o substrato antes de agir.
# Esta extensão sonda CADA FASE da rede (todas as camadas intermediárias),
# mede H_alpha e discriminabilidade de classes por fase,
# e identifica onde α pode melhor residir para aquele substrato.
#
# Critério de residência:
#   score_f = coh_alpha_f × var_entre_classes_f
#   (coerência × discriminabilidade — onde α distingue e se estabiliza)
#
# FASES da arquitetura [135 → 55 → 89 → 137 → 1]:
#   Fase 0 (substrato)  : texto → janelas de caracteres (entrada bruta)
#   Fase 1 (compressão) : 135 → 55   (primeira bifurcação, Fibonacci)
#   Fase 2 (expansão)   : 55  → 89   (segunda bifurcação, Fibonacci)
#   Fase 3 (nativo α)   : 89  → 137  (espaço 1/α — régua nativa)
#
# CONFIGS:
#   AA — baseline          (sem semente)
#   BB — semente Fase 3    (espaço 137 fixo — método anterior)
#   CC — pré-função        (N_SONDA ciclos de sondagem → fase_otima → eco ali)
#
# DIAGNÓSTICO:
#   Se max(score_fase) < LIMIAR_SUBSTRATO → substrato inadequado
#   (α não encontra residência — sinal semântico ausente em todas as fases)
#
# HIPÓTESE: CC identifica corretamente a fase de maior acoplamento α,
#           e melhora eficiência de convergência sobre BB quando a fase
#           ótima difere da Fase 3.

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
LOG_ALPHA = np.log(1.0 / ALPHA)      # log(137) ≈ 4.920
N_BASE    = int(round(1.0 / ALPHA))  # 137
LIMIAR    = 0.99 * PHI**3

N_JANELAS = 5
N_CHARS   = 27   # a-z + espaço

wm = 1.0 / PHI           # peso memória  ≈ 0.618
wn = 1.0 - 1.0 / PHI    # peso novo     ≈ 0.382

N_SONDA  = 5    # ciclos de sondagem antes de fixar fase_otima
N_EPOCHS = 30
BATCH    = 128
LR_BASE  = 0.01

LIMIAR_SUBSTRATO = 1e-5  # score mínimo para substrato ser considerado adequado

print("=" * 65)
print(f"phi              = {PHI:.10f}")
print(f"alpha            = {ALPHA:.10f}  (1/137)")
print(f"log(137)         = {LOG_ALPHA:.6f}  (régua de entropia)")
print(f"phi^3            = {PHI**3:.6f}  (limiar campo)")
print(f"N_sonda          = {N_SONDA}  (ciclos de sondagem)")
print(f"limiar_substrato = {LIMIAR_SUBSTRATO}  (diagnóstico adequação)")
print("=" * 65)
print("PRÉ-FUNÇÃO DE SONDAGEM DE FASE — ECO-RESSONANTE ADAPTATIVO")
print("Arquitetura: [135, 55, 89, 137, 1]  (5×27 → Fibonacci → α-nativo)")
print("=" * 65)

# ─── Substrato — janelas de caracteres ───────────────────────────────────────
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

# ─── Medição de fase ─────────────────────────────────────────────────────────
def H_alpha_ativacoes(h):
    """
    Entropia normalizada das ativações de uma camada.
    h: (batch, n_units)
    Trata cada unidade como bin de energia: h_norm = |h| / sum(|h|)
    Retorna: (H_alpha_mean, H_alpha_std)
    """
    h_abs  = np.abs(h) + 1e-10
    h_norm = h_abs / h_abs.sum(axis=1, keepdims=True)
    H      = -np.sum(h_norm * np.log(h_norm + 1e-15), axis=1)
    # escala: log(n_units) é o máximo teórico; LOG_ALPHA é a régua de α
    H_max   = max(np.log(h.shape[1]), LOG_ALPHA)
    H_alpha = np.clip(H / H_max, 0.0, 1.0)
    return float(H_alpha.mean()), float(H_alpha.std())

def var_entre_classes(h, y):
    """
    Discriminabilidade: variância entre médias das classes positivo/negativo.
    h: (batch, n_units), y: (batch,) binário
    Retorna: float — maior = maior distinção semântica nesta fase
    """
    mask0, mask1 = y < 0.5, y >= 0.5
    if mask0.sum() == 0 or mask1.sum() == 0:
        return 0.0
    mu0 = h[mask0].mean(axis=0)
    mu1 = h[mask1].mean(axis=0)
    return float(np.mean((mu1 - mu0) ** 2))

def coh_na_fase(h):
    """Coerência α (1 - H_alpha) das ativações."""
    H_mean, _ = H_alpha_ativacoes(h)
    return 1.0 - H_mean

# ─── Rede ─────────────────────────────────────────────────────────────────────
def relu(x):    return np.maximum(0.0, x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

class RedeAdaptativa:
    """Rede FC com pré-função de sondagem de fase."""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.dims = [N_JANELAS * N_CHARS, 55, 89, 137, 1]
        self.n_fases = len(self.dims) - 2  # fases com ativação: 1, 2, 3
        self.W = [np.random.randn(self.dims[i], self.dims[i+1])
                  * np.sqrt(2.0 / self.dims[i])
                  for i in range(len(self.dims)-1)]
        self.b = [np.zeros(self.dims[i+1]) for i in range(len(self.dims)-1)]

    def forward_sonda(self, X, y):
        """
        Forward com sondagem de cada fase intermediária.
        Retorna perfil: lista de dicts por fase com H_alpha, disc, score.
        """
        h      = X
        perfil = []
        for i, (W, b) in enumerate(zip(self.W[:-1], self.b[:-1])):
            h = relu(h @ W + b)
            H_mean, H_std = H_alpha_ativacoes(h)
            disc  = var_entre_classes(h, y)
            coh   = 1.0 - H_mean
            score = coh * disc
            perfil.append({
                'fase':    i + 1,
                'dim':     self.dims[i + 1],
                'H_alpha': H_mean,
                'H_std':   H_std,
                'coh':     coh,
                'disc':    disc,
                'score':   score,
            })
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

    def medir_coh_fase(self, fase_idx, X_batch):
        """
        Propaga até fase_idx e mede coerência α das ativações.
        Aplica piso α (floor = α × max_ativacao) antes de medir.
        fase_idx: 1-based (1=55, 2=89, 3=137)
        """
        h = X_batch
        for i, (W, b) in enumerate(zip(self.W[:-1], self.b[:-1])):
            h = relu(h @ W + b)
            if (i + 1) == fase_idx:
                # piso α: nenhuma ativação positiva abaixo de α × máximo
                h_abs = np.abs(h)
                piso  = ALPHA * h_abs.max(axis=1, keepdims=True)
                h_com_piso = np.where(h > 0, np.maximum(h, piso), h)
                return coh_na_fase(h_com_piso)
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
print(f"Balance classes — treino: {y_tr.mean():.3f}  val: {y_val.mean():.3f}")

# ─── Configurações ────────────────────────────────────────────────────────────
N_SEEDS = 10
rng     = np.random.default_rng(42)
SEEDS   = [int(x) for x in rng.integers(1, 2**31, N_SEEDS)]
print(f"\nSeeds ({N_SEEDS}): {SEEDS[:3]}...{SEEDS[-1]}")

CONFIGS = {
    "AA — baseline       (sem semente)      ": {"modo": "aa"},
    "BB — semente Fase 3 (espaço 137 fixo)  ": {"modo": "bb"},
    "CC — pré-função     (fase adaptativa)  ": {"modo": "cc"},
}

# ─── Loop principal ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
resultados = {}

for cfg_nome, cfg in CONFIGS.items():
    print(f"\n{cfg_nome}")
    accs          = []
    fases_otimas  = []
    perfis_sonda  = []
    diagnosticos  = []

    for seed in SEEDS:
        rede = RedeAdaptativa(seed=seed)
        np.random.seed(seed)

        fase_otima   = 3       # default: Fase 3 (espaço 137)
        perfil_final = None
        scores_ema   = np.zeros(rede.n_fases)
        beta         = 1.0
        bm           = 1.0
        sondagem_ok  = False

        for epoch in range(N_EPOCHS):
            perm      = np.random.permutation(len(X_tr))
            n_batches = len(X_tr) // BATCH

            # ── Sondagem (CC, primeiros N_SONDA ciclos) ───────────────────
            if cfg["modo"] == "cc" and epoch < N_SONDA:
                Xb = X_tr[perm[:BATCH]]
                yb = y_tr[perm[:BATCH]]
                perfil = rede.forward_sonda(Xb, yb)
                scores = np.array([p['score'] for p in perfil])
                scores_ema = ALPHA * scores + (1.0 - ALPHA) * scores_ema

                if epoch == N_SONDA - 1:
                    fase_otima   = int(np.argmax(scores_ema)) + 1
                    perfil_final = perfil
                    sondagem_ok  = True

            # ── LR com beta ───────────────────────────────────────────────
            lr_epoch = LR_BASE * max(beta, 0.1)

            # ── Batches ───────────────────────────────────────────────────
            for b_idx in range(n_batches):
                idx_b = perm[b_idx * BATCH:(b_idx + 1) * BATCH]
                Xb, yb = X_tr[idx_b], y_tr[idx_b]
                rede.forward(Xb)
                rede.backward(Xb, yb, lr_epoch)

            # ── Semente após batches (BB: Fase 3 fixa; CC: fase_otima) ────
            if cfg["modo"] == "bb":
                coh  = rede.medir_coh_fase(3, X_tr[:BATCH])
                ba   = float(PHI ** (3.0 * coh))
                beta = wn * ba + wm * bm
                bm   = beta

            elif cfg["modo"] == "cc":
                f    = fase_otima
                coh  = rede.medir_coh_fase(f, X_tr[:BATCH])
                ba   = float(PHI ** (3.0 * coh))
                beta = wn * ba + wm * bm
                bm   = beta

        # ── Resumo por seed ───────────────────────────────────────────────
        acc = float(np.mean((rede.predict(X_val) > 0.5) == y_val))
        accs.append(acc)
        fases_otimas.append(fase_otima)

        if perfil_final is not None:
            perfis_sonda.append(perfil_final)
            max_score = max(p['score'] for p in perfil_final)
            adequado  = max_score > LIMIAR_SUBSTRATO
            diagnosticos.append(adequado)
            tag_diag = "adequado" if adequado else "inadequado"
            tag_fase = f"F{fase_otima}({rede.dims[fase_otima]})"
            print(f"  seed {seed} → acc={acc:.4f}  fase={tag_fase}"
                  f"  max_score={max_score:.6f}  substrato={tag_diag}")
        else:
            print(f"  seed {seed} → acc={acc:.4f}")

    resultados[cfg_nome] = {
        "acc":         np.mean(accs),
        "std":         np.std(accs),
        "accs":        accs,
        "fases":       fases_otimas,
        "perfis":      perfis_sonda,
        "diagnosticos": diagnosticos,
    }

    from collections import Counter
    if fases_otimas and max(fases_otimas) > 0:
        dist = Counter(fases_otimas)
        print(f"  → acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}"
              f"  fases: {dict(dist)}")
    else:
        print(f"  → acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

# ─── Síntese ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SÍNTESE — PRÉ-FUNÇÃO DE SONDAGEM DE FASE")
print("=" * 65)

configs_list = list(resultados.keys())
aa_accs = resultados[configs_list[0]]["accs"]
aa_acc  = resultados[configs_list[0]]["acc"]

print(f"\n{'Config':<47} {'Acc':>7} {'Std':>6} {'Delta':>7}")
print("-" * 72)
for nome, r in resultados.items():
    delta = r["acc"] - aa_acc
    print(f"{nome:<47} {r['acc']:>7.4f} {r['std']:>6.4f} {delta:>+7.4f}")

print()
for nome in configs_list[1:]:
    _, p = stats.ttest_rel(resultados[nome]["accs"], aa_accs)
    print(f"Teste t: {nome.strip()[:40]} vs AA  p={p:.4f}")

# Perfil de sondagem (CC)
cc_perfis = resultados[configs_list[2]]["perfis"]
if cc_perfis:
    print("\nPerfil de sondagem médio — CC (pré-função):")
    print(f"  {'Fase':<6} {'Dim':<5} {'H_alpha':>8} {'coh':>7}"
          f" {'disc':>12} {'score':>10}")
    n_fases = len(cc_perfis[0])
    scores_medios = []
    for fi in range(n_fases):
        H_v    = [p[fi]['H_alpha'] for p in cc_perfis]
        coh_v  = [p[fi]['coh']     for p in cc_perfis]
        disc_v = [p[fi]['disc']    for p in cc_perfis]
        sc_v   = [p[fi]['score']   for p in cc_perfis]
        dim    = cc_perfis[0][fi]['dim']
        sm     = float(np.mean(sc_v))
        scores_medios.append(sm)
        marker = " ←" if fi == int(np.argmax(scores_medios[:fi+1]
                                             if fi == n_fases-1
                                             else scores_medios)) else ""
        print(f"  Fase {fi+1:<2}  {dim:<5}"
              f"  {np.mean(H_v):>8.4f}"
              f"  {np.mean(coh_v):>7.4f}"
              f"  {np.mean(disc_v):>12.8f}"
              f"  {sm:>10.8f}{marker}")

    # Refaz marcação de fase ótima
    fase_otima_media = int(np.argmax(scores_medios)) + 1
    dim_otima = cc_perfis[0][fase_otima_media - 1]['dim']
    print(f"\n  → Fase ótima (média): Fase {fase_otima_media} ({dim_otima} units)")

    # Diagnóstico de substrato
    max_score = max(scores_medios)
    if max_score > LIMIAR_SUBSTRATO:
        print(f"  → Diagnóstico: substrato ADEQUADO  (max_score={max_score:.6f})")
    else:
        print(f"  → Diagnóstico: substrato INADEQUADO (max_score={max_score:.8f})")
        print(f"     α não encontra residência — sinal semântico ausente")
        print(f"     em todas as fases. Substrato incompatível com tarefa.")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0d1117')
fig.suptitle(
    "PRÉ-FUNÇÃO DE SONDAGEM DE FASE — ECO-RESSONANTE ADAPTATIVO\n"
    "score_f = coh_alpha_f × var_entre_classes_f  |  α reside no máximo",
    fontsize=11, color='white')

for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

nomes  = [k.strip()[:32] for k in resultados.keys()]
accs_v = [v["acc"] for v in resultados.values()]
cores  = ["#8B949E", "#00BFFF", "#FF9944"]

# Plot 1: acurácia
axes[0].bar(range(len(nomes)), accs_v, color=cores)
axes[0].set_xticks(range(len(nomes)))
axes[0].set_xticklabels(nomes, rotation=15, ha='right', fontsize=7)
axes[0].set_title("Acurácia SST-2", color='white')
axes[0].set_ylim(0.4, 0.9)
axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='random')
axes[0].legend(fontsize=8)

# Plot 2: scores por fase (CC)
if cc_perfis:
    fases_labels = [f"Fase {p['fase']}\n({p['dim']})" for p in cc_perfis[0]]
    cores_fase   = ["#FF9944" if i == int(np.argmax(scores_medios))
                    else "#30363d" for i in range(len(scores_medios))]
    axes[1].bar(range(len(fases_labels)), scores_medios, color=cores_fase)
    axes[1].set_xticks(range(len(fases_labels)))
    axes[1].set_xticklabels(fases_labels, fontsize=9, color='white')
    axes[1].set_title("Score por fase (CC) — coh × disc", color='white')
    axes[1].set_ylabel("score", color='white')
    axes[1].axhline(LIMIAR_SUBSTRATO, color='red', linestyle='--',
                    alpha=0.7, label=f'limiar={LIMIAR_SUBSTRATO}')
    axes[1].legend(fontsize=8)

# Plot 3: H_alpha e disc por fase
if cc_perfis:
    x = np.arange(len(cc_perfis[0]))
    H_medios   = [np.mean([p[fi]['H_alpha'] for p in cc_perfis])
                  for fi in range(len(cc_perfis[0]))]
    disc_medios= [np.mean([p[fi]['disc']    for p in cc_perfis])
                  for fi in range(len(cc_perfis[0]))]
    disc_norm  = np.array(disc_medios) / (max(disc_medios) + 1e-10)

    ax2 = axes[2]
    ax2b = ax2.twinx()
    ax2b.set_facecolor('#161b22')
    ax2b.tick_params(colors='#00FF88', labelsize=8)

    ax2.bar(x - 0.2, H_medios,    0.35, color='#00BFFF', label='H_alpha', alpha=0.8)
    ax2b.bar(x + 0.2, disc_norm,  0.35, color='#00FF88', label='disc (norm)', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fases_labels, fontsize=9, color='white')
    ax2.set_title("H_alpha e Discriminabilidade por fase", color='white')
    ax2.set_ylabel("H_alpha", color='#00BFFF')
    ax2b.set_ylabel("disc (normalizado)", color='#00FF88')
    ax2.legend(loc='upper left', fontsize=8)
    ax2b.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("alphaphi_prefuncao_fase.png", dpi=120, bbox_inches="tight",
            facecolor='#0d1117')
plt.close()

print("\nalpha-phi | pré-função sondagem de fase | alphaphi_prefuncao_fase.png")
print("=" * 65)
