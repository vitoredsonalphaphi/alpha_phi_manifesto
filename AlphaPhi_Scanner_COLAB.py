# AlphaPhi_Scanner_COLAB.py
# Vitor Edson Delavi · Florianópolis · 2026
#
# SCANNER α-φ — INSTRUMENTO DE MAPEAMENTO PRÉ-AÇÃO
#
# FÓRMULA CENTRAL:
#
#   S_f = coh_f × disc_f
#
#   coh_f  = (1/φ)·coh_A + (1-1/φ)·(coh_B + coh_C)/2
#   f*     = argmax EMA_α(S_f)
#   meta_coh = 1 − H_α(S) / log(N_fases)
#   β      = φ^(3 × coh_f*)
#
# ESPECTROS (3 por fase):
#   A — energia   : H_α( |h| / Σ|h| )        distribuição de energia por neurônio
#   B — ativação  : fração de neurônios ativos (pós-ReLU)
#   C — variância : H_α( Var(h) / ΣVar(h) )  variância inter-amostras
#
# DISCRIMINABILIDADE (2 medidas):
#   disc_lin = E[(μ₁ − μ₀)²]          variância entre centroides de classe
#   disc_ang = 1 − |cos(μ₀, μ₁)|      distância angular entre centroides
#
# META-COERÊNCIA:
#   O eco mede a qualidade da própria observação.
#   H_α dos scores S_f entre as fases.
#   Alta → scores concentrados → observação em fase → age.
#   Baixa → scores dispersos → observação incerta → expande.
#
# DIAGNÓSTICO:
#   meta_coh < META_COH_LIMIAR após N_SONDA_MAX ciclos
#   → substrato inadequado: α não encontra residência

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

# ══════════════════════════════════════════════════════════════════
# SCANNER α-φ — classe principal
# ══════════════════════════════════════════════════════════════════

class ScannerAlphaPhi:
    """
    Scanner α-φ — instrumento de mapeamento pré-ação.

    Uso:
        scanner = ScannerAlphaPhi(rede)
        for epoch in range(N_EPOCHS):
            if not scanner.pronto:
                scanner.escaneia(X_batch, y_batch)
            lr = LR_BASE * scanner.beta(X_batch)
            # ... treino ...
        print(scanner.relatorio())
    """

    PHI              = (1 + np.sqrt(5)) / 2
    ALPHA            = 1 / 137.035999084
    LOG_ALPHA        = np.log(1.0 / ALPHA)    # log(137) ≈ 4.920
    wm               = 1.0 / PHI              # 1/φ ≈ 0.618 — peso primário
    wn               = 1.0 - 1.0 / PHI       # 1-1/φ ≈ 0.382
    META_COH_LIMIAR  = 0.70
    LIMIAR_SUBSTRATO = 1e-3
    N_SONDA_MIN      = 3
    N_SONDA_MAX      = 10

    def __init__(self, rede,
                 n_sonda_min=None, n_sonda_max=None,
                 meta_coh_limiar=None, limiar_substrato=None):
        self.rede              = rede
        self._n_sonda_min      = n_sonda_min      or self.__class__.N_SONDA_MIN
        self._n_sonda_max      = n_sonda_max      or self.__class__.N_SONDA_MAX
        self._meta_coh_limiar  = meta_coh_limiar  or self.__class__.META_COH_LIMIAR
        self._limiar_substrato = limiar_substrato or self.__class__.LIMIAR_SUBSTRATO

        self._scores_ema   = None
        self._meta_coh_hist = []
        self._perfil_final = None
        self._n_ciclos     = 0
        self._fase_otima   = None
        self._bm           = 1.0
        self._beta         = 1.0
        self.pronto        = False

    # ── Espectros ───────────────────────────────────────────────────

    def _coh_A(self, h):
        """Espectro de energia: H_α da distribuição de |ativações|."""
        h_abs  = np.abs(h) + 1e-10
        h_norm = h_abs / h_abs.sum(axis=1, keepdims=True)
        H      = -np.sum(h_norm * np.log(h_norm + 1e-15), axis=1)
        H_max  = max(np.log(h.shape[1]), self.LOG_ALPHA)
        return float(1.0 - np.clip(H / H_max, 0.0, 1.0).mean())

    def _coh_B(self, h):
        """Espectro de ativação: fração de neurônios ativos pós-ReLU."""
        return float((h > 0).mean())

    def _coh_C(self, h):
        """Espectro de variância: H_α da variância inter-amostras."""
        v      = h.var(axis=0) + 1e-10
        v_norm = v / v.sum()
        H_v    = float(-np.sum(v_norm * np.log(v_norm + 1e-15)))
        H_max  = max(np.log(h.shape[1]), self.LOG_ALPHA)
        return float(1.0 - np.clip(H_v / H_max, 0.0, 1.0))

    def _coh_composta(self, h):
        """coh_f = (1/φ)·coh_A + (1-1/φ)·(coh_B + coh_C)/2"""
        return self.wm * self._coh_A(h) + self.wn * (self._coh_B(h) + self._coh_C(h)) / 2.0

    def _disc_lin(self, h, y):
        """Discriminabilidade linear: E[(μ₁ − μ₀)²]"""
        m0, m1 = y < 0.5, y >= 0.5
        if m0.sum() == 0 or m1.sum() == 0:
            return 0.0
        return float(np.mean((h[m1].mean(0) - h[m0].mean(0)) ** 2))

    def _disc_ang(self, h, y):
        """Discriminabilidade angular: 1 − |cos(μ₀, μ₁)|"""
        m0, m1 = y < 0.5, y >= 0.5
        if m0.sum() == 0 or m1.sum() == 0:
            return 0.0
        mu0, mu1 = h[m0].mean(0), h[m1].mean(0)
        n0, n1   = np.linalg.norm(mu0), np.linalg.norm(mu1)
        if n0 < 1e-10 or n1 < 1e-10:
            return 0.0
        return float(1.0 - abs(np.dot(mu0, mu1) / (n0 * n1)))

    def _score(self, h, y):
        """S_f = coh_f × disc_f"""
        coh  = self._coh_composta(h)
        disc = (self._disc_lin(h, y) + self._disc_ang(h, y)) / 2.0
        return coh * disc, {
            'coh_A': self._coh_A(h),   'coh_B': self._coh_B(h),
            'coh_C': self._coh_C(h),   'coh':   coh,
            'disc_lin': self._disc_lin(h, y),
            'disc_ang': self._disc_ang(h, y),
            'disc': disc, 'score': coh * disc,
        }

    def _meta_coh(self, scores):
        """meta_coh = 1 − H_α(S) / log(N_fases)"""
        s      = np.array(scores) + 1e-10
        s_norm = s / s.sum()
        H_meta = float(-np.sum(s_norm * np.log(s_norm + 1e-15)))
        H_max  = np.log(max(len(s), 2))
        return float(1.0 - np.clip(H_meta / H_max, 0.0, 1.0))

    # ── Interface principal ─────────────────────────────────────────

    def escaneia(self, X_batch, y_batch):
        """
        Executa um ciclo de sondagem multi-espectral.
        Deve ser chamado uma vez por época, antes do treino.
        """
        if self.pronto:
            return

        # propaga pela rede e sonda cada fase intermediária
        h       = X_batch
        perfil  = []
        scores  = []
        dims    = self.rede.dims

        for i, (W, b) in enumerate(zip(self.rede.W[:-1], self.rede.b[:-1])):
            h = np.maximum(0.0, h @ W + b)   # ReLU
            s, p = self._score(h, y_batch)
            p['fase'] = i + 1
            p['dim']  = dims[i + 1]
            perfil.append(p)
            scores.append(s)

        # EMA_α dos scores (α como fator de suavização)
        s_arr = np.array(scores)
        if self._scores_ema is None:
            self._scores_ema = s_arr.copy()
        else:
            self._scores_ema = self.ALPHA * s_arr + (1.0 - self.ALPHA) * self._scores_ema

        mc = self._meta_coh(self._scores_ema)
        self._meta_coh_hist.append(mc)
        self._n_ciclos += 1

        parar  = self._n_ciclos >= self._n_sonda_min and mc >= self._meta_coh_limiar
        limite = self._n_ciclos >= self._n_sonda_max

        if parar or limite:
            self._fase_otima   = int(np.argmax(self._scores_ema)) + 1
            self._perfil_final = perfil
            self.pronto        = True

    def beta(self, X_batch):
        """
        β = φ^(3 × coh_f*)
        Aplica piso α na fase ótima, mede coerência, retorna β.
        """
        if self._fase_otima is None:
            return 1.0
        h = X_batch
        for i, (W, b) in enumerate(zip(self.rede.W[:-1], self.rede.b[:-1])):
            h = np.maximum(0.0, h @ W + b)
            if (i + 1) == self._fase_otima:
                h_abs = np.abs(h)
                piso  = self.ALPHA * h_abs.max(axis=1, keepdims=True)
                h_cp  = np.where(h > 0, np.maximum(h, piso), h)
                coh   = self._coh_A(h_cp)
                ba    = float(self.PHI ** (3.0 * coh))
                self._beta = self.wn * ba + self.wm * self._bm
                self._bm   = self._beta
                return max(self._beta, 0.1)
        return 1.0

    def relatorio(self):
        """Retorna dict com resultado completo do scan."""
        if self._perfil_final is None:
            return {"pronto": False}
        scores  = [p['score'] for p in self._perfil_final]
        mc_final = self._meta_coh_hist[-1] if self._meta_coh_hist else 0.0
        adequado = (max(scores) > self._limiar_substrato
                    and mc_final >= self._meta_coh_limiar)
        return {
            "pronto":        True,
            "fase_otima":    self._fase_otima,
            "dim_otima":     self.rede.dims[self._fase_otima],
            "scores":        scores,
            "meta_coh":      mc_final,
            "n_ciclos":      self._n_ciclos,
            "adequado":      adequado,
            "perfil":        self._perfil_final,
            "beta_atual":    self._beta,
        }


# ══════════════════════════════════════════════════════════════════
# REDE — arquitetura Fibonacci → α-nativo
# ══════════════════════════════════════════════════════════════════

class RedeAlphaPhi:
    """Rede FC [135→55→89→137→1] — Fibonacci + espaço nativo α."""

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.dims = [5 * 27, 55, 89, 137, 1]
        self.W = [np.random.randn(self.dims[i], self.dims[i+1])
                  * np.sqrt(2.0 / self.dims[i])
                  for i in range(len(self.dims)-1)]
        self.b = [np.zeros(self.dims[i+1]) for i in range(len(self.dims)-1)]
        self._cache = []

    def forward(self, X):
        self._cache = [X]
        h = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(0.0, h @ W + b)
            self._cache.append(h)
        out = 1.0 / (1.0 + np.exp(-np.clip(h @ self.W[-1] + self.b[-1], -30, 30)))
        self._cache.append(out.ravel())
        return out.ravel()

    def backward(self, X, y, lr):
        n    = len(y)
        dout = ((self._cache[-1] - y) / n).reshape(-1, 1)
        for i in range(len(self.W)-1, -1, -1):
            self.W[i] -= lr * (self._cache[i].T @ dout)
            self.b[i] -= lr * dout.sum(axis=0)
            if i > 0:
                dh = dout @ self.W[i].T
                dh[self._cache[i] <= 0] = 0
                dout = dh

    def predict(self, X):
        h = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(0.0, h @ W + b)
        return (1.0 / (1.0 + np.exp(-np.clip(h @ self.W[-1] + self.b[-1], -30, 30)))).ravel()


# ══════════════════════════════════════════════════════════════════
# SUBSTRATO — janelas de caracteres (SST-2)
# ══════════════════════════════════════════════════════════════════

def char_freq(text):
    counts = np.zeros(27)
    for c in text.lower():
        if 'a' <= c <= 'z': counts[ord(c) - ord('a')] += 1
        elif c == ' ':       counts[26] += 1
    total = counts.sum()
    return counts / total if total > 0 else np.ones(27) / 27

def para_janelas(text, n_jan=5):
    n = max(len(text), 1); step = n / n_jan
    freqs = np.zeros((n_jan, 27))
    for j in range(n_jan):
        s = int(j * step); e = max(int((j+1) * step), s+1)
        freqs[j] = char_freq(text[s:e])
    return freqs.ravel()


# ══════════════════════════════════════════════════════════════════
# DEMO — Scanner α-φ no SST-2
# ══════════════════════════════════════════════════════════════════

PHI   = ScannerAlphaPhi.PHI
ALPHA = ScannerAlphaPhi.ALPHA

print("=" * 65)
print(f"φ = {PHI:.10f}   α = {ALPHA:.10f}   φ³ = {PHI**3:.6f}")
print("SCANNER α-φ — DEMO SST-2 (substrato de caracteres)")
print("=" * 65)

print("\nCarregando SST-2...")
_ds = None
for _name, _cfg, _field in [
    ("stanfordnlp/sst2", None, "sentence"),
    ("SetFit/sst2",      None, "text"),
    ("glue",             "sst2", "sentence"),
]:
    try:
        _ds = load_dataset(_name) if _cfg is None else load_dataset(_name, _cfg)
        _F  = _field; print(f"  {_name}"); break
    except Exception as e:
        print(f"  {_name}: {type(e).__name__}")
if _ds is None:
    raise RuntimeError("SST-2 não carregou")

np.random.seed(42)
i_tr  = np.random.choice(len(_ds['train']),      3000, replace=False)
i_val = np.random.choice(len(_ds['validation']),  872, replace=False)
X_tr  = np.array([para_janelas(_ds['train'][int(i)][_F])      for i in i_tr])
X_val = np.array([para_janelas(_ds['validation'][int(i)][_F]) for i in i_val])
y_tr  = np.array([_ds['train'][int(i)]['label']      for i in i_tr],  dtype=float)
y_val = np.array([_ds['validation'][int(i)]['label'] for i in i_val], dtype=float)
print(f"Treino: {X_tr.shape}  Val: {X_val.shape}")

N_SEEDS  = 10
N_EPOCHS = 30
BATCH    = 128
LR_BASE  = 0.01
rng      = np.random.default_rng(42)
SEEDS    = [int(x) for x in rng.integers(1, 2**31, N_SEEDS)]

CONFIGS = {
    "AA — baseline   (sem Scanner)   ": False,
    "CC — Scanner α-φ (fase adaptiva)": True,
}

resultados = {}
print(f"\nSeeds: {SEEDS[:3]}...{SEEDS[-1]}\n")

for nome, usa_scanner in CONFIGS.items():
    print(f"{'='*65}\n{nome}")
    accs, relatorios = [], []

    for seed in SEEDS:
        rede    = RedeAlphaPhi(seed)
        scanner = ScannerAlphaPhi(rede) if usa_scanner else None
        np.random.seed(seed)

        for epoch in range(N_EPOCHS):
            perm = np.random.permutation(len(X_tr))

            # ── Scanner escaneia antes do treino ──────────────────────
            if scanner and not scanner.pronto:
                scanner.escaneia(X_tr[perm[:BATCH]], y_tr[perm[:BATCH]])

            # ── LR orientado pelo Scanner ──────────────────────────────
            lr = LR_BASE
            if scanner:
                lr = LR_BASE * scanner.beta(X_tr[:BATCH])

            # ── Treino ────────────────────────────────────────────────
            for k in range(len(X_tr) // BATCH):
                idx = perm[k*BATCH:(k+1)*BATCH]
                rede.forward(X_tr[idx])
                rede.backward(X_tr[idx], y_tr[idx], lr)

        acc = float(np.mean((rede.predict(X_val) > 0.5) == y_val))
        accs.append(acc)

        if scanner:
            r = scanner.relatorio()
            relatorios.append(r)
            mc  = r['meta_coh']
            f   = r['fase_otima']
            nc  = r['n_ciclos']
            tag = "OK" if r['adequado'] else "INADEQUADO"
            print(f"  seed {seed} → acc={acc:.4f}  F{f}({r['dim_otima']})"
                  f"  meta_coh={mc:.3f}  ciclos={nc}  β={r['beta_atual']:.3f}  {tag}")
        else:
            print(f"  seed {seed} → acc={acc:.4f}")

    resultados[nome] = {"acc": np.mean(accs), "std": np.std(accs),
                        "accs": accs, "relatorios": relatorios}
    print(f"  → acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

# ── Síntese ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SÍNTESE — SCANNER α-φ")
print("=" * 65)
aa_accs = list(resultados.values())[0]["accs"]
aa_acc  = list(resultados.values())[0]["acc"]
for n, r in resultados.items():
    print(f"  {n}  acc={r['acc']:.4f} ± {r['std']:.4f}  Δ={r['acc']-aa_acc:+.4f}")

cc_rels = list(resultados.values())[1]["relatorios"]
if cc_rels:
    mc_med  = np.mean([r['meta_coh']   for r in cc_rels])
    nc_med  = np.mean([r['n_ciclos']   for r in cc_rels])
    b_med   = np.mean([r['beta_atual'] for r in cc_rels])
    print(f"\n  meta_coh médio : {mc_med:.4f}  (limiar={ScannerAlphaPhi.META_COH_LIMIAR})")
    print(f"  ciclos médio   : {nc_med:.1f}")
    print(f"  β médio        : {b_med:.4f}")
    adequados = sum(r['adequado'] for r in cc_rels)
    print(f"  substrato      : {adequados}/10 adequados")
    if mc_med < ScannerAlphaPhi.META_COH_LIMIAR:
        print(f"\n  → Diagnóstico: substrato INADEQUADO")
        print(f"    meta_coh não convergiu — sinal semântico ausente neste substrato")

_, p = stats.ttest_rel(list(resultados.values())[1]["accs"], aa_accs)
print(f"\n  Teste t CC vs AA: p={p:.4f}")
print(f"\n  φ = {PHI:.6f}   α = {ALPHA:.10f}   φ³ = {PHI**3:.6f}")
print("=" * 65)
print("\nalpha-phi | scanner | 13.06.2026")
print("=" * 65)
