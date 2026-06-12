# AlphaPhi_Semente_COLAB.py
# Vitor Edson Delavi · Florianopolis · 2026
#
# CODIGO VIRGEM — construido do zero a partir da semente
#
# Nenhum codigo anterior herdado.
# Nenhum parametro de sucesso imposto.
# Os resultados sao o unico parametro de avaliacao.
#
# ARQUITETURA:
#
#   SEMENTE  : alpha como unidade de entropia
#              an = clip(mag/sum, ALPHA, 1.0)
#              H_alpha = H / log(1/alpha) = H / log(137)
#
#   BROTO    : coh_alpha = 1 - H_alpha
#              primeira expressao do atrator (nao imposta — observada)
#
#   CRESCIMENTO — 5 estagios E/X (N_k = 137 / phi^k):
#              k=0 [E]: N=137  ce_scale = 1 + 137*alpha  (comprime, ancora, raiz)
#              k=1 [X]: N=84   ce_scale = 1 - 84*alpha/phi (expande, sobe, sol)
#              k=2 [E]: N=52   ce_scale = 1 + 52*alpha
#              k=3 [X]: N=32   ce_scale = 1 - 32*alpha/phi
#              k=4 [E]: N=20   ce_scale = 1 + 20*alpha   (segunda estabilizacao)
#              N_k segue Fibonacci decrescente via phi
#
#   ATRATOR  : beta = phi^(3 * coh_alpha) — driven by alpha-native coherence
#              campo harmonico: beta_max >= phi^3
#
#   SUBSTRATO: ruido branco puro (entropia maxima — tabula rasa)
#              Nenhuma estrutura phi pre-existente no sinal
#
# PERGUNTA CENTRAL:
#   O campo harmonico pode emergir do ruido branco
#   quando alpha e a semente e phi e o atrator?
#
# DIAGNOSTICO ADICIONAL:
#   Onde R_alpha converge? Qual o alpha* deste substrato?
#   H_alpha desce? A entropia diminui com o crescimento?

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
ALPHA_FINE = 1 / 137.035999084
LOG_ALPHA  = np.log(1.0 / ALPHA_FINE)    # log(137) ~= 4.919
N_BASE     = int(round(1.0 / ALPHA_FINE))  # 137

FS        = 44100
DURACAO   = 1.5
N_STEPS   = 5
N_CICLOS  = 20
N_OBS     = 10
LIMIAR    = 0.99 * PHI**3

print(f"phi          = {PHI:.10f}")
print(f"alpha        = {ALPHA_FINE:.10f}  (1/{N_BASE})")
print(f"log(1/alpha) = {LOG_ALPHA:.6f}  (regua de entropia em alpha-nats)")
print(f"phi^3        = {PHI**3:.6f}  (limiar do campo harmonico)")
print(f"\n5 Estagios de crescimento (N_k = {N_BASE}/phi^k):")
for k in range(N_STEPS):
    Nk  = N_BASE / PHI**k
    d   = Nk * ALPHA_FINE
    tag = "E"; sc = min(1.0 + d, PHI**2)
    if k % 2 != 0: tag = "X"; sc = max(1.0 - d/PHI, 0.1)
    print(f"  k={k} [{tag}]: N={Nk:.1f}  delta={d:.4f}  ce_scale={sc:.4f}")
print("=" * 65)
print("AlphaPhi Semente — codigo virgem")
print("Substrato: ruido branco puro (tabula rasa)")
print("Semente:   alpha como entropia")
print("Atrator:   phi observado (nao imposto)")
print("=" * 65)

# ─── Funcoes base ─────────────────────────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max: break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    return [(max(0, int(f_lo / (FS / n))),
             min(int(f_hi / (FS / n)) + 1, n // 2 + 1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

def gerar_ruido_branco(duracao, seed=42):
    """
    Substrato virgem: ruido branco puro.
    Entropia maxima. Nenhuma estrutura phi pre-existente.
    """
    rng = np.random.default_rng(seed)
    N   = int(FS * duracao)
    return normalizar(rng.standard_normal(N))

def medir_R_alpha(cohs_alpha):
    c      = np.asarray(cohs_alpha, dtype=float)
    E_geom = float(np.mean((1.0 - c)**2))
    E_coer = float(np.mean(c**2))
    return E_geom / (E_coer + 1e-10)

# ─── Passo de crescimento — semente alpha-nativa ──────────────────────────────
def passo_crescimento(x, bins_phi, beta_bands, coh_mem, ce_scale=1.0):
    """
    SEMENTE:    an = clip(mag/sum, ALPHA_FINE, 1.0)
    BROTO:      H_alpha = H / log(137)  ->  coh_alpha = 1 - H_alpha
    CRESCIMENTO: envelope phi sobre coh_alpha com ce_scale
    Beta e driven por coh_alpha (espaco alpha-nativo).
    """
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    coh_mem    = np.atleast_1d(np.asarray(coh_mem,    dtype=float))

    N, F   = len(x), np.fft.rfft(x)
    F_out  = F.copy()
    cohs   = []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI

    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi    = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb    = F[b_lo:b_hi]
        mag   = np.abs(Fb)
        phase = np.angle(Fb)

        # SEMENTE: alpha como piso
        an = np.clip(mag / (mag.sum() + 1e-8), ALPHA_FINE, 1.0)
        an = an / an.sum()
        H  = -np.sum(an * np.log(an + 1e-15))

        # BROTO: coerencia em alpha-nats
        coh_alpha = float(np.clip(1.0 - H / LOG_ALPHA, 0.0, 1.0))

        # Memoria alpha-nativa
        cm = float(coh_mem[i]) if i < len(coh_mem) else coh_alpha
        ce = wn * coh_alpha + wm * cm

        # Crescimento
        ce_eff = float(np.clip(ce * ce_scale, 0.0, PHI**2))
        cohs.append(coh_alpha)

        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce_eff * PHI**bi) * np.cos(2*np.pi*nk/PHI),
                      0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)

    r = np.fft.irfft(F_out, n=N)
    return r / (np.max(np.abs(r)) + 1e-10), np.array(cohs)

# ─── 5 estagios de crescimento ────────────────────────────────────────────────
def cinco_estagios(sinal, beta_bands, bins_phi, coh_mem_init, N_base_rol):
    s   = sinal.copy()
    cm  = coh_mem_init.copy()
    passos = []

    for k in range(N_STEPS):
        if N_base_rol == 0:
            ce_scale = 1.0
        else:
            Nk = float(N_base_rol) / (PHI**k)
            d  = Nk * ALPHA_FINE
            if k % 2 == 0:
                ce_scale = float(np.clip(1.0 + d, 1.0, PHI**2))
            else:
                ce_scale = float(np.clip(1.0 - d / PHI, 0.1, 1.0))

        se, ca = passo_crescimento(s, bins_phi, beta_bands, cm, ce_scale)
        cm = ca
        se = normalizar(se)
        passos.append(se.copy())
        s = se.copy()

    return passos, cm   # cm = cohs_alpha do ultimo passo

# ─── Substrato virgem ─────────────────────────────────────────────────────────
sinal_virgem = gerar_ruido_branco(DURACAO, seed=42)
BANDAS       = gerar_bandas_phi()
BINS         = bandas_para_bins(BANDAS, len(sinal_virgem))
nb           = len(BINS)

# H_alpha inicial (deve ser proximo de 1.0 — ruido branco = entropia maxima)
F0   = np.fft.rfft(sinal_virgem)
mag0 = np.abs(F0[:nb])
an0  = np.clip(mag0 / (mag0.sum() + 1e-8), ALPHA_FINE, 1.0)
an0  = an0 / an0.sum()
H0   = -np.sum(an0 * np.log(an0 + 1e-15))
H0_a = H0 / LOG_ALPHA

print(f"\nSubstrato virgem (ruido branco):")
print(f"  H_alpha inicial = {H0_a:.4f}  (1.0 = max desordem em alpha-nats)")
print(f"  Bandas phi: {len(BANDAS)}  |  Amostras: {len(sinal_virgem)}")

# ─── Configuracoes ────────────────────────────────────────────────────────────
CONFIGS = {
    "AA — ruido puro, sem crescimento  ": {
        "N_rol": 0,      "N_obs": N_CICLOS,
        "color": "#8B949E", "label": "AA ruido puro"},
    "BB — semente + 5 estagios         ": {
        "N_rol": N_BASE, "N_obs": N_OBS,
        "color": "#DAA520", "label": "BB semente+crescimento"},
}

# ─── Loop principal ───────────────────────────────────────────────────────────
resultados = {}

print("\n" + "=" * 65)
for cfg_nome, cfg in CONFIGS.items():
    print(f"\n{cfg_nome}")

    beta  = np.ones(nb)
    bm    = beta.copy()
    wm_b  = 1.0 / PHI
    wn_b  = 1.0 - 1.0 / PHI

    cm    = np.zeros(nb)
    R_ema = 1.0

    hist_R     = []
    hist_coh   = []
    hist_beta  = []
    hist_H     = []

    ciclo_conv  = N_CICLOS
    sinal_final = sinal_virgem.copy()

    for ciclo in range(N_CICLOS):
        N_rol = cfg["N_rol"] if ciclo >= cfg["N_obs"] else 0

        passos, cohs = cinco_estagios(
            sinal_virgem, beta, BINS, cm, N_rol
        )
        cm = cohs.copy()

        # Beta driven by coh_alpha (espaco alpha-nativo)
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI**(3 * cr)
        beta = wn_b * ba + wm_b * bm
        bm   = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)

        R_c   = medir_R_alpha(cohs)
        R_ema = ALPHA_FINE * R_c + (1.0 - ALPHA_FINE) * R_ema

        beta_max = float(beta.max())
        H_med    = float(np.mean(1.0 - cohs))   # H_alpha = 1 - coh_alpha

        hist_R.append(R_ema)
        hist_coh.append(float(cohs.mean()))
        hist_beta.append(beta_max)
        hist_H.append(H_med)

        sinal_final = passos[-1]

        if beta_max >= LIMIAR and ciclo_conv == N_CICLOS:
            ciclo_conv = ciclo + 1
            print(f"  CAMPO HARMONICO -> ciclo {ciclo_conv:2d}  beta={beta_max:.4f}")

        if (ciclo + 1) % 5 == 0:
            fase = "OBS" if ciclo < cfg["N_obs"] else f"CRESCIMENTO(N={N_rol})"
            print(f"  ciclo {ciclo+1:2d} [{fase:>22}]"
                  f"  beta={beta_max:.4f}"
                  f"  R_alpha={R_ema:.5f}({R_ema/ALPHA_FINE:.1f}a)"
                  f"  H_alpha={H_med:.4f}"
                  f"  coh_alp={hist_coh[-1]:.4f}")

    campo_ok = ciclo_conv < N_CICLOS
    R_f = hist_R[-1]

    print(f"  {'CAMPO FORMADO ciclo '+str(ciclo_conv) if campo_ok else 'campo NAO formado'}")
    print(f"  R_alpha_final = {R_f:.5f} = {R_f/ALPHA_FINE:.1f}*alpha")
    print(f"  H_alpha_final = {hist_H[-1]:.5f}  (inicial: {H0_a:.5f})")
    print(f"  coh_alp_final = {hist_coh[-1]:.5f}")

    resultados[cfg_nome] = {
        "hist_R"    : hist_R,
        "hist_coh"  : hist_coh,
        "hist_beta" : hist_beta,
        "hist_H"    : hist_H,
        "ciclo_conv": ciclo_conv,
        "campo_ok"  : campo_ok,
        "R_final"   : R_f,
        "label"     : cfg["label"],
        "color"     : cfg["color"],
    }

    try:
        from scipy.io import wavfile as wf
        tag   = cfg["label"].split()[0].lower()
        fname = f"alphaphi_semente_{tag}.wav"
        wf.write(fname, FS, (sinal_final * 32767).astype(np.int16))
        print(f"  -> {fname}")
    except Exception:
        pass

# ─── Sintese ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SEMENTE — sintese")
print("=" * 65)
aa = resultados["AA — ruido puro, sem crescimento  "]
bb = resultados["BB — semente + 5 estagios         "]

print(f"""
Substrato: ruido branco puro
H_alpha inicial: {H0_a:.4f}  (entropia maxima)

AA (ruido, sem crescimento):
  Campo: {'sim c'+str(aa['ciclo_conv']) if aa['campo_ok'] else 'nao'}
  R_alpha_final = {aa['R_final']:.5f} = {aa['R_final']/ALPHA_FINE:.1f}*alpha
  H_alpha_final = {aa['hist_H'][-1]:.5f}

BB (semente + 5 estagios de crescimento):
  Campo: {'sim c'+str(bb['ciclo_conv']) if bb['campo_ok'] else 'nao'}
  R_alpha_final = {bb['R_final']:.5f} = {bb['R_final']/ALPHA_FINE:.1f}*alpha
  H_alpha_final = {bb['hist_H'][-1]:.5f}
  H_alpha desceu? {(bb['hist_H'][-1] < H0_a - 0.01)}  ({H0_a:.4f} -> {bb['hist_H'][-1]:.4f})

Alpha* emergente (BB):  R_alpha_final / alpha = {bb['R_final']/ALPHA_FINE:.1f}
  (no audio 880Hz: R_natural ≈ 228*alpha ≈ phi/alpha)
  (aqui: ?)

Pergunta central respondida pelos dados:
  O campo harmonico emergiu do ruido branco com alpha como semente?
  BB campo: {'SIM' if bb['campo_ok'] else 'NAO'}
""")
print("alpha-phi  |  semente  |  codigo virgem")

# ─── Visualizacao ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#0d1117")
ciclos_eixo = list(range(1, N_CICLOS + 1))

titulos = [
    ("R_alpha por ciclo\n(espaco alpha-nativo)", "R_alpha"),
    ("beta_max por ciclo\n(campo harmonico emergindo?)", "beta_max"),
    ("H_alpha por ciclo\n(entropia descendo = semente germinando)", "H_alpha"),
]
dados = [
    ([res["hist_R"]   for res in resultados.values()], ALPHA_FINE, "R_alpha=1.0", "phi"),
    ([res["hist_beta"] for res in resultados.values()], PHI**3,    f"phi^3={PHI**3:.3f}", None),
    ([res["hist_H"]   for res in resultados.values()], 0.5,        "H_alpha=0.5", None),
]

for ax, (titulo, ylabel), (series, ref1, lab1, _) in zip(axes, titulos, dados):
    ax.set_facecolor("#161b22")
    ax.set_title(titulo, color="#DAA520", fontweight="bold", fontsize=10)
    for serie, (cfg_nome, res) in zip(series, resultados.items()):
        ax.plot(ciclos_eixo, serie, color=res["color"], linewidth=2,
                label=res["label"], marker="o", markersize=3)
    ax.axhline(ref1, color="#FF4466", linestyle="--", linewidth=2,
               label=lab1, alpha=0.9)
    if ylabel == "beta_max":
        pass
    ax.axvline(N_OBS + 0.5, color="#888888", linestyle="--", alpha=0.4, label="fase 2")
    ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
    ax.tick_params(colors="#8B949E")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.set_xlabel("Ciclo", color="#8B949E")
    ax.set_ylabel(ylabel, color="#8B949E")
    ax.grid(True, alpha=0.15)

fig.suptitle(
    f"AlphaPhi Semente — codigo virgem  |  substrato: ruido branco puro\n"
    f"Semente: alpha como entropia  |  Atrator: phi observado (nao imposto)\n"
    f"phi={PHI:.4f}  alpha=1/{N_BASE}  log(1/alpha)={LOG_ALPHA:.3f}  |  Florianopolis 2026",
    color="#DAA520", fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig("alphaphi_semente.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("alphaphi_semente.png")
