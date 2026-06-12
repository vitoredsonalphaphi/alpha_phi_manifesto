# AlphaPhi_Eco_Beep880_Semente_COLAB.py
# Vitor Edson Delavi · Florianopolis · 2026
#
# ECO-RESSONANTE 880Hz — REFORMULADO COM SEMENTE ALPHA
#
# Identico ao eco-ressonante original (roldanas phi-alternadas)
# com uma unica diferenca estrutural:
#
#   ANTES: coh = 1 - H / log(N_bins)   (regua natural)
#   AGORA: coh = 1 - H / log(137)      (regua alpha — semente)
#          an  = clip(mag/sum, alpha, 1.0)  (piso alpha na distribuicao)
#
# Alpha esta literalmente na entropia.
# Phi permanece o atrator (phi^3 = limiar do campo harmonico).
#
# ABLACAO (identica ao original):
#   AA: sem roldana    (N_base=0)
#   BB: phi-alt N=14   (moderado)
#   CC: phi-alt N=137  (maximo, obs+roldana)
#   DD: phi-alt N=137  (desde ciclo 1, sem fase de obs)
#
# PERGUNTAS vs original:
#   (1) Campo ainda forma no ciclo 10 em todas as configs?
#   (2) R_natural muda com a regua alpha? (AA: R/phi = ?)
#   (3) Roldanas diferenciam com coh_alpha?
#   (4) H_alpha_equilibrio por config (alpha* no eixo entropico)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import Audio, display

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
ALPHA_FINE = 1 / 137.035999084
LOG_ALPHA  = np.log(1.0 / ALPHA_FINE)   # log(137) ~= 4.920
N_BASE     = int(round(1.0 / ALPHA_FINE))  # 137

FS       = 44100
F_BEEP   = 880.0
F_ORG    = 220.0
F_M      = F_ORG / PHI
BETA_FM  = PHI
DURACAO  = 1.5
N_STEPS  = 5
N_CICLOS = 20
N_OBS    = 10
LIMIAR   = 0.99 * PHI**3

N_BASE_MOD = 14
N_BASE_MAX = N_BASE   # 137

wm = 1.0 / PHI
wn = 1.0 - 1.0 / PHI

print(f"phi           = {PHI:.10f}")
print(f"alpha         = {ALPHA_FINE:.10f}  (1/{N_BASE})")
print(f"log(1/alpha)  = {LOG_ALPHA:.6f}  (regua de entropia — semente)")
print(f"phi^3 (limiar)= {PHI**3:.6f}")
print(f"\nAlternancia phi-alternada (N_base=137):")
for k in range(N_STEPS):
    Nk  = N_BASE_MAX / PHI**k
    d   = Nk * ALPHA_FINE
    tag = "E" if k%2==0 else "X"
    sc  = float(np.clip(1.0+d,1.0,PHI**2)) if k%2==0 \
          else float(np.clip(1.0-d/PHI,0.1,1.0))
    print(f"  p{k} [{tag}]: N={Nk:.1f}  delta={d:.4f}  ce_scale={sc:.4f}")
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

def gerar_beep(duracao, alpha_sinal=1.0 / 3.0):
    N = int(FS * duracao)
    t = np.linspace(0, duracao, N, endpoint=False)
    beep = np.sign(np.sin(2 * np.pi * F_BEEP * t))
    fm   = np.sin(2 * np.pi * F_M * t + BETA_FM * np.sin(2 * np.pi * F_M * t / PHI))
    return normalizar(alpha_sinal * beep + (1 - alpha_sinal) * fm)

def medir_R(cohs):
    c = np.asarray(cohs, dtype=float)
    return float(np.mean((1-c)**2)) / (float(np.mean(c**2)) + 1e-10)

# ─── Semente alpha — unica mudanca vs original ────────────────────────────────
def eco_eq_semente(x, bins_phi, beta_bands, coh_mem, ce_scale=1.0):
    """
    Passo do equalizador com semente alpha.
    Diferenca do original: an = clip(mag/sum, alpha, 1.0) + H/log(137).
    """
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    coh_mem    = np.atleast_1d(np.asarray(coh_mem,    dtype=float))

    N, F  = len(x), np.fft.rfft(x)
    F_out = F.copy()
    cohs  = []

    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi    = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb    = F[b_lo:b_hi]
        mag   = np.abs(Fb)
        phase = np.angle(Fb)

        # SEMENTE: alpha como piso na distribuicao
        an  = np.clip(mag / (mag.sum() + 1e-8), ALPHA_FINE, 1.0)
        an  = an / an.sum()
        H   = -np.sum(an * np.log(an + 1e-15))

        # Coerencia em alpha-nats (regua irrevogavel)
        coh = float(np.clip(1.0 - H / LOG_ALPHA, 0.0, 1.0))

        cm     = float(coh_mem[i]) if i < len(coh_mem) else coh
        ce     = wn * coh + wm * cm
        ce_eff = float(np.clip(ce * ce_scale, 0.0, PHI**2))
        cohs.append(coh)

        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce_eff * PHI**bi) * np.cos(2*np.pi*nk/PHI),
                      0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)

    r = np.fft.irfft(F_out, n=N)
    return normalizar(r), np.array(cohs)

# ─── Cascata phi-alternada ────────────────────────────────────────────────────
def cascata_phi(sinal, beta_bands, bins_phi, coh_mem_init=None, N_base=0):
    s  = sinal.copy()
    cm = coh_mem_init.copy() if coh_mem_init is not None else np.zeros(len(bins_phi))

    cohs_final = cm.copy()
    for k in range(N_STEPS):
        if N_base == 0:
            ce_scale = 1.0
        else:
            Nk    = float(N_base) / (PHI**k)
            delta = Nk * ALPHA_FINE
            if k % 2 == 0:
                ce_scale = float(np.clip(1.0 + delta, 1.0, PHI**2))
            else:
                ce_scale = float(np.clip(1.0 - delta / PHI, 0.1, 1.0))

        se, cohs = eco_eq_semente(s, bins_phi, beta_bands, cm, ce_scale)
        cm = cohs
        s  = normalizar(se)
        cohs_final = cohs

    return s, cohs_final

# ─── Substrato e bandas ───────────────────────────────────────────────────────
sinal_base = gerar_beep(DURACAO)
BANDAS     = gerar_bandas_phi()
BINS       = bandas_para_bins(BANDAS, len(sinal_base))
nb         = len(BINS)
print(f"Substrato 880Hz: {len(sinal_base)} amostras  |  Bandas phi: {nb}\n")

# ─── Configs ──────────────────────────────────────────────────────────────────
CONFIGS = {
    "AA — sem roldana    (N=0)          ": {"N_base": 0,          "N_obs": N_OBS, "color": "#8B949E"},
    "BB — phi-alt N=14   (moderado)     ": {"N_base": N_BASE_MOD, "N_obs": N_OBS, "color": "#00BFFF"},
    "CC — phi-alt N=137  (obs+roldana)  ": {"N_base": N_BASE_MAX, "N_obs": N_OBS, "color": "#00FF88"},
    "DD — phi-alt N=137  (desde ciclo 1)": {"N_base": N_BASE_MAX, "N_obs": 0,     "color": "#FF9944"},
}

# ─── Loop principal ───────────────────────────────────────────────────────────
resultados  = {}
sinais_finais = {}

print("=" * 65)
for cfg_nome, cfg in CONFIGS.items():
    N_base_cfg = cfg["N_base"]
    N_obs_cfg  = cfg["N_obs"]
    print(f"\n{cfg_nome}")

    beta    = np.ones(nb)
    bm      = beta.copy()
    coh_mem = np.zeros(nb)
    R_ema   = 1.669

    hist_R, hist_beta, hist_coh = [], [], []
    ciclo_conv  = N_CICLOS
    sinal_final = sinal_base.copy()

    for ciclo in range(N_CICLOS):
        N_ativo = 0 if ciclo < N_obs_cfg else N_base_cfg

        s_out, cohs = cascata_phi(sinal_base, beta, BINS, coh_mem, N_ativo)
        coh_mem = cohs.copy()

        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn * ba + wm * bm
        bm   = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)

        R_ciclo = medir_R(cohs)
        R_ema   = ALPHA_FINE * R_ciclo + (1.0 - ALPHA_FINE) * R_ema

        beta_max = float(beta.max())
        coh_med  = float(cohs.mean())

        hist_R.append(R_ema)
        hist_beta.append(beta_max)
        hist_coh.append(coh_med)

        sinal_final = s_out

        if beta_max >= LIMIAR and ciclo_conv == N_CICLOS:
            ciclo_conv = ciclo + 1
            print(f"  CAMPO HARMONICO -> ciclo {ciclo_conv:2d}"
                  f"  beta={beta_max:.4f}")

        if (ciclo + 1) % 5 == 0:
            fase = "OBS" if ciclo < N_obs_cfg else f"ROL(N={N_ativo})"
            print(f"  ciclo {ciclo+1:2d} [{fase:>12}]"
                  f"  beta={beta_max:.4f}"
                  f"  R_ema={R_ema:.5f}({R_ema/ALPHA_FINE:.1f}a)"
                  f"  coh={coh_med:.4f}")

    campo_ok = ciclo_conv <= N_CICLOS
    R_final  = hist_R[-1]
    print(f"  {'CAMPO ciclo '+str(ciclo_conv) if campo_ok else 'campo NAO formado'}")
    print(f"  R_final={R_final:.5f}={R_final/ALPHA_FINE:.1f}a  R/phi={R_final/PHI:.4f}")

    resultados[cfg_nome]    = {"hist_R": hist_R, "hist_beta": hist_beta,
                                "hist_coh": hist_coh, "ciclo_conv": ciclo_conv,
                                "R_final": R_final, "campo": campo_ok,
                                "color": cfg["color"]}
    sinais_finais[cfg_nome] = sinal_final.copy()

# ─── Sintese ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SINTESE — Eco Beep 880Hz reformulado (semente alpha)")
print("=" * 65)
print(f"\n{'Config':<42} {'Campo':>6} {'Ciclo':>6} {'R_final':>14} {'R/phi':>8}")
print("-" * 80)
for nome, res in resultados.items():
    c = f"c{res['ciclo_conv']}" if res["campo"] else "NAO"
    print(f"{nome:<42} {'SIM':>6} {c:>6}"
          f"  {res['R_final']:.5f}={res['R_final']/ALPHA_FINE:.1f}a"
          f"  {res['R_final']/PHI:.4f}")
print(f"\nReferencia historica (coh_raw): R_natural = phi = {PHI:.5f}  (ratio=1.0)")
print(f"Com semente alpha (coh_alpha):  R_natural = ?  (ratio acima)")

# ─── Audio — concatena AA+BB+CC+DD (4 configs × 1.5s = 6s) ───────────────────
audio_concat = np.concatenate([sinais_finais[n] for n in CONFIGS])
audio_concat = normalizar(audio_concat)
audio_int    = (audio_concat * 32767 * 0.9).astype(np.int16)
wavfile.write("alphaphi_beep880_semente.wav", FS, audio_int)
print(f"\nAudio: {len(audio_concat)/FS:.1f}s  → alphaphi_beep880_semente.wav")
print("  0.0–1.5s: AA (sem roldana)")
print("  1.5–3.0s: BB (N=14)")
print("  3.0–4.5s: CC (N=137 obs+rol)")
print("  4.5–6.0s: DD (N=137 desde c1)")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Eco Beep 880Hz — Semente α  (α na entropia, φ como atrator)",
             fontsize=12, fontweight="bold")

ciclos = np.arange(1, N_CICLOS + 1)

ax = axes[0]
for nome, res in resultados.items():
    ax.plot(ciclos, res["hist_R"], color=res["color"],
            label=nome[:20].strip(), linewidth=1.8)
ax.axhline(PHI, color="gold", linestyle="--", linewidth=1.5, label=f"φ={PHI:.4f}")
ax.axhline(ALPHA_FINE, color="red", linestyle=":", linewidth=1.0, label="α")
ax.set_title("R_ema"); ax.legend(fontsize=7)

ax = axes[1]
for nome, res in resultados.items():
    ax.plot(ciclos, res["hist_beta"], color=res["color"],
            label=nome[:20].strip(), linewidth=1.8)
ax.axhline(PHI**3, color="gold", linestyle="--", linewidth=1.5, label=f"φ³={PHI**3:.3f}")
ax.set_title("beta_max"); ax.legend(fontsize=7)

ax = axes[2]
for nome, res in resultados.items():
    ax.plot(ciclos, res["hist_coh"], color=res["color"],
            label=nome[:20].strip(), linewidth=1.8)
ax.set_title("Coerencia media (coh_alpha)"); ax.legend(fontsize=7)

for a in axes:
    a.set_facecolor("#0d1117"); a.grid(alpha=0.2); a.set_xlabel("Ciclo")
    for sp in a.spines.values(): sp.set_edgecolor("#444")
    a.tick_params(colors="#ccc"); a.title.set_color("#eee")
fig.patch.set_facecolor("#0d1117")

plt.tight_layout()
plt.savefig("alphaphi_beep880_semente.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()

print("\nalpha-phi | eco beep 880 semente | alphaphi_beep880_semente.png")
display(Audio(audio_concat, rate=FS))
