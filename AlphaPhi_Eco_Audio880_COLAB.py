# AlphaPhi_Eco_Audio880_COLAB.py
# Vitor Edson Delavi · Florianopolis · 2026
#
# ECO-RESSONANTE 880Hz — VERSAO AURICULAR
# Alpha como entropia (semente). Phi como atrator.
#
# ESTRUTURA:
#   FASE 1 — Formacao do campo (20 ciclos, config BB: coh_alpha)
#             Acompanha R_ema, beta, H_alpha ate campo harmonico (phi^3)
#
#   FASE 2 — 5 dobras E/X sobre o campo formado
#             Cada dobra = 1.6s de audio
#             Total: 5 x 1.6s = 8 segundos
#             Audio concatenado: ouves a evolucao atraves das dobras
#
#   AUDIO FINAL:
#     Dobra 0 [E]: N=137  — comprime, ancora, raiz
#     Dobra 1 [X]: N=84   — expande, respira, sol
#     Dobra 2 [E]: N=52   — comprime suave
#     Dobra 3 [X]: N=32   — expande suave
#     Dobra 4 [E]: N=20   — segunda estabilizacao
#
#   Alpha esta literalmente na entropia:
#     an = clip(mag/sum, alpha, 1.0)
#     H_alpha = H / log(137)
#     coh_alpha = 1 - H_alpha

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import Audio, display

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
ALPHA_FINE = 1 / 137.035999084
LOG_ALPHA  = np.log(1.0 / ALPHA_FINE)
N_BASE     = int(round(1.0 / ALPHA_FINE))  # 137

FS        = 44100
F_BEEP    = 880.0
F_ORG     = 220.0
F_M       = F_ORG / PHI
BETA_FM   = PHI
DURACAO   = 1.6            # 1.6s por dobra → 5 dobras = 8s
N_STEPS   = 5
N_CICLOS  = 20
N_OBS     = 10
LIMIAR    = 0.99 * PHI**3

wm = 1.0 / PHI
wn = 1.0 - 1.0 / PHI

print(f"phi      = {PHI:.6f}")
print(f"alpha    = {ALPHA_FINE:.8f}  (1/{N_BASE})")
print(f"phi^3    = {PHI**3:.6f}  (limiar campo)")
print(f"log(137) = {LOG_ALPHA:.6f}  (regua de entropia)")
print(f"\n5 dobras E/X  (N_k = {N_BASE}/phi^k):")
for k in range(N_STEPS):
    Nk  = N_BASE / PHI**k
    d   = Nk * ALPHA_FINE
    tag = "E" if k % 2 == 0 else "X"
    sc  = float(np.clip(1.0+d, 1.0, PHI**2)) if k%2==0 \
          else float(np.clip(1.0-d/PHI, 0.1, 1.0))
    print(f"  k={k} [{tag}]: N={Nk:.1f}  delta={d:.4f}  ce_scale={sc:.4f}  dur={DURACAO}s")
print(f"Total audio: {N_STEPS * DURACAO:.1f}s")
print("=" * 60)

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

# ─── Semente alpha ────────────────────────────────────────────────────────────
def semente_banda(mag):
    an = np.clip(mag / (mag.sum() + 1e-8), ALPHA_FINE, 1.0)
    an = an / an.sum()
    H  = -np.sum(an * np.log(an + 1e-15))
    H_a = float(np.clip(H / LOG_ALPHA, 0.0, 1.0))
    return H_a, 1.0 - H_a   # H_alpha, coh_alpha

# ─── Passo semente ────────────────────────────────────────────────────────────
def passo_semente(x, bins_phi, beta_bands, coh_mem, ce_scale=1.0):
    """
    Envelope phi sobre coh_alpha (alpha-nativo).
    ce_scale: dobra E/X externa.
    """
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    coh_mem    = np.atleast_1d(np.asarray(coh_mem,    dtype=float))

    N, F  = len(x), np.fft.rfft(x)
    F_out = F.copy()
    cohs, H_alphas = [], []

    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi    = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb    = F[b_lo:b_hi]
        mag   = np.abs(Fb)
        phase = np.angle(Fb)

        H_a, coh = semente_banda(mag)
        cm  = float(coh_mem[i]) if i < len(coh_mem) else coh
        ce  = wn * coh + wm * cm
        ce_eff = float(np.clip(ce * ce_scale, 0.0, PHI**2))

        cohs.append(coh)
        H_alphas.append(H_a)

        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce_eff * PHI**bi) * np.cos(2*np.pi*nk/PHI),
                      0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)

    r = np.fft.irfft(F_out, n=N)
    return (normalizar(r), np.array(cohs), np.array(H_alphas))

# ─── Substrato ────────────────────────────────────────────────────────────────
sinal_base = gerar_beep(DURACAO)
BANDAS     = gerar_bandas_phi()
BINS       = bandas_para_bins(BANDAS, len(sinal_base))
nb         = len(BINS)
print(f"Substrato 880Hz: {len(sinal_base)} amostras  |  Bandas phi: {nb}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1: Formacao do campo harmonico (20 ciclos, semente BB)
# ═══════════════════════════════════════════════════════════════════════════════
print("FASE 1 — Formacao do campo")
print("-" * 60)

beta     = np.ones(nb)
bm       = beta.copy()
coh_mem  = np.zeros(nb)
R_ema    = 1.669
ciclo_campo = N_CICLOS
sinal_campo = sinal_base.copy()

hist_R, hist_beta, hist_H = [], [], []

for ciclo in range(N_CICLOS):
    s_out, cohs, H_alphas = passo_semente(sinal_base, BINS, beta, coh_mem)
    coh_mem = cohs.copy()

    cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
    ba   = PHI ** (3 * cr)
    beta = wn * ba + wm * bm
    bm   = beta.copy()
    beta = np.clip(beta, 0.05, PHI**3)

    R_ciclo = medir_R(cohs)
    R_ema   = ALPHA_FINE * R_ciclo + (1.0 - ALPHA_FINE) * R_ema

    beta_max  = float(beta.max())
    H_med     = float(H_alphas.mean())

    hist_R.append(R_ema)
    hist_beta.append(beta_max)
    hist_H.append(H_med)

    if beta_max >= LIMIAR and ciclo_campo == N_CICLOS:
        ciclo_campo  = ciclo + 1
        sinal_campo  = s_out.copy()   # estado do campo no momento da formacao
        print(f"  >>> CAMPO HARMONICO formado: ciclo {ciclo_campo}"
              f"  beta={beta_max:.4f}  R={R_ema:.5f}={R_ema/ALPHA_FINE:.1f}a"
              f"  H_alpha={H_med:.4f}")

    if (ciclo + 1) % 5 == 0:
        print(f"  ciclo {ciclo+1:2d}  beta={beta_max:.4f}"
              f"  R_ema={R_ema:.5f}({R_ema/ALPHA_FINE:.1f}a)"
              f"  H_alpha={H_med:.4f}")

H_eq = float(np.mean(hist_H[-5:]))
print(f"\n  H_alpha_equilibrio (substrato 880Hz): {H_eq:.4f}")
print(f"  R_final = {hist_R[-1]:.5f} = {hist_R[-1]/ALPHA_FINE:.1f}*alpha"
      f"  |  R/phi = {hist_R[-1]/PHI:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2: 5 dobras E/X sobre o campo formado → 8 segundos de audio
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFASE 2 — 5 dobras E/X sobre o campo")
print("-" * 60)

dobras_audio = []
s_dobra = sinal_campo.copy()
coh_dobra = coh_mem.copy()

nomes_dobra = ["k=0 [E] ancora-raiz    ",
               "k=1 [X] expande-sol    ",
               "k=2 [E] comprime-suave ",
               "k=3 [X] expande-suave  ",
               "k=4 [E] 2a estabilizacao"]

for k in range(N_STEPS):
    Nk      = float(N_BASE) / (PHI**k)
    delta   = Nk * ALPHA_FINE
    if k % 2 == 0:
        ce_scale = float(np.clip(1.0 + delta, 1.0, PHI**2))
    else:
        ce_scale = float(np.clip(1.0 - delta / PHI, 0.1, 1.0))

    s_out, cohs, H_alphas = passo_semente(
        s_dobra, BINS, beta, coh_dobra, ce_scale=ce_scale
    )
    s_dobra   = s_out.copy()
    coh_dobra = cohs.copy()

    R_k   = medir_R(cohs)
    H_k   = float(H_alphas.mean())
    beta_k = float(beta.max())

    dobras_audio.append(s_out.copy())

    print(f"  {nomes_dobra[k]}  N={Nk:.0f}  ce={ce_scale:.4f}"
          f"  R={R_k:.4f}={R_k/ALPHA_FINE:.1f}a"
          f"  H_alpha={H_k:.4f}"
          f"  beta_max={beta_k:.4f}")

# ─── Concatena 5 dobras → 8 segundos ─────────────────────────────────────────
audio_8s = np.concatenate(dobras_audio)
audio_8s = normalizar(audio_8s)
audio_int = (audio_8s * 32767 * 0.9).astype(np.int16)

wavfile.write("alphaphi_eco_880_8s.wav", FS, audio_int)
print(f"\n  Audio: {len(audio_8s)/FS:.1f}s  → alphaphi_eco_880_8s.wav")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("AlphaPhi Eco 880Hz — α como entropia, φ como atrator\n"
             "Fase 1: formação do campo  |  Fase 2: 5 dobras E/X (8s audio)",
             fontsize=12, fontweight="bold")

ciclos = np.arange(1, N_CICLOS + 1)
ax = axes[0, 0]
ax.plot(ciclos, hist_R, color="#00BFFF", linewidth=2)
ax.axhline(PHI, color="gold", linestyle="--", linewidth=1.5, label=f"φ = {PHI:.4f}")
ax.axvline(ciclo_campo, color="#FF9944", linestyle=":", linewidth=1.5,
           label=f"campo c{ciclo_campo}")
ax.set_title("R_ema — Fase 1"); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(ciclos, hist_beta, color="#00FF88", linewidth=2)
ax.axhline(PHI**3, color="gold", linestyle="--", linewidth=1.5, label=f"φ³={PHI**3:.3f}")
ax.axvline(ciclo_campo, color="#FF9944", linestyle=":", linewidth=1.5)
ax.set_title("beta_max — Fase 1"); ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(ciclos, hist_H, color="#FF88CC", linewidth=2)
ax.axhline(H_eq, color="white", linestyle=":", linewidth=1.2,
           label=f"H_eq={H_eq:.4f}")
ax.set_title("H_alpha — eixo entropico"); ax.legend(fontsize=8)

ax = axes[1, 1]
t8 = np.linspace(0, len(audio_8s)/FS, len(audio_8s))
ax.plot(t8, audio_8s, color="#00BFFF", linewidth=0.4, alpha=0.8)
for k in range(N_STEPS):
    tk = k * DURACAO
    tag = "E" if k%2==0 else "X"
    ax.axvline(tk, color="#FF9944" if k%2==0 else "#FF88CC",
               linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(tk + 0.05, 0.85, f"k{k}[{tag}]", color="white", fontsize=8)
ax.set_title("Waveform — 8s audio (5 dobras E/X)")
ax.set_xlabel("Tempo (s)")

for a in axes.flat:
    a.set_facecolor("#0d1117"); a.grid(alpha=0.2)
    for sp in a.spines.values(): sp.set_edgecolor("#444")
    a.tick_params(colors="#ccc"); a.title.set_color("#eee")
fig.patch.set_facecolor("#0d1117")

plt.tight_layout()
plt.savefig("alphaphi_eco_880_8s.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()

# ─── Audio Colab ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("AUDIO — 5 dobras E/X  (8 segundos)")
print("  k0 [E] ancora-raiz     0.0s – 1.6s")
print("  k1 [X] expande-sol     1.6s – 3.2s")
print("  k2 [E] comprime-suave  3.2s – 4.8s")
print("  k3 [X] expande-suave   4.8s – 6.4s")
print("  k4 [E] 2a estabilizacao 6.4s – 8.0s")
print("=" * 60)
display(Audio(audio_8s, rate=FS))
