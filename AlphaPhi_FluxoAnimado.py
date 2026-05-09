"""
AlphaPhi_FluxoAnimado.py
Animação — Fluxo da Frequência em Observação

Janela deslizante (±250 ms) sobre o sinal completo (8.25 s).
Cor transita P→S→T conforme posição temporal.
Mesmo sinal. Mesmo espaço (tempo × amplitude).
Nenhuma forma desenhada. O que emergir é o que está lá.

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import hilbert, butter, filtfilt
from IPython.display import display, Video

# ── constantes ORIGINAIS — não modificar ─────────────────────
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5
N_STEPS    = 5
N_CICLOS   = 20
FADE       = int(0.15 * FS)

print("=" * 60)
print("  AlphaPhi · Fluxo Animado")
print("  Janela deslizante · transição P→S→T")
print("=" * 60)

# ── funções eco originais ─────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s)); return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max: break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    return [(max(0, int(f_lo/(FS/n))),
             min(int(f_hi/(FS/n))+1, n//2+1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0/PHI, 1.0 - 1.0/PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi   = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb   = F[b_lo:b_hi]
        mag  = np.abs(Fb); phase = np.angle(Fb)
        an   = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh  = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        ce   = (wn*coh + wm*float(coh_mem[i])
                if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk   = np.arange(len(Fb))
        env  = np.clip(1.0+(ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag*env)*np.exp(1j*phase)
    r = np.fft.irfft(F_out, n=N)
    return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    cm = np.zeros(len(bins_phi))
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se); cas.append(se); s = se.copy()
    return cas, cohs

def agente_eco(sinal, bins_phi, n_ciclos=20):
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
    return beta, cas

def concatenar(cas):
    out = cas[0].copy()
    for s in cas[1:]:
        fade_n = min(FADE, len(out), len(s))
        t_fade = np.linspace(0.0, 1.0, fade_n)
        out[-fade_n:] = out[-fade_n:]*(1-t_fade) + s[:fade_n]*t_fade
        out = np.concatenate([out, s[fade_n:]])
    return normalizar(out)

def lowpass(s, fc, fs=FS, order=4):
    b, a = butter(order, fc/(fs/2), btype='low')
    return filtfilt(b, a, s)

# ── gerar sinal ───────────────────────────────────────────────
print("\n  Gerando sinal…")
t_seg = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_seg)))
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_seg
                          + BETA_FM*np.sin(2*np.pi*F_M*t_seg)))
x_mix = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)
beta_f, cas = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal    = concatenar(cas)
dur      = len(sinal) / FS
t_full   = np.arange(len(sinal)) / FS
env_full = lowpass(np.abs(hilbert(sinal)), 200.0)
print(f"  {dur:.2f}s  β_max={beta_f.max():.4f}  φ³={PHI**3:.4f}")

# ── cor transitando P→S→T ────────────────────────────────────
_C = {
    'P': np.array([0.0,   1.0,   0.533]),   # #00FF88
    'S': np.array([1.0,   0.722, 0.0  ]),   # #FFB800
    'T': np.array([1.0,   0.267, 0.400]),   # #FF4466
}
_PT = [(4.10, _C['P']), (5.50, _C['S']), (7.10, _C['T'])]

def cor_rgb(t):
    (t0, c0), (t1, c1), (t2, c2) = _PT
    if t <= t0:   return c0
    elif t <= t1: return (1-(t-t0)/(t1-t0))*c0 + ((t-t0)/(t1-t0))*c1
    elif t <= t2: return (1-(t-t1)/(t2-t1))*c1 + ((t-t1)/(t2-t1))*c2
    else:         return c2

# ── parâmetros da animação ────────────────────────────────────
JANELA   = 1.5     # meia-janela (±1.5 s — textura de barras + arcos visíveis)
FPS      = 24
DUR_ANIM = 24      # segundos de animação

N_FRAMES   = int(FPS * DUR_ANIM)
t_centers  = np.linspace(JANELA, dur - JANELA, N_FRAMES)

COR_BG  = '#0D0D1A'
COR_TXT = '#CCCCDD'
COR_GRD = '#22223A'

# posições dos pontos de dobra para marcação sutil
T_DOBRAS = [4.10, 5.50, 7.10]

# ── montar animação ───────────────────────────────────────────
print("\n  Montando animação…")
print(f"  {N_FRAMES} frames · {FPS} fps · {DUR_ANIM}s · janela ±{JANELA:.1f}s")

fig, ax = plt.subplots(figsize=(13, 3.8))
fig.patch.set_facecolor('#080810')

def animate(i):
    ax.cla()
    ax.set_facecolor(COR_BG)
    ax.set_ylim(-1.08, 1.08)
    for sp in ax.spines.values(): sp.set_color(COR_GRD)
    ax.tick_params(colors=COR_TXT, labelsize=7)

    tc  = t_centers[i]
    t0  = tc - JANELA
    t1  = tc + JANELA
    i0  = int(t0 * FS)
    i1  = int(t1 * FS)

    seg   = sinal[i0:i1]
    t_w   = t_full[i0:i1]
    env_w = env_full[i0:i1]

    cor = cor_rgb(tc)

    # renderização idêntica ao gráfico verde baseline — sem envelope separado
    ax.plot(t_w, seg, color=cor, lw=0.6, alpha=0.9)

    # pontos de dobra visíveis na janela — linhas apenas, sem texto
    for td in T_DOBRAS:
        if t0 < td < t1:
            ax.axvline(td, color='white', lw=0.6, ls=':', alpha=0.30)

    ax.set_xlim(t0, t1)
    ax.set_xlabel('Tempo (s)', color=COR_TXT, fontsize=7)
    ax.set_ylabel('Amp',       color=COR_TXT, fontsize=7)

    progresso = tc / dur
    ax.set_title(
        f'AlphaPhi · Fluxo  —  t = {tc:.3f}s  '
        f'({int(progresso*100)}%  do sinal)',
        color=COR_TXT, fontsize=8
    )
    return []

anim = animation.FuncAnimation(
    fig, animate, frames=N_FRAMES,
    interval=1000/FPS, blit=False
)

# ── salvar ────────────────────────────────────────────────────
fname = '/content/alphaphi_fluxo.mp4'
writer = animation.FFMpegWriter(
    fps=FPS, bitrate=2400,
    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
)
print("  Renderizando… (pode demorar alguns minutos)")
anim.save(fname, writer=writer, dpi=110,
          savefig_kwargs={'facecolor': '#080810'})
plt.close()
print(f"  → alphaphi_fluxo.mp4  ({DUR_ANIM}s · {N_FRAMES} frames)")

display(Video(fname, embed=True, width=900))

print(f"\n{'='*60}")
print(f"  Fluxo animado gerado.")
print(f"  Janela: ±{int(JANELA*1000)} ms")
print(f"  Cor: P=#00FF88 → S=#FFB800 → T=#FF4466 (interpolada)")
print(f"  Observe a grade emergindo ao aproximar do T (7.1s)")
print(f"{'='*60}")
