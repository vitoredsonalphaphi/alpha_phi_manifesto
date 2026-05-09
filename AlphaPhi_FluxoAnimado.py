"""
AlphaPhi_FluxoAnimado.py
Animação — O Sinal Se Desenhando

O mesmo gráfico verde, animado progressivamente.
Eixo completo (0–8.25s) sempre visível e fixo.
A linha cresce da esquerda para a direita, revelando
cada fase e cada ponto de dobra conforme os alcança.
Mesma renderização do baseline: lw=0.6, alpha=0.9, #00FF88.

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
print("  Sinal se desenhando · gráfico verde progressivo")
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
print(f"  {dur:.2f}s  β_max={beta_f.max():.4f}  φ³={PHI**3:.4f}")

# ── parâmetros da animação ────────────────────────────────────
FPS      = 24
DUR_ANIM = 33    # 33s = velocidade 0.25x do áudio (8.25 ÷ 0.25)

N_FRAMES = int(FPS * DUR_ANIM)

# pontos de dobra — mesmos do gráfico verde baseline
DOBRAS = [
    (4.10, 'P  4.1s', '#00FF88'),
    (5.50, 'S  5.5s', '#FFB800'),
    (7.10, 'T  7.1s', '#FF4466'),
]

COR_BG  = '#0D0D1A'
COR_TXT = '#CCCCDD'
COR_GRD = '#22223A'

print(f"\n  Montando animação…")
print(f"  {N_FRAMES} frames · {FPS} fps · {DUR_ANIM}s")
print(f"  Sinal se desenha progressivamente: 0 → {dur:.2f}s")

# ── figura — mesmas dimensões do gráfico verde baseline ──────
fig, ax = plt.subplots(figsize=(14, 4))
fig.patch.set_facecolor('#080810')

def animate(i):
    ax.cla()
    ax.set_facecolor(COR_BG)
    ax.set_xlim(0, dur)
    ax.set_ylim(-1.05, 1.05)
    for sp in ax.spines.values(): sp.set_color(COR_GRD)
    ax.tick_params(colors=COR_TXT, labelsize=8)

    # quanto do sinal já foi desenhado
    t_atual = dur * (i + 1) / N_FRAMES
    i_atual = min(int(t_atual * FS), len(sinal))

    # sinal desenhado até agora — idêntico ao gráfico verde
    ax.plot(t_full[:i_atual], sinal[:i_atual],
            color='#00FF88', lw=0.6, alpha=0.9)

    # pontos de dobra aparecem quando a linha os alcança
    for t_d, label, cor in DOBRAS:
        if t_atual >= t_d:
            ax.axvline(t_d, color=cor, lw=1.2, ls='--',
                       alpha=0.70, label=label)

    ax.set_xlabel('Tempo (s)', color=COR_TXT, fontsize=9)
    ax.set_ylabel('Amplitude', color=COR_TXT, fontsize=9)
    ax.legend(fontsize=8, facecolor='#111', labelcolor='#CCCCDD',
              loc='upper left', framealpha=0.7)
    ax.set_title(
        f'AlphaPhi · Beep {F_BEEP:.0f}Hz · α*=1/3 · '
        f't = {t_atual:.2f}s / {dur:.2f}s',
        color=COR_TXT, fontsize=9
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
anim.save(fname, writer=writer, dpi=150,
          savefig_kwargs={'facecolor': '#080810'})
plt.close()
print(f"  → alphaphi_fluxo.mp4  ({DUR_ANIM}s · {N_FRAMES} frames)")

display(Video(fname, embed=True, width=960))

print(f"\n{'='*60}")
print(f"  Gráfico verde animado — sinal se desenhando")
print(f"  {dur:.2f}s de sinal em {DUR_ANIM}s de animação (0.25x)")
print(f"  P=4.1s · S=5.5s · T=7.1s revelados progressivamente")
print(f"{'='*60}")
