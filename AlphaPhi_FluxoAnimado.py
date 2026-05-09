"""
AlphaPhi_FluxoAnimado.py
Osciloscópio com Persistência — Frequência em Movimento

O sinal chega na borda direita. O passado recede para a esquerda.
O presente é sempre o ponto mais brilhante.
Nenhuma imagem estática. Nenhuma câmera.
A frequência produz a imagem enquanto se move.

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
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
print("  AlphaPhi · Frequência em Movimento")
print("  Osciloscópio com persistência · 880Hz · α*=1/3")
print("=" * 60)

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

# ── parâmetros ────────────────────────────────────────────────
WINDOW   = DURACAO    # 1.5s — exatamente 1 segmento do eco
FPS      = 24
DUR_ANIM = 33         # 0.25× do áudio
N_FRAMES = int(FPS * DUR_ANIM)
N_FADE   = 16         # camadas de persistência

# pontos de dobra
DOBRAS = [(4.10, 'P', '#00FF88'), (5.50, 'S', '#FFB800'), (7.10, 'T', '#FF4466')]

COR_BG  = '#0D0D1A'
COR_TXT = '#CCCCDD'
COR_GRD = '#22223A'

print(f"\n  {N_FRAMES} frames · {FPS}fps · {DUR_ANIM}s")
print(f"  Janela: {WINDOW:.1f}s · {N_FADE} camadas de persistência")

fig, ax = plt.subplots(figsize=(14, 4))
fig.patch.set_facecolor('#080810')

def animate(i):
    ax.cla()
    ax.set_facecolor(COR_BG)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, WINDOW)
    for sp in ax.spines.values(): sp.set_color(COR_GRD)
    ax.tick_params(colors=COR_TXT, labelsize=7)

    t_now   = dur * (i + 1) / N_FRAMES
    t_start = max(0.0, t_now - WINDOW)
    i0      = int(t_start * FS)
    i1      = min(len(sinal), int(t_now * FS))
    n       = i1 - i0

    if n < 2:
        return []

    seg    = sinal[i0:i1]
    # x = 0 (mais antigo visível) → WINDOW (presente)
    t_disp = np.linspace(0, t_now - t_start, n)

    # persistência em camadas — do mais antigo (esmaecido) ao mais novo (vivo)
    layer  = max(1, n // N_FADE)
    for k in range(N_FADE):
        i_s = k * layer
        i_e = min(n, (k + 1) * layer + 1)
        if i_s >= n:
            break
        progress = (k + 1) / N_FADE        # 0 → 1 (antigo → novo)
        alpha = 0.06 + 0.88 * progress
        lw    = 0.30 + 0.30 * progress
        ax.plot(t_disp[i_s:i_e], seg[i_s:i_e],
                color='#00FF88', lw=lw, alpha=alpha)

    # pontos de dobra — surgem como marcadores de texto quando o presente os alcança
    for t_d, lbl, cor in DOBRAS:
        if t_start <= t_d <= t_now:
            t_rel = t_d - t_start          # posição na janela visível
            ax.axvline(t_rel, color=cor, lw=0.8, ls=':', alpha=0.55)
            ax.text(t_rel + 0.02, 0.88, lbl,
                    color=cor, fontsize=9, va='top', ha='left',
                    transform=ax.get_xaxis_transform())

    ax.set_ylabel('Amplitude', color=COR_TXT, fontsize=8)
    ax.set_xlabel('← passado · · · presente →', color=COR_TXT, fontsize=8)

    # fase do sinal
    if t_now < 4.10:
        fase = 'digital compacto'
    elif t_now < 5.50:
        fase = 'P — onset'
    elif t_now < 7.10:
        fase = 'S — encorpando'
    else:
        fase = 'T — campo'

    ax.set_title(
        f'AlphaPhi · {F_BEEP:.0f}Hz · α*=1/3 · {fase}  '
        f'[t={t_now:.2f}s]',
        color=COR_TXT, fontsize=9
    )
    return []

anim = animation.FuncAnimation(
    fig, animate, frames=N_FRAMES,
    interval=1000/FPS, blit=False
)

fname  = '/content/alphaphi_fluxo.mp4'
writer = animation.FFMpegWriter(
    fps=FPS, bitrate=3000,
    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
)
print("\n  Renderizando…")
anim.save(fname, writer=writer, dpi=150,
          savefig_kwargs={'facecolor': '#080810'})
plt.close()
print(f"  → alphaphi_fluxo.mp4  ({DUR_ANIM}s · {N_FRAMES} frames)")

display(Video(fname, embed=True, width=960))

print(f"\n{'='*60}")
print(f"  Frequência em movimento — osciloscópio com persistência")
print(f"  Janela: {WINDOW:.1f}s · passado esmaece · presente vivo")
print(f"  P=4.1s · S=5.5s · T=7.1s")
print(f"{'='*60}")
