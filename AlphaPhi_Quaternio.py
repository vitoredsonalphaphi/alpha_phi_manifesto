"""
AlphaPhi_Quaternio.py
Atrator do Envelope — Embedding de Atraso

Três coordenadas extraídas do próprio envelope do sinal:
  x = env(t)
  y = env(t + τ)      τ = 250 ms
  z = env(t + 2τ)

O envelope é a resposta ao hilbert transform do sinal completo.
Nenhuma informação externa adicionada — apenas o sinal observado
em três momentos separados por τ.

O atrator resultante muda de forma nos pontos de dobra:
fase digital → desenvolvimento dos arcos → campo estabilizado.

τ = 250ms foi escolhido por ser da ordem da duração de um segmento
de transição visível no gráfico verde (arcos do envelope).

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt, hilbert
from IPython.display import display, Video, Image

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
print("  AlphaPhi · Atrator do Envelope")
print("  Embedding de atraso — env(t) × env(t+τ) × env(t+2τ)")
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

# ── envelope suavizado ────────────────────────────────────────
env_full = lowpass(np.abs(hilbert(sinal)), 50.0)   # 50 Hz → suave
env_full = env_full / (env_full.max() + 1e-10)

# subamostrar a 200 Hz
SS       = FS // 200
env_sub  = env_full[::SS]
t_sub    = t_full[::SS]
N_env    = len(env_sub)

# ── embedding de atraso — τ = 250 ms = 50 passos a 200 Hz ────
TAU      = 50   # 250 ms
N_TRAJ   = N_env - 2 * TAU

x_traj = env_sub[:N_TRAJ]
y_traj = env_sub[TAU : N_TRAJ + TAU]
z_traj = env_sub[2*TAU : N_TRAJ + 2*TAU]
t_traj = t_sub[:N_TRAJ]

print(f"\n  Atrator: {N_TRAJ} pontos")
print(f"  τ = {TAU * (1/200)*1000:.0f} ms")
print(f"  Amplitude: [{x_traj.min():.3f}, {x_traj.max():.3f}]")

# ── cores por fase ────────────────────────────────────────────
T_P, T_S, T_T = 4.10, 5.50, 7.10
COR_P = np.array([0.0,  1.0,  0.533])
COR_S = np.array([1.0,  0.722, 0.0  ])
COR_T = np.array([1.0,  0.267, 0.400])
COR_0 = np.array([0.15, 0.15, 0.35 ])

def cor_fase(t):
    if t < T_P:
        r = np.clip(t / T_P, 0, 1)
        return (1-r)*COR_0 + r*COR_P
    elif t < T_S:
        r = (t - T_P) / (T_S - T_P)
        return (1-r)*COR_P + r*COR_S
    elif t < T_T:
        r = (t - T_S) / (T_T - T_S)
        return (1-r)*COR_S + r*COR_T
    else:
        return COR_T

cores = np.array([cor_fase(t) for t in t_traj])

COR_BG  = '#080810'
COR_TXT = '#CCCCDD'

# ── imagem estática — 4 ângulos ───────────────────────────────
print("\n  Imagem estática (4 ângulos)…")
fig_s, axes_s = plt.subplots(1, 4, figsize=(18, 5),
                              subplot_kw={'projection': '3d'})
fig_s.patch.set_facecolor(COR_BG)
for ax_s, azim in zip(axes_s, [30, 80, 130, 180]):
    ax_s.set_facecolor(COR_BG)
    seg = max(1, N_TRAJ // 60)
    for k in range(0, N_TRAJ - seg, seg):
        ax_s.plot(x_traj[k:k+seg+1],
                  y_traj[k:k+seg+1],
                  z_traj[k:k+seg+1],
                  color=cores[k], lw=1.0, alpha=0.85)
    for t_d, lbl, cor in [(T_P,'P','#00FF88'),(T_S,'S','#FFB800'),(T_T,'T','#FF4466')]:
        idx = np.argmin(np.abs(t_traj - t_d))
        ax_s.scatter([x_traj[idx]], [y_traj[idx]], [z_traj[idx]],
                     c=cor, s=60, zorder=5)
        ax_s.text(x_traj[idx], y_traj[idx], z_traj[idx],
                  f' {lbl}', color=cor, fontsize=8)
    ax_s.set_xlabel('env(t)', color=COR_TXT, fontsize=7)
    ax_s.set_ylabel('env(t+τ)', color=COR_TXT, fontsize=7)
    ax_s.set_zlabel('env(t+2τ)', color=COR_TXT, fontsize=7)
    ax_s.tick_params(colors=COR_TXT, labelsize=5)
    for pane in [ax_s.xaxis.pane, ax_s.yaxis.pane, ax_s.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('#22223A')
    ax_s.view_init(elev=25, azim=azim)
    ax_s.set_title(f'azim={azim}°', color=COR_TXT, fontsize=8)

fig_s.suptitle(
    f'AlphaPhi · Atrator do Envelope  τ=250ms\n'
    f'azul→verde=antes P · verde→âmbar=P→S · âmbar→vermelho=S→T',
    color=COR_TXT, fontsize=9)
plt.tight_layout()
fname_s = '/content/alphaphi_atrator_estatico.png'
plt.savefig(fname_s, dpi=130, bbox_inches='tight', facecolor=COR_BG)
plt.close()
print(f"  → alphaphi_atrator_estatico.png")
display(Image(fname_s))

# ── animação ──────────────────────────────────────────────────
FPS      = 24
DUR_ANIM = 33
N_FRAMES = int(FPS * DUR_ANIM)

DOBRAS_3D = [
    (T_P, 'P', '#00FF88'),
    (T_S, 'S', '#FFB800'),
    (T_T, 'T', '#FF4466'),
]

print(f"\n  Animação: {N_FRAMES} frames · {FPS}fps · {DUR_ANIM}s")

fig = plt.figure(figsize=(9, 7))
fig.patch.set_facecolor(COR_BG)

def animate(i):
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(COR_BG)

    t_atual = dur * (i + 1) / N_FRAMES
    i_now   = min(int(t_atual * 200), N_TRAJ - 1)

    # trajetória completa — tênue
    ax.plot(x_traj, y_traj, z_traj,
            color='#1a2a1a', lw=0.5, alpha=0.30)

    # trajetória percorrida — colorida por fase
    if i_now > 1:
        seg = max(1, i_now // 50)
        for k in range(0, i_now - seg, seg):
            ax.plot(x_traj[k:k+seg+1],
                    y_traj[k:k+seg+1],
                    z_traj[k:k+seg+1],
                    color=cores[k], lw=1.0, alpha=0.85)

    # ponto atual
    cor_now = cores[min(i_now, len(cores)-1)]
    ax.scatter([x_traj[i_now]], [y_traj[i_now]], [z_traj[i_now]],
               c=[cor_now], s=30, zorder=5)

    # dobras
    for t_d, lbl, cor in DOBRAS_3D:
        if t_atual >= t_d:
            idx = np.argmin(np.abs(t_traj - t_d))
            ax.scatter([x_traj[idx]], [y_traj[idx]], [z_traj[idx]],
                       c=cor, s=60, zorder=6)
            ax.text(x_traj[idx], y_traj[idx], z_traj[idx],
                    f' {lbl}', color=cor, fontsize=8)

    ax.set_xlabel('env(t)', color=COR_TXT, fontsize=7, labelpad=3)
    ax.set_ylabel('env(t+τ)', color=COR_TXT, fontsize=7, labelpad=3)
    ax.set_zlabel('env(t+2τ)', color=COR_TXT, fontsize=7, labelpad=3)
    ax.tick_params(colors=COR_TXT, labelsize=5)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('#22223A')

    ax.view_init(elev=25, azim=30 + (i / N_FRAMES) * 120)
    ax.set_title(
        f'AlphaPhi · Atrator  t={t_atual:.2f}s/{dur:.2f}s  τ=250ms\n'
        f'env(t) × env(t+τ) × env(t+2τ)',
        color=COR_TXT, fontsize=8, pad=8)
    return []

anim = animation.FuncAnimation(
    fig, animate, frames=N_FRAMES,
    interval=1000/FPS, blit=False
)

fname = '/content/alphaphi_atrator.mp4'
writer = animation.FFMpegWriter(
    fps=FPS, bitrate=3000,
    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
)
print("  Renderizando…")
anim.save(fname, writer=writer, dpi=120,
          savefig_kwargs={'facecolor': COR_BG})
plt.close()
print(f"  → alphaphi_atrator.mp4  ({DUR_ANIM}s · {N_FRAMES} frames)")

display(Video(fname, embed=True, width=800))

print(f"\n{'='*60}")
print(f"  Atrator do envelope — embedding de atraso τ=250ms")
print(f"  P={T_P}s  S={T_S}s  T={T_T}s")
print(f"  β_max={beta_f.max():.4f}  φ³={PHI**3:.4f}")
print(f"{'='*60}")
