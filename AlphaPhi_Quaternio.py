"""
AlphaPhi_Quaternio.py
Trajetória 3D — Componentes no Espaço de Fase

Três componentes extraídos do próprio sinal eco,
pelas mesmas bandas φ que o agente usa:

  x = banda F_M     ≈ 135.8 Hz  → [84.7 Hz, 137.1 Hz]
  y = banda F_ORG   = 220.0 Hz  → [137.1 Hz, 221.8 Hz]
  z = banda F_BEEP  = 880.0 Hz  → [580.7 Hz, 939.6 Hz]

Nada adicionado. Nenhuma escala ajustada para parecer com algo.
O sinal já contém esses três componentes desde a geração.
O espaço de fase apenas os separa para observação simultânea.

Se emergir uma hélice, estava lá.
Se emergir um atrator, estava lá.
Se emergir uma bifurcação em P/S/T, estava lá.

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt, hilbert
from IPython.display import display, Video

# ── constantes ORIGINAIS — não modificar ─────────────────────
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI          # ≈ 135.8 Hz
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5
N_STEPS    = 5
N_CICLOS   = 20
FADE       = int(0.15 * FS)

print("=" * 60)
print("  AlphaPhi · Trajetória 3D")
print("  Componentes F_M / F_ORG / F_BEEP no espaço de fase")
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

# ── identificar bandas φ que contêm cada frequência ──────────
# as mesmas bandas do agente_eco — nenhuma nova estrutura
bandas_full = gerar_bandas_phi()
def banda_de(freq):
    for f_lo, f_hi in bandas_full:
        if f_lo <= freq < f_hi:
            return f_lo, f_hi
    return bandas_full[-1]

b_fm   = banda_de(F_M)      # banda de F_M   ≈ 135.8 Hz
b_org  = banda_de(F_ORG)    # banda de F_ORG = 220.0 Hz
b_beep = banda_de(F_BEEP)   # banda de F_BEEP = 880.0 Hz

print(f"\n  Bandas φ identificadas (mesmas do eco):")
print(f"  F_M   ≈{F_M:.1f}Hz  → [{b_fm[0]:.1f}, {b_fm[1]:.1f}] Hz")
print(f"  F_ORG = {F_ORG:.1f}Hz → [{b_org[0]:.1f}, {b_org[1]:.1f}] Hz")
print(f"  F_BEEP= {F_BEEP:.1f}Hz → [{b_beep[0]:.1f}, {b_beep[1]:.1f}] Hz")

# ── extrair cada componente via bandpass nas bordas φ ─────────
def bp_envelope(sig, f_lo, f_hi, fs=FS, order=4):
    nyq = fs / 2.0
    lo  = max(f_lo / nyq, 1e-4)
    hi  = min(f_hi / nyq, 0.9999)
    b, a = butter(order, [lo, hi], btype='band')
    filt = filtfilt(b, a, sig)
    env  = np.abs(hilbert(filt))
    return filt, env

print("\n  Extraindo componentes…")
_, env_fm   = bp_envelope(sinal, *b_fm)
_, env_org  = bp_envelope(sinal, *b_org)
_, env_beep = bp_envelope(sinal, *b_beep)

# normalizar cada envelope ao seu próprio máximo
env_fm   = env_fm   / (env_fm.max()   + 1e-10)
env_org  = env_org  / (env_org.max()  + 1e-10)
env_beep = env_beep / (env_beep.max() + 1e-10)

# subamostrar para visualização (200 Hz → ~1650 pontos)
SUBSAMPLE = FS // 200
x_traj = env_fm[::SUBSAMPLE]
y_traj = env_org[::SUBSAMPLE]
z_traj = env_beep[::SUBSAMPLE]
t_traj = t_full[::SUBSAMPLE]
N_TRAJ = len(t_traj)

print(f"  Trajetória: {N_TRAJ} pontos ({200} Hz efetivo)")

# ── mapear cores ao longo do tempo (P→S→T) ───────────────────
T_P, T_S, T_T = 4.10, 5.50, 7.10
COR_P = np.array([0.0,  1.0,  0.533])   # #00FF88
COR_S = np.array([1.0,  0.722, 0.0  ])  # #FFB800
COR_T = np.array([1.0,  0.267, 0.400])  # #FF4466
COR_0 = np.array([0.2,  0.2,  0.3  ])   # fase inicial — azul escuro

def cor_fase(t):
    if t < T_P:
        r = np.clip((t / T_P), 0, 1)
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

# ── parâmetros da animação ────────────────────────────────────
FPS      = 24
DUR_ANIM = 33    # 0.25× do áudio
N_FRAMES = int(FPS * DUR_ANIM)
TAIL     = int(N_TRAJ * 1.0)   # trajetória inteira visível (tênue)

print(f"\n  Animação: {N_FRAMES} frames · {FPS} fps · {DUR_ANIM}s")

COR_BG  = '#080810'
COR_TXT = '#CCCCDD'

# ── figura 3D ─────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor(COR_BG)
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor(COR_BG)

# trajetória completa — tênue (o "potencial")
ax.plot(x_traj, y_traj, z_traj,
        color='#334433', lw=0.5, alpha=0.25, zorder=1)

# linha ativa (atualizada a cada frame)
_line, = ax.plot([], [], [], lw=1.2, alpha=0.90, zorder=3)
_dot,  = ax.plot([], [], [], 'o', ms=5, zorder=4)

# marcadores de dobra
DOBRAS_3D = [
    (T_P, 'P', '#00FF88'),
    (T_S, 'S', '#FFB800'),
    (T_T, 'T', '#FF4466'),
]
_dobra_dots = []
for t_d, lbl, cor in DOBRAS_3D:
    idx = np.argmin(np.abs(t_traj - t_d))
    xd, yd, zd = x_traj[idx], y_traj[idx], z_traj[idx]
    dot, = ax.plot([xd], [yd], [zd], 'o', ms=8,
                   color=cor, alpha=0.0, zorder=5)
    _dobra_dots.append((t_d, dot, xd, yd, zd, lbl, cor))

# estética
ax.set_xlabel(f'F_M {F_M:.0f}Hz', color=COR_TXT, fontsize=8, labelpad=6)
ax.set_ylabel(f'F_ORG {F_ORG:.0f}Hz', color=COR_TXT, fontsize=8, labelpad=6)
ax.set_zlabel(f'F_BEEP {F_BEEP:.0f}Hz', color=COR_TXT, fontsize=8, labelpad=6)
ax.tick_params(colors=COR_TXT, labelsize=6)
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#22223A')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)

_title = ax.set_title('', color=COR_TXT, fontsize=9, pad=10)

def animate(i):
    t_atual = dur * (i + 1) / N_FRAMES
    i_now   = min(int(t_atual / (1/200)), N_TRAJ - 1)

    # trajetória percorrida — colorida por fase
    if i_now > 1:
        xs = x_traj[:i_now]
        ys = y_traj[:i_now]
        zs = z_traj[:i_now]
        # cor da porção mais recente (última cor da fase)
        cor_atual = cores[max(0, i_now-1)]
        _line.set_data_3d(xs, ys, zs)
        _line.set_color(cor_atual)

    # ponto atual
    if i_now >= 0:
        cor_atual = cores[min(i_now, len(cores)-1)]
        _dot.set_data_3d([x_traj[i_now]], [y_traj[i_now]], [z_traj[i_now]])
        _dot.set_color(cor_atual)

    # pontos de dobra surgem quando alcançados
    for t_d, dot, xd, yd, zd, lbl, cor in _dobra_dots:
        if t_atual >= t_d:
            dot.set_alpha(0.90)

    # rotação lenta — revela a forma 3D progressivamente
    az = 30 + (i / N_FRAMES) * 120   # 30° → 150°
    ax.view_init(elev=25, azim=az)

    _title.set_text(
        f'AlphaPhi · Espaço de Fase  ·  t = {t_atual:.2f}s / {dur:.2f}s\n'
        f'F_M × F_ORG × F_BEEP  —  bandas φ do eco'
    )
    return [_line, _dot, _title] + [d[1] for d in _dobra_dots]

anim = animation.FuncAnimation(
    fig, animate, frames=N_FRAMES,
    interval=1000/FPS, blit=False
)

# ── salvar ────────────────────────────────────────────────────
fname = '/content/alphaphi_quaternio.mp4'
writer = animation.FFMpegWriter(
    fps=FPS, bitrate=3000,
    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
)
print("\n  Renderizando…")
anim.save(fname, writer=writer, dpi=120,
          savefig_kwargs={'facecolor': COR_BG})
plt.close()
print(f"  → alphaphi_quaternio.mp4  ({DUR_ANIM}s · {N_FRAMES} frames)")

display(Video(fname, embed=True, width=800))

print(f"\n{'='*60}")
print(f"  Trajetória 3D — espaço de fase")
print(f"  x = F_M  {F_M:.1f}Hz  (banda φ [{b_fm[0]:.0f}–{b_fm[1]:.0f}])")
print(f"  y = F_ORG {F_ORG:.0f}Hz  (banda φ [{b_org[0]:.0f}–{b_org[1]:.0f}])")
print(f"  z = F_BEEP {F_BEEP:.0f}Hz (banda φ [{b_beep[0]:.0f}–{b_beep[1]:.0f}])")
print(f"  P={T_P}s  S={T_S}s  T={T_T}s")
print(f"{'='*60}")
