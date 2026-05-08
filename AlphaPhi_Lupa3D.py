"""
AlphaPhi_Lupa3D.py
Lupa 2D + Visão 3D da Geometria Esférica nos Pontos de Dobra

A partir do mesmo sinal e código que gerou o gráfico verde:

  Figura 1 — LUPA 2D: zoom ±80ms em cada ponto de dobra
    · Sinal bruto (linhas verdes individuais visíveis)
    · Envelope de amplitude preenchido
    · Arco ajustado (círculo) sobre o pico do envelope
    · Raio de curvatura R anotado

  Figura 2 — HÉLICE 3D: sinal analítico z(t) = x(t) + j·H[x(t)]
    · Eixo X = tempo, Y = parte real, Z = parte imaginária
    · A hélice tem raio = envelope instantâneo
    · Nos pontos de dobra: esferas de arame (latitude rings)
    · Projeção da hélice no plano inferior (o "gráfico verde" visto de cima)

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.signal import hilbert, butter, filtfilt
from scipy.io import wavfile
from IPython.display import Image, display

# ── constantes ORIGINAIS ──────────────────────────────────────
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

print("=" * 60)
print("  AlphaPhi · Lupa 2D + Hélice 3D · Geometria Esférica")
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

def concatenar(cas, fade=None):
    if fade is None: fade = int(0.15 * FS)
    out = cas[0].copy()
    for s in cas[1:]:
        fade_n = min(fade, len(out), len(s))
        t_fade = np.linspace(0.0, 1.0, fade_n)
        out[-fade_n:] = out[-fade_n:]*(1-t_fade) + s[:fade_n]*t_fade
        out = np.concatenate([out, s[fade_n:]])
    return normalizar(out)

def lowpass(s, fc, fs=FS, order=4):
    b, a = butter(order, fc/(fs/2), btype='low')
    return filtfilt(b, a, s)

# ── gerar sinal ───────────────────────────────────────────────
print("\n  Gerando sinal original…")
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_sig)))
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_sig + BETA_FM*np.sin(2*np.pi*F_M*t_sig)))
x_mix = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)
beta_f, cas = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal = concatenar(cas)
dur   = len(sinal) / FS
t_full = np.arange(len(sinal)) / FS
print(f"  Sinal: {dur:.2f}s  β_max={beta_f.max():.4f}")

# sinal analítico
z_analitico = hilbert(sinal)
env_full    = np.abs(z_analitico)
x_real      = np.real(z_analitico)
x_imag      = np.imag(z_analitico)

# ── pontos de dobra ───────────────────────────────────────────
DOBRAS = [
    {'nome': 'P', 't': 4.10, 'cor': '#00FF88', 'label': 'P  4.1s — onset'},
    {'nome': 'S', 't': 5.50, 'cor': '#FFB800', 'label': 'S  5.5s — encorpando'},
    {'nome': 'T', 't': 7.10, 'cor': '#FF4466', 'label': 'T  7.1s — campo firmado'},
]
LUPA_MS = 80    # ±80ms = janela de 160ms

# ── FIGURA 1: LUPA 2D ────────────────────────────────────────
print("\n  Gerando Figura 1 — Lupa 2D…")
COR_BG = '#0D0D1A'; COR_TXT = '#CCCCDD'

fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
fig1.patch.set_facecolor('#080810')

for ax, d in zip(axes, DOBRAS):
    tc  = d['t']
    cor = d['cor']
    i0  = max(0, int((tc - LUPA_MS/1000) * FS))
    i1  = min(len(sinal), int((tc + LUPA_MS/1000) * FS))
    seg = sinal[i0:i1]
    t_w = t_full[i0:i1]
    env_w = lowpass(np.abs(hilbert(seg)), 200.0)

    # ── ajuste de arco circular sobre o pico do envelope ─────
    env_norm = env_w / (env_w.max() + 1e-12)
    mask_top = env_norm > 0.55                  # topo do arco
    t_top    = t_w[mask_top] - tc               # centralizado
    e_top    = env_w[mask_top]
    if len(t_top) > 3:
        coeffs = np.polyfit(t_top, e_top, 2)   # parábola ≈ arco circular
        a_coef = coeffs[0]
        R_curv = abs(1.0 / (2 * a_coef + 1e-12))  # raio de curvatura (s)
        t_arc  = np.linspace(t_top[0], t_top[-1], 300)
        e_arc  = np.polyval(coeffs, t_arc)
    else:
        R_curv = float('nan'); t_arc = np.array([]); e_arc = np.array([])

    ax.set_facecolor(COR_BG)
    # sinal bruto — linhas finas (o "gráfico verde" com lupa)
    ax.plot(t_w, seg, color=cor, lw=0.28, alpha=0.80)
    # envelope preenchido
    ax.fill_between(t_w, -env_w, env_w, color=cor, alpha=0.14)
    ax.plot(t_w,  env_w, color=cor, lw=1.6, alpha=0.85)
    ax.plot(t_w, -env_w, color=cor, lw=1.6, alpha=0.85)
    # arco ajustado
    if len(t_arc):
        ax.plot(t_arc + tc, e_arc,  color='white', lw=2.2, ls='--',
                alpha=0.90, label=f'Arco  R≈{R_curv*1000:.1f}ms')
        ax.plot(t_arc + tc, -e_arc, color='white', lw=2.2, ls='--', alpha=0.90)
    # marcador central
    ax.axvline(tc, color='white', lw=1.0, ls=':', alpha=0.5)
    ax.set_title(d['label'] + f'\nR curvatura ≈ {R_curv*1000:.1f} ms',
                 color=COR_TXT, fontsize=9)
    ax.set_xlabel('t (s)', color=COR_TXT, fontsize=8)
    ax.set_ylabel('Amp', color=COR_TXT, fontsize=8)
    ax.tick_params(colors=COR_TXT, labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#22223A')
    ax.set_xlim(t_w[0], t_w[-1])
    if len(t_arc):
        ax.legend(fontsize=8, facecolor='#111', labelcolor='white', loc='upper right')

    R_ms = R_curv * 1000
    print(f"  {d['nome']} ({tc:.1f}s)  raio de curvatura = {R_ms:.1f} ms")

fig1.suptitle(
    f'AlphaPhi · LUPA 2D — Arcos do Campo Esférico (±{LUPA_MS}ms)\n'
    f'Beep {F_BEEP:.0f}Hz · α*=1/3 · Linha tracejada branca = arco circular ajustado',
    color=COR_TXT, fontsize=11, y=1.03
)
plt.tight_layout()
plt.savefig('/content/lupa_arcos_2d.png', dpi=160, bbox_inches='tight',
            facecolor='#080810')
plt.close()
print("  → lupa_arcos_2d.png")
display(Image('/content/lupa_arcos_2d.png'))

# ── FIGURA 2: HÉLICE 3D ──────────────────────────────────────
print("\n  Gerando Figura 2 — Hélice 3D…")

DS = 30        # decimação: 44100 → 1470 pts/s (suficiente para forma)
t_ds   = t_full[::DS]
xr_ds  = x_real[::DS]
xi_ds  = x_imag[::DS]
env_ds = env_full[::DS]

# normalizar envelope para colormap
env_n = (env_ds - env_ds.min()) / (env_ds.max() - env_ds.min() + 1e-12)

fig2 = plt.figure(figsize=(16, 10), facecolor='#080810')
ax3d = fig2.add_subplot(111, projection='3d')
ax3d.set_facecolor('#080810')

# ── hélice segmentada com cor por envelope ────────────────────
cmap = plt.get_cmap('plasma')
N_SEG = len(t_ds) - 1
for i in range(0, N_SEG, 3):          # a cada 3 para velocidade
    c = cmap(env_n[i])
    ax3d.plot(t_ds[i:i+2], xr_ds[i:i+2], xi_ds[i:i+2],
              color=c, lw=0.6, alpha=0.55)

# ── projeção no plano inferior (o "gráfico verde" de cima) ────
z_floor = -1.35
ax3d.plot(t_ds, xr_ds, np.full_like(t_ds, z_floor),
          color='#00FF88', lw=0.25, alpha=0.30)

# ── esferas de arame em cada ponto de dobra ───────────────────
THETA = np.linspace(0, 2*np.pi, 80)
for d in DOBRAS:
    tc  = d['t']
    cor = d['cor']
    i_c = int(tc * FS)
    r_c = float(env_full[i_c])        # raio = envelope no ponto

    # largura temporal da esfera (80ms de cada lado)
    dt_esf = 0.08
    n_aneis = 28
    t_aneis = np.linspace(tc - dt_esf, tc + dt_esf, n_aneis)

    for t_anel in t_aneis:
        frac = np.sqrt(max(0.0, 1.0 - ((t_anel - tc)/dt_esf)**2))
        r_anel = r_c * frac
        if r_anel < 0.02:
            continue
        ax3d.plot([t_anel]*len(THETA),
                  r_anel * np.cos(THETA),
                  r_anel * np.sin(THETA),
                  color=cor, lw=0.9, alpha=0.55)

    # anel equatorial mais espesso
    ax3d.plot([tc]*len(THETA),
              r_c * np.cos(THETA),
              r_c * np.sin(THETA),
              color=cor, lw=2.5, alpha=0.95,
              label=f'{d["label"]}  r={r_c:.3f}')

    # linha vertical ao plano de projeção
    ax3d.plot([tc, tc], [0, 0], [z_floor, -r_c],
              color=cor, lw=0.8, ls=':', alpha=0.4)

# ── eixos e câmera ────────────────────────────────────────────
ax3d.set_xlabel('Tempo (s)', color=COR_TXT, fontsize=9, labelpad=8)
ax3d.set_ylabel('Re[z(t)]',  color=COR_TXT, fontsize=9, labelpad=8)
ax3d.set_zlabel('Im[z(t)]',  color=COR_TXT, fontsize=9, labelpad=8)
ax3d.tick_params(colors=COR_TXT, labelsize=7)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('#22223A')
ax3d.yaxis.pane.set_edgecolor('#22223A')
ax3d.zaxis.pane.set_edgecolor('#22223A')
ax3d.set_xlim(0, dur)
ax3d.set_ylim(-1.4, 1.4)
ax3d.set_zlim(z_floor, 1.4)
ax3d.view_init(elev=22, azim=-55)

ax3d.legend(fontsize=8, facecolor='#111', labelcolor=COR_TXT,
            loc='upper left', framealpha=0.6)

fig2.suptitle(
    f'AlphaPhi · HÉLICE 3D — z(t) = x(t) + j·H[x(t)]\n'
    f'Raio da hélice = envelope instantâneo · Esferas = pontos de dobra\n'
    f'Projeção inferior = sinal 2D (gráfico verde visto de topo)',
    color=COR_TXT, fontsize=10, y=1.01
)
plt.tight_layout()
plt.savefig('/content/helice_3d_dobras.png', dpi=160, bbox_inches='tight',
            facecolor='#080810')
plt.close()
print("  → helice_3d_dobras.png")
display(Image('/content/helice_3d_dobras.png'))

print(f"\n{'='*60}")
print(f"  GEOMETRIA ESFÉRICA — RESUMO")
print(f"{'='*60}")
for d in DOBRAS:
    i_c = int(d['t'] * FS)
    r   = float(env_full[i_c])
    print(f"  {d['nome']} ({d['t']:.1f}s)  raio esférico = {r:.4f}  "
          f"(RMS local ≈ {r/np.sqrt(2):.4f})")
print(f"\n  A hélice 3D demonstra:")
print(f"  · Cada ponto de dobra = expansão do raio da hélice = esfera")
print(f"  · A projeção 2D (gráfico verde) = secção plana dessa geometria")
print(f"  · Os arcos que você viu são as 'sombras' das esferas φ")
print(f"{'='*60}")
