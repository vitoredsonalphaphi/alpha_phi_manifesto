"""
AlphaPhi_Lupa3D.py  v2 — sincero
Lupa 2D + Retrato de Fase + Hélice 3D

Princípio: nenhuma forma pré-desenhada. Nenhum arco ajustado.
Nenhuma esfera de arame imposta. O código que gera o sinal não
muda. Apenas a ferramenta de observação é refinada.

  Figura 1 — LUPA 2D (3 painéis)
    · Zoom ±80ms — linhas individuais visíveis como no gráfico verde
    · Envelope natural (lowpass de |hilbert|) preenchido — sem ajuste
    · Resultado: o que está lá, sem adição

  Figura 2 — RETRATO DE FASE (3 painéis)
    · Re[z(t)] × Im[z(t)] para cada janela de 160ms
    · z(t) = sinal analítico (transformada de Hilbert)
    · Circunferência = campo coerente puro
    · Forma emergente = o que o campo realmente é — sem imposição

  Figura 3 — HÉLICE 3D (1 painel)
    · z(t) plotado em (tempo, Re, Im) — raio = envelope instantâneo
    · Colorido pelo envelope (plasma): brilhante = campo forte
    · Projeção no plano inferior = sinal 2D (gráfico verde de cima)
    · Nenhuma geometria adicionada sobre os dados

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.signal import hilbert, butter, filtfilt
from IPython.display import Image, display

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

print("=" * 60)
print("  AlphaPhi · Lupa 2D + Retrato de Fase + Hélice 3D")
print("  (observação sincera — sem interpretação imposta)")
print("=" * 60)

# ── funções eco originais — idênticas ────────────────────────
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

# ── gerar sinal (idêntico) ────────────────────────────────────
print("\n  Gerando sinal original…")
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_sig)))
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_sig + BETA_FM*np.sin(2*np.pi*F_M*t_sig)))
x_mix = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)
beta_f, cas = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal  = concatenar(cas)
dur    = len(sinal) / FS
t_full = np.arange(len(sinal)) / FS

# sinal analítico — z(t) = x(t) + j·H[x(t)]
z_anal  = hilbert(sinal)
env_full = np.abs(z_anal)
xr_full  = np.real(z_anal)
xi_full  = np.imag(z_anal)
print(f"  Sinal: {dur:.2f}s  β_max={beta_f.max():.4f}  φ³={PHI**3:.4f}")

# ── pontos de dobra observados ────────────────────────────────
DOBRAS = [
    {'nome': 'P', 't': 4.10, 'cor': '#00FF88'},
    {'nome': 'S', 't': 5.50, 'cor': '#FFB800'},
    {'nome': 'T', 't': 7.10, 'cor': '#FF4466'},
]
LUPA_MS = 80    # ±80ms de cada lado

COR_BG  = '#0D0D1A'
COR_TXT = '#CCCCDD'
COR_GRD = '#22223A'

# ═══════════════════════════════════════════════════════════════
# FIGURA 1 — LUPA 2D
# Mesmo sinal, mais resolução — nenhuma curva adicionada
# ═══════════════════════════════════════════════════════════════
print("\n  Gerando Figura 1 — Lupa 2D…")

fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
fig1.patch.set_facecolor('#080810')

for ax, d in zip(axes1, DOBRAS):
    tc  = d['t']
    cor = d['cor']
    i0  = max(0, int((tc - LUPA_MS/1000) * FS))
    i1  = min(len(sinal), int((tc + LUPA_MS/1000) * FS))
    seg  = sinal[i0:i1]
    t_w  = t_full[i0:i1]
    env_w = lowpass(np.abs(hilbert(seg)), 200.0)

    ax.set_facecolor(COR_BG)

    # envelope natural preenchido — sem ajuste, sem curva adicionada
    ax.fill_between(t_w, -env_w, env_w, color=cor, alpha=0.13)
    ax.plot(t_w,  env_w, color=cor, lw=1.4, alpha=0.70)
    ax.plot(t_w, -env_w, color=cor, lw=1.4, alpha=0.70)

    # sinal bruto — linhas finas, mesmo estilo do gráfico verde
    ax.plot(t_w, seg, color=cor, lw=0.25, alpha=0.85)

    # marcador do ponto de dobra — apenas posição, sem interpretação
    ax.axvline(tc, color='white', lw=0.8, ls=':', alpha=0.40)

    ax.set_title(f'{d["nome"]}  t={tc:.2f}s', color=COR_TXT, fontsize=10)
    ax.set_xlabel('t (s)', color=COR_TXT, fontsize=8)
    ax.set_ylabel('Amp', color=COR_TXT, fontsize=8)
    ax.tick_params(colors=COR_TXT, labelsize=7)
    for sp in ax.spines.values(): sp.set_color(COR_GRD)
    ax.set_xlim(t_w[0], t_w[-1])

fig1.suptitle(
    f'Figura 1 — LUPA 2D  ±{LUPA_MS}ms  ·  Beep {F_BEEP:.0f}Hz α*=1/3\n'
    f'Sinal bruto + envelope natural — nenhuma curva adicionada',
    color=COR_TXT, fontsize=11, y=1.03
)
plt.tight_layout()
plt.savefig('/content/lupa_2d.png', dpi=170,
            bbox_inches='tight', facecolor='#080810')
plt.close()
display(Image('/content/lupa_2d.png'))
print("  → lupa_2d.png")

# ═══════════════════════════════════════════════════════════════
# FIGURA 2 — RETRATO DE FASE
# Re[z(t)] × Im[z(t)] — a trajetória do campo no plano complexo
# A forma que emerge é a forma real — não desenhada
# ═══════════════════════════════════════════════════════════════
print("\n  Gerando Figura 2 — Retrato de Fase…")

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.patch.set_facecolor('#080810')

for ax, d in zip(axes2, DOBRAS):
    tc  = d['t']
    cor = d['cor']
    i0  = max(0, int((tc - LUPA_MS/1000) * FS))
    i1  = min(len(sinal), int((tc + LUPA_MS/1000) * FS))
    xr_w = xr_full[i0:i1]
    xi_w = xi_full[i0:i1]
    env_w = env_full[i0:i1]

    # colorir por tempo para ver a trajetória — do mais escuro ao mais brilhante
    n = len(xr_w)
    cmap_local = plt.get_cmap('plasma')
    for k in range(0, n-1, 4):
        c = cmap_local(k / n)
        ax.plot(xr_w[k:k+2], xi_w[k:k+2], color=c, lw=0.4, alpha=0.75)

    # envelope médio como escala de referência
    r_medio = float(np.mean(env_w))
    ax.set_facecolor(COR_BG)
    ax.set_aspect('equal')
    ax.tick_params(colors=COR_TXT, labelsize=7)
    for sp in ax.spines.values(): sp.set_color(COR_GRD)
    ax.set_title(f'{d["nome"]}  t={tc:.2f}s\nr médio={r_medio:.3f}',
                 color=COR_TXT, fontsize=9)
    ax.set_xlabel('Re[z]', color=COR_TXT, fontsize=8)
    ax.set_ylabel('Im[z]', color=COR_TXT, fontsize=8)
    # origem
    ax.axhline(0, color=COR_GRD, lw=0.5, alpha=0.5)
    ax.axvline(0, color=COR_GRD, lw=0.5, alpha=0.5)

fig2.suptitle(
    f'Figura 2 — RETRATO DE FASE  Re[z(t)] × Im[z(t)]  ±{LUPA_MS}ms\n'
    f'Cor = progressão do tempo (plasma).  Forma = geometria real do campo.',
    color=COR_TXT, fontsize=11, y=1.03
)
plt.tight_layout()
plt.savefig('/content/retrato_fase.png', dpi=170,
            bbox_inches='tight', facecolor='#080810')
plt.close()
display(Image('/content/retrato_fase.png'))
print("  → retrato_fase.png")

# ═══════════════════════════════════════════════════════════════
# FIGURA 3 — HÉLICE 3D
# z(t) no espaço (t, Re, Im) — raio = envelope, cor = envelope
# Nenhuma geometria adicionada além dos dados
# ═══════════════════════════════════════════════════════════════
print("\n  Gerando Figura 3 — Hélice 3D…")

DS = 25   # decimação: 44100 → ~1764 pts/s
t_ds   = t_full[::DS]
xr_ds  = xr_full[::DS]
xi_ds  = xi_full[::DS]
env_ds = env_full[::DS]
env_n  = (env_ds - env_ds.min()) / (env_ds.max() - env_ds.min() + 1e-12)

fig3 = plt.figure(figsize=(16, 9), facecolor='#080810')
ax3d = fig3.add_subplot(111, projection='3d')
ax3d.set_facecolor('#080810')

cmap3d = plt.get_cmap('plasma')
N_pts  = len(t_ds)

# hélice segmentada por cor de envelope — sem sobreposição de formas
for i in range(0, N_pts - 1, 2):
    c = cmap3d(env_n[i])
    ax3d.plot(t_ds[i:i+2], xr_ds[i:i+2], xi_ds[i:i+2],
              color=c, lw=0.7, alpha=0.60)

# projeção no plano inferior — o gráfico verde visto de cima
z_floor = -1.45
ax3d.plot(t_ds, xr_ds, np.full_like(t_ds, z_floor),
          color='#00FF88', lw=0.20, alpha=0.25)

# marcadores dos pontos de dobra: apenas linha vertical pontilhada no eixo
# sem esferas, sem wireframes — só indica a posição no tempo
for d in DOBRAS:
    tc = d['t']
    i_c = int(tc * FS // DS)
    if i_c < len(t_ds):
        r_c = float(env_ds[i_c])
        ax3d.plot([tc, tc], [xr_ds[i_c], xr_ds[i_c]],
                  [z_floor, xi_ds[i_c]],
                  color=d['cor'], lw=1.0, ls=':', alpha=0.55,
                  label=f"{d['nome']} {tc:.1f}s")

# eixos
ax3d.set_xlabel('Tempo (s)', color=COR_TXT, fontsize=9, labelpad=8)
ax3d.set_ylabel('Re[z(t)]',  color=COR_TXT, fontsize=9, labelpad=8)
ax3d.set_zlabel('Im[z(t)]',  color=COR_TXT, fontsize=9, labelpad=8)
ax3d.tick_params(colors=COR_TXT, labelsize=7)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor(COR_GRD)
ax3d.yaxis.pane.set_edgecolor(COR_GRD)
ax3d.zaxis.pane.set_edgecolor(COR_GRD)
ax3d.set_xlim(0, dur)
ax3d.set_ylim(-1.4, 1.4)
ax3d.set_zlim(z_floor, 1.4)
ax3d.view_init(elev=20, azim=-52)
ax3d.legend(fontsize=9, facecolor='#111', labelcolor=COR_TXT,
            loc='upper left', framealpha=0.6)

fig3.suptitle(
    f'Figura 3 — HÉLICE 3D  z(t) = x(t) + j·H[x(t)]\n'
    f'Cor = envelope (plasma: escuro→brilhante = fraco→forte)\n'
    f'Projeção inferior = sinal 2D.  P/S/T = pontilhado vertical.',
    color=COR_TXT, fontsize=10, y=1.01
)
plt.tight_layout()
plt.savefig('/content/helice_3d.png', dpi=160,
            bbox_inches='tight', facecolor='#080810')
plt.close()
display(Image('/content/helice_3d.png'))
print("  → helice_3d.png")

print(f"\n{'='*60}")
print(f"  3 figuras geradas — observação sincera")
print(f"  · lupa_2d.png      — zoom ±{LUPA_MS}ms, sem ajuste")
print(f"  · retrato_fase.png — Re×Im, forma real do campo")
print(f"  · helice_3d.png    — hélice 3D, cor=envelope")
print(f"{'='*60}")
