"""
AlphaPhi_EscalaObservacao.py
Escala Graduada de Observação — Pontos de Dobra P / S / T

Princípio: o mesmo sinal, o mesmo espaço de representação (tempo × amplitude),
seis níveis progressivos de zoom centrados em cada ponto de dobra.
Nenhuma transformação de espaço. Nenhuma interpretação adicionada.
O que emergir em cada escala é o que está lá.

Escalas (centradas em t_dobra):
  1 — ± 4.0 s   (contexto completo, o gráfico verde)
  2 — ± 1.0 s   (região do ponto de dobra)
  3 — ± 250 ms  (estrutura de amplitude)
  4 — ±  60 ms  (arcos do envelope visíveis)
  5 — ±  15 ms  (ciclos individuais de 880Hz emergindo)
  6 — ±   4 ms  (~3 ciclos de 880Hz — estrutura de ciclo)

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
FADE       = int(0.15 * FS)

print("=" * 60)
print("  AlphaPhi · Escala Graduada de Observação")
print("  6 níveis de zoom · P / S / T")
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
print("\n  Gerando sinal original…")
t_seg = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_seg)))
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_seg
                          + BETA_FM*np.sin(2*np.pi*F_M*t_seg)))
x_mix = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)
beta_f, cas = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal  = concatenar(cas)
dur    = len(sinal) / FS
t_full = np.arange(len(sinal)) / FS
env_full = lowpass(np.abs(hilbert(sinal)), 200.0)
print(f"  {dur:.2f}s  β_max={beta_f.max():.4f}  φ³={PHI**3:.4f}")

# ── definições ────────────────────────────────────────────────
DOBRAS = [
    {'nome': 'P', 't': 4.10, 'cor': '#00FF88'},
    {'nome': 'S', 't': 5.50, 'cor': '#FFB800'},
    {'nome': 'T', 't': 7.10, 'cor': '#FF4466'},
]

# 6 meias-janelas em segundos
ESCALAS_S = [4.0, 1.0, 0.250, 0.060, 0.015, 0.004]
ESCALAS_LABEL = [
    '± 4.0 s  — contexto completo',
    '± 1.0 s  — região do ponto',
    '± 250 ms — estrutura de amplitude',
    '±  60 ms — arcos do envelope',
    '±  15 ms — ciclos emergindo',
    '±   4 ms — estrutura de ciclo',
]

COR_BG  = '#0D0D1A'
COR_TXT = '#CCCCDD'
COR_GRD = '#22223A'

# ── figura por ponto de dobra ─────────────────────────────────
for d in DOBRAS:
    tc  = d['t']
    cor = d['cor']
    nome = d['nome']
    print(f"\n  Gerando figura — ponto {nome} ({tc:.2f}s)…")

    fig, axes = plt.subplots(6, 1, figsize=(14, 18))
    fig.patch.set_facecolor('#080810')

    for row, (meia_jan, lbl) in enumerate(zip(ESCALAS_S, ESCALAS_LABEL)):
        ax  = axes[row]
        t0  = max(0.0,  tc - meia_jan)
        t1  = min(dur,  tc + meia_jan)
        i0  = int(t0 * FS)
        i1  = int(t1 * FS)

        seg   = sinal[i0:i1]
        t_w   = t_full[i0:i1]
        env_w = env_full[i0:i1]

        ax.set_facecolor(COR_BG)

        # envelope preenchido — aparece em todas as escalas
        ax.fill_between(t_w, -env_w, env_w, color=cor, alpha=0.12)
        ax.plot(t_w,  env_w, color=cor, lw=1.0, alpha=0.60)
        ax.plot(t_w, -env_w, color=cor, lw=1.0, alpha=0.60)

        # sinal bruto — lw adaptado à escala: fino no zoom amplo, espesso no zoom fino
        lw_tab = [0.4, 0.35, 0.45, 0.60, 1.0, 1.4]
        lw_sig = lw_tab[row]
        ax.plot(t_w, seg, color=cor, lw=lw_sig, alpha=0.92)

        # marcador do ponto de dobra
        ax.axvline(tc, color='white', lw=0.7, ls=':', alpha=0.35)

        ax.set_xlim(t0, t1)
        ax.set_ylim(-1.08, 1.08)
        ax.set_ylabel('Amp', color=COR_TXT, fontsize=7)
        ax.tick_params(colors=COR_TXT, labelsize=7)
        for sp in ax.spines.values(): sp.set_color(COR_GRD)

        # título apenas na primeira linha
        if row == 0:
            ax.set_title(
                f'Ponto {nome}  t={tc:.2f}s  —  escala 1: {lbl}',
                color=COR_TXT, fontsize=9
            )
        else:
            ax.set_title(f'Escala {row+1}: {lbl}', color=COR_TXT, fontsize=8)

        if row == 5:
            ax.set_xlabel('Tempo (s)', color=COR_TXT, fontsize=8)

        # ciclo de referência 880Hz na escala mais fina
        if row >= 4:
            ciclo = 1.0 / F_BEEP
            n_ciclos_vis = int((t1 - t0) / ciclo)
            ax.set_title(
                f'Escala {row+1}: {lbl}  '
                f'(~{n_ciclos_vis} ciclos de {F_BEEP:.0f}Hz visíveis)',
                color=COR_TXT, fontsize=8
            )

    fig.suptitle(
        f'AlphaPhi · Escala Graduada — Ponto {nome}  t={tc:.2f}s\n'
        f'Beep {F_BEEP:.0f}Hz · α*=1/3 · sinal {dur:.2f}s\n'
        f'Mesmo sinal, mesmo espaço — 6 níveis de zoom',
        color=COR_TXT, fontsize=10, y=1.01
    )
    plt.tight_layout()
    fname = f'/content/escala_{nome}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='#080810')
    plt.close()
    print(f"  → escala_{nome}.png")
    display(Image(fname))

print(f"\n{'='*60}")
print(f"  3 figuras geradas:")
print(f"  · escala_P.png  — ponto P (4.10s)")
print(f"  · escala_S.png  — ponto S (5.50s)")
print(f"  · escala_T.png  — ponto T (7.10s)")
print(f"\n  Cada figura: 6 escalas progressivas")
print(f"  Da visão completa ao ciclo individual de {F_BEEP:.0f}Hz")
print(f"{'='*60}")
