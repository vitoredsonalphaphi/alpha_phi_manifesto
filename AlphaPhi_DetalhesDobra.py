"""
AlphaPhi_DetalhesDobra.py
Detalhamento dos 3 Pontos de Dobra — campo eco-ressonante 880Hz · α*=1/3

Aprofundamento da observação iniciada no gráfico verde:
  P ≈ 4.1s — emergência inicial
  S ≈ 5.5s — campo se encorpando
  T ≈ 7.1s — campo estabilizado (ápice da sensação ergonómica)

Para cada ponto: zoom do sinal, envelope instantâneo, coerência φ-bandas,
e espectro de potência — comparando como o campo evolui de P → S → T.

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hilbert, butter, filtfilt
from scipy.io import wavfile
from IPython.display import Image, Audio, display

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
print("  AlphaPhi · Detalhamento · Três Pontos de Dobra")
print("=" * 60)

# ── funções eco originais ─────────────────────────────────────
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
    return [(max(0, int(f_lo/(FS/n))),
             min(int(f_hi/(FS/n))+1, n//2+1),
             f_lo, f_hi)
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
    cas_f = None
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        cas_f = cas
    return beta, cas_f

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

def coerencia_phi(seg, bins_phi):
    """Coerência espectral por bandas φ para um segmento."""
    F = np.fft.rfft(seg)
    cohs, freqs_c = [], []
    for (b_lo, b_hi, f_lo, f_hi) in bins_phi:
        Fb = F[b_lo:b_hi]
        if len(Fb) == 0:
            continue
        an = np.clip(np.abs(Fb)/(np.abs(Fb).sum()+1e-8), 1e-10, 1.0)
        coh = float(1.0 - (-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        cohs.append(coh)
        freqs_c.append(np.sqrt(f_lo * f_hi))   # centro geométrico da banda
    return np.array(cohs), np.array(freqs_c)

# ── gerar sinal ───────────────────────────────────────────────
print("\n  Gerando sinal original…")
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_sig)))
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_sig + BETA_FM*np.sin(2*np.pi*F_M*t_sig)))
x_mix = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)

beta_f, cas = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal = concatenar(cas, fade=int(0.15*FS))
dur   = len(sinal) / FS
t_sig_full = np.arange(len(sinal)) / FS
print(f"  β_max={beta_f.max():.4f}  φ³={PHI**3:.4f}  duração={dur:.2f}s")

# ── 3 pontos de dobra (posições observadas) ───────────────────
DOBRAS = {
    'P': {'t': 4.10, 'cor': '#00FF88', 'label': 'P — onset (~4.1s)',
          'desc': 'emergência inicial'},
    'S': {'t': 5.50, 'cor': '#FFB800', 'label': 'S — inflexão (~5.5s)',
          'desc': 'campo encorpando'},
    'T': {'t': 7.10, 'cor': '#FF4466', 'label': 'T — ápice (~7.1s)',
          'desc': 'campo estabilizado'},
}
NOMES  = list(DOBRAS.keys())
JANELA = 0.55   # ±0.275s ao redor de cada ponto

# ── extrair janelas ───────────────────────────────────────────
janelas = {}
for nome, d in DOBRAS.items():
    tc = d['t']
    i0 = max(0, int((tc - JANELA/2) * FS))
    i1 = min(len(sinal), int((tc + JANELA/2) * FS))
    seg      = sinal[i0:i1]
    t_local  = np.arange(len(seg)) / FS + tc - JANELA/2
    env      = lowpass(np.abs(hilbert(seg)), 200.0)
    freqs_fft = np.fft.rfftfreq(len(seg), 1/FS)
    mag_fft  = np.abs(np.fft.rfft(seg))
    cohs, fc = coerencia_phi(seg, bandas_para_bins(BANDAS, len(seg)))
    janelas[nome] = dict(seg=seg, t=t_local, env=env,
                         freqs=freqs_fft, mag=mag_fft,
                         cohs=cohs, fc=fc, tc=tc)
    coh_med = float(np.mean(cohs)) if len(cohs) else 0.0
    print(f"  {nome} ({tc:.2f}s): coerência média φ-bandas = {coh_med:.4f}")

# ── visualização principal ────────────────────────────────────
print("\n  Gerando figura de detalhamento…")
COR_BG = '#0D0D1A'; COR_TXT = '#CCCCDD'; COR_GRD = '#22223A'

fig = plt.figure(figsize=(16, 14), facecolor='#080810')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)

for col, nome in enumerate(NOMES):
    d  = DOBRAS[nome]
    jn = janelas[nome]
    cor = d['cor']

    # ── linha 0: zoom do sinal (o "gráfico verde" local) ─────
    ax0 = fig.add_subplot(gs[0, col])
    ax0.set_facecolor(COR_BG)
    ax0.plot(jn['t'], jn['seg'], color=cor, lw=0.35, alpha=0.9)
    ax0.fill_between(jn['t'],  jn['env'], -jn['env'],
                     color=cor, alpha=0.12)
    ax0.plot(jn['t'],  jn['env'], color=cor, lw=1.2, alpha=0.7)
    ax0.plot(jn['t'], -jn['env'], color=cor, lw=1.2, alpha=0.7)
    ax0.axvline(d['t'], color='white', lw=1.0, ls='--', alpha=0.6)
    ax0.set_title(f"{d['label']}\n{d['desc']}", color=COR_TXT, fontsize=8.5)
    ax0.set_xlabel('t (s)', color=COR_TXT, fontsize=7)
    ax0.set_ylabel('Amp', color=COR_TXT, fontsize=7)
    ax0.tick_params(colors=COR_TXT, labelsize=7)
    for sp in ax0.spines.values(): sp.set_color(COR_GRD)
    ax0.set_xlim(jn['t'][0], jn['t'][-1])

    # ── linha 1: envelope instantâneo — o "campo" ────────────
    ax1 = fig.add_subplot(gs[1, col])
    ax1.set_facecolor(COR_BG)
    ax1.fill_between(jn['t'], 0, jn['env'], color=cor, alpha=0.35)
    ax1.plot(jn['t'], jn['env'], color=cor, lw=1.5)
    # variância local do envelope (textura do campo)
    w_env = int(0.02 * FS)   # janela 20ms
    var_env = np.array([jn['env'][i:i+w_env].var()
                        for i in range(len(jn['env'])-w_env)])
    t_var   = jn['t'][:len(var_env)]
    ax1_b   = ax1.twinx()
    ax1_b.plot(t_var, var_env, color='white', lw=0.7, alpha=0.4, ls=':')
    ax1_b.tick_params(colors=COR_GRD, labelsize=6)
    ax1_b.set_ylabel('var(env)', color=COR_GRD, fontsize=6)
    ax1.axvline(d['t'], color='white', lw=1.0, ls='--', alpha=0.5)
    ax1.set_title('Envelope instantâneo  (variância = pontilhado)',
                  color=COR_TXT, fontsize=8)
    ax1.set_xlabel('t (s)', color=COR_TXT, fontsize=7)
    ax1.set_ylabel('|hilbert|', color=COR_TXT, fontsize=7)
    ax1.tick_params(colors=COR_TXT, labelsize=7)
    for sp in ax1.spines.values(): sp.set_color(COR_GRD)
    ax1.set_xlim(jn['t'][0], jn['t'][-1])

    # ── linha 2: coerência φ-bandas ───────────────────────────
    ax2 = fig.add_subplot(gs[2, col])
    ax2.set_facecolor(COR_BG)
    if len(jn['cohs']) > 0:
        # mostrar apenas bandas audíveis (20Hz–8kHz)
        mask_f = jn['fc'] <= 8000
        fc_show  = jn['fc'][mask_f]
        coh_show = jn['cohs'][mask_f]
        ax2.bar(range(len(coh_show)), coh_show, color=cor, alpha=0.75, width=0.8)
        # linha PHI³ como referência
        ax2.axhline(1.0, color='white', lw=0.6, ls=':', alpha=0.3)
        ax2.axhline(np.mean(coh_show), color='white', lw=1.0, ls='--', alpha=0.6,
                    label=f'média={np.mean(coh_show):.3f}')
        ax2.set_xticks([])
        ax2.legend(fontsize=7, facecolor='#111', labelcolor='white')
    ax2.set_title('Coerência por bandas φ  (20Hz→8kHz)',
                  color=COR_TXT, fontsize=8)
    ax2.set_ylabel('Coerência', color=COR_TXT, fontsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(colors=COR_TXT, labelsize=7)
    for sp in ax2.spines.values(): sp.set_color(COR_GRD)

    # ── linha 3: espectro de potência ─────────────────────────
    ax3 = fig.add_subplot(gs[3, col])
    ax3.set_facecolor(COR_BG)
    # mostrar até 4kHz para detalhe
    mask_sp = jn['freqs'] <= 4000
    ax3.semilogy(jn['freqs'][mask_sp], jn['mag'][mask_sp]+1e-10,
                 color=cor, lw=0.8, alpha=0.9)
    # marcar harmônicas de 880Hz
    for h in range(1, 6):
        fh = F_BEEP * h
        if fh <= 4000:
            ax3.axvline(fh, color='white', lw=0.5, ls=':', alpha=0.35)
    # marcar F_ORG e harmônicas
    for h in range(1, 10):
        fh = F_ORG * h
        if fh <= 4000:
            ax3.axvline(fh, color='cyan', lw=0.4, ls=':', alpha=0.2)
    ax3.set_title('Espectro (branco=880Hz harm. · ciano=220Hz harm.)',
                  color=COR_TXT, fontsize=8)
    ax3.set_xlabel('Hz', color=COR_TXT, fontsize=7)
    ax3.set_ylabel('Mag (log)', color=COR_TXT, fontsize=7)
    ax3.tick_params(colors=COR_TXT, labelsize=7)
    for sp in ax3.spines.values(): sp.set_color(COR_GRD)
    ax3.set_xlim(0, 4000)

fig.suptitle(
    f'AlphaPhi · Detalhes dos 3 Pontos de Dobra\n'
    f'Beep {F_BEEP:.0f}Hz · α*=1/3 · sinal {dur:.2f}s\n'
    f'Linha 1: zoom sinal  |  Linha 2: envelope + variância  |  '
    f'Linha 3: coerência φ  |  Linha 4: espectro',
    color=COR_TXT, fontsize=10, y=1.01
)
plt.savefig('/content/detalhes_tres_dobras.png', dpi=150,
            bbox_inches='tight', facecolor='#080810')
plt.close()
print("  → detalhes_tres_dobras.png")
display(Image('/content/detalhes_tres_dobras.png'))

# ── figura complementar: visão global + 3 janelas destacadas ─
print("\n  Gerando figura contextual (gráfico verde + janelas)…")
fig2, axes2 = plt.subplots(2, 1, figsize=(16, 7))
fig2.patch.set_facecolor('#080810')
for ax in axes2:
    ax.set_facecolor(COR_BG)
    ax.tick_params(colors=COR_TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(COR_GRD)

# painel superior: sinal completo (o "gráfico verde" original)
axes2[0].plot(t_sig_full, sinal, color='#00FF88', lw=0.35, alpha=0.88)
for nome, d in DOBRAS.items():
    jw = (d['t'] - JANELA/2, d['t'] + JANELA/2)
    axes2[0].axvspan(jw[0], jw[1], color=d['cor'], alpha=0.10)
    axes2[0].axvline(d['t'], color=d['cor'], lw=1.5, ls='--', alpha=0.85,
                     label=f"{nome}={d['t']:.1f}s")
axes2[0].set_title('Sinal completo — 3 janelas de análise destacadas',
                   color=COR_TXT, fontsize=10)
axes2[0].legend(fontsize=9, facecolor='#111', labelcolor=COR_TXT, loc='upper left')
axes2[0].set_ylabel('Amp', color=COR_TXT, fontsize=8)
axes2[0].set_xlim(0, dur)

# painel inferior: coerência média comparativa
nomes_ord = list(DOBRAS.keys())
cohs_med  = [float(np.mean(janelas[n]['cohs'])) if len(janelas[n]['cohs'])>0 else 0
             for n in nomes_ord]
cors_bar  = [DOBRAS[n]['cor'] for n in nomes_ord]
labels_bar = [f"{n}\n{DOBRAS[n]['t']:.1f}s" for n in nomes_ord]
bars = axes2[1].bar(nomes_ord, cohs_med, color=cors_bar, alpha=0.8, width=0.5)
for bar, val in zip(bars, cohs_med):
    axes2[1].text(bar.get_x() + bar.get_width()/2, val + 0.005,
                  f'{val:.4f}', ha='center', va='bottom',
                  color=COR_TXT, fontsize=10, fontweight='bold')
axes2[1].set_xticklabels(labels_bar, color=COR_TXT, fontsize=10)
axes2[1].set_title('Coerência φ-bandas média nos 3 pontos de dobra  (P→S→T)',
                   color=COR_TXT, fontsize=10)
axes2[1].set_ylabel('Coerência média', color=COR_TXT, fontsize=9)
axes2[1].set_ylim(0, 1.0)
axes2[1].tick_params(colors=COR_TXT)
axes2[1].set_facecolor(COR_BG)
for sp in axes2[1].spines.values(): sp.set_color(COR_GRD)

fig2.suptitle(
    f'AlphaPhi · Emergência do Campo — P→S→T · Beep {F_BEEP:.0f}Hz α*=1/3',
    color=COR_TXT, fontsize=11, y=1.01
)
plt.tight_layout()
plt.savefig('/content/emergencia_campo_pst.png', dpi=150,
            bbox_inches='tight', facecolor='#080810')
plt.close()
print("  → emergencia_campo_pst.png")
display(Image('/content/emergencia_campo_pst.png'))

print(f"\n{'='*60}")
print(f"  RESULTADO")
print(f"{'='*60}")
for nome in NOMES:
    d  = DOBRAS[nome]
    jn = janelas[nome]
    cm = float(np.mean(jn['cohs'])) if len(jn['cohs']) > 0 else 0
    print(f"  {nome} ({d['t']:.1f}s) — {d['desc']}")
    print(f"    coerência φ média : {cm:.4f}")
    print(f"    energia (RMS)     : {float(np.sqrt(np.mean(jn['seg']**2))):.4f}")
print(f"\n  Observar:")
print(f"  · detalhes_tres_dobras.png — 4 análises × 3 pontos")
print(f"  · emergencia_campo_pst.png — sinal completo + coerência comparativa")
print(f"{'='*60}")
