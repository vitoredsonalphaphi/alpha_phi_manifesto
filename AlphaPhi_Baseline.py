"""
AlphaPhi_Baseline.py
BASE DE REFERÊNCIA — Áudio + Gráfico Verde

Este arquivo preserva exatamente dois resultados fundamentais:

  1. ÁUDIO ORIGINAL
     beep880_baseline.wav — 880Hz, α*=1/3, eco φ completo
     O sinal que produziu a sensação ergonómica verificada.
     Nenhuma alteração, nenhuma simplificação, nenhum tile.

  2. GRÁFICO VERDE ORIGINAL
     grafico_verde_baseline.png — sinal completo 0–8.25s
     O gráfico que revelou os arcos esféricos nos pontos de dobra.
     Mesmos parâmetros de plotagem que produziram a observação.

PARÂMETROS IMUTÁVEIS:
  F_BEEP     = 880.0 Hz
  α*         = 1/3  (0.3333...)
  DURACAO    = 1.5 s
  N_STEPS    = 5
  N_CICLOS   = 20
  fade       = 0.15 s  (int(0.15 * 44100) amostras)
  FS         = 44100 Hz

Qualquer refinamento futuro usa este arquivo como parâmetro.
Não modificar.

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
from IPython.display import Audio, Image, display

# ══════════════════════════════════════════════════════════════
# PARÂMETROS — NÃO MODIFICAR
# ══════════════════════════════════════════════════════════════
PHI        = (1 + np.sqrt(5)) / 2   # 1.6180339...
FS         = 44100                   # Hz
F_BEEP     = 880.0                   # Hz
F_ORG      = 220.0                   # Hz
F_M        = F_ORG / PHI             # Hz
BETA_FM    = PHI                     # índice FM
ALPHA_STAR = 1.0 / 3.0              # α* = 1/3
DURACAO    = 1.5                     # s por segmento
N_STEPS    = 5                       # passos da cascata
N_CICLOS   = 20                      # ciclos do agente eco
FADE       = int(0.15 * FS)          # amostras de crossfade

print("=" * 60)
print("  AlphaPhi · BASELINE")
print("  Áudio Original + Gráfico Verde")
print("=" * 60)
print(f"\n  PHI        = {PHI:.10f}")
print(f"  F_BEEP     = {F_BEEP:.1f} Hz")
print(f"  α*         = {ALPHA_STAR:.10f}  (1/3)")
print(f"  DURACAO    = {DURACAO:.1f} s")
print(f"  N_STEPS    = {N_STEPS}")
print(f"  N_CICLOS   = {N_CICLOS}")
print(f"  FADE       = {FADE} amostras ({0.15*1000:.0f} ms)")
print(f"  FS         = {FS} Hz")

# ══════════════════════════════════════════════════════════════
# FUNÇÕES ECO — IDÊNTICAS AO ORIGINAL
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# GERAR SINAL
# ══════════════════════════════════════════════════════════════
print("\n  Gerando sinal…")
t_seg = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_seg)))
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_seg
                          + BETA_FM*np.sin(2*np.pi*F_M*t_seg)))
x_mix = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)

beta_f, cas = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal = concatenar(cas)
dur   = len(sinal) / FS
t_eixo = np.arange(len(sinal)) / FS

print(f"  β_max = {beta_f.max():.4f}  (φ³ = {PHI**3:.4f})")
print(f"  Duração = {dur:.4f} s  ({len(sinal)} amostras)")
print(f"  Estimativa teórica = "
      f"{DURACAO*(N_STEPS+1) - 0.15*N_STEPS:.4f} s")

# ══════════════════════════════════════════════════════════════
# 1. ÁUDIO ORIGINAL
# ══════════════════════════════════════════════════════════════
print("\n  Salvando áudio baseline…")
s16 = np.int16(np.clip(sinal, -1, 1) * 32767)
wavfile.write('/content/beep880_baseline.wav', FS, s16)
print(f"  → beep880_baseline.wav  ({dur:.4f}s, {FS}Hz, 16bit)")
print(f"  SHA: {hash(s16.tobytes()) & 0xFFFFFFFF:08X}  "
      f"(verificação de integridade)")

display(Audio('/content/beep880_baseline.wav'))

# ══════════════════════════════════════════════════════════════
# 2. GRÁFICO VERDE BASELINE
# Mesmos parâmetros de plotagem do gráfico original
# ══════════════════════════════════════════════════════════════
print("\n  Gerando gráfico verde baseline…")

fig, ax = plt.subplots(1, 1, figsize=(14, 4))
fig.patch.set_facecolor('#080810')
ax.set_facecolor('#0D0D1A')

# sinal completo — parâmetros idênticos ao original
ax.plot(t_eixo, sinal, color='#00FF88', lw=0.6, alpha=0.9)

# pontos de dobra observados — apenas linhas verticais, sem interpretação
for t_d, label, cor in [
    (4.10, 'P  4.1s', '#00FF88'),
    (5.50, 'S  5.5s', '#FFB800'),
    (7.10, 'T  7.1s', '#FF4466'),
]:
    ax.axvline(t_d, color=cor, lw=1.2, ls='--', alpha=0.70, label=label)

ax.set_xlim(0, dur)
ax.set_ylim(-1.05, 1.05)
ax.set_xlabel('Tempo (s)', color='#CCCCDD', fontsize=9)
ax.set_ylabel('Amplitude', color='#CCCCDD', fontsize=9)
ax.tick_params(colors='#CCCCDD', labelsize=8)
for sp in ax.spines.values(): sp.set_color('#22223A')
ax.legend(fontsize=8, facecolor='#111', labelcolor='#CCCCDD',
          loc='upper left', framealpha=0.7)

fig.suptitle(
    f'AlphaPhi · GRÁFICO VERDE BASELINE\n'
    f'Beep {F_BEEP:.0f}Hz · α*=1/3 · {dur:.2f}s · '
    f'lw=0.6 · cor=#00FF88 · α=0.9',
    color='#CCCCDD', fontsize=10, y=1.02
)
plt.tight_layout()
plt.savefig('/content/grafico_verde_baseline.png', dpi=150,
            bbox_inches='tight', facecolor='#080810')
plt.close()
print("  → grafico_verde_baseline.png")
display(Image('/content/grafico_verde_baseline.png'))

# ══════════════════════════════════════════════════════════════
# REGISTRO DOS PARÂMETROS DE PLOTAGEM
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  BASELINE REGISTRADO")
print(f"{'='*60}")
print(f"  ÁUDIO:")
print(f"    arquivo  : beep880_baseline.wav")
print(f"    duração  : {dur:.4f} s")
print(f"    amostras : {len(sinal)}")
print(f"    pico     : {np.max(np.abs(sinal)):.6f}")
print(f"    RMS      : {np.sqrt(np.mean(sinal**2)):.6f}")
print(f"\n  GRÁFICO:")
print(f"    arquivo  : grafico_verde_baseline.png")
print(f"    figsize  : (14, 4)")
print(f"    cor      : #00FF88")
print(f"    lw       : 0.6")
print(f"    alpha    : 0.9")
print(f"    xlim     : (0, {dur:.4f})")
print(f"    ylim     : (-1.05, 1.05)")
print(f"    dpi      : 150")
print(f"\n  PONTOS DE DOBRA (observados por escuta):")
print(f"    P : 4.10 s — onset")
print(f"    S : 5.50 s — campo encorpando")
print(f"    T : 7.10 s — campo firmado")
print(f"\n  Este arquivo é referência imutável.")
print(f"  Próximas verificações partem deste ponto.")
print(f"{'='*60}")
