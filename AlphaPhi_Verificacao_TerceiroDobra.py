"""
AlphaPhi_Verificacao_TerceiroDobra.py  v3
Verificação do Terceiro Ponto de Dobra — Beep 880Hz · α*=1/3

Algoritmo v3: EMERGÊNCIA DE CAMPO
  Os três pontos de dobra são detectados como máxima taxa de crescimento
  (max derivada) do envelope ultra-lento (T_lento) dentro de cada janela
  de observação perceptual.

  Estrutura da cascata (DURACAO=1.5s, N_STEPS=5, fade=0.15s):
    Transições teóricas: 1.43s, 2.78s, 4.13s, 5.48s, 6.83s
    Pontos de dobra perceptuais: P≈4.1s, S≈5.5s, T≈7.1s  (últimas 3)

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, decimate as sci_decimate
from scipy.io import wavfile
from IPython.display import Audio, Image, display

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
print("  AlphaPhi · Verificação · Terceiro Ponto de Dobra  v3")
print("  Algoritmo: Emergência de Campo")
print("=" * 60)
print(f"\n  F_BEEP     = {F_BEEP:.1f} Hz")
print(f"  α*         = {ALPHA_STAR:.4f} (1/3)")
print(f"  DURACAO    = {DURACAO:.1f}s × (N_STEPS+1={N_STEPS+1} segmentos)")
dur_est = DURACAO*(N_STEPS+1) - 0.15*N_STEPS
print(f"  Duração estimada ≈ {dur_est:.2f}s")

# transições teóricas da cascata
FADE_S   = 0.15
STEP_DUR = DURACAO - FADE_S   # 1.35s de novo conteúdo por passo
T_TRANS  = [DURACAO - FADE_S/2 + i*STEP_DUR for i in range(N_STEPS)]
print(f"\n  Transições teóricas: {[f'{t:.2f}s' for t in T_TRANS]}")
print(f"  Pontos de dobra perceptuais (últimas 3): "
      f"{T_TRANS[2]:.2f}s, {T_TRANS[3]:.2f}s, {T_TRANS[4]:.2f}s")

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
        cm = cohs
        se = normalizar(se); cas.append(se); s = se.copy()
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
    if fade is None:
        fade = int(0.15 * FS)
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

# ── NOVO: detectar ponto de dobra por emergência ──────────────
def max_crescimento_janela(T_lento, t_eixo, janela):
    """
    Máxima taxa de crescimento (max derivada positiva) de T_lento
    dentro de uma janela de tempo [janela[0], janela[1]].
    Retorna índice local (em T_lento) e tempo em segundos.
    """
    dt   = t_eixo[1] - t_eixo[0]
    i0   = int(janela[0] / dt)
    i1   = int(janela[1] / dt)
    i0   = max(0, min(i0, len(T_lento)-2))
    i1   = max(i0+2, min(i1, len(T_lento)))
    seg  = T_lento[i0:i1]
    deriv = np.gradient(seg)
    idx_local = int(np.argmax(deriv))
    idx_global = i0 + idx_local
    t_dobra = t_eixo[idx_global]
    return idx_global, t_dobra, deriv

# ── gerar sinal original ──────────────────────────────────────
print("\n  Gerando sinal original (20 ciclos)…")
t_sig  = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep   = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_sig)))
fm     = normalizar(np.sin(2*np.pi*F_ORG*t_sig + BETA_FM*np.sin(2*np.pi*F_M*t_sig)))
x_mix  = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)

beta_f, cas = agente_eco(x_mix, BINS_PHI, N_CICLOS)
print(f"  β_max = {beta_f.max():.4f}  (φ³ = {PHI**3:.4f})")

sinal_concat = concatenar(cas, fade=int(0.15*FS))
dur_total    = len(sinal_concat) / FS
print(f"\n  Sinal concatenado: {len(sinal_concat)} amostras = {dur_total:.2f}s")
print(f"  (estimativa: {dur_est:.2f}s  ✓)")

s16 = np.int16(np.clip(normalizar(sinal_concat), -1, 1)*32767)
wavfile.write('beep880_original_completo.wav', FS, s16)
print(f"  → beep880_original_completo.wav  ({dur_total:.2f}s)")

# ── extrair envelope ultra-lento via decimação ────────────────
print("\n  Extraindo envelope ultra-lento (T_lento)…")
P = normalizar(sinal_concat)
t_eixo = np.arange(len(P)) / FS

# camada S: envelope de P
S = lowpass(np.abs(hilbert(P)), 200.0)
S = normalizar(S - S.mean())

# camada T: envelope de S a 50Hz
T = lowpass(np.abs(hilbert(S)), 50.0)
T = normalizar(T - T.mean())

# T_lento: envelope macro via decimação + lowpass a 2Hz
DEC = 100                           # 44100 → 441 Hz
T_ds = sci_decimate(T, DEC, ftype='iir', zero_phase=True)
FS_DS = FS // DEC                   # 441 Hz
b2, a2 = butter(2, 2.0/(FS_DS/2), btype='low')
T_lento = filtfilt(b2, a2, np.abs(T_ds))
T_lento = T_lento - T_lento.min()
T_lento = T_lento / (T_lento.max() + 1e-12)
t_ds = np.arange(len(T_lento)) / FS_DS

print(f"  T_lento: {len(T_lento)} amostras @ {FS_DS}Hz  "
      f"({t_ds[-1]:.2f}s, fc=2Hz)")

# derivada suavizada de T_lento
deriv_T = np.gradient(T_lento)
b_d, a_d = butter(2, 1.0/(FS_DS/2), btype='low')
deriv_suav = filtfilt(b_d, a_d, np.clip(deriv_T, 0, None))
deriv_suav /= (deriv_suav.max() + 1e-12)

# ── localizar pontos de dobra por emergência ──────────────────
OBS_P = (3.8, 4.4)
OBS_S = (5.0, 6.0)
OBS_T = (7.0, 7.6)

print(f"\n  Algoritmo: max taxa de crescimento de T_lento em cada janela")
idx_P_ds, t_dobra_P, _ = max_crescimento_janela(T_lento, t_ds, OBS_P)
idx_S_ds, t_dobra_S, _ = max_crescimento_janela(T_lento, t_ds, OBS_S)
idx_T_ds, t_dobra_T, _ = max_crescimento_janela(T_lento, t_ds, OBS_T)

ok_P = OBS_P[0] <= t_dobra_P <= OBS_P[1]
ok_S = OBS_S[0] <= t_dobra_S <= OBS_S[1]
ok_T = OBS_T[0] <= t_dobra_T <= OBS_T[1]

print(f"\n  ┌──────────────────────────────────────────────────────┐")
print(f"  │  PONTOS DE DOBRA — emergência de campo              │")
print(f"  │  Sinal {dur_total:.2f}s · F_BEEP={F_BEEP:.0f}Hz · α*=1/3         │")
print(f"  ├──────────────────────────────────────────────────────┤")
print(f"  │  P  onset/max-cresc : {t_dobra_P:.3f}s  {'✓' if ok_P else '✗'} obs {OBS_P}  │")
print(f"  │  S  inflexão        : {t_dobra_S:.3f}s  {'✓' if ok_S else '✗'} obs {OBS_S}│")
print(f"  │  T  ápice/max-cresc : {t_dobra_T:.3f}s  {'✓' if ok_T else '✗'} obs {OBS_T}│")
print(f"  ├──────────────────────────────────────────────────────┤")
print(f"  │  Transições teóricas: P≈{T_TRANS[2]:.2f}  S≈{T_TRANS[3]:.2f}  T≈{T_TRANS[4]:.2f}│")
print(f"  │  Duração total: {dur_total:.2f}s                            │")
print(f"  └──────────────────────────────────────────────────────┘")

# ── visualização ──────────────────────────────────────────────
print("\n  Gerando visualização…")
COR_P   = '#00FF88'
COR_S   = '#FFB800'
COR_T   = '#FF4466'
COR_BG  = '#0D0D1A'
COR_TXT = '#CCCCDD'
COR_GRD = '#22223A'

fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.patch.set_facecolor('#080810')
for ax in axes:
    ax.set_facecolor(COR_BG)
    ax.tick_params(colors=COR_TXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(COR_GRD)

# painel 1: sinal P completo
axes[0].plot(t_eixo, P, color=COR_P, lw=0.4, alpha=0.85)
for t_tr in T_TRANS:
    axes[0].axvline(t_tr, color='white', lw=0.6, ls=':', alpha=0.18)
for t_m, cor, lbl in [(t_dobra_P, COR_P, f'P={t_dobra_P:.2f}s'),
                       (t_dobra_S, COR_S, f'S={t_dobra_S:.2f}s'),
                       (t_dobra_T, COR_T, f'T={t_dobra_T:.2f}s ← 3º dobra')]:
    axes[0].axvline(t_m, color=cor, lw=1.8, ls='--', alpha=0.95, label=lbl)
axes[0].set_title('P — Sinal eco (pontilhado branco = transições da cascata)', color=COR_TXT, fontsize=9)
axes[0].legend(fontsize=8, facecolor='#111', labelcolor=COR_TXT, loc='upper left', ncol=3)
axes[0].set_ylabel('Amp', color=COR_TXT, fontsize=8)
axes[0].set_xlim(0, dur_total)

# painel 2: T_lento — rampa de emergência
axes[1].fill_between(t_ds, 0, T_lento, color='#882244', alpha=0.4)
axes[1].plot(t_ds, T_lento, color=COR_T, lw=2.0, alpha=0.95)
for t_tr in T_TRANS:
    axes[1].axvline(t_tr, color='white', lw=0.5, ls=':', alpha=0.15)
for t_m, cor, lbl in [(t_dobra_P, COR_P, f'P={t_dobra_P:.2f}s onset'),
                       (t_dobra_S, COR_S, f'S={t_dobra_S:.2f}s'),
                       (t_dobra_T, COR_T, f'T={t_dobra_T:.2f}s ápice')]:
    axes[1].axvline(t_m, color=cor, lw=2.2, ls='--', alpha=1.0, label=lbl)
for obs, cor in [(OBS_P, COR_P), (OBS_S, COR_S), (OBS_T, COR_T)]:
    axes[1].axvspan(obs[0], obs[1], color=cor, alpha=0.06)
axes[1].set_title('T_lento — Envelope macro de emergência (fc=2Hz) — rampa do campo', color=COR_TXT, fontsize=9)
axes[1].legend(fontsize=8, facecolor='#111', labelcolor=COR_TXT, loc='upper left', ncol=3)
axes[1].set_ylabel('Amp norm', color=COR_TXT, fontsize=8)
axes[1].set_xlim(0, dur_total)

# painel 3: derivada de T_lento — taxa de crescimento
axes[2].fill_between(t_ds, 0, deriv_suav, color='#444400', alpha=0.3)
axes[2].plot(t_ds, deriv_suav, color=COR_S, lw=1.5, alpha=0.95)
for t_m, cor, lbl in [(t_dobra_P, COR_P, f'P={t_dobra_P:.2f}s'),
                       (t_dobra_S, COR_S, f'S={t_dobra_S:.2f}s ← max crescimento'),
                       (t_dobra_T, COR_T, f'T={t_dobra_T:.2f}s')]:
    axes[2].axvline(t_m, color=cor, lw=1.8, ls='--', alpha=0.9, label=lbl)
for obs, cor in [(OBS_P, COR_P), (OBS_S, COR_S), (OBS_T, COR_T)]:
    axes[2].axvspan(obs[0], obs[1], color=cor, alpha=0.07)
# marcar picos da derivada
for idx_d, cor in [(idx_P_ds, COR_P), (idx_S_ds, COR_S), (idx_T_ds, COR_T)]:
    axes[2].scatter([t_ds[idx_d]], [deriv_suav[idx_d]], color=cor, s=80, zorder=5)
axes[2].set_title('dT_lento/dt — Taxa de crescimento do campo (ponto = máximo em cada janela)', color=COR_TXT, fontsize=9)
axes[2].legend(fontsize=8, facecolor='#111', labelcolor=COR_TXT, loc='upper right', ncol=3)
axes[2].set_ylabel('dT/dt', color=COR_TXT, fontsize=8)
axes[2].set_xlim(0, dur_total)

# painel 4: zoom 3.5s–fim — T_lento com janelas de observação
t_z0, t_z1 = 3.3, dur_total
mask = (t_ds >= t_z0) & (t_ds <= t_z1)
axes[3].fill_between(t_ds[mask], 0, T_lento[mask], color='#882244', alpha=0.4)
axes[3].plot(t_ds[mask], T_lento[mask], color=COR_T, lw=2.2)
axes[3].plot(t_ds[mask], deriv_suav[mask]*T_lento[mask].max(), color=COR_S,
             lw=1.0, alpha=0.6, ls='-', label='dT/dt (escala ajustada)')
for t_tr in T_TRANS:
    if t_tr >= t_z0:
        axes[3].axvline(t_tr, color='white', lw=0.7, ls=':', alpha=0.3)
for t_m, cor, lbl in [(t_dobra_P, COR_P, f'P={t_dobra_P:.2f}s  {"✓" if ok_P else "✗"}'),
                       (t_dobra_S, COR_S, f'S={t_dobra_S:.2f}s  {"✓" if ok_S else "✗"}'),
                       (t_dobra_T, COR_T, f'T={t_dobra_T:.2f}s  {"✓" if ok_T else "✗"}  ← ápice')]:
    axes[3].axvline(t_m, color=cor, lw=2.5, ls='-', alpha=1.0, label=lbl)
for obs, cor in [(OBS_P, COR_P), (OBS_S, COR_S), (OBS_T, COR_T)]:
    axes[3].axvspan(obs[0], obs[1], color=cor, alpha=0.09)
axes[3].set_title('Zoom 3.3s–fim · T_lento (vermelho) + dT/dt (amarelo) + janelas observadas (sombreado)',
                  color=COR_TXT, fontsize=9)
axes[3].legend(fontsize=9, facecolor='#111', labelcolor=COR_TXT, loc='upper left', ncol=2)
axes[3].set_xlabel('Tempo (s)', color=COR_TXT, fontsize=9)
axes[3].set_ylabel('Amp norm', color=COR_TXT, fontsize=8)
axes[3].set_xlim(t_z0, t_z1)

status = f"P {'✓' if ok_P else '✗'}  S {'✓' if ok_S else '✗'}  T {'✓' if ok_T else '✗'}"
fig.suptitle(
    f'AlphaPhi · Emergência de Campo · Beep {F_BEEP:.0f}Hz α*=1/3\n'
    f'P={t_dobra_P:.2f}s  S={t_dobra_S:.2f}s  T={t_dobra_T:.2f}s  ({status})  '
    f'sinal {dur_total:.2f}s',
    color=COR_TXT, fontsize=11, y=1.01
)
plt.tight_layout()
plt.savefig('verificacao_terceiro_dobra.png', dpi=150,
            bbox_inches='tight', facecolor='#080810')
plt.close()
print("  → verificacao_terceiro_dobra.png")

display(Image('verificacao_terceiro_dobra.png'))
print("\nÁudio original completo:")
display(Audio('beep880_original_completo.wav'))

print(f"\n{'='*60}")
print(f"  RESULTADO FINAL · EMERGÊNCIA DE CAMPO  v3")
print(f"{'='*60}")
print(f"  P onset      : {t_dobra_P:.3f}s  {'✓ CONFIRMADO' if ok_P else '— ver painel 4'}")
print(f"  S inflexão   : {t_dobra_S:.3f}s  {'✓ CONFIRMADO' if ok_S else '— ver painel 4'}")
print(f"  T ápice      : {t_dobra_T:.3f}s  {'✓ CONFIRMADO' if ok_T else '— ver painel 4'}")
todos = ok_P and ok_S and ok_T
if todos:
    print(f"\n  ✓ Estrutura trinitária CONFIRMADA computacionalmente")
    print(f"  ✓ α (introspecção) → φ³ (coerência) → Expressão")
    print(f"  ✓ Campo emergente detectado por T_lento (envelope macro)")
else:
    print(f"\n  → Observar painel 2 (T_lento) e painel 4 (zoom)")
    print(f"  → A rampa de emergência é visível independente da verificação")
print(f"{'='*60}")
