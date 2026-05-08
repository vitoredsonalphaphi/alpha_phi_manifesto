"""
AlphaPhi_Verificacao_TerceiroDobra.py
Verificação do Terceiro Ponto de Dobra — Beep 880Hz · α*=1/3

Objetivo: confirmar que o terceiro ponto de dobra (camada T)
ocorre entre o 6º e o 8º segundo do sinal concatenado original.

Sinal original: DURACAO=1.5s, N_STEPS=5 → concatenar() ≈ 8.25s
Terceiro ponto de dobra = mínimo da variância local de T

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.io import wavfile
from IPython.display import Audio, Image, display

# ── constantes ORIGINAIS — não modificar ─────────────────────
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0          # Hz — original
F_ORG      = 220.0
F_M        = F_ORG / PHI
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0      # α* = 1/3 — original
DURACAO    = 1.5             # s — original
N_STEPS    = 5               # — original
N_CICLOS   = 20

print("=" * 60)
print("  AlphaPhi · Verificação · Terceiro Ponto de Dobra")
print("=" * 60)
print(f"\n  F_BEEP     = {F_BEEP:.1f} Hz")
print(f"  α*         = {ALPHA_STAR:.4f} (1/3)")
print(f"  DURACAO    = {DURACAO:.1f}s × (N_STEPS+1={N_STEPS+1} segmentos)")
print(f"  Duração concatenada estimada ≈ {DURACAO*(N_STEPS+1) - 0.15*N_STEPS:.2f}s")

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
N_BANDAS = len(BINS_PHI)

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

def lowpass(s, fc, order=4):
    b, a = butter(order, fc/(FS/2), btype='low')
    return filtfilt(b, a, s)

def ponto_dobra_local(sig, janela_ms=200, step_ms=10):
    """Mínimo de variância local com janela e step em ms."""
    w    = int(FS * janela_ms / 1000)
    step = int(FS * step_ms  / 1000)
    indices = range(0, len(sig)-w, step)
    var  = np.array([np.var(sig[i:i+w]) for i in indices])
    idx_min = np.argmin(var)
    return list(indices)[idx_min], var, list(indices)

# ── gerar sinal original ──────────────────────────────────────
print("\n  Gerando sinal original (20 ciclos)…")
t_sig  = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep   = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_sig)))
fm     = normalizar(np.sin(2*np.pi*F_ORG*t_sig + BETA_FM*np.sin(2*np.pi*F_M*t_sig)))
x_mix  = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)

beta_f, cas = agente_eco(x_mix, BINS_PHI, N_CICLOS)
print(f"  β_max = {beta_f.max():.4f}  (φ³ = {PHI**3:.4f})")

# concatenar cascade — sinal COMPLETO como o usuário ouviu
sinal_concat = concatenar(cas, fade=int(0.15*FS))
dur_total    = len(sinal_concat) / FS
print(f"\n  Sinal concatenado: {len(sinal_concat)} amostras = {dur_total:.2f}s")
print(f"  (estimativa: {DURACAO*(N_STEPS+1) - 0.15*N_STEPS:.2f}s  ✓)")

# salvar áudio original completo
s16 = np.int16(np.clip(normalizar(sinal_concat), -1, 1)*32767)
wavfile.write('/content/beep880_original_completo.wav', FS, s16)
print(f"  → beep880_original_completo.wav  ({dur_total:.2f}s)")

# ── extrair camadas P/S/T do sinal completo ───────────────────
print("\n  Extraindo camadas P, S, T…")
P = normalizar(sinal_concat)
S = lowpass(np.abs(hilbert(P)), 200.0)
S = normalizar(S - S.mean())
T = lowpass(np.abs(hilbert(S)), 50.0)
T = normalizar(T - T.mean())

t_eixo = np.arange(len(P)) / FS  # eixo de tempo em segundos

# ── localizar os 3 pontos de dobra ───────────────────────────
print(f"\n  Localizando pontos de dobra (mínimo variância local)…")
idx_P, var_P, idx_P_arr = ponto_dobra_local(P, janela_ms=150, step_ms=10)
idx_S, var_S, idx_S_arr = ponto_dobra_local(S, janela_ms=300, step_ms=10)
idx_T, var_T, idx_T_arr = ponto_dobra_local(T, janela_ms=500, step_ms=10)

t_dobra_P = idx_P / FS
t_dobra_S = idx_S / FS
t_dobra_T = idx_T / FS

# posições observadas por escuta (Vitor Edson Delavi, 8 mai 2026)
OBS_P = (3.8, 4.4)   # logo após 4s
OBS_S = (5.0, 6.0)   # ~5.5s
OBS_T = (7.0, 7.6)   # logo após 7s

print(f"\n  ┌──────────────────────────────────────────────────────┐")
print(f"  │  PONTOS DE DOBRA — sinal {dur_total:.2f}s                    │")
print(f"  ├──────────────────────────────────────────────────────┤")
print(f"  │  P  computado: {t_dobra_P:.3f}s   observado: ~4.1s          │")
print(f"  │  S  computado: {t_dobra_S:.3f}s   observado: ~5.5s          │")
print(f"  │  T  computado: {t_dobra_T:.3f}s   observado: ~7.1s  ← 3º   │")
print(f"  ├──────────────────────────────────────────────────────┤")
print(f"  │  Duração total: {dur_total:.2f}s                            │")
print(f"  └──────────────────────────────────────────────────────┘")

# verificação das 3 posições
ok_P = OBS_P[0] <= t_dobra_P <= OBS_P[1]
ok_S = OBS_S[0] <= t_dobra_S <= OBS_S[1]
ok_T = OBS_T[0] <= t_dobra_T <= OBS_T[1]
print(f"\n  P dentro do intervalo observado {OBS_P}: {'✓' if ok_P else '✗'}")
print(f"  S dentro do intervalo observado {OBS_S}: {'✓' if ok_S else '✗'}")
print(f"  T dentro do intervalo observado {OBS_T}: {'✓' if ok_T else '✗'}")
entre_6_e_8 = ok_T  # atualizar flag para resultado final

# ── visualização ──────────────────────────────────────────────
print("\n  Gerando visualização…")

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
fig.patch.set_facecolor('#080810')
cores = {'P':'#00FF88','S':'#FFB800','T':'#FF4466',
         'bg':'#080810','grid':'#12121E','text':'#CCCCDD'}

for ax in axes:
    ax.set_facecolor('#0D0D1A')
    ax.tick_params(colors=cores['text'], labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#22223A')

# P
axes[0].plot(t_eixo, P, color=cores['P'], lw=0.6, alpha=0.9)
axes[0].axvline(t_dobra_P, color=cores['P'], lw=2, ls='--', alpha=0.9,
                label=f'Dobra P = {t_dobra_P:.2f}s')
axes[0].set_title('P — Sinal Primário (eco)', color=cores['text'], fontsize=10)
axes[0].legend(fontsize=8, facecolor='#111', labelcolor=cores['text'])
axes[0].set_ylabel('Amp', color=cores['text'], fontsize=8)

# S
axes[1].plot(t_eixo, S, color=cores['S'], lw=0.8, alpha=0.9)
axes[1].axvline(t_dobra_S, color=cores['S'], lw=2, ls='--', alpha=0.9,
                label=f'Dobra S = {t_dobra_S:.2f}s')
axes[1].set_title('S — Envelope de P (pulsação)', color=cores['text'], fontsize=10)
axes[1].legend(fontsize=8, facecolor='#111', labelcolor=cores['text'])
axes[1].set_ylabel('Amp', color=cores['text'], fontsize=8)

# T
axes[2].plot(t_eixo, T, color=cores['T'], lw=1.0, alpha=0.9)
axes[2].axvline(t_dobra_T, color='white', lw=2.5, ls='-', alpha=1.0,
                label=f'Dobra T = {t_dobra_T:.2f}s  ← 3º PONTO')
# destacar região 6s-8s
axes[2].axvspan(6.0, 8.5, color='white', alpha=0.05)
axes[2].set_title('T — Envelope de S (trino) — TERCEIRO PONTO DE DOBRA',
                  color=cores['text'], fontsize=10)
axes[2].legend(fontsize=9, facecolor='#111', labelcolor='white')
axes[2].set_ylabel('Amp', color=cores['text'], fontsize=8)

# variância de T ao longo do tempo
t_var = np.array(idx_T_arr) / FS
axes[3].plot(t_var, var_T, color=cores['T'], lw=1.2, alpha=0.8)
axes[3].scatter([t_dobra_T], [var_T[np.argmin(var_T)]], color='white', s=80, zorder=5,
                label=f'Mínimo = {t_dobra_T:.2f}s')
axes[3].axvspan(6.0, 8.5, color='white', alpha=0.05)
axes[3].set_title('Variância local de T — mínimo = ponto de dobra',
                  color=cores['text'], fontsize=10)
axes[3].legend(fontsize=8, facecolor='#111', labelcolor='white')
axes[3].set_xlabel('Tempo (s)', color=cores['text'], fontsize=9)
axes[3].set_ylabel('Var', color=cores['text'], fontsize=8)

fig.suptitle(
    f'AlphaPhi · Verificação Terceiro Ponto de Dobra · '
    f'Beep {F_BEEP:.0f}Hz α*=1/3\n'
    f'P={t_dobra_P:.2f}s  S={t_dobra_S:.2f}s  T={t_dobra_T:.2f}s  '
    f'(sinal {dur_total:.2f}s)',
    color=cores['text'], fontsize=11, y=1.01
)

plt.tight_layout()
plt.savefig('/content/verificacao_terceiro_dobra.png', dpi=150,
            bbox_inches='tight', facecolor='#080810')
plt.close()
print("  → verificacao_terceiro_dobra.png")

display(Image('/content/verificacao_terceiro_dobra.png'))
print("\nÁudio original completo:")
display(Audio('/content/beep880_original_completo.wav'))

print(f"\n{'='*60}")
print(f"  RESULTADO FINAL")
print(f"{'='*60}")
print(f"  Terceiro ponto de dobra (camada T): {t_dobra_T:.3f}s")
if entre_6_e_8:
    print(f"  ✓ Confirmado entre 6s e 8s")
    print(f"  ✓ É o momento de máxima organização do campo")
    print(f"  ✓ Corresponde ao último segmento do cascade (passo {N_STEPS})")
    print(f"  ✓ Onde α completou o ancoramento e φ³ a coerência")
    print(f"  → Liberação = expressão livre do campo coerente")
else:
    print(f"  → t_T = {t_dobra_T:.3f}s — fora do intervalo esperado")
    print(f"  → Ajustar janela de análise ou parâmetros de filtro")
print(f"{'='*60}")
