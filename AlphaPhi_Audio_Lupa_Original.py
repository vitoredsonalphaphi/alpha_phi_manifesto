"""
AlphaPhi Lupa Original — Beep 880Hz eco α*=0.3333
Observação pura do áudio original — sem filtros, sem modificação.

Método: STFT (Short-Time Fourier Transform) = espectrograma
  → FFT em janelas deslizantes sobre o áudio original
  → Mostra como o espectro evolui no tempo
  → NÃO modifica o sinal — só observa

As 3 camadas de vibração são observadas DENTRO do original:
  [P] Banda primária   880Hz ± harmônicos ímpares (onda quadrada)
  [S] Banda secundária FM sidebands 80–350Hz
  [T] Banda terciária  batimentos 20–80Hz (o "trino")

Os pontos de dobra são marcados no tempo como linhas verticais.

Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.signal import stft, butter, filtfilt
from scipy.io import wavfile
from IPython.display import HTML, Audio, display

# ── constantes ────────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI           # ≈ 136 Hz
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5                    # segundos por janela
N_STEPS    = 5

# Bandas de observação (não filtram — apenas definem onde olhar no espectro)
BANDAS_OBS = {
    'P': (600,  5000, '#00FF88', 'P — Primária  (880Hz + harmônicos)'),
    'S': (80,   350,  '#FFB800', 'S — Secundária (FM sidebands 80–350Hz)'),
    'T': (20,   80,   '#FF4466', 'T — Terciária  (batimentos 20–80Hz)'),
}

CORES = {'bg': '#080810', 'grid': '#12121E', 'text': '#CCCCDD',
         'dobra': '#FFFFFF'}

print("AlphaPhi Lupa Original — observação pura via STFT")
print(f"Beep {F_BEEP:.0f}Hz | α*={ALPHA_STAR:.4f} | sem modificação do áudio\n")

# ── pipeline eco (idêntico ao original) ──────────────────────────────────────
def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max:
            break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    result = []
    for f_lo, f_hi in bandas:
        b_lo = max(0, int(f_lo / (FS / n)))
        b_hi = min(int(f_hi / (FS / n)) + 1, n // 2 + 1)
        result.append((b_lo, b_hi, f_lo, f_hi))
    return result

N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    N = len(x)
    F = np.fft.rfft(x)
    F_out = F.copy()
    cohs = []
    w_mem, w_now = 1.0/PHI, 1.0-1.0/PHI
    for i, (b_low, b_high, _, _) in enumerate(bins_phi):
        beta_i = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        F_band = F[b_low:b_high]
        mag, phase = np.abs(F_band), np.angle(F_band)
        an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        coh_ef = (w_now*coh + w_mem*float(coh_mem[i])
                  if coh_mem is not None and i < len(coh_mem) else coh)
        cohs.append(coh)
        n_idx = np.arange(len(F_band))
        env   = np.clip(1.0+(coh_ef*PHI**beta_i)*np.cos(2*np.pi*n_idx/PHI),
                        0.05, None)
        F_out[b_low:b_high] = (mag*env)*np.exp(1j*phase)
    r = np.fft.irfft(F_out, n=N)
    return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

def normalizar(s):
    m = np.max(np.abs(s))
    return s/m if m > 1e-12 else s

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    coh_mem = np.zeros(N_BANDAS)
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem = cohs
        s_e = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()
    return cas, cohs

def agente_eco(sinal, bins_phi, n_ciclos=20):
    n_b = len(bins_phi)
    beta = np.ones(n_b)
    beta_mem = beta.copy()
    w_mem, w_now = 1.0/PHI, 1.0-1.0/PHI
    cas_f = None
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        coh_rel   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        beta_alvo = PHI**(3*coh_rel)
        beta      = w_now*beta_alvo + w_mem*beta_mem
        beta_mem  = beta.copy()
        beta      = np.clip(beta, 0.05, PHI**3)
        cas_f     = cas
    return beta, cas_f

def gerar_beep():
    t = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
    return normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t)))

def gerar_fm():
    t = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
    return normalizar(np.sin(2*np.pi*F_ORG*t + BETA_FM*np.sin(2*np.pi*F_M*t)))

def crossfade(a, b, fade=2000):
    fade = min(fade, len(a), len(b))
    t    = np.linspace(0.0, 1.0, fade)
    out  = a.copy()
    out[-fade:] = a[-fade:]*(1-t) + b[:fade]*t
    return np.concatenate([out, b[fade:]])

# ── gerar áudio original ──────────────────────────────────────────────────────
print("[1/4] Gerando áudio original (cascata completa)...")
beep  = gerar_beep()
fm    = gerar_fm()
x_mix = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)
beta_final, cas = agente_eco(x_mix, BINS_PHI, n_ciclos=20)

# concatena os 6 passos com crossfade curto — exatamente como no original
FADE_N = int(0.02 * FS)  # 20ms crossfade
audio_original = cas[0].copy()
for passo in cas[1:]:
    audio_original = crossfade(audio_original, passo, fade=FADE_N)
audio_original = normalizar(audio_original)

DUR_TOTAL = len(audio_original) / FS
# tempos dos pontos de dobra (início de cada novo passo)
T_DOBRAS  = [DURACAO * (i+1) for i in range(N_STEPS)]

print(f"  β_max={beta_final.max():.4f}  φ³={PHI**3:.4f}")
print(f"  Duração total: {DUR_TOTAL:.2f}s")
print(f"  Pontos de dobra: {[f'{t:.1f}s' for t in T_DOBRAS]}\n")

# salva o original intacto
s16 = np.int16(np.clip(audio_original, -1.0, 1.0) * 32767)
wavfile.write("beep880_original_completo.wav", FS, s16)
print("  → beep880_original_completo.wav (áudio original, sem modificação)\n")

# ── STFT — espectrograma ──────────────────────────────────────────────────────
print("[2/4] Calculando espectrograma STFT...")

NPERSEG = 2048       # janela FFT (~46ms) — boa resolução freq e tempo
NOVERLAP = 1800      # sobreposição alta → espectrograma suave

freqs_stft, tempos_stft, Zxx = stft(
    audio_original, fs=FS,
    nperseg=NPERSEG, noverlap=NOVERLAP,
    window='hann'
)
Sxx = np.abs(Zxx)**2                          # potência
Sxx_db = 10*np.log10(Sxx + 1e-12)            # em dB

# energia por banda ao longo do tempo
def energia_banda(f_low, f_high):
    mask = (freqs_stft >= f_low) & (freqs_stft <= f_high)
    return Sxx[mask, :].mean(axis=0)

energia = {k: energia_banda(v[0], v[1]) for k, v in BANDAS_OBS.items()}

# normaliza cada banda para comparação
for k in energia:
    e = energia[k]
    energia[k] = (e - e.min()) / (e.max() - e.min() + 1e-10)

print(f"  Janela STFT: {NPERSEG/FS*1000:.0f}ms | "
      f"resolução freq: {FS/NPERSEG:.1f}Hz | "
      f"pontos tempo: {len(tempos_stft)}\n")

# ── figura principal ──────────────────────────────────────────────────────────
print("[3/4] Gerando figura principal...")

fig = plt.figure(figsize=(16, 13), facecolor=CORES['bg'])
fig.suptitle(
    f"Lupa Original — Beep 880Hz eco α*={ALPHA_STAR:.4f}\n"
    f"Observação via STFT — áudio inalterado — {DUR_TOTAL:.1f}s",
    color=CORES['text'], fontsize=11, y=1.005
)
gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45)

t_eixo = np.linspace(0, DUR_TOTAL, len(audio_original))

# ── painel 1: forma de onda ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(CORES['grid'])
ax1.plot(t_eixo, audio_original, color='#4488FF', lw=0.4, alpha=0.8)
for td in T_DOBRAS:
    ax1.axvline(td, color=CORES['dobra'], lw=0.8, ls='--', alpha=0.7)
    ax1.text(td+0.03, 0.85, f'dobra\n{td:.1f}s',
             color=CORES['dobra'], fontsize=6.5, va='top')
ax1.set_ylabel("Amplitude", color=CORES['text'], fontsize=8)
ax1.set_xlim(0, DUR_TOTAL)
ax1.set_ylim(-1.1, 1.1)
ax1.set_title("Forma de onda original — pontos de dobra marcados",
              color=CORES['text'], fontsize=9)
ax1.tick_params(colors=CORES['text'], labelsize=7)
for sp in ax1.spines.values(): sp.set_edgecolor('#1A1A2E')

# ── painel 2: espectrograma ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(CORES['grid'])
vmin = np.percentile(Sxx_db, 20)
vmax = np.percentile(Sxx_db, 99.5)
img = ax2.pcolormesh(tempos_stft, freqs_stft, Sxx_db,
                     shading='gouraud', cmap='inferno',
                     vmin=vmin, vmax=vmax)
# marca bandas
for k, (f_lo, f_hi, cor, label) in BANDAS_OBS.items():
    ax2.axhspan(f_lo, f_hi, alpha=0.12, color=cor)
    ax2.text(0.15, (f_lo+f_hi)/2, k, color=cor,
             fontsize=8, fontweight='bold', va='center')
for td in T_DOBRAS:
    ax2.axvline(td, color=CORES['dobra'], lw=0.8, ls='--', alpha=0.6)
ax2.set_ylim(15, 5000)
ax2.set_yscale('log')
ax2.set_ylabel("Frequência (Hz)", color=CORES['text'], fontsize=8)
ax2.set_title("Espectrograma STFT — 3 bandas de observação destacadas",
              color=CORES['text'], fontsize=9)
ax2.tick_params(colors=CORES['text'], labelsize=7)
for sp in ax2.spines.values(): sp.set_edgecolor('#1A1A2E')
plt.colorbar(img, ax=ax2, label='dB').ax.tick_params(
    colors=CORES['text'], labelsize=6)

# ── painel 3: energia das 3 bandas ao longo do tempo ─────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(CORES['grid'])
for k, (_, _, cor, label) in BANDAS_OBS.items():
    ax3.plot(tempos_stft, energia[k], color=cor, lw=1.1,
             label=label, alpha=0.9)
for td in T_DOBRAS:
    ax3.axvline(td, color=CORES['dobra'], lw=0.8, ls='--', alpha=0.6)
ax3.set_ylabel("Energia (norm.)", color=CORES['text'], fontsize=8)
ax3.set_xlim(0, DUR_TOTAL)
ax3.set_ylim(-0.05, 1.1)
ax3.set_title("Energia por banda P / S / T ao longo do tempo",
              color=CORES['text'], fontsize=9)
ax3.legend(fontsize=7, facecolor=CORES['grid'],
           labelcolor=CORES['text'], loc='upper right')
ax3.tick_params(colors=CORES['text'], labelsize=7)
for sp in ax3.spines.values(): sp.set_edgecolor('#1A1A2E')

# ── painel 4: zoom nos pontos de dobra (seg 2.5 a 8) ─────────────────────────
ax4 = fig.add_subplot(gs[3])
ax4.set_facecolor(CORES['grid'])
T_ZOOM_I, T_ZOOM_F = 2.5, DUR_TOTAL
mask_z = (tempos_stft >= T_ZOOM_I) & (tempos_stft <= T_ZOOM_F)
for k, (_, _, cor, _) in BANDAS_OBS.items():
    ax4.plot(tempos_stft[mask_z], energia[k][mask_z],
             color=cor, lw=1.4, alpha=0.95, label=k)
for td in T_DOBRAS:
    if T_ZOOM_I <= td <= T_ZOOM_F:
        ax4.axvline(td, color=CORES['dobra'], lw=1.0, ls='--', alpha=0.8)
        ax4.text(td+0.03, 1.02, f'dobra\n{td:.1f}s',
                 color=CORES['dobra'], fontsize=7, va='bottom')
ax4.set_ylabel("Energia (norm.)", color=CORES['text'], fontsize=8)
ax4.set_xlabel("Tempo (s)", color=CORES['text'], fontsize=8)
ax4.set_xlim(T_ZOOM_I, T_ZOOM_F)
ax4.set_ylim(-0.05, 1.15)
ax4.set_title("Zoom — pontos de dobra (2,5s → fim) — P / S / T",
              color=CORES['text'], fontsize=9)
ax4.legend(fontsize=8, facecolor=CORES['grid'],
           labelcolor=CORES['text'], loc='upper left')
ax4.tick_params(colors=CORES['text'], labelsize=7)
for sp in ax4.spines.values(): sp.set_edgecolor('#1A1A2E')

plt.tight_layout()
plt.savefig("lupa_original.png", dpi=150,
            bbox_inches='tight', facecolor=CORES['bg'])
plt.show()
print("  → lupa_original.png\n")

# ── animação osciloscópio — varredura do original ─────────────────────────────
print("[4/4] Gerando animação osciloscópio...")
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 60   # MB

N_JANELA  = int(0.08 * FS)          # janela visível: 80ms
N_TOT     = len(audio_original)
N_FRAMES  = 200
STEP_FR   = max(1, (N_TOT - N_JANELA) // N_FRAMES)

fig_osc = plt.figure(figsize=(14, 8), facecolor=CORES['bg'])
fig_osc.suptitle(
    f"Osciloscópio — Beep 880Hz eco α*={ALPHA_STAR:.4f} — áudio original\n"
    "Janela 80ms varrendo o sinal completo | P/S/T = energia por banda",
    color=CORES['text'], fontsize=10
)
gs_o = gridspec.GridSpec(3, 2, figure=fig_osc, hspace=0.5, wspace=0.3)

ax_onda = fig_osc.add_subplot(gs_o[:2, 0])   # forma de onda (2/3 da altura)
ax_esp  = fig_osc.add_subplot(gs_o[:2, 1])   # espectro instantâneo
ax_p    = fig_osc.add_subplot(gs_o[2, 0])    # energia P
ax_st_e = fig_osc.add_subplot(gs_o[2, 1])    # energia S e T

for ax in [ax_onda, ax_esp, ax_p, ax_st_e]:
    ax.set_facecolor(CORES['grid'])
    ax.tick_params(colors=CORES['text'], labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor('#1A1A2E')

ax_onda.set_ylabel("Amplitude", color=CORES['text'], fontsize=8)
ax_onda.set_xlabel("ms (janela 80ms)", color=CORES['text'], fontsize=7)
ax_esp.set_ylabel("Magnitude (dB)", color=CORES['text'], fontsize=8)
ax_esp.set_xlabel("Hz", color=CORES['text'], fontsize=7)
ax_p.set_xlabel("Tempo (s)", color=CORES['text'], fontsize=7)
ax_st_e.set_xlabel("Tempo (s)", color=CORES['text'], fontsize=7)

t_jan_ms = np.linspace(0, 80, N_JANELA)
freq_jan  = np.fft.rfftfreq(N_JANELA, d=1.0/FS)
mask_fj   = (freq_jan > 15) & (freq_jan < 5000)

linha_onda,  = ax_onda.plot([], [], color='#4488FF', lw=0.7)
linha_esp,   = ax_esp.plot([], [], color='#AAAAFF', lw=0.8)
ponto_tempo  = ax_onda.axvline(0, color='white', lw=0.4, alpha=0.4)

# barras de energia das 3 bandas
barra_p = ax_p.bar(['P'], [0], color=BANDAS_OBS['P'][2], width=0.5)
barra_s = ax_st_e.bar(['S', 'T'], [0, 0],
                      color=[BANDAS_OBS['S'][2], BANDAS_OBS['T'][2]],
                      width=0.5)

# linha de progresso no espectrograma (fundo)
ax_onda.set_title("Forma de onda", color=CORES['text'], fontsize=8)
ax_esp.set_title("Espectro instantâneo (FFT janela)", color=CORES['text'], fontsize=8)
ax_p.set_title("Energia P", color=BANDAS_OBS['P'][2], fontsize=8)
ax_st_e.set_title("Energia S / T", color=CORES['text'], fontsize=8)

# linha de posição temporal sobre painel de energia
ax_p.set_xlim(-0.5, 1.5)
ax_p.set_ylim(0, 1.2)
ax_st_e.set_xlim(-0.5, 2.5)
ax_st_e.set_ylim(0, 1.2)

tempo_label = fig_osc.text(0.5, 0.01, '', ha='center',
                            color=CORES['text'], fontsize=9)

def init_osc2():
    linha_onda.set_data([], [])
    linha_esp.set_data([], [])
    return linha_onda, linha_esp

def update_osc2(frame):
    i0 = frame * STEP_FR
    i1 = i0 + N_JANELA
    if i1 > N_TOT:
        return linha_onda, linha_esp

    janela = audio_original[i0:i1]
    t_atual = i0 / FS

    # forma de onda
    linha_onda.set_data(t_jan_ms, janela)
    ax_onda.set_xlim(0, 80)
    ax_onda.set_ylim(-1.15, 1.15)

    # marca se está numa dobra
    em_dobra = any(abs(t_atual - td) < 0.1 for td in T_DOBRAS)
    linha_onda.set_color('#FF4466' if em_dobra else '#4488FF')
    ax_onda.set_title(
        f"{'★ PONTO DE DOBRA ★' if em_dobra else 'Forma de onda'}",
        color='#FF4466' if em_dobra else CORES['text'], fontsize=8
    )

    # espectro instantâneo
    mag = 20*np.log10(np.abs(np.fft.rfft(janela))[mask_fj] + 1e-8)
    linha_esp.set_data(freq_jan[mask_fj], mag)
    ax_esp.set_xlim(15, 5000)
    ax_esp.set_xscale('log')
    ax_esp.set_ylim(mag.min()-5, mag.max()+5)

    # marcadores de banda no espectro
    ax_esp.set_facecolor(CORES['grid'])
    for k, (f_lo, f_hi, cor, _) in BANDAS_OBS.items():
        ax_esp.axvspan(f_lo, f_hi, alpha=0.12, color=cor)

    # energia por banda na janela
    def e_banda(f_lo, f_hi):
        mag_lin = np.abs(np.fft.rfft(janela))
        mask_b  = (freq_jan >= f_lo) & (freq_jan <= f_hi)
        return float(mag_lin[mask_b].mean()) if mask_b.any() else 0.0

    ep = e_banda(*BANDAS_OBS['P'][:2])
    es = e_banda(*BANDAS_OBS['S'][:2])
    et = e_banda(*BANDAS_OBS['T'][:2])
    mx = max(ep, es, et, 1e-6)

    for rect, h in zip(barra_p, [ep/mx]):
        rect.set_height(h)
    for rect, h in zip(barra_s, [es/mx, et/mx]):
        rect.set_height(h)

    tempo_label.set_text(
        f"t = {t_atual:.2f}s / {DUR_TOTAL:.1f}s  |  "
        f"P={ep/mx:.2f}  S={es/mx:.2f}  T={et/mx:.2f}  |  "
        f"β_max={beta_final.max():.3f}"
    )
    return linha_onda, linha_esp

anim2 = FuncAnimation(
    fig_osc, update_osc2, init_func=init_osc2,
    frames=N_FRAMES, interval=50, blit=False
)

print("Exibindo osciloscópio animado...")
display(HTML(anim2.to_jshtml()))

# ── playback do original ──────────────────────────────────────────────────────
print("\n── Playback — áudio original completo (sem modificação) ──")
display(Audio("beep880_original_completo.wav"))

print(f"""
── Síntese ──
  Duração total : {DUR_TOTAL:.2f}s
  Pontos de dobra: {[f'{t:.1f}s' for t in T_DOBRAS]}

  Bandas observadas (sem filtrar):
    P: 600–5000Hz  (fundamental 880Hz + harmônicos ímpares da onda quadrada)
    S:  80–350Hz   (FM sidebands: 220±136Hz = 84Hz e 356Hz)
    T:  20–80Hz    (batimentos: 55−34=21Hz, 55+34=89Hz, atrator ÷4=10.5Hz)

  No espectrograma: observe como cada banda muda ao cruzar cada dobra.
  No gráfico de energia: veja o cruzamento P↔S↔T nos pontos de dobra.
  O trino (T) deve aparecer como flutuação lenta sob as outras duas.

Concluído.
""")
