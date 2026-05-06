"""
AlphaPhi Osciloscópio v2 — Lupa Multinível por Envelope Hilbert
Beep 880Hz eco α*=0.3333 — Ponto de Dobra

Extrai as 3 camadas de vibração perceptíveis no ponto de dobra:

  [P] Primário   — o sinal eco: a vibração principal (o campo)
  [S] Secundário — envelope de amplitude de P via Hilbert
                   como P pulsa internamente — o "tênue em outro ritmo"
  [T] Terciário  — envelope de amplitude de S via Hilbert
                   como S pulsa — o "trino", quase inaudível

A pergunta do experimento:
  Em que frequência vibra S em relação a P?
  Em que frequência vibra T em relação a S?
  A razão é φ? É 3? É outra coisa?

O código não assume resposta — deixa o sinal falar.

Célula única para Google Colab.
Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.signal import hilbert, butter, filtfilt, spectrogram
from scipy.io import wavfile
from IPython.display import HTML, Audio, display

# ── constantes ────────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5
N_STEPS    = 5

CORES = {
    'P': '#00FF88',
    'S': '#FFB800',
    'T': '#FF4466',
    'bg': '#080810',
    'grid': '#12121E',
    'text': '#CCCCDD',
}

print("AlphaPhi Osciloscópio v2 — Lupa Multinível")
print(f"Beep {F_BEEP:.0f}Hz | α*={ALPHA_STAR:.4f} | Hilbert multinível\n")

# ── bandas φ ──────────────────────────────────────────────────────────────────
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

# ── eco_eq + cascata + agente ─────────────────────────────────────────────────
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
        env   = np.clip(1.0+(coh_ef*PHI**beta_i)*np.cos(2*np.pi*n_idx/PHI), 0.05, None)
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
        coh_rel  = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        beta_alvo = PHI**(3*coh_rel)
        beta = w_now*beta_alvo + w_mem*beta_mem
        beta_mem = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        cas_f = cas
    return beta, cas_f

def gerar_beep():
    t = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
    return normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t)))

def gerar_fm():
    t = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
    return normalizar(np.sin(2*np.pi*F_ORG*t + BETA_FM*np.sin(2*np.pi*F_M*t)))

# ── extração multinível por Hilbert ──────────────────────────────────────────
def lowpass(sig, fc, fs=FS, order=4):
    """Filtro passa-baixo Butterworth."""
    b, a = butter(order, fc/(fs/2), btype='low')
    return filtfilt(b, a, sig)

def envelope_hilbert(sig, fc_suavizacao=None, fs=FS):
    """Extrai envelope de amplitude via transformada de Hilbert."""
    env = np.abs(hilbert(sig))
    if fc_suavizacao is not None:
        env = lowpass(env, fc_suavizacao, fs)
    return env

def extrair_3_camadas(sinal_eco, fc_s=200.0, fc_t=50.0):
    """
    Extrai P, S, T do sinal eco por envelope aninhado.

    P = sinal_eco normalizado           (a vibração principal)
    S = envelope de amplitude de P      (como P pulsa — o tênue)
    T = envelope de amplitude de S      (como S pulsa — o trino)

    fc_s: frequência de corte para suavizar o envelope de P → S
    fc_t: frequência de corte para suavizar o envelope de S → T
    """
    P = normalizar(sinal_eco)
    S = envelope_hilbert(P, fc_suavizacao=fc_s)
    S = normalizar(S - S.mean())          # remove DC, normaliza
    T = envelope_hilbert(S, fc_suavizacao=fc_t)
    T = normalizar(T - T.mean())
    return P, S, T

def freq_dominante(sig, fs=FS):
    """Frequência dominante do sinal (excluindo DC)."""
    F = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0/fs)
    F[freqs < 1.0] = 0
    return freqs[np.argmax(F)]

def salvar_wav(sig, nome, fs=FS):
    s16 = np.int16(np.clip(normalizar(sig), -1.0, 1.0) * 32767)
    wavfile.write(nome, fs, s16)
    print(f"  → {nome}  ({len(sig)/fs:.2f}s  |  fs={fs}Hz)")

# ── pipeline ──────────────────────────────────────────────────────────────────
print("[1/4] Gerando sinais e rodando agente_eco...")
beep  = gerar_beep()
fm    = gerar_fm()
x_mix = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)
beta_final, cas = agente_eco(x_mix, BINS_PHI, n_ciclos=20)
print(f"  β_max={beta_final.max():.4f}  (φ³={PHI**3:.4f})")

# passo 3 = primeiro pós-dobra principal (onde a sensação começa)
PASSO = 3
sinal_dobra = np.array(cas[PASSO])
print(f"  Passo selecionado: {PASSO} (pós-dobra principal)\n")

# ── varredura de parâmetros fc_s e fc_t ──────────────────────────────────────
# Mostra P, S, T para 3 configurações de corte, para comparação
CONFIGS_FC = [
    (300.0, 80.0,  "largo"),
    (150.0, 40.0,  "médio"),
    ( 60.0, 15.0,  "estreito"),
]

print("[2/4] Extraindo 3 camadas (P, S, T) em 3 configurações de corte...")

resultados = []
for fc_s, fc_t, nome in CONFIGS_FC:
    P, S, T = extrair_3_camadas(sinal_dobra, fc_s=fc_s, fc_t=fc_t)
    fp = freq_dominante(P)
    fs_d = freq_dominante(S)
    ft = freq_dominante(T)
    ratio_sp = fs_d / (fp + 1e-10)
    ratio_ts = ft  / (fs_d + 1e-10)
    resultados.append((P, S, T, fp, fs_d, ft, ratio_sp, ratio_ts, nome, fc_s, fc_t))
    print(f"  [{nome}]  fc_s={fc_s:.0f}Hz  fc_t={fc_t:.0f}Hz")
    print(f"    P: {fp:.2f}Hz  |  S: {fs_d:.2f}Hz  |  T: {ft:.2f}Hz")
    print(f"    S/P = {ratio_sp:.4f}  |  T/S = {ratio_ts:.4f}")
    print(f"    φ   = {PHI:.4f}  |  1/φ  = {1/PHI:.4f}  |  1/3 = {1/3:.4f}")
    print()

# configuração principal: médio
P, S, T, fp, fs_d, ft, ratio_sp, ratio_ts, _, fc_s_ref, fc_t_ref = resultados[1]

# ── salvamento de áudio ───────────────────────────────────────────────────────
print("[3/4] Salvando áudios...")

# normal speed
salvar_wav(P, "oscilo_P_normal.wav")
salvar_wav(S, "oscilo_S_normal.wav")
salvar_wav(T, "oscilo_T_normal.wav")

# 0.25x via downsample (divide fs por 4)
FS_025 = FS // 4
salvar_wav(P, "oscilo_P_025x.wav", fs=FS_025)
salvar_wav(S, "oscilo_S_025x.wav", fs=FS_025)
salvar_wav(T, "oscilo_T_025x.wav", fs=FS_025)
print()

# ── figura principal — P, S, T no tempo ──────────────────────────────────────
print("[4/4] Gerando figuras...")

T_EIXO = np.linspace(0, DURACAO*1000, N_SINAL)
FREQ_FFT = np.fft.rfftfreq(N_SINAL, d=1.0/FS)
MASK_F   = (FREQ_FFT > 1.0) & (FREQ_FFT < 8000)

fig = plt.figure(figsize=(16, 12), facecolor=CORES['bg'])
fig.suptitle(
    f"Lupa Multinível — Beep 880Hz eco α*={ALPHA_STAR:.4f} — Passo {PASSO}\n"
    f"P: {fp:.1f}Hz  |  S: {fs_d:.1f}Hz  (S/P={ratio_sp:.3f})"
    f"  |  T: {ft:.1f}Hz  (T/S={ratio_ts:.3f})"
    f"  |  φ={PHI:.4f}  |  1/3={1/3:.4f}",
    color=CORES['text'], fontsize=10, y=1.01
)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)

camadas = [
    (P, 'P — Primário (eco)',          CORES['P'], fp),
    (S, 'S — Secundário (envelope P)', CORES['S'], fs_d),
    (T, 'T — Terciário / trino (env S)',CORES['T'], ft),
]

for linha, (sig, label, cor, f_dom) in enumerate(camadas):
    mag = np.abs(np.fft.rfft(sig))

    # Tempo — visão completa
    ax_t = fig.add_subplot(gs[linha, 0])
    ax_t.set_facecolor(CORES['grid'])
    ax_t.plot(T_EIXO, sig, color=cor, lw=0.6)
    ax_t.axhline(0, color=CORES['text'], lw=0.3, alpha=0.4)
    ax_t.set_title(f"{label}\nf_dom={f_dom:.1f}Hz", color=cor, fontsize=8)
    ax_t.set_xlabel("ms", color=CORES['text'], fontsize=7)
    ax_t.tick_params(colors=CORES['text'], labelsize=6)
    rms = np.sqrt(np.mean(sig**2))
    ax_t.set_ylim(-max(rms*5, 0.05), max(rms*5, 0.05))
    ax_t.set_xlim(0, DURACAO*1000)
    for sp in ax_t.spines.values():
        sp.set_edgecolor('#1A1A2E')

    # Tempo — zoom nos primeiros 50ms (lupa)
    ax_z = fig.add_subplot(gs[linha, 1])
    ax_z.set_facecolor(CORES['grid'])
    n_zoom = int(0.05 * FS)
    ax_z.plot(T_EIXO[:n_zoom], sig[:n_zoom], color=cor, lw=0.8)
    ax_z.axhline(0, color=CORES['text'], lw=0.3, alpha=0.4)
    ax_z.set_title(f"Zoom — primeiros 50ms", color=CORES['text'], fontsize=8)
    ax_z.set_xlabel("ms", color=CORES['text'], fontsize=7)
    ax_z.tick_params(colors=CORES['text'], labelsize=6)
    ax_z.set_ylim(-max(rms*5, 0.05), max(rms*5, 0.05))
    for sp in ax_z.spines.values():
        sp.set_edgecolor('#1A1A2E')

    # FFT
    ax_f = fig.add_subplot(gs[linha, 2])
    ax_f.set_facecolor(CORES['grid'])
    ax_f.semilogy(FREQ_FFT[MASK_F], mag[MASK_F]+1e-8, color=cor, lw=0.8)
    ax_f.axvline(f_dom, color='white', lw=0.8, ls='--', alpha=0.6)
    ax_f.set_title(f"FFT", color=CORES['text'], fontsize=8)
    ax_f.set_xlabel("Hz", color=CORES['text'], fontsize=7)
    ax_f.tick_params(colors=CORES['text'], labelsize=6)
    for sp in ax_f.spines.values():
        sp.set_edgecolor('#1A1A2E')

# Linha extra: sobreposição dos 3 envelopes
ax_ov = fig.add_subplot(gs[3, :])
ax_ov.set_facecolor(CORES['grid'])
ax_ov.plot(T_EIXO, P/np.max(np.abs(P)+1e-10),
           color=CORES['P'], lw=0.5, alpha=0.7, label='P (norm)')
ax_ov.plot(T_EIXO, S/np.max(np.abs(S)+1e-10),
           color=CORES['S'], lw=0.8, alpha=0.85, label='S (norm)')
ax_ov.plot(T_EIXO, T/np.max(np.abs(T)+1e-10),
           color=CORES['T'], lw=1.0, alpha=0.95, label='T (norm)')
ax_ov.axhline(0, color=CORES['text'], lw=0.3, alpha=0.3)
ax_ov.set_title("Sobreposição P + S + T normalizados — relação de fases",
                color=CORES['text'], fontsize=9)
ax_ov.set_xlabel("ms", color=CORES['text'], fontsize=8)
ax_ov.set_xlim(0, DURACAO*1000)
ax_ov.legend(loc='upper right', fontsize=8, facecolor=CORES['grid'],
             labelcolor=CORES['text'])
ax_ov.tick_params(colors=CORES['text'], labelsize=7)
for sp in ax_ov.spines.values():
    sp.set_edgecolor('#1A1A2E')

plt.tight_layout()
plt.savefig("oscilo_multinivel.png", dpi=150,
            bbox_inches='tight', facecolor=CORES['bg'])
plt.show()
print("  → oscilo_multinivel.png")

# ── animação osciloscópio — varredura em tempo real das 3 camadas ─────────────
print("Gerando animação osciloscópio...")

# janela de varredura: 50ms (como feixe de elétrons varrendo a tela)
N_JANELA  = int(0.05 * FS)          # 50ms visíveis por vez
N_FRAMES  = N_SINAL - N_JANELA      # total de frames
STEP      = max(1, N_FRAMES // 180) # ~180 frames para animação fluida

fig_anim = plt.figure(figsize=(14, 9), facecolor=CORES['bg'])
fig_anim.suptitle(
    f"Osciloscópio — Beep 880Hz eco α*={ALPHA_STAR:.4f}\n"
    f"P={fp:.0f}Hz  S={fs_d:.0f}Hz  T={ft:.0f}Hz  "
    f"|  S/P={ratio_sp:.3f}  T/S={ratio_ts:.3f}  |  φ={PHI:.4f}",
    color=CORES['text'], fontsize=10, y=1.01
)

gs_a = gridspec.GridSpec(3, 2, figure=fig_anim, hspace=0.5, wspace=0.35)

# painéis esquerda = forma de onda (varredura)
# painéis direita  = FFT acumulada (estática, atualiza suavemente)
ax_pt = fig_anim.add_subplot(gs_a[0, 0])
ax_st = fig_anim.add_subplot(gs_a[1, 0])
ax_tt = fig_anim.add_subplot(gs_a[2, 0])
ax_pf = fig_anim.add_subplot(gs_a[0, 1])
ax_sf = fig_anim.add_subplot(gs_a[1, 1])
ax_tf = fig_anim.add_subplot(gs_a[2, 1])

for ax, cor, label in [
    (ax_pt, CORES['P'], 'P — Primário'),
    (ax_st, CORES['S'], 'S — Secundário (tênue)'),
    (ax_tt, CORES['T'], 'T — Terciário (trino)'),
]:
    ax.set_facecolor(CORES['grid'])
    ax.set_ylabel(label, color=cor, fontsize=8)
    ax.tick_params(colors=CORES['text'], labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor('#1A1A2E')

for ax in [ax_pf, ax_sf, ax_tf]:
    ax.set_facecolor(CORES['grid'])
    ax.set_ylabel("FFT", color=CORES['text'], fontsize=8)
    ax.tick_params(colors=CORES['text'], labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor('#1A1A2E')

ax_tt.set_xlabel("ms (janela 50ms)", color=CORES['text'], fontsize=7)
ax_tf.set_xlabel("Hz", color=CORES['text'], fontsize=7)

t_janela_ms = np.linspace(0, 50, N_JANELA)
freq_j      = np.fft.rfftfreq(N_JANELA, d=1.0/FS)
mask_fj     = (freq_j > 10) & (freq_j < 6000)

lp_t, = ax_pt.plot([], [], color=CORES['P'], lw=0.7)
ls_t, = ax_st.plot([], [], color=CORES['S'], lw=0.8)
lt_t, = ax_tt.plot([], [], color=CORES['T'], lw=0.9)
lp_f, = ax_pf.semilogy([], [], color=CORES['P'], lw=0.8)
ls_f, = ax_sf.semilogy([], [], color=CORES['S'], lw=0.8)
lt_f, = ax_tf.semilogy([], [], color=CORES['T'], lw=0.8)

# linha vertical "ponteiro" do osciloscópio
vlines = [ax.axvline(0, color='white', lw=0.5, alpha=0.4)
          for ax in [ax_pt, ax_st, ax_tt]]

tempo_txt = fig_anim.text(0.5, 0.02, '', ha='center',
                          color=CORES['text'], fontsize=9)

def init_osc():
    for l in [lp_t, ls_t, lt_t, lp_f, ls_f, lt_f]:
        l.set_data([], [])
    return lp_t, ls_t, lt_t, lp_f, ls_f, lt_f

def update_osc(frame):
    i0 = frame * STEP
    i1 = i0 + N_JANELA
    if i1 > N_SINAL:
        return lp_t, ls_t, lt_t, lp_f, ls_f, lt_f

    p_j = P[i0:i1]
    s_j = S[i0:i1]
    t_j = T[i0:i1]

    # forma de onda
    lp_t.set_data(t_janela_ms, p_j)
    ls_t.set_data(t_janela_ms, s_j)
    lt_t.set_data(t_janela_ms, t_j)

    for ax, sig in [(ax_pt, p_j), (ax_st, s_j), (ax_tt, t_j)]:
        rms = np.sqrt(np.mean(sig**2))
        lim = max(rms * 5, 0.02)
        ax.set_xlim(0, 50)
        ax.set_ylim(-lim, lim)

    # FFT da janela
    mp = np.abs(np.fft.rfft(p_j)) + 1e-8
    ms = np.abs(np.fft.rfft(s_j)) + 1e-8
    mt = np.abs(np.fft.rfft(t_j)) + 1e-8
    lp_f.set_data(freq_j[mask_fj], mp[mask_fj])
    ls_f.set_data(freq_j[mask_fj], ms[mask_fj])
    lt_f.set_data(freq_j[mask_fj], mt[mask_fj])
    for ax in [ax_pf, ax_sf, ax_tf]:
        ax.set_xlim(10, 6000)
        ax.relim()
        ax.autoscale_view(scaley=True)

    t_ms_atual = i0 / FS * 1000
    tempo_txt.set_text(
        f"t = {t_ms_atual:.1f}ms / {DURACAO*1000:.0f}ms  "
        f"|  frame {frame+1}/{N_FRAMES//STEP}"
    )
    return lp_t, ls_t, lt_t, lp_f, ls_f, lt_f

anim_osc = FuncAnimation(
    fig_anim, update_osc, init_func=init_osc,
    frames=N_FRAMES // STEP,
    interval=40,       # 40ms por frame ≈ 25fps
    blit=False
)

print("Exibindo osciloscópio animado (varredura das 3 camadas)...")
display(HTML(anim_osc.to_jshtml()))

# ── varredura de fc: como muda a frequência dominante de T? ──────────────────
fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor=CORES['bg'])
ax2.set_facecolor(CORES['grid'])

FCS_RANGE = np.logspace(1, np.log10(500), 40)  # 10 → 500 Hz
ft_vals   = []
for fc in FCS_RANGE:
    _, S_v, T_v = extrair_3_camadas(sinal_dobra, fc_s=fc*3, fc_t=fc)
    ft_vals.append(freq_dominante(T_v))

ax2.semilogx(FCS_RANGE, ft_vals, color=CORES['T'], lw=1.2)
ax2.axhline(fp/3,  color='white', lw=0.6, ls=':', alpha=0.5,
            label=f'f_P/3 = {fp/3:.1f}Hz')
ax2.axhline(fs_d/PHI, color=CORES['S'], lw=0.6, ls=':', alpha=0.5,
            label=f'f_S/φ = {fs_d/PHI:.1f}Hz')
ax2.set_title("Frequência de T em função do corte fc_t\n"
              "(o 'trino' é estável? Em que frequência converge?)",
              color=CORES['text'], fontsize=9)
ax2.set_xlabel("fc_t (Hz)", color=CORES['text'], fontsize=8)
ax2.set_ylabel("f_dom(T) Hz", color=CORES['T'], fontsize=8)
ax2.tick_params(colors=CORES['text'], labelsize=7)
ax2.legend(fontsize=8, facecolor=CORES['grid'], labelcolor=CORES['text'])
for sp in ax2.spines.values():
    sp.set_edgecolor('#1A1A2E')
plt.tight_layout()
plt.savefig("oscilo_varredura_fc.png", dpi=150,
            bbox_inches='tight', facecolor=CORES['bg'])
plt.show()
print("  → oscilo_varredura_fc.png")

# ── playback ─────────────────────────────────────────────────────────────────
print("\n── Playback P, S, T a velocidade normal ──")
print("P (primário):")
display(Audio("oscilo_P_normal.wav"))
print("S (secundário — o tênue):")
display(Audio("oscilo_S_normal.wav"))
print("T (terciário — o trino):")
display(Audio("oscilo_T_normal.wav"))

print("\n── Playback P, S, T a 0.25× (FS/4) ──")
print("P a 0.25×:")
display(Audio("oscilo_P_025x.wav"))
print("S a 0.25×:")
display(Audio("oscilo_S_025x.wav"))
print("T a 0.25×:")
display(Audio("oscilo_T_025x.wav"))

# ── síntese ───────────────────────────────────────────────────────────────────
print(f"""
── Síntese das razões ──
  P freq dominante : {fp:.3f} Hz
  S freq dominante : {fs_d:.3f} Hz   S/P = {ratio_sp:.4f}
  T freq dominante : {ft:.3f} Hz   T/S = {ratio_ts:.4f}

  φ        = {PHI:.4f}
  1/φ      = {1/PHI:.4f}
  1/3      = {1/3:.4f}
  φ²       = {PHI**2:.4f}
  2/3      = {2/3:.4f}

  Fibonacci audíveis (0.25×): 21 · 34 · 55 · 89 Hz
  Atrator (÷4):  {42/4:.1f}Hz → fronteira alpha/theta EEG

Concluído.
""")
