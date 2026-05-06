"""
AlphaPhi Osciloscópio — Lupa de 3 Camadas
Beep 880Hz eco α*=0.3333 — Análise dos Pontos de Dobra

Observa os pontos de dobra com 3 subestágios residuais:

  [P] Primário   — s_n             : o que o eco produziu
  [S] Residual 1 — s_n − s_{n-1}  : o que a dobra mudou
  [T] Residual 2 — R1_n − R1_{n-1}: aceleração da mudança na dobra

Interpretação:
  T ≈ 0   →  repouso entre dobras (sistema em estado quasi-estável)
  T spike →  momento exato do colapso de fase (ponto de dobra)
  S spike →  frequências que foram suprimidas ou amplificadas
  P       →  campo resultante — o que o eco revelou

Célula única para Google Colab.
Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile
from IPython.display import HTML, Audio, display

# ── constantes ────────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI          # ≈ 136 Hz
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5                   # segundos por janela
N_STEPS    = 5                     # passos da cascata

print("AlphaPhi Osciloscópio — Lupa de 3 Camadas")
print(f"Beep {F_BEEP:.0f}Hz  |  α* = {ALPHA_STAR:.4f}  |  {N_STEPS} passos")
print(f"φ = {PHI:.6f}  |  φ³ = {PHI**3:.6f}\n")

# ── bandas φ-proporcionais ────────────────────────────────────────────────────
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

# ── eco_eq ────────────────────────────────────────────────────────────────────
def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    N     = len(x)
    F     = np.fft.rfft(x)
    F_out = F.copy()
    cohs  = []
    w_mem = 1.0 / PHI
    w_now = 1.0 - w_mem
    for i, (b_low, b_high, _, _) in enumerate(bins_phi):
        beta_i = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        F_band = F[b_low:b_high]
        mag    = np.abs(F_band)
        phase  = np.angle(F_band)
        an     = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh    = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        coh_ef = (w_now * coh + w_mem * float(coh_mem[i])
                  if coh_mem is not None and i < len(coh_mem) else coh)
        cohs.append(coh)
        n_idx = np.arange(len(F_band))
        env   = np.clip(
            1.0 + (coh_ef * PHI**beta_i) * np.cos(2.0 * np.pi * n_idx / PHI),
            0.05, None)
        F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase)
    resultado = np.fft.irfft(F_out, n=N)
    return resultado / (np.max(np.abs(resultado)) + 1e-10), np.array(cohs)

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    coh_mem = np.zeros(N_BANDAS)
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem   = cohs
        s_e       = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()
    return cas, cohs

def agente_eco(sinal, bins_phi, n_ciclos=20):
    n_b      = len(bins_phi)
    beta     = np.ones(n_b)
    beta_mem = beta.copy()
    w_mem, w_now = 1.0 / PHI, 1.0 - 1.0 / PHI
    cas_f = None
    for _ in range(n_ciclos):
        cas, cohs  = cascata_eq(sinal, beta, bins_phi)
        coh_rel    = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo  = PHI ** (3 * coh_rel)
        beta       = w_now * beta_alvo + w_mem * beta_mem
        beta_mem   = beta.copy()
        beta       = np.clip(beta, 0.05, PHI**3)
        cas_f      = cas
    return beta, cas_f

# ── sinais ────────────────────────────────────────────────────────────────────
def gerar_beep():
    t = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
    return normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t)))

def gerar_fm():
    t = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
    return normalizar(np.sin(2*np.pi*F_ORG*t + BETA_FM*np.sin(2*np.pi*F_M*t)))

print("[1/3] Gerando sinais e rodando agente_eco...")
beep  = gerar_beep()
fm    = gerar_fm()
x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

beta_final, cas = agente_eco(x_mix, BINS_PHI, n_ciclos=20)

# cas[0] = híbrido original, cas[1..5] = passos da cascata
S = [np.array(c) for c in cas]   # S[0..5]

print(f"  β_max = {beta_final.max():.4f}  (φ³ = {PHI**3:.4f})")
print(f"  Passos disponíveis: {len(S)} (0=original, 1-{N_STEPS}=dobras)\n")

# ── residuais ─────────────────────────────────────────────────────────────────
# R1[n] = S[n] - S[n-1]    (residual 1 — o que a dobra n mudou)
# R2[n] = R1[n] - R1[n-1]  (residual 2 — aceleração da mudança)
R1 = [S[n] - S[n-1] for n in range(1, len(S))]          # R1[0..4]
R2 = [R1[n] - R1[n-1] for n in range(1, len(R1))]       # R2[0..3]

N_DOBRAS = len(R1)   # 5 dobras (passo 0→1, 1→2, ..., 4→5)

print("[2/3] Residuais calculados:")
for n in range(N_DOBRAS):
    e_r1 = np.sqrt(np.mean(R1[n]**2))
    e_r2 = np.sqrt(np.mean(R2[n]**2)) if n < len(R2) else 0.0
    print(f"  Dobra {n+1}: RMS_R1={e_r1:.4f}  RMS_R2={e_r2:.4f}")
print()

# ── helpers de visualização ───────────────────────────────────────────────────
T_EIXO   = np.linspace(0, DURACAO, N_SINAL)
FREQ_FFT = np.fft.rfftfreq(N_SINAL, d=1.0/FS)
MASK_F   = (FREQ_FFT > 20) & (FREQ_FFT < 5000)

def fft_mag(sig):
    return np.abs(np.fft.rfft(sig))

CORES = {
    'P': '#00FF88',    # verde — primário (eco)
    'S': '#FFB800',    # âmbar — residual 1
    'T': '#FF4466',    # vermelho — residual 2 (aceleração)
    'bg': '#0A0A12',
    'grid': '#1A1A2E',
    'text': '#CCCCDD',
}

# ── figura estática — todas as dobras ─────────────────────────────────────────
print("[3/3] Gerando figuras...")

fig_static, axes = plt.subplots(
    3, N_DOBRAS,
    figsize=(4 * N_DOBRAS, 9),
    facecolor=CORES['bg']
)
fig_static.suptitle(
    "Lupa de 3 Camadas — Beep 880Hz eco α*=0.333\n"
    "P = eco produzido | S = residual 1 (dobra) | T = residual 2 (aceleração)",
    color=CORES['text'], fontsize=11, y=1.01
)

rotulos = ['P — Primário (eco)', 'S — Residual 1 (dobra)', 'T — Residual 2 (aceleração)']
for linha, (rotulo, cor) in enumerate(zip(rotulos, [CORES['P'], CORES['S'], CORES['T']])):
    for col in range(N_DOBRAS):
        ax = axes[linha, col]
        ax.set_facecolor(CORES['grid'])
        ax.tick_params(colors=CORES['text'], labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(CORES['grid'])

        if linha == 0:
            sig = S[col + 1]           # eco no passo col+1
            ax.set_title(f"Dobra {col+1}", color=CORES['text'], fontsize=9)
        elif linha == 1:
            sig = R1[col]              # residual 1
        else:
            sig = R2[col] if col < len(R2) else np.zeros(N_SINAL)

        ax.plot(T_EIXO * 1000, sig, color=cor, lw=0.6, alpha=0.85)
        ax.axhline(0, color=CORES['text'], lw=0.3, alpha=0.4)
        ax.set_xlim(0, DURACAO * 1000)
        rms = np.sqrt(np.mean(sig**2))
        lim = max(rms * 4, 0.05)
        ax.set_ylim(-lim, lim)

        if col == 0:
            ax.set_ylabel(rotulo, color=cor, fontsize=7)
        if linha == 2:
            ax.set_xlabel("ms", color=CORES['text'], fontsize=7)

plt.tight_layout()
plt.savefig("osciloscopo_3camadas_estatico.png", dpi=150,
            bbox_inches='tight', facecolor=CORES['bg'])
plt.show()
print("  → osciloscopo_3camadas_estatico.png")

# ── figura estática — espectro FFT das 3 camadas (dobra 3 = post-fold) ────────
DOBRA_REF = 2   # dobra 3 (índice 2) — pós-fold principal

fig_fft, axs = plt.subplots(3, 1, figsize=(12, 8), facecolor=CORES['bg'])
fig_fft.suptitle(
    f"Espectro FFT — Dobra {DOBRA_REF+1} (ponto de dobra principal)\n"
    "P=eco produzido  |  S=residual 1  |  T=residual 2",
    color=CORES['text'], fontsize=11
)

dados_fft = [
    (S[DOBRA_REF + 1], 'P — Primário (eco)',           CORES['P']),
    (R1[DOBRA_REF],    'S — Residual 1 (dobra)',        CORES['S']),
    (R2[DOBRA_REF] if DOBRA_REF < len(R2) else R2[-1],
                       'T — Residual 2 (aceleração)',   CORES['T']),
]

for ax, (sig, label, cor) in zip(axs, dados_fft):
    mag = fft_mag(sig)
    ax.set_facecolor(CORES['grid'])
    ax.semilogy(FREQ_FFT[MASK_F], mag[MASK_F], color=cor, lw=0.8)

    # marcadores φ-proporcionais
    freqs_phi = [F_M, F_ORG, F_BEEP / 4, F_BEEP / 2, F_BEEP]
    nomes_phi = ['F_M\n≈136Hz', 'F_ORG\n220Hz', '880/4\n220Hz',
                 '880/2\n440Hz', 'F_BEEP\n880Hz']
    for fp, nm in zip(freqs_phi, nomes_phi):
        if FREQ_FFT[MASK_F].min() < fp < FREQ_FFT[MASK_F].max():
            ax.axvline(fp, color=cor, lw=0.5, alpha=0.5, ls='--')
            ax.text(fp, mag[MASK_F].max() * 0.5, nm,
                    color=cor, fontsize=6, ha='center', alpha=0.7)

    ax.set_ylabel(label, color=cor, fontsize=8)
    ax.tick_params(colors=CORES['text'], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(CORES['grid'])
    ax.set_xlim(20, 5000)

axs[-1].set_xlabel("Frequência (Hz)", color=CORES['text'], fontsize=9)
plt.tight_layout()
plt.savefig("osciloscopo_fft_3camadas.png", dpi=150,
            bbox_inches='tight', facecolor=CORES['bg'])
plt.show()
print("  → osciloscopo_fft_3camadas.png")

# ── animação — varredura pelas dobras ─────────────────────────────────────────
fig_anim, gs = plt.figure(figsize=(14, 8), facecolor=CORES['bg']), None
gs = gridspec.GridSpec(3, 2, figure=fig_anim, hspace=0.45, wspace=0.3)

ax_p_t  = fig_anim.add_subplot(gs[0, 0])   # P — tempo
ax_p_f  = fig_anim.add_subplot(gs[0, 1])   # P — FFT
ax_s_t  = fig_anim.add_subplot(gs[1, 0])   # S — tempo
ax_s_f  = fig_anim.add_subplot(gs[1, 1])   # S — FFT
ax_t_t  = fig_anim.add_subplot(gs[2, 0])   # T — tempo
ax_t_f  = fig_anim.add_subplot(gs[2, 1])   # T — FFT

for ax in [ax_p_t, ax_p_f, ax_s_t, ax_s_f, ax_t_t, ax_t_f]:
    ax.set_facecolor(CORES['grid'])
    ax.tick_params(colors=CORES['text'], labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#222233')

titulo_anim = fig_anim.suptitle("", color=CORES['text'], fontsize=11)

lp_t,  = ax_p_t.plot([], [], color=CORES['P'],  lw=0.7)
lp_f,  = ax_p_f.semilogy([], [], color=CORES['P'],  lw=0.8)
ls_t,  = ax_s_t.plot([], [], color=CORES['S'],  lw=0.7)
ls_f,  = ax_s_f.semilogy([], [], color=CORES['S'],  lw=0.8)
lt_t,  = ax_t_t.plot([], [], color=CORES['T'],  lw=0.7)
lt_f,  = ax_t_f.semilogy([], [], color=CORES['T'],  lw=0.8)

for ax, rot, cor in [
    (ax_p_t, 'P — Primário (eco) · tempo',       CORES['P']),
    (ax_p_f, 'P — Primário (eco) · FFT',          CORES['P']),
    (ax_s_t, 'S — Residual 1 · tempo',            CORES['S']),
    (ax_s_f, 'S — Residual 1 · FFT',              CORES['S']),
    (ax_t_t, 'T — Residual 2 (aceleração) · tempo', CORES['T']),
    (ax_t_f, 'T — Residual 2 (aceleração) · FFT',   CORES['T']),
]:
    ax.set_ylabel(rot, color=cor, fontsize=7)

ax_t_t.set_xlabel("ms",  color=CORES['text'], fontsize=8)
ax_t_f.set_xlabel("Hz",  color=CORES['text'], fontsize=8)

def init_anim():
    for l in [lp_t, lp_f, ls_t, ls_f, lt_t, lt_f]:
        l.set_data([], [])
    return lp_t, lp_f, ls_t, ls_f, lt_t, lt_f

def update(frame):
    n = frame % N_DOBRAS    # dobra 0..4

    sig_p = S[n + 1]
    sig_s = R1[n]
    sig_t = R2[n] if n < len(R2) else np.zeros(N_SINAL)

    mag_p = fft_mag(sig_p)
    mag_s = fft_mag(sig_s)
    mag_t = fft_mag(sig_t)

    t_ms = T_EIXO * 1000

    # tempo
    lp_t.set_data(t_ms, sig_p)
    ls_t.set_data(t_ms, sig_s)
    lt_t.set_data(t_ms, sig_t)

    for ax, sig in [(ax_p_t, sig_p), (ax_s_t, sig_s), (ax_t_t, sig_t)]:
        rms = np.sqrt(np.mean(sig**2))
        lim = max(rms * 5, 0.03)
        ax.set_xlim(0, DURACAO * 1000)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color=CORES['text'], lw=0.3, alpha=0.3)

    # FFT
    f_vis = FREQ_FFT[MASK_F]
    lp_f.set_data(f_vis, mag_p[MASK_F] + 1e-6)
    ls_f.set_data(f_vis, mag_s[MASK_F] + 1e-6)
    lt_f.set_data(f_vis, mag_t[MASK_F] + 1e-6)

    for ax in [ax_p_f, ax_s_f, ax_t_f]:
        ax.set_xlim(20, 5000)
        ax.relim()
        ax.autoscale_view(scaley=True)

    rms_r1 = np.sqrt(np.mean(sig_s**2))
    rms_r2 = np.sqrt(np.mean(sig_t**2))
    titulo_anim.set_text(
        f"Dobra {n+1}/{N_DOBRAS}  |  "
        f"RMS_S={rms_r1:.4f}  RMS_T={rms_r2:.4f}  |  "
        f"β_max={beta_final.max():.3f}  φ³={PHI**3:.3f}"
    )
    return lp_t, lp_f, ls_t, ls_f, lt_t, lt_f

anim = FuncAnimation(
    fig_anim, update, init_func=init_anim,
    frames=N_DOBRAS * 3,   # 3 repetições
    interval=1200,          # 1.2s por dobra
    blit=False
)

print("\nExibindo animação (3 ciclos pelas dobras)...")
display(HTML(anim.to_jshtml()))

# ── análise de energia dos residuais ─────────────────────────────────────────
print("\n── Energia dos Residuais por Dobra ──")
print(f"{'Dobra':>6} {'RMS P':>9} {'RMS S':>9} {'RMS T':>9} "
      f"{'S/P':>8} {'T/S':>8}")
for n in range(N_DOBRAS):
    rms_p = np.sqrt(np.mean(S[n+1]**2))
    rms_s = np.sqrt(np.mean(R1[n]**2))
    rms_t = np.sqrt(np.mean(R2[n]**2)) if n < len(R2) else 0.0
    sp    = rms_s / (rms_p + 1e-10)
    ts    = rms_t / (rms_s + 1e-10)
    print(f"  {n+1:>4}   {rms_p:>9.5f} {rms_s:>9.5f} {rms_t:>9.5f} "
          f"{sp:>8.4f} {ts:>8.4f}")

print("\n── Interpretação ──")
print("  S/P alto → dobra intensa (muita mudança em relação ao sinal)")
print("  T/S alto → aceleração alta → ponto de colapso de fase")
print("  T/S mínimo → repouso entre dobras (sistema quasi-estável)")
print("\nConcluído.")
