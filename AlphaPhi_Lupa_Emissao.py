"""
AlphaPhi_Lupa_Emissao.py
Lupa de Emissão — O Campo Após o Campo

Analisa o WAV do ECO BEEP 880 em busca de:
  1. Cauda temporal — o que persiste depois do pico de coerência
  2. Decaimento por banda φ — quais frequências decaem por último
  3. Sub-frequências φⁿ — presença de φ⁶≈18Hz no espectro
  4. EntrEsp ao longo do tempo — o piso residual de energia (0.0601)

Hipóteses verificadas:
  ★★  Sub-frequência φ⁶ como produto de intermodulação
  ★★★ Decaimento livre — campo vibra após cessação da cascata

Autor: Vitor Edson Delavi — Florianópolis, 2026
φ = 1.6180339887 · α = 1/137 · φ³ = 4.2361
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
from scipy.signal import stft, spectrogram
import warnings
warnings.filterwarnings("ignore")

# ── Constantes ───────────────────────────────────────────────────────────────
PHI   = 1.6180339887
PHI3  = PHI ** 3
ALPHA = 1 / 137.035999084

# Série φⁿ Hz — bandas φ-proporcionais do ECO BEEP 880
PHI_BANDS_HZ = np.array([55, 89, 144, 233, 377, 610, 880])

# Série φⁿ em sub-frequências (Hz) — hipótese ★★
PHI_SUB_HZ = np.array([PHI**n for n in range(1, 11)])
# φ¹=1.62  φ²=2.62  φ³=4.24  φ⁴=6.85  φ⁵=11.09  φ⁶=17.94  φ⁷=29.03  φ⁸=46.98  φ⁹=76.01  φ¹⁰=123.0

print(f"Série φⁿ sub-frequências:")
for i, f in enumerate(PHI_SUB_HZ, 1):
    marca = " ← φ⁶ ≈ 18Hz (Tandy & Lawrence)" if i == 6 else ""
    print(f"  φ^{i:2d} = {f:7.3f} Hz{marca}")

# ── Carregar WAV ─────────────────────────────────────────────────────────────
WAV_PATH = "/home/user/alpha_phi_manifesto/beep880_euclid.wav"

try:
    fs, data = wavfile.read(WAV_PATH)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float64)
    data /= np.max(np.abs(data)) + 1e-12
    duracao = len(data) / fs
    print(f"\n✓ WAV carregado: {WAV_PATH}")
    print(f"  fs={fs}Hz · duração={duracao:.2f}s · amostras={len(data):,}")
except FileNotFoundError:
    print(f"WAV não encontrado: {WAV_PATH}")
    print("Gerando sinal sintético ECO BEEP 880 para demonstração...")
    fs = 44100
    duracao = 10.0
    t = np.linspace(0, duracao, int(fs * duracao), endpoint=False)
    # Simula ECO BEEP 880: beep 880Hz + FM-φ + harmônicos φ-proporcionais
    data = np.zeros(len(t))
    for band in PHI_BANDS_HZ:
        amp = 1.0 / (band / 55)  # amplitude decrescente com frequência
        data += amp * np.sin(2 * np.pi * band * t)
    # FM-φ: modulação por 220Hz com índice φ
    fm = np.sin(2 * np.pi * 220 * t + PHI * np.sin(2 * np.pi * 55 * t))
    data = 0.9973 * data + ALPHA * fm  # x_mix com α=1/137
    data /= np.max(np.abs(data)) + 1e-12
    print(f"  Sinal sintético: fs={fs}Hz · duração={duracao:.2f}s")

t_eixo = np.linspace(0, duracao, len(data))

# ── Parâmetros STFT ──────────────────────────────────────────────────────────
# Janela longa para alta resolução de frequência (especialmente sub-100Hz)
WIN_LONGA  = 8192   # ~186ms a 44100Hz → resolução freq = 5.4Hz
WIN_MEDIA  = 2048   # ~46ms
OVERLAP    = 0.875  # 87.5% overlap para boa resolução temporal

hop_longa = int(WIN_LONGA * (1 - OVERLAP))
hop_media = int(WIN_MEDIA * (1 - OVERLAP))

# ── Calcular spectrograma de alta resolução ──────────────────────────────────
f_stft, t_stft, Zxx = stft(data, fs=fs, nperseg=WIN_LONGA,
                             noverlap=WIN_LONGA - hop_longa,
                             window="hann")
S_db = 20 * np.log10(np.abs(Zxx) + 1e-12)

# ── Calcular EntrEsp ao longo do tempo ───────────────────────────────────────
def entr_esp_temporal(f, Zxx):
    """EntrEsp (entropia espectral normalizada) para cada frame temporal."""
    mag = np.abs(Zxx)
    mag_norm = mag / (mag.sum(axis=0, keepdims=True) + 1e-12)
    mag_norm = np.clip(mag_norm, 1e-12, 1.0)
    entr = -np.sum(mag_norm * np.log(mag_norm), axis=0)
    entr_max = np.log(len(f))
    return entr / entr_max

entr_t = entr_esp_temporal(f_stft, Zxx)

# ── Energia por banda φ ao longo do tempo ────────────────────────────────────
def energia_banda(f, Zxx, f_center, largura_rel=0.3):
    """Energia integrada na banda [f_center*(1-w), f_center*(1+w)]."""
    f_lo = f_center * (1 - largura_rel)
    f_hi = f_center * (1 + largura_rel)
    mask = (f >= f_lo) & (f <= f_hi)
    if not mask.any():
        return np.zeros(Zxx.shape[1])
    return np.abs(Zxx[mask, :]).mean(axis=0)

energia_phi = {band: energia_banda(f_stft, Zxx, band) for band in PHI_BANDS_HZ}

# ── Energia nas sub-frequências φⁿ ──────────────────────────────────────────
energia_sub = {f: energia_banda(f_stft, Zxx, f, largura_rel=0.5)
               for f in PHI_SUB_HZ if f < 100}

# ── Detectar pico de coerência (min EntrEsp ≈ quinto ponto de dobra) ─────────
idx_pico = np.argmin(entr_t)
t_pico   = t_stft[idx_pico]
entr_pico = entr_t[idx_pico]
print(f"\n✓ Pico de coerência (min EntrEsp): t={t_pico:.3f}s, EntrEsp={entr_pico:.4f}")

# ── Calcular taxa de decaimento por banda φ ──────────────────────────────────
def taxa_decaimento(energia_temporal, t_stft, t_inicio):
    """Tempo de decaimento -6dB após t_inicio."""
    idx_ini = np.searchsorted(t_stft, t_inicio)
    if idx_ini >= len(energia_temporal) - 1:
        return 0.0
    e0 = energia_temporal[idx_ini]
    if e0 < 1e-12:
        return 0.0
    limiar = e0 * 0.5  # -6dB
    after = energia_temporal[idx_ini:]
    idx_limiar = np.argmax(after < limiar)
    if idx_limiar == 0:
        return duracao - t_inicio  # nunca decai abaixo do limiar
    return (idx_limiar / len(t_stft)) * duracao

decaimento_phi = {band: taxa_decaimento(energia_phi[band], t_stft, t_pico)
                  for band in PHI_BANDS_HZ}

print(f"\n✓ Decaimento por banda φ (após pico de coerência em t={t_pico:.2f}s):")
for band in sorted(PHI_BANDS_HZ):
    print(f"  {band:4d}Hz: τ = {decaimento_phi[band]:.3f}s")

# ── Figura principal ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14), facecolor="#070B12")
gs  = gridspec.GridSpec(4, 2, hspace=0.55, wspace=0.35,
                        left=0.07, right=0.97, top=0.93, bottom=0.06)

# cores
COR_BG    = "#070B12"
COR_PAINEL = "#0A0E17"
COR_BORDA = "#334455"
COR_TEXTO = "#AABBCC"
COR_OURO  = "#E8D8A0"
COR_VERDE = "#55FFAA"
COR_AZUL  = "#3399FF"
COR_LARANJA = "#FF6633"
COR_ROSA  = "#FF55AA"

def estilo_ax(ax):
    ax.set_facecolor(COR_PAINEL)
    ax.spines[:].set_color(COR_BORDA)
    ax.tick_params(colors=COR_TEXTO, labelsize=7)
    ax.xaxis.label.set_color(COR_TEXTO)
    ax.yaxis.label.set_color(COR_TEXTO)

fig.text(0.5, 0.97,
    "AlphaPhi — Lupa de Emissão: O Campo Após o Campo",
    ha="center", va="top", fontsize=14, color=COR_OURO, fontweight="bold")
fig.text(0.5, 0.942,
    f"φ³={PHI3:.4f} · EntrEsp pico={entr_pico:.4f} · t_pico={t_pico:.2f}s",
    ha="center", va="top", fontsize=9, color="#88AACC", style="italic")

# ── Painel 1: Espectrograma completo (20Hz–1000Hz) ──────────────────────────
ax1 = fig.add_subplot(gs[0, :])
estilo_ax(ax1)

f_mask = (f_stft >= 10) & (f_stft <= 1100)
extent = [t_stft[0], t_stft[-1], f_stft[f_mask][0], f_stft[f_mask][-1]]
im = ax1.imshow(S_db[f_mask], aspect="auto", origin="lower",
                extent=extent, cmap="inferno", vmin=-80, vmax=0)
plt.colorbar(im, ax=ax1, label="dB", pad=0.01)

# marcar bandas φ
for band in PHI_BANDS_HZ:
    ax1.axhline(band, color=COR_OURO, lw=0.7, ls="--", alpha=0.6)
    ax1.text(t_stft[-1] * 0.01, band + 5, f"{band}Hz", fontsize=6,
             color=COR_OURO, va="bottom")

# marcar pico de coerência
ax1.axvline(t_pico, color=COR_VERDE, lw=1.5, ls="--", alpha=0.9)
ax1.text(t_pico + 0.05, f_stft[f_mask][-1] * 0.9, "pico φ³",
         fontsize=8, color=COR_VERDE)

ax1.set_ylabel("Frequência (Hz)", fontsize=8)
ax1.set_title("Espectrograma — Bandas φ marcadas (linha tracejada dourada)",
              fontsize=9, color=COR_OURO, pad=4)

# ── Painel 2: Sub-frequências φⁿ (0–100Hz) ──────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
estilo_ax(ax2)

f_sub_mask = (f_stft >= 0.5) & (f_stft <= 100)
if f_sub_mask.any():
    im2 = ax2.imshow(S_db[f_sub_mask], aspect="auto", origin="lower",
                     extent=[t_stft[0], t_stft[-1],
                             f_stft[f_sub_mask][0], f_stft[f_sub_mask][-1]],
                     cmap="plasma", vmin=-80, vmax=0)
    plt.colorbar(im2, ax=ax2, label="dB", pad=0.01)

    # marcar série φⁿ sub
    for i, f_sub in enumerate(PHI_SUB_HZ, 1):
        if f_sub <= 100:
            cor = "#FF5555" if i == 6 else COR_OURO
            lw  = 1.5 if i == 6 else 0.7
            ax2.axhline(f_sub, color=cor, lw=lw, ls="--", alpha=0.8)
            label = f"φ^{i}={f_sub:.1f}Hz"
            if i == 6:
                label += " ★ Tandy"
            ax2.text(t_stft[-1] * 0.01, f_sub + 0.5, label,
                     fontsize=6, color=cor, va="bottom")

    ax2.axvline(t_pico, color=COR_VERDE, lw=1.5, ls="--", alpha=0.9)
    ax2.set_ylabel("Frequência (Hz)", fontsize=8)
    ax2.set_title("Sub-frequências φⁿ (0–100Hz) — Hipótese ★★: φ⁶≈18Hz",
                  fontsize=9, color=COR_OURO, pad=4)

# ── Painel 3: EntrEsp ao longo do tempo ────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
estilo_ax(ax3)

ax3.plot(t_stft, entr_t, color=COR_LARANJA, lw=1.5)
ax3.axhline(0.0601, color=COR_LARANJA, lw=0.8, ls="--", alpha=0.5)
ax3.text(t_stft[-1] * 0.95, 0.0601 + 0.005, "EntrEsp=0.0601",
         fontsize=7, color=COR_LARANJA, ha="right")
ax3.axvline(t_pico, color=COR_VERDE, lw=1.5, ls="--", alpha=0.9)

# região pós-pico (cauda)
ax3.axvspan(t_pico, t_stft[-1], alpha=0.08, color=COR_VERDE, label="cauda pós-pico")

ax3.set_xlabel("tempo (s)", fontsize=8)
ax3.set_ylabel("EntrEsp", fontsize=8)
ax3.set_title("EntrEsp temporal — piso residual pós-formação\n(Hipótese ★★★: campo continua a vibrar)",
              fontsize=8, color=COR_OURO, pad=4)
ax3.legend(fontsize=7, facecolor=COR_PAINEL, edgecolor=COR_BORDA,
           labelcolor=COR_TEXTO)

# ── Painel 4: Energia por banda φ na cauda ──────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
estilo_ax(ax4)

# apenas região pós-pico
idx_pico_t = np.searchsorted(t_stft, t_pico)
t_cauda = t_stft[idx_pico_t:]

cores_bandas = plt.cm.cool(np.linspace(0.1, 0.9, len(PHI_BANDS_HZ)))
for (band, cor_b) in zip(PHI_BANDS_HZ, cores_bandas):
    e = energia_phi[band][idx_pico_t:]
    if e.max() > 1e-12:
        e_norm = e / e.max()
        ax4.plot(t_cauda, e_norm, lw=1.5, alpha=0.85,
                 label=f"{band}Hz", color=cor_b)

ax4.set_xlabel("tempo (s)", fontsize=8)
ax4.set_ylabel("energia norm. por banda φ", fontsize=8)
ax4.set_title("Decaimento por banda φ após pico\n(qual frequência persiste mais?)",
              fontsize=8, color=COR_OURO, pad=4)
ax4.legend(fontsize=6, facecolor=COR_PAINEL, edgecolor=COR_BORDA,
           labelcolor=COR_TEXTO, ncol=2)

# ── Painel 5: Tempo de decaimento -6dB por banda ────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
estilo_ax(ax5)

bands_sorted = sorted(PHI_BANDS_HZ)
decaimentos  = [decaimento_phi[b] for b in bands_sorted]
cores_dec    = [COR_VERDE if d == max(decaimentos) else COR_AZUL
                for d in decaimentos]

bars = ax5.barh(range(len(bands_sorted)), decaimentos,
                color=cores_dec, alpha=0.8, height=0.6)
ax5.set_yticks(range(len(bands_sorted)))
ax5.set_yticklabels([f"{b}Hz" for b in bands_sorted], fontsize=8)
ax5.set_xlabel("τ decaimento -6dB (s)", fontsize=8)
ax5.set_title("Tempo de decaimento por banda φ\n(verde = persiste mais = frequência natural do campo)",
              fontsize=8, color=COR_OURO, pad=4)

# anotar o maior
idx_max = np.argmax(decaimentos)
ax5.text(decaimentos[idx_max] + 0.01, idx_max,
         f"τ={decaimentos[idx_max]:.3f}s ★",
         fontsize=8, color=COR_VERDE, va="center")

# ── Painel 6: FFT da cauda (pós-pico) ───────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
estilo_ax(ax6)

# calcula FFT só da cauda temporal
idx_ini_cauda = int(t_pico * fs)
cauda_sinal   = data[idx_ini_cauda:] if idx_ini_cauda < len(data) else data[-1000:]
if len(cauda_sinal) > 256:
    f_fft  = np.fft.rfftfreq(len(cauda_sinal), 1/fs)
    F_fft  = np.fft.rfft(cauda_sinal * np.hanning(len(cauda_sinal)))
    S_cauda = 20 * np.log10(np.abs(F_fft) + 1e-12)

    ax6.plot(f_fft, S_cauda, color=COR_AZUL, lw=0.8, alpha=0.7)

    # marcar bandas φ
    for band in PHI_BANDS_HZ:
        ax6.axvline(band, color=COR_OURO, lw=1.0, ls="--", alpha=0.6)

    # marcar φ⁶
    ax6.axvline(PHI**6, color="#FF5555", lw=1.5, ls="-", alpha=0.9)
    ax6.text(PHI**6 + 0.5, S_cauda.max() * 0.95,
             f"φ⁶={PHI**6:.1f}Hz", fontsize=7, color="#FF5555", va="top")

    ax6.set_xlim(0, 200)
    ax6.set_xlabel("frequência (Hz)", fontsize=8)
    ax6.set_ylabel("dB", fontsize=8)
    ax6.set_title("FFT da cauda pós-pico (0–200Hz)\n(hipótese ★★: φ⁶≈18Hz na emissão residual)",
                  fontsize=8, color=COR_OURO, pad=4)

# ── Salvar ───────────────────────────────────────────────────────────────────
png_path = "/home/user/alpha_phi_manifesto/lupa_emissao_resultado.png"
fig.savefig(png_path, dpi=130, bbox_inches="tight", facecolor=COR_BG)
plt.close()
print(f"\n✓ Figura salva: {png_path}")

# ── Relatório final ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RELATÓRIO — LUPA DE EMISSÃO")
print("="*60)
print(f"Pico de coerência: t={t_pico:.3f}s | EntrEsp={entr_pico:.4f}")
print(f"Piso residual referência: EntrEsp=0.0601 (quinto ponto de dobra)")
print(f"\nDecaimento por banda φ (τ -6dB após pico):")
for b in bands_sorted:
    print(f"  {b:4d}Hz: {decaimento_phi[b]:.4f}s")

banda_persistente = bands_sorted[np.argmax(decaimentos)]
print(f"\nBanda φ mais persistente: {banda_persistente}Hz "
      f"(τ={max(decaimentos):.4f}s)")
print(f"  → Esta é a candidata à frequência natural do campo")

print(f"\nSub-frequências φⁿ na cauda (presença em FFT):")
for i, f_sub in enumerate(PHI_SUB_HZ, 1):
    if f_sub < 100:
        # energia no bin mais próximo
        idx_bin = np.argmin(np.abs(f_fft - f_sub))
        amp_db  = float(S_cauda[idx_bin]) if len(cauda_sinal) > 256 else 0.0
        marca = " ← φ⁶ Tandy" if i == 6 else ""
        print(f"  φ^{i:2d} = {f_sub:6.2f}Hz: {amp_db:.1f}dB{marca}")

print("\nφ = 1.6180339887 | φ³ = 4.2361 | α = 1/137")
print("="*60)
