# © Vitor Edson Delavi · Florianópolis · 2026
# Análise comparativa — Original vs Campo Harmônico φ

import urllib.request
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from IPython.display import Audio, display

BASE = "https://raw.githubusercontent.com/vitoredsonalphaphi/alpha_phi_manifesto/claude/good-morning-N6f3S/"

print("Baixando arquivos...")
urllib.request.urlretrieve(BASE + "violao_original.wav",  "original.wav")
urllib.request.urlretrieve(BASE + "violao_campo_phi.wav", "campo_phi.wav")

rate, d1 = wavfile.read("original.wav")
_,    d2 = wavfile.read("campo_phi.wav")

s1 = d1.astype(np.float64) / 32768.0
s2 = d2.astype(np.float64) / 32768.0
PHI = (1 + np.sqrt(5)) / 2

# ── Players ──────────────────────────────────────────────────────────────
print("\n▶ ORIGINAL")
display(Audio(d1, rate=rate))
print("▶ CAMPO HARMÔNICO φ")
display(Audio(d2, rate=rate))

# ── Métricas ─────────────────────────────────────────────────────────────
def bits_efetivos(s, bins=256):
    c, _ = np.histogram(s, bins=bins)
    c = c[c > 0]; p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))

def entropia_espectral(s, fs):
    f, psd = signal.welch(s, fs, nperseg=4096)
    psd = psd[psd > 0]; p = psd / psd.sum()
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))

def autocorr(s):
    s = s - s.mean()
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])

print("\n" + "=" * 50)
print(f"{'Métrica':<28} {'Original':>10} {'Campo φ':>10}")
print("=" * 50)
metricas = [
    ("Bits efetivos",       bits_efetivos(s1),       bits_efetivos(s2)),
    ("Entropia espectral",  entropia_espectral(s1,rate), entropia_espectral(s2,rate)),
    ("Autocorrelação",      autocorr(s1),             autocorr(s2)),
    ("RMS (energia)",       float(np.sqrt(np.mean(s1**2))), float(np.sqrt(np.mean(s2**2)))),
    ("Pico absoluto",       float(np.max(np.abs(s1))), float(np.max(np.abs(s2)))),
]
for nome, v1, v2 in metricas:
    delta = "↑" if v2 > v1 else "↓"
    print(f"{nome:<28} {v1:>10.4f} {v2:>10.4f} {delta}")

# ── Gráficos ─────────────────────────────────────────────────────────────
t = np.linspace(0, len(s1)/rate, len(s1))
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Forma de onda
axes[0,0].plot(t, s1, color='gray',    lw=0.4, alpha=0.8)
axes[0,0].set_title("Forma de onda — Original"); axes[0,0].set_xlabel("s")
axes[0,1].plot(t[:len(s2)], s2, color='#00cc66', lw=0.4, alpha=0.8)
axes[0,1].set_title(f"Forma de onda — Campo φ  (β→φ³={PHI**3:.4f})")

# Espectrograma
for ax, sig, titulo, cor in [
    (axes[1,0], s1, "Espectrograma — Original",    'gray'),
    (axes[1,1], s2, "Espectrograma — Campo φ",     'plasma'),
]:
    f_spec, t_spec, Sxx = signal.spectrogram(sig, rate, nperseg=1024)
    ax.pcolormesh(t_spec, f_spec[:200], 10*np.log10(Sxx[:200]+1e-10),
                  shading='gouraud', cmap=cor)
    ax.set_title(titulo); ax.set_ylabel("Hz"); ax.set_xlabel("s")
    ax.set_ylim(0, 4000)

# Espectro de frequências
N = min(len(s1), len(s2))
freqs = np.fft.rfftfreq(N, 1/rate)
sp1 = np.abs(np.fft.rfft(s1[:N]))
sp2 = np.abs(np.fft.rfft(s2[:N]))
mask = freqs < 8000
axes[2,0].semilogy(freqs[mask], sp1[mask], color='gray',    lw=0.8, alpha=0.9, label='Original')
axes[2,0].semilogy(freqs[mask], sp2[mask], color='#00cc66', lw=0.8, alpha=0.9, label='Campo φ')
axes[2,0].set_title("Espectro comparado (0–8kHz)")
axes[2,0].set_xlabel("Hz"); axes[2,0].legend()

# Diferença espectral
diff = sp2[mask] - sp1[mask]
axes[2,1].fill_between(freqs[mask], diff, 0,
    where=diff>=0, color='#00cc66', alpha=0.6, label='Campo φ > Original')
axes[2,1].fill_between(freqs[mask], diff, 0,
    where=diff<0,  color='#ff4444', alpha=0.6, label='Original > Campo φ')
axes[2,1].axhline(0, color='white', lw=0.5)
axes[2,1].set_title("Diferença espectral (Campo φ − Original)")
axes[2,1].set_xlabel("Hz"); axes[2,1].legend()

plt.tight_layout()
plt.savefig("analise_comparativa.png", dpi=120, bbox_inches='tight')
plt.show()
print("\nGráfico salvo: analise_comparativa.png")
