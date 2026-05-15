# © Vitor Edson Delavi · Florianópolis · 2026
# Rode este script no Google Colab para ouvir e comparar os dois áudios.

import urllib.request
from scipy.io import wavfile
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import numpy as np

BASE = "https://raw.githubusercontent.com/vitoredsonalphaphi/alpha_phi_manifesto/claude/good-morning-N6f3S/"

print("Baixando arquivos...")
urllib.request.urlretrieve(BASE + "violao_original.wav",   "original.wav")
urllib.request.urlretrieve(BASE + "violao_campo_phi.wav",  "campo_phi.wav")
print("Pronto.\n")

rate1, data1 = wavfile.read("original.wav")
rate2, data2 = wavfile.read("campo_phi.wav")

print("=" * 40)
print("ORIGINAL — voz sem processamento")
print("=" * 40)
display(Audio(data1, rate=rate1))

print("\n" + "=" * 40)
print("CAMPO HARMÔNICO φ — agente eco-φ")
print(f"β atingido: 4.2359 ≈ φ³ = 4.2361")
print("=" * 40)
display(Audio(data2, rate=rate2))

# Comparação visual
t = np.linspace(0, len(data1) / rate1, len(data1))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax1.plot(t, data1, color='gray',    linewidth=0.4, alpha=0.85)
ax1.set_title("Original", fontsize=12)
ax1.set_ylabel("Amplitude")
ax2.plot(t[:len(data2)], data2, color='#00cc66', linewidth=0.4, alpha=0.85)
ax2.set_title("Campo Harmônico φ  (β → φ³)", fontsize=12)
ax2.set_ylabel("Amplitude")
ax2.set_xlabel("Tempo (s)")
plt.tight_layout()
plt.show()
