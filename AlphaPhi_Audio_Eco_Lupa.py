"""
AlphaPhi Audio Eco — Lupa v3
Célula única para Google Colab.

Correção: eco agora modifica AMPLITUDES (via envelope φ-quasiperiódico).
Versão anterior rotacionava só fase — inaudível ao ouvido humano.

Quatro sinais:
  1. Original
  2. Eco direto (φ-envelope aplicado ao espectro de amplitudes)
  3. Lupa ×10
  4. Microscópio ×20 (só a diferença)
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

PHI         = (1 + np.sqrt(5)) / 2
K_MIN       = np.sqrt(2)
FS          = 44100
DURACAO     = 2.0
GANHO_LUPA  = 10.0
GANHO_MICRO = 20.0

def medir_campo(x):
    f = np.fft.fft(x)
    a = np.abs(f)
    a = np.clip(a / (a.sum() + 1e-8), 1e-10, 1.0)
    e = -np.sum(a * np.log(a))
    c = float(1.0 - e / np.log(len(x)))
    return K_MIN + (PHI - K_MIN) * c, c

def eco_ressonante(x):
    k, coh = medir_campo(x)
    F     = np.fft.rfft(x)
    mag   = np.abs(F)
    phase = np.angle(F)
    n_idx = np.arange(len(F))

    # envelope quasiperiódico de φ: cos(2π·n/φ) nunca se repete (φ irracional)
    # redistribui amplitudes — uns harmônicos sobem, outros descem → timbre muda
    phi_env = 1.0 + coh * np.cos(2.0 * np.pi * n_idx / PHI)
    phi_env = np.clip(phi_env, 0.05, None)

    # aplica ao espectro de amplitudes + rotação de fase original
    F_eco = (mag * phi_env) * np.exp(1j * phase * k)
    s_eco = np.fft.irfft(F_eco, n=len(x))
    return s_eco, k, coh

def gerar_harmonicos(freq_base=220.0, n_harmonicos=6):
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    sinal = sum((1.0/n) * np.sin(2*np.pi*freq_base*n*t) for n in range(1, n_harmonicos+1))
    return sinal / np.max(np.abs(sinal))

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def salvar_wav(sinal, nome):
    wavfile.write(nome, FS, np.int16(np.clip(sinal, -1.0, 1.0) * 32767))

sinal_orig              = gerar_harmonicos()
sinal_eco_raw, k, coh  = eco_ressonante(sinal_orig)
sinal_eco_raw           = normalizar(sinal_eco_raw)

diff        = sinal_eco_raw - sinal_orig
sinal_lupa  = normalizar(sinal_orig + diff * GANHO_LUPA)
sinal_micro = normalizar(diff * GANHO_MICRO)

salvar_wav(sinal_orig,    "audio_original.wav")
salvar_wav(sinal_eco_raw, "audio_eco.wav")
salvar_wav(sinal_lupa,    "audio_eco_lupa.wav")
salvar_wav(sinal_micro,   "audio_eco_micro.wav")

F_orig = np.fft.rfft(sinal_orig)
F_eco  = np.fft.rfft(sinal_eco_raw)
freqs  = np.fft.rfftfreq(len(sinal_orig), 1/FS)
idx    = freqs <= 1000
e_orig = np.abs(F_orig)[idx].sum()
e_eco  = np.abs(F_eco)[idx].sum()

print(f"k={k:.5f}  coerência={coh:.5f}  razão espectral 0-1kHz={e_eco/e_orig:.4f}")
print()
print("Ganho φ-eco por harmônico:")
for n in range(1, 7):
    f_n   = 220.0 * n
    bin_n = round(f_n * len(sinal_orig) / FS)
    if bin_n < len(F_orig):
        g = np.abs(F_eco[bin_n]) / (np.abs(F_orig[bin_n]) + 1e-12)
        bar = "+" if g >= 1.0 else "-"
        print(f"  {n}º harm {f_n:.0f} Hz: {g:.3f}×  {bar}")
print()
print("1) ORIGINAL")
display(Audio("audio_original.wav"))
print("2) ECO direto (φ-envelope nas amplitudes)")
display(Audio("audio_eco.wav"))
print("3) LUPA: original + diferença ×10")
display(Audio("audio_eco_lupa.wav"))
print("4) MICROSCÓPIO: só a diferença ×20")
display(Audio("audio_eco_micro.wav"))
