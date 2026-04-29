"""
AlphaPhi Audio Eco — Lupa (v2)
Célula única para Google Colab.

Três sinais para comparação:
  1. Original          — harmônicos puros
  2. Eco x10 (lupa)    — diferença amplificada 10×, sobreposta ao original
  3. Diferença pura    — só o que o eco mudou, amplificada 20× (microscópio)
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

# ── constantes ────────────────────────────────────────────────────────────────
PHI      = (1 + np.sqrt(5)) / 2          # 1.618034…
K_MIN    = np.sqrt(2)                    # 1.414214…
N_ECO    = 5                             # iterações (era 3)
FS       = 44100
DURACAO  = 2.0
GANHO_LUPA  = 10.0   # amplifica diferença 10× no sinal eco
GANHO_MICRO = 20.0   # amplifica diferença 20× no sinal "microscópio"

# ── núcleo eco ────────────────────────────────────────────────────────────────
def medir_campo(x):
    f = np.fft.fft(x)
    a = np.abs(f)
    a = np.clip(a / (a.sum() + 1e-8), 1e-10, 1.0)
    e = -np.sum(a * np.log(a))
    c = float(1.0 - e / np.log(len(x)))
    return K_MIN + (PHI - K_MIN) * c, c

def eco_ressonante(x):
    k, coh = medir_campo(x)
    s = x.copy()
    for _ in range(N_ECO):
        f  = np.fft.fft(s)
        r  = np.real(np.fft.ifft(np.abs(f) * np.exp(1j * np.angle(f) * k)))
        s  = s + (r - x) * PHI
    return s, k, coh

# ── síntese de harmônicos ──────────────────────────────────────────────────────
def gerar_harmonicos(freq_base=220.0, n_harmonicos=6):
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    sinal = sum((1.0/n) * np.sin(2*np.pi*freq_base*n*t) for n in range(1, n_harmonicos+1))
    return sinal / np.max(np.abs(sinal))

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def salvar_wav(sinal, nome):
    wavfile.write(nome, FS, np.int16(np.clip(sinal, -1.0, 1.0) * 32767))

# ── processamento ─────────────────────────────────────────────────────────────
sinal_orig = gerar_harmonicos()

sinal_eco_raw, k, coh = eco_ressonante(sinal_orig)
sinal_eco_raw = normalizar(sinal_eco_raw)

diff = sinal_eco_raw - sinal_orig                          # o que o eco mudou

# lupa: original + diferença amplificada GANHO_LUPA vezes
sinal_lupa = normalizar(sinal_orig + diff * GANHO_LUPA)

# microscópio: só a diferença, amplificada GANHO_MICRO vezes
sinal_micro = normalizar(diff * GANHO_MICRO)

salvar_wav(sinal_orig,  "audio_original.wav")
salvar_wav(sinal_lupa,  "audio_eco_lupa.wav")
salvar_wav(sinal_micro, "audio_eco_micro.wav")

# ── razão espectral ────────────────────────────────────────────────────────────
freqs = np.fft.rfftfreq(len(sinal_orig), 1/FS)
idx   = freqs <= 1000
e_orig = np.abs(np.fft.rfft(sinal_orig))[idx].sum()
e_eco  = np.abs(np.fft.rfft(sinal_eco_raw))[idx].sum()

print(f"k={k:.5f}  coerência={coh:.5f}  razão espectral (eco/orig, 0-1kHz)={e_eco/e_orig:.4f}")
print(f"GANHO_LUPA={GANHO_LUPA}×  GANHO_MICRO={GANHO_MICRO}×  N_ECO={N_ECO}")
print()
print("1) ORIGINAL")
display(Audio("audio_original.wav"))
print("2) ECO LUPA (diferença ×10 sobre o original) — compare com 1")
display(Audio("audio_eco_lupa.wav"))
print("3) MICROSCÓPIO (só a diferença, ×20) — isso é o que o eco adiciona")
display(Audio("audio_eco_micro.wav"))
