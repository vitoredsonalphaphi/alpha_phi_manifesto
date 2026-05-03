"""
AlphaPhi Audio — FM φ + Cascata Eco
Célula única para Google Colab.

FM com proporção φ:
  portadora  f_c = 220 Hz
  moduladora f_m = f_c / φ  ≈ 135.9 Hz   (razão irracional → parciais inarmônicos)
  índice     β   = φ        ≈ 1.618

Cascata de eco sobre o sinal FM:
  eco×1 → eco×2 → ... → eco×N
  play ascendente: eco×N → ... → original
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

PHI     = (1 + np.sqrt(5)) / 2
K_MIN   = np.sqrt(2)
FS      = 44100
DURACAO = 1.5
N_STEPS = 5
BETA    = PHI           # índice de modulação = φ
F_C     = 220.0         # portadora (Hz)
F_M     = F_C / PHI    # moduladora ≈ 135.9 Hz

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
    phi_env = 1.0 + coh * np.cos(2.0 * np.pi * n_idx / PHI)
    phi_env = np.clip(phi_env, 0.05, None)
    F_eco = (mag * phi_env) * np.exp(1j * phase * k)
    return np.fft.irfft(F_eco, n=len(x)), k, coh

def gerar_fm():
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    s = np.sin(2*np.pi*F_C*t + BETA * np.sin(2*np.pi*F_M*t))
    return s / np.max(np.abs(s))

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def crossfade(a, b, fade=2000):
    t = np.linspace(0, 1, fade)
    return np.concatenate([a[:-fade], a[-fade:]*(1-t) + b[:fade]*t, b[fade:]])

def salvar_wav(sinal, nome):
    wavfile.write(nome, FS, np.int16(np.clip(sinal, -1.0, 1.0) * 32767))

# cascata
sinal_fm = gerar_fm()
cascata  = [sinal_fm]
s = sinal_fm.copy()
for _ in range(N_STEPS):
    s, k, coh = eco_ressonante(s)
    cascata.append(normalizar(s))

# parciais FM: f_c ± n*f_m para n=0..5, frequências positivas únicas
N   = int(FS * DURACAO)
ref = np.abs(np.fft.rfft(cascata[0]))

vistos = set()
parciais = []
for n in range(6):
    for f in [abs(F_C + n*F_M), abs(F_C - n*F_M)]:
        f = round(f, 1)
        if 20 < f < FS/2 and f not in vistos:
            vistos.add(f)
            parciais.append(f)
parciais = sorted(parciais)[:8]

print(f"FM φ:  f_c={F_C:.0f}Hz  f_m={F_M:.1f}Hz  β={BETA:.3f}  (f_m = f_c / φ)\n")
print(f"{'':>6}", end="")
for f in parciais:
    print(f"  {f:>6.1f}Hz", end="")
print()

for i, sig in enumerate(cascata):
    F_sig = np.fft.rfft(sig)
    label = "orig  " if i == 0 else f"eco×{i} "
    print(label, end="")
    for f in parciais:
        b = round(f * N / FS)
        r = np.abs(F_sig[b]) / (ref[b] + 1e-12) if b < len(F_sig) else 0.0
        print(f"  {r:>6.3f}", end="")
    print()

def concatenar(seq):
    out = seq[0].copy()
    for sig in seq[1:]:
        out = crossfade(out, sig)
    return normalizar(out)

salvar_wav(concatenar(cascata),        "fm_descendente.wav")
salvar_wav(concatenar(cascata[::-1]),  "fm_ascendente.wav")

print(f"\nk={k:.5f}  coerência={coh:.5f}")
print("\nDESCENDENTE — FM original → eco×5:")
display(Audio("fm_descendente.wav"))
print("ASCENDENTE — eco×5 → FM original:")
display(Audio("fm_ascendente.wav"))
