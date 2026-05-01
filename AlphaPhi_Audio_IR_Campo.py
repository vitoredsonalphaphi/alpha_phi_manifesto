"""
AlphaPhi Audio — IR Acústica φ vs IR Neural φ + Eco Campo
Célula única para Google Colab.

Dois campos físicos modelados com parâmetros φ:

IR ACÚSTICA (sala):
  - Som direto + reflexões com atrasos φ^n × d/c e amplitudes 1/φ^n
  - Distância fonte-receptor: d = φ metros
  - RT60 = φ × 0.5 ≈ 0.809s (tempo de reverberação)

IR NEURAL (crânio):
  - Cascata de 3 filtros LP representando camadas biológicas
  - Escalpo  : f_c = 100 Hz
  - Crânio   : f_c = 100/φ² ≈ 38 Hz  (dominante — mais resistivo)
  - LCR      : f_c = 100×φ ≈ 162 Hz  (mais condutivo)
  - Reproduz atenuação de volume conduction do EEG

eco_campo_ir: convolve sinal com IR → mede coerência do campo →
              aplica φ-envelope → devolve correção ao sinal original
eco_ponto:    age no sinal individualmente (fonte)
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

PHI     = (1 + np.sqrt(5)) / 2
K_MIN   = np.sqrt(2)
FS      = 44100
DURACAO = 1.5
N_STEPS = 5
F_C     = 220.0
F_M     = F_C / PHI    # ≈ 135.9 Hz

# ── IR acústica (sala φ) ──────────────────────────────────────────────────────
def gerar_ir_acustica(n_samples, rt60=PHI*0.5, n_ref=8):
    h = np.zeros(n_samples)
    h[0] = 1.0                          # som direto
    c = 343.0                           # velocidade do som (m/s)
    d = PHI                             # distância fonte-receptor (m)
    for n in range(1, n_ref + 1):
        delay_s = (PHI**n * d) / c      # atraso em s: φ^n × d/c
        ds = int(delay_s * FS)
        if ds < n_samples:
            h[ds] += (1.0 / PHI)**n    # amplitude: 1/φ^n
    t = np.arange(n_samples) / FS
    h *= np.exp(-t * np.log(1000) / rt60)   # decaimento RT60
    return h / (np.max(np.abs(h)) + 1e-10)

# ── IR neural (crânio φ) ──────────────────────────────────────────────────────
def gerar_ir_neural(n_samples):
    """
    Cascata de 3 filtros passa-baixa 1ª ordem com f_c em proporções φ.
    Modela atenuação cumulativa escalpo → crânio → LCR.
    """
    freqs = np.fft.rfftfreq(n_samples, 1.0 / FS)
    H = np.ones(len(freqs), dtype=complex)
    f_base = 100.0
    for f_c in [f_base, f_base / PHI**2, f_base * PHI]:
        H *= 1.0 / (1.0 + 1j * freqs / (f_c + 1e-10))
    h = np.fft.irfft(H, n=n_samples)
    return h / (np.max(np.abs(h)) + 1e-10), np.abs(H)

# ── eco ───────────────────────────────────────────────────────────────────────
def medir_k(x):
    fb = np.fft.fft(x)
    am = np.abs(fb)
    an = np.clip(am / (am.sum() + 1e-8), 1e-10, 1.0)
    e  = -np.sum(an * np.log(an))
    c  = float(1.0 - e / np.log(len(x)))
    return K_MIN + (PHI - K_MIN) * c, c

def phi_env(n_idx, coh):
    return np.clip(1.0 + coh * np.cos(2.0 * np.pi * n_idx / PHI), 0.05, None)

def eco_ponto(x):
    k, coh = medir_k(x)
    F     = np.fft.rfft(x)
    n_idx = np.arange(len(F))
    F_eco = (np.abs(F) * phi_env(n_idx, coh)) * np.exp(1j * np.angle(F) * k)
    return np.fft.irfft(F_eco, n=len(x)), k, coh

def eco_campo_ir(x, ir):
    """
    Campo físico via IR:
    1. Convolve x com IR → sinal no campo
    2. Mede coerência do campo
    3. φ-envelope sobre espectro do campo
    4. Correção = campo_eco − campo_original
    5. Retorna x + correção (sinal corrigido pelo campo)
    """
    x_campo = np.convolve(x, ir, mode='same')
    x_campo = x_campo / (np.max(np.abs(x_campo)) + 1e-10)
    k, coh  = medir_k(x_campo)
    F       = np.fft.rfft(x_campo)
    n_idx   = np.arange(len(F))
    F_eco   = (np.abs(F) * phi_env(n_idx, coh)) * np.exp(1j * np.angle(F) * k)
    correcao = np.fft.irfft(F_eco, n=len(x)) - x_campo
    resultado = x + correcao
    return resultado / (np.max(np.abs(resultado)) + 1e-10), k, coh

# ── síntese FM-φ ──────────────────────────────────────────────────────────────
def gerar_fm():
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    s = np.sin(2*np.pi*F_C*t + PHI * np.sin(2*np.pi*F_M*t))
    return s / np.max(np.abs(s))

def normalizar(s):
    m = np.max(np.abs(s)); return s / m if m > 1e-12 else s

def crossfade(a, b, fade=2000):
    t = np.linspace(0, 1, fade)
    return np.concatenate([a[:-fade], a[-fade:]*(1-t) + b[:fade]*t, b[fade:]])

def concatenar(seq):
    out = seq[0].copy()
    for sig in seq[1:]: out = crossfade(out, sig)
    return normalizar(out)

def salvar_wav(s, nome):
    wavfile.write(nome, FS, np.int16(np.clip(s, -1.0, 1.0) * 32767))

# ── gerar IRs e cascatas ──────────────────────────────────────────────────────
N_SINAL = int(FS * DURACAO)
ir_ac        = gerar_ir_acustica(N_SINAL)
ir_ne, H_ne  = gerar_ir_neural(N_SINAL)
sinal_fm     = gerar_fm()

def cascata_ir(ir, label):
    cas = [sinal_fm]
    s   = sinal_fm.copy()
    ks, cohs = [], []
    for _ in range(N_STEPS):
        s_c, k, coh = eco_campo_ir(s, ir)   # campo (ressonância)
        s_u, kp, _  = eco_ponto(s_c)        # ponto (fonte)
        s_u = normalizar(s_u)
        cas.append(s_u); s = s_u.copy()
        ks.append(k); cohs.append(coh)
    print(f"  {label:<20}: k_campo={ks[-1]:.5f}  coh={cohs[-1]:.4f}")
    return cas

print(f"FM-φ:  f_c={F_C:.0f}Hz  f_m={F_M:.1f}Hz  β={PHI:.3f}\n")
print("Gerando cascatas...")
cas_ac = cascata_ir(ir_ac, "IR Acústica (sala)")
cas_ne = cascata_ir(ir_ne, "IR Neural  (crânio)")

# ── tabela de parciais ────────────────────────────────────────────────────────
ref    = np.abs(np.fft.rfft(sinal_fm))
vistos, parciais = set(), []
for n in range(6):
    for f in [abs(F_C + n*F_M), abs(F_C - n*F_M)]:
        f = round(f, 1)
        if 20 < f < FS/2 and f not in vistos:
            vistos.add(f); parciais.append(f)
parciais = sorted(parciais)[:8]

def tabela(cas, titulo):
    print(f"\n{titulo}")
    print(f"{'':>6}", end="")
    for f in parciais: print(f"  {f:>6.1f}Hz", end="")
    print()
    for i, sig in enumerate(cas):
        F_sig = np.fft.rfft(sig)
        label = "orig  " if i == 0 else f"eco×{i} "
        print(label, end="")
        for f in parciais:
            b = round(f * N_SINAL / FS)
            r = np.abs(F_sig[b]) / (ref[b] + 1e-12) if b < len(F_sig) else 0.0
            print(f"  {r:>6.3f}", end="")
        print()

tabela(cas_ac, "── IR ACÚSTICA φ — sala (d=φm, RT60=φ×0.5s) ────────")
tabela(cas_ne, "── IR NEURAL φ  — crânio (3 camadas biológicas) ────")

# perfil de atenuação da IR neural
print("\nPerfil IR Neural (atenuação por frequência):")
for f_check in [10, 38, 100, 162, 220, 356, 440, 1000]:
    b = round(f_check * N_SINAL / FS)
    if b < len(H_ne):
        db = 20 * np.log10(H_ne[b] + 1e-10)
        mark = " ← par φ" if f_check in [220, 356] else ""
        print(f"  {f_check:>5} Hz : {db:>7.2f} dB{mark}")

# ── áudio ─────────────────────────────────────────────────────────────────────
salvar_wav(concatenar(cas_ac),        "ac_descendente.wav")
salvar_wav(concatenar(cas_ac[::-1]),  "ac_ascendente.wav")
salvar_wav(concatenar(cas_ne),        "ne_descendente.wav")
salvar_wav(concatenar(cas_ne[::-1]),  "ne_ascendente.wav")

print("\n── IR ACÚSTICA φ (sala) ──────────────────────────────")
print("Descendente:"); display(Audio("ac_descendente.wav"))
print("Ascendente:");  display(Audio("ac_ascendente.wav"))
print("\n── IR NEURAL φ (crânio) ─────────────────────────────")
print("Descendente:"); display(Audio("ne_descendente.wav"))
print("Ascendente:");  display(Audio("ne_ascendente.wav"))
