"""
AlphaPhi Audio — IR Cascata: Sala φ → Crânio φ
Célula única para Google Colab.

Encadeamento físico: ambiente acústico → substrato neural
  1. Sinal FM-φ entra na sala (IR acústica)
  2. Saída da sala entra no crânio (IR neural)
  3. Cada IR aplica eco_campo_ir independentemente

A sala seleciona 220Hz (portadora).
O crânio seleciona 356Hz (modulada = portadora × φ).
O encadeamento revela qual membro do par φ sobrevive
à tensão entre os dois substratos.

Comparação:
  cas_ac   — só sala
  cas_ne   — só crânio
  cas_enc  — sala → crânio (encadeado)
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

# ── IRs ───────────────────────────────────────────────────────────────────────
def gerar_ir_acustica(n_samples, rt60=PHI*0.5, n_ref=8):
    h = np.zeros(n_samples)
    h[0] = 1.0
    c = 343.0; d = PHI
    for n in range(1, n_ref + 1):
        delay_s = (PHI**n * d) / c
        ds = int(delay_s * FS)
        if ds < n_samples:
            h[ds] += (1.0 / PHI)**n
    t = np.arange(n_samples) / FS
    h *= np.exp(-t * np.log(1000) / rt60)
    return h / (np.max(np.abs(h)) + 1e-10)

def gerar_ir_neural(n_samples):
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
    x_campo  = np.convolve(x, ir, mode='same')
    x_campo  = x_campo / (np.max(np.abs(x_campo)) + 1e-10)
    k, coh   = medir_k(x_campo)
    F        = np.fft.rfft(x_campo)
    n_idx    = np.arange(len(F))
    F_eco    = (np.abs(F) * phi_env(n_idx, coh)) * np.exp(1j * np.angle(F) * k)
    correcao = np.fft.irfft(F_eco, n=len(x)) - x_campo
    resultado = x + correcao
    return resultado / (np.max(np.abs(resultado)) + 1e-10), k, coh

# ── síntese e utilitários ──────────────────────────────────────────────────────
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

# ── cascatas ──────────────────────────────────────────────────────────────────
N_SINAL      = int(FS * DURACAO)
ir_ac        = gerar_ir_acustica(N_SINAL)
ir_ne, H_ne  = gerar_ir_neural(N_SINAL)
sinal_fm     = gerar_fm()

def cascata_simples(ir, label):
    cas = [sinal_fm]; s = sinal_fm.copy()
    ks, cohs = [], []
    for _ in range(N_STEPS):
        s_c, k, coh = eco_campo_ir(s, ir)
        s_u, _, _   = eco_ponto(s_c)
        s_u = normalizar(s_u)
        cas.append(s_u); s = s_u.copy()
        ks.append(k); cohs.append(coh)
    print(f"  {label:<22}: k={ks[-1]:.5f}  coh={cohs[-1]:.4f}")
    return cas

def cascata_encadeada():
    """Sala → crânio: cada passo aplica IR acústica e depois IR neural."""
    cas = [sinal_fm]; s = sinal_fm.copy()
    ks_ac, ks_ne, cohs_ac, cohs_ne = [], [], [], []
    for _ in range(N_STEPS):
        # 1. campo acústico (sala)
        s_sala, k_ac, coh_ac = eco_campo_ir(s, ir_ac)
        s_sala = normalizar(s_sala)
        # 2. campo neural (crânio) sobre saída da sala
        s_cranio, k_ne, coh_ne = eco_campo_ir(s_sala, ir_ne)
        # 3. ponto (fonte individual) sobre resultado encadeado
        s_u, _, _ = eco_ponto(s_cranio)
        s_u = normalizar(s_u)
        cas.append(s_u); s = s_u.copy()
        ks_ac.append(k_ac); ks_ne.append(k_ne)
        cohs_ac.append(coh_ac); cohs_ne.append(coh_ne)
    print(f"  {'Encadeada (sala→crânio)':<22}: k_ac={ks_ac[-1]:.5f} coh_ac={cohs_ac[-1]:.4f} | k_ne={ks_ne[-1]:.5f} coh_ne={cohs_ne[-1]:.4f}")
    return cas

print(f"FM-φ:  f_c={F_C:.0f}Hz  f_m={F_M:.1f}Hz  β={PHI:.3f}\n")
print("Gerando cascatas...")
cas_ac  = cascata_simples(ir_ac, "IR Acústica (sala)")
cas_ne  = cascata_simples(ir_ne, "IR Neural  (crânio)")
cas_enc = cascata_encadeada()

# ── tabela comparativa ────────────────────────────────────────────────────────
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

tabela(cas_ac,  "── SÓ SALA φ ──────────────────────────────────────")
tabela(cas_ne,  "── SÓ CRÂNIO φ ─────────────────────────────────────")
tabela(cas_enc, "── ENCADEADA sala→crânio φ ─────────────────────────")

# resumo do par φ no eco×5
print("\n── Par φ no eco×5 ───────────────────────────────────")
print(f"{'':>22}  {'220Hz':>8}  {'356Hz':>8}  {'razão 356/220':>14}")
for cas, label in [(cas_ac,"Só sala"),(cas_ne,"Só crânio"),(cas_enc,"Encadeada")]:
    sig = cas[-1]
    F_sig = np.fft.rfft(sig)
    b220 = round(220.0 * N_SINAL / FS)
    b356 = round(356.0 * N_SINAL / FS)
    r220 = np.abs(F_sig[b220]) / (ref[b220] + 1e-12)
    r356 = np.abs(F_sig[b356]) / (ref[b356] + 1e-12)
    razao = r356 / (r220 + 1e-12)
    dom = "356Hz↑" if razao > 1 else "220Hz↑"
    print(f"  {label:<20}  {r220:>8.4f}  {r356:>8.4f}  {razao:>8.4f}  {dom}")

# ── áudio ─────────────────────────────────────────────────────────────────────
salvar_wav(concatenar(cas_ac),         "cascata_sala.wav")
salvar_wav(concatenar(cas_ac[::-1]),   "cascata_sala_asc.wav")
salvar_wav(concatenar(cas_ne),         "cascata_cranio.wav")
salvar_wav(concatenar(cas_ne[::-1]),   "cascata_cranio_asc.wav")
salvar_wav(concatenar(cas_enc),        "cascata_encadeada.wav")
salvar_wav(concatenar(cas_enc[::-1]),  "cascata_encadeada_asc.wav")

print("\n── SÓ SALA φ ─────────────────────────────────────────")
print("Descendente:"); display(Audio("cascata_sala.wav"))
print("Ascendente:");  display(Audio("cascata_sala_asc.wav"))
print("\n── SÓ CRÂNIO φ ───────────────────────────────────────")
print("Descendente:"); display(Audio("cascata_cranio.wav"))
print("Ascendente:");  display(Audio("cascata_cranio_asc.wav"))
print("\n── ENCADEADA sala→crânio φ ───────────────────────────")
print("Descendente:"); display(Audio("cascata_encadeada.wav"))
print("Ascendente:");  display(Audio("cascata_encadeada_asc.wav"))
