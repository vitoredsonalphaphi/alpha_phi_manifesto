"""
AlphaPhi Audio — Eco Ponto + Campo sobre FM-φ
Célula única para Google Colab.

eco_ponto  : age em cada passo da cascata individualmente (fonte)
eco_campo  : age no espectro médio acumulado da cascata (ressonância)
eco_unissono: campo → ponto em sequência

Sinal base: FM com razão φ
  f_c = 220 Hz  |  f_m = f_c / φ ≈ 135.9 Hz  |  β = φ
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

PHI     = (1 + np.sqrt(5)) / 2
K_MIN   = np.sqrt(2)
FS      = 44100
DURACAO = 1.5
N_STEPS = 5
BETA    = PHI
F_C     = 220.0
F_M     = F_C / PHI   # ≈ 135.9 Hz

# ── núcleo ─────────────────────────────────────────────────────────────────────
def medir_k(x_or_X):
    """Aceita sinal 1D ou batch 2D (usa espectro médio)."""
    X = np.atleast_2d(x_or_X)
    fb = np.fft.fft(X, axis=-1)
    am = np.abs(fb).mean(axis=0)
    an = np.clip(am / (am.sum() + 1e-8), 1e-10, 1.0)
    e  = -np.sum(an * np.log(an))
    c  = float(1.0 - e / np.log(X.shape[-1]))
    return K_MIN + (PHI - K_MIN) * c, c

def phi_env(n_idx, coh):
    env = 1.0 + coh * np.cos(2.0 * np.pi * n_idx / PHI)
    return np.clip(env, 0.05, None)

# ── eco ponto (fonte) ──────────────────────────────────────────────────────────
def eco_ponto(x):
    """Age no sinal individual."""
    k, coh = medir_k(x)
    F     = np.fft.rfft(x)
    n_idx = np.arange(len(F))
    F_eco = (np.abs(F) * phi_env(n_idx, coh)) * np.exp(1j * np.angle(F) * k)
    return np.fft.irfft(F_eco, n=len(x)), k, coh

# ── eco campo (ressonância) ────────────────────────────────────────────────────
def eco_campo_fit(cascata):
    """
    Fit: espectro médio de toda a cascata = campo acústico acumulado.
    Retorna correção 1D a aplicar em cada passo.
    """
    # empilha todos os passos e calcula espectro médio (o campo)
    stack  = np.stack(cascata, axis=0)          # (n_passos, n_amostras)
    k, coh = medir_k(stack)

    F_mean = np.fft.rfft(stack, axis=-1).mean(axis=0)
    mag    = np.abs(F_mean)
    phase  = np.angle(F_mean)
    n_idx  = np.arange(len(F_mean))

    F_eco_campo = (mag * phi_env(n_idx, coh)) * np.exp(1j * phase * k)

    N          = stack.shape[-1]
    correcao   = np.fft.irfft(F_eco_campo, n=N) - np.fft.irfft(F_mean, n=N)
    return correcao, k, coh

def eco_campo_apply(x, correcao):
    """Aplica correção de campo ao sinal."""
    return x + correcao

# ── geração e utilitários ─────────────────────────────────────────────────────
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

def concatenar(seq):
    out = seq[0].copy()
    for sig in seq[1:]:
        out = crossfade(out, sig)
    return normalizar(out)

def salvar_wav(sinal, nome):
    wavfile.write(nome, FS, np.int16(np.clip(sinal, -1.0, 1.0) * 32767))

# ── cascatas ──────────────────────────────────────────────────────────────────
sinal_fm = gerar_fm()

# — cascata A: só eco ponto (referência) —
cas_P = [sinal_fm]
s = sinal_fm.copy()
for _ in range(N_STEPS):
    s, _, _ = eco_ponto(s)
    cas_P.append(normalizar(s))

# — cascata B: eco uníssono (campo fit após cada passo, depois ponto) —
cas_U = [sinal_fm]
s = sinal_fm.copy()
for _ in range(N_STEPS):
    # campo: usa todos os passos acumulados até agora como "ambiente"
    corr, k_c, coh_c = eco_campo_fit(cas_U)
    s_c = eco_campo_apply(s, corr)          # corrige pelo campo
    s_c = normalizar(s_c)
    s_u, _, _ = eco_ponto(s_c)             # eco ponto dentro do campo
    s_u = normalizar(s_u)
    cas_U.append(s_u)
    s = s_u.copy()

# ── tabela de parciais ────────────────────────────────────────────────────────
N   = int(FS * DURACAO)
ref = np.abs(np.fft.rfft(sinal_fm))

vistos, parciais = set(), []
for n in range(6):
    for f in [abs(F_C + n*F_M), abs(F_C - n*F_M)]:
        f = round(f, 1)
        if 20 < f < FS/2 and f not in vistos:
            vistos.add(f); parciais.append(f)
parciais = sorted(parciais)[:8]

def tabela(cascata, titulo):
    print(f"\n{titulo}")
    print(f"{'':>6}", end="")
    for f in parciais: print(f"  {f:>6.1f}Hz", end="")
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

tabela(cas_P, "── ECO PONTO (fonte apenas) ──────────────────────────")
tabela(cas_U, "── ECO UNÍSSONO (campo + ponto) ─────────────────────")

# diagnóstico do campo
corr_final, k_campo, coh_campo = eco_campo_fit(cas_U)
_, k_ponto, coh_ponto = eco_ponto(sinal_fm)
print(f"\nk_ponto={k_ponto:.5f}  coh_ponto={coh_ponto:.4f}")
print(f"k_campo={k_campo:.5f}  coh_campo={coh_campo:.4f}")

# ── áudio ─────────────────────────────────────────────────────────────────────
salvar_wav(concatenar(cas_P),        "ponto_descendente.wav")
salvar_wav(concatenar(cas_P[::-1]),  "ponto_ascendente.wav")
salvar_wav(concatenar(cas_U),        "unissono_descendente.wav")
salvar_wav(concatenar(cas_U[::-1]),  "unissono_ascendente.wav")

print("\n── ECO PONTO ─────────────────────")
print("Descendente:"); display(Audio("ponto_descendente.wav"))
print("Ascendente:");  display(Audio("ponto_ascendente.wav"))
print("\n── ECO UNÍSSONO (campo + ponto) ──")
print("Descendente:"); display(Audio("unissono_descendente.wav"))
print("Ascendente:");  display(Audio("unissono_ascendente.wav"))
