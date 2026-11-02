# ECO BEEP 880 — α* = 1/3 (extraído do código original do Manifesto 01, sem alterar nenhuma linha de cálculo)
# Entrada: x = (1 − α)·beep 880 Hz + α·FM-φ, com α = 1/3. Eco-φ em 5 passos (pontos de dobra), agente com 20 ciclos.
# Saída: um único áudio com a entrada e os 5 pontos de dobra em sequência.

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

PHI     = (1 + np.sqrt(5)) / 2
FS      = 44100
DURACAO = 1.5
N_STEPS = 5
ALPHA   = 0.33333333   # α* = 1/3, com o mesmo arredondamento (8 casas) da varredura original
F_ORG   = 220.0
F_M     = F_ORG / PHI
F_BEEP  = 880.0

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max:
            break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n_samples):
    resultado = []
    for f_low, f_high in bandas:
        b_low  = max(int(np.floor(f_low  * n_samples / FS)), 0)
        b_high = min(int(np.ceil(f_high  * n_samples / FS)), n_samples // 2 + 1)
        if b_high - b_low >= 3:
            resultado.append((b_low, b_high, f_low, f_high))
    return resultado

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
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
        if coh_mem is not None and i < len(coh_mem):
            coh_ef = w_now * coh + w_mem * float(coh_mem[i])
        else:
            coh_ef = coh
        cohs.append(coh)
        n_idx = np.arange(len(F_band))
        env   = np.clip(
            1.0 + (coh_ef * PHI**beta_i) * np.cos(2.0 * np.pi * n_idx / PHI),
            0.05, None
        )
        F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase)
    resultado = np.fft.irfft(F_out, n=N)
    return resultado / (np.max(np.abs(resultado)) + 1e-10), np.array(cohs, dtype=float)

def cascata_eq(sinal_entrada, beta_bands, bins_phi):
    cas        = [sinal_entrada]
    s          = sinal_entrada.copy()
    n_bandas   = len(bins_phi)
    coh_mem    = np.zeros(n_bandas, dtype=float)
    cohs_final = np.zeros(n_bandas, dtype=float)
    for _ in range(N_STEPS):
        s_e, cohs  = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem    = cohs
        cohs_final = cohs
        s_e        = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()
    return cas, cohs_final

def agente_eco(sinal_entrada, bins_phi, n_ciclos=20):
    n_bandas  = len(bins_phi)
    beta      = np.ones(n_bandas, dtype=float)
    beta_mem  = beta.copy()
    w_mem     = 1.0 / PHI
    w_now     = 1.0 - w_mem
    cas_final = None
    for _ in range(n_ciclos):
        cas, cohs  = cascata_eq(sinal_entrada, beta, bins_phi)
        coh_rel    = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo  = PHI ** (3 * coh_rel)
        beta       = w_now * beta_alvo + w_mem * beta_mem
        beta_mem   = beta.copy()
        beta       = np.clip(beta, 0.05, PHI**3)
        cas_final  = cas
    return beta, cas_final

def gerar_beep(freq, duracao=DURACAO):
    t   = np.linspace(0, duracao, int(FS * duracao), endpoint=False)
    sig = np.sign(np.sin(2 * np.pi * freq * t))
    return normalizar(sig)

def gerar_fm_phi():
    t   = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    sig = np.sin(2 * np.pi * F_ORG * t + PHI * np.sin(2 * np.pi * F_M * t))
    return normalizar(sig)

def crossfade(a, b, fade=2000):
    t = np.linspace(0, 1, fade)
    return np.concatenate([a[:-fade], a[-fade:] * (1-t) + b[:fade] * t, b[fade:]])

def concatenar(seq):
    out = seq[0].copy()
    for sig in seq[1:]:
        out = crossfade(out, sig)
    return normalizar(out)

# ── ECO BEEP 880, α* = 1/3 ────────────────────────────────────────────────────
BINS_PHI = bandas_para_bins(gerar_bandas_phi(), int(FS * DURACAO))
x_mix    = normalizar((1.0 - ALPHA) * gerar_beep(F_BEEP) + ALPHA * gerar_fm_phi())
beta, cas = agente_eco(x_mix, BINS_PHI, n_ciclos=20)

eco_880 = concatenar(cas)
wavfile.write("eco_beep_880.wav", FS, np.int16(np.clip(eco_880, -1.0, 1.0) * 32767))
print("ECO BEEP 880 · α* = 1/3 · entrada + 5 pontos de dobra")
display(Audio("eco_beep_880.wav"))
