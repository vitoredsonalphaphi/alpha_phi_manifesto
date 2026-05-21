"""
Gera beep880_puro.wav — ECO BEEP 880 puro (passo 5 da cascata, N_CICLOS=20)
com 10s de silêncio padded após o sinal, para teste de decaimento livre.

Parâmetros idênticos ao experimento original:
  F_BEEP=880Hz · F_ORG=220Hz · ALPHA=1/3 · N_STEPS=5 · N_CICLOS=20
"""

import numpy as np
from scipy.io import wavfile

PHI      = (1 + np.sqrt(5)) / 2
FS       = 44100
F_BEEP   = 880.0
F_ORG    = 220.0
F_M      = F_ORG / PHI
BETA_FM  = PHI
ALPHA    = 1.0 / 3.0
N_STEPS  = 5
N_CICLOS = 20
DURACAO  = 7.0     # idêntico ao BEEP880_17S.py
PAD_S    = 15.0    # segundos de silêncio após o sinal (para decaimento livre)

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max:
            break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    return [(max(0, int(f_lo / (FS / n))), min(int(f_hi / (FS / n)) + 1, n // 2 + 1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb  = F[b_lo:b_hi]; mag = np.abs(Fb); phase = np.angle(Fb)
        an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        ce  = (wn * coh + wm * float(coh_mem[i])
               if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce * PHI ** bi) * np.cos(2 * np.pi * nk / PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)
    r = np.fft.irfft(F_out, n=N)
    return r / (np.max(np.abs(r)) + 1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi, coh_mem_init=None):
    s  = sinal.copy()
    cm = coh_mem_init.copy() if coh_mem_init is not None else np.zeros(len(bins_phi))
    passos = []
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se); passos.append(se.copy()); s = se.copy()
    return passos, cohs

def gerar_beep(duracao):
    N = int(FS * duracao); t = np.linspace(0, duracao, N, endpoint=False)
    beep = np.sign(np.sin(2 * np.pi * F_BEEP * t))
    fm   = np.sin(2 * np.pi * F_M * t + BETA_FM * np.sin(2 * np.pi * F_M * t / PHI))
    return normalizar(ALPHA * beep + (1 - ALPHA) * fm)

def autocorr(s):
    s = s - s.mean(); return float(np.corrcoef(s[:-1], s[1:])[0, 1])

def entr_esp(s):
    mag = np.abs(np.fft.rfft(s)) ** 2
    p   = mag / (mag.sum() + 1e-12); p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))

print("=" * 55)
print("ECO BEEP 880 PURO — gerando campo no passo 5")
print(f"F_BEEP={F_BEEP}Hz · F_ORG={F_ORG}Hz · α={ALPHA:.4f} · N_CICLOS={N_CICLOS}")
print("=" * 55)

sinal_bip = gerar_beep(DURACAO)
BANDAS    = gerar_bandas_phi()
BINS      = bandas_para_bins(BANDAS, len(sinal_bip))

nb       = len(BINS)
beta     = np.ones(nb); bm = beta.copy()
wm_b, wn_b = 1.0 / PHI, 1.0 - 1.0 / PHI
limiar   = 0.99 * PHI ** 3
passo5   = sinal_bip.copy(); coh_final = np.zeros(nb)

for ciclo in range(N_CICLOS):
    passos, cohs = cascata_eq(sinal_bip, beta, BINS)
    cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
    ba   = PHI ** (3 * cr)
    beta = wn_b * ba + wm_b * bm; bm = beta.copy()
    beta = np.clip(beta, 0.05, PHI ** 3)
    passo5 = passos[-1]; coh_final = cohs
    if beta.max() >= limiar:
        print(f"  Campo harmônico atingido — ciclo {ciclo + 1}/{N_CICLOS}  β_max={beta.max():.4f}")
        break

print(f"\nPasso 5:")
print(f"  AutoCorr = {autocorr(passo5):.4f}  (alvo: 1.0000)")
print(f"  EntrEsp  = {entr_esp(passo5):.4f}  (alvo: 0.0601)")

# passo5 + silêncio padded
silencio  = np.zeros(int(PAD_S * FS), dtype=passo5.dtype)
sequencia = np.concatenate([passo5, silencio])
sequencia = normalizar(sequencia)

wavfile.write("beep880_puro.wav", FS, (sequencia * 32767).astype(np.int16))
print(f"\nSalvo: beep880_puro.wav  ({len(sequencia)/FS:.1f}s = {DURACAO}s sinal + {PAD_S:.0f}s silêncio)")
print("Pronto para AlphaPhi_Lupa_Emissao.py")
