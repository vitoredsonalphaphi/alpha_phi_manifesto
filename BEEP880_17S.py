# © Vitor Edson Delavi · Florianópolis · 2026
# ECO BEEP 880 — 17 segundos
#
# [0 – 7s]   ECO BEEP 880 idêntico ao original (5 fases, campo forma)
# [7 – 17s]  extensão 10s com expertise do violão (coh_mem propagado)
#
# Rode no Google Colab.

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

PHI     = (1 + np.sqrt(5)) / 2
FS      = 44100
F_BEEP  = 880.0
F_ORG   = 220.0
F_M     = F_ORG / PHI
BETA_FM = PHI
ALPHA   = 1.0 / 3.0
N_STEPS = 5
N_CICLOS= 20
DUR_BIP = 7.0    # segundos de ECO BEEP 880
DUR_EXT = 10.0   # segundos de extensão
CHUNK   = 1.5    # tamanho do chunk na extensão (igual ao beep original)

def normalizar(s):
    m = np.max(np.abs(s)); return s/m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f*PHI, f_max); bandas.append((f, f_next))
        if f_next >= f_max: break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    return [(max(0, int(f_lo/(FS/n))), min(int(f_hi/(FS/n))+1, n//2+1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb  = F[b_lo:b_hi]; mag = np.abs(Fb); phase = np.angle(Fb)
        an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        ce  = (wn*coh + wm*float(coh_mem[i])
               if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk  = np.arange(len(Fb))
        env = np.clip(1.0+(ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag*env)*np.exp(1j*phase)
    r = np.fft.irfft(F_out, n=N)
    return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi, coh_mem_init=None):
    s  = sinal.copy()
    cm = coh_mem_init.copy() if coh_mem_init is not None else np.zeros(len(bins_phi))
    passos = []
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se); passos.append(se.copy()); s = se.copy()
    return passos, cohs

def gerar_beep(duracao):
    N = int(FS*duracao); t = np.linspace(0, duracao, N, endpoint=False)
    beep = np.sign(np.sin(2*np.pi*F_BEEP*t))
    fm   = np.sin(2*np.pi*F_M*t + BETA_FM*np.sin(2*np.pi*F_M*t/PHI))
    return normalizar(ALPHA*beep + (1-ALPHA)*fm)

# ── [0–7s] ECO BEEP 880 idêntico ─────────────────────────────────────────────
print("="*50)
print("ECO BEEP 880 — 17 segundos")
print(f"[0–{DUR_BIP:.0f}s] ECO BEEP 880  |  [{DUR_BIP:.0f}–{DUR_BIP+DUR_EXT:.0f}s] extensão")
print("="*50)

sinal_bip = gerar_beep(DUR_BIP)
BANDAS    = gerar_bandas_phi()
BINS      = bandas_para_bins(BANDAS, len(sinal_bip))

print(f"\n[>] Convergência ({N_CICLOS} ciclos, idêntico ao original)...")
nb   = len(BINS)
beta = np.ones(nb); bm = beta.copy()
wm, wn  = 1.0/PHI, 1.0-1.0/PHI
limiar  = 0.99*PHI**3
passo5  = sinal_bip.copy(); coh_final = np.zeros(nb)

for ciclo in range(N_CICLOS):
    passos, cohs = cascata_eq(sinal_bip, beta, BINS)
    cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
    ba   = PHI**(3*cr)
    beta = wn*ba + wm*bm; bm = beta.copy()
    beta = np.clip(beta, 0.05, PHI**3)
    passo5 = passos[-1]; coh_final = cohs
    if beta.max() >= limiar:
        print(f"    Campo harmônico — ciclo {ciclo+1}/{N_CICLOS}  β máx={beta.max():.4f}")
        break

# ── [7–17s] extensão com coh_mem propagado ───────────────────────────────────
print(f"\n[>] Extensão {DUR_EXT:.0f}s (chunks de {CHUNK}s, coh_mem propagado)...")
N_CHUNKS = int(np.ceil(DUR_EXT / CHUNK))
chunks, cm = [], coh_final.copy()
FADE_N = int(0.03*FS)

for i in range(N_CHUNKS):
    chunk  = gerar_beep(CHUNK)
    BINS_C = bandas_para_bins(BANDAS, len(chunk))
    b_c = beta[:len(BINS_C)]
    if len(b_c) < len(BINS_C):
        b_c = np.pad(b_c, (0,len(BINS_C)-len(b_c)), constant_values=PHI**3)
    cm_c = cm[:len(BINS_C)]
    if len(cm_c) < len(BINS_C):
        cm_c = np.pad(cm_c, (0,len(BINS_C)-len(cm_c)), constant_values=0.0)
    passos_c, cohs_c = cascata_eq(chunk, b_c, BINS_C, coh_mem_init=cm_c)
    chunks.append(normalizar(passos_c[-1])); cm = cohs_c

# Montar extensão
extensao = chunks[0].copy()
for c in chunks[1:]:
    f = min(FADE_N, len(extensao), len(c)); fade = np.linspace(0,1,f)
    extensao[-f:] = extensao[-f:]*(1-fade) + c[:f]*fade
    extensao = np.concatenate([extensao, c[f:]])
extensao = normalizar(extensao[:int(DUR_EXT*FS)])

# Montar sequência completa [passo5 | extensão]
f = min(FADE_N, len(passo5), len(extensao)); fade = np.linspace(0,1,f)
p5 = passo5.copy(); p5[-f:] = passo5[-f:]*(1-fade) + extensao[:f]*fade
sequencia = normalizar(np.concatenate([p5, extensao[f:]]))

print(f"    Total: {len(sequencia)/FS:.1f}s")

# ── Métricas ──────────────────────────────────────────────────────────────────
def autocorr(s):
    s=s-s.mean(); return float(np.corrcoef(s[:-1],s[1:])[0,1])
def entr_esp(s):
    mag=np.abs(np.fft.rfft(s))**2; p=mag/(mag.sum()+1e-12); p=p[p>1e-15]
    return float(-np.sum(p*np.log2(p)))

print(f"\n{'='*50}")
print(f"{'Fase':<26} {'AutoCorr':>9} {'EntrEsp':>9}")
print("-"*50)
seg = len(sinal_bip)//4
for nome, sig in [
    ("ECO BEEP 880 (passo5)",    passo5[:seg]),
    ("Extensão (início)",        extensao[:seg]),
    ("Extensão (final)",         extensao[-seg:]),
]:
    print(f"{nome:<26} {autocorr(sig):>9.4f} {entr_esp(sig):>9.4f}")
print(f"{'Alvo':<26} {'1.0000':>9} {'0.0601':>9}")

# ── Salvar e reproduzir ───────────────────────────────────────────────────────
wavfile.write("beep880_17s.wav", FS, (sequencia*32767).astype(np.int16))
print(f"\nSalvo: beep880_17s.wav")

print(f"\n{'='*50}")
print(f"▶  ECO BEEP 880 — {len(sequencia)/FS:.0f}s  (campo + extensão)")
display(Audio(sequencia, rate=FS))

print("\nOBSERVE:")
print("  A sensação do ponto de dobra 5 persiste nos 10s de extensão?")
print("  Há quebra perceptível na passagem do 7° para o 8° segundo?")
