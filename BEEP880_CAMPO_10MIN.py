# © Vitor Edson Delavi · Florianópolis · 2026
# ECO BEEP 880 — Campo Harmônico Estendido 10 minutos
#
# Processo IDÊNTICO ao original:
#   F_BEEP=880Hz  F_ORG=220Hz  α=1/3  FM-φ  N_STEPS=5  N_CICLOS=20
#
# Quando o campo harmônico forma (ponto de dobra 5):
#   β_conv aplicado em chunks de 1.5s com coh_mem propagado —
#   o campo não recomeça, continua de onde parou.
#
# Rode no Google Colab.

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

PHI      = (1 + np.sqrt(5)) / 2
FS       = 44100
F_BEEP   = 880.0
F_ORG    = 220.0
F_M      = F_ORG / PHI
BETA_FM  = PHI
ALPHA    = 1.0 / 3.0
DURACAO  = 1.5
N_STEPS  = 5
N_CICLOS = 20
EXTENSAO = 10 * 60   # 10 minutos em segundos

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max: break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    return [(max(0, int(f_lo/(FS/n))),
             min(int(f_hi/(FS/n))+1, n//2+1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0/PHI, 1.0 - 1.0/PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi   = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb   = F[b_lo:b_hi]
        mag  = np.abs(Fb); phase = np.angle(Fb)
        an   = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh  = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        ce   = (wn*coh + wm*float(coh_mem[i])
                if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk   = np.arange(len(Fb))
        env  = np.clip(1.0+(ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag*env)*np.exp(1j*phase)
    r = np.fft.irfft(F_out, n=N)
    return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi, coh_mem_init=None):
    s  = sinal.copy()
    cm = coh_mem_init.copy() if coh_mem_init is not None else np.zeros(len(bins_phi))
    passos = []
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se)
        passos.append(se.copy()); s = se.copy()
    return passos, cohs

def gerar_beep(duracao):
    N = int(FS * duracao)
    t = np.linspace(0, duracao, N, endpoint=False)
    beep = np.sign(np.sin(2*np.pi*F_BEEP*t))
    fm   = np.sin(2*np.pi*F_M*t + BETA_FM*np.sin(2*np.pi*F_M*t/PHI))
    return normalizar(ALPHA*beep + (1-ALPHA)*fm)

# ── Fase 1: convergência IDÊNTICA ao original ─────────────────────────────────
print("=" * 55)
print("ECO BEEP 880 — Campo Harmônico Estendido 10 minutos")
print(f"φ={PHI:.6f}  φ³={PHI**3:.4f}  α={ALPHA}  N_STEPS={N_STEPS}")
print("=" * 55)

sinal_base = gerar_beep(DURACAO)
BANDAS     = gerar_bandas_phi()
BINS       = bandas_para_bins(BANDAS, len(sinal_base))

print(f"\n[1] Convergência ({N_CICLOS} ciclos, idêntico ao original)...")
nb   = len(BINS)
beta = np.ones(nb); bm = beta.copy()
wm, wn = 1.0/PHI, 1.0-1.0/PHI
limiar = 0.99 * PHI**3
historico = []; ciclo_conv = N_CICLOS
passo5 = sinal_base.copy(); coh_final = np.zeros(nb)

for ciclo in range(N_CICLOS):
    passos, cohs = cascata_eq(sinal_base, beta, BINS)
    cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
    ba   = PHI**(3*cr)
    beta = wn*ba + wm*bm; bm = beta.copy()
    beta = np.clip(beta, 0.05, PHI**3)
    historico.append(beta.copy())
    passo5    = passos[-1]
    coh_final = cohs
    if beta.max() >= limiar and ciclo_conv == N_CICLOS:
        ciclo_conv = ciclo + 1
        print(f"    Campo harmônico formado no ciclo {ciclo_conv}/{N_CICLOS}")
        print(f"    β máx={beta.max():.4f}  (φ³={PHI**3:.4f})")

print(f"    Ponto de dobra 5 — AutoCorr e EntrEsp calculados abaixo.")

# ── Fase 2: extensão 10 minutos em chunks de 1.5s ─────────────────────────────
N_CHUNKS = int(np.ceil(EXTENSAO / DURACAO))
print(f"\n[2] Estendendo campo por {EXTENSAO/60:.0f} minutos")
print(f"    {N_CHUNKS} chunks de {DURACAO}s  |  coh_mem propagado entre chunks")

FADE_N = int(0.03 * FS)   # 30ms de fade entre chunks
chunks = []
cm     = coh_final.copy()

for i in range(N_CHUNKS):
    chunk     = gerar_beep(DURACAO)
    BINS_C    = bandas_para_bins(BANDAS, len(chunk))
    b_c = beta[:len(BINS_C)]
    if len(b_c) < len(BINS_C):
        b_c = np.pad(b_c, (0, len(BINS_C)-len(b_c)), constant_values=PHI**3)
    cm_c = cm[:len(BINS_C)]
    if len(cm_c) < len(BINS_C):
        cm_c = np.pad(cm_c, (0, len(BINS_C)-len(cm_c)), constant_values=0.0)
    passos_c, cohs_c = cascata_eq(chunk, b_c, BINS_C, coh_mem_init=cm_c)
    chunks.append(normalizar(passos_c[-1]))
    cm = cohs_c
    if (i+1) % 40 == 0:
        print(f"    chunk {i+1}/{N_CHUNKS}  ({(i+1)*DURACAO/60:.1f}min)  "
              f"coh_mem médio={cohs_c.mean():.4f}")

# Montar extensão com fade entre chunks
print("\n[3] Montando áudio final...")
extensao = chunks[0].copy()
for c in chunks[1:]:
    f = min(FADE_N, len(extensao), len(c))
    fade = np.linspace(0, 1, f)
    extensao[-f:] = extensao[-f:]*(1-fade) + c[:f]*fade
    extensao = np.concatenate([extensao, c[f:]])
extensao = normalizar(extensao[:int(EXTENSAO * FS)])

# Fade entre ponto5 e extensão
f = min(FADE_N, len(passo5), len(extensao))
fade = np.linspace(0, 1, f)
p5_fade = passo5.copy()
p5_fade[-f:] = passo5[-f:]*(1-fade) + extensao[:f]*fade
sequencia = normalizar(np.concatenate([p5_fade, extensao[f:]]))

dur_min = len(sequencia)/FS/60
print(f"    Áudio final: {dur_min:.1f} minutos")

# ── Métricas ──────────────────────────────────────────────────────────────────
def autocorr(s):
    s=s-s.mean(); return float(np.corrcoef(s[:-1],s[1:])[0,1])
def entr_esp(s):
    mag=np.abs(np.fft.rfft(s))**2; p=mag/(mag.sum()+1e-12); p=p[p>1e-15]
    return float(-np.sum(p*np.log2(p)))

print(f"\n{'='*55}")
print(f"{'Fase':<28} {'AutoCorr':>9} {'EntrEsp':>9}")
print("-"*55)
for nome, sig in [
    ("Beep bruto",              sinal_base),
    ("Ponto de dobra 5",        passo5),
    ("Extensão (primeiros 1.5s)", extensao[:len(sinal_base)]),
]:
    print(f"{nome:<28} {autocorr(sig):>9.4f} {entr_esp(sig):>9.4f}")
print(f"{'Alvo':<28} {'1.0000':>9} {'0.0601':>9}")

# ── Salvar ────────────────────────────────────────────────────────────────────
wavfile.write("beep880_campo_10min.wav", FS, (sequencia * 32767).astype(np.int16))
print(f"\nSalvo: beep880_campo_10min.wav  ({dur_min:.1f} min)")

# ── Player ────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"▶ ECO BEEP 880 — Campo Harmônico {dur_min:.0f} minutos")
print("  (inicia no ponto de dobra 5 — campo já formado)")
display(Audio(sequencia, rate=FS))

print("\nOBSERVE:")
print("  A sensação do ponto de dobra 5 se sustenta nos 10 minutos?")
print("  O campo é contínuo ou há quebras entre os chunks?")
