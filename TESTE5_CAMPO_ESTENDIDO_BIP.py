# © Vitor Edson Delavi · Florianópolis · 2026
# TESTE 5 — ECO BEEP 880 com campo estendido 10s
#
# Processo IDÊNTICO ao teste original do BIP 880:
#   F_BEEP=880Hz, F_ORG=220Hz, α=1/3, FM-φ, N_STEPS=5, N_CICLOS=20
#   β começa em 1.0, converge para φ³ em 20 ciclos
#
# Novidade (expertise do violão):
#   Quando o campo forma no ponto de dobra 5 (ciclo convergido),
#   β_conv é aplicado em chunks de 1.5s com coh_mem propagado entre eles.
#   Isso mantém a continuidade temporal do campo — não congela β em φ³.
#
# Saída:
#   1. Beep bruto (1.5s) — referência
#   2. Ponto de dobra 5 (1.5s) — campo máximo
#   3. Extensão 10s — campo sustentado chunk a chunk
#
# Rode no Google Colab.

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display
import matplotlib.pyplot as plt

# ── Constantes ORIGINAIS — não modificadas ───────────────────────────────────
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
FADE     = int(0.05 * FS)   # fade curto entre segmentos

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

# ── Funções ORIGINAIS — não modificadas ──────────────────────────────────────
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

def agente_eco(sinal, bins_phi):
    """Agente ORIGINAL — idêntico ao teste original do BIP 880."""
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    historico = []
    ultimo_passo5 = sinal.copy()
    ultimo_cohs   = np.zeros(nb)
    for _ in range(N_CICLOS):
        passos, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        historico.append(beta.copy())
        ultimo_passo5 = passos[-1]
        ultimo_cohs   = cohs
    return beta, ultimo_passo5, historico, ultimo_cohs

def gerar_beep(duracao):
    N = int(FS * duracao)
    t = np.linspace(0, duracao, N, endpoint=False)
    beep = np.sign(np.sin(2*np.pi*F_BEEP*t))
    fm   = np.sin(2*np.pi*F_M*t + BETA_FM*np.sin(2*np.pi*F_M*t/PHI))
    return normalizar(ALPHA*beep + (1-ALPHA)*fm)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 58)
print("TESTE 5 — ECO BEEP 880 + Campo Estendido 10s")
print(f"F_BEEP={F_BEEP}Hz  F_ORG={F_ORG}Hz  α={ALPHA:.4f}  φ³={PHI**3:.4f}")
print("=" * 58)

sinal_base = gerar_beep(DURACAO)
BANDAS     = gerar_bandas_phi()
BINS       = bandas_para_bins(BANDAS, len(sinal_base))

# ── Fase 1: Convergência IDÊNTICA ao original ─────────────────────────────────
print(f"\nFase 1 — Convergência original (β=1.0, {N_CICLOS} ciclos)...")
beta_conv, passo5, hist, cohs_conv = agente_eco(sinal_base, BINS)

limiar    = 0.99 * PHI**3
ciclo_c   = next((i+1 for i,h in enumerate(hist) if h.max() >= limiar), N_CICLOS)
print(f"β convergido — médio: {beta_conv.mean():.4f}  máx: {beta_conv.max():.4f}")
print(f"Convergiu no ciclo: {ciclo_c}/{N_CICLOS}")
print(f"Ponto de dobra 5 gerado — {DURACAO}s de campo máximo")

# ── Fase 2: Extensão 10s em chunks de 1.5s com coh_mem propagado ─────────────
print(f"\nFase 2 — Extensão 10s (chunks de {DURACAO}s, coh_mem propagado)...")

EXTENSAO   = 10.0
N_CHUNKS   = int(np.ceil(EXTENSAO / DURACAO))   # 7 chunks de 1.5s ≈ 10.5s
chunks_out = []
coh_mem_atual = cohs_conv.copy()   # inicia com coerência do ponto de dobra 5

for i in range(N_CHUNKS):
    t_ini = i * DURACAO
    chunk = gerar_beep(DURACAO)
    BINS_C = bandas_para_bins(BANDAS, len(chunk))

    # β_conv ajustado para o tamanho do chunk
    b_c = beta_conv[:len(BINS_C)]
    if len(b_c) < len(BINS_C):
        b_c = np.pad(b_c, (0, len(BINS_C)-len(b_c)), constant_values=PHI**3)

    # coh_mem_atual ajustado para o tamanho do chunk
    cm_c = coh_mem_atual[:len(BINS_C)]
    if len(cm_c) < len(BINS_C):
        cm_c = np.pad(cm_c, (0, len(BINS_C)-len(cm_c)), constant_values=0.0)

    # Cascata com β_conv e coh_mem propagado — não congela, não adapta
    passos_c, cohs_c = cascata_eq(chunk, b_c, BINS_C, coh_mem_init=cm_c)
    chunk_out = normalizar(passos_c[-1])
    chunks_out.append(chunk_out)
    coh_mem_atual = cohs_c   # propaga coh_mem para o próximo chunk
    print(f"  chunk {i+1}/{N_CHUNKS}  [{t_ini:.1f}–{t_ini+DURACAO:.1f}s]  "
          f"coh_mem médio: {cohs_c.mean():.4f}")

# Concatenar extensão com fade entre chunks
extensao = chunks_out[0].copy()
for c in chunks_out[1:]:
    fade = np.linspace(0, 1, min(FADE, len(extensao), len(c)))
    extensao[-len(fade):] = extensao[-len(fade):]*(1-fade) + c[:len(fade)]*fade
    extensao = np.concatenate([extensao, c[len(fade):]])
extensao = normalizar(extensao[:int(EXTENSAO * FS)])

print(f"Extensão gerada: {len(extensao)/FS:.1f}s")

# ── Sequência completa ────────────────────────────────────────────────────────
# fade entre passo5 e extensão
fade_t = np.linspace(0, 1, min(FADE, len(passo5), len(extensao)))
passo5_fade = passo5.copy()
passo5_fade[-len(fade_t):] = (passo5[-len(fade_t):]*(1-fade_t) +
                               extensao[:len(fade_t)]*fade_t)
sequencia = normalizar(np.concatenate([
    sinal_base,
    passo5_fade,
    extensao[len(fade_t):]
]))

dur_total = len(sequencia)/FS
print(f"\nSequência completa: {dur_total:.1f}s")
print(f"  [0.0 – {DURACAO:.1f}s]   beep bruto")
print(f"  [{DURACAO:.1f} – {DURACAO*2:.1f}s]   ponto de dobra 5 (campo máximo)")
print(f"  [{DURACAO*2:.1f} – {dur_total:.1f}s]  extensão com coh_mem propagado")

# ── Métricas ──────────────────────────────────────────────────────────────────
def autocorr(s):
    s=s-s.mean(); return float(np.corrcoef(s[:-1],s[1:])[0,1])
def entr_esp(s):
    mag=np.abs(np.fft.rfft(s))**2; p=mag/(mag.sum()+1e-12); p=p[p>1e-15]
    return float(-np.sum(p*np.log2(p)))
def bits_ef(s):
    c,_=np.histogram(s,bins=256); c=c[c>0]; p=c/c.sum()
    return float(-np.sum(p*np.log2(p)))

print(f"\n{'='*58}")
print(f"{'Fase':<30} {'AutoCorr':>9} {'EntrEsp':>9} {'Bits':>7}")
print("-"*58)
for nome, sig in [
    ("Beep bruto",         sinal_base),
    ("Ponto de dobra 5",   passo5),
    ("Extensão 10s",       extensao),
]:
    print(f"{nome:<30} {autocorr(sig):>9.4f} {entr_esp(sig):>9.4f} {bits_ef(sig):>7.4f}")

# ── Players ───────────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print("▶ 1. BEEP BRUTO — sem campo")
display(Audio(sinal_base, rate=FS))

print("▶ 2. PONTO DE DOBRA 5 — campo máximo (1.5s)")
display(Audio(passo5, rate=FS))

print("▶ 3. EXTENSÃO 10s — campo sustentado com coh_mem propagado")
display(Audio(extensao, rate=FS))

print("▶ 4. SEQUÊNCIA COMPLETA — bruto → dobra 5 → extensão")
display(Audio(sequencia, rate=FS))

# ── Salvar ────────────────────────────────────────────────────────────────────
wavfile.write("t5_beep_bruto.wav",   FS, (sinal_base * 32767).astype(np.int16))
wavfile.write("t5_ponto5.wav",       FS, (passo5     * 32767).astype(np.int16))
wavfile.write("t5_extensao_10s.wav", FS, (extensao   * 32767).astype(np.int16))
wavfile.write("t5_sequencia.wav",    FS, (sequencia  * 32767).astype(np.int16))

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

betas_max = [h.max() for h in hist]
axes[0].plot(range(1, N_CICLOS+1), betas_max, 'o-', color='#00cc66', lw=1.5)
axes[0].axhline(PHI**3, color='gold', lw=1.5, ls=':', label=f'φ³={PHI**3:.4f}')
axes[0].axvline(ciclo_c, color='white', lw=1, ls='--', label=f'convergência ciclo {ciclo_c}')
axes[0].set_title("Convergência β — ECO BEEP 880 (original)")
axes[0].set_xlabel("Ciclo"); axes[0].set_ylabel("β máximo")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

t_seq = np.linspace(0, len(sequencia)/FS, len(sequencia))
axes[1].plot(t_seq, sequencia, color='#00cc66', lw=0.4, alpha=0.85)
axes[1].axvline(DURACAO,   color='gold',  lw=1.5, ls='--', label='ponto de dobra 5')
axes[1].axvline(DURACAO*2, color='white', lw=1.5, ls='--', label='início extensão')
axes[1].set_title("Sequência completa — bruto → dobra 5 → extensão 10s")
axes[1].set_xlabel("Tempo (s)"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig("teste5_campo_estendido.png", dpi=120, bbox_inches='tight')
plt.show()

print("\nOBSERVE:")
print("  1. A sensação do ponto de dobra 5 se sustenta na extensão?")
print("  2. A extensão soa diferente do ponto de dobra 5?")
print("  3. Há diferença perceptível entre os chunks da extensão?")
print("  4. Compare com o TESTE 1 original — a extensão é mais fiel?")
