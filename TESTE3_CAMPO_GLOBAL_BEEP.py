# © Vitor Edson Delavi · Florianópolis · 2026
# TESTE 3 — Campo global informado aplicado ao ECO BEEP 880
#
# Ferramenta original preservada integralmente:
#   F_BEEP=880Hz, F_ORG=220Hz, α=1/3, FM-φ, eco_eq, cascata_eq, agente_eco
#
# Único acréscimo: antes do agente adaptive, analisa o espectro
# global do sinal completo e inicializa β com esse conhecimento.
# β não começa em 1.0 — começa informado pelo campo.
#
# Outputs:
#   1. Beep bruto        — sinal original sem processamento
#   2. Ponto de dobra 5  — com β inicializado pelo campo global
#   3. Campo estendido   — sinal longo (9.5s) processado desde o início
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
FADE     = int(0.15 * FS)

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

def cascata_eq(sinal, beta_bands, bins_phi):
    s = sinal.copy()
    cm = np.zeros(len(bins_phi))
    passos = []
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se)
        passos.append(se.copy()); s = se.copy()
    return passos, cohs

def agente_eco(sinal, bins_phi, beta_init=None):
    """Agente original + suporte a β inicial informado (opcional)."""
    nb = len(bins_phi)
    beta = beta_init.copy() if beta_init is not None else np.ones(nb)
    bm   = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    historico = []
    for _ in range(N_CICLOS):
        passos, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        historico.append(beta.copy())
    return beta, passos[-1], historico

# ── Geração do sinal ECO BEEP 880 ORIGINAL ───────────────────────────────────
def gerar_beep(duracao):
    N = int(FS * duracao)
    t = np.linspace(0, duracao, N, endpoint=False)
    beep = np.sign(np.sin(2*np.pi*F_BEEP*t))
    fm   = np.sin(2*np.pi*F_M*t + BETA_FM*np.sin(2*np.pi*F_M*t/PHI))
    return normalizar(ALPHA*beep + (1-ALPHA)*fm)

print("=" * 58)
print("TESTE 3 — Campo Global Informado no ECO BEEP 880")
print(f"F_BEEP={F_BEEP}Hz  F_ORG={F_ORG}Hz  α={ALPHA}  φ={PHI:.6f}")
print("=" * 58)

# Sinal base (1.5s — original)
sinal_base = gerar_beep(DURACAO)
BANDAS     = gerar_bandas_phi()
BINS       = bandas_para_bins(BANDAS, len(sinal_base))

# ── ACRÉSCIMO: Inicialização por campo global ─────────────────────────────────
print("\n[+] Analisando campo global do beep...")
F_glob = np.fft.rfft(sinal_base)
cohs_glob = []
for b_lo, b_hi, _, _ in BINS:
    Fb  = F_glob[b_lo:b_hi]
    mag = np.abs(Fb)
    an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
    coh = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
    cohs_glob.append(coh)
cohs_glob  = np.array(cohs_glob)
beta_init  = np.clip(PHI**(3*cohs_glob), 0.05, PHI**3)
print(f"    β inicial (global) — médio: {beta_init.mean():.4f}  máx: {beta_init.max():.4f}")
print(f"    β sem informação   — médio: 1.0000  máx: 1.0000")
print(f"    φ³ alvo            =        {PHI**3:.4f}")

# ── Agente com β informado ────────────────────────────────────────────────────
print("\n[>] Agente eco-φ com β inicializado pelo campo global...")
beta_conv, passo5, historico = agente_eco(sinal_base, BINS, beta_init=beta_init)

limiar = 0.99 * PHI**3
ciclo_conv = next((i+1 for i,h in enumerate(historico) if h.max() >= limiar), 20)
print(f"    β convergido — médio: {beta_conv.mean():.4f}  máx: {beta_conv.max():.4f}")
print(f"    Atingiu 99% de φ³ no ciclo: {ciclo_conv}/20")

# ── Agente SEM informação (referência — método original) ─────────────────────
print("\n[>] Agente original (β=1.0 inicial, referência)...")
beta_orig, passo5_orig, hist_orig = agente_eco(sinal_base, BINS, beta_init=None)
ciclo_orig = next((i+1 for i,h in enumerate(hist_orig) if h.max() >= limiar), 20)
print(f"    β convergido — médio: {beta_orig.mean():.4f}  máx: {beta_orig.max():.4f}")
print(f"    Atingiu 99% de φ³ no ciclo: {ciclo_orig}/20")

# ── Sinal longo: campo global aplicado desde o início (9.5s) ─────────────────
print("\n[>] Gerando sinal longo (9.5s) e aplicando campo desde o início...")
sinal_longo = gerar_beep(DURACAO + 8.0)   # 9.5s
BINS_L      = bandas_para_bins(BANDAS, len(sinal_longo))
beta_L      = np.clip(beta_conv[:len(BINS_L)], 0.05, PHI**3)
if len(beta_L) < len(BINS_L):
    beta_L = np.pad(beta_L, (0, len(BINS_L)-len(beta_L)), constant_values=PHI**3)
passos_L, _ = cascata_eq(sinal_longo, beta_L, BINS_L)
sinal_estendido = normalizar(passos_L[-1])
print(f"    Sinal longo processado: {len(sinal_estendido)/FS:.1f}s")

# ── Métricas ──────────────────────────────────────────────────────────────────
def autocorr(s):
    s = s-s.mean()
    return float(np.corrcoef(s[:-1], s[1:])[0,1])
def entr_esp(s):
    mag = np.abs(np.fft.rfft(s))**2
    p = mag/(mag.sum()+1e-12); p=p[p>1e-15]
    return float(-np.sum(p*np.log2(p)))
def bits_ef(s):
    c,_=np.histogram(s,bins=256); c=c[c>0]; p=c/c.sum()
    return float(-np.sum(p*np.log2(p)))

print(f"\n{'='*58}")
print(f"{'Sinal':<32} {'AutoCorr':>9} {'EntrEsp':>9} {'Bits':>7}")
print("-"*58)
for nome, sig in [
    ("Beep bruto",                    sinal_base),
    ("Ponto dobra 5 (original β=1)", passo5_orig),
    ("Ponto dobra 5 (β informado)",   passo5),
    ("Sinal longo 9.5s (β informado)",sinal_estendido[:len(sinal_base)]),
]:
    print(f"{nome:<32} {autocorr(sig):>9.4f} {entr_esp(sig):>9.4f} {bits_ef(sig):>7.4f}")

# ── Salvar ────────────────────────────────────────────────────────────────────
wavfile.write("t3_beep_bruto.wav",    FS, (sinal_base      * 32767).astype(np.int16))
wavfile.write("t3_ponto5_orig.wav",   FS, (passo5_orig     * 32767).astype(np.int16))
wavfile.write("t3_ponto5_inform.wav", FS, (passo5          * 32767).astype(np.int16))
wavfile.write("t3_campo_longo.wav",   FS, (sinal_estendido * 32767).astype(np.int16))

# ── Players ───────────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print("▶ 1. BEEP BRUTO — original sem campo")
display(Audio("t3_beep_bruto.wav", rate=FS))

print("▶ 2. PONTO DE DOBRA 5 — método original (β=1.0 inicial)")
display(Audio("t3_ponto5_orig.wav", rate=FS))

print("▶ 3. PONTO DE DOBRA 5 — β informado pelo campo global")
display(Audio("t3_ponto5_inform.wav", rate=FS))

print("▶ 4. CAMPO LONGO 9.5s — β informado, aplicado desde o início")
display(Audio("t3_campo_longo.wav", rate=FS))

# ── Gráfico convergência ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

b_max_orig = [h.max()  for h in hist_orig]
b_max_info = [h.max()  for h in historico]
ciclos     = range(1, N_CICLOS+1)

axes[0].plot(ciclos, b_max_orig, 'o--', color='gray',    lw=1.5, label=f'β=1.0 inicial (conv. ciclo {ciclo_orig})')
axes[0].plot(ciclos, b_max_info, 'o-',  color='#00cc66', lw=1.5, label=f'β informado (conv. ciclo {ciclo_conv})')
axes[0].axhline(PHI**3, color='gold', lw=1.5, ls=':', label=f'φ³={PHI**3:.4f}')
axes[0].set_title("Convergência β — original vs informado")
axes[0].set_xlabel("Ciclo"); axes[0].set_ylabel("β máximo")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

t_long = np.linspace(0, len(sinal_estendido)/FS, len(sinal_estendido))
axes[1].plot(t_long, sinal_estendido, color='#00cc66', lw=0.4, alpha=0.85)
axes[1].axvline(DURACAO, color='gold', lw=1.5, ls='--', label=f'fim do beep original ({DURACAO}s)')
axes[1].set_title("Campo longo 9.5s — β informado desde o início")
axes[1].set_xlabel("Tempo (s)"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig("teste3_campo_global_beep.png", dpi=120, bbox_inches='tight')
plt.show()

print("\nOBSERVE:")
print(f"  Ciclos para convergir — original: {ciclo_orig}  |  informado: {ciclo_conv}")
print("  O áudio 3 soa diferente do áudio 2?")
print("  O campo longo (áudio 4) mantém a sensação ergonômica do ponto 5?")
print("  Há 'dublagem' nos áudios 3 e 4?")
