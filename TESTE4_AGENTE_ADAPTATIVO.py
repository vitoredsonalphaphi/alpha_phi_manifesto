# © Vitor Edson Delavi · Florianópolis · 2026
# TESTE 4 — Agente Observador Adaptativo (Chave 09)
#
# Implementação da observação de 2ª ordem:
#   sinal → o agente deriva seus próprios parâmetros
#
# Observações implementadas:
#   1. Duração → n_ciclos proporcional (escala φ, parada antecipada)
#   2. Distribuição espectral → acoplamento por banda (α_banda)
#   3. Autocorrelação natural → coh_mem inicial aquecido
#
# Ferramenta ORIGINAL preservada integralmente.
# Novas funções adicionadas: observar_2a_ordem,
# eco_eq_adaptativo, cascata_eq_adaptativo, agente_eco_adaptativo.
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

# ── Observação de 2ª ordem ────────────────────────────────────────────────────
def observar_2a_ordem(sinal, bins_phi, fs):
    """Deriva parâmetros do próprio sinal — Chave 09, Seção III."""
    N     = len(sinal)
    dur_s = N / fs
    F     = np.fft.rfft(sinal)

    # 1. Duração → n_ciclos (escala logarítmica φ-proporcional)
    #    ECO BEEP 880: 1.5s → 20 ciclos (referência)
    #    Para outros sinais: cresce com log na base φ da proporção de duração
    if dur_s <= 1.5:
        n_ciclos_max = N_CICLOS
    else:
        n_ciclos_max = max(N_CICLOS,
                           int(round(N_CICLOS * (1.0 + np.log(dur_s/1.5)/np.log(PHI)/PHI))))
    print(f"    [obs 1] duração={dur_s:.2f}s → n_ciclos_max={n_ciclos_max}")

    # 2. Distribuição espectral → α_banda (acoplamento por banda)
    #    Banda com mais energia do sinal → campo se acopla mais nela
    energias = np.array([
        np.sum(np.abs(F[b_lo:b_hi])**2)
        for b_lo, b_hi, _, _ in bins_phi
    ])
    e_total = energias.sum() + 1e-10
    # Mapeia [0.0, 1.0] de energia relativa → α_banda em [0.1, 1.0]
    alpha_banda = 0.1 + 0.9 * energias / (energias.max() + 1e-10)
    print(f"    [obs 2] α_banda — mín: {alpha_banda.min():.4f}  máx: {alpha_banda.max():.4f}  "
          f"(banda dominante em {bins_phi[int(np.argmax(energias))][2]:.0f}–"
          f"{bins_phi[int(np.argmax(energias))][3]:.0f}Hz)")

    # 3. Autocorrelação natural → coh_mem inicial aquecido
    #    Cada banda inicia coh_mem com a coerência espectral já observada
    coh_mem_0 = []
    for b_lo, b_hi, _, _ in bins_phi:
        Fb  = F[b_lo:b_hi]
        if len(Fb) < 2:
            coh_mem_0.append(0.0); continue
        mag = np.abs(Fb)
        an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        coh_mem_0.append(coh)
    coh_mem_0 = np.array(coh_mem_0)
    print(f"    [obs 3] coh_mem_0 — médio: {coh_mem_0.mean():.4f}  máx: {coh_mem_0.max():.4f}")

    return n_ciclos_max, alpha_banda, coh_mem_0

# ── Funções adaptativas ───────────────────────────────────────────────────────
def eco_eq_adaptativo(x, bins_phi, beta_bands, alpha_banda, coh_mem=None):
    """eco_eq com acoplamento por banda derivado do sinal (observação 2ª ordem)."""
    beta_bands  = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    alpha_banda = np.atleast_1d(np.asarray(alpha_banda, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0/PHI, 1.0 - 1.0/PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi  = float(beta_bands[i])  if i < len(beta_bands)  else 1.0
        ai  = float(alpha_banda[i]) if i < len(alpha_banda) else 1.0
        Fb  = F[b_lo:b_hi]
        mag = np.abs(Fb); phase = np.angle(Fb)
        an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        ce  = (wn*coh + wm*float(coh_mem[i])
               if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk  = np.arange(len(Fb))
        # Diferença central: ai (α_banda) modula o acoplamento por banda
        env = np.clip(1.0+(ai*ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag*env)*np.exp(1j*phase)
    r = np.fft.irfft(F_out, n=N)
    return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

def cascata_eq_adaptativa(sinal, beta_bands, bins_phi, alpha_banda, coh_mem_0):
    s  = sinal.copy()
    cm = coh_mem_0.copy()   # aquecido — não inicia em zeros
    passos = []
    for _ in range(N_STEPS):
        se, cohs = eco_eq_adaptativo(s, bins_phi, beta_bands, alpha_banda, cm)
        cm = cohs; se = normalizar(se)
        passos.append(se.copy()); s = se.copy()
    return passos, cohs

def agente_eco_adaptativo(sinal, bins_phi, beta_init, n_ciclos_max, alpha_banda, coh_mem_0):
    """Agente com parâmetros derivados do sinal — parada antecipada por convergência."""
    nb   = len(bins_phi)
    beta = beta_init.copy()
    bm   = beta.copy()
    wm, wn  = 1.0/PHI, 1.0-1.0/PHI
    limiar  = 0.99 * PHI**3
    historico, ciclo_conv = [], n_ciclos_max
    for ciclo in range(n_ciclos_max):
        passos, cohs = cascata_eq_adaptativa(sinal, beta, bins_phi, alpha_banda, coh_mem_0)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        historico.append(beta.copy())
        # Parada antecipada — o sinal diz quando convergiu
        if beta.max() >= limiar and ciclo_conv == n_ciclos_max:
            ciclo_conv = ciclo + 1
            break   # co-ressonância atingida — não processa além
    return beta, passos[-1], historico, ciclo_conv

# ── Pipeline ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("TESTE 4 — Agente Observador Adaptativo (Chave 09)")
print(f"F_BEEP={F_BEEP}Hz  F_ORG={F_ORG}Hz  α={ALPHA}  φ={PHI:.6f}")
print("=" * 60)

sinal_base = gerar_beep(DURACAO)
BANDAS     = gerar_bandas_phi()
BINS       = bandas_para_bins(BANDAS, len(sinal_base))

# ── Referência: agente original (β=1.0, sem informação) ──────────────────────
print("\n[REF] Agente original (β=1.0, N_CICLOS=20, sem observação)...")
beta_ref, passo5_ref, hist_ref = agente_eco(sinal_base, BINS, beta_init=None)
limiar  = 0.99 * PHI**3
conv_ref = next((i+1 for i,h in enumerate(hist_ref) if h.max() >= limiar), 20)
print(f"      β convergido — médio: {beta_ref.mean():.4f}  máx: {beta_ref.max():.4f}")
print(f"      Convergiu no ciclo: {conv_ref}/20")

# ── Referência: agente com β global (TESTE 3) ─────────────────────────────────
print("\n[T3]  Agente com β inicializado pelo campo global (TESTE 3)...")
F_glob = np.fft.rfft(sinal_base)
cohs_g = []
for b_lo, b_hi, _, _ in BINS:
    Fb = F_glob[b_lo:b_hi]; mag = np.abs(Fb)
    an = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
    cohs_g.append(float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2))))
beta_t3   = np.clip(PHI**(3*np.array(cohs_g)), 0.05, PHI**3)
beta_t3c, passo5_t3, hist_t3 = agente_eco(sinal_base, BINS, beta_init=beta_t3)
conv_t3 = next((i+1 for i,h in enumerate(hist_t3) if h.max() >= limiar), 20)
print(f"      β convergido — médio: {beta_t3c.mean():.4f}  máx: {beta_t3c.max():.4f}")
print(f"      Convergiu no ciclo: {conv_t3}/20")

# ── TESTE 4: Observação de 2ª ordem ──────────────────────────────────────────
print("\n[T4]  Observação de 2ª ordem...")
n_ciclos_max, alpha_banda, coh_mem_0 = observar_2a_ordem(sinal_base, BINS, FS)

print("\n[T4]  Agente adaptativo (β informado + α_banda + coh_mem aquecido)...")
beta_t4, passo5_t4, hist_t4, conv_t4 = agente_eco_adaptativo(
    sinal_base, BINS,
    beta_init     = beta_t3,       # β inicial do campo global (Chave 09: acumulativo)
    n_ciclos_max  = n_ciclos_max,
    alpha_banda   = alpha_banda,
    coh_mem_0     = coh_mem_0
)
print(f"      β convergido — médio: {beta_t4.mean():.4f}  máx: {beta_t4.max():.4f}")
print(f"      Convergiu no ciclo: {conv_t4}/{n_ciclos_max}  "
      f"(parada antecipada: {'sim' if conv_t4 < n_ciclos_max else 'não'})")

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

print(f"\n{'='*62}")
print(f"{'Sinal':<36} {'AutoCorr':>9} {'EntrEsp':>9} {'Bits':>6}")
print("-"*62)
for nome, sig in [
    ("Beep bruto",                     sinal_base),
    ("REF — original β=1.0",           passo5_ref),
    ("T3  — β global informado",       passo5_t3),
    ("T4  — agente adaptativo",        passo5_t4),
]:
    print(f"{nome:<36} {autocorr(sig):>9.4f} {entr_esp(sig):>9.4f} {bits_ef(sig):>6.4f}")
print(f"{'Alvo (ECO BEEP 880 ponto 5)':<36} {'1.0000':>9} {'0.0601':>9} {'7.8931':>6}")

# ── Salvar ────────────────────────────────────────────────────────────────────
wavfile.write("t4_beep_bruto.wav",    FS, (sinal_base   * 32767).astype(np.int16))
wavfile.write("t4_ref_original.wav",  FS, (passo5_ref   * 32767).astype(np.int16))
wavfile.write("t4_t3_global.wav",     FS, (passo5_t3    * 32767).astype(np.int16))
wavfile.write("t4_adaptativo.wav",    FS, (passo5_t4    * 32767).astype(np.int16))

# ── Players ───────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("▶ 1. BEEP BRUTO — sem campo")
display(Audio("t4_beep_bruto.wav", rate=FS))

print("▶ 2. REF — agente original (β=1.0, 20 ciclos)")
display(Audio("t4_ref_original.wav", rate=FS))

print("▶ 3. T3 — β global informado (referência TESTE 3)")
display(Audio("t4_t3_global.wav", rate=FS))

print("▶ 4. T4 — agente adaptativo (obs. 2ª ordem)")
display(Audio("t4_adaptativo.wav", rate=FS))

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Convergência β
ciclos_ref = range(1, len(hist_ref)+1)
ciclos_t3  = range(1, len(hist_t3)+1)
ciclos_t4  = range(1, len(hist_t4)+1)

axes[0].plot(ciclos_ref, [h.max() for h in hist_ref], 'o--', color='gray',    lw=1.5, label=f'REF β=1.0 (ciclo {conv_ref})')
axes[0].plot(ciclos_t3,  [h.max() for h in hist_t3],  'o-',  color='#00aaff', lw=1.5, label=f'T3 β global (ciclo {conv_t3})')
axes[0].plot(ciclos_t4,  [h.max() for h in hist_t4],  'o-',  color='#00cc66', lw=1.5, label=f'T4 adaptativo (ciclo {conv_t4})')
axes[0].axhline(PHI**3, color='gold', lw=1.5, ls=':', label=f'φ³={PHI**3:.4f}')
axes[0].set_title("Convergência β — comparativo")
axes[0].set_xlabel("Ciclo"); axes[0].set_ylabel("β máximo")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# α por banda (observação 2ª ordem)
x_b = range(len(alpha_banda))
axes[1].bar(x_b, alpha_banda, color='#00cc66', alpha=0.7)
axes[1].axhline(ALPHA, color='gold', lw=1.5, ls=':', label=f'α original={ALPHA:.4f}')
axes[1].set_title("α_banda — acoplamento derivado do sinal")
axes[1].set_xlabel("Banda φ"); axes[1].set_ylabel("α_banda")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2)

# coh_mem_0 (observação 3ª — autocorrelação natural)
axes[2].bar(x_b, coh_mem_0, color='#00aaff', alpha=0.7)
axes[2].set_title("coh_mem inicial — coerência natural por banda")
axes[2].set_xlabel("Banda φ"); axes[2].set_ylabel("coh_mem_0")
axes[2].grid(alpha=0.2)

plt.tight_layout()
plt.savefig("teste4_agente_adaptativo.png", dpi=120, bbox_inches='tight')
plt.show()

print("\nOBSERVE:")
print(f"  1. O agente T4 convergiu no ciclo {conv_t4} (parada antecipada pelo sinal)?")
print("  2. O áudio 4 soa diferente dos áudios 2 e 3?")
print("  3. As métricas do T4 se aproximam do alvo (AutoCorr→1, EntrEsp→0.06)?")
print("  4. O α_banda preserva a identidade espectral do beep original?")
