# © Vitor Edson Delavi · Florianópolis · 2026
# TESTE 4B — Agente Observador Adaptativo aplicado ao Violão
#
# Agente de 2ª ordem: o violão informa seus próprios parâmetros.
#   1. Campo global do violão → β inicial
#   2. Distribuição espectral do violão → α_banda (acoplamento por banda)
#   3. Autocorrelação natural do violão → coh_mem_0 aquecido
#   4. Duração do violão → n_ciclos adaptativo
#
# Convergência nos primeiros 8s — campo aplicado ao áudio completo.
# Ferramenta ORIGINAL preservada integralmente.
#
# Rode no Google Colab.

import urllib.request
import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display
import matplotlib.pyplot as plt

BASE = "https://raw.githubusercontent.com/vitoredsonalphaphi/alpha_phi_manifesto/claude/good-morning-N6f3S/"

print("Baixando violão...")
urllib.request.urlretrieve(BASE + "violao_20260513.wav", "violao_20260513.wav")
rate, data = wavfile.read("violao_20260513.wav")
sinal_completo = data.astype(np.float64) / 32768.0
sinal_completo = sinal_completo / (np.max(np.abs(sinal_completo)) + 1e-10)
print(f"Áudio: {len(sinal_completo)/rate:.1f}s  |  {rate}Hz\n")

PHI     = (1 + np.sqrt(5)) / 2
N_STEPS = 5
N_CICLOS = 20

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=None):
    if f_max is None: f_max = rate / 2
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max: break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    return [(max(0, int(f_lo * n / rate)),
             min(int(f_hi * n / rate) + 1, n // 2 + 1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

# ── Funções ORIGINAIS — não modificadas ──────────────────────────────────────
def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0/PHI, 1.0 - 1.0/PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb  = F[b_lo:b_hi]
        mag = np.abs(Fb); phase = np.angle(Fb)
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
        cm = cohs; se = normalizar(se)
        passos.append(se.copy()); s = se.copy()
    return s, cohs

# ── eco_eq adaptativo (α_banda por banda) ────────────────────────────────────
def eco_eq_adaptativo(x, bins_phi, beta_bands, alpha_banda, coh_mem=None):
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
        env = np.clip(1.0+(ai*ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag*env)*np.exp(1j*phase)
    r = np.fft.irfft(F_out, n=N)
    return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

def cascata_eq_adaptativa(sinal, beta_bands, bins_phi, alpha_banda, coh_mem_0):
    s  = sinal.copy()
    cm = coh_mem_0.copy()
    passos = []
    for _ in range(N_STEPS):
        se, cohs = eco_eq_adaptativo(s, bins_phi, beta_bands, alpha_banda, cm)
        cm = cohs; se = normalizar(se)
        passos.append(se.copy()); s = se.copy()
    return s, cohs

# ── Observação de 2ª ordem ────────────────────────────────────────────────────
def observar_2a_ordem(sinal, bins_phi, fs):
    N     = len(sinal)
    dur_s = N / fs
    F     = np.fft.rfft(sinal)

    # 1. Duração → n_ciclos (log φ-proporcional, mín 20)
    n_ciclos_max = max(N_CICLOS,
                       int(round(N_CICLOS * (1.0 + np.log(max(dur_s,1.5)/1.5)/np.log(PHI)/PHI))))
    print(f"  [obs 1] duração={dur_s:.1f}s → n_ciclos_max={n_ciclos_max}")

    # 2. Distribuição espectral → α_banda
    energias = np.array([np.sum(np.abs(F[b_lo:b_hi])**2) for b_lo, b_hi, _, _ in bins_phi])
    alpha_banda = 0.1 + 0.9 * energias / (energias.max() + 1e-10)
    dom = int(np.argmax(energias))
    print(f"  [obs 2] α_banda — mín={alpha_banda.min():.3f} máx={alpha_banda.max():.3f}  "
          f"banda dominante: {bins_phi[dom][2]:.0f}–{bins_phi[dom][3]:.0f}Hz")

    # 3. Autocorrelação natural → coh_mem_0
    coh_mem_0 = []
    for b_lo, b_hi, _, _ in bins_phi:
        Fb = F[b_lo:b_hi]
        if len(Fb) < 2: coh_mem_0.append(0.0); continue
        mag = np.abs(Fb)
        an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        coh_mem_0.append(coh)
    coh_mem_0 = np.array(coh_mem_0)
    print(f"  [obs 3] coh_mem_0 — médio={coh_mem_0.mean():.4f} máx={coh_mem_0.max():.4f}")

    return n_ciclos_max, alpha_banda, coh_mem_0

def agente_eco_adaptativo(sinal, bins_phi, beta_init, n_ciclos_max, alpha_banda, coh_mem_0):
    beta = beta_init.copy(); bm = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    limiar = 0.99 * PHI**3
    historico, ciclo_conv = [], n_ciclos_max
    for ciclo in range(n_ciclos_max):
        _, cohs = cascata_eq_adaptativa(sinal, beta, bins_phi, alpha_banda, coh_mem_0)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        historico.append(beta.copy())
        if beta.max() >= limiar and ciclo_conv == n_ciclos_max:
            ciclo_conv = ciclo + 1
            break
    return beta, historico, ciclo_conv

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

# ─────────────────────────────────────────────────────────────────────────────
BANDAS         = gerar_bandas_phi()
N_CONV_S       = 8
N_conv         = min(int(N_CONV_S * rate), len(sinal_completo))
sinal_conv     = sinal_completo[:N_conv]
BINS_CONV      = bandas_para_bins(BANDAS, N_conv)
BINS_COMPLETO  = bandas_para_bins(BANDAS, len(sinal_completo))

# ── FASE 1: Campo global → β inicial ─────────────────────────────────────────
print("=" * 58)
print("FASE 1 — Campo global do violão (β inicial)")
print("=" * 58)
F_global = np.fft.rfft(sinal_completo)
cohs_g   = []
for b_lo, b_hi, _, _ in BINS_COMPLETO:
    Fb  = F_global[b_lo:b_hi]; mag = np.abs(Fb)
    an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
    cohs_g.append(float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2))))
cohs_g    = np.array(cohs_g)
beta_glob = np.clip(PHI**(3*cohs_g), 0.05, PHI**3)
print(f"β inicial (campo global) — médio: {beta_glob.mean():.4f}  máx: {beta_glob.max():.4f}")

# Ajustar beta_glob para bins de convergência
beta_init_conv = beta_glob[:len(BINS_CONV)]
if len(beta_init_conv) < len(BINS_CONV):
    beta_init_conv = np.pad(beta_init_conv, (0, len(BINS_CONV)-len(beta_init_conv)),
                            constant_values=1.0)

# ── FASE 2: Observação 2ª ordem + convergência (8s) ──────────────────────────
print(f"\n{'='*58}")
print(f"FASE 2 — Observação de 2ª ordem + convergência ({N_CONV_S}s)")
print("=" * 58)
n_max, alpha_conv, coh_mem_0 = observar_2a_ordem(sinal_conv, BINS_CONV, rate)

print(f"\n[>] Agente adaptativo (primeiros {N_CONV_S}s)...")
beta_conv, hist_conv, ciclo_conv = agente_eco_adaptativo(
    sinal_conv, BINS_CONV, beta_init_conv, n_max, alpha_conv, coh_mem_0)
print(f"    β convergido — médio: {beta_conv.mean():.4f}  máx: {beta_conv.max():.4f}")
print(f"    Convergiu no ciclo: {ciclo_conv}/{n_max}  "
      f"parada antecipada: {'sim' if ciclo_conv < n_max else 'não'}")

# ── FASE 3: Referência T3 (sem adaptativo) para comparação ───────────────────
print(f"\n{'='*58}")
print("FASE 3 — Referência: campo global sem adaptativo (T3)")
print("=" * 58)

def agente_eco_informado(sinal, bins_phi, beta_init, n_ciclos=20):
    beta = beta_init.copy(); bm = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    historico = []
    for _ in range(n_ciclos):
        _, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        historico.append(beta.copy())
    return beta, historico

beta_t3, hist_t3 = agente_eco_informado(sinal_conv, BINS_CONV, beta_init_conv)
limiar_t3 = 0.99 * PHI**3
ciclo_t3  = next((i+1 for i,h in enumerate(hist_t3) if h.max() >= limiar_t3), 20)
print(f"T3 β convergido — médio: {beta_t3.mean():.4f}  máx: {beta_t3.max():.4f}  ciclo: {ciclo_t3}/20")

# ── FASE 4: Aplicar ao áudio completo ────────────────────────────────────────
print(f"\n{'='*58}")
print("FASE 4 — Aplicando campo ao violão completo")
print("=" * 58)

# T3 aplicado ao completo
beta_t3_full = beta_t3[:len(BINS_COMPLETO)]
if len(beta_t3_full) < len(BINS_COMPLETO):
    beta_t3_full = np.pad(beta_t3_full, (0, len(BINS_COMPLETO)-len(beta_t3_full)),
                          constant_values=PHI**3)
sinal_t3, _ = cascata_eq(sinal_completo, beta_t3_full, BINS_COMPLETO)
sinal_t3 = normalizar(sinal_t3)
print("T3 aplicado ao violão completo.")

# T4 adaptativo aplicado ao completo
beta_conv_full = beta_conv[:len(BINS_COMPLETO)]
if len(beta_conv_full) < len(BINS_COMPLETO):
    beta_conv_full = np.pad(beta_conv_full, (0, len(BINS_COMPLETO)-len(beta_conv_full)),
                            constant_values=PHI**3)
alpha_full = alpha_conv[:len(BINS_COMPLETO)]
if len(alpha_full) < len(BINS_COMPLETO):
    alpha_full = np.pad(alpha_full, (0, len(BINS_COMPLETO)-len(alpha_full)),
                        constant_values=1.0)
coh_mem_full = coh_mem_0[:len(BINS_COMPLETO)]
if len(coh_mem_full) < len(BINS_COMPLETO):
    coh_mem_full = np.pad(coh_mem_full, (0, len(BINS_COMPLETO)-len(coh_mem_full)),
                          constant_values=0.0)

sinal_t4, _ = cascata_eq_adaptativa(sinal_completo, beta_conv_full, BINS_COMPLETO,
                                     alpha_full, coh_mem_full)
sinal_t4 = normalizar(sinal_t4)
print("T4 adaptativo aplicado ao violão completo.")

# ── Métricas ──────────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"{'Sinal':<32} {'AutoCorr':>9} {'EntrEsp':>9} {'Bits':>6}")
print("-"*58)
for nome, sig in [
    ("Violão original",          sinal_completo),
    ("T3 — campo global",        sinal_t3),
    ("T4 — agente adaptativo",   sinal_t4),
]:
    print(f"{nome:<32} {autocorr(sig):>9.4f} {entr_esp(sig):>9.4f} {bits_ef(sig):>6.4f}")
print(f"{'Ref. ponto dobra 5 (beep)':<32} {'1.0000':>9} {'0.0601':>9} {'7.8931':>6}  ← alvo")

# ── Salvar ────────────────────────────────────────────────────────────────────
wavfile.write("violao_original.wav",  rate, (sinal_completo * 32767).astype(np.int16))
wavfile.write("violao_t3_global.wav", rate, (sinal_t3       * 32767).astype(np.int16))
wavfile.write("violao_t4_adapt.wav",  rate, (sinal_t4       * 32767).astype(np.int16))

# ── Players ───────────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print("▶ 1. VIOLÃO ORIGINAL")
display(Audio("violao_original.wav", rate=rate))

print("▶ 2. T3 — campo global informado")
display(Audio("violao_t3_global.wav", rate=rate))

print("▶ 3. T4 — agente adaptativo (obs. 2ª ordem)")
display(Audio("violao_t4_adapt.wav", rate=rate))

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Convergência β
ciclos_t3 = range(1, len(hist_t3)+1)
ciclos_t4 = range(1, len(hist_conv)+1)
axes[0].plot(ciclos_t3, [h.max() for h in hist_t3],   'o--', color='#00aaff', lw=1.5,
             label=f'T3 global (ciclo {ciclo_t3})')
axes[0].plot(ciclos_t4, [h.max() for h in hist_conv],  'o-',  color='#00cc66', lw=1.5,
             label=f'T4 adaptativo (ciclo {ciclo_conv})')
axes[0].axhline(PHI**3, color='gold', lw=1.5, ls=':', label=f'φ³={PHI**3:.4f}')
axes[0].set_title(f"Convergência β — violão ({N_CONV_S}s)")
axes[0].set_xlabel("Ciclo"); axes[0].set_ylabel("β máximo")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# α_banda do violão
axes[1].bar(range(len(alpha_conv)), alpha_conv, color='#00cc66', alpha=0.7)
axes[1].set_title("α_banda — acoplamento derivado do violão")
axes[1].set_xlabel("Banda φ"); axes[1].set_ylabel("α_banda"); axes[1].grid(alpha=0.2)

# Forma de onda comparada
t = np.linspace(0, len(sinal_completo)/rate, len(sinal_completo))
axes[2].plot(t, sinal_completo, color='gray',    lw=0.3, alpha=0.6, label='Original')
axes[2].plot(t, sinal_t4,       color='#00cc66', lw=0.3, alpha=0.8, label='T4 adaptativo')
axes[2].set_title("Forma de onda — original vs T4")
axes[2].set_xlabel("s"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.2)

plt.tight_layout()
plt.savefig("teste4b_violao.png", dpi=120, bbox_inches='tight')
plt.show()

print("\nOBSERVE:")
print("  1. O violão T4 tem campo harmônico desde o início?")
print("  2. T4 soa diferente de T3?")
print("  3. A identidade do violão (timbre, harmônicos) foi preservada?")
print(f"  4. O agente convergiu no ciclo {ciclo_conv}/{n_max} — a parada antecipada funcionou?")
