# © Vitor Edson Delavi · Florianópolis · 2026
# Violão + Campo Harmônico φ — resultado definitivo
#
# Parâmetros validados pelos testes TESTE4 / TESTE4B:
#   N_STEPS = 5        — cascata completa (campo máximo)
#   wet     = 1/3      — acoplamento α original (co-ressonância validada)
#   β_init  = campo global do violão
#   α_banda = distribuição espectral do violão (obs. 2ª ordem)
#   coh_mem = coerência natural do violão (obs. 2ª ordem)
#   convergência nos primeiros 8s → campo aplicado ao áudio completo
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
sinal = data.astype(np.float64) / 32768.0
sinal = sinal / (np.max(np.abs(sinal)) + 1e-10)
print(f"Áudio: {len(sinal)/rate:.1f}s  |  {rate}Hz\n")

PHI     = (1 + np.sqrt(5)) / 2
N_STEPS = 5
WET     = 1.0 / 3.0     # acoplamento validado — proporção α original

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_max=None):
    if f_max is None: f_max = rate / 2
    bandas, f = [], 20.0
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

def eco_eq_adaptativo(x, bins_phi, beta_bands, alpha_banda, coh_mem=None):
    beta_bands  = np.atleast_1d(np.asarray(beta_bands,  dtype=float))
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

def cascata(sinal, beta, bins, alpha_banda, coh_mem_0):
    s, cm = sinal.copy(), coh_mem_0.copy()
    for _ in range(N_STEPS):
        se, cohs = eco_eq_adaptativo(s, bins, beta, alpha_banda, cm)
        cm = cohs; se = normalizar(se); s = se.copy()
    return s, cohs

def agente(sinal, bins, beta_init, alpha_banda, coh_mem_0, n_ciclos=20):
    beta = beta_init.copy(); bm = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    limiar = 0.99 * PHI**3
    historico, ciclo_conv = [], n_ciclos
    for ciclo in range(n_ciclos):
        _, cohs = cascata(sinal, beta, bins, alpha_banda, coh_mem_0)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        historico.append(beta.copy())
        if beta.max() >= limiar and ciclo_conv == n_ciclos:
            ciclo_conv = ciclo + 1
            break
    return beta, historico, ciclo_conv

# ── Observação do violão ──────────────────────────────────────────────────────
print("=" * 55)
print("Observando violão...")
print("=" * 55)
BANDAS        = gerar_bandas_phi()
BINS_COMPLETO = bandas_para_bins(BANDAS, len(sinal))

# Campo global → β inicial
F_glob  = np.fft.rfft(sinal)
cohs_g  = []
for b_lo, b_hi, _, _ in BINS_COMPLETO:
    Fb  = F_glob[b_lo:b_hi]; mag = np.abs(Fb)
    an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
    cohs_g.append(float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2))))
beta_glob = np.clip(PHI**(3*np.array(cohs_g)), 0.05, PHI**3)
print(f"β inicial — médio: {beta_glob.mean():.4f}  máx: {beta_glob.max():.4f}")

# Segmento de convergência (8s)
N_CONV    = min(int(8 * rate), len(sinal))
s_conv    = sinal[:N_CONV]
BINS_CONV = bandas_para_bins(BANDAS, N_CONV)

# α_banda e coh_mem_0 do violão
F_conv    = np.fft.rfft(s_conv)
energias  = np.array([np.sum(np.abs(F_conv[b_lo:b_hi])**2) for b_lo,b_hi,_,_ in BINS_CONV])
alpha_b   = 0.1 + 0.9 * energias / (energias.max() + 1e-10)
coh_m0    = []
for b_lo, b_hi, _, _ in BINS_CONV:
    Fb = F_conv[b_lo:b_hi]
    if len(Fb) < 2: coh_m0.append(0.0); continue
    mag = np.abs(Fb)
    an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
    coh_m0.append(float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2))))
coh_m0 = np.array(coh_m0)

beta_init = beta_glob[:len(BINS_CONV)]

# ── Convergência (8s) ─────────────────────────────────────────────────────────
print("\nConvergindo (8s)...")
beta_conv, hist, ciclo_conv = agente(s_conv, BINS_CONV, beta_init, alpha_b, coh_m0)
print(f"β convergido — médio: {beta_conv.mean():.4f}  máx: {beta_conv.max():.4f}  "
      f"ciclo: {ciclo_conv}")

# ── Campo ao sinal completo ───────────────────────────────────────────────────
print("\nAplicando campo ao violão completo...")
b_full = beta_conv[:len(BINS_COMPLETO)]
if len(b_full) < len(BINS_COMPLETO):
    b_full = np.pad(b_full, (0, len(BINS_COMPLETO)-len(b_full)), constant_values=PHI**3)
a_full = alpha_b[:len(BINS_COMPLETO)]
if len(a_full) < len(BINS_COMPLETO):
    a_full = np.pad(a_full, (0, len(BINS_COMPLETO)-len(a_full)), constant_values=1.0)
c_full = coh_m0[:len(BINS_COMPLETO)]
if len(c_full) < len(BINS_COMPLETO):
    c_full = np.pad(c_full, (0, len(BINS_COMPLETO)-len(c_full)), constant_values=0.0)

campo, _ = cascata(sinal, b_full, BINS_COMPLETO, a_full, c_full)
campo     = normalizar(campo)

# Mistura wet=1/3
resultado = normalizar(WET * campo + (1.0 - WET) * sinal)
print(f"Campo aplicado. wet={WET:.4f}  (1/3)")

# ── Métricas ──────────────────────────────────────────────────────────────────
def autocorr(s):
    s=s-s.mean(); return float(np.corrcoef(s[:-1],s[1:])[0,1])
def entr_esp(s):
    mag=np.abs(np.fft.rfft(s))**2; p=mag/(mag.sum()+1e-12); p=p[p>1e-15]
    return float(-np.sum(p*np.log2(p)))
def bits_ef(s):
    c,_=np.histogram(s,bins=256); c=c[c>0]; p=c/c.sum()
    return float(-np.sum(p*np.log2(p)))

print(f"\n{'='*55}")
print(f"{'Sinal':<30} {'AutoCorr':>9} {'EntrEsp':>9} {'Bits':>6}")
print("-"*55)
for nome, sig in [
    ("Violão original",    sinal),
    ("Campo φ  wet=1/3",   resultado),
]:
    print(f"{nome:<30} {autocorr(sig):>9.4f} {entr_esp(sig):>9.4f} {bits_ef(sig):>6.4f}")

# ── Salvar ────────────────────────────────────────────────────────────────────
wavfile.write("violao_campo_phi_v2.wav", rate, (resultado * 32767).astype(np.int16))
print("Salvo: violao_campo_phi_v2.wav")

# ── Players ───────────────────────────────────────────────────────────────────
N_PLAY = min(int(40 * rate), len(sinal))

print(f"\n{'='*55}")
print("▶ 1. VIOLÃO ORIGINAL")
display(Audio(sinal[:N_PLAY], rate=rate))

print(f"▶ 2. VIOLÃO + CAMPO φ  (wet=1/3)")
display(Audio(resultado[:N_PLAY], rate=rate))

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot([h.max() for h in hist], 'o-', color='#00cc66', lw=1.5)
axes[0].axhline(PHI**3, color='gold', lw=1.5, ls=':', label=f'φ³={PHI**3:.4f}')
axes[0].set_title(f"Convergência β — violão (convergiu ciclo {ciclo_conv})")
axes[0].set_xlabel("Ciclo"); axes[0].set_ylabel("β máximo")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

t = np.linspace(0, len(sinal)/rate, len(sinal))
axes[1].plot(t, sinal,     color='gray',    lw=0.3, alpha=0.7, label='Original')
axes[1].plot(t, resultado, color='#00cc66', lw=0.3, alpha=0.85, label='Campo φ wet=1/3')
axes[1].set_title("Forma de onda — original vs campo φ")
axes[1].set_xlabel("s"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig("violao_campo_phi_v2.png", dpi=120, bbox_inches='tight')
plt.show()
