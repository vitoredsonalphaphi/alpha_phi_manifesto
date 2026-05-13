# © Vitor Edson Delavi · Florianópolis · 2026
# TESTE 2 — Convergência informada pelo campo global
#
# Hipótese: o agente observa o espectro do áudio inteiro para
# inicializar β. Converge em apenas 4 segundos (β já sabe onde está
# indo). Aplica o campo estabilizado ao áudio completo.
#
# Diferença do teste anterior: β não começa em 1.0 — começa
# informado pelo campo total. A convergência é acelerada porque
# o ponto de partida já carrega o conhecimento do todo.
#
# Rode no Google Colab.

import urllib.request
import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display
import matplotlib.pyplot as plt

BASE = "https://raw.githubusercontent.com/vitoredsonalphaphi/alpha_phi_manifesto/claude/good-morning-N6f3S/"

print("Baixando áudio...")
urllib.request.urlretrieve(BASE + "violao_original.wav", "original.wav")
rate, data = wavfile.read("original.wav")
sinal_completo = data.astype(np.float64) / 32768.0
sinal_completo = sinal_completo / (np.max(np.abs(sinal_completo)) + 1e-10)
print(f"Áudio: {len(sinal_completo)/rate:.1f}s  |  {rate}Hz\n")

PHI     = (1 + np.sqrt(5)) / 2
N_STEPS = 5
FADE    = int(0.15 * rate)

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

def cascata_eq(sinal, beta_bands, bins_phi):
    s = sinal.copy()
    cm = np.zeros(len(bins_phi))
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se); s = se.copy()
    return s, cohs

def agente_eco_informado(sinal_curto, bins_phi, beta_init, n_ciclos=20):
    """Agente com β inicializado pelo campo global — não começa em 1.0."""
    nb = len(bins_phi)
    beta = beta_init.copy(); bm = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    historico = []
    for _ in range(n_ciclos):
        _, cohs = cascata_eq(sinal_curto, beta, bins_phi)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        historico.append(beta.copy())
    return beta, historico

# ── FASE 1: Analisar campo global ────────────────────────────────────────────
print("=" * 55)
print("FASE 1 — Observação do campo global (áudio completo)")
print("=" * 55)

BANDAS = gerar_bandas_phi()
BINS_COMPLETO = bandas_para_bins(BANDAS, len(sinal_completo))

# Coerência espectral de cada banda do sinal completo
F_global = np.fft.rfft(sinal_completo)
cohs_global = []
for b_lo, b_hi, _, _ in BINS_COMPLETO:
    Fb  = F_global[b_lo:b_hi]
    mag = np.abs(Fb)
    an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
    coh = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
    cohs_global.append(coh)

cohs_global = np.array(cohs_global)
# β inicial informado pelo campo global
beta_global = PHI**(3 * cohs_global)
beta_global = np.clip(beta_global, 0.05, PHI**3)

print(f"Bandas analisadas: {len(BINS_COMPLETO)}")
print(f"β inicial (campo global) — médio: {beta_global.mean():.4f}  máx: {beta_global.max():.4f}")
print(f"Comparação: β em branco seria 1.0000 | φ³ = {PHI**3:.4f}")

# ── FASE 2: Convergência nos primeiros 4 segundos ────────────────────────────
print(f"\n{'='*55}")
print("FASE 2 — Convergência nos primeiros 4s (β informado)")
print("=" * 55)

N_4S = min(int(4 * rate), len(sinal_completo))
sinal_4s = sinal_completo[:N_4S]
BINS_4S  = bandas_para_bins(BANDAS, N_4S)

# Ajustar beta_global para o tamanho dos bins de 4s
beta_init_4s = beta_global[:len(BINS_4S)]
if len(beta_init_4s) < len(BINS_4S):
    beta_init_4s = np.pad(beta_init_4s, (0, len(BINS_4S)-len(beta_init_4s)), constant_values=1.0)

beta_convergido, hist_4s = agente_eco_informado(sinal_4s, BINS_4S, beta_init_4s)

print(f"β convergido — médio: {beta_convergido.mean():.4f}  máx: {beta_convergido.max():.4f}")
print(f"φ³ = {PHI**3:.4f}  →  diferença máx: {abs(beta_convergido.max()-PHI**3):.6f}")

# Ciclos para atingir 99% de φ³
limiar = 0.99 * PHI**3
ciclo_convergencia = next((i+1 for i,h in enumerate(hist_4s) if h.max() >= limiar), 20)
print(f"Atingiu 99% de φ³ no ciclo: {ciclo_convergencia}/20")

# ── FASE 3: Aplicar campo ao áudio completo ──────────────────────────────────
print(f"\n{'='*55}")
print("FASE 3 — Aplicando campo ao áudio completo")
print("=" * 55)

# Beta para o sinal completo: usar beta_convergido (tamanho pode diferir)
beta_full = beta_convergido[:len(BINS_COMPLETO)]
if len(beta_full) < len(BINS_COMPLETO):
    beta_full = np.pad(beta_full, (0, len(BINS_COMPLETO)-len(beta_full)),
                       constant_values=PHI**3)

sinal_processado, cohs_final = cascata_eq(sinal_completo, beta_full, BINS_COMPLETO)
sinal_processado = normalizar(sinal_processado)
print("Campo aplicado ao áudio completo.")

# ── Métricas ─────────────────────────────────────────────────────────────────
def autocorr(s):
    s = s - s.mean()
    return float(np.corrcoef(s[:-1], s[1:])[0,1])

def entr_esp(s):
    mag = np.abs(np.fft.rfft(s))**2
    p   = mag/(mag.sum()+1e-12); p = p[p>1e-15]
    return float(-np.sum(p*np.log2(p)))

def bits_ef(s):
    c,_ = np.histogram(s, bins=256); c=c[c>0]; p=c/c.sum()
    return float(-np.sum(p*np.log2(p)))

print(f"\n{'='*55}")
print(f"{'Sinal':<30} {'AutoCorr':>9} {'EntrEsp':>9} {'Bits':>7}")
print("-"*55)
for nome, sig in [
    ("Original (completo)",      sinal_completo),
    ("Processado (campo global)", sinal_processado),
]:
    print(f"{nome:<30} {autocorr(sig):>9.4f} {entr_esp(sig):>9.4f} {bits_ef(sig):>7.4f}")

# Referência do ponto de dobra 5 puro (do teste anterior)
print(f"{'Ponto dobra 5 puro (ref.)':<30} {'1.0000':>9} {'0.0601':>9} {'7.8931':>7}  ← alvo")

# ── Salvar e ouvir ────────────────────────────────────────────────────────────
wavfile.write("campo_global_processado.wav", rate,
              (sinal_processado * 32767).astype(np.int16))

print(f"\n{'='*55}")
print("▶ ORIGINAL")
display(Audio("original.wav", rate=rate))

print("▶ CAMPO GLOBAL — convergência 4s → aplicado ao áudio completo")
display(Audio("campo_global_processado.wav", rate=rate))

# ── Gráficos ─────────────────────────────────────────────────────────────────
betas_max  = [h.max()  for h in hist_4s]
betas_mean = [h.mean() for h in hist_4s]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Convergência β
axes[0,0].plot(range(1,21), betas_max,  'o-', color='#00cc66', label='β máximo')
axes[0,0].plot(range(1,21), betas_mean, 's--',color='#88ddaa', label='β médio')
axes[0,0].axhline(PHI**3, color='gold', lw=1.5, ls=':', label=f'φ³={PHI**3:.4f}')
axes[0,0].axhline(beta_global.max(), color='cyan', lw=1, ls=':',
                  label=f'β_init={beta_global.max():.4f}')
axes[0,0].set_title("Convergência β (4s, informado pelo campo global)")
axes[0,0].set_xlabel("Ciclo"); axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)

# β por banda — global vs convergido
x = range(len(cohs_global))
axes[0,1].bar(x, beta_global,    alpha=0.5, color='cyan',    label='β inicial (campo global)')
axes[0,1].bar(x, beta_convergido[:len(cohs_global)], alpha=0.5,
              color='#00cc66', label='β convergido (4s)')
axes[0,1].axhline(PHI**3, color='gold', lw=1.5, ls=':', label=f'φ³')
axes[0,1].set_title("β por banda φ — inicial vs convergido")
axes[0,1].set_xlabel("Banda"); axes[0,1].legend(fontsize=8); axes[0,1].grid(alpha=0.2)

# Formas de onda
t_all = np.linspace(0, len(sinal_completo)/rate, len(sinal_completo))
axes[1,0].plot(t_all, sinal_completo,   color='gray',    lw=0.4, alpha=0.8)
axes[1,0].set_title("Original"); axes[1,0].set_xlabel("s")
axes[1,1].plot(t_all[:len(sinal_processado)], sinal_processado,
               color='#00cc66', lw=0.4, alpha=0.8)
axes[1,1].set_title("Campo global aplicado"); axes[1,1].set_xlabel("s")

plt.tight_layout()
plt.savefig("teste2_campo_global.png", dpi=120, bbox_inches='tight')
plt.show()

print("\nOBSERVE:")
print("1. Quantos ciclos foram necessários para convergir? (vs 16-20 sem informação)")
print("2. A sensação ergonômica está presente desde o início do áudio?")
print("3. As métricas se aproximam do ponto de dobra 5 puro (AutoCorr→1, EntrEsp→0.06)?")
