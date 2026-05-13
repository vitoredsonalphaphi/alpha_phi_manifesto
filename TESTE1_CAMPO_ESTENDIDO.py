# © Vitor Edson Delavi · Florianópolis · 2026
# TESTE 1 — ECO BEEP 880 estendido após convergência
#
# Hipótese: o campo harmônico que emerge no ponto de dobra 5
# pode ser sustentado e estendido se β for congelado em φ³.
# Se a sensação ergonômica acompanha o campo — ela também se estende.
#
# Rode no Google Colab.

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display
import matplotlib.pyplot as plt

PHI      = (1 + np.sqrt(5)) / 2
FS       = 44100
F_BEEP   = 880.0
F_ORG    = 220.0
F_M      = F_ORG / PHI
BETA_FM  = PHI
ALPHA    = 1.0 / 3.0
DURACAO  = 1.5        # duração do beep de convergência
EXTENSAO = 8.0        # segundos de extensão com β congelado em φ³
N_STEPS  = 5
N_CICLOS = 20
FADE     = int(0.15 * FS)

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
               if (coh_mem is not None and i<len(coh_mem)) else coh)
        cohs.append(coh)
        nk  = np.arange(len(Fb))
        env = np.clip(1.0+(ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag*env)*np.exp(1j*phase)
    r = np.fft.irfft(F_out, n=N)
    return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi):
    s = sinal.copy()
    cm = np.zeros(len(bins_phi))
    passos = []
    for p in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se)
        passos.append(se.copy()); s = se.copy()
    return passos, cohs

def agente_eco(sinal, bins_phi):
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    historico_beta = []
    ultimo_passo5 = sinal.copy()
    for ciclo in range(N_CICLOS):
        passos, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        historico_beta.append(beta.copy())
        ultimo_passo5 = passos[-1]  # ponto de dobra 5 deste ciclo
    return beta, ultimo_passo5, historico_beta

# ── Gerar sinal ECO BEEP 880 ──────────────────────────────────────────────────
N = int(FS * DURACAO)
t = np.linspace(0, DURACAO, N, endpoint=False)

beep = np.sign(np.sin(2*np.pi*F_BEEP*t))
fm   = np.sin(2*np.pi*F_M*t + BETA_FM*np.sin(2*np.pi*F_M*t/PHI))
sinal_base = normalizar(ALPHA*beep + (1-ALPHA)*fm)

BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N)

print("=" * 55)
print("TESTE 1 — Campo Harmônico Estendido")
print(f"ECO BEEP 880Hz  |  {N_CICLOS} ciclos  |  {N_STEPS} passos")
print("=" * 55)

# ── Fase 1: Convergência adaptativa ──────────────────────────────────────────
print("\nFase 1: convergência adaptativa...")
beta_final, sinal_passo5, hist_beta = agente_eco(sinal_base, BINS_PHI)

print(f"β final — médio: {beta_final.mean():.4f}  máx: {beta_final.max():.4f}")
print(f"φ³ = {PHI**3:.4f}  →  diferença: {abs(beta_final.max() - PHI**3):.6f}")

# ── Fase 2: Extensão com β congelado em φ³ ───────────────────────────────────
print(f"\nFase 2: extensão de {EXTENSAO}s com β congelado em φ³...")

N_EXT    = int(FS * EXTENSAO)
t_ext    = np.linspace(0, EXTENSAO, N_EXT, endpoint=False)
beep_ext = np.sign(np.sin(2*np.pi*F_BEEP*t_ext))
fm_ext   = np.sin(2*np.pi*F_M*t_ext + BETA_FM*np.sin(2*np.pi*F_M*t_ext/PHI))
sinal_ext_base = normalizar(ALPHA*beep_ext + (1-ALPHA)*fm_ext)

BINS_EXT = bandas_para_bins(BANDAS, N_EXT)

# Aplicar β congelado em φ³ — sem adaptação
beta_frozen = np.full(len(BINS_EXT), PHI**3)
passos_ext, _ = cascata_eq(sinal_ext_base, beta_frozen, BINS_EXT)
sinal_extensao = passos_ext[-1]  # passo 5 direto

print(f"Extensão processada: {len(sinal_extensao)/FS:.1f}s com β=φ³={PHI**3:.4f}")

# ── Montar sequência completa ─────────────────────────────────────────────────
# [beep bruto] + [ponto de dobra 5] + [extensão congelada]
fade_n  = min(FADE, len(sinal_passo5), len(sinal_extensao))
t_fade  = np.linspace(0, 1, fade_n)
transicao = sinal_passo5.copy()
transicao[-fade_n:] = (sinal_passo5[-fade_n:]*(1-t_fade) +
                       sinal_extensao[:fade_n]*t_fade)
sequencia_completa = normalizar(np.concatenate([
    sinal_base,          # beep bruto (sem campo)
    transicao,           # ponto de dobra 5 (campo emerge)
    sinal_extensao[fade_n:]  # extensão com β=φ³ congelado
]))

print(f"\nSequência completa: {len(sequencia_completa)/FS:.1f}s total")
print(f"  [0.0 – {DURACAO:.1f}s]   beep bruto (sem campo)")
print(f"  [{DURACAO:.1f} – {DURACAO*2:.1f}s]   ponto de dobra 5 (campo emerge)")
print(f"  [{DURACAO*2:.1f} – {DURACAO*2+EXTENSAO:.1f}s]  extensão β=φ³ congelado")

# ── Salvar ────────────────────────────────────────────────────────────────────
wavfile.write("campo_ponto5.wav",   FS, (sinal_passo5    * 32767).astype(np.int16))
wavfile.write("campo_estendido.wav",FS, (sequencia_completa * 32767).astype(np.int16))
wavfile.write("beep_bruto.wav",     FS, (sinal_base      * 32767).astype(np.int16))

# ── Players ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("▶ BEEP BRUTO — sem campo (referência)")
display(Audio("beep_bruto.wav", rate=FS))

print("▶ PONTO DE DOBRA 5 — campo harmônico emergindo")
display(Audio("campo_ponto5.wav", rate=FS))

print("▶ SEQUÊNCIA COMPLETA — bruto → dobra 5 → extensão β=φ³")
display(Audio("campo_estendido.wav", rate=FS))

# ── Métricas por fase ─────────────────────────────────────────────────────────
def autocorr(s):
    s = s - s.mean()
    return float(np.corrcoef(s[:-1], s[1:])[0,1])

def entr_esp(s):
    mag = np.abs(np.fft.rfft(s))**2
    p   = mag/(mag.sum()+1e-12); p = p[p>1e-15]
    return float(-np.sum(p*np.log2(p)))

def bits_ef(s):
    c, _ = np.histogram(s, bins=256); c = c[c>0]; p = c/c.sum()
    return float(-np.sum(p*np.log2(p)))

print("\n" + "=" * 55)
print(f"{'Fase':<28} {'AutoCorr':>9} {'EntrEsp':>9} {'Bits':>7}")
print("-" * 55)
for nome, sig in [
    ("Beep bruto",            sinal_base),
    ("Ponto de dobra 5",      sinal_passo5),
    ("Extensão β=φ³",         sinal_extensao),
]:
    print(f"{nome:<28} {autocorr(sig):>9.4f} {entr_esp(sig):>9.4f} {bits_ef(sig):>7.4f}")

# ── Convergência de β ─────────────────────────────────────────────────────────
betas_max  = [h.max()  for h in hist_beta]
betas_mean = [h.mean() for h in hist_beta]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(range(1, N_CICLOS+1), betas_max,  'o-', color='#00cc66', label='β máximo')
axes[0].plot(range(1, N_CICLOS+1), betas_mean, 's--',color='#88ddaa', label='β médio')
axes[0].axhline(PHI**3, color='gold', lw=1.5, ls=':', label=f'φ³={PHI**3:.4f}')
axes[0].axhline(PHI,    color='gray', lw=1,   ls=':', label=f'φ={PHI:.4f}')
axes[0].set_title("Convergência de β por ciclo")
axes[0].set_xlabel("Ciclo"); axes[0].set_ylabel("β")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

t_seq = np.linspace(0, len(sequencia_completa)/FS, len(sequencia_completa))
axes[1].plot(t_seq, sequencia_completa, color='#00cc66', lw=0.4, alpha=0.85)
axes[1].axvline(DURACAO,   color='gold',  lw=1.5, ls='--', label='Ponto de dobra 5')
axes[1].axvline(DURACAO*2, color='white', lw=1.5, ls='--', label='Início extensão β=φ³')
axes[1].set_title("Forma de onda — sequência completa")
axes[1].set_xlabel("Tempo (s)"); axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig("teste1_campo_estendido.png", dpi=120, bbox_inches='tight')
plt.show()
print("\nGráfico salvo: teste1_campo_estendido.png")
print("\nOBSERVE: a sensação ergonômica persiste na extensão?")
print("Se sim → β=φ³ pode ser aplicado diretamente ao violão.")
