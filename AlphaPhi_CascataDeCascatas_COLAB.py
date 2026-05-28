"""
AlphaPhi_CascataDeCascatas_COLAB.py  v1
Cascata de Cascatas — para Google Colab

Cole tudo numa única célula e execute.

Três níveis:
  N1: cone original          → β → φ³  (verificado)
  N2: 5 cones em uníssono    → meta-frequência
  N3: eco sobre meta-freq    → atrator: φ³? φ⁴? campo?

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
from IPython.display import Audio, Image, display

# ── constantes ORIGINAIS — não modificar ─────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5
N_STEPS    = 5
N_CICLOS   = 20
FADE_S     = 0.15

N_SINAL    = int(FS * DURACAO)
DUR_TOTAL  = DURACAO * (N_STEPS + 1) - FADE_S * N_STEPS   # 8.25s
OFFSET_S   = DUR_TOTAL / N_STEPS                            # 1.65s

print("=" * 60)
print("  AlphaPhi · Cascata de Cascatas  v1")
print("=" * 60)
print(f"  φ³ = {PHI**3:.6f}  |  OFFSET = {OFFSET_S:.2f}s  |  DUR = {DUR_TOTAL:.2f}s")

# ── funções core ──────────────────────────────────────────────────────────────
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
    return [(max(0, int(f_lo / (FS / n))),
             min(int(f_hi / (FS / n)) + 1, n // 2 + 1),
             f_lo, f_hi)
            for f_lo, f_hi in bandas]

BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb  = F[b_lo:b_hi]
        mag = np.abs(Fb); phase = np.angle(Fb)
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

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    cm = np.zeros(len(bins_phi))
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs
        se = normalizar(se); cas.append(se); s = se.copy()
    return cas, cohs

def agente_eco(sinal, bins_phi, n_ciclos=N_CICLOS):
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    cas_f = None
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn * ba + wm * bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI ** 3)
        cas_f = cas
    return beta, cas_f

def concatenar(cas, fade=None):
    if fade is None: fade = int(FADE_S * FS)
    out = cas[0].copy()
    for s in cas[1:]:
        fade_n = min(fade, len(out), len(s))
        t_fade = np.linspace(0.0, 1.0, fade_n)
        out[-fade_n:] = out[-fade_n:] * (1 - t_fade) + s[:fade_n] * t_fade
        out = np.concatenate([out, s[fade_n:]])
    return normalizar(out)

# ── NÍVEL 1: cone original ────────────────────────────────────────────────────
print("\n  Gerando N1 — cone original...")
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t_sig)))
fm    = normalizar(np.sin(2 * np.pi * F_ORG * t_sig
                          + BETA_FM * np.sin(2 * np.pi * F_M * t_sig)))
x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

beta_N1, cas_N1 = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal_N1 = concatenar(cas_N1)
print(f"  β_max N1 = {beta_N1.max():.6f}  (φ³ = {PHI**3:.6f})")

# ── NÍVEL 2: cascata de cascatas ──────────────────────────────────────────────
print("\n  Gerando N2 — cascata de cascatas (5 instâncias)...")
OFFSET_N  = int(OFFSET_S * FS)
LEN_CONE  = len(sinal_N1)
LEN_META  = LEN_CONE + 4 * OFFSET_N

meta = np.zeros(LEN_META)
for i in range(N_STEPS):
    inicio = i * OFFSET_N
    fim    = inicio + LEN_CONE
    meta[inicio:fim] += sinal_N1
meta = normalizar(meta)

JANELA_INICIO = 4 * OFFSET_N
meta_ativa = normalizar(meta[JANELA_INICIO: JANELA_INICIO + N_SINAL])

beta_N2, _ = agente_eco(meta_ativa, BINS_PHI, N_CICLOS)
print(f"  β_max N2 = {beta_N2.max():.6f}  Δ vs N1 = {beta_N2.max()-beta_N1.max():+.6f}")

# campo contínuo 30s
TARGET_S   = 30.0
N_LOOPS    = int(np.ceil(TARGET_S / DUR_TOTAL)) + 2
campo_cont = np.zeros(int(TARGET_S * FS) + LEN_CONE)
for i in range(N_STEPS):
    for loop in range(N_LOOPS):
        inicio = i * OFFSET_N + loop * LEN_CONE
        fim    = min(inicio + LEN_CONE, len(campo_cont))
        if inicio >= len(campo_cont): break
        campo_cont[inicio:fim] += sinal_N1[:fim - inicio]
campo_cont = normalizar(campo_cont[:int(TARGET_S * FS)])

# ── NÍVEL 3: eco sobre a meta-frequência ─────────────────────────────────────
print("\n  Gerando N3 — eco sobre meta-frequência...")
beta_N3, cas_N3 = agente_eco(meta_ativa, BINS_PHI, N_CICLOS)
sinal_N3 = concatenar(cas_N3)
print(f"  β_max N3 = {beta_N3.max():.6f}  Δ vs N1 = {beta_N3.max()-beta_N1.max():+.6f}")

# ── diagnóstico ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DIAGNÓSTICO DO ATRATOR")
print("=" * 60)
for nivel, bmax, label in [
        (1, beta_N1.max(), "Cone original  "),
        (2, beta_N2.max(), "Meta-frequência"),
        (3, beta_N3.max(), "Eco sobre meta ")]:
    pot = np.log(bmax) / np.log(PHI)
    print(f"  N{nivel} {label}: {bmax:.6f}  (φ^{pot:.3f})")

# ── visualização ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 10), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Cascata de Cascatas  v1', color='white', fontsize=13)

COR     = ['#00aaff', '#00ffaa', '#ffaa00']
TITULOS = ['N1 — Cone original (8.25s)',
           'N2 — Meta-frequência (5 cones)',
           'N3 — Eco sobre meta-frequência']
SINAIS  = [sinal_N1,
           meta[:int(min(len(meta), int(8.25*FS)))],
           sinal_N3]
BETAS   = [beta_N1, beta_N2, beta_N3]

for i, (sig, beta, cor, titulo) in enumerate(zip(SINAIS, BETAS, COR, TITULOS)):
    ax_s, ax_b = axes[i][0], axes[i][1]
    t = np.linspace(0, len(sig) / FS, len(sig))
    ax_s.plot(t, sig, color=cor, lw=0.4, alpha=0.8)
    ax_s.set_facecolor('#111111'); ax_s.set_title(titulo, color='white', fontsize=10)
    ax_s.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_s.tick_params(colors='#888888')
    for sp in ax_s.spines.values(): sp.set_color('#333333')

    ax_b.bar(range(len(beta)), beta, color=cor, alpha=0.7)
    ax_b.axhline(PHI**3, color='white',  lw=1.2, ls='--', label=f'φ³={PHI**3:.3f}')
    ax_b.axhline(PHI**4, color='yellow', lw=0.8, ls=':',  label=f'φ⁴={PHI**4:.3f}')
    ax_b.set_facecolor('#111111')
    ax_b.set_title(f'β-bandas φ  (max={beta.max():.4f})', color='white', fontsize=10)
    ax_b.legend(fontsize=8, facecolor='#222222', labelcolor='white')
    ax_b.tick_params(colors='#888888')
    for sp in ax_b.spines.values(): sp.set_color('#333333')

plt.tight_layout()
plt.savefig('/content/cascata_de_cascatas_resultado.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
display(Image('/content/cascata_de_cascatas_resultado.png'))

# ── salvar e reproduzir áudios ────────────────────────────────────────────────
def salvar_e_tocar(sinal, nome, label):
    s16 = np.int16(np.clip(normalizar(sinal), -1, 1) * 32767)
    wavfile.write(nome, FS, s16)
    print(f"\n  ▶ {label}  ({len(sinal)/FS:.1f}s)")
    display(Audio(nome))

salvar_e_tocar(sinal_N1,   '/content/cascata_N1_original.wav',
               'N1 — Cone original (8.25s)')

salvar_e_tocar(campo_cont, '/content/cascata_N2_campo_continuo_30s.wav',
               'N2 — Campo contínuo 30s (para experimento físico)')

salvar_e_tocar(sinal_N3,   '/content/cascata_N3_eco_meta.wav',
               'N3 — Eco sobre meta-frequência (8.25s)')

print("\n" + "=" * 60)
print("  Comparar N1 × N2 × N3 pelo ouvido:")
print("  N1 = cone único — o ECO BEEP 880 original")
print("  N2 = 5 cones simultâneos — mais denso, mesmo campo")
print("  N3 = eco sobre N2 — o eco reconhece o que já está lá")
print("=" * 60)
