"""
AlphaPhi_CascataDeCascatas.py  v1
Cascata de Cascatas — Campo Harmônico Contínuo

Três níveis de organização:
  N1: cone original          → β_max → φ³  (verificado)
  N2: 5 cones em uníssono    → meta-frequência (sinal novo)
  N3: eco sobre meta-freq    → atrator: φ³? φ⁴? campo de atratores?

Arquitetura N2:
  5 instâncias do cascata completo (8.25s cada)
  defasadas por OFFSET = 8.25s / 5 = 1.65s
  somadas → meta-sinal que contém todos os 5 estados espectrais simultaneamente

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io import wavfile

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
OFFSET_S   = DUR_TOTAL / N_STEPS                            # 1.65s por defasagem

print("=" * 60)
print("  AlphaPhi · Cascata de Cascatas  v1")
print("=" * 60)
print(f"\n  φ           = {PHI:.6f}")
print(f"  φ³          = {PHI**3:.6f}")
print(f"  DUR_TOTAL   = {DUR_TOTAL:.2f}s  (um cone completo)")
print(f"  OFFSET_S    = {OFFSET_S:.2f}s  (defasagem entre instâncias)")
print(f"  5 instâncias cobrem: {5 * OFFSET_S:.2f}s = 1 ciclo completo")

# ── funções core (originais) ──────────────────────────────────────────────────
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

def medir_beta(sinal):
    beta, _ = agente_eco(sinal, BINS_PHI, N_CICLOS)
    return beta

# ── NÍVEL 1: cone original ────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  NÍVEL 1 — Cone original")
print("─" * 60)

t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t_sig)))
fm    = normalizar(np.sin(2 * np.pi * F_ORG * t_sig
                          + BETA_FM * np.sin(2 * np.pi * F_M * t_sig)))
x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

beta_N1, cas_N1 = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal_N1 = concatenar(cas_N1)

print(f"  β_max N1  = {beta_N1.max():.6f}")
print(f"  φ³        = {PHI**3:.6f}")
print(f"  Δ         = {abs(beta_N1.max() - PHI**3):.6f}")

# ── NÍVEL 2: cascata de cascatas — 5 instâncias defasadas ────────────────────
print("\n" + "─" * 60)
print("  NÍVEL 2 — Cascata de cascatas (5 instâncias em uníssono)")
print("─" * 60)

OFFSET_N  = int(OFFSET_S * FS)              # 1.65s em amostras
LEN_CONE  = len(sinal_N1)                   # 8.25s em amostras
LEN_META  = LEN_CONE + 4 * OFFSET_N        # comprimento total

meta = np.zeros(LEN_META)
for i in range(N_STEPS):
    inicio = i * OFFSET_N
    fim    = inicio + LEN_CONE
    meta[inicio:fim] += sinal_N1

meta = normalizar(meta)

# janela onde todas 5 instâncias estão ativas: a partir do 5° offset
JANELA_INICIO = 4 * OFFSET_N
meta_ativa = meta[JANELA_INICIO: JANELA_INICIO + N_SINAL]
meta_ativa = normalizar(meta_ativa)

beta_N2 = medir_beta(meta_ativa)
print(f"  β_max N2  = {beta_N2.max():.6f}")
print(f"  φ³        = {PHI**3:.6f}")
print(f"  φ⁴        = {PHI**4:.6f}")
print(f"  Δ vs N1   = {beta_N2.max() - beta_N1.max():+.6f}")

# campo contínuo de 30 segundos para o experimento físico
TARGET_S   = 30.0
N_LOOPS    = int(np.ceil(TARGET_S / DUR_TOTAL)) + 2
campo_cont = np.zeros(int(TARGET_S * FS) + LEN_CONE)
for i in range(N_STEPS):
    for loop in range(N_LOOPS):
        inicio = i * OFFSET_N + loop * LEN_CONE
        fim    = inicio + LEN_CONE
        if inicio >= len(campo_cont): break
        fim = min(fim, len(campo_cont))
        campo_cont[inicio:fim] += sinal_N1[:fim - inicio]

campo_cont = normalizar(campo_cont[:int(TARGET_S * FS)])

# ── NÍVEL 3: eco sobre a meta-frequência ─────────────────────────────────────
print("\n" + "─" * 60)
print("  NÍVEL 3 — Eco sobre a meta-frequência")
print("─" * 60)

beta_N3, cas_N3 = agente_eco(meta_ativa, BINS_PHI, N_CICLOS)
sinal_N3 = concatenar(cas_N3)

print(f"  β_max N3  = {beta_N3.max():.6f}")
print(f"  φ³        = {PHI**3:.6f}")
print(f"  φ⁴        = {PHI**4:.6f}")
print(f"  φ⁵        = {PHI**5:.6f}")
print(f"  Δ vs N1   = {beta_N3.max() - beta_N1.max():+.6f}")
print(f"  Δ vs N2   = {beta_N3.max() - beta_N2.max():+.6f}")

# ── diagnóstico do atrator ────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  DIAGNÓSTICO DO ATRATOR")
print("─" * 60)

for nivel, bmax, label in [
        (1, beta_N1.max(), "Cone original"),
        (2, beta_N2.max(), "Meta-frequência"),
        (3, beta_N3.max(), "Eco sobre meta")]:
    razao = bmax / (PHI ** 3)
    if abs(razao - 1.0) < 0.01:
        status = f"≈ φ³  (estável)"
    elif bmax > PHI ** 3 + 0.01:
        pot = np.log(bmax) / np.log(PHI)
        status = f"> φ³  (φ^{pot:.3f} — escalamento)"
    else:
        status = f"< φ³  (comprimido)"
    print(f"  N{nivel} {label:20s}: {bmax:.6f}  {status}")

# ── salvar áudios ─────────────────────────────────────────────────────────────
def salvar_wav(sinal, nome):
    s16 = np.int16(np.clip(normalizar(sinal), -1, 1) * 32767)
    wavfile.write(nome, FS, s16)
    print(f"  → {nome}  ({len(sinal)/FS:.1f}s)")

print("\n  Salvando áudios…")
salvar_wav(sinal_N1,    'cascata_N1_original.wav')
salvar_wav(campo_cont,  'cascata_N2_campo_continuo_30s.wav')
salvar_wav(sinal_N3,    'cascata_N3_eco_meta.wav')

# ── visualização ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 10),
                          facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Cascata de Cascatas  v1',
             color='white', fontsize=13, y=0.98)

COR = ['#00aaff', '#00ffaa', '#ffaa00']
TITULOS = ['N1 — Cone original', 'N2 — Meta-frequência', 'N3 — Eco sobre meta']
SINAIS  = [sinal_N1, meta[:int(min(len(meta), 8*FS))], sinal_N3]
BETAS   = [beta_N1,  beta_N2,  beta_N3]

for i, (sig, beta, cor, titulo) in enumerate(zip(SINAIS, BETAS, COR, TITULOS)):
    ax_s = axes[i][0]
    ax_b = axes[i][1]

    t = np.linspace(0, len(sig) / FS, len(sig))
    ax_s.plot(t, sig, color=cor, lw=0.4, alpha=0.8)
    ax_s.set_facecolor('#111111')
    ax_s.set_title(titulo, color='white', fontsize=10)
    ax_s.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_s.tick_params(colors='#888888')
    for spine in ax_s.spines.values(): spine.set_color('#333333')

    ax_b.bar(range(len(beta)), beta, color=cor, alpha=0.7, width=0.8)
    ax_b.axhline(PHI**3, color='white',  lw=1.0, ls='--', label=f'φ³={PHI**3:.3f}')
    ax_b.axhline(PHI**4, color='yellow', lw=0.7, ls=':', label=f'φ⁴={PHI**4:.3f}')
    ax_b.set_facecolor('#111111')
    ax_b.set_title(f'β-bandas  (max={beta.max():.4f})', color='white', fontsize=10)
    ax_b.legend(fontsize=7, facecolor='#222222', labelcolor='white')
    ax_b.tick_params(colors='#888888')
    for spine in ax_b.spines.values(): spine.set_color('#333333')

plt.tight_layout()
plt.savefig('cascata_de_cascatas_resultado.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
print("\n  → cascata_de_cascatas_resultado.png")

# ── sumário final ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTADO FINAL")
print("=" * 60)
print(f"\n  N1  β_max = {beta_N1.max():.6f}  (φ³ = {PHI**3:.6f})")
print(f"  N2  β_max = {beta_N2.max():.6f}  (φ⁴ = {PHI**4:.6f})")
print(f"  N3  β_max = {beta_N3.max():.6f}  (φ⁵ = {PHI**5:.6f})\n")

if beta_N3.max() > beta_N1.max() + 0.01:
    print("  ✓ Atrator escalou — meta-frequência contém mais organização potencial")
elif abs(beta_N3.max() - beta_N1.max()) < 0.01:
    print("  → Atrator estável em φ³ — eco reconhece campo já presente")
else:
    print("  → Atrator comprimido — investigar")

print(f"\n  Campo contínuo 30s salvo: cascata_N2_campo_continuo_30s.wav")
print("  (para experimento físico — borracha + glicerina)")
print("=" * 60)
