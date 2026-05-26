"""
AlphaPhi_CascataDeCascatas_COLAB.py  v2
Cascata de Cascatas — para Google Colab  · pesos φ nas instâncias

Cole tudo numa única célula e execute.

Três níveis:
  N1: cone original          → β → φ³  (verificado)
  N2: 5 cones em uníssono    → meta-frequência  (ponderada por φ)
  N3: eco sobre meta-freq    → atrator suave · agradabilidade φ

Arquitetura φ-ponderada (v2):
  Instância 1 — BASE (início):        peso = 1.0
  Instância 2 — transição:            peso = 1/φ²  ≈ 0.382
  Instância 3 — meio (mais recuada):  peso = 1/φ³  ≈ 0.236
  Instância 4 — transição:            peso = 1/φ²  ≈ 0.382
  Instância 5 — C.H. (campo harm.):  peso = 1/φ   ≈ 0.618

  BASE e CAMPO HARMÔNICO: proeminentes (duas linhas paralelas)
  Instâncias 2,3,4: segundo plano — o processo, não o destaque

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

# ── pesos φ por instância (v2) ────────────────────────────────────────────────
PESOS_PHI = np.array([
    1.0,           # instância 1 — BASE (sinal inicial, alta entropia)
    1.0 / PHI**2,  # instância 2 — ≈ 0.382 (segundo plano)
    1.0 / PHI**3,  # instância 3 — ≈ 0.236 (mais recuada)
    1.0 / PHI**2,  # instância 4 — ≈ 0.382 (segundo plano)
    1.0 / PHI,     # instância 5 — C.H. ≈ 0.618 (campo harmônico)
])

print("=" * 60)
print("  AlphaPhi · Cascata de Cascatas  v2  (pesos φ)")
print("=" * 60)
print(f"  φ³ = {PHI**3:.6f}  |  OFFSET = {OFFSET_S:.2f}s  |  DUR = {DUR_TOTAL:.2f}s")
print(f"\n  Pesos das 5 instâncias:")
labels = ["BASE     ", "transição", "meio     ", "transição", "C.H.     "]
for i, (lb, w) in enumerate(zip(labels, PESOS_PHI)):
    bar = "█" * int(w * 20)
    print(f"    [{i+1}] {lb} {w:.3f}  {bar}")
print(f"\n  Duas linhas paralelas:")
print(f"    BASE (1.000) ————— o sinal original, alta entropia")
print(f"    C.H. (0.618) ————— o campo harmônico resolvido")
print(f"    [2,3,4]  recuadas — o processo, não o destaque")

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

# ── NÍVEL 2: cascata de cascatas — ponderada por φ ────────────────────────────
print("\n  Gerando N2 — cascata de cascatas φ-ponderada (5 instâncias)...")
OFFSET_N  = int(OFFSET_S * FS)
LEN_CONE  = len(sinal_N1)
LEN_META  = LEN_CONE + 4 * OFFSET_N

meta = np.zeros(LEN_META)
for i in range(N_STEPS):
    inicio = i * OFFSET_N
    fim    = inicio + LEN_CONE
    meta[inicio:fim] += sinal_N1 * PESOS_PHI[i]   # ← pesos φ aplicados aqui
meta = normalizar(meta)

JANELA_INICIO = 4 * OFFSET_N
meta_ativa = normalizar(meta[JANELA_INICIO: JANELA_INICIO + N_SINAL])

beta_N2, _ = agente_eco(meta_ativa, BINS_PHI, N_CICLOS)
print(f"  β_max N2 = {beta_N2.max():.6f}  Δ vs N1 = {beta_N2.max()-beta_N1.max():+.6f}")

# campo contínuo 30s — ponderado por φ
TARGET_S   = 30.0
N_LOOPS    = int(np.ceil(TARGET_S / DUR_TOTAL)) + 2
campo_cont = np.zeros(int(TARGET_S * FS) + LEN_CONE)
for i in range(N_STEPS):
    for loop in range(N_LOOPS):
        inicio = i * OFFSET_N + loop * LEN_CONE
        fim    = min(inicio + LEN_CONE, len(campo_cont))
        if inicio >= len(campo_cont): break
        campo_cont[inicio:fim] += sinal_N1[:fim - inicio] * PESOS_PHI[i]  # ← pesos φ
campo_cont = normalizar(campo_cont[:int(TARGET_S * FS)])

# ── NÍVEL 3: eco sobre a meta-frequência ─────────────────────────────────────
print("\n  Gerando N3 — eco sobre meta-frequência φ-ponderada...")
beta_N3, cas_N3 = agente_eco(meta_ativa, BINS_PHI, N_CICLOS)
sinal_N3 = concatenar(cas_N3)
print(f"  β_max N3 = {beta_N3.max():.6f}  Δ vs N1 = {beta_N3.max()-beta_N1.max():+.6f}")

# ── diagnóstico ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DIAGNÓSTICO DO ATRATOR")
print("=" * 60)
for nivel, bmax, label in [
        (1, beta_N1.max(), "Cone original  "),
        (2, beta_N2.max(), "Meta φ-ponderada"),
        (3, beta_N3.max(), "Eco sobre meta ")]:
    pot = np.log(bmax) / np.log(PHI)
    print(f"  N{nivel} {label}: {bmax:.6f}  (φ^{pot:.3f})")

# ── visualização ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 10), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Cascata de Cascatas  v2  (pesos φ)',
             color='white', fontsize=13)

COR     = ['#00aaff', '#00ffaa', '#ffaa00']
TITULOS = ['N1 — Cone original (8.25s)',
           'N2 — Meta φ-ponderada (BASE + C.H. proeminentes)',
           'N3 — Eco sobre meta φ (gradiente suave)']
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

# diagrama de pesos — painel extra
fig2, ax_w = plt.subplots(figsize=(8, 3), facecolor='#0a0a0a')
ax_w.set_facecolor('#111111')
cores_inst = ['#00aaff', '#4488cc', '#336699', '#4488cc', '#00ffaa']
for i, (w, lb) in enumerate(zip(PESOS_PHI, labels)):
    ax_w.bar(i + 1, w, color=cores_inst[i], alpha=0.85, width=0.6)
    ax_w.text(i + 1, w + 0.02, f'{w:.3f}', ha='center', color='white', fontsize=9)
ax_w.set_title('Pesos φ das 5 instâncias  (BASE=1.0  ·  C.H.=1/φ  ·  [2,3,4]=segundo plano)',
               color='white', fontsize=10)
ax_w.set_xticks([1,2,3,4,5])
ax_w.set_xticklabels(['BASE\n(1.000)', '1/φ²\n(0.382)', '1/φ³\n(0.236)',
                       '1/φ²\n(0.382)', 'C.H.\n1/φ (0.618)'],
                      color='white', fontsize=8)
ax_w.tick_params(colors='#888888')
ax_w.set_ylim(0, 1.2)
for sp in ax_w.spines.values(): sp.set_color('#333333')
ax_w.axhline(1.0/PHI,   color='#00ffaa', lw=0.8, ls=':', alpha=0.6)
ax_w.axhline(1.0,       color='#00aaff', lw=0.8, ls=':', alpha=0.6)
fig2.tight_layout()
fig2.savefig('/content/cascata_pesos_phi.png', dpi=120,
             bbox_inches='tight', facecolor='#0a0a0a')

plt.tight_layout()
plt.savefig('/content/cascata_de_cascatas_v2_resultado.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
display(Image('/content/cascata_de_cascatas_v2_resultado.png'))
display(Image('/content/cascata_pesos_phi.png'))

# ── espectrogramas — a plástica interna da frequência ─────────────────────────
print("\n  Gerando espectrogramas...")
from scipy.signal import spectrogram as sp_specgram

# frequências das bandas φ (linhas horizontais de referência)
FREQS_BANDA = []
f_b = 20.0
while f_b < 22050:
    FREQS_BANDA.append(f_b)
    f_b = min(f_b * PHI, 22050)

fig3, axes3 = plt.subplots(3, 1, figsize=(14, 12), facecolor='#0a0a0a')
fig3.suptitle(
    'AlphaPhi · Espectrogramas — a plástica interna da frequência\n'
    'eixo X = tempo (s) · eixo Y = frequência (Hz, escala log) · cor = intensidade',
    color='white', fontsize=11)

SINAIS_ESP  = [sinal_N1,
               meta[:int(min(len(meta), int(10 * FS)))],
               sinal_N3]
TITULOS_ESP = [
    'N1 — Cone original · linhas amarelas = 5 pontos de dobra',
    'N2 — Meta φ-ponderada · linhas ciano = início de cada instância · cor = sobreposição',
    'N3 — Eco sobre meta · gradiente suave · sem dobras discretas visíveis'
]
CMAPS = ['Blues', 'Greens', 'YlOrRd']

# frequências-chave para anotar
F_CHAVE = [(18,  'φ⁶≈18Hz'), (52, '~52Hz'), (220, '220Hz'), (880, '880Hz')]

for i, (sig, titulo, cmap) in enumerate(zip(SINAIS_ESP, TITULOS_ESP, CMAPS)):
    ax = axes3[i]
    ax.set_facecolor('#050505')

    # calcula espectrograma
    f_s, t_s, Sxx = sp_specgram(sig, fs=FS, nperseg=2048, noverlap=1920,
                                  window='hann')
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
    Sxx_db -= Sxx_db.max()                          # normaliza 0 dB no topo

    # plota
    ax.pcolormesh(t_s, f_s, Sxx_db,
                  cmap=cmap, vmin=-55, vmax=0, shading='gouraud')

    # escala log — mostra bandas graves e agudas com clareza
    ax.set_yscale('log')
    ax.set_ylim(20, 22050)

    # linhas das bandas φ (horizontais) — a grade da frequência
    for f_phi in FREQS_BANDA[1:-1]:
        ax.axhline(f_phi, color='white', lw=0.3, alpha=0.2, ls=':')

    # marcações temporais
    if i == 0:
        # N1: 5 pontos de dobra
        for step in range(N_STEPS + 1):
            t_mark = step * DURACAO * (1 - FADE_S / DURACAO)
            ax.axvline(t_mark, color='yellow', lw=0.7, alpha=0.5, ls='--')
        ax.text(0.05, 18000, 'P1', color='yellow', fontsize=7, alpha=0.8)
        ax.text(DURACAO * 0.9, 18000, 'P2', color='yellow', fontsize=7, alpha=0.8)
        ax.text(DURACAO * 1.85, 18000, 'P3', color='yellow', fontsize=7, alpha=0.8)
        ax.text(DURACAO * 2.8, 18000, 'P4', color='yellow', fontsize=7, alpha=0.8)
        ax.text(DURACAO * 3.75, 18000, 'P5\n(C.H.)', color='yellow', fontsize=7, alpha=0.9)
    elif i == 1:
        # N2: início de cada instância
        for inst in range(N_STEPS):
            ax.axvline(inst * OFFSET_S, color='cyan', lw=0.7, alpha=0.5, ls='--')
            ax.text(inst * OFFSET_S + 0.05, 18000, f'I{inst+1}',
                    color='cyan', fontsize=7, alpha=0.8)

    # frequências-chave (horizontais coloridas)
    for f_k, lbl in F_CHAVE:
        ax.axhline(f_k, color='#ffcc44', lw=0.7, alpha=0.55, ls='-')
        ax.text(len(sig) / FS * 0.01, f_k * 1.15, lbl,
                color='#ffcc44', fontsize=7, alpha=0.85)

    ax.set_title(titulo, color='white', fontsize=9, pad=4)
    ax.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax.set_ylabel('frequência Hz (log)', color='#888888', fontsize=8)
    ax.tick_params(colors='#888888', labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#333333')

plt.tight_layout(h_pad=2.5)
plt.savefig('/content/cascata_espectrogramas.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
display(Image('/content/cascata_espectrogramas.png'))
print("  → espectrograma gerado: cascata_espectrogramas.png")
print("  eixo Y (vertical) = frequência em Hz, escala log")
print("  eixo X (horizontal) = tempo de 0 a 8.25s")
print("  cor = intensidade — quanto mais brilhante, mais energia nessa freq/tempo")
print("  linhas φ-band (brancas pontilhadas) = as 15 bandas da cascata")
print("  linhas amarelas (N1) = os 5 pontos de dobra")
print("  linhas ciano (N2) = início de cada instância defasada")

# ── salvar e reproduzir áudios ────────────────────────────────────────────────
def salvar_e_tocar(sinal, nome, label):
    s16 = np.int16(np.clip(normalizar(sinal), -1, 1) * 32767)
    wavfile.write(nome, FS, s16)
    print(f"\n  ▶ {label}  ({len(sinal)/FS:.1f}s)")
    display(Audio(nome))

salvar_e_tocar(sinal_N1,   '/content/cascata_v2_N1_original.wav',
               'N1 — Cone original (8.25s) — igual à v1')

salvar_e_tocar(campo_cont, '/content/cascata_v2_N2_campo_continuo_30s.wav',
               'N2 — Campo contínuo 30s · BASE + C.H. proeminentes · [2,3,4] recuadas')

salvar_e_tocar(sinal_N3,   '/content/cascata_v2_N3_eco_meta.wav',
               'N3 — Eco sobre meta φ · gradiente suave (8.25s)')

print("\n" + "=" * 60)
print("  v2 — o que mudou:")
print("  N1 = igual (referência)")
print("  N2 = BASE não grita mais · C.H. presente · meio recuado")
print("  N2 = duas linhas paralelas: início (BE) + fim (C.H.)")
print("  N3 = agradabilidade macia preservada · atrator φ³")
print("=" * 60)
print(f"\n  Pesos usados:")
for i, (lb, w) in enumerate(zip(labels, PESOS_PHI)):
    print(f"    [{i+1}] {lb}: {w:.4f}")
print("=" * 60)
