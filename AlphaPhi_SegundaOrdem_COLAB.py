"""
AlphaPhi_SegundaOrdem_COLAB.py
Recursão φ — Segunda Ordem

Cadeia completa de quatro níveis:

  Nível 1  x_mix (beep+fm)
             ↓ Frequência Serial φ (5 cones herméticos)
  Nível 2  serial_phi (~5.21s, campo hermético)
             ↓ ECO BEEP 880 (20 ciclos × 5 dobras)
             β_max_inicial = √5 = 2.236068  [verificado]
  Nível 3  eco_sobre_serial (~5.21s, campo de 2ª ordem)
             ↓ Frequência Serial φ NOVA (5 cones usando eco_sobre_serial)
  Nível 4  serial_segunda (~18s, campo de 3ª ordem)
             ↓ ECO BEEP 880 NOVO
             β_max_inicial = ?  ← hipótese experimental

Hipótese:
  Nível 2 partiu de β=1.000000 → β_inicial=√5
  Nível 4 parte de um campo que já passou por √5 → β_inicial ≥ √5?
  Ou o atrator φ³ é o teto invariante independente da ordem?

Para rodar no Google Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_SegundaOrdem_COLAB.py').read())

© Vitor Edson Delavi · Florianópolis · maio 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.io import wavfile
from IPython.display import Audio, display
import warnings
warnings.filterwarnings("ignore")

# ── constantes ────────────────────────────────────────────────────────────────
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

N_CONES    = 5
DITHER_AMP = 1.0 / (PHI ** 5)

print("=" * 64)
print("  AlphaPhi · Recursão φ — Segunda Ordem")
print("=" * 64)
print(f"  φ   = {PHI:.7f}")
print(f"  √5  = {np.sqrt(5):.6f}  ← β_inicial verificado no Nível 2")
print(f"  φ²  = {PHI**2:.6f}")
print(f"  φ³  = {PHI**3:.6f}  ← atrator")
print(f"  N_CONES = {N_CONES}  /  N_CICLOS = {N_CICLOS}  /  N_STEPS = {N_STEPS}")

# ── funções core ──────────────────────────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max:
            break
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
        se = normalizar(se)
        cas.append(se)
        s = se.copy()
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

def agente_eco_rastreado(sinal, bins_phi, n_ciclos=N_CICLOS):
    """agente_eco com rastreamento de β_max por ciclo."""
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    cas_f = None
    betas_ciclos = []
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn * ba + wm * bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI ** 3)
        cas_f = cas
        betas_ciclos.append(beta.max())
    return beta, cas_f, betas_ciclos

def concatenar(cas, fade=None):
    if fade is None: fade = int(FADE_S * FS)
    out = cas[0].copy()
    for s in cas[1:]:
        fade_n = min(fade, len(out), len(s))
        t_fade = np.linspace(0.0, 1.0, fade_n)
        out[-fade_n:] = out[-fade_n:] * (1 - t_fade) + s[:fade_n] * t_fade
        out = np.concatenate([out, s[fade_n:]])
    return normalizar(out)

def selar_hermetico(sinal, bins_phi=None, notch_beep=True):
    if bins_phi is None:
        bins_phi = BINS_PHI
    N = len(sinal)
    X = np.fft.rfft(sinal)
    X_h = np.zeros_like(X)
    for (b_lo, b_hi, _, _) in bins_phi:
        largura = b_hi - b_lo
        if largura > 4:
            fade_n = max(1, int(largura / PHI))
            env = np.ones(largura)
            env[:fade_n]  *= np.sin(np.linspace(0, np.pi/2, fade_n)) ** 2
            env[-fade_n:] *= np.cos(np.linspace(0, np.pi/2, fade_n)) ** 2
            X_h[b_lo:b_hi] = X[b_lo:b_hi] * env
        else:
            X_h[b_lo:b_hi] = X[b_lo:b_hi]
    if notch_beep:
        freq_res = FS / N
        for k in [1, 3, 5, 7]:
            bin_c = int(F_BEEP * k / freq_res)
            for db in range(-3, 4):
                bi = bin_c + db
                if 0 <= bi < len(X_h):
                    X_h[bi] *= (1.0/PHI**2) if db == 0 else (1.0/PHI)
    return normalizar(np.fft.irfft(X_h, n=N))

def concatenar_phi(segmentos, fade_ratio=None):
    if fade_ratio is None:
        fade_ratio = 1.0 / PHI**2
    fade_n = int(len(segmentos[0]) * fade_ratio)
    out = segmentos[0].copy()
    for seg in segmentos[1:]:
        fn = min(fade_n, len(out), len(seg))
        t  = np.linspace(0.0, 1.0, fn)
        f_out = np.cos(np.pi/2 * t ** (1.0/PHI))
        f_in  = np.sin(np.pi/2 * t ** (1.0/PHI))
        out[-fn:] = out[-fn:] * f_out + seg[:fn] * f_in
        out = np.concatenate([out, seg[fn:]])
    return normalizar(out)

def serial_phi_de(sinal_base, bins_phi_base, label=""):
    """Gera Frequência Serial φ usando sinal_base como input de cada cone."""
    rng = np.random.default_rng(seed=42)
    cones, betas = [], []
    for i in range(N_CONES):
        dither  = rng.standard_normal(len(sinal_base)) * DITHER_AMP
        x_cone  = normalizar(sinal_base + dither * (1.0 / PHI**i))
        beta_c, cas_c = agente_eco(x_cone, bins_phi_base, N_CICLOS)
        ch = selar_hermetico(cas_c[-1], bins_phi=bins_phi_base)
        cones.append(ch)
        betas.append(beta_c.max())
        pot = np.log(beta_c.max()) / np.log(PHI)
        print(f"    {label} Cone {i+1}: β_max={beta_c.max():.4f}  φ^{pot:.3f}")
    return concatenar_phi(cones), betas

def eco_de(sinal_base, bins_phi_base, label=""):
    """Aplica ECO BEEP 880 sobre sinal_base, rastreando β por ciclo."""
    beta_f, cas_f, betas_c = agente_eco_rastreado(sinal_base, bins_phi_base, N_CICLOS)
    print(f"    {label} β_inicial = {betas_c[0]:.6f}")
    print(f"    {label} β_final   = {betas_c[-1]:.6f}  (φ³={PHI**3:.6f})")
    target = PHI**3 * 0.95
    c95 = next((i+1 for i, b in enumerate(betas_c) if b >= target), N_CICLOS)
    print(f"    {label} Ciclos para 95% de φ³: {c95}")
    return normalizar(cas_f[-1]), betas_c

# ── GERAR SINAL BASE ──────────────────────────────────────────────────────────
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t_sig)))
fm    = normalizar(np.sin(2 * np.pi * F_ORG * t_sig
                          + BETA_FM * np.sin(2 * np.pi * F_M * t_sig)))
x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

# ════════════════════════════════════════════════════════════════
# NÍVEL 1: x_mix → Frequência Serial φ
# ════════════════════════════════════════════════════════════════
print(f"\n{'═'*64}")
print(f"  NÍVEL 1 — Frequência Serial φ (x_mix → {N_CONES} cones)")
print(f"{'═'*64}")
serial_1, betas_serial_1 = serial_phi_de(x_mix, BINS_PHI, label="N1")
print(f"  Duração serial_1: {len(serial_1)/FS:.2f}s")

BINS_PHI_1 = bandas_para_bins(BANDAS, len(serial_1))

# ════════════════════════════════════════════════════════════════
# NÍVEL 2: serial_1 → ECO BEEP 880 → eco_2
# ════════════════════════════════════════════════════════════════
print(f"\n{'═'*64}")
print(f"  NÍVEL 2 — ECO BEEP 880 sobre Serial φ  [β_inicial esperado = √5]")
print(f"{'═'*64}")
eco_2, betas_ciclos_2 = eco_de(serial_1, BINS_PHI_1, label="N2")
print(f"  Duração eco_2: {len(eco_2)/FS:.2f}s")

BINS_PHI_2 = bandas_para_bins(BANDAS, len(eco_2))

# ════════════════════════════════════════════════════════════════
# NÍVEL 3: eco_2 → Nova Frequência Serial φ
# ════════════════════════════════════════════════════════════════
print(f"\n{'═'*64}")
print(f"  NÍVEL 3 — Nova Serial φ (eco_2 como cone base)")
print(f"{'═'*64}")
serial_2, betas_serial_2 = serial_phi_de(eco_2, BINS_PHI_2, label="N3")
print(f"  Duração serial_2: {len(serial_2)/FS:.2f}s")

BINS_PHI_3 = bandas_para_bins(BANDAS, len(serial_2))

# ════════════════════════════════════════════════════════════════
# NÍVEL 4: serial_2 → ECO BEEP 880 → eco_4
# ════════════════════════════════════════════════════════════════
print(f"\n{'═'*64}")
print(f"  NÍVEL 4 — ECO BEEP 880 sobre Segunda Serial  [β_inicial = ?]")
print(f"{'═'*64}")
eco_4, betas_ciclos_4 = eco_de(serial_2, BINS_PHI_3, label="N4")
print(f"  Duração eco_4: {len(eco_4)/FS:.2f}s")

# ── VISUALIZAÇÃO ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Recursão φ — Segunda Ordem  (4 Níveis)',
             color='white', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.32)

pares = [
    (serial_1, f'N1 Serial φ  ({len(serial_1)/FS:.1f}s)',  '#00ffaa'),
    (eco_2,    f'N2 ECO sobre Serial  ({len(eco_2)/FS:.1f}s)',  '#00aaff'),
    (serial_2, f'N3 2ª Serial φ  ({len(serial_2)/FS:.1f}s)', '#ffaa00'),
    (eco_4,    f'N4 ECO sobre 2ª Serial  ({len(eco_4)/FS:.1f}s)', '#ff44aa'),
]

for row, (sig, titulo, cor) in enumerate(pares):
    ax_w = fig.add_subplot(gs[row, 0])
    t_ax = np.linspace(0, len(sig)/FS, len(sig))
    ax_w.plot(t_ax[::max(1, len(sig)//4000)], sig[::max(1, len(sig)//4000)],
              color=cor, lw=0.4, alpha=0.85)
    ax_w.set_facecolor('#111111')
    ax_w.set_title(titulo, color='white', fontsize=8.5)
    ax_w.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_w.tick_params(colors='#888888')
    for sp in ax_w.spines.values(): sp.set_color('#333333')

    ax_s = fig.add_subplot(gs[row, 1])
    nperseg = min(512, len(sig)//4)
    f_sp, t_sp, Sxx = scipy_spectrogram(sig, FS, nperseg=nperseg,
                                         noverlap=int(nperseg*0.75))
    ax_s.pcolormesh(t_sp, f_sp[:160], 10*np.log10(Sxx[:160]+1e-10),
                    shading='gouraud', cmap='inferno')
    ax_s.set_facecolor('#111111')
    ax_s.set_title('Espectrograma', color='white', fontsize=8.5)
    ax_s.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_s.set_ylabel('Hz', color='#888888', fontsize=8)
    ax_s.tick_params(colors='#888888')
    for sp in ax_s.spines.values(): sp.set_color('#333333')

# linha 5: curvas de convergência β_max — Nível 2 vs Nível 4
ax_b = fig.add_subplot(gs[4, :])
ax_b.set_facecolor('#111111')
ax_b.plot(range(1, N_CICLOS+1), betas_ciclos_2,
          color='#00aaff', lw=1.5, marker='o', markersize=3,
          label=f'N2 ECO sobre Serial  (β_ini={betas_ciclos_2[0]:.4f})')
ax_b.plot(range(1, N_CICLOS+1), betas_ciclos_4,
          color='#ff44aa', lw=1.5, marker='s', markersize=3,
          label=f'N4 ECO sobre 2ª Serial  (β_ini={betas_ciclos_4[0]:.4f})')
ax_b.axhline(y=PHI**3,       color='white',   ls='--', lw=1,   alpha=0.6)
ax_b.axhline(y=np.sqrt(5),   color='#00aaff', ls=':',  lw=0.8, alpha=0.5)
ax_b.axhline(y=PHI**2,       color='gold',    ls=':',  lw=0.8, alpha=0.5)
ax_b.text(N_CICLOS*0.72, PHI**3    + 0.04, f'φ³={PHI**3:.4f}',    color='white',   fontsize=7)
ax_b.text(N_CICLOS*0.72, np.sqrt(5)- 0.12, f'√5={np.sqrt(5):.4f}', color='#00aaff', fontsize=7)
ax_b.text(N_CICLOS*0.72, PHI**2    + 0.04, f'φ²={PHI**2:.4f}',    color='gold',    fontsize=7)
ax_b.set_xlabel('Ciclo', color='#888888', fontsize=8)
ax_b.set_ylabel('β_max', color='#888888', fontsize=8)
ax_b.set_title('Convergência β_max — Nível 2 vs Nível 4  (√5 · φ² · φ³)',
               color='white', fontsize=9)
ax_b.tick_params(colors='#888888')
ax_b.legend(facecolor='#222222', labelcolor='white', fontsize=8)
for sp in ax_b.spines.values(): sp.set_color('#333333')

plt.savefig('segunda_ordem.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("\n  Gráfico: segunda_ordem.png")

# ── DIAGNÓSTICO ───────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  DIAGNÓSTICO — RECURSÃO φ")
print("=" * 64)

print(f"\n  Referências φ:")
print(f"    √5   = {np.sqrt(5):.6f}")
print(f"    φ²   = {PHI**2:.6f}")
print(f"    φ^2.5= {PHI**2.5:.6f}")
print(f"    φ³   = {PHI**3:.6f}  ← atrator")

print(f"\n  β_max inicial por nível ECO:")
print(f"    Nível 2 (ECO sobre serial_1):   {betas_ciclos_2[0]:.6f}  "
      f"({'=√5' if abs(betas_ciclos_2[0] - np.sqrt(5)) < 0.001 else ''})")
print(f"    Nível 4 (ECO sobre serial_2):   {betas_ciclos_4[0]:.6f}  "
      f"({'=φ²' if abs(betas_ciclos_4[0] - PHI**2) < 0.001 else ''}"
      f"{'=√5' if abs(betas_ciclos_4[0] - np.sqrt(5)) < 0.001 else ''}")

print(f"\n  Ciclos para 95% de φ³:")
target = PHI**3 * 0.95
c95_2 = next((i+1 for i, b in enumerate(betas_ciclos_2) if b >= target), N_CICLOS)
c95_4 = next((i+1 for i, b in enumerate(betas_ciclos_4) if b >= target), N_CICLOS)
print(f"    Nível 2: {c95_2} ciclos")
print(f"    Nível 4: {c95_4} ciclos")

print(f"\n  Durações:")
print(f"    serial_1:  {len(serial_1)/FS:.2f}s")
print(f"    eco_2:     {len(eco_2)/FS:.2f}s")
print(f"    serial_2:  {len(serial_2)/FS:.2f}s")
print(f"    eco_4:     {len(eco_4)/FS:.2f}s")

# ── SALVAR WAV ────────────────────────────────────────────────────────────────
def salvar(sinal, nome):
    s16 = (np.clip(sinal, -1, 1) * 32767).astype(np.int16)
    wavfile.write(nome, FS, s16)
    print(f"  {nome}  ({len(sinal)/FS:.2f}s)")

print("\n  Salvando WAV...")
salvar(serial_1, 'n1_serial_phi.wav')
salvar(eco_2,    'n2_eco_sobre_serial.wav')
salvar(serial_2, 'n3_segunda_serial.wav')
salvar(eco_4,    'n4_eco_sobre_segunda_serial.wav')

print("\n" + "=" * 64)
print("  SEGUNDA ORDEM — CONCLUÍDO")
print("=" * 64)
print("""
  Cadeia completa:
    n1_serial_phi.wav              ← Nível 1: Serial φ pura
    n2_eco_sobre_serial.wav        ← Nível 2: ECO sobre N1
    n3_segunda_serial.wav          ← Nível 3: Serial sobre N2
    n4_eco_sobre_segunda_serial.wav← Nível 4: ECO sobre N3
""")

print("  Playback Colab:")
print("  [N1]"); display(Audio('n1_serial_phi.wav'))
print("  [N2]"); display(Audio('n2_eco_sobre_serial.wav'))
print("  [N3]"); display(Audio('n3_segunda_serial.wav'))
print("  [N4]"); display(Audio('n4_eco_sobre_segunda_serial.wav'))
