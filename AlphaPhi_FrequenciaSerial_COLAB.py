"""
AlphaPhi_FrequenciaSerial_COLAB.py
Frequência Serial φ — Campos Harmônicos Sequenciais Herméticos

Três estados:
  [1] Beep puro          0→8.25s   sem modulação
  [2] ECO BEEP 880       0→8.25s   processo completo (beep audível no início)
  [3] Frequência Serial φ 0→8.25s  N_CONES selados hermeticamente

Selagem hermética:
  Cada cone executa o processo completo (ECO BEEP 880).
  Apenas o resultado φ-ressonante atravessa para a saída.
  O beep e o processo ficam no "terceiro ponto" — sub-perceptual.

Contraste com metafrequência (Cascata de Cascatas v2):
  Metafrequência = 5 instâncias sobrepostas simultaneamente
  Frequência Serial = N_CONES em sequência, cada um completo e hermético

Para rodar no Google Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_FrequenciaSerial_COLAB.py').read())

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

# ── constantes idênticas ao ECO BEEP 880 ──────────────────────────────────────
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

# número de cones na frequência serial
N_CONES    = 5

# perturbação φ por cone: garante processo independente em cada cone
DITHER_AMP = 1.0 / (PHI ** 5)   # ≈ 0.09 — sub-perceptual

print("=" * 60)
print("  AlphaPhi · Frequência Serial φ")
print("=" * 60)
print(f"  φ  = {PHI:.7f}")
print(f"  φ³ = {PHI**3:.6f}")
print(f"  N_CONES   = {N_CONES}  ×  {DURACAO}s  =  {N_CONES*DURACAO:.2f}s base")
print(f"  DUR_TOTAL ≈ {DUR_TOTAL:.2f}s (com crossfade φ)")
print(f"  Dither por cone: {DITHER_AMP:.4f}  (1/φ⁵ — sub-perceptual)")
print(f"  Selagem hermética: bandas φ + notch beep")

# ── funções core (idênticas à Cascata de Cascatas v2) ─────────────────────────
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

def concatenar(cas, fade=None):
    if fade is None: fade = int(FADE_S * FS)
    out = cas[0].copy()
    for s in cas[1:]:
        fade_n = min(fade, len(out), len(s))
        t_fade = np.linspace(0.0, 1.0, fade_n)
        out[-fade_n:] = out[-fade_n:] * (1 - t_fade) + s[:fade_n] * t_fade
        out = np.concatenate([out, s[fade_n:]])
    return normalizar(out)

# ── SELAGEM HERMÉTICA ──────────────────────────────────────────────────────────
def selar_hermetico(sinal, bins_phi=None, notch_beep=True):
    """
    Hermetic sealing:
      1. Mantém apenas componentes φ-ressonantes (máscara espectral φ)
      2. Atenua harmônicos do beep dentro das bandas φ
    O beep e o processo ficam no terceiro ponto da triangulação —
    sub-perceptuais, não atravessam para a saída expressa.
    """
    if bins_phi is None:
        bins_phi = BINS_PHI
    N = len(sinal)
    X = np.fft.rfft(sinal)
    X_h = np.zeros_like(X)

    # Máscara φ com suavização de bordas
    for (b_lo, b_hi, _, _) in bins_phi:
        largura = b_hi - b_lo
        if largura > 4:
            fade_n = max(1, int(largura / PHI))
            env = np.ones(largura)
            fade_in  = np.sin(np.linspace(0, np.pi/2, fade_n)) ** 2
            fade_out = np.cos(np.linspace(0, np.pi/2, fade_n)) ** 2
            env[:fade_n]  *= fade_in
            env[-fade_n:] *= fade_out
            X_h[b_lo:b_hi] = X[b_lo:b_hi] * env
        else:
            X_h[b_lo:b_hi] = X[b_lo:b_hi]

    # Notch suave nos harmônicos ímpares do beep (880, 2640, 4400, 6160 Hz)
    if notch_beep:
        freq_res = FS / N
        for k in [1, 3, 5, 7]:
            f_notch = F_BEEP * k
            bin_c = int(f_notch / freq_res)
            for db in range(-3, 4):
                bi = bin_c + db
                if 0 <= bi < len(X_h):
                    # atenuação φ-proporcional: centro = 1/φ², bordas = 1/φ
                    depth = (1.0 / PHI**2) if db == 0 else (1.0 / PHI)
                    X_h[bi] *= depth

    resultado = np.fft.irfft(X_h, n=N)
    return normalizar(resultado)

# ── CROSSFADE φ-PROPORCIONAL ──────────────────────────────────────────────────
def concatenar_phi(segmentos, fade_ratio=None):
    """
    Stitch com crossfade φ-proporcional.
    Curva: sen/cos com expoente 1/φ — transição mais suave que linear.
    """
    if fade_ratio is None:
        fade_ratio = 1.0 / PHI**2   # ≈ 38% da duração do segmento
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

# ── GERAR SINAL BASE ──────────────────────────────────────────────────────────
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t_sig)))
fm    = normalizar(np.sin(2 * np.pi * F_ORG * t_sig
                          + BETA_FM * np.sin(2 * np.pi * F_M * t_sig)))
x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

# ── [1] BEEP PURO ─────────────────────────────────────────────────────────────
print("\n  [1] Beep puro...")
n_puro   = int(FS * DUR_TOTAL)
t_puro   = np.linspace(0, DUR_TOTAL, n_puro, endpoint=False)
beep_puro = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t_puro)))

# ── [2] ECO BEEP 880 (original) ──────────────────────────────────────────────
print("  [2] ECO BEEP 880...")
beta_ref, cas_ref = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal_eco = concatenar(cas_ref)
print(f"      β_max = {beta_ref.max():.6f}  (φ³ = {PHI**3:.6f})")

# ── [3] FREQUÊNCIA SERIAL φ ───────────────────────────────────────────────────
print(f"  [3] Frequência Serial φ  ({N_CONES} cones herméticos)...")

rng = np.random.default_rng(seed=42)
cones_ch = []
betas_cones = []

for i in range(N_CONES):
    # Cada cone: processo independente com dither φ-proporcional
    dither   = rng.standard_normal(N_SINAL) * DITHER_AMP
    x_cone   = normalizar(x_mix + dither * (1.0 / PHI**i))
    beta_c, cas_c = agente_eco(x_cone, BINS_PHI, N_CICLOS)

    # Extrair campo harmônico (passo final) e selar hermeticamente
    ch_bruto = cas_c[-1]
    ch_selado = selar_hermetico(ch_bruto)
    cones_ch.append(ch_selado)
    betas_cones.append(beta_c.max())

    pot = np.log(beta_c.max()) / np.log(PHI)
    print(f"      Cone {i+1}: β_max = {beta_c.max():.4f}  (φ^{pot:.3f})")

serial_phi = concatenar_phi(cones_ch)
print(f"      Serial φ: {len(serial_phi)/FS:.2f}s")

# ── VISUALIZAÇÃO ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Frequência Serial φ — Selagem Hermética',
             color='white', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.32)

sinais  = [beep_puro,  sinal_eco,  serial_phi]
titulos = [
    '[1] Beep Puro — sinal bruto, sem processo',
    '[2] ECO BEEP 880 — processo completo (beep audível)',
    '[3] Frequência Serial φ — selagem hermética (só C.H.)',
]
cores   = ['#ff4444', '#00aaff', '#00ffaa']

for i, (sig, titulo, cor) in enumerate(zip(sinais, titulos, cores)):
    ax_w = fig.add_subplot(gs[i, 0])
    t_ax = np.linspace(0, len(sig)/FS, len(sig))
    ax_w.plot(t_ax[::50], sig[::50], color=cor, lw=0.4, alpha=0.85)
    ax_w.set_facecolor('#111111')
    ax_w.set_title(titulo, color='white', fontsize=9)
    ax_w.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_w.tick_params(colors='#888888')
    for sp in ax_w.spines.values():
        sp.set_color('#333333')
    if i == 2:
        # Marcadores dos N_CONES na serial
        dur_cone_aprox = len(serial_phi) / FS / N_CONES
        for ci in range(1, N_CONES):
            ax_w.axvline(x=ci * dur_cone_aprox, color='gold',
                         alpha=0.5, linewidth=0.8, linestyle='--')

    ax_s = fig.add_subplot(gs[i, 1])
    f_sp, t_sp, Sxx = scipy_spectrogram(sig, FS, nperseg=512, noverlap=384)
    ax_s.pcolormesh(t_sp, f_sp[:160],
                    10*np.log10(Sxx[:160] + 1e-10),
                    shading='gouraud', cmap='inferno')
    ax_s.set_facecolor('#111111')
    ax_s.set_title('Espectrograma', color='white', fontsize=9)
    ax_s.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_s.set_ylabel('Hz',        color='#888888', fontsize=8)
    ax_s.tick_params(colors='#888888')
    for sp in ax_s.spines.values():
        sp.set_color('#333333')

plt.savefig('frequencia_serial_phi.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("\n  Gráfico: frequencia_serial_phi.png")

# ── DIAGNÓSTICO ───────────────────────────────────────────────────────────────
freq_res = FS / N_SINAL
print("\n" + "=" * 60)
print("  DIAGNÓSTICO — TRÊS ESTADOS")
print("=" * 60)
print(f"  {'Sinal':<22} {'E_φ':>8}  {'E_¬φ':>8}  {'Selagem':>8}")
print(f"  {'-'*50}")

for label, seg in [
    ("[1] Beep Puro    ", beep_puro[:N_SINAL]),
    ("[2] ECO BEEP 880 ", cas_ref[-1]),
    ("[3] Serial φ (C5)", cones_ch[-1]),
]:
    X     = np.abs(np.fft.rfft(seg))
    X_n   = X / (X.sum() + 1e-10)
    e_phi = sum(X_n[b_lo:b_hi].sum() for b_lo, b_hi, _, _ in BINS_PHI)
    e_fora = max(0.0, 1.0 - e_phi)
    ok    = "✓ hermético" if e_fora < 0.05 else f"  {e_fora:.3f}"
    print(f"  {label}  {e_phi:.4f}   {e_fora:.4f}   {ok}")

print(f"\n  β por cone (serial φ):")
for i, bm in enumerate(betas_cones):
    pot = np.log(bm) / np.log(PHI)
    bar = "█" * int(bm / (PHI**3) * 20)
    print(f"    Cone {i+1}: {bm:.4f}  φ^{pot:.3f}  {bar}")

print(f"\n  β_max referência (ECO BEEP 880): {beta_ref.max():.6f}")
print(f"  φ³ (atrator):                    {PHI**3:.6f}")

# ── SALVAR WAV ────────────────────────────────────────────────────────────────
def salvar(sinal, nome):
    s16 = (np.clip(sinal, -1, 1) * 32767).astype(np.int16)
    wavfile.write(nome, FS, s16)
    print(f"  {nome}  ({len(sinal)/FS:.2f}s)")

print("\n  Salvando arquivos WAV...")
salvar(beep_puro,  'beep_puro.wav')
salvar(sinal_eco,  'eco_beep_880.wav')
salvar(serial_phi, 'frequencia_serial_phi.wav')

print("\n" + "=" * 60)
print("  FREQUÊNCIA SERIAL φ — CONCLUÍDO")
print("=" * 60)
print(f"""
  Três arquivos para comparação auditiva:

    beep_puro.wav              ← sinal bruto, sem processo
    eco_beep_880.wav           ← processo com beep audível
    frequencia_serial_phi.wav  ← processo com selagem hermética

  O que muda da tentativa anterior (beep estendido):
    Tentativa anterior: capturou só o estado C.H. → sem propulsão → cortes
    Frequência Serial:  cada cone executa processo completo (beep → C.H.)
                        somente o C.H. hermético atravessa para a saída
                        crossfade φ elimina rupturas entre cones

  Linha expressa  →  frequencia_serial_phi.wav  (só C.H.)
  Linhas herméticas → beep + processo (sub-perceptuais, não gravados)
""")

print("  Playback Colab:")
print("  [1]"); display(Audio('beep_puro.wav'))
print("  [2]"); display(Audio('eco_beep_880.wav'))
print("  [3]"); display(Audio('frequencia_serial_phi.wav'))
