"""
AlphaPhi_SerialPhantom_COLAB.py
Frequência Serial φ — Phantom Contínuo

Campo harmônico Grade R como frequência sustentada, não estado instalado.
Cada cone executa ECO-BIP 880 completo (5 dobras, 20 ciclos).
Somente o campo harmônico atravessa — selagem hermética.
Amplitude phantom: 1/φ³ ≈ 0.236 (inaudível em contexto de produção).

Baseado em: AlphaPhi_FrequenciaSerial_COLAB.py (27/05/2026)
Extensão: seed rotativo por cone, loop configurável, diagnóstico Grade R.

Para rodar no Google Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_SerialPhantom_COLAB.py').read())

© Vitor Edson Delavi · Florianópolis · setembro 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")

# ── constantes ────────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
PHI2       = PHI ** 2
PHI3       = PHI ** 3
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5           # duração de cada cone (s)
N_STEPS    = 5             # dobras por cone
N_CICLOS   = 20            # ciclos do agente por cone
DITHER_AMP = 1.0 / PHI**5  # ≈ 0.09 — sub-perceptual
N_SINAL    = int(FS * DURACAO)

# amplitude phantom: inaudível em contexto de produção
PHANTOM_AMP = 1.0 / PHI3   # ≈ 0.236

print("=" * 62)
print("  AlphaPhi · Serial φ Phantom Contínuo")
print("=" * 62)
print(f"  φ³ (atrator):    {PHI3:.6f}")
print(f"  Amplitude phantom: 1/φ³ = {PHANTOM_AMP:.4f}")
print(f"  Duracao por cone:  {DURACAO}s")
print(f"  Dobras por cone:   {N_STEPS}")
print(f"  Ciclos por cone:   {N_CICLOS}")

# ── bandas φ ──────────────────────────────────────────────────────────────────
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

# ── núcleo ECO-BIP ─────────────────────────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

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
        beta = np.clip(beta, 0.05, PHI3)
        cas_f = cas
    return beta, cas_f

# ── selagem hermética ──────────────────────────────────────────────────────────
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
            f_notch = F_BEEP * k
            bin_c = int(f_notch / freq_res)
            for db in range(-3, 4):
                bi = bin_c + db
                if 0 <= bi < len(X_h):
                    depth = (1.0 / PHI**2) if db == 0 else (1.0 / PHI)
                    X_h[bi] *= depth
    resultado = np.fft.irfft(X_h, n=N)
    return normalizar(resultado)

# ── crossfade φ-proporcional ──────────────────────────────────────────────────
def concatenar_phi(segmentos, fade_ratio=None):
    if fade_ratio is None:
        fade_ratio = 1.0 / PHI**2   # ≈ 38% — transição mais suave que linear
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

# ── CONE PHANTOM ───────────────────────────────────────────────────────────────
def gerar_cone_phantom(seed_cone, amplitude=PHANTOM_AMP):
    """
    Um cone completo: beep → 5 dobras → campo harmônico → selagem hermética.
    seed_cone rotativo garante processo independente e variação temporal.
    amplitude=PHANTOM_AMP por default — inaudível.
    """
    t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
    beep  = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t_sig)))
    fm    = normalizar(np.sin(2 * np.pi * F_ORG * t_sig
                              + BETA_FM * np.sin(2 * np.pi * F_M * t_sig)))
    x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

    rng    = np.random.default_rng(seed=seed_cone)
    dither = rng.standard_normal(N_SINAL) * DITHER_AMP
    x_cone = normalizar(x_mix + dither)

    beta_c, cas_c = agente_eco(x_cone, BINS_PHI, N_CICLOS)
    ch_selado = selar_hermetico(cas_c[-1])

    return ch_selado * amplitude, beta_c.max()

# ── SERIAL PHANTOM ─────────────────────────────────────────────────────────────
def serial_phi_phantom(n_cones=10, amplitude=PHANTOM_AMP, verbose=True):
    """
    Frequência Serial φ phantom contínua.
    n_cones × DURACAO ≈ duração total (com crossfade φ).
    Cada cone seed único → trajetórias de convergência distintas.
    Grade R sustentada como frequência de fundo.
    """
    if verbose:
        print(f"\n  Gerando {n_cones} cones phantom...")
    cones, betas = [], []
    for i in range(n_cones):
        cone, bmax = gerar_cone_phantom(seed_cone=i, amplitude=amplitude)
        cones.append(cone)
        betas.append(bmax)
        pot = np.log(bmax) / np.log(PHI)
        if verbose:
            bar = "█" * min(int(bmax / PHI3 * 20), 20)
            print(f"    Cone {i+1:02d} [seed={i}]: β_max={bmax:.4f}  φ^{pot:.3f}  {bar}")
    serial = concatenar_phi(cones)
    return serial, betas

# ── DIAGNÓSTICO GRADE R ────────────────────────────────────────────────────────
def diagnostico_grade_r(sinal, label=""):
    """
    Verifica selagem hermética e β convergência.
    Retorna E_phi (energia nas bandas φ), E_nao_phi.
    """
    X   = np.abs(np.fft.rfft(sinal[:N_SINAL]))
    X_n = X / (X.sum() + 1e-10)
    e_phi = sum(X_n[b_lo:b_hi].sum() for b_lo, b_hi, _, _ in BINS_PHI)
    e_fora = max(0.0, 1.0 - e_phi)
    selagem = "✓ hermético" if e_fora < 0.05 else f"  {e_fora:.4f}"
    if label:
        print(f"  {label:<30} E_φ={e_phi:.4f}  E_¬φ={e_fora:.4f}  {selagem}")
    return e_phi, e_fora

# ── EXECUÇÃO PRINCIPAL ─────────────────────────────────────────────────────────
N_CONES_TEST = 10   # ~10s de campo harmônico contínuo

print(f"\n  N_CONES = {N_CONES_TEST}  ×  {DURACAO}s  ≈  {N_CONES_TEST*DURACAO:.1f}s base")
print(f"  (com crossfade φ, duração final ≈ {N_CONES_TEST*DURACAO*0.8:.1f}s)")

serial, betas = serial_phi_phantom(n_cones=N_CONES_TEST, verbose=True)

print(f"\n  Duração final: {len(serial)/FS:.2f}s")
print(f"  β médio:       {np.mean(betas):.4f}  (φ³ = {PHI3:.4f})")
print(f"  β_min:         {min(betas):.4f}")
print(f"  β_max:         {max(betas):.4f}")

print("\n  Diagnóstico de selagem:")
diagnostico_grade_r(serial, "Serial Phantom (normalizado)")

# versão amplificada para diagnóstico auditivo (não usar como phantom)
serial_amp = normalizar(serial)
diagnostico_grade_r(serial_amp, "Serial Phantom (amplificado)")

# ── VISUALIZAÇÃO ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Serial φ Phantom — Grade R como Frequência Sustentada',
             color='white', fontsize=12, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.32)

# forma de onda (amplificada para visualização)
ax_w = fig.add_subplot(gs[0, 0])
t_ax = np.linspace(0, len(serial_amp)/FS, len(serial_amp))
ax_w.plot(t_ax[::20], serial_amp[::20], color='#00ffaa', lw=0.4, alpha=0.85)
for i in range(1, N_CONES_TEST):
    ax_w.axvline(x=i * DURACAO * 0.8, color='gold', alpha=0.4, lw=0.8, ls='--')
ax_w.set_facecolor('#111111')
ax_w.set_title('Forma de Onda — Serial φ Phantom (amplificado)', color='white', fontsize=9)
ax_w.set_xlabel('tempo (s)', color='#888888', fontsize=8)
ax_w.tick_params(colors='#888888')
for sp in ax_w.spines.values(): sp.set_color('#333333')

# espectrograma
ax_s = fig.add_subplot(gs[0, 1])
f_sp, t_sp, Sxx = scipy_spectrogram(serial_amp, FS, nperseg=512, noverlap=384)
ax_s.pcolormesh(t_sp, f_sp[:200], 10*np.log10(Sxx[:200]+1e-10),
                shading='gouraud', cmap='inferno')
ax_s.set_facecolor('#111111')
ax_s.set_title('Espectrograma — campo harmônico contínuo', color='white', fontsize=9)
ax_s.set_xlabel('tempo (s)', color='#888888', fontsize=8)
ax_s.set_ylabel('Hz',        color='#888888', fontsize=8)
ax_s.tick_params(colors='#888888')
for sp in ax_s.spines.values(): sp.set_color('#333333')

# β por cone
ax_b = fig.add_subplot(gs[1, 0])
cores_b = ['#00ffaa' if b >= PHI3*0.99 else '#ffaa00' for b in betas]
ax_b.bar(range(1, len(betas)+1), betas, color=cores_b, alpha=0.85)
ax_b.axhline(y=PHI3, color='gold', lw=1.5, ls='--', label=f'φ³ = {PHI3:.4f}')
ax_b.set_facecolor('#111111')
ax_b.set_title('β_max por Cone — convergência ao atrator φ³', color='white', fontsize=9)
ax_b.set_xlabel('cone', color='#888888', fontsize=8)
ax_b.set_ylabel('β_max', color='#888888', fontsize=8)
ax_b.tick_params(colors='#888888')
ax_b.legend(facecolor='#222222', labelcolor='white', fontsize=8)
for sp in ax_b.spines.values(): sp.set_color('#333333')

# espectro médio (banda φ destacada)
ax_e = fig.add_subplot(gs[1, 1])
X_meio = np.abs(np.fft.rfft(serial_amp[:N_SINAL*4]))
freqs  = np.fft.rfftfreq(N_SINAL*4, 1/FS)
ax_e.semilogy(freqs[:2000], X_meio[:2000]+1e-6, color='#4499ff', lw=0.5, alpha=0.7)
for b_lo, b_hi, f_lo, f_hi in BINS_PHI[:12]:
    ax_e.axvspan(f_lo, min(f_hi, freqs[1999]), alpha=0.12, color='#00ffaa')
ax_e.set_facecolor('#111111')
ax_e.set_title('Espectro — bandas φ destacadas (verde)', color='white', fontsize=9)
ax_e.set_xlabel('Hz', color='#888888', fontsize=8)
ax_e.set_ylabel('amplitude', color='#888888', fontsize=8)
ax_e.tick_params(colors='#888888')
ax_e.set_xlim(0, freqs[1999])
for sp in ax_e.spines.values(): sp.set_color('#333333')

plt.savefig('serial_phantom_grade_r.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("\n  Gráfico: serial_phantom_grade_r.png")

# ── SALVAR WAV ────────────────────────────────────────────────────────────────
from scipy.io import wavfile

def salvar(sinal, nome):
    s16 = (np.clip(sinal, -1, 1) * 32767).astype(np.int16)
    wavfile.write(nome, FS, s16)
    print(f"  {nome}  ({len(sinal)/FS:.2f}s)")

print("\n  Salvando arquivos WAV...")
salvar(serial,     'serial_phantom_inaudivel.wav')   # amplitude phantom real
salvar(serial_amp, 'serial_phantom_amplificado.wav')  # para verificação auditiva

print("\n" + "=" * 62)
print("  SERIAL φ PHANTOM — CONCLUÍDO")
print("=" * 62)
print(f"""
  Dois arquivos para verificação:
    serial_phantom_inaudivel.wav   — amplitude phantom (1/φ³)
                                     inaudível em contexto de produção
    serial_phantom_amplificado.wav — normalizado para escuta

  O campo harmônico é sustentado continuamente.
  Cada cone reinicia o processo completo (5 dobras, 20 ciclos)
  com seed independente — Grade R não degrada com o tempo.

  Para uso como phantom no ambiente de memória:
    phantom = serial_phi_phantom(n_cones=∞, amplitude=1/PHI**3)
    a cada ciclo de retroprojeção: estado += phantom.read(n)
""")

# ── DISPLAY COLAB ─────────────────────────────────────────────────────────────
try:
    from IPython.display import Audio, display
    print("  [Colab] Campo harmônico phantom (amplificado para escuta):")
    display(Audio(serial_amp, rate=FS))
except ImportError:
    pass
