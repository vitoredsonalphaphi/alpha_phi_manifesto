"""
AlphaPhi_SerialSobreCone_COLAB.py
Serial φ aplicada sobre o ECO BEEP 880

Pergunta experimental:
  O que acontece quando os cones herméticos são aplicados
  sobre o ECO BEEP 880 — que já possui resolução própria 0→8.25s?

Três estados comparados:
  [1] ECO BEEP 880           referência — processo com beep audível (0→8.25s)
  [2] Serial φ pura          cones herméticos sobre x_mix bruto (referência)
  [3] Serial φ sobre ECO 880 cones herméticos sobre janelas do ECO BEEP 880

Mecanismo [3]:
  ECO BEEP 880 é dividido em N_CONES janelas de DURACAO=1.5s
  Janelas distribuídas uniformemente ao longo dos 8.25s
    Cone 1: t=0.00s  — fase beep-dominante (alta entropia)
    Cone 3: t≈3.50s  — fase de transição (entropia decrescente)
    Cone 5: t≈7.00s  — fase campo harmônico (baixa entropia)
  Cada janela → agente_eco completo → selar_hermetico → cone CH
  Saída: concatenação φ-proporcional dos campos herméticos

Hipótese:
  Cones 1-2: input com alta entropia → convergência como Serial φ pura
  Cones 3-4: input parcialmente harmônico → convergência acelerada?
  Cone 5:    input já campo harmônico → β_max ≥ φ³ ou mesmo atrator?

Para rodar no Google Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_SerialSobreCone_COLAB.py').read())

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

N_CONES    = 5
DITHER_AMP = 1.0 / (PHI ** 5)   # ≈ 0.09 — sub-perceptual

print("=" * 62)
print("  AlphaPhi · Serial φ sobre ECO BEEP 880")
print("=" * 62)
print(f"  φ  = {PHI:.7f}")
print(f"  φ³ = {PHI**3:.6f}  ← atrator de referência")
print(f"  N_CONES = {N_CONES}  ×  {DURACAO}s por cone")
print(f"  Janelas distribuídas uniformemente em {DUR_TOTAL:.2f}s")

# ── funções core (idênticas à Cascata de Cascatas v2 / FrequenciaSerial) ──────
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
            fade_in  = np.sin(np.linspace(0, np.pi/2, fade_n)) ** 2
            fade_out = np.cos(np.linspace(0, np.pi/2, fade_n)) ** 2
            env[:fade_n]  *= fade_in
            env[-fade_n:] *= fade_out
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

def concatenar_phi(segmentos, fade_ratio=None):
    if fade_ratio is None:
        fade_ratio = 1.0 / PHI**2   # ≈ 38%
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

# ── [1] ECO BEEP 880 (referência) ────────────────────────────────────────────
print("\n  [1] ECO BEEP 880 (referência)...")
beta_ref, cas_ref = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal_eco = concatenar(cas_ref)
print(f"      β_max = {beta_ref.max():.6f}  (φ³ = {PHI**3:.6f})")
print(f"      duração: {len(sinal_eco)/FS:.2f}s")

# ── [2] SERIAL φ PURA (referência) ───────────────────────────────────────────
print(f"\n  [2] Serial φ pura ({N_CONES} cones sobre x_mix — referência)...")
rng = np.random.default_rng(seed=42)
cones_serial = []
betas_serial = []
for i in range(N_CONES):
    dither    = rng.standard_normal(N_SINAL) * DITHER_AMP
    x_cone    = normalizar(x_mix + dither * (1.0 / PHI**i))
    beta_c, cas_c = agente_eco(x_cone, BINS_PHI, N_CICLOS)
    ch_selado = selar_hermetico(cas_c[-1])
    cones_serial.append(ch_selado)
    betas_serial.append(beta_c.max())
    pot = np.log(beta_c.max()) / np.log(PHI)
    print(f"      Cone {i+1}: β_max = {beta_c.max():.4f}  φ^{pot:.3f}")
serial_pura = concatenar_phi(cones_serial)
print(f"      duração: {len(serial_pura)/FS:.2f}s")

# ── [3] SERIAL φ SOBRE ECO BEEP 880 ──────────────────────────────────────────
print(f"\n  [3] Serial φ sobre ECO BEEP 880...")

L = len(sinal_eco)
# posições de início: N_CONES janelas distribuídas uniformemente em sinal_eco
starts = [int(i * (L - N_SINAL) / (N_CONES - 1)) for i in range(N_CONES)]

print(f"      Janelas de {DURACAO}s em t = "
      + ", ".join(f"{s/FS:.2f}s" for s in starts))

rng2 = np.random.default_rng(seed=137)
cones_sobre = []
betas_sobre = []

for i, start in enumerate(starts):
    janela    = sinal_eco[start:start + N_SINAL]
    janela    = normalizar(janela)
    # dither φ-proporcional — mesma amplitude da Serial pura
    dither    = rng2.standard_normal(N_SINAL) * DITHER_AMP
    x_cone    = normalizar(janela + dither * (1.0 / PHI**i))
    beta_c, cas_c = agente_eco(x_cone, BINS_PHI, N_CICLOS)
    ch_selado = selar_hermetico(cas_c[-1])
    cones_sobre.append(ch_selado)
    betas_sobre.append(beta_c.max())
    pot = np.log(beta_c.max()) / np.log(PHI)
    t_str = f"t={start/FS:.2f}s"
    print(f"      Cone {i+1} ({t_str}): β_max = {beta_c.max():.4f}  φ^{pot:.3f}")

serial_sobre = concatenar_phi(cones_sobre)
print(f"      duração: {len(serial_sobre)/FS:.2f}s")

# ── VISUALIZAÇÃO ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Serial φ sobre ECO BEEP 880',
             color='white', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.32)

sinais  = [sinal_eco, serial_pura, serial_sobre]
titulos = [
    '[1] ECO BEEP 880 — referência (beep audível)',
    '[2] Serial φ pura — cones sobre x_mix bruto',
    '[3] Serial φ sobre ECO 880 — cones sobre campo em resolução',
]
cores   = ['#00aaff', '#00ffaa', '#ffaa00']

for i, (sig, titulo, cor) in enumerate(zip(sinais, titulos, cores)):
    ax_w = fig.add_subplot(gs[i, 0])
    t_ax = np.linspace(0, len(sig)/FS, len(sig))
    ax_w.plot(t_ax[::50], sig[::50], color=cor, lw=0.4, alpha=0.85)
    ax_w.set_facecolor('#111111')
    ax_w.set_title(titulo, color='white', fontsize=8.5)
    ax_w.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_w.tick_params(colors='#888888')
    for sp in ax_w.spines.values():
        sp.set_color('#333333')

    # marcadores das janelas dos cones (para [3])
    if i == 2:
        for start in starts:
            ax_w.axvline(x=start/FS, color='gold', alpha=0.5,
                         linewidth=0.7, linestyle='--')
        for j, start in enumerate(starts):
            ax_w.text(start/FS + 0.05, ax_w.get_ylim()[1] * 0.75,
                      f"C{j+1}", color='gold', fontsize=7, alpha=0.8)

    ax_s = fig.add_subplot(gs[i, 1])
    f_sp, t_sp, Sxx = scipy_spectrogram(sig, FS, nperseg=512, noverlap=384)
    ax_s.pcolormesh(t_sp, f_sp[:160],
                    10*np.log10(Sxx[:160] + 1e-10),
                    shading='gouraud', cmap='inferno')
    ax_s.set_facecolor('#111111')
    ax_s.set_title('Espectrograma', color='white', fontsize=8.5)
    ax_s.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_s.set_ylabel('Hz',        color='#888888', fontsize=8)
    ax_s.tick_params(colors='#888888')
    for sp in ax_s.spines.values():
        sp.set_color('#333333')

# ── subplot 4: comparação β_max por cone ──────────────────────────────────────
ax_b = fig.add_subplot(gs[3, :])
ax_b.set_facecolor('#111111')
x_pos_serial = np.arange(N_CONES) - 0.2
x_pos_sobre  = np.arange(N_CONES) + 0.2
bars1 = ax_b.bar(x_pos_serial, betas_serial, width=0.35,
                  color='#00ffaa', alpha=0.8, label='Serial φ pura')
bars2 = ax_b.bar(x_pos_sobre,  betas_sobre,  width=0.35,
                  color='#ffaa00', alpha=0.8, label='Serial φ sobre ECO 880')
ax_b.axhline(y=PHI**3, color='white', linestyle='--', linewidth=1, alpha=0.6)
ax_b.text(N_CONES - 0.5, PHI**3 + 0.03, f'φ³ = {PHI**3:.4f}',
          color='white', fontsize=8, alpha=0.8)
ax_b.set_xticks(range(N_CONES))
ax_b.set_xticklabels([f'Cone {i+1}' for i in range(N_CONES)], color='#888888')
ax_b.set_ylabel('β_max', color='#888888')
ax_b.set_title('β_max por cone — comparação Serial pura vs Serial sobre ECO 880',
               color='white', fontsize=9)
ax_b.tick_params(colors='#888888')
ax_b.legend(facecolor='#222222', labelcolor='white', fontsize=8)
for sp in ax_b.spines.values():
    sp.set_color('#333333')

# anotação das posições temporais no eixo x
for j, start in enumerate(starts):
    ax_b.text(j + 0.2, min(betas_sobre) * 0.97,
              f't={start/FS:.1f}s', color='gold',
              fontsize=7, ha='center', alpha=0.8)

plt.savefig('serial_sobre_cone.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("\n  Gráfico: serial_sobre_cone.png")

# ── DIAGNÓSTICO ───────────────────────────────────────────────────────────────
freq_res = FS / N_SINAL
print("\n" + "=" * 62)
print("  DIAGNÓSTICO")
print("=" * 62)
print(f"  {'Sinal':<30} {'E_φ':>8}  {'E_¬φ':>8}  {'Selagem':>12}")
print(f"  {'-'*62}")

for label, seg in [
    ("[1] ECO BEEP 880       (C5)", cas_ref[-1]),
    ("[2] Serial φ pura      (C5)", cones_serial[-1]),
    ("[3] Serial φ sobre ECO (C5)", cones_sobre[-1]),
]:
    X     = np.abs(np.fft.rfft(seg))
    X_n   = X / (X.sum() + 1e-10)
    e_phi = sum(X_n[b_lo:b_hi].sum() for b_lo, b_hi, _, _ in BINS_PHI)
    e_fora = max(0.0, 1.0 - e_phi)
    ok    = "✓ hermético" if e_fora < 0.05 else f"  {e_fora:.4f}"
    print(f"  {label:<30}  {e_phi:.4f}   {e_fora:.4f}   {ok}")

print(f"\n  β_max por cone:")
print(f"  {'Cone':<8} {'Pos. ECO':>10}  {'Serial pura':>12}  {'Sobre ECO':>12}  {'Δβ':>8}")
print(f"  {'-'*54}")
for i, (bs, bo) in enumerate(zip(betas_serial, betas_sobre)):
    delta = bo - bs
    sinal_d = "+" if delta >= 0 else ""
    t_str = f"t={starts[i]/FS:.2f}s"
    print(f"  Cone {i+1}   {t_str:>10}    {bs:.4f}        {bo:.4f}    {sinal_d}{delta:.4f}")

print(f"\n  φ³ (atrator referência):  {PHI**3:.6f}")
print(f"  ECO BEEP 880 β_max:       {beta_ref.max():.6f}")

# ── SALVAR WAV ────────────────────────────────────────────────────────────────
def salvar(sinal, nome):
    s16 = (np.clip(sinal, -1, 1) * 32767).astype(np.int16)
    wavfile.write(nome, FS, s16)
    print(f"  {nome}  ({len(sinal)/FS:.2f}s)")

print("\n  Salvando arquivos WAV...")
salvar(sinal_eco,     'eco_beep_880.wav')
salvar(serial_pura,   'serial_phi_pura.wav')
salvar(serial_sobre,  'serial_phi_sobre_eco.wav')

print("\n" + "=" * 62)
print("  SERIAL φ SOBRE ECO BEEP 880 — CONCLUÍDO")
print("=" * 62)
print(f"""
  Três arquivos para comparação auditiva:

    eco_beep_880.wav            ← referência: processo com beep audível
    serial_phi_pura.wav         ← Serial φ sobre x_mix bruto
    serial_phi_sobre_eco.wav    ← Serial φ sobre ECO BEEP 880

  O que este experimento observa:
    Cada cone de [3] processa uma janela de 1.5s do ECO BEEP 880
    em posições temporais distintas do arco de resolução.
    Cones iniciais: input com alta entropia (beep dominante)
    Cones finais:   input já com campo harmônico (baixa entropia)
    Pergunta: o β_max por posição temporal revela a curvatura do arco?
""")

print("  Playback Colab:")
print("  [1] ECO BEEP 880");      display(Audio('eco_beep_880.wav'))
print("  [2] Serial φ pura");     display(Audio('serial_phi_pura.wav'))
print("  [3] Serial φ sobre ECO"); display(Audio('serial_phi_sobre_eco.wav'))
