"""
AlphaPhi_EcoSobreSerial_COLAB.py
ECO BEEP 880 aplicado sobre a Frequência Serial φ

Pergunta experimental:
  O que acontece quando o processo ECO BEEP 880 — com seus 5 pontos
  de dobra progressivos — é aplicado sobre a Frequência Serial φ
  como sinal de entrada?

  Na forma original:
    x_mix (beep+fm, alta entropia) → ECO BEEP 880 → campo harmônico

  Neste experimento:
    serial_phi (campo harmônico, baixa entropia) → ECO BEEP 880 → ?

Estados produzidos:
  [1] ECO BEEP 880 original   — referência (x_mix → 5 dobras)
  [2] Frequência Serial φ     — referência (N_CONES herméticos)
  [3] Eco sobre Serial        — serial_phi → 5 dobras progressivas
                                Arco completo: step 0 → step 5

  O resultado [3] não tem a plástica do ECO BEEP 880 (que parte de beep bruto)
  nem a plástica da Serial φ (que é sequência de cones herméticos).
  É o processo progressivo do ECO BEEP 880 agindo sobre um campo
  que já possui coerência φ.

Hipótese:
  Input com coerência φ → agente_eco parte de β já elevado
  → convergência mais rápida? β_max ≥ φ³? ou mesmo atrator?
  A primeira dobra de [3] já parte de onde [1] termina?

Para rodar no Google Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_EcoSobreSerial_COLAB.py').read())

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
DUR_TOTAL  = DURACAO * (N_STEPS + 1) - FADE_S * N_STEPS   # 8.25s

N_CONES    = 5
DITHER_AMP = 1.0 / (PHI ** 5)   # ≈ 0.09

print("=" * 62)
print("  AlphaPhi · ECO BEEP 880 sobre Frequência Serial φ")
print("=" * 62)
print(f"  φ  = {PHI:.7f}")
print(f"  φ³ = {PHI**3:.6f}  ← atrator de referência")
print(f"  N_CONES Serial = {N_CONES} · N_STEPS ECO = {N_STEPS}")

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

# ── GERAR SINAL BASE ──────────────────────────────────────────────────────────
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t_sig)))
fm    = normalizar(np.sin(2 * np.pi * F_ORG * t_sig
                          + BETA_FM * np.sin(2 * np.pi * F_M * t_sig)))
x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

# ── [1] ECO BEEP 880 (referência) ────────────────────────────────────────────
print("\n  [1] ECO BEEP 880 (referência, x_mix → 5 dobras)...")
beta_ref, cas_ref = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal_eco = concatenar(cas_ref)
print(f"      β_max = {beta_ref.max():.6f}  /  φ³ = {PHI**3:.6f}")
print(f"      duração: {len(sinal_eco)/FS:.2f}s")

# ── [2] FREQUÊNCIA SERIAL φ (referência) ─────────────────────────────────────
print(f"\n  [2] Frequência Serial φ ({N_CONES} cones herméticos — referência)...")
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
serial_phi = concatenar_phi(cones_serial)
print(f"      β_max médio = {np.mean(betas_serial):.4f}")
print(f"      duração: {len(serial_phi)/FS:.2f}s")

# ── [3] ECO BEEP 880 SOBRE SERIAL φ ──────────────────────────────────────────
print(f"\n  [3] ECO BEEP 880 sobre Serial φ...")
print(f"      Input: serial_phi ({len(serial_phi)/FS:.2f}s, campo harmônico)")
print(f"      Processo: agente_eco com BINS_PHI adaptado ao comprimento serial")

# BINS_PHI recalculado para o comprimento da serial_phi
N_SERIAL       = len(serial_phi)
BINS_PHI_SER   = bandas_para_bins(BANDAS, N_SERIAL)
print(f"      N_SERIAL = {N_SERIAL} samples  ({N_SERIAL/FS:.2f}s)")
print(f"      Número de bandas φ: {len(BINS_PHI_SER)}")

# acompanhar β_max por ciclo durante a convergência
print(f"      Rodando {N_CICLOS} ciclos × {N_STEPS} dobras...")

nb_ser = len(BINS_PHI_SER)
beta_s = np.ones(nb_ser)
bm_s   = beta_s.copy()
wm_s, wn_s = 1.0 / PHI, 1.0 - 1.0 / PHI
cas_final = None
betas_ciclos = []   # β_max por ciclo — curva de convergência

for ciclo in range(N_CICLOS):
    cas, cohs = cascata_eq(serial_phi, beta_s, BINS_PHI_SER)
    cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
    ba   = PHI ** (3 * cr)
    beta_s = wn_s * ba + wm_s * bm_s; bm_s = beta_s.copy()
    beta_s = np.clip(beta_s, 0.05, PHI ** 3)
    cas_final = cas
    betas_ciclos.append(beta_s.max())

print(f"      β_max inicial (ciclo 1):  {betas_ciclos[0]:.6f}")
print(f"      β_max final   (ciclo {N_CICLOS}): {betas_ciclos[-1]:.6f}")
print(f"      φ³ referência:             {PHI**3:.6f}")

# arco progressivo: 6 steps (step0=serial_phi, steps 1-5 = dobras progressivas)
# cada step tem duração de serial_phi (~5.21s)
eco_serial_arco = concatenar(cas_final)
dur_arco = len(eco_serial_arco) / FS
print(f"\n      Arco completo (6 steps × {len(serial_phi)/FS:.2f}s): {dur_arco:.2f}s")

# step final isolado — resultado da 5ª dobra sobre serial_phi
eco_serial_campo = normalizar(cas_final[-1])
print(f"      Campo final (step 5):  {len(eco_serial_campo)/FS:.2f}s")

# ── VISUALIZAÇÃO ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 16), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · ECO BEEP 880 sobre Frequência Serial φ',
             color='white', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.32)

# linha 1: ECO BEEP 880 referência
ax_w1 = fig.add_subplot(gs[0, 0])
t1 = np.linspace(0, len(sinal_eco)/FS, len(sinal_eco))
ax_w1.plot(t1[::50], sinal_eco[::50], color='#00aaff', lw=0.4, alpha=0.85)
ax_w1.set_facecolor('#111111')
ax_w1.set_title('[1] ECO BEEP 880 — referência (x_mix → 5 dobras)',
                color='white', fontsize=8.5)
ax_w1.set_xlabel('tempo (s)', color='#888888', fontsize=8)
ax_w1.tick_params(colors='#888888')
for sp in ax_w1.spines.values(): sp.set_color('#333333')

ax_s1 = fig.add_subplot(gs[0, 1])
f_sp, t_sp, Sxx = scipy_spectrogram(sinal_eco, FS, nperseg=512, noverlap=384)
ax_s1.pcolormesh(t_sp, f_sp[:160], 10*np.log10(Sxx[:160]+1e-10),
                 shading='gouraud', cmap='inferno')
ax_s1.set_facecolor('#111111')
ax_s1.set_title('Espectrograma', color='white', fontsize=8.5)
ax_s1.set_xlabel('tempo (s)', color='#888888', fontsize=8)
ax_s1.set_ylabel('Hz', color='#888888', fontsize=8)
ax_s1.tick_params(colors='#888888')
for sp in ax_s1.spines.values(): sp.set_color('#333333')

# linha 2: Serial φ referência
ax_w2 = fig.add_subplot(gs[1, 0])
t2 = np.linspace(0, len(serial_phi)/FS, len(serial_phi))
ax_w2.plot(t2[::50], serial_phi[::50], color='#00ffaa', lw=0.4, alpha=0.85)
ax_w2.set_facecolor('#111111')
ax_w2.set_title('[2] Frequência Serial φ — referência (N_CONES herméticos)',
                color='white', fontsize=8.5)
ax_w2.set_xlabel('tempo (s)', color='#888888', fontsize=8)
ax_w2.tick_params(colors='#888888')
for sp in ax_w2.spines.values(): sp.set_color('#333333')

ax_s2 = fig.add_subplot(gs[1, 1])
f_sp, t_sp, Sxx = scipy_spectrogram(serial_phi, FS, nperseg=512, noverlap=384)
ax_s2.pcolormesh(t_sp, f_sp[:160], 10*np.log10(Sxx[:160]+1e-10),
                 shading='gouraud', cmap='inferno')
ax_s2.set_facecolor('#111111')
ax_s2.set_title('Espectrograma', color='white', fontsize=8.5)
ax_s2.set_xlabel('tempo (s)', color='#888888', fontsize=8)
ax_s2.set_ylabel('Hz', color='#888888', fontsize=8)
ax_s2.tick_params(colors='#888888')
for sp in ax_s2.spines.values(): sp.set_color('#333333')

# linha 3: Eco sobre Serial — campo final (step 5)
ax_w3 = fig.add_subplot(gs[2, 0])
t3 = np.linspace(0, len(eco_serial_campo)/FS, len(eco_serial_campo))
ax_w3.plot(t3[::50], eco_serial_campo[::50], color='#ffaa00', lw=0.4, alpha=0.85)
ax_w3.set_facecolor('#111111')
ax_w3.set_title('[3] ECO sobre Serial — campo final (5ª dobra sobre serial_phi)',
                color='white', fontsize=8.5)
ax_w3.set_xlabel('tempo (s)', color='#888888', fontsize=8)
ax_w3.tick_params(colors='#888888')
for sp in ax_w3.spines.values(): sp.set_color('#333333')

ax_s3 = fig.add_subplot(gs[2, 1])
f_sp, t_sp, Sxx = scipy_spectrogram(eco_serial_campo, FS, nperseg=512, noverlap=384)
ax_s3.pcolormesh(t_sp, f_sp[:160], 10*np.log10(Sxx[:160]+1e-10),
                 shading='gouraud', cmap='inferno')
ax_s3.set_facecolor('#111111')
ax_s3.set_title('Espectrograma', color='white', fontsize=8.5)
ax_s3.set_xlabel('tempo (s)', color='#888888', fontsize=8)
ax_s3.set_ylabel('Hz', color='#888888', fontsize=8)
ax_s3.tick_params(colors='#888888')
for sp in ax_s3.spines.values(): sp.set_color('#333333')

# linha 4: curva de convergência β_max por ciclo
ax_b = fig.add_subplot(gs[3, :])
ax_b.set_facecolor('#111111')
ax_b.plot(range(1, N_CICLOS+1), betas_ciclos,
          color='#ffaa00', lw=1.5, marker='o', markersize=3, label='ECO sobre Serial')
ax_b.axhline(y=PHI**3, color='white', linestyle='--', linewidth=1, alpha=0.6)
ax_b.axhline(y=beta_ref.max(), color='#00aaff', linestyle=':', linewidth=1, alpha=0.6)
ax_b.text(N_CICLOS * 0.75, PHI**3 + 0.04,
          f'φ³ = {PHI**3:.4f}', color='white', fontsize=8, alpha=0.8)
ax_b.text(N_CICLOS * 0.75, beta_ref.max() - 0.08,
          f'ECO BEEP 880: {beta_ref.max():.4f}', color='#00aaff', fontsize=8, alpha=0.8)
ax_b.set_xlabel('Ciclo (iteração do agente_eco)', color='#888888', fontsize=8)
ax_b.set_ylabel('β_max', color='#888888', fontsize=8)
ax_b.set_title('Convergência β_max — ECO sobre Serial φ (20 ciclos × 5 dobras)',
               color='white', fontsize=9)
ax_b.tick_params(colors='#888888')
ax_b.legend(facecolor='#222222', labelcolor='white', fontsize=8)
for sp in ax_b.spines.values(): sp.set_color('#333333')

plt.savefig('eco_sobre_serial.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("\n  Gráfico: eco_sobre_serial.png")

# ── DIAGNÓSTICO ───────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  DIAGNÓSTICO")
print("=" * 62)
print(f"  {'Sinal':<32} {'E_φ':>8}  {'E_¬φ':>8}  {'β_max':>8}")
print(f"  {'-'*60}")

for label, seg, bm in [
    ("[1] ECO BEEP 880 (campo final)", cas_ref[-1], beta_ref.max()),
    ("[2] Serial φ (cone 5)",          cones_serial[-1], betas_serial[-1]),
    ("[3] ECO sobre Serial (step 5)",  eco_serial_campo, betas_ciclos[-1]),
]:
    X     = np.abs(np.fft.rfft(seg[:N_SINAL]))  # primeiro segmento para comparação uniforme
    X_n   = X / (X.sum() + 1e-10)
    e_phi = sum(X_n[b_lo:b_hi].sum() for b_lo, b_hi, _, _ in BINS_PHI)
    e_fora = max(0.0, 1.0 - e_phi)
    print(f"  {label:<32}  {e_phi:.4f}   {e_fora:.4f}   {bm:.4f}")

print(f"\n  Convergência β_max (ECO sobre Serial):")
print(f"  Ciclo  1: {betas_ciclos[0]:.6f}  (ECO BEEP 880 parte de: 1.000000)")
for c in [4, 9, 14, 19]:
    if c < len(betas_ciclos):
        print(f"  Ciclo {c+1:2d}: {betas_ciclos[c]:.6f}")
print(f"  φ³:        {PHI**3:.6f}")

# ciclos para atingir 95% de φ³
target = PHI**3 * 0.95
ciclos_95 = next((i+1 for i, b in enumerate(betas_ciclos) if b >= target), N_CICLOS)
print(f"\n  Ciclos para atingir 95% de φ³: {ciclos_95}  (ECO BEEP 880 padrão: ~{N_CICLOS})")

# ── SALVAR WAV ────────────────────────────────────────────────────────────────
def salvar(sinal, nome):
    s16 = (np.clip(sinal, -1, 1) * 32767).astype(np.int16)
    wavfile.write(nome, FS, s16)
    print(f"  {nome}  ({len(sinal)/FS:.2f}s)")

print("\n  Salvando arquivos WAV...")
salvar(sinal_eco,          'eco_beep_880.wav')
salvar(serial_phi,         'frequencia_serial_phi.wav')
salvar(eco_serial_campo,   'eco_sobre_serial_campo.wav')    # step final
salvar(eco_serial_arco,    'eco_sobre_serial_arco.wav')     # arco completo 6 steps

print("\n" + "=" * 62)
print("  ECO SOBRE SERIAL — CONCLUÍDO")
print("=" * 62)
print(f"""
  Quatro arquivos para comparação auditiva:

    eco_beep_880.wav              ← [1] referência ECO BEEP 880
    frequencia_serial_phi.wav     ← [2] referência Serial φ
    eco_sobre_serial_campo.wav    ← [3] 5ª dobra sobre serial_phi
    eco_sobre_serial_arco.wav     ← [3] arco completo (step 0→5, {dur_arco:.1f}s)

  eco_sobre_serial_campo.wav: o resultado da aplicação do ECO BEEP 880
  à Frequência Serial φ — a 5ª dobra do processo progressivo sobre
  um sinal que já possui coerência φ.

  eco_sobre_serial_arco.wav: o arco inteiro — step 0 (serial_phi original)
  progressindo pelos 5 pontos de dobra até o campo de segunda ordem.
  A plástica deste arco é diferente de ambas as referências.
""")

print("  Playback Colab:")
print("  [1]"); display(Audio('eco_beep_880.wav'))
print("  [2]"); display(Audio('frequencia_serial_phi.wav'))
print("  [3a] campo final"); display(Audio('eco_sobre_serial_campo.wav'))
print("  [3b] arco completo"); display(Audio('eco_sobre_serial_arco.wav'))
