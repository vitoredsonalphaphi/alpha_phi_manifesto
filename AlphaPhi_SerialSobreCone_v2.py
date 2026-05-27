"""
AlphaPhi_SerialSobreCone_v2.py
Serial φ sobre o Cone — ECO BEEP 880 como base dos cones herméticos

Pergunta experimental:
  O que acontece quando os cones da Serial φ partem do ECO BEEP 880
  (campo parcialmente harmônico, 8.25s) como sinal base — em vez de
  partir do x_mix bruto?

Três estados comparados:
  [1] ECO BEEP 880           referência (x_mix → 5 dobras, 8.25s)
  [2] Serial φ pura          referência (x_mix → 5 cones, 5.21s)
  [3] Serial φ sobre Cone    sinal_eco como base dos N_CONES cones

Mecanismo [3]:
  sinal_eco (8.25s) é o sinal base de cada cone
  N_CONES cones: agente_eco(sinal_eco + dither, BINS_ECO, N_CICLOS)
  BINS_ECO recalculado para o comprimento de sinal_eco
  Cada campo hermético: selar_hermetico com bandas do comprimento ECO
  Saída: concatenação φ-proporcional

Hipótese:
  Os cones partem de um input mais harmônico que x_mix →
  β por cone converge mais rápido que na Serial pura?
  O campo resultante: β_inicial para ECO = √5 (invariante)?
  Ou a impureza de sinal_eco (não hermeticamente selado) quebra o invariante?

Diferença da v1 (SerialSobreCone original):
  v1: janelas de 1.5s do sinal_eco como input (fragmentos)
  v2: sinal_eco completo (8.25s) como input de cada cone

Para rodar no Google Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_SerialSobreCone_v2.py').read())

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
DUR_TOTAL  = DURACAO * (N_STEPS + 1) - FADE_S * N_STEPS

N_CONES    = 5
DITHER_AMP = 1.0 / (PHI ** 5)

print("=" * 62)
print("  AlphaPhi · Serial φ sobre Cone (v2)")
print("=" * 62)
print(f"  φ  = {PHI:.7f}")
print(f"  φ³ = {PHI**3:.6f}  ← atrator")
print(f"  Diferença da v1: sinal_eco COMPLETO como base dos cones")

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
        env = np.clip(1.0 + (ce * PHI**bi) * np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)
    r = np.fft.irfft(F_out, n=N)
    return r / (np.max(np.abs(r)) + 1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    cm = np.zeros(len(bins_phi))
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se); cas.append(se); s = se.copy()
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
        beta = np.clip(beta, 0.05, PHI**3)
        cas_f = cas
    return beta, cas_f

def concatenar(cas, fade=None):
    if fade is None: fade = int(FADE_S * FS)
    out = cas[0].copy()
    for s in cas[1:]:
        fn = min(fade, len(out), len(s))
        t  = np.linspace(0.0, 1.0, fn)
        out[-fn:] = out[-fn:] * (1-t) + s[:fn] * t
        out = np.concatenate([out, s[fn:]])
    return normalizar(out)

def selar_hermetico(sinal, bins_phi=None):
    if bins_phi is None: bins_phi = BINS_PHI
    N = len(sinal)
    X = np.fft.rfft(sinal)
    X_h = np.zeros_like(X)
    for (b_lo, b_hi, _, _) in bins_phi:
        larg = b_hi - b_lo
        if larg > 4:
            fn = max(1, int(larg / PHI))
            env = np.ones(larg)
            env[:fn]  *= np.sin(np.linspace(0, np.pi/2, fn))**2
            env[-fn:] *= np.cos(np.linspace(0, np.pi/2, fn))**2
            X_h[b_lo:b_hi] = X[b_lo:b_hi] * env
        else:
            X_h[b_lo:b_hi] = X[b_lo:b_hi]
    freq_res = FS / N
    for k in [1, 3, 5, 7]:
        bc = int(F_BEEP * k / freq_res)
        for db in range(-3, 4):
            bi = bc + db
            if 0 <= bi < len(X_h):
                X_h[bi] *= (1/PHI**2) if db == 0 else (1/PHI)
    return normalizar(np.fft.irfft(X_h, n=N))

def concatenar_phi(segs, fade_ratio=None):
    if fade_ratio is None: fade_ratio = 1.0 / PHI**2
    fade_n = int(len(segs[0]) * fade_ratio)
    out = segs[0].copy()
    for seg in segs[1:]:
        fn = min(fade_n, len(out), len(seg))
        t  = np.linspace(0.0, 1.0, fn)
        fo = np.cos(np.pi/2 * t**(1/PHI))
        fi = np.sin(np.pi/2 * t**(1/PHI))
        out[-fn:] = out[-fn:] * fo + seg[:fn] * fi
        out = np.concatenate([out, seg[fn:]])
    return normalizar(out)

# ── GERAR SINAL BASE ──────────────────────────────────────────────────────────
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_sig)))
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_sig + BETA_FM*np.sin(2*np.pi*F_M*t_sig)))
x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

# ── [1] ECO BEEP 880 ──────────────────────────────────────────────────────────
print("\n  [1] ECO BEEP 880 (referência)...")
beta_ref, cas_ref = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal_eco = concatenar(cas_ref)
print(f"      β_max = {beta_ref.max():.6f}  /  duração: {len(sinal_eco)/FS:.2f}s")

# ── [2] SERIAL φ PURA (referência) ───────────────────────────────────────────
print(f"\n  [2] Serial φ pura (referência)...")
rng = np.random.default_rng(seed=42)
cones_pura, betas_pura = [], []
for i in range(N_CONES):
    d = rng.standard_normal(N_SINAL) * DITHER_AMP
    x = normalizar(x_mix + d * (1/PHI**i))
    bc, cc = agente_eco(x, BINS_PHI, N_CICLOS)
    ch = selar_hermetico(cc[-1], BINS_PHI)
    cones_pura.append(ch); betas_pura.append(bc.max())
serial_pura = concatenar_phi(cones_pura)
print(f"      β_max médio = {np.mean(betas_pura):.4f}  /  duração: {len(serial_pura)/FS:.2f}s")

# ── [3] SERIAL φ SOBRE CONE (v2 — sinal_eco completo) ────────────────────────
print(f"\n  [3] Serial φ sobre Cone v2 (sinal_eco completo como base)...")
N_ECO      = len(sinal_eco)
BINS_ECO   = bandas_para_bins(BANDAS, N_ECO)
print(f"      sinal_eco: {N_ECO} samples ({N_ECO/FS:.2f}s)  /  bandas φ: {len(BINS_ECO)}")

rng2 = np.random.default_rng(seed=99)
cones_sc, betas_sc = [], []
for i in range(N_CONES):
    d  = rng2.standard_normal(N_ECO) * DITHER_AMP
    xc = normalizar(sinal_eco + d * (1/PHI**i))
    bc, cc = agente_eco(xc, BINS_ECO, N_CICLOS)
    ch = selar_hermetico(cc[-1], BINS_ECO)
    cones_sc.append(ch); betas_sc.append(bc.max())
    pot = np.log(bc.max()) / np.log(PHI)
    print(f"      Cone {i+1}: β_max={bc.max():.4f}  φ^{pot:.3f}")

serial_sc = concatenar_phi(cones_sc)
print(f"      duração: {len(serial_sc)/FS:.2f}s")

# ── verificar invariante √5 ───────────────────────────────────────────────────
print(f"\n  Verificando invariante √5 no campo Serial sobre Cone...")
BINS_SC  = bandas_para_bins(BANDAS, len(serial_sc))
nb_sc    = len(BINS_SC)
beta_chk = np.ones(nb_sc); bm_chk = beta_chk.copy()
wm, wn   = 1/PHI, 1-1/PHI
cas_chk, cohs_chk = cascata_eq(serial_sc, beta_chk, BINS_SC)
cr_chk   = (cohs_chk - cohs_chk.min()) / (cohs_chk.max() - cohs_chk.min() + 1e-10)
ba_chk   = PHI**(3 * cr_chk)
beta_1ciclo = (wn * ba_chk + wm * bm_chk).max()
print(f"      β_max ciclo 1 = {beta_1ciclo:.6f}")
print(f"      √5            = {np.sqrt(5):.6f}  {'← INVARIANTE MANTIDO' if abs(beta_1ciclo - np.sqrt(5)) < 0.01 else '← INVARIANTE QUEBRADO'}")

# ── VISUALIZAÇÃO ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Serial φ sobre Cone v2 — sinal_eco como base',
             color='white', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.32)

pares = [
    (sinal_eco,  '[1] ECO BEEP 880 — referência', '#00aaff'),
    (serial_pura,'[2] Serial φ pura — referência', '#00ffaa'),
    (serial_sc,  '[3] Serial φ sobre Cone v2',    '#ff8800'),
]
for row, (sig, titulo, cor) in enumerate(pares):
    ax_w = fig.add_subplot(gs[row, 0])
    t_ax = np.linspace(0, len(sig)/FS, len(sig))
    ax_w.plot(t_ax[::max(1,len(sig)//4000)], sig[::max(1,len(sig)//4000)],
              color=cor, lw=0.4, alpha=0.85)
    ax_w.set_facecolor('#111111'); ax_w.set_title(titulo, color='white', fontsize=8.5)
    ax_w.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_w.tick_params(colors='#888888')
    for sp in ax_w.spines.values(): sp.set_color('#333333')

    ax_s = fig.add_subplot(gs[row, 1])
    np2 = min(512, len(sig)//4)
    f_sp, t_sp, Sxx = scipy_spectrogram(sig, FS, nperseg=np2, noverlap=int(np2*0.75))
    ax_s.pcolormesh(t_sp, f_sp[:160], 10*np.log10(Sxx[:160]+1e-10),
                    shading='gouraud', cmap='inferno')
    ax_s.set_facecolor('#111111'); ax_s.set_title('Espectrograma', color='white', fontsize=8.5)
    ax_s.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_s.set_ylabel('Hz', color='#888888', fontsize=8)
    ax_s.tick_params(colors='#888888')
    for sp in ax_s.spines.values(): sp.set_color('#333333')

plt.savefig('serial_sobre_cone_v2.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("\n  Gráfico: serial_sobre_cone_v2.png")

# ── DIAGNÓSTICO ───────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  DIAGNÓSTICO")
print("="*62)
print(f"  β_max por cone:")
print(f"  {'Cone':<6} {'Serial pura':>12}  {'Serial s/ Cone':>14}")
print(f"  {'-'*36}")
for i, (bp, bs) in enumerate(zip(betas_pura, betas_sc)):
    print(f"  {i+1:<6} {bp:.4f}        {bs:.4f}")
print(f"\n  Invariante √5 no campo Serial sobre Cone:")
print(f"  β_max ciclo 1 = {beta_1ciclo:.6f}  (√5 = {np.sqrt(5):.6f})")

def salvar(sig, nome):
    s16 = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    wavfile.write(nome, FS, s16)
    print(f"  {nome}  ({len(sig)/FS:.2f}s)")

print("\n  Salvando WAV...")
salvar(sinal_eco,   'eco_beep_880.wav')
salvar(serial_pura, 'serial_phi_pura.wav')
salvar(serial_sc,   'serial_sobre_cone_v2.wav')

print("\n  Playback:")
print("  [1]"); display(Audio('eco_beep_880.wav'))
print("  [2]"); display(Audio('serial_phi_pura.wav'))
print("  [3]"); display(Audio('serial_sobre_cone_v2.wav'))
