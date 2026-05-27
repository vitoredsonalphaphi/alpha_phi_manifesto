"""
AlphaPhi_HiperbolicoCadeia_COLAB.py
Cadeia Hiperbólica Completa — ECO hiperbólico → Serial φ → ECO

Cadeia para cada teto de β:

  serial_phi (√5-coerente)
      ↓ ECO hiperbólico  (teto: φ³ / φ⁴ / φ⁶ / sem teto)
  eco_hiper_[X]
      ↓ Nova Serial φ  (N_CONES herméticos sobre eco_hiper)
  serial_hiper_[X]
      ↓ ECO BEEP 880  (teto φ³ — operador organizador final)
  eco_final_[X]

Princípio (Entrada 71):
  Serial φ prepara. ECO organiza.
  Válido mesmo para campos hiperbólicos (β > φ³)?

Hipóteses:
  1. Invariante √5: Serial φ sobre eco_hiper mantém β_inicial=√5?
     (verificado para x_mix, serial_phi, sinal_eco — agora para campo hiperbólico)
  2. ECO final: β_max = φ³ mesmo partindo de campo hiperbólico?
  3. O teto hiperbólico afeta o resultado final ou o ECO final sempre
     reconduz ao mesmo atrator φ³?

Para rodar no Google Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_HiperbolicoCadeia_COLAB.py').read())

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
print("  AlphaPhi · Cadeia Hiperbólica Completa")
print("=" * 64)
print(f"  φ³ = {PHI**3:.4f}  φ⁴ = {PHI**4:.4f}  φ⁶ = {PHI**6:.4f}  ∞ = sem teto")
print(f"  Cadeia: serial_phi → ECO(teto) → Serial φ → ECO(φ³)")
print(f"  4 tetos × 3 estágios = 12 agente_eco execuções")

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

def agente_eco_rastreado(sinal, bins_phi, n_ciclos=N_CICLOS, beta_teto=None):
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    cas_f, betas_c = None, []
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn * ba + wm * bm; bm = beta.copy()
        if beta_teto is not None:
            beta = np.clip(beta, 0.05, beta_teto)
        else:
            beta = np.clip(beta, 0.05, None)
        cas_f = cas
        betas_c.append(beta.max())
    return beta, cas_f, betas_c

def selar_hermetico(sinal, bins_phi=None):
    if bins_phi is None: bins_phi = BINS_PHI
    N = len(sinal); X = np.fft.rfft(sinal); X_h = np.zeros_like(X)
    for (b_lo, b_hi, _, _) in bins_phi:
        larg = b_hi - b_lo
        if larg > 4:
            fn = max(1, int(larg / PHI)); env = np.ones(larg)
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

def serial_de(sinal_base, bins_base, label=""):
    rng = np.random.default_rng(seed=42)
    cones, betas = [], []
    for i in range(N_CONES):
        d  = rng.standard_normal(len(sinal_base)) * DITHER_AMP
        xc = normalizar(sinal_base + d * (1/PHI**i))
        _, cas, _ = agente_eco_rastreado(xc, bins_base, N_CICLOS, beta_teto=PHI**3)
        ch = selar_hermetico(cas[-1], bins_base)
        cones.append(ch)
        betas.append(_[...].max() if hasattr(_,'max') else PHI**3)
    return concatenar_phi(cones)

# ── GERAR SERIAL φ BASE ───────────────────────────────────────────────────────
print("\n  Gerando serial_phi (input √5-coerente)...")
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_sig)))
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_sig + BETA_FM*np.sin(2*np.pi*F_M*t_sig)))
x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

rng0 = np.random.default_rng(seed=42)
cones0 = []
for i in range(N_CONES):
    d = rng0.standard_normal(N_SINAL) * DITHER_AMP
    x = normalizar(x_mix + d * (1/PHI**i))
    _, cas, _ = agente_eco_rastreado(x, BINS_PHI, N_CICLOS, beta_teto=PHI**3)
    cones0.append(selar_hermetico(cas[-1], BINS_PHI))
serial_phi = concatenar_phi(cones0)
N_SER    = len(serial_phi)
BINS_SER = bandas_para_bins(BANDAS, N_SER)
print(f"  serial_phi: {N_SER/FS:.2f}s  /  bandas φ: {len(BINS_SER)}")

# ── QUATRO TETOS ──────────────────────────────────────────────────────────────
tetos = [
    ('A', PHI**3, f'φ³={PHI**3:.3f}', '#00aaff'),
    ('B', PHI**4, f'φ⁴={PHI**4:.3f}', '#00ffaa'),
    ('C', PHI**6, f'φ⁶={PHI**6:.3f}', '#ffaa00'),
    ('D', None,   '∞ sem teto',        '#ff44aa'),
]

cadeias = {}

for label, teto, descricao, cor in tetos:
    print(f"\n{'═'*64}")
    print(f"  CADEIA [{label}] — teto {descricao}")
    print(f"{'═'*64}")

    # ── Estágio 1: ECO hiperbólico ────────────────────────────────────────────
    print(f"  [1] ECO hiperbólico (teto {descricao})...")
    _, cas_h, betas_h = agente_eco_rastreado(serial_phi, BINS_SER, N_CICLOS, beta_teto=teto)
    eco_hiper = normalizar(cas_h[-1])
    b_ini_h = betas_h[0]; b_fin_h = betas_h[-1]
    pot_h = np.log(max(b_fin_h, 1e-6)) / np.log(PHI)
    print(f"      β_ini={b_ini_h:.4f}  β_fin={b_fin_h:.4f} (φ^{pot_h:.3f})")

    # ── Estágio 2: Nova Serial φ sobre eco_hiper ──────────────────────────────
    print(f"  [2] Nova Serial φ sobre eco_hiper...")
    BINS_H = bandas_para_bins(BANDAS, len(eco_hiper))
    rng2 = np.random.default_rng(seed=42)
    cones_h = []
    for i in range(N_CONES):
        d  = rng2.standard_normal(len(eco_hiper)) * DITHER_AMP
        xc = normalizar(eco_hiper + d * (1/PHI**i))
        _, cas_c, _ = agente_eco_rastreado(xc, BINS_H, N_CICLOS, beta_teto=PHI**3)
        cones_h.append(selar_hermetico(cas_c[-1], BINS_H))
    serial_hiper = concatenar_phi(cones_h)
    print(f"      serial_hiper: {len(serial_hiper)/FS:.2f}s")

    # verificar invariante √5
    BINS_SH = bandas_para_bins(BANDAS, len(serial_hiper))
    nb_sh   = len(BINS_SH)
    beta_chk = np.ones(nb_sh); bm_chk = beta_chk.copy()
    _, cohs_chk = eco_eq(serial_hiper, BINS_SH, beta_chk)
    cr_chk = (cohs_chk - cohs_chk.min()) / (cohs_chk.max() - cohs_chk.min() + 1e-10)
    ba_chk = PHI**(3 * cr_chk)
    b_inv  = ((1-1/PHI) * ba_chk + (1/PHI) * bm_chk).max()
    inv_ok = abs(b_inv - np.sqrt(5)) < 0.01
    print(f"      Invariante √5: β_ciclo1={b_inv:.6f}  {'✓ MANTIDO' if inv_ok else '✗ QUEBRADO'}")

    # ── Estágio 3: ECO final (teto φ³) ───────────────────────────────────────
    print(f"  [3] ECO final (teto φ³) sobre serial_hiper...")
    _, cas_f, betas_f = agente_eco_rastreado(serial_hiper, BINS_SH, N_CICLOS, beta_teto=PHI**3)
    eco_final = normalizar(cas_f[-1])
    b_ini_f = betas_f[0]; b_fin_f = betas_f[-1]
    pot_f = np.log(max(b_fin_f, 1e-6)) / np.log(PHI)
    target = PHI**3 * 0.95
    c95 = next((i+1 for i,b in enumerate(betas_f) if b >= target), N_CICLOS)
    print(f"      β_ini={b_ini_f:.6f}  β_fin={b_fin_f:.6f} (φ^{pot_f:.3f})")
    print(f"      Ciclos para 95% de φ³: {c95}")

    cadeias[label] = {
        'eco_hiper':   eco_hiper,
        'serial_hiper': serial_hiper,
        'eco_final':   eco_final,
        'betas_hiper': betas_h,
        'betas_final': betas_f,
        'b_inv':       b_inv,
        'inv_ok':      inv_ok,
        'c95':         c95,
        'descricao':   descricao,
        'cor':         cor,
        'b_fin_hiper': b_fin_h,
        'b_fin_final': b_fin_f,
    }

# ── VISUALIZAÇÃO ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Cadeia Hiperbólica — ECO(teto) → Serial φ → ECO(φ³)',
             color='white', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.32)

# linhas 0-3: eco_final de cada cadeia
for row, (label, teto, descricao, cor) in enumerate(tetos):
    c = cadeias[label]
    sig = c['eco_final']
    ax_w = fig.add_subplot(gs[row, 0])
    t_ax = np.linspace(0, len(sig)/FS, len(sig))
    ax_w.plot(t_ax[::max(1,len(sig)//4000)], sig[::max(1,len(sig)//4000)],
              color=cor, lw=0.4, alpha=0.85)
    ax_w.set_facecolor('#111111')
    inv_str = "✓√5" if c['inv_ok'] else "✗√5"
    ax_w.set_title(f'[{label}] {descricao} — eco_final  {inv_str}  ({len(sig)/FS:.1f}s)',
                   color='white', fontsize=8.5)
    ax_w.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_w.tick_params(colors='#888888')
    for sp in ax_w.spines.values(): sp.set_color('#333333')

    ax_s = fig.add_subplot(gs[row, 1])
    np2 = min(512, len(sig)//4)
    f_sp, t_sp, Sxx = scipy_spectrogram(sig, FS, nperseg=np2, noverlap=int(np2*0.75))
    ax_s.pcolormesh(t_sp, f_sp[:160], 10*np.log10(Sxx[:160]+1e-10),
                    shading='gouraud', cmap='inferno')
    ax_s.set_facecolor('#111111')
    ax_s.set_title('Espectrograma', color='white', fontsize=8.5)
    ax_s.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax_s.set_ylabel('Hz', color='#888888', fontsize=8)
    ax_s.tick_params(colors='#888888')
    for sp in ax_s.spines.values(): sp.set_color('#333333')

# linha 4: curvas de convergência do ECO final para todos os tetos
ax_c = fig.add_subplot(gs[4, :])
ax_c.set_facecolor('#111111')
for label, teto, descricao, cor in tetos:
    c = cadeias[label]
    ax_c.plot(range(1, N_CICLOS+1), c['betas_final'],
              color=cor, lw=1.5, marker='o', markersize=3,
              label=f'[{label}] {descricao}  β_ini={c["betas_final"][0]:.4f}  c95={c["c95"]}')
ax_c.axhline(y=PHI**3,     color='white',   ls='--', lw=1,   alpha=0.7)
ax_c.axhline(y=np.sqrt(5), color='#aaaaaa', ls=':',  lw=0.8, alpha=0.5)
ax_c.text(N_CICLOS*0.75, PHI**3+0.04,     f'φ³={PHI**3:.4f}',   color='white',   fontsize=7)
ax_c.text(N_CICLOS*0.75, np.sqrt(5)-0.12, f'√5={np.sqrt(5):.4f}', color='#aaaaaa', fontsize=7)
ax_c.set_xlabel('Ciclo (ECO final)', color='#888888', fontsize=8)
ax_c.set_ylabel('β_max', color='#888888', fontsize=8)
ax_c.set_title('Convergência ECO final — todos os tetos hiperbólicos',
               color='white', fontsize=9)
ax_c.tick_params(colors='#888888')
ax_c.legend(facecolor='#222222', labelcolor='white', fontsize=8)
for sp in ax_c.spines.values(): sp.set_color('#333333')

plt.savefig('hiperbolico_cadeia.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("\n  Gráfico: hiperbolico_cadeia.png")

# ── DIAGNÓSTICO ───────────────────────────────────────────────────────────────
print("\n" + "="*64)
print("  DIAGNÓSTICO — CADEIA HIPERBÓLICA COMPLETA")
print("="*64)
print(f"\n  {'[X]':<5} {'Teto':>16}  {'β_eco_hiper':>12}  {'√5?':>8}  {'β_final':>8}  {'c95':>5}")
print(f"  {'-'*58}")
for label, teto, descricao, cor in tetos:
    c = cadeias[label]
    pot = np.log(max(c['b_fin_hiper'],1e-6)) / np.log(PHI)
    inv = "✓" if c['inv_ok'] else "✗"
    print(f"  [{label}]  {descricao:>16}    φ^{pot:.3f}       {inv}     {c['b_fin_final']:.4f}   {c['c95']:>3}")

print(f"\n  √5 = {np.sqrt(5):.6f}  ←  invariante esperado em β_ciclo1 do ECO final")
print(f"  φ³ = {PHI**3:.6f}  ←  atrator esperado em β_final do ECO final")

print(f"\n  β_ciclo1 do ECO final por cadeia:")
for label, teto, descricao, cor in tetos:
    c = cadeias[label]
    print(f"    [{label}] {descricao}: β_ciclo1 = {c['betas_final'][0]:.6f}  "
          f"({'=√5' if c['inv_ok'] else '≠√5'})")

# ── SALVAR WAV ────────────────────────────────────────────────────────────────
def salvar(sig, nome):
    s16 = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    wavfile.write(nome, FS, s16)
    print(f"  {nome}  ({len(sig)/FS:.2f}s)")

print("\n  Salvando WAV...")
for label, teto, descricao, cor in tetos:
    c = cadeias[label]
    salvar(c['eco_hiper'],    f'hiper_{label}_eco_hiper.wav')
    salvar(c['serial_hiper'], f'hiper_{label}_serial.wav')
    salvar(c['eco_final'],    f'hiper_{label}_eco_final.wav')

print("\n" + "="*64)
print("  CADEIA HIPERBÓLICA — CONCLUÍDO")
print("="*64)
print("""
  Arquivos por cadeia (X = A/B/C/D):
    hiper_X_eco_hiper.wav   ← ECO hiperbólico sobre serial_phi
    hiper_X_serial.wav      ← Serial φ sobre eco_hiper
    hiper_X_eco_final.wav   ← ECO final (φ³) sobre serial_hiper
""")

print("  Playback:")
for label, teto, descricao, cor in tetos:
    print(f"\n  [{label}] {descricao}")
    print(f"    ECO hiperbólico:")
    display(Audio(f'hiper_{label}_eco_hiper.wav'))
    print(f"    Serial sobre hiper:")
    display(Audio(f'hiper_{label}_serial.wav'))
    print(f"    ECO final:")
    display(Audio(f'hiper_{label}_eco_final.wav'))
