"""
AlphaPhi_Hiperbolico_COLAB.py
Extensão Hiperbólica do ECO-φ — β além de φ³

Pergunta experimental:
  O teto β_max = φ³ é imposto pelo clip `np.clip(beta, 0.05, PHI**3)`.
  O que acontece se removermos ou elevarmos esse teto?

  Tentativa anterior (ECO BEEP 880 sobre x_mix, alta entropia):
    Sem teto → divergência / colapso espectral

  Hipótese agora (fractabilidade):
    Input em √5-coerência (saída da Serial φ) possui estrutura suficiente
    para sustentar β > φ³ sem colapsar.
    A recursividade fractal Serial→ECO→Serial fornece o scaffolding.

Tetos testados:
  [A] φ³ ≈ 4.236  (baseline — teto atual)
  [B] φ⁴ ≈ 6.854  (primeira extensão)
  [C] φ⁶ ≈ 17.94  (segunda extensão)
  [D] sem teto    (expansão livre)

Input: serial_phi (campo √5-coerente da Frequência Serial φ)

Diagnóstico:
  Para cada teto: β_max por ciclo, E_φ, E_¬φ, qualidade espectral
  Pergunta central: β estabiliza num novo atrator acima de φ³?
  Ou diverge? Ou colapsa de volta a φ³?

Para rodar no Google Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_Hiperbolico_COLAB.py').read())

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
print("  AlphaPhi · Extensão Hiperbólica — β além de φ³")
print("=" * 64)
print(f"  φ  = {PHI:.7f}")
print(f"  φ³ = {PHI**3:.6f}  ← teto atual")
print(f"  φ⁴ = {PHI**4:.6f}  ← teto [B]")
print(f"  φ⁶ = {PHI**6:.6f}  ← teto [C]")
print(f"  ∞  = sem teto      ← teto [D]")
print(f"\n  Input: serial_phi (√5-coerente)")

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

def agente_eco_hiperbolico(sinal, bins_phi, n_ciclos=N_CICLOS, beta_teto=None):
    """agente_eco com teto de β configurável. beta_teto=None = sem teto."""
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

# ── GERAR SERIAL φ (input √5-coerente) ───────────────────────────────────────
print("\n  Gerando Serial φ (input √5-coerente para os testes hiperbólicos)...")
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_sig)))
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_sig + BETA_FM*np.sin(2*np.pi*F_M*t_sig)))
x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

rng = np.random.default_rng(seed=42)
cones_s = []
for i in range(N_CONES):
    d = rng.standard_normal(N_SINAL) * DITHER_AMP
    x = normalizar(x_mix + d * (1/PHI**i))
    bc, cc = agente_eco_hiperbolico(x, BINS_PHI, N_CICLOS, beta_teto=PHI**3)
    cones_s.append(selar_hermetico(cc[-1], BINS_PHI))
serial_phi = concatenar_phi(cones_s)
N_SERIAL   = len(serial_phi)
BINS_SER   = bandas_para_bins(BANDAS, N_SERIAL)
print(f"  serial_phi: {N_SERIAL/FS:.2f}s  /  bandas φ: {len(BINS_SER)}")

# ── QUATRO TETOS ──────────────────────────────────────────────────────────────
tetos = [
    ('A', PHI**3,  f'φ³ = {PHI**3:.3f}',  '#00aaff', 'baseline'),
    ('B', PHI**4,  f'φ⁴ = {PHI**4:.3f}',  '#00ffaa', '1ª extensão'),
    ('C', PHI**6,  f'φ⁶ = {PHI**6:.3f}',  '#ffaa00', '2ª extensão'),
    ('D', None,    '∞  = sem teto',        '#ff44aa', 'livre'),
]

resultados = {}
for label, teto, descricao, cor, nome in tetos:
    print(f"\n  [{label}] {descricao} — {nome}...")
    beta_f, cas_f, betas_c = agente_eco_hiperbolico(
        serial_phi, BINS_SER, N_CICLOS, beta_teto=teto)
    campo = normalizar(cas_f[-1])
    resultados[label] = {
        'campo': campo,
        'betas_ciclos': betas_c,
        'beta_max_final': betas_c[-1],
        'descricao': descricao,
        'cor': cor,
        'nome': nome,
    }
    b_ini = betas_c[0]
    b_fin = betas_c[-1]
    pot_ini = np.log(max(b_ini, 1e-6)) / np.log(PHI)
    pot_fin = np.log(max(b_fin, 1e-6)) / np.log(PHI)
    print(f"      β_inicial = {b_ini:.4f}  (φ^{pot_ini:.3f})")
    print(f"      β_final   = {b_fin:.4f}  (φ^{pot_fin:.3f})")
    target = (teto or b_fin) * 0.95
    c95 = next((i+1 for i,b in enumerate(betas_c) if b >= target), N_CICLOS)
    print(f"      Ciclos para 95% do teto: {c95}")

# ── VISUALIZAÇÃO ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 16), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Extensão Hiperbólica — β além de φ³',
             color='white', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.32)

# linhas 0-1: campos A (baseline) e D (sem teto) — os extremos
for row, label in enumerate(['A', 'D']):
    r = resultados[label]
    sig = r['campo']; cor = r['cor']
    ax_w = fig.add_subplot(gs[row, 0])
    t_ax = np.linspace(0, len(sig)/FS, len(sig))
    ax_w.plot(t_ax[::max(1,len(sig)//4000)], sig[::max(1,len(sig)//4000)],
              color=cor, lw=0.4, alpha=0.85)
    ax_w.set_facecolor('#111111')
    ax_w.set_title(f'[{label}] {r["descricao"]} — campo final', color='white', fontsize=8.5)
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

# linha 2: curvas de convergência de todos os tetos
ax_c = fig.add_subplot(gs[2, :])
ax_c.set_facecolor('#111111')
for label, teto, descricao, cor, nome in tetos:
    r = resultados[label]
    ax_c.plot(range(1, N_CICLOS+1), r['betas_ciclos'],
              color=cor, lw=1.5, label=f'[{label}] {descricao}  (final={r["beta_max_final"]:.3f})')

# linhas de referência φ
for exp, ls in [(3,'--'),(4,':'),(5,'-.'),(6,':')]:
    val = PHI**exp
    if val < max(r['beta_max_final'] for r in resultados.values()) * 1.1:
        ax_c.axhline(y=val, color='white', ls=ls, lw=0.7, alpha=0.4)
        ax_c.text(N_CICLOS*0.82, val+0.05, f'φ^{exp}={val:.2f}',
                  color='white', fontsize=7, alpha=0.6)

ax_c.set_xlabel('Ciclo', color='#888888', fontsize=8)
ax_c.set_ylabel('β_max', color='#888888', fontsize=8)
ax_c.set_title('Convergência β_max — quatro tetos  (input: serial_phi √5-coerente)',
               color='white', fontsize=9)
ax_c.tick_params(colors='#888888')
ax_c.legend(facecolor='#222222', labelcolor='white', fontsize=8)
for sp in ax_c.spines.values(): sp.set_color('#333333')

plt.savefig('hiperbolico.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("\n  Gráfico: hiperbolico.png")

# ── DIAGNÓSTICO ───────────────────────────────────────────────────────────────
print("\n" + "="*64)
print("  DIAGNÓSTICO — EXTENSÃO HIPERBÓLICA")
print("="*64)
print(f"\n  {'Teto':<6} {'β_ini':>8}  {'β_final':>8}  {'φ^?':>6}  {'E_φ':>8}  {'E_¬φ':>8}")
print(f"  {'-'*56}")

for label, teto, descricao, cor, nome in tetos:
    r = resultados[label]
    seg = r['campo'][:N_SINAL]
    X   = np.abs(np.fft.rfft(seg))
    Xn  = X / (X.sum() + 1e-10)
    e_phi  = sum(Xn[b_lo:b_hi].sum() for b_lo, b_hi, _, _ in BINS_PHI)
    e_fora = max(0.0, 1.0 - e_phi)
    bfin   = r['beta_max_final']
    pot    = np.log(max(bfin, 1e-6)) / np.log(PHI)
    bini   = r['betas_ciclos'][0]
    print(f"  [{label}]    {bini:.4f}    {bfin:.4f}   φ^{pot:.2f}   {e_phi:.4f}   {e_fora:.4f}")

print(f"\n  Referências:")
print(f"  √5  = {np.sqrt(5):.6f}  ← β_inicial esperado (invariante)")
print(f"  φ³  = {PHI**3:.6f}  ← atrator baseline")
print(f"  φ⁴  = {PHI**4:.6f}")
print(f"  φ⁵  = {PHI**5:.6f}  ≈ 11Hz (faixa alpha-EEG)")
print(f"  φ⁶  = {PHI**6:.6f}  ≈ 18Hz (Tandy & Lawrence)")

# ── SALVAR WAV ────────────────────────────────────────────────────────────────
def salvar(sig, nome):
    s16 = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    wavfile.write(nome, FS, s16)
    print(f"  {nome}  ({len(sig)/FS:.2f}s)")

print("\n  Salvando WAV...")
for label, teto, descricao, cor, nome in tetos:
    salvar(resultados[label]['campo'], f'hiperbolico_{label}.wav')

print("\n  Playback:")
for label, teto, descricao, cor, nome in tetos:
    print(f"  [{label}] {descricao}")
    display(Audio(f'hiperbolico_{label}.wav'))
