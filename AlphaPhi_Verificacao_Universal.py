"""
AlphaPhi_Verificacao_Universal.py — v2
Verificação da universalidade: semente α-φ + cascata → assinatura φ-harmônica

Protocolo de prévia — NÃO toca nenhum código oficial existente.

Pergunta: sinais Euclidianos distintos do EcoBIP desenvolvem alinhamento
          φ-harmônico após semente α-φ + cascata?

Cenários:
  A — EcoBIP 880Hz          (referência — resultado já confirmado)
  B — Onda quadrada 440Hz   (Euclidiano puro, sem processamento)
  C — Onda quadrada 440Hz   + semente α-φ + cascata
  D — Senoide pura 333Hz    + semente α-φ + cascata
  E — Ruído branco          + semente α-φ + cascata

Métrica — PHI_score: fração de energia espectral em posições φ-harmônicas
  PHI_score(C) > PHI_score(B) → semente/cascata criou alinhamento φ real
  PHI_score universal ≈ PHI_score(A) → universalidade confirmada
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec

# ── Constantes irrevogáveis ──────────────────────────────────────────────────
PHI   = 1.6180339887
ALPHA = 1 / 137.035999
SEAL  = 1 / PHI

FS  = 44100
DUR = 0.5
N   = int(FS * DUR)

# ── Utilidades ───────────────────────────────────────────────────────────────
def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)

# ── Semente espectral α-φ (INALTERADA — prévia isolada) ─────────────────────
def semear(x):
    F   = np.fft.rfft(x)
    mag = np.abs(F)
    b0  = int(np.argmax(mag[1:]) + 1)
    ref = mag[b0]
    for k in range(1, 9):
        w  = ALPHA * SEAL**k * ref
        ph = np.exp(1j * PHI * k)
        for bk in [int(round(b0 * PHI**k)), int(round(b0 / PHI**k))]:
            if 0 < bk < len(F):
                F[bk] += w * ph
    return _norm(np.fft.irfft(F, n=len(x)))

# ── Cascata eco_eq — 5 dobras ────────────────────────────────────────────────
def cascata(x, n_steps=5):
    n_bins = len(np.fft.rfft(x))
    fib = [1, 1]
    while fib[-1] < n_bins:
        fib.append(fib[-1] + fib[-2])
    fib_bins = [f for f in fib if f < n_bins]

    mem = np.zeros(len(x))
    sig = x.copy()
    for _ in range(n_steps):
        F   = np.fft.rfft(sig)
        env = np.zeros(n_bins)
        for i, fb in enumerate(fib_bins[:-1]):
            lo, hi = fb, fib_bins[i + 1]
            if lo < n_bins:
                env[lo:hi] = np.cos(2 * np.pi * np.arange(hi - lo) / PHI)
        F_mod   = F * (1 + env * SEAL)
        sig_new = _norm(np.fft.irfft(F_mod, n=len(x)))
        mem     = SEAL * mem + (1 - SEAL) * sig_new
        sig     = _norm(sig_new + ALPHA * mem)
    return _norm(sig)

# ── Métricas ─────────────────────────────────────────────────────────────────
def phi_score(x, n_harm=8):
    """Fração de energia espectral em posições φ-harmônicas do pico dominante."""
    F   = np.fft.rfft(x)
    mag = np.abs(F)
    b0  = int(np.argmax(mag[1:]) + 1)
    bw  = max(1, int(b0 * 0.08))          # janela de 8% ao redor de cada bin φ
    total = float(np.sum(mag[1:]**2)) + 1e-12
    phi_e = 0.0
    for k in range(1, n_harm + 1):
        for bk in [int(round(b0 * PHI**k)), int(round(b0 / PHI**k))]:
            if 1 <= bk < len(mag):
                lo = max(1, bk - bw)
                hi = min(len(mag), bk + bw + 1)
                phi_e += float(np.sum(mag[lo:hi]**2))
    return float(phi_e / total)

def coh_espectral(x):
    mag = np.abs(np.fft.rfft(x))[1:]
    an  = np.clip(mag / (mag.sum() + 1e-9), 1e-10, 1.0)
    return float(1.0 - (-np.sum(an * np.log(an))) / np.log(len(an)))

def scanner_topo(x, n_bins=100):
    """T(ω,τ) para visualização."""
    esp = np.abs(np.fft.rfft(x))[1:n_bins]
    cep = np.abs(np.fft.irfft(np.log(np.abs(np.fft.rfft(x)) + 1e-9)))[1:n_bins]
    Sn  = (esp - esp.min()) / (esp.max() - esp.min() + 1e-9)
    Cn  = (cep - cep.min()) / (cep.max() - cep.min() + 1e-9)
    return np.outer(Sn, Cn)

# ── Geradores de sinal ────────────────────────────────────────────────────────
t = np.linspace(0, DUR, N, endpoint=False)
rng = np.random.default_rng(42)

def ecobip():
    dig = _norm(np.sign(np.sin(2 * np.pi * 880 * t)))
    org = _norm(np.sin(2*np.pi*220*t + PHI*np.sin(2*np.pi*(220/PHI)*t)))
    return _norm((1 - ALPHA) * dig + ALPHA * org)

def sq440():        return _norm(np.sign(np.sin(2 * np.pi * 440 * t)))
def sin333():       return _norm(np.sin(2 * np.pi * 333 * t))
def ruido():        return _norm(rng.standard_normal(N))

cenarios = {
    'A\nEcoBIP 880Hz\n(referência)':              ecobip(),
    'B\nQuadrada 440Hz\n(Euclidiana pura)':        sq440(),
    'C\nQuadrada 440Hz\n+ semente + cascata':      cascata(semear(sq440())),
    'D\nSenoide 333Hz\n+ semente + cascata':       cascata(semear(sin333())),
    'E\nRuído branco\n+ semente + cascata':        cascata(semear(ruido())),
}

# ── Calcular métricas ─────────────────────────────────────────────────────────
res = {}
for nome, sig in cenarios.items():
    res[nome] = {
        'sig':  sig,
        'T':    scanner_topo(sig),
        'coh':  coh_espectral(sig),
        'phi':  phi_score(sig),
    }

# ── Visualização ──────────────────────────────────────────────────────────────
n = len(cenarios)
fig = plt.figure(figsize=(4.2 * n, 12), facecolor='#080808')
fig.suptitle(
    'Verificação de Universalidade — Assinatura φ-Harmônica em Diferentes Substratos\n'
    'PHI_score = fração de energia espectral em posições φ-harmônicas (b₀·φᵏ e b₀/φᵏ)',
    color='white', fontsize=12, fontweight='bold', y=0.99
)
gs = GridSpec.GridSpec(3, n, figure=fig, hspace=0.50, wspace=0.25)

phi_ref = res[list(res.keys())[0]]['phi']   # A como referência

for col, (nome, r) in enumerate(res.items()):
    sig = r['sig']
    T   = r['T']
    coh = r['coh']
    phi = r['phi']

    # ── Linha 0: sinal temporal ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, col])
    ax.plot(t[:3000] * 1000, sig[:3000], color='#00e5ff', lw=0.5, alpha=0.8)
    ax.set_facecolor('#111111')
    ax.set_title(nome, color='white', fontsize=8, pad=3)
    ax.set_xlabel('ms', color='#888888', fontsize=6)
    ax.tick_params(colors='#666666', labelsize=5)
    for sp in ax.spines.values(): sp.set_color('#2a2a2a')

    # ── Linha 1: T(ω,τ) ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, col])
    im = ax.imshow(np.log1p(T * 80).T, aspect='auto', origin='lower',
                   cmap='inferno', interpolation='nearest')
    ax.set_facecolor('#000000')
    ax.set_title(f'T(ω,τ)   COH={coh:.3f}', color='#ffcc00', fontsize=7, pad=2)
    ax.set_xlabel('ω (espectral)', color='#888888', fontsize=6)
    ax.set_ylabel('τ (cepstral)', color='#888888', fontsize=6)
    ax.tick_params(colors='#666666', labelsize=5)
    for sp in ax.spines.values(): sp.set_color('#2a2a2a')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(
        colors='#888888', labelsize=4)

    # ── Linha 2: PHI_score ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, col])
    delta = phi - res[list(res.keys())[1]]['phi']  # vs. B (Euclidiana pura)
    cor = '#00ff88' if phi >= phi_ref * 0.85 else ('#ffaa00' if phi >= phi_ref * 0.60 else '#ff4444')
    ax.bar(['PHI_score'], [phi], color=cor, width=0.5, alpha=0.9)
    ax.axhline(phi_ref, color='#ffffff', lw=0.8, ls='--', alpha=0.4, label=f'ref A={phi_ref:.4f}')
    ax.set_ylim(0, max(phi_ref * 1.4, phi * 1.2, 0.01))
    ax.set_facecolor('#111111')
    ax.set_title(f'φ-score = {phi:.4f}\nΔ vs B = {delta:+.4f}',
                 color=cor, fontsize=8, fontweight='bold', pad=3)
    ax.tick_params(colors='#666666', labelsize=6)
    for sp in ax.spines.values(): sp.set_color('#2a2a2a')
    ax.legend(fontsize=5, facecolor='#1a1a1a', labelcolor='white', loc='upper right')

plt.savefig('AlphaPhi_Verificacao_Universal.png', dpi=150,
            bbox_inches='tight', facecolor='#080808')
plt.show()

# ── Relatório ─────────────────────────────────────────────────────────────────
nomes = list(res.keys())
phi_B = res[nomes[1]]['phi']

print('\n' + '═' * 68)
print('  RELATÓRIO — Universalidade da Assinatura φ-Harmônica')
print('═' * 68)
print(f'  {"Cenário":<30}  {"COH":>6}  {"PHI_score":>9}  {"Δ vs B":>8}')
print('─' * 68)
for nome, r in res.items():
    label = nome.replace('\n', ' ').strip()
    delta = r['phi'] - phi_B
    print(f'  {label:<30}  {r["coh"]:>6.3f}  {r["phi"]:>9.5f}  {delta:>+8.5f}')
print('═' * 68)
print()
print(f'  PHI_score referência (A / EcoBIP) = {phi_ref:.5f}')
print()
for nome in nomes[2:]:   # C, D, E
    label = nome.replace('\n', ' ').strip()
    phi_x = res[nome]['phi']
    delta  = phi_x - phi_B
    pct    = phi_x / phi_ref * 100
    print(f'  {label}')
    print(f'    PHI_score = {phi_x:.5f}  ({pct:.1f}% do nível A)')
    if delta > 0.005:
        print(f'    ✓ Semente + cascata aumentou o alinhamento φ ({delta:+.5f} vs B)')
    elif delta > 0:
        print(f'    ~ Aumento pequeno ({delta:+.5f} vs B). Verificar visualmente T(ω,τ).')
    else:
        print(f'    ✗ Não houve aumento ({delta:+.5f} vs B). Analisar substrato.')
    print()
