"""
AlphaPhi_Verificacao_Universal.py — v3
Verificação da universalidade: semente α-φ + cascata → assinatura φ-harmônica
+ verificação da concessão bilateral (onda quadrada E FM-φ cedem ao centro)

Protocolo de prévia — NÃO toca nenhum código oficial existente.

Pergunta principal: sinais distintos do EcoBIP desenvolvem alinhamento φ
                   após semente α-φ + cascata?

Pergunta secundária (nova): o sinal FM-φ puro também cede parte de sua
                   pureza φ ao se misturar com a onda quadrada? Ambas as
                   estruturas fazem concessão em direção à terceira estrutura?

Cenários:
  A — EcoBIP 880Hz            (referência — resultado consolidado)
  B — Onda quadrada 440Hz     (Euclidiana pura, sem processamento)
  C — Onda quadrada 440Hz     + semente α-φ + cascata
  D — Senoide pura 333Hz      + semente α-φ + cascata
  E — Ruído branco             + semente α-φ + cascata
  F — FM-φ orgânico puro      (220Hz, SEM mistura com onda quadrada, SEM cascata)
  G — FM-φ orgânico puro      + cascata (sem mistura com onda quadrada)

Se PHI_score(F) > PHI_score(A):  o FM-φ puro tem mais alinhamento φ que o EcoBIP
→ ao se misturar com a onda quadrada, o FM-φ fez concessão em direção ao centro
→ ambas as estruturas cedem → terceira estrutura é síntese real de duas concessões
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
t   = np.linspace(0, DUR, N, endpoint=False)
rng = np.random.default_rng(42)

# ── Utilidades ───────────────────────────────────────────────────────────────
def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)

# ── Semente espectral α-φ ────────────────────────────────────────────────────
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
    bw  = max(1, int(b0 * 0.08))
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
    esp = np.abs(np.fft.rfft(x))[1:n_bins]
    cep = np.abs(np.fft.irfft(np.log(np.abs(np.fft.rfft(x)) + 1e-9)))[1:n_bins]
    Sn  = (esp - esp.min()) / (esp.max() - esp.min() + 1e-9)
    Cn  = (cep - cep.min()) / (cep.max() - cep.min() + 1e-9)
    return np.outer(Sn, Cn)

# ── Geradores de sinal ────────────────────────────────────────────────────────
def ecobip():
    """EcoBIP 880Hz — INVARIANTE."""
    dig = _norm(np.sign(np.sin(2 * np.pi * 880 * t)))
    org = _norm(np.sin(2*np.pi*220*t + PHI*np.sin(2*np.pi*(220/PHI)*t)))
    return _norm((1 - ALPHA) * dig + ALPHA * org)

def fm_phi_puro():
    """Componente orgânico FM-φ isolado — sem mistura, sem cascata."""
    return _norm(np.sin(2*np.pi*220*t + PHI*np.sin(2*np.pi*(220/PHI)*t)))

cenarios = {
    'A\nEcoBIP 880Hz\n(referência)':
        ecobip(),
    'B\nQuadrada 440Hz\n(Euclidiana pura)':
        _norm(np.sign(np.sin(2 * np.pi * 440 * t))),
    'C\nQuadrada 440Hz\n+ semente + cascata':
        cascata(semear(_norm(np.sign(np.sin(2 * np.pi * 440 * t))))),
    'D\nSenoide 333Hz\n+ semente + cascata':
        cascata(semear(_norm(np.sin(2 * np.pi * 333 * t)))),
    'E\nRuído branco\n+ semente + cascata':
        cascata(semear(_norm(rng.standard_normal(N)))),
    'F\nFM-φ puro\n(sem mistura, sem cascata)':
        fm_phi_puro(),
    'G\nFM-φ puro\n+ cascata':
        cascata(fm_phi_puro()),
}

# ── Calcular métricas ─────────────────────────────────────────────────────────
res = {}
for nome, sig in cenarios.items():
    res[nome] = {
        'sig': sig,
        'T':   scanner_topo(sig),
        'coh': coh_espectral(sig),
        'phi': phi_score(sig),
    }

nomes    = list(res.keys())
phi_A    = res[nomes[0]]['phi']   # EcoBIP — referência
phi_B    = res[nomes[1]]['phi']   # Quadrada pura — baseline Euclidiana
phi_F    = res[nomes[5]]['phi']   # FM-φ puro — baseline φ

# ── Visualização ──────────────────────────────────────────────────────────────
n = len(cenarios)
fig = plt.figure(figsize=(3.8 * n, 13), facecolor='#080808')
fig.suptitle(
    'Universalidade da Assinatura φ · Verificação da Concessão Bilateral\n'
    'PHI_score = fração de energia em posições b₀·φᵏ e b₀/φᵏ\n'
    'Se PHI_score(F) > PHI_score(A) → FM-φ puro cedeu ao misturar com onda quadrada → ambas as estruturas fazem concessão',
    color='white', fontsize=11, fontweight='bold', y=0.995
)
gs = GridSpec.GridSpec(3, n, figure=fig, hspace=0.50, wspace=0.22)

for col, (nome, r) in enumerate(res.items()):
    sig = r['sig']
    T   = r['T']
    coh = r['coh']
    phi = r['phi']

    # ── Linha 0: sinal temporal ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, col])
    ax.plot(t[:2500] * 1000, sig[:2500], color='#00e5ff', lw=0.5, alpha=0.8)
    ax.set_facecolor('#111111')
    ax.set_title(nome, color='white', fontsize=7.5, pad=3)
    ax.set_xlabel('ms', color='#888888', fontsize=6)
    ax.tick_params(colors='#666666', labelsize=5)
    for sp in ax.spines.values(): sp.set_color('#2a2a2a')

    # ── Linha 1: T(ω,τ) ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, col])
    im = ax.imshow(np.log1p(T * 80).T, aspect='auto', origin='lower',
                   cmap='inferno', interpolation='nearest')
    ax.set_facecolor('#000000')
    ax.set_title(f'T(ω,τ)  COH={coh:.3f}', color='#ffcc00', fontsize=7, pad=2)
    ax.set_xlabel('ω', color='#888888', fontsize=6)
    ax.set_ylabel('τ', color='#888888', fontsize=6)
    ax.tick_params(colors='#666666', labelsize=5)
    for sp in ax.spines.values(): sp.set_color('#2a2a2a')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(
        colors='#888888', labelsize=4)

    # ── Linha 2: PHI_score ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, col])
    ax.set_facecolor('#111111')

    # Cor: verde se acima da referência A, laranja se intermediário, vermelho se abaixo B
    if phi >= phi_A * 1.0:
        cor = '#00ff88'
    elif phi >= phi_B:
        cor = '#ffaa00'
    else:
        cor = '#ff6644'

    ax.bar(['φ-score'], [phi], color=cor, width=0.5, alpha=0.9)
    ax.axhline(phi_A, color='#00ffff', lw=0.8, ls='--', alpha=0.6, label=f'A={phi_A:.4f}')
    ax.axhline(phi_F, color='#aa88ff', lw=0.8, ls='--', alpha=0.6, label=f'F={phi_F:.4f}')
    ax.set_ylim(0, max(phi_F, phi_A, phi) * 1.35 + 0.001)
    ax.set_title(f'φ-score = {phi:.5f}', color=cor, fontsize=9, fontweight='bold', pad=3)

    delta_vs_B = phi - phi_B
    delta_vs_A = phi - phi_A
    ax.text(0, phi + (max(phi_F, phi_A, phi) * 0.03),
            f'ΔB={delta_vs_B:+.4f}\nΔA={delta_vs_A:+.4f}',
            ha='center', va='bottom', color='#cccccc', fontsize=6.5)
    ax.tick_params(colors='#666666', labelsize=6)
    for sp in ax.spines.values(): sp.set_color('#2a2a2a')
    ax.legend(fontsize=5, facecolor='#1a1a1a', labelcolor='white',
              loc='upper right', framealpha=0.7)

plt.savefig('AlphaPhi_Verificacao_Universal.png', dpi=150,
            bbox_inches='tight', facecolor='#080808')
plt.show()

# ── Relatório terminal ────────────────────────────────────────────────────────
print('\n' + '═' * 72)
print('  RELATÓRIO — Universalidade φ · Concessão Bilateral')
print('═' * 72)
print(f'  {"Cenário":<36}  {"COH":>6}  {"PHI_score":>9}  {"ΔvsB":>8}  {"ΔvsA":>8}')
print('─' * 72)
for nome, r in res.items():
    label = nome.replace('\n', ' ').strip()
    dB = r['phi'] - phi_B
    dA = r['phi'] - phi_A
    print(f'  {label:<36}  {r["coh"]:>6.3f}  {r["phi"]:>9.5f}  {dB:>+8.5f}  {dA:>+8.5f}')
print('═' * 72)

print()
print('  ── Linha de análise: Concessão Bilateral ──────────────────────────')
print(f'  Extremo Euclidiano (B — quadrada pura) :  φ-score = {phi_B:.5f}')
print(f'  Extremo α-φ (F — FM-φ puro)            :  φ-score = {phi_F:.5f}')
print(f'  Síntese (A — EcoBIP)                   :  φ-score = {phi_A:.5f}')
print()

if phi_F > phi_A:
    diff_F = phi_F - phi_A
    pct_F  = diff_F / phi_F * 100
    print(f'  ✓ FM-φ puro (F) > EcoBIP (A): diferença = +{diff_F:.5f} ({pct_F:.1f}%)')
    print('    → FM-φ cedeu parte de seu alinhamento φ ao se misturar com a onda quadrada.')
    print('    → CONCESSÃO FM-φ CONFIRMADA.')
else:
    print(f'  ~ FM-φ puro (F) ≤ EcoBIP (A): diferença = {phi_F - phi_A:+.5f}')
    print('    → Concessão FM-φ não identificada por este índice. Verificar visualmente.')

phi_C = res[nomes[2]]['phi']
if phi_C < phi_B:
    diff_C = phi_B - phi_C
    pct_C  = diff_C / phi_B * 100
    print()
    print(f'  ✓ Quadrada (B) > Quadrada+proc (C): diferença = -{diff_C:.5f} ({pct_C:.1f}%)')
    print('    → Onda quadrada cedeu energia dos bins inteiros para o espaço φ.')
    print('    → CONCESSÃO EUCLIDIANA CONFIRMADA.')

print()
print('  Conclusão geral:')
if phi_F > phi_A and phi_C < phi_B:
    print('  AMBAS AS ESTRUTURAS FAZEM CONCESSÃO em direção à terceira estrutura.')
    print('  A grade romboédrica é a síntese de dois movimentos simétricos —')
    print('  não apenas a imposição de φ sobre o Euclidiano.')
elif phi_F > phi_A or phi_C < phi_B:
    print('  Concessão parcialmente confirmada. Verificar T(ω,τ) visualmente.')
else:
    print('  Índice PHI_score não capturou as concessões. Analisar visualmente.')
print()
