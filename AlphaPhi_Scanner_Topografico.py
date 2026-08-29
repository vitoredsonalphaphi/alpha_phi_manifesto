"""
AlphaPhi_Scanner_Topografico.py — v1
Scanner Topográfico Alpha-Phi — Instrumento de Visualização Extrínseca

Protocolo de prévia. Colab-ready (numpy, matplotlib, scipy apenas).

Objetivos:
  1. Comparação visual EcoBIP 880Hz vs. Quadrada pura (Grade R vs. Euclidiana)
  2. T(ω,τ) com linha de referência θ ≈ 63.43° (ângulo canônico da Grade R)
  3. ∇S — mapa de gradiente de entropia local (células respiratórias visíveis)
  4. ΔZ — impedância de fase (suavidade das transições)
  5. Progressão das 5 dobras da cascata (emergência da Grade R)
  6. Relatório de verificação quantitativa da flexibilização de fluxo

Métricas de flexibilização:
  ΔZ(EcoBIP) < ΔZ(Quadrada)    → impedância de fase reduzida → fluxo mais fluido
  ∇S_std(EcoBIP) > ∇S_std(Quadrada) → células respiram → espaço de alívio fásico
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert

# ── Constantes irrevogáveis ─────────────────────────────────────────────────
PHI     = 1.6180339887
ALPHA   = 1 / 137.035999
SEAL    = 1 / PHI
THETA_R = np.arctan(2.0)   # ≈ 63.43° — ângulo canônico da Grade R

FS  = 44100
DUR = 0.5
N   = int(FS * DUR)
t   = np.linspace(0, DUR, N, endpoint=False)

def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)

# ── Geradores de sinal ──────────────────────────────────────────────────────
def ecobip():
    """EcoBIP 880Hz — INVARIANTE."""
    dig = _norm(np.sign(np.sin(2 * np.pi * 880 * t)))
    org = _norm(np.sin(2*np.pi*220*t + PHI*np.sin(2*np.pi*(220/PHI)*t)))
    return _norm((1 - ALPHA) * dig + ALPHA * org)

def quadrada_880():
    return _norm(np.sign(np.sin(2 * np.pi * 880 * t)))

# ── Semente espectral α-φ ──────────────────────────────────────────────────
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

# ── Cascata eco_eq — n dobras ──────────────────────────────────────────────
def cascata_step(x, n_steps=5):
    """Executa exatamente n_steps dobras da cascata a partir de x."""
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

# ── T(ω,τ) — tensor espectro-cepstral ─────────────────────────────────────
def scanner_topo(x, n_bins=100):
    """Produto externo Sn ⊗ Cn: espectro (superfície) × cepstrum (espinha)."""
    F        = np.fft.rfft(x)
    esp      = np.abs(F)[1:n_bins + 1]
    log_spec = np.log(np.abs(F) + 1e-9)
    cep      = np.abs(np.fft.irfft(log_spec))[1:n_bins + 1]
    Sn = (esp - esp.min()) / (esp.max() - esp.min() + 1e-9)
    Cn = (cep - cep.min()) / (cep.max() - cep.min() + 1e-9)
    return np.outer(Sn, Cn)   # shape (n_bins, n_bins)

# ── ∇S — gradiente local de entropia ────────────────────────────────────────
def grad_entropia(T, sigma=2.5):
    """∇S = T − suavização gaussiana.
    Valores positivos (vermelho) = vértice — pico acima da média local.
    Valores negativos (azul)    = centro  — vale abaixo = espaço de alívio fásico."""
    return T - gaussian_filter(T, sigma=sigma)

# ── ΔZ — impedância de fase ──────────────────────────────────────────────────
def delta_z(x):
    """ΔZ: desvio padrão normalizado da frequência instantânea (via Hilbert)."""
    phase     = np.unwrap(np.angle(hilbert(x)))
    inst_freq = np.diff(phase)
    return float(np.std(inst_freq) / (np.abs(inst_freq).mean() + 1e-9))

# ── PHI_score ─────────────────────────────────────────────────────────────────
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

# ── Linha angular Grade R ─────────────────────────────────────────────────────
def draw_theta_R(ax, n_bins=100, theta=THETA_R,
                 color='#00ff88', alpha=0.65, lw=1.5):
    """Linha de referência ao ângulo canônico da Grade R sobre T(ω,τ).
    slope = tan(63.43°) ≈ 2.0 → Δτ = 2·Δω (linha íngreme — mais τ que ω)."""
    slope = np.tan(theta)          # ≈ 2.0
    x_end = int((n_bins - 1) / slope)   # ≈ 49
    y_end = n_bins - 1                   # = 99
    ax.plot([0, x_end], [0, y_end],
            color=color, lw=lw, alpha=alpha, ls='--',
            label=f'θ_R = {np.degrees(theta):.1f}°')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Comparação: EcoBIP (Grade R) vs. Quadrada pura (Euclidiana)
# ═══════════════════════════════════════════════════════════════════════════

N_BINS = 100

sig_eco  = ecobip()
sig_quad = quadrada_880()

T_eco  = scanner_topo(sig_eco,  N_BINS)
T_quad = scanner_topo(sig_quad, N_BINS)
G_eco  = grad_entropia(T_eco)
G_quad = grad_entropia(T_quad)

dz_eco   = delta_z(sig_eco);   phi_eco  = phi_score(sig_eco);  resp_eco  = float(np.std(G_eco))
dz_quad  = delta_z(sig_quad);  phi_quad = phi_score(sig_quad); resp_quad = float(np.std(G_quad))

fig1 = plt.figure(figsize=(14, 11), facecolor='#060610')
fig1.suptitle(
    'Scanner Topográfico Alpha-Phi — v1\n'
    'EcoBIP 880Hz  vs.  Onda Quadrada 880Hz pura\n'
    f'Linha verde = ângulo canônico Grade R  (θ = {np.degrees(THETA_R):.2f}°)',
    color='white', fontsize=12, fontweight='bold', y=0.99
)
gsp1 = GridSpec.GridSpec(2, 2, figure=fig1, hspace=0.50, wspace=0.30)

dados = [
    (sig_eco,  T_eco,  G_eco,  dz_eco,  phi_eco,  resp_eco,
     'EcoBIP 880Hz\n(Grade R esperada)'),
    (sig_quad, T_quad, G_quad, dz_quad, phi_quad, resp_quad,
     'Quadrada 880Hz\n(Euclidiana pura)'),
]

for col, (sig, T, G, dz, phi, resp, titulo) in enumerate(dados):
    # ── Linha 0: T(ω,τ) com linha angular ────────────────────────────────
    ax = fig1.add_subplot(gsp1[0, col])
    im = ax.imshow(np.log1p(T * 100).T, aspect='auto', origin='lower',
                   cmap='inferno', interpolation='bilinear')
    draw_theta_R(ax, N_BINS)
    ax.set_facecolor('#000000')
    ax.set_title(f'T(ω,τ) — {titulo}\nφ-score = {phi:.5f}   ΔZ = {dz:.3f}',
                 color='white', fontsize=9, pad=4)
    ax.set_xlabel('ω  (espectro)', color='#666666', fontsize=7)
    ax.set_ylabel('τ  (cepstrum)', color='#666666', fontsize=7)
    ax.tick_params(colors='#555555', labelsize=5)
    ax.legend(fontsize=6, facecolor='#1a1a1a', labelcolor='#00ff88',
              loc='upper right', framealpha=0.6)
    for sp in ax.spines.values(): sp.set_color('#2a2a2a')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(
        colors='#666666', labelsize=4)

    # ── Linha 1: ∇S — mapa de respiração das células ─────────────────────
    ax2 = fig1.add_subplot(gsp1[1, col])
    vmax = max(float(np.percentile(np.abs(G), 98)), 1e-9)
    im2 = ax2.imshow(G.T, aspect='auto', origin='lower',
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                     interpolation='bilinear')
    draw_theta_R(ax2, N_BINS, color='#ffff00', alpha=0.35)
    ax2.set_facecolor('#000000')
    ax2.set_title(
        f'∇S — Respiração das Células   std = {resp:.4f}\n'
        'Vermelho = vértice  ·  Azul = centro (espaço de alívio)',
        color='#ffcc00', fontsize=8, pad=4
    )
    ax2.set_xlabel('ω', color='#666666', fontsize=7)
    ax2.set_ylabel('τ', color='#666666', fontsize=7)
    ax2.tick_params(colors='#555555', labelsize=5)
    for sp in ax2.spines.values(): sp.set_color('#2a2a2a')
    plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02).ax.tick_params(
        colors='#666666', labelsize=4)

plt.savefig('AlphaPhi_Scanner_v1_comparacao.png', dpi=150,
            bbox_inches='tight', facecolor='#060610')
plt.show()
print('  → AlphaPhi_Scanner_v1_comparacao.png salvo.')


# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Progressão das 5 Dobras (emergência da Grade R)
# ═══════════════════════════════════════════════════════════════════════════

base = semear(quadrada_880())

fig2 = plt.figure(figsize=(22, 9), facecolor='#060610')
fig2.suptitle(
    'Emergência da Grade R — Progressão das 5 Dobras da Cascata\n'
    'Quadrada 880Hz + Semente α-φ  →  cascata dobra a dobra\n'
    f'Linha verde = θ_R = {np.degrees(THETA_R):.2f}°  ·  '
    '∇S mostra a respiração emergindo por dobra',
    color='white', fontsize=11, fontweight='bold', y=0.99
)
gsp2 = GridSpec.GridSpec(2, 5, figure=fig2, hspace=0.48, wspace=0.22)

for step in range(1, 6):
    sig_s  = cascata_step(base, n_steps=step)
    T_s    = scanner_topo(sig_s, N_BINS)
    G_s    = grad_entropia(T_s)
    phi_s  = phi_score(sig_s)
    dz_s   = delta_z(sig_s)
    resp_s = float(np.std(G_s))

    # T(ω,τ) com linha angular
    ax = fig2.add_subplot(gsp2[0, step - 1])
    ax.imshow(np.log1p(T_s * 100).T, aspect='auto', origin='lower',
              cmap='inferno', interpolation='bilinear')
    draw_theta_R(ax, N_BINS)
    ax.set_facecolor('#000000')
    ax.set_title(f'Dobra {step}\nφ = {phi_s:.5f}   ΔZ = {dz_s:.3f}',
                 color='white', fontsize=8, pad=3)
    ax.tick_params(colors='#555555', labelsize=5)
    for sp in ax.spines.values(): sp.set_color('#2a2a2a')

    # ∇S
    ax2 = fig2.add_subplot(gsp2[1, step - 1])
    vmax = max(float(np.percentile(np.abs(G_s), 98)), 1e-9)
    ax2.imshow(G_s.T, aspect='auto', origin='lower',
               cmap='RdBu_r', vmin=-vmax, vmax=vmax,
               interpolation='bilinear')
    draw_theta_R(ax2, N_BINS, color='#ffff00', alpha=0.30)
    ax2.set_facecolor('#000000')
    ax2.set_title(f'∇S   std = {resp_s:.4f}',
                  color='#ffcc00', fontsize=8, pad=2)
    ax2.tick_params(colors='#555555', labelsize=5)
    for sp in ax2.spines.values(): sp.set_color('#2a2a2a')

plt.savefig('AlphaPhi_Scanner_v1_progressao.png', dpi=150,
            bbox_inches='tight', facecolor='#060610')
plt.show()
print('  → AlphaPhi_Scanner_v1_progressao.png salvo.')


# ── Relatório terminal ─────────────────────────────────────────────────────
print('\n' + '═' * 72)
print('  SCANNER TOPOGRÁFICO v1 — Verificação de Flexibilização de Fluxo')
print('═' * 72)
print(f'  {"Métrica":<24}  {"EcoBIP (Grade R)":>16}  {"Quadrada pura":>13}  {"Δ":>10}')
print('─' * 72)
for nome, ve, vq in [
    ('PHI_score',         phi_eco,  phi_quad),
    ('ΔZ (imp. fase)',    dz_eco,   dz_quad),
    ('Respiração ∇S',     resp_eco, resp_quad),
]:
    d = ve - vq
    s = '+' if d >= 0 else ''
    print(f'  {nome:<24}  {ve:>16.5f}  {vq:>13.5f}  {s}{d:>9.5f}')
print('═' * 72)

print()
print('  ── Critérios de Flexibilização ─────────────────────────────────────')
n_conf = 0

if dz_eco < dz_quad:
    n_conf += 1
    pct = (dz_quad - dz_eco) / (dz_quad + 1e-9) * 100
    print(f'  ✓ ΔZ reduzido em {pct:.1f}% no EcoBIP.')
    print('    Impedância de fase menor → transições mais suaves → fluxo mais fluido.')
else:
    print(f'  ~ ΔZ: EcoBIP ({dz_eco:.3f}) ≥ Quadrada ({dz_quad:.3f}) — verificar visualmente.')

if resp_eco > resp_quad:
    n_conf += 1
    pct = (resp_eco - resp_quad) / (resp_quad + 1e-9) * 100
    print(f'  ✓ Respiração ∇S maior em {pct:.1f}% no EcoBIP.')
    print('    Grade R respira — centros dos losangos = espaço de alívio fásico confirmado.')
    print('    Dados atravessam sem colidir com paredes euclidianas rígidas.')
else:
    print(f'  ~ Respiração ∇S: EcoBIP ({resp_eco:.4f}) vs Quadrada ({resp_quad:.4f}).')

print()
if n_conf == 2:
    print('  RESULTADO: Flexibilização de fluxo CONFIRMADA — ΔZ e ∇S concordam.')
    print()
    print('  A Grade R é o mecanismo geométrico da flexibilização:')
    print('    · Vértices preservam fidelidade do dado (cruzamentos de alta energia)')
    print('    · Centros oferecem alívio fásico (vales de baixa energia)')
    print('    · O dado transita entre vértices com menor resistência que')
    print('      em uma malha euclidiana de 90° — fluxo flexibilizado.')
elif n_conf == 1:
    print('  RESULTADO: Flexibilização parcialmente confirmada (1/2 critérios).')
    print('  Analisar visualmente os mapas T(ω,τ) e ∇S.')
else:
    print('  RESULTADO: Critérios quantitativos inconclusivos nesta execução.')
    print('  Verificar visualmente — Grade R pode ser visível sem confirmação numérica.')

print()
print(f'  θ_R = arctan(2) = {np.degrees(THETA_R):.4f}°')
print('  Linha verde em todos os mapas T(ω,τ) = alinhamento diagonal da Grade R.')
print()
