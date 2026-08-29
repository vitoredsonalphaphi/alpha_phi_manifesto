# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_SementeLatente.py
Vitor Edson Delavi · Florianópolis · 2026

Protocolo de Validação do Axioma da Precedência (Entradas 205–209):
    A semente α-φ PRECEDE qualquer fractal.
    Fractal sem semente → replica rigidez euclidiana em subescala.
    Fractal com semente → conduto orgânico, atrito → α.

Três cenários comparativos:

    A: EcoBIP 880Hz original — sinal já portador de semente α-φ implícita
       através do termo ALPHA×sig_org na construção (INVARIANTE — intocado).

    B: Onda quadrada pura 880Hz SEM semente — referência euclidiana.
       Mede o patamar basal do fractal sem conduto α-φ.

    C: Onda quadrada pura 880Hz COM semente α-φ explícita no estágio zero.
       Teste genuíno do Axioma: φ primeiro → α segundo → fractal emerge.

Resultado esperado (Axioma da Precedência):
    COH_C >> COH_B  (semente gera ganho real sobre Euclidiano puro)
    COH_A ≈ COH_C   (ambos semedados, métodos diferentes)

Métricas: COH e ENTR em cada uma das 5 dobras, via T(ω,τ).
Saída: painel comparativo (15 mapas T + curvas de convergência).

INVARIANTE: O Cenário A NÃO é modificado.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999
SEAL  = 1 / PHI
FS    = 44100
DUR   = 1.5
N_STEPS = 5
F_BEEP  = 880.0
F_ORG   = 220.0


# ── Semente Latente α-φ ───────────────────────────────────────────────────────

def inicializar_alpha_phi(x: np.ndarray) -> np.ndarray:
    """
    Estágio zero: semente α-φ no domínio espectral.

    Ordem irrevogável (Axioma da Precedência):
        1. φ primeiro  — define as POSIÇÕES φ-harmônicas (curvatura do espaço espectral)
        2. α segundo   — âncora o peso de cada harmônica inserida (ALPHA × SEAL^k)

    Operação: insere energia nas posições b_k = b₀ × φ^k e b₀ / φ^k do espectro.
    Peso: ALPHA × SEAL^k (decrescente com φ⁻ᵏ, irredutível em α).
    Fase: exp(i·φ·k) — curvatura de fase φ-ressonante.

    Ao contrário da versão AM (obsoleta), esta não redistribui energia existente —
    adiciona energia nova nas posições φ-harmônicas. O eco_eq então amplifica
    preferencialmente essas posições, elevando COH_φ real.
    """
    N = len(x)
    F = np.fft.rfft(x)
    mag = np.abs(F)
    # Frequência dominante (evita DC)
    b0 = int(np.argmax(mag[1:]) + 1)
    ref_amp = mag[b0]
    # Insere energia nas posições φ^k e φ^{-k}
    for k in range(1, 9):
        weight = ALPHA * SEAL**k * ref_amp          # α ancora; SEAL^k decai
        phase_k = np.exp(1j * PHI * k)             # φ curva a fase
        for bk in [int(round(b0 * PHI**k)),
                   int(round(b0 / PHI**k))]:
            if 0 < bk < len(F):
                F[bk] += weight * phase_k
    return _norm(np.fft.irfft(F, n=N))


# ── Funções do EcoBIP Scanner Top (preservadas intactas) ─────────────────────

def _norm(s: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def _bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max:
            break
        f = f_next
    return bandas

def _bins(bandas, n_samples):
    out = []
    for fl, fh in bandas:
        lo = max(int(np.floor(fl * n_samples / FS)), 0)
        hi = min(int(np.ceil(fh * n_samples / FS)), n_samples // 2 + 1)
        if hi - lo >= 3:
            out.append((lo, hi, fl, fh))
    return out

def _gerar_base(f_beep=F_BEEP, f_org=F_ORG):
    """EcoBIP 880Hz — portador de semente α-φ implícita via ALPHA×FM-φ."""
    t = np.linspace(0, DUR, int(FS * DUR), endpoint=False)
    sig_dig = _norm(np.sign(np.sin(2 * np.pi * f_beep * t)))
    sig_org = _norm(np.sin(2 * np.pi * f_org * t + PHI * np.sin(2 * np.pi * (f_org/PHI) * t)))
    return _norm((1.0 - ALPHA) * sig_dig + ALPHA * sig_org)

def _gerar_euclidiano(f_beep=F_BEEP):
    """Onda quadrada pura 880Hz — sem qualquer mistura FM-φ. Sinal euclidiano."""
    t = np.linspace(0, DUR, int(FS * DUR), endpoint=False)
    return _norm(np.sign(np.sin(2 * np.pi * f_beep * t)))

def _eco_eq(x, bins_phi, beta, coh_mem):
    N = len(x)
    F = np.fft.rfft(x)
    F_out = F.copy()
    cohs = []
    w_m, w_n = 1.0/PHI, 1.0 - 1.0/PHI
    for i, (lo, hi, *_) in enumerate(bins_phi):
        b_i = float(beta[i]) if i < len(beta) else 1.0
        band = F[lo:hi]
        mag = np.abs(band)
        an = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        coh_ef = w_n * coh + w_m * float(coh_mem[i]) if i < len(coh_mem) else coh
        cohs.append(coh)
        n_idx = np.arange(len(band))
        env = np.clip(1.0 + coh_ef * PHI**b_i * np.cos(2.0 * np.pi * n_idx / PHI), 0.05, None)
        F_out[lo:hi] = (mag * env) * np.exp(1j * np.angle(band))
    return _norm(np.fft.irfft(F_out, n=N)), np.array(cohs)

def _cascata(x0, bins_phi):
    laminas = [x0]
    s = x0.copy()
    nb = len(bins_phi)
    beta = np.ones(nb)
    coh_mem = np.zeros(nb)
    for _ in range(N_STEPS):
        s, cohs = _eco_eq(s, bins_phi, beta, coh_mem)
        coh_mem = cohs
        cr = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta = np.clip(0.382 * PHI**(3*cr) + 0.618 * beta, 0.05, PHI**3)
        laminas.append(s.copy())
    return laminas

def _topografia(sig, n_bins=180, bins_phi=None):
    esp = np.abs(np.fft.rfft(sig))
    cep = np.abs(np.fft.irfft(np.log(esp + 1e-9)))
    S = esp[1:n_bins]; C = cep[1:n_bins]
    Sn = (S - S.min()) / (S.max() - S.min() + 1e-9)
    Cn = (C - C.min()) / (C.max() - C.min() + 1e-9)
    T = np.outer(Sn, Cn)
    # COH espectral (concentração genérica)
    an = np.clip(Sn / (Sn.sum() + 1e-9), 1e-10, 1.0)
    entr = float(-np.sum(an * np.log(an)) / np.log(len(an)))
    coh = float(1.0 - entr)
    # COH_φ v2: perfil de decaimento nas bandas Fibonacci
    # Mede se E_{k+1}/E_k ≈ SEAL em cada banda — perfil áureo de energia
    coh_phi = 0.0
    if bins_phi is not None and len(bins_phi) >= 2:
        energias = [float(np.sum(esp[lo:hi]**2)) for lo, hi, *_ in bins_phi]
        ratios = []
        for i in range(len(energias) - 1):
            if energias[i] > 1e-18:
                ratios.append(energias[i + 1] / energias[i])
        if ratios:
            erros_norm = [((r - SEAL) / SEAL)**2 for r in ratios]
            coh_phi = float(max(0.0, 1.0 - np.mean(erros_norm)))
    return T, coh, entr, coh_phi


# ── Protocolo Comparativo A / B / C ──────────────────────────────────────────

def comparar_cenarios(salvar='SementeLatente_ValidacaoComparativa.png'):
    """
    Executa os três cenários e gera painel comparativo.

    A — EcoBIP 880Hz original (semente α-φ implícita via ALPHA×FM-φ).
    B — Onda quadrada pura 880Hz SEM semente (referência euclidiana basal).
    C — Onda quadrada pura 880Hz COM semente α-φ explícita (teste do Axioma).

    Hipótese (Axioma da Precedência):
        COH_C >> COH_B  →  a semente gera ganho real sobre o Euclidiano puro.
        COH_A ≈ COH_C   →  métodos diferentes de semeação convergem.
    """
    N = int(FS * DUR)
    bins = _bins(_bandas_phi(), N)

    x_base      = _gerar_base()
    x_euclidiano = _gerar_euclidiano()

    print('Cenário A — EcoBIP 880Hz (semente implícita)...')
    lam_A = _cascata(x_base, bins)

    print('Cenário B — Onda quadrada pura (sem semente)...')
    lam_B = _cascata(x_euclidiano, bins)

    print('Cenário C — Onda quadrada pura + semente α-φ explícita...')
    lam_C = _cascata(inicializar_alpha_phi(x_euclidiano), bins)

    def _metricas(laminas):
        cohs, entrs, cohs_phi = [], [], []
        for i in range(N_STEPS):
            _, c, e, cp = _topografia(laminas[i + 1], bins_phi=bins)
            cohs.append(c); entrs.append(e); cohs_phi.append(cp)
        return cohs, entrs, cohs_phi

    cohs_A, entrs_A, cphi_A = _metricas(lam_A)
    cohs_B, entrs_B, cphi_B = _metricas(lam_B)
    cohs_C, entrs_C, cphi_C = _metricas(lam_C)

    # ── Painel visual ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 16), facecolor='#07080E')
    fig.suptitle(
        r'Protocolo de Validação — Axioma da Precedência α-φ'
        '\nA: EcoBIP 880Hz (semente implícita)   '
        'B: Quadrada pura (sem semente)   '
        'C: Quadrada pura + semente α-φ',
        fontsize=11, color='#C8BBAA', fontweight='bold'
    )

    gs = GridSpec(4, N_STEPS, figure=fig, hspace=0.5, wspace=0.18,
                  top=0.91, bottom=0.07, left=0.04, right=0.98)

    CENARIOS = [
        (lam_A, cohs_A, entrs_A, cphi_A, 'A — EcoBIP (semente implícita)', 'inferno'),
        (lam_B, cohs_B, entrs_B, cphi_B, 'B — Euclidiano puro (sem semente)', 'magma'),
        (lam_C, cohs_C, entrs_C, cphi_C, 'C — Euclidiano + semente α-φ (espectral)', 'plasma'),
    ]
    CORES = ['#FF8C42', '#AAAAAA', '#4AFFE8']

    for row, (laminas, cohs, entrs, cohsphi, lbl, cmap) in enumerate(CENARIOS):
        for col in range(N_STEPS):
            ax = fig.add_subplot(gs[row, col])
            T, _, _, _ = _topografia(laminas[col + 1], bins_phi=bins)
            ax.imshow(T + 1e-4, cmap=cmap, origin='lower', aspect='auto',
                      norm=mcolors.LogNorm(vmin=1e-3, vmax=1.0))
            ax.set_title(
                f'{lbl}\nD{col+1}  COH:{cohs[col]:.4f}\n'
                f'COH_φ:{cohsphi[col]:.4f}  E:{entrs[col]:.4f}',
                fontsize=6.5, color='#C8BBAA'
            )
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#1E2840')

    dobras = list(range(1, N_STEPS + 1))

    # Curva COH
    ax_c = fig.add_subplot(gs[3, :3], facecolor='#0D1525')
    ax_c.plot(dobras, cohs_A, 'o--', color=CORES[0], lw=1.8, label='A — EcoBIP (COH)')
    ax_c.plot(dobras, cohs_B, 's:',  color=CORES[1], lw=1.5, label='B — Euclidiano (COH)')
    ax_c.plot(dobras, cohs_C, 'D-',  color=CORES[2], lw=2.2, label='C — Eucl+Semente (COH)')
    ax_c.fill_between(dobras, cohs_B, cohs_C,
                      where=[c > b for b, c in zip(cohs_B, cohs_C)],
                      alpha=0.20, color=CORES[2], label='Ganho do Axioma (C−B)')
    ax_c.set_xlabel('Dobra', color='#C8BBAA')
    ax_c.set_ylabel('COH', color='#C8BBAA')
    ax_c.set_title('Evolução de COH — Axioma da Precedência', color='#C8BBAA', fontsize=9)
    ax_c.legend(fontsize=7.5, facecolor='#0D1525', labelcolor='#C8BBAA', loc='upper left')
    ax_c.tick_params(colors='#2E4055')
    ax_c.grid(alpha=0.2, color='#2E4055')

    # Curva ENTR
    ax_e = fig.add_subplot(gs[3, 3:], facecolor='#0D1525')
    ax_e.plot(dobras, cphi_A, 'o--', color=CORES[0], lw=1.8, label='A — EcoBIP (COH_φ)')
    ax_e.plot(dobras, cphi_B, 's:',  color=CORES[1], lw=1.5, label='B — Euclidiano (COH_φ)')
    ax_e.plot(dobras, cphi_C, 'D-',  color=CORES[2], lw=2.2, label='C — Eucl+Semente (COH_φ)')
    ax_e.fill_between(dobras, cphi_B, cphi_C,
                      where=[c > b for b, c in zip(cphi_B, cphi_C)],
                      alpha=0.25, color=CORES[2], label='Ganho COH_φ (C−B)')
    ax_e.set_xlabel('Dobra', color='#C8BBAA')
    ax_e.set_ylabel('COH_φ  (perfil decaimento SEAL nas bandas)', color='#C8BBAA')
    ax_e.set_title('COH_φ v2 — Decaimento Áureo entre Bandas Fibonacci', color='#C8BBAA', fontsize=9)
    ax_e.legend(fontsize=7.5, facecolor='#0D1525', labelcolor='#C8BBAA', loc='upper left')
    ax_e.tick_params(colors='#2E4055')
    ax_e.grid(alpha=0.2, color='#2E4055')

    plt.savefig(salvar, dpi=150, bbox_inches='tight', facecolor='#07080E')
    print(f'\nSalvo: {salvar}')
    plt.show()

    # ── Relatório textual ──────────────────────────────────────────────────────
    sep = '=' * 84
    print(f'\n{sep}')
    print('RELATÓRIO — Axioma da Precedência α-φ  (v3 — semente espectral + COH_φ)')
    print(f'φ = {PHI:.10f}   α = {ALPHA:.10f}   SEAL = {SEAL:.10f}')
    print(sep)
    print(f'{"D":>3}  {"COH_A":>7}  {"COH_B":>7}  {"COH_C":>7}'
          f'  {"C-B":>7}  {"A-B":>7}'
          f'  {"CPhi_A":>7}  {"CPhi_B":>7}  {"CPhi_C":>7}  {"CφC-B":>7}')
    print('-' * 84)
    for i in range(N_STEPS):
        dcb  = cohs_C[i] - cohs_B[i]
        dab  = cohs_A[i] - cohs_B[i]
        dphi = cphi_C[i] - cphi_B[i]
        print(f'{i+1:>3}  {cohs_A[i]:>7.4f}  {cohs_B[i]:>7.4f}  {cohs_C[i]:>7.4f}'
              f'  {dcb:>+7.4f}  {dab:>+7.4f}'
              f'  {cphi_A[i]:>7.5f}  {cphi_B[i]:>7.5f}  {cphi_C[i]:>7.5f}  {dphi:>+7.5f}')
    print(sep)
    ganho_coh  = cohs_C[-1] - cohs_B[-1]
    ganho_impl = cohs_A[-1] - cohs_B[-1]
    ganho_phi  = cphi_C[-1] - cphi_B[-1]
    print(f'COH  — Ganho C−B (espectral genérico):     {ganho_coh:+.4f}')
    print(f'COH  — Ganho A−B (EcoBIP vs Euclidiano):   {ganho_impl:+.4f}')
    print(f'COH_φ — Ganho C−B (alinhamento φ-real):    {ganho_phi:+.5f}')
    print(f'Critério selagem hermética: COH ≥ {1 - ALPHA*PHI:.4f}')
    print(sep)
    if ganho_phi > 0:
        print('✓ AXIOMA CONFIRMADO via COH_φ: C tem mais energia φ-harmônica que B')
        print(f'  A semente espectral gerou +{ganho_phi:.5f} de alinhamento φ-real.')
    else:
        print('○ COH_φ(C) ≤ COH_φ(B) — investigar bins φ-harmônicos e eco_eq.')
    if ganho_coh > 0:
        print('  COH espectral também confirma (C > B).')
    else:
        print('  COH espectral: C < B — concentração genérica menor (esperado para sinal FM).')
    print(sep)

    return {
        'cohs_A': cohs_A, 'entrs_A': entrs_A, 'cphi_A': cphi_A,
        'cohs_B': cohs_B, 'entrs_B': entrs_B, 'cphi_B': cphi_B,
        'cohs_C': cohs_C, 'entrs_C': entrs_C, 'cphi_C': cphi_C,
    }


if __name__ == '__main__':
    comparar_cenarios()
