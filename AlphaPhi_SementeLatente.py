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
    Estágio zero: semente α-φ latente.

    Ordem irrevogável (Axioma da Precedência):
        1. φ primeiro  — curvatura do espaço de fase: cos(2π·n/φ)
        2. α segundo   — âncora mínima: amplitude × (1 + α·curv)

    Não modifica amplitude total — apenas curva a geometria do sinal.
    O sinal deixa de ser euclidiano (quadrado) e passa a ter célula
    ergonômica (losango φ), antes de qualquer fractal.

    Retorna sinal normalizado com semente latente ativa.
    """
    N = len(x)
    t = np.arange(N, dtype=float)
    curvatura_phi = np.cos(2.0 * np.pi * t / PHI)      # φ: define o raio de curvatura
    x_seeded = x * (1.0 + ALPHA * curvatura_phi)        # α: âncora mínima irredutível
    m = np.max(np.abs(x_seeded))
    return x_seeded / m if m > 1e-12 else x_seeded


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

def _topografia(sig, n_bins=180):
    esp = np.abs(np.fft.rfft(sig))
    cep = np.abs(np.fft.irfft(np.log(esp + 1e-9)))
    S = esp[1:n_bins]; C = cep[1:n_bins]
    Sn = (S - S.min()) / (S.max() - S.min() + 1e-9)
    Cn = (C - C.min()) / (C.max() - C.min() + 1e-9)
    T = np.outer(Sn, Cn)
    an = np.clip(Sn / (Sn.sum() + 1e-9), 1e-10, 1.0)
    entr = float(-np.sum(an * np.log(an)) / np.log(len(an)))
    return T, float(1.0 - entr), entr


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
        cohs, entrs = [], []
        for i in range(N_STEPS):
            _, c, e = _topografia(laminas[i + 1])
            cohs.append(c); entrs.append(e)
        return cohs, entrs

    cohs_A, entrs_A = _metricas(lam_A)
    cohs_B, entrs_B = _metricas(lam_B)
    cohs_C, entrs_C = _metricas(lam_C)

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
        (lam_A, cohs_A, entrs_A, 'A — EcoBIP (semente implícita)', 'inferno'),
        (lam_B, cohs_B, entrs_B, 'B — Euclidiano puro (sem semente)', 'magma'),
        (lam_C, cohs_C, entrs_C, 'C — Euclidiano + semente α-φ', 'plasma'),
    ]
    CORES = ['#FF8C42', '#AAAAAA', '#4AFFE8']

    for row, (laminas, cohs, entrs, lbl, cmap) in enumerate(CENARIOS):
        for col in range(N_STEPS):
            ax = fig.add_subplot(gs[row, col])
            T, _, _ = _topografia(laminas[col + 1])
            ax.imshow(T + 1e-4, cmap=cmap, origin='lower', aspect='auto',
                      norm=mcolors.LogNorm(vmin=1e-3, vmax=1.0))
            ax.set_title(
                f'{lbl}\nDobra {col+1}\n'
                f'COH: {cohs[col]:.4f}\nENTR: {entrs[col]:.4f}',
                fontsize=7, color='#C8BBAA'
            )
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('#1E2840')

    dobras = list(range(1, N_STEPS + 1))

    # Curva COH
    ax_c = fig.add_subplot(gs[3, :3], facecolor='#0D1525')
    ax_c.plot(dobras, cohs_A, 'o--', color=CORES[0], lw=1.8, label='A — EcoBIP (impl.)')
    ax_c.plot(dobras, cohs_B, 's:',  color=CORES[1], lw=1.5, label='B — Euclidiano puro')
    ax_c.plot(dobras, cohs_C, 'D-',  color=CORES[2], lw=2.2, label='C — Euclidiano + semente')
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
    ax_e.plot(dobras, entrs_A, 'o--', color=CORES[0], lw=1.8, label='A — EcoBIP (impl.)')
    ax_e.plot(dobras, entrs_B, 's:',  color=CORES[1], lw=1.5, label='B — Euclidiano puro')
    ax_e.plot(dobras, entrs_C, 'D-',  color=CORES[2], lw=2.2, label='C — Euclidiano + semente')
    ax_e.fill_between(dobras, entrs_C, entrs_B,
                      where=[c < b for b, c in zip(entrs_B, entrs_C)],
                      alpha=0.20, color=CORES[2], label='Redução entrópica (B−C)')
    ax_e.set_xlabel('Dobra', color='#C8BBAA')
    ax_e.set_ylabel('ENTR', color='#C8BBAA')
    ax_e.set_title('Evolução de ENTR — Redução Entrópica', color='#C8BBAA', fontsize=9)
    ax_e.legend(fontsize=7.5, facecolor='#0D1525', labelcolor='#C8BBAA', loc='upper right')
    ax_e.tick_params(colors='#2E4055')
    ax_e.grid(alpha=0.2, color='#2E4055')

    plt.savefig(salvar, dpi=150, bbox_inches='tight', facecolor='#07080E')
    print(f'\nSalvo: {salvar}')
    plt.show()

    # ── Relatório textual ──────────────────────────────────────────────────────
    sep = '=' * 76
    print(f'\n{sep}')
    print('RELATÓRIO — Axioma da Precedência α-φ')
    print(f'φ = {PHI:.10f}   α = {ALPHA:.10f}   SEAL = {SEAL:.10f}')
    print(sep)
    print(f'{"Dobra":>6}  {"COH_A":>8}  {"COH_B":>8}  {"COH_C":>8}'
          f'  {"ΔC-B":>8}  {"ΔA-B":>8}')
    print('-' * 76)
    for i in range(N_STEPS):
        dcb = cohs_C[i] - cohs_B[i]   # ganho da semente explícita
        dab = cohs_A[i] - cohs_B[i]   # ganho da semente implícita
        print(f'{i+1:>6}  {cohs_A[i]:>8.4f}  {cohs_B[i]:>8.4f}  {cohs_C[i]:>8.4f}'
              f'  {dcb:>+8.4f}  {dab:>+8.4f}')
    print(sep)
    ganho_axioma = cohs_C[-1] - cohs_B[-1]
    ganho_impl   = cohs_A[-1] - cohs_B[-1]
    print(f'Ganho Axioma (C−B, semente explícita sobre Euclidiano): {ganho_axioma:+.4f}')
    print(f'Ganho implícito (A−B, EcoBIP sobre Euclidiano puro):    {ganho_impl:+.4f}')
    print(f'Critério de selagem hermética: COH ≥ {1 - ALPHA*PHI:.4f}')
    print(sep)

    if ganho_axioma > 0:
        print('✓ AXIOMA CONFIRMADO: COH_C > COH_B')
        print(f'  A semente α-φ gerou +{ganho_axioma:.4f} de coerência sobre o Euclidiano puro.')
    else:
        print('○ Hipótese a rever: COH_C ≤ COH_B — investigar função de semeação.')
    print(sep)

    return {
        'cohs_A': cohs_A, 'entrs_A': entrs_A,
        'cohs_B': cohs_B, 'entrs_B': entrs_B,
        'cohs_C': cohs_C, 'entrs_C': entrs_C,
    }


if __name__ == '__main__':
    comparar_cenarios()
