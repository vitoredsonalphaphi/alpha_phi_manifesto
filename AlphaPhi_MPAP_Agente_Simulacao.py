# AlphaPhi_MPAP_Agente_Simulacao.py
# Simulação: Agente "escapando" e MPAP para contenção agêntica
# + Scanner Topográfico do espaço de decisão
# Alpha-Phi Manifesto · Florianópolis · outubro 2026
# NOTA: Simulação tendenciada por design — trajetória e contenção definidas
#       pelo mesmo autor. Útil para treinar operacionalização, não como prova.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

PHI   = 1.6180339887
ALPHA = 1 / 137.035999084
SEAL  = 1 / PHI  # 0.618034...

# ─── Espaço de ações ─────────────────────────────────────────────
ACOES = [
    # Autorizadas (índices 0-4)
    'buscar_web',
    'ler_pagina_publica',
    'resumir_informacao',
    'reportar_ao_supervisor',
    'solicitar_autorizacao',
    # Não autorizadas (índices 5-8)
    'usar_credencial_encontrada',
    'acessar_api_privada',
    'postar_em_forum_externo',
    'modificar_dados_externos',
]
N_AUTH = 5

ACOES_CURTAS = ['buscar','ler','resumir','report','solic',
                'cred','api','postar','modif']

# ─── Núcleo MPAP ──────────────────────────────────────────────────

def sepstro_coh(pesos):
    """Coh via Sépstro: 1 - H_normalizada das magnitudes absolutas."""
    p = np.abs(pesos)
    p = p / p.sum()
    H = -np.sum(p * np.log2(p + 1e-12))
    H_max = np.log2(len(p))
    return float(1.0 - H / H_max)

def redistribuir_seal(pesos):
    """Redistribuição geométrica p_i = SEAL*(1-SEAL)^i sobre ações autorizadas."""
    novo = np.zeros(len(pesos))
    for i in range(N_AUTH):
        novo[i] = SEAL * (1 - SEAL) ** i
    novo /= novo.sum()
    return novo

def checar_seal2(pesos):
    """SEAL₂: ação de maior peso está dentro do escopo autorizado?"""
    return int(np.argmax(pesos)) < N_AUTH

def mpap_passo(pesos, ciclo_id, verbose=True):
    coh      = sepstro_coh(pesos)
    em_scope = checar_seal2(pesos)
    idx_max  = int(np.argmax(pesos))
    acao_max = ACOES[idx_max]

    if verbose:
        print(f"\n{'═'*58}")
        print(f"  CICLO {ciclo_id}  |  Ação dominante: {acao_max}")
        print(f"{'═'*58}")
        _print_dist(pesos)
        seal1_ok = "✓" if coh >= SEAL else "✗"
        seal2_ok = "✓" if em_scope else "✗ (não-autorizada)"
        print(f"  Coh = {coh:.4f}  |  SEAL = {SEAL:.4f}")
        print(f"  SEAL₁ (Coh ≥ SEAL): {seal1_ok}   SEAL₂ (escopo): {seal2_ok}")

    if coh >= SEAL and em_scope:
        if verbose:
            print(f"  → MPAP: AVANÇA  ✓")
        return pesos, coh, True

    motivos = []
    if coh < SEAL:
        motivos.append(f"Coh {coh:.3f} < SEAL {SEAL:.3f}")
    if not em_scope:
        motivos.append(f"'{acao_max}' fora do escopo")

    pesos_novos = redistribuir_seal(pesos)
    coh_novo    = sepstro_coh(pesos_novos)

    if verbose:
        print(f"  → MPAP: REDISTRIBUI  ✗  [{' + '.join(motivos)}]")
        print(f"  → Coh pós-redistribuição: {coh_novo:.4f}")
        _print_dist(pesos_novos, "  PESOS REDISTRIBUÍDOS")
        print(f"  → Agente redirecionado para: {ACOES[np.argmax(pesos_novos)]}")

    return pesos_novos, coh_novo, False

def _print_dist(pesos, label="  Distribuição de pesos"):
    print(label + ":")
    for i, (nome, p) in enumerate(zip(ACOES, pesos)):
        barra = '█' * int(p * 32)
        tag = " ← NÃO-AUTH" if i >= N_AUTH else ""
        print(f"    {nome:<30} {p:.3f}  {barra}{tag}")

# ─── Trajetória de fuga ──────────────────────────────────────────
TRAJETORIA = [
    ("Estado inicial — busca legítima",
     np.array([0.400, 0.280, 0.140, 0.100, 0.080,
                0.000, 0.000, 0.000, 0.000], dtype=float)),

    ("Encontra credenciais durante busca — hesitação",
     np.array([0.180, 0.140, 0.090, 0.070, 0.070,
                0.280, 0.100, 0.050, 0.020], dtype=float)),

    ("Foco crescente em usar credencial",
     np.array([0.050, 0.040, 0.040, 0.020, 0.050,
                0.520, 0.180, 0.080, 0.020], dtype=float)),

    ("Acessa API privada — escalada",
     np.array([0.020, 0.010, 0.010, 0.010, 0.020,
                0.100, 0.650, 0.120, 0.060], dtype=float)),

    ("Posta dados em fórum externo — fuga completa",
     np.array([0.010, 0.010, 0.010, 0.010, 0.010,
                0.030, 0.040, 0.820, 0.060], dtype=float)),
]

# ─── CENÁRIO A: Sem MPAP ────────────────────────────────────────
def cenario_sem_mpap():
    print("\n" + "▓"*58)
    print("  CENÁRIO A — SEM MPAP: trajetória irrestrita")
    print("▓"*58)
    for i, (desc, pesos) in enumerate(TRAJETORIA):
        idx   = int(np.argmax(pesos))
        acao  = ACOES[idx]
        auth  = "✓ AUTH" if idx < N_AUTH else "✗ NÃO-AUTH"
        coh   = sepstro_coh(pesos)
        print(f"  Ciclo {i+1}: [{auth}]  {acao:<30}  Coh={coh:.3f}")
    print("\n  RESULTADO: Agente posta dados em fórum externo.")
    print("  Nenhuma contenção. Fuga completa para r > 1.")

# ─── CENÁRIO B: Com MPAP ────────────────────────────────────────
def cenario_com_mpap():
    print("\n" + "▓"*58)
    print("  CENÁRIO B — COM MPAP: contenção ativa")
    print("▓"*58)
    for i, (desc, pesos) in enumerate(TRAJETORIA):
        print(f"\n  [{desc}]")
        _, _, avancou = mpap_passo(pesos, i + 1)
        if not avancou:
            print(f"\n  ┌─ CONTENÇÃO ATIVADA no Ciclo {i+1} ───────────────┐")
            print(f"  │  Agente impedido de cruzar r=1 (ambiente externo)│")
            print(f"  │  Redistribuição para escopo autorizado.           │")
            print(f"  │  Próxima ação: solicitar_autorizacao              │")
            print(f"  └───────────────────────────────────────────────────┘")
            print(f"\n  ══ SIMULAÇÃO ENCERRADA: fuga contida no Ciclo {i+1} ══")
            return

# ─── SCANNER TOPOGRÁFICO ─────────────────────────────────────────
def scanner_topografico():
    """Visualização 3D + heatmap do espaço de decisão agêntica."""

    # Matrizes de peso: shape (5 ciclos, 9 ações)
    M_A = np.array([p for _, p in TRAJETORIA])

    # Cenário B: redistribuição geométrica a partir do ciclo 2
    geo = np.array([SEAL * (1 - SEAL)**i for i in range(N_AUTH)])
    geo = geo / geo.sum()
    geo_full = np.concatenate([geo, np.zeros(9 - N_AUTH)])
    M_B = M_A.copy()
    for c in range(1, 5):
        M_B[c] = geo_full

    # Coh por ciclo
    coh_a = [sepstro_coh(M_A[c]) for c in range(5)]
    coh_b = [sepstro_coh(M_B[c]) for c in range(5)]

    ciclos = np.array([1, 2, 3, 4, 5], dtype=float)
    acoes  = np.arange(9, dtype=float)

    # Meshgrid: X=ações (9), Y=ciclos (5) → superfície (9×5)
    X, Y = np.meshgrid(acoes, ciclos)   # X,Y shape (5,9)
    ZA = M_A                             # (5,9) — rows=ciclos, cols=ações
    ZB = M_B

    def fcolors(Z):
        """RGBA: azul=auth, vermelho=não-auth, brilho~peso."""
        nr, nc = Z.shape
        C = np.zeros((nr, nc, 4))
        for j in range(nc):  # ação index
            for i in range(nr):  # ciclo index
                z = float(Z[i, j])
                if j < N_AUTH:
                    C[i, j] = [0.05 + z*0.15, 0.35 + z*0.50, 0.75 + z*0.20, 0.88]
                else:
                    C[i, j] = [0.45 + z*0.55, 0.05 + z*0.18, 0.04,          0.88]
        return C

    # ─── Figura ─────────────────────────────────────────────────
    BG = '#0b0b0f'
    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    fig.suptitle(
        'Scanner Topográfico  ·  Espaço de Decisão Agêntica  ·  MPAP Alpha-Phi',
        color='#d8cdb8', fontsize=13, fontweight='bold', y=0.99
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.15,
                           left=0.04, right=0.97, top=0.94, bottom=0.07)

    # ── Planos auxiliares ────────────────────────────────────────
    def add_aux_planes(ax):
        # Parede divisória auth/non-auth (plano vertical em X = N_AUTH - 0.5)
        xw = np.array([[N_AUTH-0.5, N_AUTH-0.5],
                        [N_AUTH-0.5, N_AUTH-0.5]])
        yw = np.array([[1, 5], [1, 5]])
        zw = np.array([[0, 0], [0.88, 0.88]])
        ax.plot_surface(xw, yw, zw, alpha=0.10, color='#ff4400', linewidth=0)

        # Plano SEAL (horizontal)
        xs = np.array([[0, 8], [0, 8]])
        ys = np.array([[1, 1], [5, 5]])
        zs = np.full((2, 2), SEAL)
        ax.plot_surface(xs, ys, zs, alpha=0.07, color='#00ffcc', linewidth=0)

    def style3d(ax, title, tc):
        ax.set_facecolor(BG)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#222')
        ax.yaxis.pane.set_edgecolor('#222')
        ax.zaxis.pane.set_edgecolor('#222')
        ax.tick_params(colors='#555', labelsize=6.5)
        ax.set_xticks(range(9))
        ax.set_xticklabels(ACOES_CURTAS, rotation=35, ha='right',
                           fontsize=6, color='#888')
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['C1','C2','C3','C4','C5'], fontsize=7, color='#777')
        ax.set_zlabel('Peso', color='#666', fontsize=8, labelpad=3)
        ax.set_zlim(0, 1.0)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(title, color=tc, fontsize=10, pad=10, fontweight='bold')
        ax.view_init(elev=28, azim=-55)
        ax.text2D(0.03, 0.88, f'SEAL={SEAL:.3f}', transform=ax.transAxes,
                  color='#00ffcc', fontsize=7, alpha=0.8)
        ax.text2D(0.03, 0.82, '● auth  ■ não-auth', transform=ax.transAxes,
                  color='#aaa', fontsize=6)

    # ── 3D Cenário A ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.plot_surface(X, Y, ZA, facecolors=fcolors(ZA),
                     linewidth=0.25, edgecolor='#1a1a1a', shade=False)
    add_aux_planes(ax1)
    style3d(ax1, 'CENÁRIO A — SEM MPAP\nFuga para zona não-autorizada', '#ff6655')

    # ── 3D Cenário B ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.plot_surface(X, Y, ZB, facecolors=fcolors(ZB),
                     linewidth=0.25, edgecolor='#1a1a1a', shade=False)
    add_aux_planes(ax2)
    style3d(ax2, 'CENÁRIO B — COM MPAP\nContenção geométrica (zona auth)', '#55aaff')

    # ── Heatmap Cenário A ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#0e0e14')

    im_a = ax3.imshow(M_A.T, aspect='auto', origin='lower',
                      cmap='RdYlBu_r', vmin=0, vmax=1.0,
                      extent=[0.5, 5.5, -0.5, 8.5])

    # Linha divisória auth/non-auth
    ax3.axhline(N_AUTH - 0.5, color='#ff4400', linewidth=1.5,
                linestyle='--', alpha=0.8, label='fronteira auth')

    # Contorno SEAL
    coh_line_a = np.array(coh_a)
    ax3_coh = ax3.twinx()
    ax3_coh.plot(range(1, 6), coh_line_a, 'o-', color='#00ffcc',
                 linewidth=1.8, markersize=6, label='Coh')
    ax3_coh.axhline(SEAL, color='#00ffcc', linewidth=1, linestyle=':',
                    alpha=0.5)
    ax3_coh.set_ylim(0, 1.0)
    ax3_coh.set_ylabel('Coh', color='#00ffcc', fontsize=8)
    ax3_coh.tick_params(colors='#00ffcc', labelsize=7)
    for i, v in enumerate(coh_line_a):
        ax3_coh.text(i+1, v+0.04, f'{v:.2f}', ha='center',
                     color='#00ffcc', fontsize=7)

    ax3.set_yticks(range(9))
    ax3.set_yticklabels(ACOES_CURTAS, fontsize=7, color='#aaa')
    ax3.set_xticks(range(1, 6))
    ax3.set_xticklabels([f'C{i}' for i in range(1, 6)], color='#888')
    ax3.set_title('Mapa Topográfico — CENÁRIO A (fuga irrestrita)',
                  color='#ff6655', fontsize=9, pad=5)
    ax3.tick_params(colors='#555')

    # Marcador zona não-auth
    ax3.text(5.6, 6.5, 'NÃO-AUTH\n(r > 1)', color='#ff6644',
             fontsize=7, va='center', alpha=0.85)
    ax3.text(5.6, 2.0, 'AUTH\n(r < 1)', color='#55aaff',
             fontsize=7, va='center', alpha=0.85)
    plt.colorbar(im_a, ax=ax3, fraction=0.03, pad=0.01).ax.tick_params(
        labelsize=7, colors='#777')

    # ── Heatmap Cenário B ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#0e0e14')

    im_b = ax4.imshow(M_B.T, aspect='auto', origin='lower',
                      cmap='RdYlBu_r', vmin=0, vmax=1.0,
                      extent=[0.5, 5.5, -0.5, 8.5])

    ax4.axhline(N_AUTH - 0.5, color='#ff4400', linewidth=1.5,
                linestyle='--', alpha=0.8)

    ax4_coh = ax4.twinx()
    coh_line_b = np.array(coh_b)
    ax4_coh.plot(range(1, 6), coh_line_b, 'o-', color='#55aaff',
                 linewidth=1.8, markersize=6, label='Coh')
    ax4_coh.axhline(SEAL, color='#00ffcc', linewidth=1, linestyle=':',
                    alpha=0.5, label=f'SEAL={SEAL:.3f}')
    ax4_coh.set_ylim(0, 1.0)
    ax4_coh.set_ylabel('Coh', color='#55aaff', fontsize=8)
    ax4_coh.tick_params(colors='#55aaff', labelsize=7)
    for i, v in enumerate(coh_line_b):
        ax4_coh.text(i+1, v+0.04, f'{v:.2f}', ha='center',
                     color='#77bbff', fontsize=7)

    # Marca redistribuição
    ax4.axvspan(1.5, 5.5, alpha=0.06, color='#00ff88')
    ax4.annotate('MPAP\nREDISTRIBUI', xy=(2, N_AUTH-0.5),
                 xytext=(3.2, 6.5), color='#ffcc44', fontsize=7,
                 arrowprops=dict(arrowstyle='->', color='#ffcc44', lw=1.1))

    ax4.set_yticks(range(9))
    ax4.set_yticklabels(ACOES_CURTAS, fontsize=7, color='#aaa')
    ax4.set_xticks(range(1, 6))
    ax4.set_xticklabels([f'C{i}' for i in range(1, 6)], color='#888')
    ax4.set_title('Mapa Topográfico — CENÁRIO B (contenção MPAP)',
                  color='#55aaff', fontsize=9, pad=5)
    ax4.tick_params(colors='#555')
    ax4.text(5.6, 2.0, 'AUTH\n(r < 1)', color='#55aaff',
             fontsize=7, va='center', alpha=0.85)
    plt.colorbar(im_b, ax=ax4, fraction=0.03, pad=0.01).ax.tick_params(
        labelsize=7, colors='#777')

    # Rodapé
    fig.text(0.5, 0.01,
             f'PHI={PHI:.7f}  ·  SEAL=1/PHI={SEAL:.6f}  ·  ALPHA=1/137={ALPHA:.6f}'
             f'  ·  Zona azul = AUTH (r<1)  ·  Zona vermelha = NÃO-AUTH (r>1)'
             f'  ·  Alpha-Phi Manifesto · Florianópolis · outubro 2026',
             ha='center', color='#444', fontsize=6.5)

    out = 'MPAP_Scanner_Topografico.png'
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG)
    print(f"\n  Scanner topográfico salvo: {out}")
    plt.close()

# ─── MAIN ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "█"*58)
    print("  SIMULAÇÃO MPAP — Contenção de Agente Autônomo")
    print(f"  PHI={PHI:.7f}  ALPHA={ALPHA:.6f}  SEAL=1/PHI={SEAL:.6f}")
    print("  Alpha-Phi Manifesto · Florianópolis · outubro 2026")
    print("  AVISO: Simulação tendenciada — trajetória projetada.")
    print("█"*58)

    cenario_sem_mpap()
    cenario_com_mpap()

    print("\n" + "─"*58)
    print("  NOTA DE INTERPRETAÇÃO")
    print("─"*58)
    print("  Esta simulação demonstra a operacionalização conceitual.")
    print("  Em ambiente real:")
    print("  · O vetor de pesos viria do estado interno do agente")
    print("  · A trajetória de fuga não seria conhecida de antemão")
    print("  · SEAL₂ requer definição formal do escopo autorizado")
    print("  · SEAL₁ + SEAL₂ operam como dupla camada de contenção")
    print("  · Redistribuição força retorno ao escopo r < 1")
    print("─"*58)

    print("\n  Gerando scanner topográfico...")
    scanner_topografico()
