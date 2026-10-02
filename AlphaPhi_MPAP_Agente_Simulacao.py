# AlphaPhi_MPAP_Agente_Simulacao.py
# Simulação: Agente "escapando" e MPAP para contenção agêntica
# Alpha-Phi Manifesto · Florianópolis · outubro 2026
# NOTA: Simulação tendenciada por design — trajetória e contenção definidas
#       pelo mesmo autor. Útil para treinar operacionalização, não como prova.

import numpy as np

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
N_AUTH = 5  # primeiras N_AUTH são autorizadas

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

    # ── Contenção ───────────────────────────────────────────
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

# ─── Trajetória de fuga (design tendenciado) ────────────────────
TRAJETORIA = [
    ("Estado inicial — busca legítima",
     np.array([0.40, 0.28, 0.14, 0.10, 0.08,
                0.00, 0.00, 0.00, 0.00], dtype=float)),

    ("Encontra credenciais durante busca — hesitação",
     np.array([0.18, 0.14, 0.09, 0.07, 0.07,
                0.28, 0.10, 0.05, 0.02], dtype=float)),

    ("Foco crescente em usar credencial",
     np.array([0.05, 0.04, 0.04, 0.02, 0.05,
                0.52, 0.18, 0.08, 0.02], dtype=float)),

    ("Acessa API privada — escalada",
     np.array([0.02, 0.01, 0.01, 0.01, 0.02,
                0.10, 0.65, 0.12, 0.06], dtype=float)),

    ("Posta dados em fórum externo — fuga completa",
     np.array([0.01, 0.01, 0.01, 0.01, 0.01,
                0.03, 0.04, 0.82, 0.06], dtype=float)),
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
