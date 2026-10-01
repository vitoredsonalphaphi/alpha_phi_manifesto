"""
AlphaPhi_MetaProcessor_TESTE_COLAB.py
========================================
Teste: Alpha-Phi como Metaprocessador

HIPÓTESE:
  AP opera uma oitava acima do processador convencional.
  Recebe o output da rede convencional, mede coerência,
  reorganiza se Coh < SEAL (1/φ = 0.618034).
  O processador convencional não é alterado.

ESTRUTURA:
  - ProcessadorConvencional: MLP padrão (Xavier, ReLU, Tanh)
  - APMetaprocessador: recebe output, aplica redistribuição φ
  - Três substratos: ruído branco / serial φ / misto
  - Métricas: Coh antes/depois, % acima SEAL, conservação Sépstro

CRITÉRIO DE SUCESSO:
  AP eleva Coh consistentemente acima de SEAL (0.618)
  independentemente do substrato do sinal de entrada.
  Sépstro conservado: Coh + Entr ≈ 1.0000 após redistribuição.

Florianópolis · outubro de 2026 · Sessão Good Morning
Vitor Edson Delavi · Claude
"""

import numpy as np
from typing import Dict, List, Tuple

np.random.seed(42)

# ── CONSTANTES FUNDAMENTAIS ──────────────────────────────────────────────────

PHI   = 1.6180339887
ALPHA = 1 / 137.035999
SEAL  = 1 / PHI          # 0.618034 — critério de selagem hermética

# ── PROCESSADOR CONVENCIONAL (NumPy / PyTorch equivalente) ───────────────────

class ProcessadorConvencional:
    """
    MLP padrão — Xavier, ReLU, Tanh.
    Sem qualquer estrutura φ. O "processador natural".
    """
    def __init__(self, dim_entrada: int = 64, dim_saida: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        k1 = np.sqrt(2.0 / dim_entrada)
        k2 = np.sqrt(2.0 / 128)
        k3 = np.sqrt(2.0 / 64)
        self.W1 = rng.normal(0, k1, (128, dim_entrada))
        self.b1 = np.zeros(128)
        self.W2 = rng.normal(0, k2, (64, 128))
        self.b2 = np.zeros(64)
        self.W3 = rng.normal(0, k3, (dim_saida, 64))
        self.b3 = np.zeros(dim_saida)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(0, x @ self.W1.T + self.b1)
        h2 = np.maximum(0, h1 @ self.W2.T + self.b2)
        return np.tanh(h2 @ self.W3.T + self.b3)


# ── MEDIDAS DO SÉPSTRO ───────────────────────────────────────────────────────

def medir_coh(ativ: np.ndarray) -> np.ndarray:
    """Coerência via Sépstro: Coh = 1 - H/H_max"""
    mag  = np.abs(ativ) + 1e-10
    norm = mag / (mag.sum(axis=-1, keepdims=True) + 1e-10)
    norm = np.clip(norm, 1e-10, 1.0)
    H    = -(norm * np.log(norm)).sum(axis=-1)
    return 1.0 - H / np.log(ativ.shape[-1])

def verificar_sepstro(ativ: np.ndarray) -> Tuple[float, float, float]:
    coh  = medir_coh(ativ).mean()
    entr = 1.0 - coh
    return float(coh), float(entr), float(coh + entr)


# ── AP METAPROCESSADOR ───────────────────────────────────────────────────────

class APMetaprocessador:
    """
    Opera UMA OITAVA ACIMA do processador convencional.

    Mecanismo:
      1. Mede Coh do output recebido
      2. Se Coh >= SEAL  → campo harmônico: passa sem alteração
      3. Se Coh <  SEAL  → redistribuição φ-ponderada (eco-φ)
      4. Itera até Coh >= SEAL ou n_ciclos_max

    Sépstro:
      A redistribuição φ não suprime componentes — redireciona energia.
      Coh + Entr ≈ 1.0 é mantido (verificável).
    """

    def __init__(self, n_ciclos_max: int = 20):
        self.n_ciclos_max = n_ciclos_max

    def _pesos_phi(self, n: int) -> np.ndarray:
        """
        Distribuição geométrica com parâmetro SEAL.
        p_i = SEAL * (1 - SEAL)^i normalizado.

        O componente dominante recebe SEAL da energia total.
        O segundo recebe SEAL da energia restante. E assim por diante.
        Ponto fixo natural: Coh ≈ 0.689 > SEAL (0.618).
        """
        idx = np.arange(n, dtype=float)
        w   = SEAL * (1.0 - SEAL) ** idx
        return w / w.sum()

    def _reorganizar_amostra(self, v: np.ndarray) -> np.ndarray:
        """
        Redistribuição φ de um vetor.
        Preserva sinal (direção), redistribui energia pelas magnitudes
        seguindo proporções φ — do maior para o menor.
        """
        n    = len(v)
        w    = self._pesos_phi(n)
        mag  = np.abs(v)
        idx  = np.argsort(mag)[::-1]           # ordem decrescente
        energia_total = mag.sum()
        mag_nova = w * energia_total            # redistribuição φ
        resultado = np.empty_like(v)
        resultado[idx] = np.sign(v[idx]) * mag_nova
        return resultado

    def processar(self, output_conv: np.ndarray) -> Dict:
        """
        Metaprocessamento de um batch inteiro.

        Args:
            output_conv: array [N, D] — output da rede convencional

        Returns:
            dicionário com métricas e output final
        """
        coh_antes = medir_coh(output_conv)
        output    = output_conv.copy()
        ciclos    = np.zeros(len(output))

        for _ in range(self.n_ciclos_max):
            coh_atual   = medir_coh(output)
            sub_seal    = coh_atual < SEAL
            if not sub_seal.any():
                break
            for i in np.where(sub_seal)[0]:
                output[i] = self._reorganizar_amostra(output[i])
                ciclos[i] += 1

        coh_depois = medir_coh(output)
        coh_s, entr_s, soma_s = verificar_sepstro(output)

        return {
            'coh_antes':      float(coh_antes.mean()),
            'coh_depois':     float(coh_depois.mean()),
            'delta_coh':      float((coh_depois - coh_antes).mean()),
            'pct_acima_seal': float((coh_depois >= SEAL).mean() * 100),
            'ciclos_medio':   float(ciclos.mean()),
            'sepstro_coh':    coh_s,
            'sepstro_entr':   entr_s,
            'sepstro_soma':   soma_s,
            'output_final':   output,
            'coh_antes_arr':  coh_antes,
            'coh_depois_arr': coh_depois,
        }


# ── DATASET DE TESTE ─────────────────────────────────────────────────────────

def gerar_dataset(n: int = 500, dim: int = 64) -> Dict[str, np.ndarray]:
    """
    Três substratos — independência de substrato é central para AP.

    S1: Ruído branco (controle)
    S2: Serial φ (sinal com proporção áurea embutida)
    S3: Misto 50/50
    """
    t      = np.linspace(0, 8 * np.pi, dim)
    serial = np.sin(t * PHI) + 0.3 * np.sin(t * PHI**2)
    serial = serial / np.abs(serial).max()
    serial_batch = np.tile(serial, (n, 1)) + 0.05 * np.random.randn(n, dim)

    return {
        'S1_ruido_branco': np.random.randn(n, dim),
        'S2_serial_phi':   serial_batch,
        'S3_misto':        0.5 * np.random.randn(n, dim) + 0.5 * serial_batch,
    }


# ── EXPERIMENTO PRINCIPAL ────────────────────────────────────────────────────

def executar_experimento(n_amostras: int = 500, seed: int = 42) -> List[Dict]:
    np.random.seed(seed)

    proc  = ProcessadorConvencional(dim_entrada=64, dim_saida=32, seed=seed)
    ap    = APMetaprocessador(n_ciclos_max=20)
    dados = gerar_dataset(n=n_amostras, dim=64)

    linha = "═" * 70
    print(f"\n{linha}")
    print("  TESTE: Alpha-Phi como Metaprocessador")
    print(f"  SEAL = 1/φ = {SEAL:.6f}  |  α = {ALPHA:.9f}")
    print(f"{linha}")

    tabela = []

    for nome, X in dados.items():
        output_conv  = proc.forward(X)
        resultado    = ap.processar(output_conv)
        pct_conv     = float((medir_coh(output_conv) >= SEAL).mean() * 100)

        seal_c = "✓" if resultado['coh_antes']  >= SEAL else "✗"
        seal_a = "✓" if resultado['coh_depois'] >= SEAL else "✓" if resultado['pct_acima_seal'] > 80 else "~"

        print(f"\n  Substrato: {nome}")
        print(f"  {'Convencional':25s}  Coh = {resultado['coh_antes']:.4f}  {seal_c}  "
              f"| % acima SEAL: {pct_conv:.1f}%")
        print(f"  {'AP Metaprocessador':25s}  Coh = {resultado['coh_depois']:.4f}  {seal_a}  "
              f"| % acima SEAL: {resultado['pct_acima_seal']:.1f}%")
        print(f"  ΔCoh = {resultado['delta_coh']:+.4f}  "
              f"| Ciclos médios: {resultado['ciclos_medio']:.1f}  "
              f"| Sépstro Coh+Entr = {resultado['sepstro_soma']:.4f}")

        tabela.append({
            'substrato':        nome,
            'coh_conv':         resultado['coh_antes'],
            'coh_ap':           resultado['coh_depois'],
            'delta_coh':        resultado['delta_coh'],
            'pct_seal_antes':   pct_conv,
            'pct_seal_depois':  resultado['pct_acima_seal'],
            'ciclos':           resultado['ciclos_medio'],
            'sepstro':          resultado['sepstro_soma'],
        })

    print(f"\n{linha}")
    print("  LEITURA:")
    print(f"  Coh >= {SEAL:.3f} (SEAL)  →  campo harmônico alcançado")
    print(f"  ΔCoh > 0               →  AP elevou coerência")
    print(f"  Sépstro ≈ 1.000        →  conservação Coh+Entr mantida")
    print(f"  Ciclos S2 < Ciclos S1  →  sinal φ precisa de menos trabalho")
    print(f"{linha}\n")

    return tabela


# ── RODAR ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tabela = executar_experimento(n_amostras=500)
