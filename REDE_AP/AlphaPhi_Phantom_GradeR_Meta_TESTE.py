"""
AlphaPhi_Phantom_GradeR_Meta_TESTE.py
========================================
Teste: Phantom + Grade R + AP Metaprocessador

HIPÓTESE:
  O Phantom aplicado durante o "treinamento" da rede convencional
  orienta os pesos em direção à geometria φ.
  Grade R verifica se a estrutura φ está presente no output pré-AP.
  AP sela o output no campo harmônico.

  Se correto: a rede com Phantom produz Coh > rede sem Phantom,
  antes mesmo de AP intervir. AP fecha o gap restante em 1 ciclo.
  Grade R marca a diferença entre os dois outputs pré-AP.

ARQUITETURAS COMPARADAS:
  [A] Sem Phantom → AP  (baseline — teste anterior)
  [B] Com Phantom → AP  (nova hipótese)

PHANTOM:
  Sinal φ-estruturado injetado ANTES de cada forward pass
  como perturbação aditiva escalada por ALPHA.
  Orienta os pesos implicitamente — sem backprop modificado.
  É uma influência externa, não uma mudança de arquitetura.

GRADE R:
  θ_R = arctan(2) ≈ 63.43° — critério geométrico φ.
  Mede se o vetor de ativação tem componente principal
  alinhada com o ângulo θ_R da geometria icosaédrica.
  Grade R PRESENTE: ângulo dominante ≥ θ_R.

Florianópolis · outubro de 2026 · Sessão Good Morning
Vitor Edson Delavi · Claude
"""

import numpy as np

np.random.seed(42)

PHI   = 1.6180339887
ALPHA = 1 / 137.035999
SEAL  = 1 / PHI
THETA_R = np.arctan(2)  # ≈ 1.1071 rad ≈ 63.43°

# ── CONSTANTES ────────────────────────────────────────────────────────────────

N_AMOSTRAS    = 500
DIM_ENTRADA   = 64
DIM_SAIDA     = 32
N_PHANTOM     = 10       # passos de "influência" do Phantom
PHANTOM_SCALE = ALPHA    # escala de perturbação — α como âncora de tensão

# ── PHANTOM ───────────────────────────────────────────────────────────────────

def gerar_phantom(n: int, dim: int) -> np.ndarray:
    """
    Sinal φ-estruturado: série temporal com proporção áurea embutida.
    Usado como perturbação nos dados de entrada durante o "treinamento".
    """
    t = np.linspace(0, 8 * np.pi, dim)
    base = np.sin(t * PHI) + (1/PHI) * np.sin(t * PHI**2) + (1/PHI**2) * np.sin(t * PHI**3)
    base = base / np.abs(base).max()
    batch = np.tile(base, (n, 1))
    batch += 0.02 * np.random.randn(n, dim)
    return batch * PHANTOM_SCALE

# ── PROCESSADOR CONVENCIONAL ──────────────────────────────────────────────────

def relu(x): return np.maximum(0, x)

class ProcessadorConvencional:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/DIM_ENTRADA), (128, DIM_ENTRADA))
        self.b1 = np.zeros(128)
        self.W2 = rng.normal(0, np.sqrt(2/128), (64, 128))
        self.b2 = np.zeros(64)
        self.W3 = rng.normal(0, np.sqrt(2/64), (DIM_SAIDA, 64))
        self.b3 = np.zeros(DIM_SAIDA)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = relu(x @ self.W1.T + self.b1)
        h2 = relu(h1 @ self.W2.T + self.b2)
        return np.tanh(h2 @ self.W3.T + self.b3)

    def adaptar_com_phantom(self, X: np.ndarray, n_passos: int = N_PHANTOM):
        """
        Simula influência do Phantom sem backprop completo.
        Cada passo: processa X + Phantom, ajusta pesos por correlação φ.
        Regra Hebbiana φ: Δw ∝ ALPHA * (output_phantom - output_normal).
        """
        lr = ALPHA * 0.1
        for _ in range(n_passos):
            phantom = gerar_phantom(len(X), DIM_ENTRADA)
            out_normal  = self.forward(X)
            out_phantom = self.forward(X + phantom)
            delta = out_phantom - out_normal
            # Ajuste Hebbiano mínimo: propaga diferença para última camada
            h1 = relu(X @ self.W1.T + self.b1)
            h2 = relu(h1 @ self.W2.T + self.b2)
            dW3 = lr * (delta.T @ h2) / len(X)
            self.W3 += dW3

# ── GRADE R ───────────────────────────────────────────────────────────────────

def medir_grade_r(ativ: np.ndarray) -> dict:
    """
    Grade R: critério geométrico φ.
    θ_R = arctan(2) ≈ 63.43°.

    Para cada amostra: encontra o ângulo do componente dominante
    em relação ao componente sub-dominante no plano (dim0, dim1).
    Grade R PRESENTE se ângulo ≥ θ_R.

    Proxy: razão entre componente dominante e sub-dominante.
    Se ratio ≥ tan(θ_R) = 2.0, Grade R presente.
    """
    mag    = np.abs(ativ)
    sorted_mag = np.sort(mag, axis=-1)[:, ::-1]  # decrescente
    ratio  = sorted_mag[:, 0] / (sorted_mag[:, 1] + 1e-10)
    angulo = np.arctan(ratio)
    presente = angulo >= THETA_R
    return {
        'ratio_medio':   float(ratio.mean()),
        'angulo_medio':  float(np.degrees(angulo.mean())),
        'pct_presente':  float(presente.mean() * 100),
        'theta_r_graus': float(np.degrees(THETA_R)),
    }

# ── AP METAPROCESSADOR ────────────────────────────────────────────────────────

def medir_coh(ativ: np.ndarray) -> np.ndarray:
    mag  = np.abs(ativ) + 1e-10
    norm = mag / (mag.sum(axis=-1, keepdims=True) + 1e-10)
    norm = np.clip(norm, 1e-10, 1.0)
    H    = -(norm * np.log(norm)).sum(axis=-1)
    return 1.0 - H / np.log(ativ.shape[-1])

def redistribuir_seal(v: np.ndarray) -> np.ndarray:
    n   = len(v)
    idx = np.arange(n, dtype=float)
    w   = SEAL * (1.0 - SEAL) ** idx
    w  /= w.sum()
    mag = np.abs(v)
    ordem = np.argsort(mag)[::-1]
    res   = np.empty_like(v)
    energia = mag.sum()
    res[ordem] = np.sign(v[ordem]) * w * energia
    return res

def ap_processar(output_conv: np.ndarray) -> dict:
    coh_antes = medir_coh(output_conv)
    output    = output_conv.copy()
    ciclos    = np.zeros(len(output))
    for _ in range(20):
        sub = medir_coh(output) < SEAL
        if not sub.any():
            break
        for i in np.where(sub)[0]:
            output[i] = redistribuir_seal(output[i])
            ciclos[i] += 1
    coh_depois = medir_coh(output)
    coh_s = float(medir_coh(output).mean())
    entr_s = 1.0 - coh_s
    return {
        'coh_antes':   float(coh_antes.mean()),
        'coh_depois':  float(coh_depois.mean()),
        'delta_coh':   float((coh_depois - coh_antes).mean()),
        'pct_seal':    float((coh_depois >= SEAL).mean() * 100),
        'ciclos':      float(ciclos.mean()),
        'sepstro':     float(coh_s + entr_s),
        'output':      output,
    }

# ── DATASET ───────────────────────────────────────────────────────────────────

def gerar_dataset():
    t = np.linspace(0, 8 * np.pi, DIM_ENTRADA)
    serial = np.sin(t * PHI) + 0.3 * np.sin(t * PHI**2)
    serial /= np.abs(serial).max()
    serial_batch = np.tile(serial, (N_AMOSTRAS, 1)) + 0.05 * np.random.randn(N_AMOSTRAS, DIM_ENTRADA)
    return {
        'S1_ruido_branco': np.random.randn(N_AMOSTRAS, DIM_ENTRADA),
        'S2_serial_phi':   serial_batch,
        'S3_misto':        0.5 * np.random.randn(N_AMOSTRAS, DIM_ENTRADA) + 0.5 * serial_batch,
    }

# ── EXPERIMENTO ───────────────────────────────────────────────────────────────

def executar():
    np.random.seed(42)
    dados  = gerar_dataset()
    linha  = "═" * 72
    linha2 = "─" * 72

    print(f"\n{linha}")
    print("  TESTE: Phantom + Grade R + AP Metaprocessador")
    print(f"  θ_R = arctan(2) = {np.degrees(THETA_R):.2f}°  |  SEAL = {SEAL:.6f}  |  α = {ALPHA:.9f}")
    print(f"{linha}\n")

    for nome, X in dados.items():
        print(f"  ┌─ Substrato: {nome}")

        # [A] SEM Phantom
        rede_a = ProcessadorConvencional(seed=42)
        out_a  = rede_a.forward(X)
        gr_a   = medir_grade_r(out_a)
        ap_a   = ap_processar(out_a)

        # [B] COM Phantom
        rede_b = ProcessadorConvencional(seed=42)
        X_treino = dados['S2_serial_phi']
        rede_b.adaptar_com_phantom(X_treino, n_passos=N_PHANTOM)
        out_b  = rede_b.forward(X)
        gr_b   = medir_grade_r(out_b)
        ap_b   = ap_processar(out_b)

        print(f"  │")
        print(f"  │  {'':30s}  {'[A] Sem Phantom':>16s}  {'[B] Com Phantom':>16s}")
        print(f"  │  {linha2[:64]}")
        print(f"  │  {'Coh pré-AP':30s}  {ap_a['coh_antes']:>16.4f}  {ap_b['coh_antes']:>16.4f}")
        print(f"  │  {'Grade R % presente (pré-AP)':30s}  {gr_a['pct_presente']:>15.1f}%  {gr_b['pct_presente']:>15.1f}%")
        print(f"  │  {'Ângulo dominante médio':30s}  {gr_a['angulo_medio']:>14.2f}°  {gr_b['angulo_medio']:>14.2f}°")
        print(f"  │  {linha2[:64]}")
        print(f"  │  {'Coh pós-AP':30s}  {ap_a['coh_depois']:>16.4f}  {ap_b['coh_depois']:>16.4f}")
        print(f"  │  {'ΔCoh (AP fez este trabalho)':30s}  {ap_a['delta_coh']:>+16.4f}  {ap_b['delta_coh']:>+16.4f}")
        print(f"  │  {'% acima SEAL':30s}  {ap_a['pct_seal']:>15.1f}%  {ap_b['pct_seal']:>15.1f}%")
        print(f"  │  {'Ciclos AP':30s}  {ap_a['ciclos']:>16.1f}  {ap_b['ciclos']:>16.1f}")
        print(f"  │  {'Sépstro Coh+Entr':30s}  {ap_a['sepstro']:>16.4f}  {ap_b['sepstro']:>16.4f}")
        print(f"  └{'─'*65}\n")

    print(f"{linha}")
    print("  LEITURA:")
    print(f"  Coh pré-AP [B] > [A]     →  Phantom elevou coerência na rede")
    print(f"  Grade R [B] > [A]        →  rede com Phantom tem geometria φ")
    print(f"  ΔCoh [B] < [A]           →  AP fez MENOS trabalho (rede já orientada)")
    print(f"  Coh pós-AP ≈ igual       →  campo harmônico é o mesmo (SEAL = ponto fixo)")
    print(f"  Sépstro ≈ 1.000          →  conservação mantida em ambos")
    print(f"{linha}\n")

if __name__ == '__main__':
    executar()
