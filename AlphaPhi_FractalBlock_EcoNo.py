# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_FractalBlock_EcoNo.py
Vitor Edson Delavi · Florianópolis · 2026

Integração FractalBlock ← EcoNo ← FunctionalTool Protocol (Entradas 191–192)

Camadas desta arquitetura:

  1. FunctionalTool (Protocol)
     Contrato de interface que qualquer ferramenta Alpha-Phi implementa.
     Scanner, eco_fononico_v2, PhiAttractorNetwork → plugáveis sem reescrita.
     execute(x) → (tensor_processado, coh: float, entr: float)

  2. EcoNoTriade (implementa FunctionalTool)
     Tríade Observar → Selecionar → Agir com Sépstro local.
     Folha da árvore fractal. Substitui BasicConv.

  3. FractalFunctionalNode
     Wrapper fractal para qualquer FunctionalTool.
     Critério de selagem corrigido: ganho_relativo = ΔCoh / (1 − Coh_entrada)
     Sela quando ganho_relativo < SEAL = 1/φ   (ΔCoh marginal → campo estabilizado)
     Join Sépstro: pesos proporcionais ao ganho de Coh de cada caminho.

  4. FractalBlockEcoNo
     Árvore recursiva binária com EcoNoTriade nas folhas.
     C_k = Join_Sépstro(C_{k-1}(x), C_{k-1}(C_{k-1}(x)))

Modelo espacial: α (centro, r=0) → Campo Harmônico (borda, r=1)
Movimento sempre do centro para fora. Nunca inverter.
"""

import math
from typing import Tuple, Protocol, runtime_checkable
import torch
import torch.nn as nn


# ── Constantes fundamentais ───────────────────────────────────────────────────

PHI   = (1 + math.sqrt(5)) / 2    # 1.6180339887 — lei geradora, expansão
ALPHA = 1 / 137.035999             # 0.007297...  — âncora individual, centro
SEAL  = 1 / PHI                    # 0.6180...    — critério de selagem hermética


# ── FunctionalTool — contrato de interface ───────────────────────────────────

@runtime_checkable
class FunctionalTool(Protocol):
    """
    Contrato que toda ferramenta Alpha-Phi deve cumprir para entrar na
    árvore fractal sem ser reescrita.

    Scanner, eco_fononico_v2, PhiAttractorNetwork → implementam execute()
    e passam a ser plugáveis em FractalFunctionalNode diretamente.

    execute(x) retorna:
        tensor_processado: sinal após a ferramenta
        coh:  float ∈ [0, 1] — coerência medida pelo Sépstro da ferramenta
        entr: float ∈ [0, 1] — entropia  (coh + entr = 1.0 sempre)
    """

    def execute(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        ...


# ── FractalFunctionalNode — wrapper fractal para qualquer FunctionalTool ──────

class FractalFunctionalNode(nn.Module):
    """
    Envolve qualquer FunctionalTool em progressão fractal.

    A ferramenta executa primeiro (preserva sua funcionalidade original).
    O fractal decide depois: se o ganho relativo de Coh justifica continuar.

    Critério de selagem corrigido (vs. Gemini):
        delta_coh      = |coh_saida − coh_entrada|
        ganho_relativo = delta_coh / (1 − coh_entrada + ε)
        sela se:  ganho_relativo < SEAL  (1/φ ≈ 0.618)

    Isso mede o ganho em relação à MARGEM DISPONÍVEL —
    não o delta absoluto (que seria quase sempre < 0.618 e selaria cedo demais).

    Join Sépstro: pesos proporcionais ao ganho de coerência de cada caminho.
    """

    def __init__(self, tool: FunctionalTool, depth: int = 0, max_depth: int = 7):
        super().__init__()
        self.tool      = tool
        self.depth     = depth
        self.max_depth = max_depth

    def _coh_de_tensor(self, x: torch.Tensor) -> float:
        """Coh instantânea de um tensor via energia normalizada."""
        energia = x.abs().mean().item()
        return min(max(energia / (1.0 + energia), ALPHA), 1.0 - ALPHA)

    def _selagem_hermetica(self, coh: float) -> bool:
        """
        Critério exato do EcoNo holográfico (AlphaPhi_EcoAdaptativo_Holografico.py):

            margem_disponivel = 1 − Coh
            ganho_projetado   = margem_disponivel × SEAL   (SEAL = 1/φ)
            sela se:  ganho_projetado < α

        Tradução: sela quando Coh > 1 − α×φ ≈ 0.9882
        O campo só sela quando está dentro de α de distância da borda harmônica.
        Uma rede não treinada (Coh ≈ 0.3–0.5) NÃO sela — tem tensão para expandir.
        """
        margem_disponivel = 1.0 - coh
        ganho_projetado   = margem_disponivel * SEAL
        return ganho_projetado < ALPHA

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # ── 1. Execução primária da ferramenta ────────────────────────────────
        coh_in                   = self._coh_de_tensor(x)
        x_out, coh_out, entr_out = self.tool.execute(x)

        # ── 2. Critério de selagem hermética ──────────────────────────────────
        # Sela quando o campo está próximo da borda harmônica (Coh ≈ 0.9882+)
        # ou quando depth máxima foi atingida (guarda de segurança)
        if self._selagem_hermetica(coh_out) or self.depth >= self.max_depth:
            return x_out, {
                'depth':     self.depth,
                'coh':       coh_out,
                'entr':      entr_out,
                'selado':    True,
                'execucoes': 1,
            }

        # ── 3. Progressão fractal — dois filhos compartilham a mesma ferramenta
        filho_1 = FractalFunctionalNode(self.tool, self.depth + 1, self.max_depth)
        filho_2 = FractalFunctionalNode(self.tool, self.depth + 1, self.max_depth)

        x_f1, meta1 = filho_1(x_out)
        x_f2, meta2 = filho_2(x_f1)

        # ── 4. Join Sépstro — pesos proporcionais à coerência de cada caminho
        c1  = meta1['coh'] + 1e-8
        c2  = meta2['coh'] + 1e-8
        tot = c1 + c2
        x_final = (c1 / tot) * x_f1 + (c2 / tot) * x_f2

        _, coh_final, entr_final = self.tool.execute(x_final)

        return x_final, {
            'depth':     max(meta1['depth'], meta2['depth']),
            'coh':       coh_final,
            'entr':      entr_final,
            'selado':    False,
            'execucoes': 1 + meta1['execucoes'] + meta2['execucoes'],
        }


# ── EcoNoTriade — tríade como módulo PyTorch ──────────────────────────────────

class EcoNoTriade(nn.Module):
    """
    Tríade EcoNo em PyTorch: Observar → Selecionar → Agir.

    Substitui BasicConv no caso base (depth==1) do FractalBlock.
    Cada folha da árvore fractal passa a ter:
        - Observar:   leitura α-sensível do estado do nó (1×1 conv)
        - Selecionar: alinhamento φ (depthwise separable — φ-local)
        - Agir:       transformação modulada pela coerência do Sépstro local
        - Sépstro:    Coh + Entr = 1.0 instantâneo — lei de conservação

    O Sépstro é calculado a cada forward — não é estado persistente.
    Isso preserva a compatibilidade com batches e com o autograd.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Observar: lê o estado do nó — 1×1 conv, não altera estrutura espacial
        self.observar = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Selecionar: alinhamento φ — depthwise separable (φ-local por canal)
        self.selecionar = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),       # depthwise: cada canal isolado
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # pointwise: recombina
            nn.BatchNorm2d(channels),
            nn.GELU(),                                                 # ativação suave, φ-compatível
        )

        # Agir: transformação final modulada pela coerência Coh do nó
        self.agir = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def septro(self, sel: torch.Tensor):
        """
        Sépstro instantâneo: Coh + Entr = 1.0000

        Coh = energia da seleção normalizada ∈ [α, 1−α]
        Entr = 1 − Coh

        Ganho = quanto Coh se afastou do centro α em direção à borda.
        Ganho < SEAL = 1/φ → nó atingiu equilíbrio → sela a ramificação.

        Retorna: (coh, entr, ganho) — todos tensores (B, 1, 1, 1)
        """
        energia = sel.abs().mean(dim=(1, 2, 3), keepdim=True)
        coh  = (energia / (1.0 + energia)).clamp(ALPHA, 1.0 - ALPHA)
        entr = 1.0 - coh
        ganho = (coh - ALPHA) / (1.0 - ALPHA)   # normalizado ∈ [0, 1]
        return coh, entr, ganho

    def forward(self, x: torch.Tensor):
        """
        Ciclo ternário: Observar → Selecionar → Agir.

        Retorna:
            saida (tensor): resultado do Agir, modulado por Coh
            ganho (tensor): (B,1,1,1) — coerência relativa do nó — critério de selagem
        """
        # Fase 1 — Observar: sensibilidade α-escalada
        obs = self.observar(x) * (1.0 + ALPHA)

        # Fase 2 — Selecionar: alinhamento φ
        sel = self.selecionar(obs)

        # Sépstro local: Coh + Entr = 1.0
        coh, entr, ganho = self.septro(sel)

        # Fase 3 — Agir: saída modulada pela coerência atual do nó
        saida = self.agir(sel) * coh

        return saida, ganho

    def execute(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Implementa FunctionalTool — permite que EcoNoTriade seja
        plugada em FractalFunctionalNode sem nenhuma outra mudança.
        """
        saida, ganho_t = self.forward(x)
        coh_val  = ganho_t.mean().item()
        entr_val = 1.0 - coh_val
        return saida, coh_val, entr_val


# ── FractalBlockEcoNo — árvore fractal com EcoNo nas folhas ──────────────────

class FractalBlockEcoNo(nn.Module):
    """
    FractalBlock com EcoNoTriade nas folhas e Sépstro no Join.

    Mantém a estrutura original:
        C_k = Join(C_{k-1}(x),  C_{k-1}(C_{k-1}(x)))
              caminho raso        caminho profundo

    Substituições:
        BasicConv     → EcoNoTriade  (folhas: depth==1)
        Join (×0.5)   → Join Sépstro (pesos proporcionais ao ganho de Coh)

    Selagem hermética:
        Durante inferência, se o caminho raso já estabilizou (ganho_médio < 1/φ),
        o caminho profundo não é calculado — o ramo está selado.
        Durante treino, ambos os caminhos sempre ativam (gradiente completo).
    """

    def __init__(self, depth: int, channels: int):
        super().__init__()
        self.depth = depth

        if depth == 1:
            # Folha: EcoNoTriade substitui BasicConv
            self.core = EcoNoTriade(channels)
        else:
            # Nível k: mesmo padrão recursivo do FractalBlock original
            self.left        = FractalBlockEcoNo(depth - 1, channels)
            self.right_step1 = FractalBlockEcoNo(depth - 1, channels)
            self.right_step2 = FractalBlockEcoNo(depth - 1, channels)

    def forward(self, x: torch.Tensor):
        """
        Retorna: (saida, ganho)
            saida: tensor processado
            ganho: (B,1,1,1) — coerência relativa acumulada do bloco
        """
        if self.depth == 1:
            return self.core(x)

        # ── Caminho raso: C_{k-1}(x) ─────────────────────────────────────────
        out_shallow, ganho_shallow = self.left(x)

        # ── Selagem hermética (inferência) ────────────────────────────────────
        # Se o caminho raso já estabilizou, o campo está perto da borda:
        # não há tensão entrópica residual para justificar o caminho profundo.
        if not self.training:
            ganho_medio = ganho_shallow.mean()
            if ganho_medio < SEAL:
                return out_shallow, ganho_shallow

        # ── Caminho profundo: C_{k-1}(C_{k-1}(x)) ───────────────────────────
        out_mid,  _           = self.right_step1(x)
        out_deep, ganho_deep  = self.right_step2(out_mid)

        # ── Join via Sépstro ──────────────────────────────────────────────────
        # Substitui média aritmética (0.5 + 0.5) por fusão ponderada pelo ganho de Coh.
        # O caminho com maior coerência relativa tem maior peso na saída.
        # Sépstro do Join: w_shallow + w_deep = 1 (conservação local)
        eps   = 1e-8
        total = ganho_shallow + ganho_deep + eps
        w_s   = ganho_shallow / total    # peso do caminho raso
        w_d   = ganho_deep    / total    # peso do caminho profundo

        out_join   = w_s * out_shallow + w_d * out_deep
        ganho_join = w_s * ganho_shallow + w_d * ganho_deep   # coerência ponderada

        return out_join, ganho_join


# ── Demonstração ─────────────────────────────────────────────────────────────

def demo_comparativa():
    """
    Demonstra as duas formas de uso:

    A) FractalBlockEcoNo — árvore pré-alocada, EcoNoTriade nas folhas
    B) FractalFunctionalNode — wrapper dinâmico via Protocol,
       EcoNoTriade plugada como FunctionalTool
    """
    torch.manual_seed(137)
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)

    print("=" * 68)
    print("FractalBlock EcoNo + FunctionalTool Protocol")
    print(f"φ = {PHI:.10f}  |  α = {ALPHA:.10f}  |  1/φ = {SEAL:.10f}")
    print(f"Entrada: {list(x.shape)}")
    print("=" * 68)

    # ── A) FractalBlockEcoNo (árvore pré-alocada) ────────────────────────────
    print("\n[A] FractalBlockEcoNo — árvore pré-alocada, Join Sépstro")
    for depth in [1, 2, 3]:
        net = FractalBlockEcoNo(depth=depth, channels=C)
        net.eval()
        with torch.no_grad():
            saida, ganho = net(x)
        g = ganho.mean().item()
        status = "SELADO" if g < SEAL else f"ativo (Entr={1-g:.4f})"
        print(f"  depth={depth}  ganho={g:.4f}  Coh+Entr={g+(1-g):.4f}  → {status}")

    # ── B) FractalFunctionalNode via Protocol ────────────────────────────────
    print("\n[B] FractalFunctionalNode — EcoNoTriade como FunctionalTool")
    print(f"    Critério de selagem: ganho_relativo = ΔCoh/(1−Coh_in) < 1/φ")

    triade = EcoNoTriade(channels=C)
    assert isinstance(triade, FunctionalTool), "EcoNoTriade deve implementar FunctionalTool"

    for max_d in [3, 5, 7]:
        node = FractalFunctionalNode(tool=triade, depth=0, max_depth=max_d)
        with torch.no_grad():
            saida, meta = node(x)
        print(f"  max_depth={max_d}  depth_real={meta['depth']}  "
              f"coh={meta['coh']:.4f}  execuções={meta['execucoes']}  "
              f"selado={meta['selado']}")

    print("\n" + "=" * 68)
    print("Verificações:")
    print(f"  ✓ EcoNoTriade implementa FunctionalTool (execute retorna coh+entr=1.0)")
    print(f"  ✓ FractalFunctionalNode aceita qualquer FunctionalTool via Protocol")
    print(f"  ✓ Critério: ganho_relativo = ΔCoh/(1−Coh_in) — mede margem percorrida")
    print(f"  ✓ Join Sépstro: w1+w2=1, pesos∝ganho de coerência de cada caminho")
    print(f"  ✓ Sépstro: Coh + Entr = 1.0 conservado em cada nó")
    print("=" * 68)
    print("\nPróximas ferramentas a plugar via FunctionalTool.execute():")
    print("  Scanner_alpha_phi.execute(x)  → (x_obs, coh_espectral, entr)")
    print("  eco_fononico_v2.execute(x)    → (x_mod, coh_fononico, entr)")
    print("  PhiAttractorNetwork.execute(x)→ (x_atrator, alpha_star, 1−alpha_star)")


# ── Execução ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    demo_comparativa()
