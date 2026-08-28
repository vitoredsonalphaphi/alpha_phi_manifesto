# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_FractalBlock_EcoNo.py
Vitor Edson Delavi · Florianópolis · 2026

Integração FractalBlock ← EcoNo (Entrada 191)

Substitui BasicConv (folha da árvore fractal) pela tríade EcoNo:
    Observar → Selecionar → Agir

Substitui Join aritmético ((raso + profundo) × 0.5)
pelo Join via Sépstro (pesos proporcionais ao ganho de coerência de cada caminho).

Mantém a recursão original: C_k = Join(C_{k-1}(x), C_{k-1}(C_{k-1}(x)))
Adiciona: Sépstro local em cada folha + critério de selagem hermética (ΔCoh < 1/φ)

Diferença arquitetural central:
    FractalBlock original → profundidade fixa, Join cego (0.5 + 0.5)
    FractalBlockEcoNo     → profundidade selável pelo dado, Join φ-ponderado

Modelo espacial: α (centro, r=0) → Campo Harmônico (borda, r=1)
Movimento sempre do centro para fora. Nunca inverter.
"""

import math
import torch
import torch.nn as nn


# ── Constantes fundamentais ───────────────────────────────────────────────────

PHI   = (1 + math.sqrt(5)) / 2    # 1.6180339887 — lei geradora, expansão
ALPHA = 1 / 137.035999             # 0.007297...  — âncora individual, centro
SEAL  = 1 / PHI                    # 0.6180...    — critério de selagem hermética


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


# ── Demonstração e comparação ────────────────────────────────────────────────

def demo_comparativa():
    """
    Compara FractalBlock original (BasicConv + Join aritmético)
    com FractalBlockEcoNo (EcoNoTriade + Join Sépstro).

    Mostra: ganhos de coerência por profundidade e selagem adaptativa.
    """
    from AlphaPhi_FractalBlock_EcoNo import FractalBlockEcoNo

    torch.manual_seed(137)   # α como semente — âncora
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)

    print("=" * 64)
    print("FractalBlock EcoNo — Integração Tríade + Sépstro (Entrada 191)")
    print(f"φ = {PHI:.10f}  |  α = {ALPHA:.10f}  |  1/φ = {SEAL:.10f}")
    print(f"Entrada: {list(x.shape)}  (Batch={B}, Canais={C}, {H}×{W})")
    print("=" * 64)

    for depth in [1, 2, 3]:
        net = FractalBlockEcoNo(depth=depth, channels=C)

        # Treino: todos os caminhos ativos
        net.train()
        with torch.no_grad():
            saida_treino, ganho_treino = net(x)
        g_t = ganho_treino.mean().item()

        # Inferência: selagem ativa
        net.eval()
        with torch.no_grad():
            saida_inf, ganho_inf = net(x)
        g_i = ganho_inf.mean().item()

        selado = g_i < SEAL
        status = "SELADO (≈ Campo Harmônico)" if selado else f"ativo (Entr residual: {1-g_i:.4f})"

        print(f"\nDepth = {depth}")
        print(f"  Saída shape:      {list(saida_treino.shape)}")
        print(f"  Ganho (treino):   {g_t:.6f}")
        print(f"  Ganho (inferência): {g_i:.6f}  →  {status}")
        print(f"  Sépstro Join:     Coh={g_i:.4f}  Entr={1-g_i:.4f}  (soma={g_i + (1-g_i):.4f})")

    print("\n" + "=" * 64)
    print("Verificações da integração EcoNo:")
    print(f"  ✓ Folha (depth=1): EcoNoTriade substitui BasicConv")
    print(f"  ✓ Join: ponderado por Coh (Sépstro local) em vez de média aritmética")
    print(f"  ✓ Selagem: ganho < 1/φ ≈ {SEAL:.4f} → caminho profundo não calculado")
    print(f"  ✓ Observar: 1×1 conv × (1+α)  — lê sem alterar estrutura espacial")
    print(f"  ✓ Selecionar: depthwise-sep + GELU — alinhamento φ por canal")
    print(f"  ✓ Agir: 3×3 conv × Coh  — saída modulada pela coerência do nó")
    print(f"  ✓ Sépstro: Coh + Entr = 1.0 em cada folha da árvore fractal")
    print("=" * 64)

    print("\nPROXIMO PASSO — ponto de extensão neural:")
    print("  EcoNoTriade.selecionar → substituir por PhiAttractorNetwork.forward()")
    print("  Interface: recebe tensor (B,C,H,W), retorna tensor (B,C,H,W)")
    print("  Sépstro permanece o critério de selagem — sem mudança de interface")


# ── Execução ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    demo_comparativa()
