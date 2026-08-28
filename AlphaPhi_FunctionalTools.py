# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_FunctionalTools.py
Vitor Edson Delavi · Florianópolis · 2026

Implementação de execute() para as ferramentas Alpha-Phi existentes.

Cada wrapper implementa o FunctionalTool Protocol:
    execute(x: Tensor) → (tensor_processado, coh: float, entr: float)

Isso permite plugar Scanner, eco_fononico_v2 e PhiAttractorNetwork
no FractalFunctionalNode sem reescrever o núcleo de cada ferramenta.

Ferramentas:
    EcoFononicoV2Tool   — wraps eco_fononico_v2 (numpy DSP)
    ScannerTool         — wraps ScannerAlphaPhi (observação pura)
    PhiAttractorTool    — wraps PhiAttractorNetwork (rede Fibonacci)

Modelo espacial:
    ScannerTool      → r = 0  (Observar — lê o campo, nunca modifica)
    PhiAttractorTool → 0<r<1  (Selecionar — atrator entre α e φ)
    EcoFononicoV2Tool→ r → 1  (Agir — produz o campo harmônico)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


PHI   = (1 + math.sqrt(5)) / 2
ALPHA = 1 / 137.035999
SEAL  = 1 / PHI


# ── EcoFononicoV2Tool ─────────────────────────────────────────────────────────

class EcoFononicoV2Tool(nn.Module):
    """
    Wrapper execute() para eco_fononico_v2.

    eco_fononico_v2 opera em numpy — este wrapper:
      1. Converte tensor → numpy
      2. Executa eco_fononico_v2 (coupling=φ, k=√2)
      3. Extrai coh do campo via medir_campo
      4. Retorna tensor processado + Sépstro (coh, entr)

    Coh retornada = coh_campo medida diretamente pelo Sépstro espectral.
    Sela quando campo atinge coerência harmônica sustentada (Coh ≈ 0.988+).
    """

    N_ECO    = 5        # ciclos de eco — mesmo padrão dos experimentos
    K_MIN    = math.sqrt(2)   # k = √2 como rotação natural do campo

    def execute(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        device = x.device
        X = x.detach().cpu().numpy()

        # Medição do campo ANTES
        freq      = np.fft.fft(X, axis=-1)
        amp_media = np.abs(freq).mean(axis=0) if X.ndim > 1 else np.abs(freq)
        amp_norm  = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
        entropia  = -np.sum(amp_norm * np.log(amp_norm))
        n_bins    = X.shape[-1]
        coh_in    = float(1.0 - entropia / np.log(n_bins))
        k         = self.K_MIN + (PHI - self.K_MIN) * coh_in

        # eco_fononico_v2: coupling = φ
        s = X.copy()
        for _ in range(self.N_ECO):
            freq_s   = np.fft.fft(s, axis=-1)
            reflexao = np.real(np.fft.ifft(
                np.abs(freq_s) * np.exp(1j * np.angle(freq_s) * k), axis=-1))
            s = s + (reflexao - X) * PHI

        # Medição do campo DEPOIS
        freq_out    = np.fft.fft(s, axis=-1)
        amp_out     = np.abs(freq_out).mean(axis=0) if s.ndim > 1 else np.abs(freq_out)
        amp_norm_out= np.clip(amp_out / (amp_out.sum() + 1e-8), 1e-10, 1.0)
        entr_out    = -np.sum(amp_norm_out * np.log(amp_norm_out))
        coh_out     = float(np.clip(1.0 - entr_out / np.log(n_bins), ALPHA, 1.0 - ALPHA))
        entr_val    = 1.0 - coh_out

        tensor_out = torch.tensor(s, dtype=x.dtype, device=device)
        return tensor_out, coh_out, entr_val


# ── ScannerTool ───────────────────────────────────────────────────────────────

class ScannerTool(nn.Module):
    """
    Wrapper execute() para o Scanner α-φ.

    INVARIANTE: o Scanner NUNCA modifica o sinal.
    execute() retorna o tensor ORIGINAL — apenas relata o estado do campo.

    Coh = média das coerências por banda φ-geométrica.
    O Scanner é Observar do sistema — r = 0, próximo de α.

    Bandas φ-geométricas: cada banda = φ × anterior (escala log-φ).
    """

    N_FFT  = 1024
    N_BAND = 9         # 9 bandas de 129Hz a 8000Hz para FS=16000

    def _bandas_phi(self, n_bins: int):
        """Fronteiras das bandas φ-geométricas em índices de bin."""
        limites = [0]
        atual   = max(1, int(n_bins * ALPHA))
        while atual < n_bins and len(limites) < self.N_BAND + 1:
            limites.append(atual)
            atual = min(n_bins, int(atual * PHI))
        limites.append(n_bins)
        return limites

    def execute(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        X = x.detach().cpu().numpy()

        # Garante 1D ou 2D (batch × amostras)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        n_bins = X.shape[-1] // 2 + 1
        bandas  = self._bandas_phi(n_bins)
        cohs    = []

        for b in range(len(bandas) - 1):
            lo, hi = bandas[b], bandas[b + 1]
            if hi <= lo:
                continue
            # Energia na banda
            freq_X  = np.fft.rfft(X, axis=-1)[:, lo:hi]
            amp     = np.abs(freq_X)
            amp_n   = np.clip(amp / (amp.sum(axis=-1, keepdims=True) + 1e-8), 1e-10, 1.0)
            H       = -np.sum(amp_n * np.log(amp_n), axis=-1).mean()
            H_max   = np.log(hi - lo) if hi - lo > 1 else 1.0
            coh_b   = float(np.clip(1.0 - H / H_max, 0.0, 1.0))
            cohs.append(coh_b)

        coh_media = float(np.mean(cohs)) if cohs else ALPHA
        coh_val   = float(np.clip(coh_media, ALPHA, 1.0 - ALPHA))
        entr_val  = 1.0 - coh_val

        # Scanner NUNCA modifica o sinal — retorna x original
        return x, coh_val, entr_val


# ── PhiAttractorTool ──────────────────────────────────────────────────────────

class PhiAttractorTool(nn.Module):
    """
    Wrapper execute() para PhiAttractorNetwork.

    PhiAttractorNetwork produz alpha_pred ∈ [1e-4, 0.35] por fonema —
    o ponto de emergência α* (não a constante de estrutura fina).

    Mapeamento para o Sépstro:
        coh = 1 − (alpha_pred / ALPHA_MAX)
            → α* baixo  = campo mais coerente (próximo da borda)
            → α* alto   = campo mais entrópico (próximo do centro α)

    A rede Fibonacci (89→55→34→21→13→8) opera no gradiente intermediário:
    r ∈ (0, 1) — entre o Scanner (Observar) e o eco_fononico (Agir).

    Parâmetros:
        net: instância de PhiAttractorNetwork já inicializada
    """

    ALPHA_MAX = 0.35

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def execute(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        # PhiAttractorNetwork espera (batch, n_features)
        # Se x for (B, C, H, W) ou (B, C, N), achatamos para (B, -1)
        shape_orig = x.shape
        x_flat = x.reshape(x.shape[0], -1).float()

        with torch.no_grad():
            resultado = self.net(x_flat)

        alpha_pred = resultado['alpha_pred']          # (batch,) — α* por item
        coerencias = resultado.get('coerencias', [])  # coerências por camada

        alpha_medio = alpha_pred.mean().item()

        # Mapeamento α* → Coh: α* baixo = mais coerente
        coh_val  = float(np.clip(1.0 - alpha_medio / self.ALPHA_MAX, ALPHA, 1.0 - ALPHA))
        entr_val = 1.0 - coh_val

        # Saída: x modulado pelo estado do atrator (coh pondera a passagem)
        # O atrator não cria campo — informa o campo existente
        x_out = x * coh_val

        return x_out, coh_val, entr_val


# ── Verificação de conformidade com o Protocol ────────────────────────────────

def verificar_tools(sinal_1d: torch.Tensor = None):
    """
    Verifica que cada tool implementa corretamente o FunctionalTool Protocol.
    Usa sinal sintético se nenhum for fornecido.
    """
    from AlphaPhi_FractalBlock_EcoNo import FunctionalTool, FractalFunctionalNode

    if sinal_1d is None:
        torch.manual_seed(137)
        sinal_1d = torch.randn(1, 1024)   # 1 sample, 1024 amostras

    print("=" * 64)
    print("AlphaPhi FunctionalTools — verificação do Protocol execute()")
    print(f"φ={PHI:.6f}  α={ALPHA:.6f}  1/φ={SEAL:.6f}")
    print(f"Sinal: {list(sinal_1d.shape)}")
    print("=" * 64)

    tools = [
        ("ScannerTool         [Observar, r=0]",     ScannerTool()),
        ("EcoFononicoV2Tool   [Agir,     r→1]",    EcoFononicoV2Tool()),
    ]

    for nome, tool in tools:
        assert isinstance(tool, FunctionalTool), f"{nome} não implementa FunctionalTool"
        saida, coh, entr = tool.execute(sinal_1d)
        soma = coh + entr
        status = "✓ Sépstro OK" if abs(soma - 1.0) < 1e-6 else f"✗ ERRO soma={soma:.6f}"
        print(f"\n{nome}")
        print(f"  Saída: {list(saida.shape)}  coh={coh:.4f}  entr={entr:.4f}  "
              f"coh+entr={soma:.6f}  {status}")

        # Plugar no FractalFunctionalNode
        node  = FractalFunctionalNode(tool=tool, depth=0, max_depth=3)
        with torch.no_grad():
            _, meta = node(sinal_1d)
        print(f"  Fractal: depth={meta['depth']}  coh={meta['coh']:.4f}  "
              f"execuções={meta['execucoes']}  selado={meta['selado']}")

    print("\n" + "=" * 64)
    print("PhiAttractorTool — requer instância treinada de PhiAttractorNetwork")
    print("  Uso: tool = PhiAttractorTool(net=minha_rede_treinada)")
    print("       node = FractalFunctionalNode(tool=tool, max_depth=5)")
    print("=" * 64)
    print("\nOrdem de inserção na esfera (modelo espacial canônico):")
    print("  r=0    ScannerTool         → Observar — lê sem modificar")
    print("  0<r<1  PhiAttractorTool    → Selecionar — atrator Fibonacci")
    print("  r→1    EcoFononicoV2Tool   → Agir — produz campo harmônico")


# ── Execução ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    verificar_tools()
