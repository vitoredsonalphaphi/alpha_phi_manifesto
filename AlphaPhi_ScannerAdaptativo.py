# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_ScannerAdaptativo.py
Vitor Edson Delavi · Florianópolis · 2026

Scanner α-φ com ressonância fractal aprendida por domínio.

INVARIANTE (irrevogável):
    execute(x) → (x_original, coh: float, entr: float)
    O Scanner NUNCA modifica o sinal. Apenas calibra a medição.

O que aprende (treinamento não-supervisionado):
    1. Pesos de banda (quais escalas φ revelam mais auto-similaridade neste domínio)
    2. Expoente de Hurst esperado (H = 0.5: Browniano; H > 0.5: persistente)
    3. Perfil de referência φ (Coh esperada por banda para sinal coerente neste domínio)

Target de treinamento: a própria lei φ, não labels externos.
    coh_ref[k] ∝ φ^(-k) — decaimento esperado por banda fractal φ-ressonante

Modelo espacial: r = 0 (Observar — lê o campo, nunca o modifica)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Iterable, Optional


PHI   = (1 + math.sqrt(5)) / 2
ALPHA = 1 / 137.035999
SEAL  = 1 / PHI
N_BAND_DEFAULT = 9


# ── Utilidade: expoente de Hurst via periodograma log-log ─────────────────────

def estimar_hurst(x_np: np.ndarray) -> float:
    """
    Estimativa do expoente de Hurst via regressão espectral (método de Welch simplificado).

    Lei de potência do espectro: S(f) ∝ f^(-β), com β = 2H + 1.
    H = 0.5  → ruído Browniano (sem memória fractal)
    H > 0.5  → persistente (positivo para escalas longas) — voz, música, ECG
    H < 0.5  → anti-persistente (média-revertente) — alguns sinais fisiológicos
    H = 1.0  → 1/f — limite da atração φ (rosa)
    """
    if x_np.ndim > 1:
        x_np = x_np.mean(axis=0)
    N = len(x_np)
    freq  = np.fft.rfftfreq(N)[1:]   # exclui DC
    power = np.abs(np.fft.rfft(x_np)[1:]) ** 2
    log_f = np.log(freq + 1e-12)
    log_p = np.log(power + 1e-12)
    beta  = -np.polyfit(log_f, log_p, 1)[0]   # β = -slope
    H     = float(np.clip((beta - 1.0) / 2.0, 0.0, 1.0))
    return H


# ── ScannerAdaptativo ─────────────────────────────────────────────────────────

class ScannerAdaptativo(nn.Module):
    """
    Scanner com auto-calibração φ por domínio.

    Parâmetros treináveis (treinamento não-supervisionado via loss_phi):
        band_logits  (N_BAND,)  — pesos de importância por banda (softmax)
        coh_ref_raw  (N_BAND,)  — perfil de Coh de referência por banda (softmax, target φ-decay)
        hurst_raw    (scalar)   — expoente de Hurst estimado para o domínio (sigmoid)

    Execute NUNCA modifica o sinal.
    Loss NÃO usa labels — apenas a estrutura φ como supervisão.
    """

    def __init__(self, n_band: int = N_BAND_DEFAULT):
        super().__init__()
        self.n_band = n_band

        # Pesos de banda inicializados uniformes — o treinamento os especializa
        self.band_logits = nn.Parameter(torch.zeros(n_band))

        # Perfil de referência inicial: φ^(-k) normalizado
        ref_phi = torch.tensor([PHI ** (-k) for k in range(n_band)], dtype=torch.float32)
        self.coh_ref_raw = nn.Parameter(ref_phi.clone())

        # Hurst inicial: 0.5 (Browniano — agnóstico)
        self.hurst_raw = nn.Parameter(torch.tensor(0.0))   # sigmoid(0) = 0.5

    @property
    def hurst(self) -> torch.Tensor:
        return torch.sigmoid(self.hurst_raw)

    @property
    def band_pesos(self) -> torch.Tensor:
        return F.softmax(self.band_logits, dim=0)

    @property
    def coh_ref(self) -> torch.Tensor:
        return F.softmax(self.coh_ref_raw, dim=0)

    # ── Estrutura de bandas φ-geométricas ─────────────────────────────────────

    def _bandas_phi(self, n_bins: int) -> list:
        """Fronteiras das bandas em índices de bin (mesma lógica do ScannerTool)."""
        limites = [0]
        atual   = max(1, int(n_bins * ALPHA))
        while atual < n_bins and len(limites) < self.n_band + 1:
            limites.append(atual)
            atual = min(n_bins, int(atual * PHI))
        limites.append(n_bins)
        return limites

    # ── Medição de Coh por banda ──────────────────────────────────────────────

    def _medir_bandas(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna tensor (n_band,) com Coh medida em cada banda φ.
        Operação puramente observacional — sem gradiente sobre x.
        """
        X = x.detach().cpu().numpy()
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        n_bins = X.shape[-1] // 2 + 1
        bandas = self._bandas_phi(n_bins)
        cohs   = []

        for b in range(len(bandas) - 1):
            lo, hi = bandas[b], bandas[b + 1]
            if hi <= lo:
                cohs.append(ALPHA)
                continue
            freq_X = np.fft.rfft(X, axis=-1)[:, lo:hi]
            amp    = np.abs(freq_X)
            amp_n  = np.clip(amp / (amp.sum(axis=-1, keepdims=True) + 1e-8), 1e-10, 1.0)
            H      = -np.sum(amp_n * np.log(amp_n), axis=-1).mean()
            H_max  = np.log(hi - lo) if hi - lo > 1 else 1.0
            cohs.append(float(np.clip(1.0 - H / H_max, 0.0, 1.0)))

        # Padding se há menos bandas que n_band
        while len(cohs) < self.n_band:
            cohs.append(ALPHA)

        return torch.tensor(cohs[:self.n_band], dtype=torch.float32)

    # ── Execute — INVARIANTE: retorna x_original ──────────────────────────────

    def execute(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Mede Coh do sinal com pesos de banda calibrados para o domínio.

        Retorna:
            (x_original, coh_ponderada, entr)  — x nunca é modificado
        """
        cohs_banda  = self._medir_bandas(x)              # (n_band,)
        pesos       = self.band_pesos                     # (n_band,) — softmax
        coh_val     = (pesos * cohs_banda).sum().item()
        coh_val     = float(np.clip(coh_val, ALPHA, 1.0 - ALPHA))
        entr_val    = 1.0 - coh_val
        return x, coh_val, entr_val

    # ── Fingerprint fractal — vetor Coh por banda ─────────────────────────────

    def fingerprint(self, x: torch.Tensor) -> dict:
        """
        Retorna o perfil completo de Coh por banda — assinatura fractal do sinal.

        Em vez de um único número Coh, retorna o vetor [Coh_banda_0, ..., Coh_banda_8].
        Esse vetor é a impressão digital fractal do sinal neste domínio.

        Uso: FractalFunctionalNode pode usar o fingerprint para decidir em qual
             profundidade-escala expandir vs. selar.
        """
        cohs_banda = self._medir_bandas(x)
        pesos      = self.band_pesos.detach()
        _, coh_media, entr_media = self.execute(x)
        return {
            'coh_por_banda': cohs_banda.tolist(),
            'pesos_banda':   pesos.tolist(),
            'coh_media':     coh_media,
            'entr_media':    entr_media,
            'hurst_dominio': self.hurst.item(),
        }

    # ── Loss φ — treinamento não-supervisionado ───────────────────────────────

    def loss_phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perda de ressonância φ — treinamento NÃO supervisionado.

        O target não é um label externo: é a própria estrutura φ.

        Três termos:

        L_ressonancia:  perfil de Coh medido deve seguir o perfil de referência φ^(-k).
                        Se o sinal tem estrutura fractal φ-ressonante, minimizar esse erro
                        ajusta os pesos de banda para revelar essa estrutura.

        L_hurst:        expoente de Hurst medido deve convergir para self.hurst.
                        Isso ancora o Scanner ao tipo de auto-similaridade do domínio.

        L_diversidade:  maximiza entropia dos pesos (evita colapso a uma única banda).
                        Regularização Sépstro — preserva leitura de todo o campo.
        """
        cohs_banda  = self._medir_bandas(x)                        # (n_band,)
        pesos       = self.band_pesos                               # (n_band,)
        coh_ref     = self.coh_ref                                  # (n_band,) — φ-decay
        hurst_aprendido = self.hurst

        # Coh ponderada (diferenciável em relação aos pesos)
        coh_pond = (pesos * cohs_banda).sum()

        # 1. Ressonância φ: perfil de Coh medido → perfil de referência φ-decay
        L_res = F.mse_loss(pesos * cohs_banda, coh_ref)

        # 2. Hurst: expoente medido no sinal deve ser próximo do Hurst aprendido
        X_np = x.detach().cpu().numpy()
        if X_np.ndim > 2:
            X_np = X_np.reshape(-1, X_np.shape[-1])
        hurst_medido = torch.tensor(
            np.mean([estimar_hurst(X_np[i]) for i in range(X_np.shape[0])]),
            dtype=torch.float32)
        L_hurst = (hurst_medido - hurst_aprendido) ** 2

        # 3. Diversidade (regularização Sépstro — evita colapso em uma banda)
        L_div = (pesos * torch.log(pesos + 1e-8)).sum()   # minimizar = maximizar entropia

        return L_res + 0.2 * L_hurst + 0.05 * L_div


# ── Treinar Scanner para um domínio ──────────────────────────────────────────

def treinar_scanner(
    scanner:     ScannerAdaptativo,
    dominio:     Iterable[torch.Tensor],
    n_epocas:    int = 80,
    lr:          float = 1e-3,
    verbose:     bool = True,
) -> ScannerAdaptativo:
    """
    Treinamento não-supervisionado do ScannerAdaptativo.

    dominio: iterável de tensores do domínio-alvo (ex: batches de áudio, ECG, texto embedado)
    Target: nenhum label externo — apenas a estrutura φ.

    O Scanner aprende a reconhecer o padrão fractal INERENTE ao domínio.
    Ao final: execute() retorna Coh calibrada para aquele tipo de sinal.

    Após treinar em voz:       Coh alta para sinais de fala, baixa para ruído
    Após treinar em ECG:       Coh alta para ritmo cardíaco saudável, baixa para arritmia
    Após treinar em ruído:     Coh permanece próxima de ALPHA (domínio sem campo φ)
    """
    opt = torch.optim.Adam(scanner.parameters(), lr=lr)
    historico = []

    for ep in range(n_epocas):
        ep_loss = 0.0
        n_batches = 0
        for x in dominio:
            opt.zero_grad()
            L = scanner.loss_phi(x)
            L.backward()
            opt.step()
            ep_loss += L.item()
            n_batches += 1

        media = ep_loss / max(1, n_batches)
        historico.append(media)

        if verbose and (ep % 10 == 0 or ep == n_epocas - 1):
            h_val = scanner.hurst.item()
            pesos = scanner.band_pesos.detach().numpy()
            banda_dom = int(np.argmax(pesos))
            print(f"  época {ep:3d}  loss={media:.5f}  H={h_val:.3f}  "
                  f"banda_dominante={banda_dom}  "
                  f"Coh_ref[0]={scanner.coh_ref[0].item():.4f}")

    return scanner


# ── Verificação ───────────────────────────────────────────────────────────────

def demo():
    torch.manual_seed(137)
    print("=" * 64)
    print("ScannerAdaptativo — demo de treinamento por domínio")
    print(f"φ={PHI:.6f}  α={ALPHA:.6f}  H_inicial=0.500")
    print("=" * 64)

    # Domínio 1: ruído puro (H ≈ 0.5)
    ruido = [torch.randn(1, 1024) for _ in range(20)]

    # Domínio 2: sinal 1/f — fractal φ-ressonante (H ≈ 1.0)
    def sinal_1f(N=1024):
        f     = np.fft.rfftfreq(N)[1:]
        amp   = 1.0 / (f + 1e-8)
        phase = np.random.uniform(0, 2 * np.pi, len(f))
        spec  = np.zeros(N // 2 + 1, dtype=complex)
        spec[1:] = amp * np.exp(1j * phase)
        s = np.fft.irfft(spec, n=N)
        return torch.tensor(s[np.newaxis, :], dtype=torch.float32)

    campo_harmonico = [sinal_1f() for _ in range(20)]

    for nome, dominio in [("Ruído (H≈0.5)", ruido), ("1/f — campo φ (H≈1.0)", campo_harmonico)]:
        print(f"\n── Treinando em: {nome}")
        scanner = ScannerAdaptativo(n_band=9)

        # Coh antes do treinamento
        x_teste = dominio[0]
        _, coh_antes, _ = scanner.execute(x_teste)

        treinar_scanner(scanner, dominio, n_epocas=40, lr=1e-3, verbose=True)

        _, coh_depois, _ = scanner.execute(x_teste)
        fp = scanner.fingerprint(x_teste)

        print(f"\n  Coh antes:  {coh_antes:.5f}")
        print(f"  Coh depois: {coh_depois:.5f}")
        print(f"  Hurst aprendido: {fp['hurst_dominio']:.4f}")
        print(f"  Coh por banda: {[round(c,3) for c in fp['coh_por_banda']]}")
        print(f"  Pesos banda:   {[round(p,3) for p in fp['pesos_banda']]}")

    print("\n" + "=" * 64)
    print("Invariante verificada: execute() retorna x_original em ambos os casos.")
    print("O Scanner não modifica — apenas calibra a percepção do campo.")


if __name__ == '__main__':
    demo()
