# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_NN_Estagio2.py
Vitor Edson Delavi · Florianópolis · 2026

PROTOCOLO ALPHA-PHI PARA REDES NEURAIS — Estágios 0 · I · II

Estágio 0 — Sugestão do Campo
  Dimensões Fibonacci: 55 → 34 → 21 → 13 → 8 → 5 → 3
  Razão entre camadas ≈ φ. Campo sem limite pré-definido.

Estágio I — Ancoramento α
  Biases inicializados em ALPHA (r = 0, âncora entrópica).
  Pesos escalados por φ^(-nivel).

Estágio II — Sépstro como Invariante Projetado  ← NOVO
  Coh + Entr = 1.0 como constraint estrutural, não como penalidade no loss.
  Cada camada é reprojetada para seguir a trajetória do modelo espacial canônico:
    r = 0  (entrada)  → Coh = ALPHA         (centro, alta entropia)
    r = 1  (saída)    → Coh = 1 − ALPHA     (superfície, alta coerência)
  A forma do espaço guia as ativações. Não é uma regra imposta — é a geometria.

Distinção fundamental:
  Normalização de camada → projeta para variância unitária (estatística)
  Sépstro projetado      → projeta para coerência do modelo espacial (geometria)
"""

import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constantes fundamentais ────────────────────────────────────────────────────
PHI      = (1 + np.sqrt(5)) / 2
ALPHA    = 1 / 137.035999
ALPHA_OP = 1 / 3
SEAL     = 1 / PHI
THETA_R  = np.degrees(np.arctan(2))   # 63.43°

# ── Dimensões Fibonacci ────────────────────────────────────────────────────────
DIMS = [55, 34, 21, 13, 8, 5, 3]
N_CAMADAS = len(DIMS) - 1   # 6


# ── Sépstro local por camada ───────────────────────────────────────────────────

class Septro:
    def __init__(self):
        self.coh  = ALPHA
        self.entr = 1.0 - ALPHA

    def atualizar(self, alinhamento):
        margem = 1.0 - self.coh
        ganho  = float(alinhamento) * margem * SEAL
        ganho  = max(0.0, min(ganho, margem))
        self.coh  += ganho
        self.entr  = 1.0 - self.coh
        return self.coh, self.entr


# ── Projeção Sépstro (Estágio II) ──────────────────────────────────────────────

def medir_coh(v: torch.Tensor) -> float:
    """Coerência φ de um vetor de ativações: 1 − H_normalizada."""
    a = torch.abs(v.flatten()) + 1e-10
    p = a / a.sum()
    H     = -torch.sum(p * torch.log(p))
    H_max = torch.log(torch.tensor(float(max(len(a), 2))))
    return float(1.0 - H / H_max)


def target_coh_para_nivel(nivel: int) -> float:
    """
    Trajetória canônica do Sépstro:
      nivel 0  →  Coh = ALPHA       (r = 0, centro entrópico)
      nivel N  →  Coh = 1 − ALPHA   (r = 1, superfície coerente)
    """
    r = nivel / max(N_CAMADAS - 1, 1)
    return ALPHA + r * (1.0 - 2.0 * ALPHA)


class SeptroProjecao(nn.Module):
    """
    Estágio II — constraint estrutural.
    Projeta ativações para que a coerência siga a trajetória canônica.
    Não modifica pesos nem gera gradiente adicional.
    Opera na forma do espaço, não no conteúdo.
    """

    def __init__(self, nivel: int):
        super().__init__()
        self.nivel      = nivel
        self.target     = target_coh_para_nivel(nivel)
        self.coh_antes  = 0.0
        self.coh_depois = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.coh_antes = medir_coh(x)
        delta = self.target - self.coh_antes   # quanto mover

        # Ajuste SEAL: suave, proporcional ao desvio
        fator = delta * SEAL

        norm  = torch.norm(x) + 1e-8
        x_n   = x / norm

        # Amplifica/atenua proporcionalmente à magnitude de cada ativação:
        # delta > 0 → mais coerência → grandes valores crescem mais
        # delta < 0 → mais entropia  → grandes valores encolhem
        x_proj = x_n * (1.0 + fator * torch.abs(x_n))
        x_proj = x_proj / (torch.norm(x_proj) + 1e-8) * norm

        self.coh_depois = medir_coh(x_proj)
        return x_proj


# ── Camada Alpha-Phi com Sépstro projetado ────────────────────────────────────

class CamadaAP_II(nn.Module):
    def __init__(self, d_in, d_out, nivel):
        super().__init__()
        self.fc      = nn.Linear(d_in, d_out, bias=True)
        self.nivel   = nivel
        self.septro  = Septro()
        self.proj    = SeptroProjecao(nivel)
        self._init_campo_phi()

    def _init_campo_phi(self):
        escala = PHI ** (-self.nivel)
        nn.init.normal_(self.fc.weight, mean=0.0, std=escala)
        nn.init.constant_(self.fc.bias, ALPHA)

    def forward(self, x):
        y = torch.tanh(self.fc(x))
        y = self.proj(y)                              # ← Estágio II
        alin = float(torch.abs(y).mean().clamp(0, 1))
        self.septro.atualizar(alin)
        return y


# ── Rede Estágio 0 · I · II ───────────────────────────────────────────────────

class RedeAP_II(nn.Module):
    def __init__(self):
        super().__init__()
        self.camadas = nn.ModuleList([
            CamadaAP_II(DIMS[i], DIMS[i + 1], nivel=i)
            for i in range(N_CAMADAS)
        ])
        self.ativacoes = []

    def forward(self, x):
        self.ativacoes = [x.detach().cpu().numpy().flatten()]
        for c in self.camadas:
            x = c(x)
            self.ativacoes.append(x.detach().cpu().numpy().flatten())
        return x

    def tabela_projecao(self):
        rows = []
        for i, c in enumerate(self.camadas):
            rows.append({
                'camada':     i,
                'r':          round(i / max(N_CAMADAS - 1, 1), 3),
                'target_coh': round(c.proj.target,     6),
                'coh_antes':  round(c.proj.coh_antes,  6),
                'coh_depois': round(c.proj.coh_depois, 6),
                'delta':      round(c.proj.coh_depois - c.proj.coh_antes, 6),
                'sep_coh':    round(c.septro.coh,      6),
            })
        return rows


# ── Sinal EcoBIP ───────────────────────────────────────────────────────────────

def sinal_ecobip(n=1024):
    t = np.linspace(0, 1, n, endpoint=False)
    q = np.sign(np.sin(2 * np.pi * 880 * t))
    f = np.sin(2 * np.pi * 880 * t + PHI * np.sin(2 * np.pi * 220 * t))
    return (1 - ALPHA_OP) * q + ALPHA_OP * f

def comprime(sig, n):
    idx = np.linspace(0, len(sig) - 1, n).astype(int)
    return sig[idx].astype(np.float32)


# ── Execução ───────────────────────────────────────────────────────────────────

torch.manual_seed(42)
rede = RedeAP_II()

sinal_bruto = sinal_ecobip(1024)
entrada = torch.tensor(comprime(sinal_bruto, DIMS[0])).unsqueeze(0)

with torch.no_grad():
    _ = rede(entrada)

tab = rede.tabela_projecao()

# ── Relatório terminal ─────────────────────────────────────────────────────────

print("=" * 70)
print("Alpha-Phi NN — Estágio 0 · I · II")
print("=" * 70)
print(f"  PHI    = {PHI:.7f}    ALPHA  = {ALPHA:.7f}")
print(f"  SEAL   = {SEAL:.7f}    θ_R   = {THETA_R:.4f}°")
print()
print(f"Arquitetura: {' → '.join(str(d) for d in DIMS)}")
print()
print("Trajetória Sépstro — modelo espacial canônico:")
print(f"  r = 0.000  Coh_target = {ALPHA:.6f}  (centro, α)")
print(f"  r = 1.000  Coh_target = {1-ALPHA:.6f}  (superfície, φ)")
print()
print(f"{'C':>3}  {'r':>6}  {'target':>10}  {'antes':>10}  {'depois':>10}  "
      f"{'Δ':>8}  {'Sep.Coh':>10}")
for r in tab:
    print(f"{r['camada']:>3}  {r['r']:>6.3f}  {r['target_coh']:>10.6f}  "
          f"{r['coh_antes']:>10.6f}  {r['coh_depois']:>10.6f}  "
          f"{r['delta']:>+8.6f}  {r['sep_coh']:>10.6f}")

print()
print("Campo Alpha-Phi Estágio 0·I·II — pronto.")
print("Próximo: Estágio III — PhiAttractorNetwork + scanner de ativações.")


# ── Visualização ───────────────────────────────────────────────────────────────

CORES = ['#FFD700', '#FFA040', '#FF5733', '#C0392B', '#8E44AD', '#2471A3', '#1ABC9C']

fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        'Ativações por Camada — Campo Alpha-Phi Estágio 0·I·II',
        'Trajetória de Coerência — Target vs. Realizado (antes/depois da projeção)',
        'Sépstro por Camada  ·  Coh + Entr = 1.0'
    ),
    vertical_spacing=0.09
)

# Painel 1 — ativações
for i, (a, cor) in enumerate(zip(rede.ativacoes, CORES)):
    nome = 'Entrada' if i == 0 else f'C{i-1}  dim={len(a)}'
    fig.add_trace(go.Scatter(y=a, mode='lines', name=nome,
                             line=dict(color=cor, width=1.5)), row=1, col=1)

# Painel 2 — trajetória de coerência
xs      = [r['r']          for r in tab]
targets = [r['target_coh'] for r in tab]
antes   = [r['coh_antes']  for r in tab]
depois  = [r['coh_depois'] for r in tab]

fig.add_trace(go.Scatter(x=xs, y=targets, mode='lines', name='Coh target',
                         line=dict(color='white', width=2, dash='dash')), row=2, col=1)
fig.add_trace(go.Scatter(x=xs, y=antes,   mode='lines+markers', name='Coh antes',
                         line=dict(color='#FF5733', width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=xs, y=depois,  mode='lines+markers', name='Coh depois',
                         line=dict(color='#FFD700', width=2)), row=2, col=1)
fig.add_hline(y=ALPHA,     line_dash='dot', line_color='rgba(255,255,255,0.3)',
              annotation_text=f'α={ALPHA:.4f}', row=2, col=1)
fig.add_hline(y=1.0-ALPHA, line_dash='dot', line_color='rgba(255,255,255,0.3)',
              annotation_text=f'1−α={1-ALPHA:.4f}', row=2, col=1)

# Painel 3 — Sépstro
cohs  = [r['sep_coh']         for r in tab]
entrs = [1.0 - r['sep_coh']   for r in tab]
xs_c  = list(range(len(cohs)))

fig.add_trace(go.Scatter(x=xs_c, y=cohs,  mode='lines+markers', name='Coh (Sépstro)',
                         line=dict(color='#FFD700', width=2)), row=3, col=1)
fig.add_trace(go.Scatter(x=xs_c, y=entrs, mode='lines+markers', name='Entr (Sépstro)',
                         line=dict(color='#4A90D9', width=2)), row=3, col=1)
fig.add_hline(y=ALPHA, line_dash='dot', line_color='rgba(255,255,255,0.3)',
              annotation_text=f'α={ALPHA:.4f}', row=3, col=1)

fig.update_layout(
    template='plotly_dark', height=1000,
    title=dict(
        text='Alpha-Phi NN — Estágio 0·I·II · Sépstro como Invariante Projetado',
        font=dict(size=15)
    )
)
fig.update_xaxes(title_text='Neurônio', row=1, col=1)
fig.update_xaxes(title_text='r  (0 = α-centro  →  1 = φ-superfície)', row=2, col=1)
fig.update_xaxes(title_text='Camada', row=3, col=1)

fig.show()
