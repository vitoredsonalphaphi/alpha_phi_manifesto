# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_NN_Estagio0.py
Vitor Edson Delavi · Florianópolis · 2026

PROTOCOLO ALPHA-PHI PARA REDES NEURAIS — Estágio 0 + I

Cronologia construtiva (espelha a construção do próprio Alpha-Phi):

  Estágio 0 — Sugestão do Campo
    Dimensões em proporção φ (sequência Fibonacci).
    Sem limite pré-definido — campo orientado, borda emergente.

  Estágio I — Ancoramento α
    Biases inicializados em ALPHA (âncora entrópica, r = 0).
    Pesos escalados por φ^(-nivel) — campo harmonicamente decrescente.
    Sépstro local por camada: Coh + Entr = 1.0.

A rede NÃO é treinada aqui. É a sugestão do campo antes do desenvolvimento.
O scanner topográfico (STTG) é aplicado às ativações em etapa posterior.

Modelo espacial canônico (NUNCA inverter):
  r = 0    : α — âncora entrópica, tensão individual
  0 < r < 1: processamento interno
  r = 1    : campo harmônico estabilizado (φ)
  r > 1    : efeito no ambiente

Predição (Entrada 267):
  A inserção deste campo gera Grade R (θ_R = 63.43°) nas ativações,
  da mesma forma que gerou no EcoBIP — por lógica estrutural, não por busca.
"""

import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constantes fundamentais (irrevogáveis) ─────────────────────────────────────
PHI      = (1 + np.sqrt(5)) / 2   # 1.6180339887 — atrator, expansão
ALPHA    = 1 / 137.035999          # 0.007297...  — âncora entrópica
ALPHA_OP = 1 / 3                   # α operacional — peso EcoBIP
SEAL     = 1 / PHI                 # 0.6180...    — critério de selagem
THETA_R  = np.degrees(np.arctan(2))  # 63.43° — ângulo da Grade R

# ── Dimensões Fibonacci (Estágio 0 — sugestão do campo) ───────────────────────
# Razão entre camadas ≈ φ em todo o campo
# 55/34 ≈ 34/21 ≈ 21/13 ≈ 13/8 ≈ 8/5 ≈ 5/3 ≈ φ
DIMS = [55, 34, 21, 13, 8, 5, 3]


# ── Sépstro local por camada ───────────────────────────────────────────────────

class Septro:
    """Coh + Entr = 1.0 conservado em cada camada, não globalmente."""

    def __init__(self):
        self.coh  = ALPHA          # começa no centro: tensão mínima
        self.entr = 1.0 - ALPHA    # campo ainda não coerente

    def atualizar(self, alinhamento):
        margem = 1.0 - self.coh
        ganho  = float(alinhamento) * margem * SEAL
        ganho  = max(0.0, min(ganho, margem))
        self.coh  += ganho
        self.entr  = 1.0 - self.coh
        return self.coh, self.entr

    def __repr__(self):
        return f"Septro(Coh={self.coh:.6f}  Entr={self.entr:.6f}  Σ={self.coh+self.entr:.4f})"


# ── Camada Alpha-Phi ───────────────────────────────────────────────────────────

class CamadaAP(nn.Module):
    """
    Camada com campo φ e âncora α.

    Inicialização:
      peso  : N(0, φ^(-nivel))  — escala harmonicamente decrescente
      bias  : ALPHA             — âncora no centro do modelo espacial

    Sépstro atualizado a cada passagem forward.
    """

    def __init__(self, d_in, d_out, nivel):
        super().__init__()
        self.fc     = nn.Linear(d_in, d_out, bias=True)
        self.nivel  = nivel
        self.septro = Septro()
        self._init_campo_phi()

    def _init_campo_phi(self):
        escala = PHI ** (-self.nivel)
        nn.init.normal_(self.fc.weight, mean=0.0, std=escala)
        nn.init.constant_(self.fc.bias, ALPHA)

    def forward(self, x):
        y = torch.tanh(self.fc(x))
        alin = float(torch.abs(y).mean().clamp(0.0, 1.0))
        self.septro.atualizar(alin)
        return y

    def coerencia_ativacoes(self, v):
        """Coerência φ das ativações: 1 - H_normalizada."""
        a = np.abs(v) + 1e-10
        a = a / a.sum()
        H = -np.sum(a * np.log(a))
        H_max = np.log(max(len(a), 2))
        return float(1.0 - H / H_max)


# ── Rede Alpha-Phi Estágio 0 + I ──────────────────────────────────────────────

class RedeAP(nn.Module):
    """
    Estágio 0: campo sugerido em proporção φ (DIMS Fibonacci).
    Estágio I: α como âncora entrópica em cada camada.
    Sem treinamento — campo puro na inicialização.

    Ponto de extensão natural:
      Estágio II → Sépstro como invariante projetado (constraint layer)
      Estágio III → Scanner de ativações (STTG sobre self.ativacoes)
    """

    def __init__(self):
        super().__init__()
        self.camadas = nn.ModuleList([
            CamadaAP(DIMS[i], DIMS[i + 1], nivel=i)
            for i in range(len(DIMS) - 1)
        ])
        self.ativacoes = []   # preenchido a cada forward — entrada do STTG

    def forward(self, x):
        self.ativacoes = [x.detach().cpu().numpy().flatten()]
        for c in self.camadas:
            x = c(x)
            self.ativacoes.append(x.detach().cpu().numpy().flatten())
        return x

    def septro_tabela(self):
        rows = []
        for i, c in enumerate(self.camadas):
            rows.append({
                'camada': i,
                'd_in':   DIMS[i],
                'd_out':  DIMS[i + 1],
                'coh':    c.septro.coh,
                'entr':   c.septro.entr,
                'soma':   c.septro.coh + c.septro.entr,
                'coh_ativ': c.coerencia_ativacoes(self.ativacoes[i + 1])
            })
        return rows

    def razoes_phi(self):
        return [round(DIMS[i] / DIMS[i + 1], 4) for i in range(len(DIMS) - 1)]


# ── Sinal EcoBIP de entrada ────────────────────────────────────────────────────

def sinal_ecobip(n_amostras=1024):
    t = np.linspace(0, 1, n_amostras, endpoint=False)
    quadrada = np.sign(np.sin(2 * np.pi * 880 * t))
    fm_phi   = np.sin(2 * np.pi * 880 * t + PHI * np.sin(2 * np.pi * 220 * t))
    return (1 - ALPHA_OP) * quadrada + ALPHA_OP * fm_phi

def comprime(sig, n):
    """Reduz sinal para n pontos por amostragem uniforme."""
    idx = np.linspace(0, len(sig) - 1, n).astype(int)
    return sig[idx].astype(np.float32)


# ── Execução ───────────────────────────────────────────────────────────────────

torch.manual_seed(42)   # semente fixa — campo reproduzível
rede = RedeAP()

sinal_bruto = sinal_ecobip(1024)
entrada = torch.tensor(comprime(sinal_bruto, DIMS[0])).unsqueeze(0)

with torch.no_grad():
    _ = rede(entrada)

# ── Relatório no terminal ──────────────────────────────────────────────────────

print("=" * 60)
print("Alpha-Phi NN — Estágio 0 + I")
print("=" * 60)
print(f"  PHI      = {PHI:.7f}")
print(f"  ALPHA    = {ALPHA:.7f}")
print(f"  SEAL     = {SEAL:.7f}")
print(f"  THETA_R  = {THETA_R:.4f}°  (referência Grade R)")
print()
print(f"Arquitetura : {' → '.join(str(d) for d in DIMS)}")
print(f"Razões φ    : {rede.razoes_phi()}")
print()
print(f"{'C':>3}  {'d_in':>5}  {'d_out':>5}  {'Coh(Sépstro)':>14}  {'Entr':>10}  {'Σ':>6}  {'Coh(ativ)':>10}")
for r in rede.septro_tabela():
    print(f"{r['camada']:>3}  {r['d_in']:>5}  {r['d_out']:>5}  "
          f"{r['coh']:>14.6f}  {r['entr']:>10.6f}  {r['soma']:>6.4f}  {r['coh_ativ']:>10.4f}")
print()
print("Campo Alpha-Phi Estágio 0+I — pronto.")
print(f"self.ativacoes: {len(rede.ativacoes)} camadas  "
      f"(dims {[len(a) for a in rede.ativacoes]})")
print()
print("Próximo:")
print("  Estágio II  → Sépstro projetado como constraint layer")
print("  Estágio III → STTG sobre rede.ativacoes  (scanner de ativações)")


# ── Visualização ───────────────────────────────────────────────────────────────

CORES = ['#FFD700', '#FFA040', '#FF5733', '#C0392B', '#8E44AD', '#2471A3', '#1ABC9C']

fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        'Sinal EcoBIP de Entrada (1024 amostras)',
        'Ativações por Camada — Campo Alpha-Phi (sem treinamento)',
        'Sépstro por Camada  ·  Coh + Entr = 1.0'
    ),
    vertical_spacing=0.1
)

# Painel 1 — sinal bruto
fig.add_trace(go.Scatter(y=sinal_bruto[:256], mode='lines',
                         line=dict(color='#FFD700', width=1), name='EcoBIP'),
              row=1, col=1)

# Painel 2 — ativações por camada
for i, (a, cor) in enumerate(zip(rede.ativacoes, CORES)):
    nome = f'Entrada' if i == 0 else f'Camada {i-1}  dim={len(a)}'
    fig.add_trace(go.Scatter(y=a, mode='lines+markers', name=nome,
                             line=dict(color=cor, width=1.5),
                             marker=dict(size=3)),
                  row=2, col=1)

# Painel 3 — Sépstro
tab   = rede.septro_tabela()
xs    = [r['camada'] for r in tab]
cohs  = [r['coh']  for r in tab]
entrs = [r['entr'] for r in tab]
covs  = [r['coh_ativ'] for r in tab]

fig.add_trace(go.Scatter(x=xs, y=cohs,  mode='lines+markers', name='Coh (Sépstro)',
                         line=dict(color='#FFD700', width=2)), row=3, col=1)
fig.add_trace(go.Scatter(x=xs, y=entrs, mode='lines+markers', name='Entr (Sépstro)',
                         line=dict(color='#4A90D9', width=2)), row=3, col=1)
fig.add_trace(go.Scatter(x=xs, y=covs,  mode='lines+markers', name='Coh (ativações)',
                         line=dict(color='#1ABC9C', width=2, dash='dot')), row=3, col=1)
fig.add_hline(y=ALPHA, line_dash='dash', line_color='rgba(255,255,255,0.4)',
              annotation_text=f'α = {ALPHA:.4f}', row=3, col=1)

fig.update_layout(
    template='plotly_dark',
    height=900,
    title=dict(text='Alpha-Phi NN — Estágio 0+I · Sugestão de Campo + Ancoramento α',
               font=dict(size=16)),
    showlegend=True
)
fig.update_xaxes(title_text='Amostra',  row=1, col=1)
fig.update_xaxes(title_text='Neurônio', row=2, col=1)
fig.update_xaxes(title_text='Camada',   row=3, col=1)

fig.show()
