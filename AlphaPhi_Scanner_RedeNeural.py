# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Scanner_RedeNeural.py
Vitor Edson Delavi · Florianópolis · 2026

SCANNER TOPOGRÁFICO — Rede Neural Alpha-Phi sem Treinamento

Pergunta: a inserção do campo φ na rede neural gera Grade R
(θ_R = 63.43°) nas ativações antes de qualquer treinamento?

Método:
  1. Instanciar RedeAP (Estágio 0+I) sem treino — campo puro
  2. Passar N entradas EcoBIP com variações de fase
  3. Coletar ativações de todas as camadas
  4. Interpolar todas para mesma grade (55 neurônios)
  5. Médias das ativações → topografia estrutural do campo
  6. Scanner 3D: X = neurônio (normalizado), Y = profundidade r, Z = |ativação|
  7. Medir o ângulo da crista dominante → comparar com θ_R

Se Grade R emerge: o campo φ organiza o espaço de representação
antes da tarefa existir — lógica estrutural, não busca.
"""

import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

# ── Constantes ─────────────────────────────────────────────────────────────────
PHI      = (1 + np.sqrt(5)) / 2
ALPHA    = 1 / 137.035999
ALPHA_OP = 1 / 3
SEAL     = 1 / PHI
THETA_R  = np.arctan(2)           # 63.43° em radianos
SR       = 44100
BASE     = 880

DIMS     = [55, 34, 21, 13, 8, 5, 3]
N_CAM    = len(DIMS) - 1           # 6 camadas

COLORSCALE = [
    [0.00, 'rgb(0,0,20)'],
    [0.15, 'rgb(5,20,80)'],
    [0.40, 'rgb(80,50,0)'],
    [0.65, 'rgb(180,120,0)'],
    [0.85, 'rgb(230,190,20)'],
    [1.00, 'rgb(255,248,100)'],
]

# ── Rede Alpha-Phi — Estágio 0+I (autônoma, sem imports externos) ──────────────

class CamadaAP(nn.Module):
    def __init__(self, d_in, d_out, nivel):
        super().__init__()
        self.fc    = nn.Linear(d_in, d_out)
        self.nivel = nivel
        self._init_campo()

    def _init_campo(self):
        escala = PHI ** (-self.nivel)
        nn.init.normal_(self.fc.weight, mean=0.0, std=escala)
        nn.init.constant_(self.fc.bias, ALPHA)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class RedeAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.camadas   = nn.ModuleList([
            CamadaAP(DIMS[i], DIMS[i+1], nivel=i)
            for i in range(N_CAM)
        ])
        self.ativacoes = []

    def forward(self, x):
        self.ativacoes = [x.detach().cpu().numpy().flatten()]
        for c in self.camadas:
            x = c(x)
            self.ativacoes.append(x.detach().cpu().numpy().flatten())
        return x


# ── Gerador de entradas EcoBIP com variações de fase ──────────────────────────

def ecobip_variado(n=1024, delta_fase=0.0):
    t  = np.linspace(0, 1, n, endpoint=False)
    qd = np.sign(np.sin(2*np.pi*BASE*t + delta_fase))
    fm = np.sin(2*np.pi*BASE*t + delta_fase + PHI*np.sin(2*np.pi*(BASE/4)*t))
    s  = (1-ALPHA_OP)*qd + ALPHA_OP*fm
    return s / (np.max(np.abs(s)) + 1e-8)

def comprimir(sig, n):
    idx = np.linspace(0, len(sig)-1, n).astype(int)
    return sig[idx].astype(np.float32)


# ── Coleta de ativações sobre N entradas ──────────────────────────────────────

torch.manual_seed(42)
rede = RedeAP()

N_ENTRADAS = 120
fases = np.linspace(0, 2*np.pi, N_ENTRADAS, endpoint=False)

print(f"Rede Alpha-Phi — campo sem treinamento")
print(f"Arquitetura: {' → '.join(str(d) for d in DIMS)}")
print(f"Passando {N_ENTRADAS} entradas EcoBIP (variações de fase 0–2π)...")

# Acumula |ativações| por camada
# Cada camada tem dims[i+1] neurônios — interpolamos para grade de 55
N_GRID   = 55
n_layers = len(DIMS)   # 7: entrada + 6 camadas

mapa_ativ = np.zeros((n_layers, N_GRID))  # [layer, neuron_normalizado]

for fase in fases:
    sig   = ecobip_variado(1024, delta_fase=fase)
    inp   = torch.tensor(comprimir(sig, DIMS[0])).unsqueeze(0)
    with torch.no_grad():
        _ = rede(inp)

    for lv, ativ in enumerate(rede.ativacoes):
        a_abs = np.abs(ativ)
        # Interpolação para N_GRID pontos
        x_orig  = np.linspace(0, 1, len(a_abs))
        x_grid  = np.linspace(0, 1, N_GRID)
        a_interp = interp1d(x_orig, a_abs, kind='linear')(x_grid)
        mapa_ativ[lv] += a_interp

mapa_ativ /= N_ENTRADAS   # média

print(f"Mapa de ativações: {mapa_ativ.shape}  (layers × neurônios normalizados)")
print(f"  min={mapa_ativ.min():.6f}  max={mapa_ativ.max():.6f}")

# ── Normalização para visualização ────────────────────────────────────────────
mapa_norm = np.log1p(mapa_ativ * 100)

# ── Grade R — posicionada no mapa de ativações ─────────────────────────────────
# Espaço normalizado: X ∈ [0,1] (neurônio), Y ∈ [0,1] (profundidade r)
# θ_R = arctan(2) → tan(θ_R) = 2
# Relação no espaço normalizado: Δy/Δx = tan(θ_R) = 2
# Linha passa pelo centro do mapa

x_n  = np.linspace(0, 1, N_GRID)
r_n  = np.linspace(0, 1, n_layers)

x_c  = 0.5
r_c  = 0.5
dx   = 0.3
dy   = dx * np.tan(THETA_R)   # ≈ 0.6

xgr  = [x_c - dx, x_c + dx]
rgr  = [r_c - dy, r_c + dy]
rgr  = [np.clip(v, 0, 1) for v in rgr]

# Em índices reais
xgr_idx = [v * (N_GRID-1)   for v in xgr]
rgr_idx = [v * (n_layers-1)  for v in rgr]

# Elevação da Grade R: levemente acima da superfície
z_gr = [mapa_norm.max() * 1.1] * 2

# ── Crista dominante — análise do ângulo real ──────────────────────────────────
# Para cada layer, encontra o neurônio de maior ativação média
picos_x = []
picos_r = []
for lv in range(n_layers):
    pk = np.argmax(mapa_norm[lv])
    picos_x.append(pk / (N_GRID - 1))        # normalizado 0-1
    picos_r.append(lv / (n_layers - 1))       # normalizado 0-1

# Regressão linear sobre os picos → ângulo da crista
picos_x = np.array(picos_x)
picos_r = np.array(picos_r)
if picos_x.std() > 1e-6:
    coef = np.polyfit(picos_x, picos_r, 1)
    angulo_crista = np.degrees(np.arctan(coef[0]))
else:
    angulo_crista = 90.0

print(f"\nAnálise da crista dominante:")
print(f"  Picos por layer (normalizado): {picos_x.round(3)}")
print(f"  Ângulo da crista: {angulo_crista:.2f}°")
print(f"  θ_R referência:  {np.degrees(THETA_R):.2f}°")
print(f"  Δ do θ_R:        {abs(angulo_crista - np.degrees(THETA_R)):.2f}°")

# ── Grade de display ──────────────────────────────────────────────────────────
X_grid, R_grid = np.meshgrid(
    np.linspace(0, 1, N_GRID),
    np.linspace(0, 1, n_layers)
)

# ── Figura principal — superfície 3D ──────────────────────────────────────────
fig = go.Figure()

# Superfície das ativações
fig.add_trace(go.Surface(
    x=X_grid, y=R_grid, z=mapa_norm,
    colorscale=COLORSCALE,
    showscale=True,
    colorbar=dict(title='log|ativ|', thickness=14, len=0.65),
    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.3),
    name='Campo AP'
))

# Grade R
fig.add_trace(go.Scatter3d(
    x=xgr, y=rgr, z=z_gr,
    mode='lines',
    line=dict(color='lime', width=7),
    name=f'Grade R θ={np.degrees(THETA_R):.2f}°'
))

# Crista dominante — trajetória dos picos por layer
z_picos = [mapa_norm[lv, int(px*(N_GRID-1))] for lv, px in enumerate(picos_x)]
fig.add_trace(go.Scatter3d(
    x=picos_x, y=picos_r, z=z_picos,
    mode='lines+markers',
    line=dict(color='cyan', width=4),
    marker=dict(size=5, color='cyan'),
    name=f'Crista dominante {angulo_crista:.1f}°'
))

# Labels de profundidade
for lv, (d_in, d_out) in enumerate(zip(DIMS[:-1], DIMS[1:])):
    r_val = lv / (n_layers - 1)
    fig.add_trace(go.Scatter3d(
        x=[0.02], y=[r_val], z=[mapa_norm.max() * 1.15],
        mode='text',
        text=[f'C{lv} ({d_in}→{d_out})'],
        textfont=dict(size=9, color='rgba(255,255,255,0.6)'),
        showlegend=False
    ))

fig.update_layout(
    title=dict(
        text=(f'Scanner Topográfico — Rede Neural Alpha-Phi (sem treino) · '
              f'θ_R ref={np.degrees(THETA_R):.2f}° · '
              f'Crista={angulo_crista:.1f}°'),
        font=dict(size=13)
    ),
    template='plotly_dark',
    height=720,
    scene=dict(
        xaxis=dict(title='Neurônio (norm.)', showgrid=True,
                   gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(title='Profundidade r  (0=α → 1=φ)', showgrid=True,
                   gridcolor='rgba(255,255,255,0.06)'),
        zaxis=dict(title='log |ativação|', showgrid=True,
                   gridcolor='rgba(255,255,255,0.06)'),
        camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9)),
        bgcolor='rgb(4,4,14)',
        aspectmode='manual',
        aspectratio=dict(x=2.0, y=1.4, z=0.7),
    ),
    legend=dict(x=0.01, y=0.95, font=dict(size=11)),
    margin=dict(l=0, r=0, t=60, b=0)
)

fig.show()

# ── Mapa de calor 2D — visão plana da topografia ──────────────────────────────
labels_y = ['Entrada\n(55)'] + [f'C{i}\n({DIMS[i+1]})' for i in range(N_CAM)]

fig2 = go.Figure()

fig2.add_trace(go.Heatmap(
    x=np.linspace(0, 1, N_GRID),
    y=list(range(n_layers)),
    z=mapa_norm,
    colorscale=COLORSCALE,
    colorbar=dict(title='log|ativ|'),
))

# Grade R no 2D
x2_gr = [x_c - dx, x_c + dx]
y2_gr = [r_c*(n_layers-1) - dy*(n_layers-1),
         r_c*(n_layers-1) + dy*(n_layers-1)]
fig2.add_trace(go.Scatter(
    x=x2_gr, y=y2_gr,
    mode='lines', line=dict(color='lime', width=2),
    name=f'Grade R {np.degrees(THETA_R):.1f}°'
))

# Crista no 2D
fig2.add_trace(go.Scatter(
    x=picos_x, y=list(range(n_layers)),
    mode='lines+markers', line=dict(color='cyan', width=2),
    marker=dict(size=6), name=f'Crista {angulo_crista:.1f}°'
))

fig2.update_layout(
    title=f'Mapa de Ativações — Rede AP sem Treino · Grade R plana',
    xaxis_title='Neurônio (normalizado 0–1)',
    yaxis_title='Camada',
    yaxis=dict(tickmode='array',
               tickvals=list(range(n_layers)),
               ticktext=labels_y),
    template='plotly_dark',
    height=420
)

fig2.show()

# ── Coerência por camada ───────────────────────────────────────────────────────
def coh_de_vetor(v):
    a = np.abs(v) + 1e-10
    a = a / a.sum()
    H = -np.sum(a * np.log(a))
    H_max = np.log(max(len(a), 2))
    return float(1.0 - H / H_max)

cohs_medias = [coh_de_vetor(mapa_ativ[lv]) for lv in range(n_layers)]
r_vals = np.linspace(0, 1, n_layers)
targets = [ALPHA + r*(1.0 - 2.0*ALPHA) for r in r_vals]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=r_vals, y=targets, mode='lines',
                           name='Coh target (canônico)',
                           line=dict(color='white', width=2, dash='dash')))
fig3.add_trace(go.Scatter(x=r_vals, y=cohs_medias, mode='lines+markers',
                           name='Coh real (campo AP)',
                           line=dict(color='#FFD700', width=2),
                           marker=dict(size=8)))
fig3.add_hline(y=ALPHA,     line_dash='dot', line_color='rgba(255,255,255,0.3)',
               annotation_text=f'α={ALPHA:.4f}')
fig3.add_hline(y=1.0-ALPHA, line_dash='dot', line_color='rgba(255,255,255,0.3)',
               annotation_text=f'1−α={1-ALPHA:.4f}')

fig3.update_layout(
    title='Trajetória de Coerência — Campo AP (sem treino) vs. Trajetória Canônica',
    xaxis_title='r  (0 = α-centro  →  1 = φ-superfície)',
    yaxis_title='Coerência',
    template='plotly_dark', height=380
)
fig3.show()

# ── Relatório final ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SCANNER TOPOGRÁFICO — Rede Neural Alpha-Phi (sem treinamento)")
print("="*65)
print(f"  N entradas EcoBIP : {N_ENTRADAS}  (fases 0–2π)")
print(f"  Arquitetura       : {' → '.join(str(d) for d in DIMS)}")
print(f"  Grade R referência: {np.degrees(THETA_R):.4f}°")
print(f"  Ângulo da crista  : {angulo_crista:.4f}°")
delta = abs(angulo_crista - np.degrees(THETA_R))
print(f"  Δ                 : {delta:.4f}°", end="  ")
if delta < 5.0:
    print("→ Grade R PRESENTE (Δ < 5°)")
elif delta < 15.0:
    print("→ Proximidade parcial com Grade R")
else:
    print("→ Grade R não detectada nesta configuração")
print()
print("Coerência por camada (campo sem treinamento):")
for lv, (coh, tgt) in enumerate(zip(cohs_medias, targets)):
    label = 'Entrada' if lv == 0 else f'C{lv-1} ({DIMS[lv]})'
    print(f"  {label:>14} : r={r_vals[lv]:.3f}  Coh={coh:.6f}  target={tgt:.6f}  "
          f"Δ={coh-tgt:+.6f}")
