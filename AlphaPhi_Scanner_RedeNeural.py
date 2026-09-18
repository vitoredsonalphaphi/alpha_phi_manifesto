# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Scanner_RedeNeural.py
Vitor Edson Delavi · Florianópolis · 2026

SCANNER TOPOGRÁFICO — Campo Puro Alpha-Phi na Rede Neural

Pergunta: a inserção do campo φ na rede neural gera Grade R
(θ_R = 63.43°) nas ativações antes de qualquer treinamento?

Método:
  1. Instanciar RedeAP (Estágio 0+I) sem treino — campo puro
  2. Passar N entradas GAUSSIANAS NEUTRAS — sem estrutura importada
     (Gaussiana normalizada: máxima entropia, zero viés espectral)
  3. Coletar ativações de todas as camadas
  4. Interpolar todas para mesma grade (55 neurônios)
  5. Média das ativações → topografia estrutural do CAMPO, não dos dados
  6. Scanner 3D: X = neurônio (normalizado), Y = profundidade r, Z = |ativação|
  7. Medir o ângulo da crista dominante → comparar com θ_R

Princípio:
  Entradas EcoBIP importariam estrutura de áudio (ALPHA_OP=1/3, FM-φ, 880Hz)
  para o campo que queremos observar como virgem. Gaussiana neutra é substrato
  sem memória — o que aparece na topografia é APENAS o campo Alpha-Phi.

Se Grade R emerge: o campo φ organiza o espaço de representação
antes da tarefa e antes dos dados — lógica estrutural, não busca.
"""

import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from scipy.interpolate import interp1d

# ── Constantes Alpha-Phi (apenas NN — sem ALPHA_OP que é domínio EcoBIP) ───────
PHI     = (1 + np.sqrt(5)) / 2
ALPHA   = 1 / 137.035999
SEAL    = 1 / PHI
THETA_R = np.arctan(2)           # 63.43° em radianos

DIMS    = [55, 34, 21, 13, 8, 5, 3]
N_CAM   = len(DIMS) - 1           # 6 camadas

COLORSCALE = [
    [0.00, 'rgb(0,0,20)'],
    [0.15, 'rgb(5,20,80)'],
    [0.40, 'rgb(80,50,0)'],
    [0.65, 'rgb(180,120,0)'],
    [0.85, 'rgb(230,190,20)'],
    [1.00, 'rgb(255,248,100)'],
]

# ── Rede Alpha-Phi — campo estrutural sem treinamento ─────────────────────────

class CamadaAP(nn.Module):
    def __init__(self, d_in, d_out, nivel):
        super().__init__()
        self.fc    = nn.Linear(d_in, d_out)
        self.nivel = nivel
        self._init_campo()

    def _init_campo(self):
        escala = PHI ** (-self.nivel)               # escala decai por φ por nível
        nn.init.normal_(self.fc.weight, mean=0.0, std=escala)
        nn.init.constant_(self.fc.bias, ALPHA)      # α como âncora estrutural

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


# ── Entradas Gaussianas Neutras — substrato virgem ────────────────────────────
# Gaussiana normalizada: máxima entropia espectral, sem estrutura prévia.
# EcoBIP, FM-φ, ruído colorido, sinais de áudio → EXCLUÍDOS.
# O que aparecer na topografia emerge SOMENTE do campo Alpha-Phi.

N_ENTRADAS = 200
torch.manual_seed(137)    # seed α — reprodutibilidade canônica
rng = np.random.default_rng(137)

entradas_raw = rng.standard_normal((N_ENTRADAS, DIMS[0])).astype(np.float32)
# Normalização por amostra: média=0, std=1
entradas_raw -= entradas_raw.mean(axis=1, keepdims=True)
entradas_raw /= (entradas_raw.std(axis=1, keepdims=True) + 1e-8)

print("Scanner Topográfico — Campo Puro Alpha-Phi")
print(f"Arquitetura : {' → '.join(str(d) for d in DIMS)}")
print(f"Entradas    : {N_ENTRADAS} × Gaussiana N(0,1) normalizada por amostra")
print(f"Seed        : 137 (α canônico)")
print(f"Sem EcoBIP, sem FM-φ, sem estrutura de áudio — campo virgem.\n")

# ── Coleta de ativações ────────────────────────────────────────────────────────
rede   = RedeAP()
N_GRID = 55
n_layers = len(DIMS)   # 7: entrada + 6 camadas

mapa_ativ = np.zeros((n_layers, N_GRID))

for i, entry in enumerate(entradas_raw):
    inp = torch.tensor(entry).unsqueeze(0)
    with torch.no_grad():
        _ = rede(inp)

    for lv, ativ in enumerate(rede.ativacoes):
        a_abs  = np.abs(ativ)
        x_orig = np.linspace(0, 1, len(a_abs))
        x_grid = np.linspace(0, 1, N_GRID)
        a_interp = interp1d(x_orig, a_abs, kind='linear')(x_grid)
        mapa_ativ[lv] += a_interp

mapa_ativ /= N_ENTRADAS
print(f"Mapa de ativações: {mapa_ativ.shape}  (camadas × neurônios normalizados)")
print(f"  min={mapa_ativ.min():.6f}  max={mapa_ativ.max():.6f}\n")

# ── Normalização logarítmica para visualização ────────────────────────────────
mapa_norm = np.log1p(mapa_ativ * 100)

# ── Análise da crista dominante ───────────────────────────────────────────────
picos_x = np.array([np.argmax(mapa_norm[lv]) / (N_GRID - 1) for lv in range(n_layers)])
picos_r = np.linspace(0, 1, n_layers)

if picos_x.std() > 1e-6:
    coef = np.polyfit(picos_x, picos_r, 1)
    angulo_crista = np.degrees(np.arctan(coef[0]))
else:
    angulo_crista = 90.0

print("Crista dominante (pico de ativação por camada):")
print(f"  Picos normalizados: {picos_x.round(3)}")
print(f"  Ângulo da crista  : {angulo_crista:.2f}°")
print(f"  θ_R referência    : {np.degrees(THETA_R):.2f}°")
delta = abs(angulo_crista - np.degrees(THETA_R))
print(f"  Δ do θ_R          : {delta:.2f}°", end="  ")
if delta < 5.0:
    print("→ Grade R PRESENTE (Δ < 5°)")
elif delta < 15.0:
    print("→ Proximidade parcial com Grade R")
else:
    print("→ Grade R não detectada nesta configuração")
print()

# ── Grade R no espaço normalizado ────────────────────────────────────────────
x_c, r_c, dx = 0.5, 0.5, 0.3
dy   = dx * np.tan(THETA_R)
xgr  = [x_c - dx, x_c + dx]
rgr  = [np.clip(r_c - dy, 0, 1), np.clip(r_c + dy, 0, 1)]
z_gr = [mapa_norm.max() * 1.1] * 2

# ── Superfície 3D ─────────────────────────────────────────────────────────────
X_grid, R_grid = np.meshgrid(np.linspace(0, 1, N_GRID), np.linspace(0, 1, n_layers))

fig = go.Figure()

fig.add_trace(go.Surface(
    x=X_grid, y=R_grid, z=mapa_norm,
    colorscale=COLORSCALE,
    showscale=True,
    colorbar=dict(title='log|ativ|', thickness=14, len=0.65),
    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.3),
    name='Campo AP puro'
))

fig.add_trace(go.Scatter3d(
    x=xgr, y=rgr, z=z_gr,
    mode='lines', line=dict(color='lime', width=7),
    name=f'Grade R θ_R={np.degrees(THETA_R):.2f}°'
))

z_picos = [mapa_norm[lv, int(px*(N_GRID-1))] for lv, px in enumerate(picos_x)]
fig.add_trace(go.Scatter3d(
    x=picos_x, y=picos_r, z=z_picos,
    mode='lines+markers',
    line=dict(color='cyan', width=4),
    marker=dict(size=5, color='cyan'),
    name=f'Crista real {angulo_crista:.1f}°'
))

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
        text=(f'Campo Puro Alpha-Phi — Rede Neural sem Treino · '
              f'Entradas Gaussianas · '
              f'θ_R={np.degrees(THETA_R):.2f}° · '
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

# ── Mapa de calor 2D ──────────────────────────────────────────────────────────
labels_y = ['Entrada\n(55)'] + [f'C{i}\n({DIMS[i+1]})' for i in range(N_CAM)]

fig2 = go.Figure()

fig2.add_trace(go.Heatmap(
    x=np.linspace(0, 1, N_GRID),
    y=list(range(n_layers)),
    z=mapa_norm,
    colorscale=COLORSCALE,
    colorbar=dict(title='log|ativ|'),
))

x2_gr = [x_c - dx, x_c + dx]
y2_gr = [r_c*(n_layers-1) - dy*(n_layers-1),
         r_c*(n_layers-1) + dy*(n_layers-1)]
fig2.add_trace(go.Scatter(
    x=x2_gr, y=y2_gr,
    mode='lines', line=dict(color='lime', width=2),
    name=f'Grade R {np.degrees(THETA_R):.1f}°'
))
fig2.add_trace(go.Scatter(
    x=picos_x, y=list(range(n_layers)),
    mode='lines+markers', line=dict(color='cyan', width=2),
    marker=dict(size=6), name=f'Crista {angulo_crista:.1f}°'
))

fig2.update_layout(
    title='Campo AP — Mapa 2D · Entradas Gaussianas Neutras',
    xaxis_title='Neurônio (normalizado 0–1)',
    yaxis_title='Camada',
    yaxis=dict(tickmode='array', tickvals=list(range(n_layers)), ticktext=labels_y),
    template='plotly_dark', height=420
)
fig2.show()

# ── Trajetória de coerência por camada ────────────────────────────────────────
def coh_de_vetor(v):
    a = np.abs(v) + 1e-10
    a = a / a.sum()
    H = -np.sum(a * np.log(a))
    return float(1.0 - H / np.log(max(len(a), 2)))

cohs_medias = [coh_de_vetor(mapa_ativ[lv]) for lv in range(n_layers)]
r_vals  = np.linspace(0, 1, n_layers)
targets = [ALPHA + r*(1.0 - 2.0*ALPHA) for r in r_vals]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=r_vals, y=targets, mode='lines',
                           name='Trajetória canônica (Sépstro)',
                           line=dict(color='white', width=2, dash='dash')))
fig3.add_trace(go.Scatter(x=r_vals, y=cohs_medias, mode='lines+markers',
                           name='Coh real — campo AP virgem',
                           line=dict(color='#FFD700', width=2),
                           marker=dict(size=8)))
fig3.add_hline(y=ALPHA,     line_dash='dot', line_color='rgba(255,255,255,0.3)',
               annotation_text=f'α={ALPHA:.4f}')
fig3.add_hline(y=1.0-ALPHA, line_dash='dot', line_color='rgba(255,255,255,0.3)',
               annotation_text=f'1−α={1-ALPHA:.4f}')

fig3.update_layout(
    title='Trajetória de Coerência — Campo AP Virgem vs. Trajetória Canônica do Sépstro',
    xaxis_title='r  (0 = α-centro  →  1 = φ-superfície)',
    yaxis_title='Coerência',
    template='plotly_dark', height=380
)
fig3.show()

# ── Relatório final ────────────────────────────────────────────────────────────
print("=" * 65)
print("SCANNER — CAMPO PURO ALPHA-PHI (sem treino, sem EcoBIP)")
print("=" * 65)
print(f"  Entradas        : {N_ENTRADAS} × Gaussiana N(0,1) normalizada")
print(f"  Arquitetura     : {' → '.join(str(d) for d in DIMS)}")
print(f"  Bias (âncora α) : {ALPHA:.8f}")
print(f"  Escala peso C_i : φ^(-i)  = φ^0, φ^-1, φ^-2, φ^-3, φ^-4, φ^-5")
print(f"  Grade R ref     : {np.degrees(THETA_R):.4f}°")
print(f"  Ângulo crista   : {angulo_crista:.4f}°")
print(f"  Δ               : {delta:.4f}°  →  ", end="")
if delta < 5.0:
    print("Grade R PRESENTE")
elif delta < 15.0:
    print("Proximidade parcial com Grade R")
else:
    print("Grade R não detectada")
print()
print("Coerência por camada vs. trajetória canônica (Sépstro):")
for lv, (coh, tgt) in enumerate(zip(cohs_medias, targets)):
    label = 'Entrada' if lv == 0 else f'C{lv-1} ({DIMS[lv]}n)'
    print(f"  {label:>16} r={r_vals[lv]:.3f}  Coh={coh:.6f}  target={tgt:.6f}  "
          f"Δ={coh-tgt:+.6f}")
