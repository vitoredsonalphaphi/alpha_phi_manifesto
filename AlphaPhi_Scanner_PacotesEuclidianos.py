# ============================================================
# SCANNER DOS PACOTES EUCLIDIANOS — Distribuição de Fase em φ²
# Sala vazia e em silêncio · Sem cascata · Sem sinal · Sem rede
# Sete pacotes ℝⁿ Fibonacci · Disco de fase em φ² = 2.618
# Confirmação: espaço euclidiano é neutro em φ² (PLV ≈ ruído)
# Vitor Edson Delavi · Florianópolis · 20 set 2026
# Sessão Good Morning · Entrada 277
# ============================================================

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PHI    = (1 + np.sqrt(5)) / 2
PHI2   = PHI**2               # 2.6180 — bin calibrado
DIMS   = [55, 34, 21, 13, 8, 5, 3]
N_SAMP = 5000
rng    = np.random.default_rng(137)
n_lv   = len(DIMS)

# ── DFT exato em k não-inteiro ────────────────────────────────
def dft_at_k(X, k):
    n      = np.arange(X.shape[1])
    kernel = np.exp(-2j * np.pi * k * n / X.shape[1])
    return X @ kernel

# ── Scanner: DFT(φ²) de vetores PUROS por pacote ────────────
print(f"SCANNER DOS PACOTES EUCLIDIANOS")
print(f"φ² = {PHI2:.6f} · {N_SAMP} vetores por pacote\n")

pacotes = []
for dim in DIMS:
    X  = rng.standard_normal((N_SAMP, dim))
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Z  = dft_at_k(X, PHI2)
    pacotes.append(Z)
    plv   = float(np.abs(np.mean(np.exp(1j * np.angle(Z)))))
    r     = float(np.abs(Z).mean())
    r_teo = float(np.sqrt(dim / 2))
    status = 'uniforme' if plv < 1.5 / np.sqrt(N_SAMP) else 'ESTRUTURADA'
    print(f"  {dim:2d}n  |Z|={r:.4f} (teórico √{dim}/2={r_teo:.4f})  "
          f"PLV={plv:.5f}  [{status}]")

# ── CORES por pacote ─────────────────────────────────────────
CORES = [
    '#4a8cff', '#3ab8e0', '#30c890', '#8ccc3c',
    '#e0b42c', '#ff7820', '#e03c78'
]
ALTURA = 2.0  # separação vertical entre discos

# ── PLOT 1: Discos 3D empilhados — sala vazia ────────────────
fig1 = go.Figure()

for lv, (dim, Z) in enumerate(zip(DIMS, pacotes)):
    h   = lv * ALTURA
    Re  = Z.real
    Im  = Z.imag
    cor = CORES[lv]
    r_m = float(np.abs(Z).mean())

    # Nuvem de pontos de fase — o disco
    fig1.add_trace(go.Scatter3d(
        x=Re, y=Im, z=np.full(N_SAMP, h),
        mode='markers',
        marker=dict(size=1.2, color=cor, opacity=0.45),
        name=f'{dim}n', showlegend=True))

    # Círculo de contorno (raio médio observado)
    theta = np.linspace(0, 2 * np.pi, 300)
    fig1.add_trace(go.Scatter3d(
        x=r_m * np.cos(theta), y=r_m * np.sin(theta),
        z=np.full(300, h),
        mode='lines', line=dict(color=cor, width=3),
        showlegend=False))

    # Círculo teórico (raio = √dim/2, tracejado)
    r_teo = np.sqrt(dim / 2)
    fig1.add_trace(go.Scatter3d(
        x=r_teo * np.cos(theta), y=r_teo * np.sin(theta),
        z=np.full(300, h),
        mode='lines',
        line=dict(color='rgba(255,255,255,0.20)', width=1.5, dash='dot'),
        showlegend=False))

    # Eixos do disco (cruzeta)
    for dx, dy in [(r_m, 0), (-r_m, 0), (0, r_m), (0, -r_m)]:
        fig1.add_trace(go.Scatter3d(
            x=[0, dx * 0.88], y=[0, dy * 0.88], z=[h, h],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.10)', width=1),
            showlegend=False))

    # Label
    fig1.add_trace(go.Scatter3d(
        x=[r_m * 1.18], y=[0], z=[h + 0.28],
        mode='text', text=[f'{dim}n  |Z|={r_m:.2f}'],
        textfont=dict(size=10, color='rgba(200,220,255,0.8)'),
        showlegend=False))

# Coluna central de orientação
zs = [lv * ALTURA for lv in range(n_lv)]
fig1.add_trace(go.Scatter3d(
    x=[0] * n_lv, y=[0] * n_lv, z=zs,
    mode='lines', line=dict(color='rgba(255,255,255,0.08)', width=1),
    showlegend=False))

fig1.update_layout(
    title=dict(
        text=(f'Pacotes Euclidianos · Fase em φ² = {PHI2:.4f} · Sala Vazia<br>'
              f'<sup>Sem cascata · Sem sinal · Disco uniforme = espaço neutro · '
              f'Tracejado branco = raio teórico √(dim/2)</sup>'),
        font=dict(size=12)),
    template='plotly_dark', height=740,
    scene=dict(
        xaxis=dict(title='Re · DFT(φ²)',
                   gridcolor='rgba(80,120,220,0.08)',
                   color='rgba(150,190,255,0.6)'),
        yaxis=dict(title='Im · DFT(φ²)',
                   gridcolor='rgba(80,120,220,0.08)',
                   color='rgba(150,190,255,0.6)'),
        zaxis=dict(title='Pacote (55n → 3n)',
                   tickvals=[k * ALTURA for k in range(n_lv)],
                   ticktext=[f'{d}n' for d in DIMS],
                   gridcolor='rgba(80,120,220,0.08)',
                   color='rgba(150,190,255,0.6)'),
        camera=dict(eye=dict(x=1.4, y=-1.7, z=0.7)),
        bgcolor='rgb(2,4,18)',
        aspectmode='manual',
        aspectratio=dict(x=1.5, y=1.5, z=2.8)),
    legend=dict(x=0.01, y=0.95, font=dict(size=11),
                bgcolor='rgba(0,0,0,0.5)'),
    margin=dict(l=0, r=0, t=80, b=0))
fig1.show()

# ── PLOT 2: Rosa de fase — histograma polar por pacote ───────
fig2 = make_subplots(
    rows=1, cols=n_lv,
    subplot_titles=[f'{d}n' for d in DIMS],
    specs=[[{'type': 'polar'}] * n_lv])

for lv, (dim, Z) in enumerate(zip(DIMS, pacotes)):
    fases = np.degrees(np.angle(Z)) % 360
    counts, edges = np.histogram(fases, bins=36, range=(0, 360))
    theta_deg = (edges[:-1] + edges[1:]) / 2
    fig2.add_trace(go.Barpolar(
        r=counts, theta=theta_deg,
        width=10, marker_color=CORES[lv],
        marker_line_width=0, opacity=0.85,
        name=f'{dim}n', showlegend=False),
        row=1, col=lv + 1)

fig2.update_layout(
    title=dict(
        text=(f'Rosa de Fase φ² por Pacote · Uniformidade = Espaço Neutro<br>'
              f'<sup>Rosa circular → nenhuma fase preferida → vazio confirmado</sup>'),
        font=dict(size=12)),
    template='plotly_dark', height=380,
    margin=dict(l=10, r=10, t=80, b=10))
fig2.update_polars(radialaxis_showticklabels=False,
                   angularaxis_color='rgba(150,190,255,0.4)',
                   bgcolor='rgb(4,6,25)')
fig2.show()

# ── PLOT 3: Raio por pacote — identidade dimensional ─────────
raios_obs = [float(np.abs(Z).mean()) for Z in pacotes]
raios_teo = [float(np.sqrt(d / 2)) for d in DIMS]
plvs      = [float(np.abs(np.mean(np.exp(1j * np.angle(Z))))) for Z in pacotes]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=[f'{d}n' for d in DIMS], y=raios_obs,
    mode='lines+markers', name='|Z| observado',
    line=dict(color='rgba(80,180,255,0.9)', width=3),
    marker=dict(size=9, symbol='diamond')))
fig3.add_trace(go.Scatter(
    x=[f'{d}n' for d in DIMS], y=raios_teo,
    mode='lines', name='√(dim/2) teórico',
    line=dict(color='rgba(255,170,0,0.7)', width=2, dash='dot')))
fig3.add_trace(go.Bar(
    x=[f'{d}n' for d in DIMS], y=plvs,
    name='PLV(φ²)', opacity=0.5,
    marker_color='rgba(60,200,120,0.6)',
    yaxis='y2'))
fig3.add_hline(
    y=1 / np.sqrt(N_SAMP),
    line_color='rgba(255,80,80,0.5)', line_dash='dash',
    annotation_text=f'Piso ruído 1/√N = {1/np.sqrt(N_SAMP):.4f}',
    annotation_font_color='rgba(255,100,100,0.8)')
fig3.update_layout(
    title=(f'Raio e PLV dos Pacotes em φ² · Identidade Dimensional<br>'
           f'<sup>|Z| ∝ √(dim/2) · PLV ≈ ruído → todos os pacotes neutros em φ²</sup>'),
    xaxis=dict(title='Pacote Euclidiano', color='#aaa'),
    yaxis=dict(title='|DFT(φ²)|', color='#aaa'),
    yaxis2=dict(title='PLV(φ²)', overlaying='y', side='right',
                color='rgba(60,200,120,0.8)', range=[0, 0.08]),
    template='plotly_dark', height=400,
    paper_bgcolor='#0a0a12',
    legend=dict(x=0.02, y=0.98))
fig3.show()
