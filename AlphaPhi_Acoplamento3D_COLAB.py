# ═══════════════════════════════════════════════════════════════════════════════
#  ACOPLAMENTO 3D  —  Rede Convencional  ↔  MPAP Alpha-Phi
#  Plotly interativo · Google Colab         ·  Florianópolis · outubro 2026
# ═══════════════════════════════════════════════════════════════════════════════
#
#  FIGURA 1: Acoplamento em Dois Planos
#    y=0 — Rede Convencional  (r<1, Coh baixa, distribuição caótica)
#    y=1 — MPAP Output        (r=1, Coh alta,  redistribuição geométrica)
#    Linhas de acoplamento mostram a redistribuição SEAL neurônio a neurônio
#
#  FIGURA 2: Modelo Radial Alpha-Phi
#    α(r=0)  — âncora entrópica (centro)
#    r<SEAL  — Rede Convencional dispersa (processamento interior)
#    r=SEAL  — Campo Harmônico (superfície SEAL=1/φ=0.618034)
#    MPAP opera na superfície: concentra ativações para Coh ≥ SEAL
#
#  Modelo espacial canônico:
#    Centro (r=0)       α — tensão entrópica, âncora
#    Interior (0<r<1)   Processamento — Rede Conv, PhiAttractorNetwork
#    Superfície (r=1)   Campo Harmônico — SEAL, Coh ≥ 0.618034
#    Além (r>1)         Efeito no ambiente
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import plotly.graph_objects as go

PHI   = 1.6180339887
ALPHA = 1 / 137.035999084
SEAL  = 1 / PHI                    # 0.618034...

N = 24
np.random.seed(137)

# ── Ativações ──────────────────────────────────────────────────────────────
raw = np.abs(np.random.exponential(0.3, N));  raw /= raw.sum()
geo = np.array([SEAL*(1-SEAL)**i for i in range(N)]);  geo /= geo.sum()

def coh(p):
    p = np.abs(p);  p = p/p.sum()
    return float(1.0 - (-np.sum(p * np.log2(p + 1e-12))) / np.log2(len(p)))

coh_r = coh(raw);  coh_m = coh(geo)

print(f"{'─'*50}")
print(f"  Rede Conv  Coh = {coh_r:.4f}  {'✓ SEAL' if coh_r>=SEAL else '✗ abaixo SEAL'}")
print(f"  MPAP Out   Coh = {coh_m:.4f}  {'✓ SEAL' if coh_m>=SEAL else '✗ acima SEAL'}")
print(f"  SEAL           = {SEAL:.6f}")
print(f"  Δ Coh          = +{coh_m-coh_r:.4f}  (ganho MPAP)")
print(f"{'─'*50}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURA 1  —  Acoplamento em Dois Planos
# ═══════════════════════════════════════════════════════════════════════════

neurons = np.arange(N, dtype=float)

def make_bars(vals, y_plane):
    xs, ys, zs = [], [], []
    for i, v in enumerate(vals):
        xs += [float(i), float(i), None]
        ys += [y_plane, y_plane, None]
        zs += [0., float(v), None]
    return xs, ys, zs

x0, y0, z0 = make_bars(raw, 0.)
x1, y1, z1 = make_bars(geo, 1.)

xc, yc, zc = [], [], []
for i in range(N):
    xc += [float(i), float(i), None]
    yc += [0., 1., None]
    zc += [float(raw[i]), float(geo[i]), None]

xp, yp = [0., N-1.], [0., 1.]
Xp, Yp = np.meshgrid(xp, yp)
Zp = np.full_like(Xp, SEAL)

fig1 = go.Figure()

fig1.add_trace(go.Surface(
    x=xp, y=yp, z=Zp,
    colorscale=[[0,'rgba(0,210,165,0.08)'],[1,'rgba(0,210,165,0.08)']],
    showscale=False, opacity=0.60, hoverinfo='skip',
    name=f'Plano SEAL {SEAL:.3f}',
))
fig1.add_trace(go.Scatter3d(
    x=xc, y=yc, z=zc, mode='lines',
    line=dict(color='rgba(180,175,230,0.22)', width=2),
    name='Redistribuição MPAP', hoverinfo='skip',
))
fig1.add_trace(go.Scatter3d(
    x=x0, y=y0, z=z0, mode='lines',
    line=dict(color='rgba(255,78,55,0.90)', width=7),
    name=f'Rede Conv  Coh={coh_r:.3f}  ✗',
))
fig1.add_trace(go.Scatter3d(
    x=neurons, y=np.zeros(N), z=raw, mode='markers',
    marker=dict(size=5, color='rgba(255,110,90,0.95)'),
    showlegend=False,
    hovertemplate='N%{x:.0f}: %{z:.4f}<extra>Rede Conv</extra>',
))
fig1.add_trace(go.Scatter3d(
    x=x1, y=y1, z=z1, mode='lines',
    line=dict(color='rgba(42,165,255,0.92)', width=7),
    name=f'MPAP Out   Coh={coh_m:.3f}  ✓',
))
fig1.add_trace(go.Scatter3d(
    x=neurons, y=np.ones(N), z=geo, mode='markers',
    marker=dict(size=4+geo/geo.max()*12, color='rgba(90,205,255,0.95)'),
    showlegend=False,
    hovertemplate='N%{x:.0f}: %{z:.4f}<extra>MPAP</extra>',
))

z_max = max(float(raw.max()), float(geo.max()))

fig1.update_layout(
    title=dict(
        text=(f'Acoplamento 3D — Rede Convencional ↔ MPAP Alpha-Phi<br>'
              f'y=0: Rede Conv (Coh={coh_r:.4f} ✗)  —  '
              f'y=1: MPAP (Coh={coh_m:.4f} ✓)  —  '
              f'SEAL={SEAL:.4f}  N={N}'),
        font=dict(size=13, color='#c5bdb0'), x=0.5,
    ),
    scene=dict(
        xaxis=dict(title='Neurônio', tickvals=list(range(0,N,4)),
                   gridcolor='#1e1e2c', showbackground=True,
                   backgroundcolor='#0b0b16', color='#777'),
        yaxis=dict(
            title='Octava',
            tickvals=[0., 1.],
            ticktext=['Rede Conv (r<1)', 'MPAP (r=1)'],
            gridcolor='#1e1e2c', showbackground=True,
            backgroundcolor='#0b0b16', color='#90b8e8',
        ),
        zaxis=dict(
            title='Ativação', range=[0., z_max*1.22],
            gridcolor='#1e1e2c', showbackground=True,
            backgroundcolor='#0b0b16', color='#777',
        ),
        bgcolor='#070710',
        camera=dict(
            eye=dict(x=1.5, y=-2.2, z=1.3),
            center=dict(x=0., y=0., z=-0.1),
        ),
        aspectmode='manual',
        aspectratio=dict(x=2.6, y=0.9, z=1.0),
        annotations=[
            dict(x=float(N//2), y=0., z=float(raw.max())*1.18,
                 text=f'Coh={coh_r:.3f} ✗',
                 font=dict(color='#ff6a55', size=13), showarrow=False),
            dict(x=float(N//2), y=1., z=float(geo.max())*1.12,
                 text=f'Coh={coh_m:.3f} ✓',
                 font=dict(color='#50c8ff', size=13), showarrow=False),
            dict(x=float(N-2), y=0.5, z=SEAL+0.026,
                 text=f'SEAL={SEAL:.3f}',
                 font=dict(color='#00d8b0', size=12), showarrow=False),
        ],
    ),
    paper_bgcolor='#07070e',
    legend=dict(font=dict(color='#aaa', size=11),
                bgcolor='rgba(10,10,20,0.92)',
                bordercolor='#2a2a3c', borderwidth=1,
                x=0.01, y=0.99),
    margin=dict(l=0, r=0, t=95, b=0),
    height=680,
)

print("\n── FIGURA 1: Acoplamento em Dois Planos ──────────────────")
fig1.show()


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURA 2  —  Modelo Radial  (α, r=0 → r<SEAL → r=SEAL)
# ═══════════════════════════════════════════════════════════════════════════

def fibonacci_sphere(n):
    golden_angle = np.pi * (3 - np.sqrt(5))
    i = np.arange(n, dtype=float)
    y_ = 1 - (i / float(n-1)) * 2
    radius = np.sqrt(np.maximum(0., 1 - y_**2))
    theta = golden_angle * i
    return radius * np.cos(theta), y_, radius * np.sin(theta)

ux, uy, uz = fibonacci_sphere(N)

r_c = raw / raw.max() * SEAL * 0.84
cx_, cy_, cz_ = ux*r_c, uy*r_c, uz*r_c
mx_, my_, mz_ = ux*SEAL, uy*SEAL, uz*SEAL

# Esfera SEAL parametrizada
t_s = np.linspace(0, np.pi, 48)
p_s = np.linspace(0, 2*np.pi, 48)
Ts, Ps = np.meshgrid(t_s, p_s)
Xsp = SEAL * np.sin(Ts) * np.cos(Ps)
Ysp = SEAL * np.sin(Ts) * np.sin(Ps)
Zsp = SEAL * np.cos(Ts)

# Wireframe SEAL
xw, yw, zw = [], [], []
for ph in np.linspace(0, 2*np.pi, 10, endpoint=False):
    t = np.linspace(0, np.pi, 60)
    xw += list(SEAL*np.sin(t)*np.cos(ph)) + [None]
    yw += list(SEAL*np.sin(t)*np.sin(ph)) + [None]
    zw += list(SEAL*np.cos(t)) + [None]
for th in np.linspace(np.pi*0.15, np.pi*0.85, 6):
    p = np.linspace(0, 2*np.pi, 60)
    r_ = SEAL*np.sin(th)
    xw += list(r_*np.cos(p)) + [None]
    yw += list(r_*np.sin(p)) + [None]
    zw += list(np.full(60, SEAL*np.cos(th))) + [None]

# α → rede conv (raios de tensão entrópica)
xa_, ya_, za_ = [], [], []
for i in range(N):
    xa_ += [0., cx_[i], None];  ya_ += [0., cy_[i], None];  za_ += [0., cz_[i], None]

# Rede conv → MPAP (redistribuição)
xr_, yr_, zr_ = [], [], []
for i in range(N):
    xr_ += [cx_[i], mx_[i], None];  yr_ += [cy_[i], my_[i], None];  zr_ += [cz_[i], mz_[i], None]

fig2 = go.Figure()

fig2.add_trace(go.Surface(
    x=Xsp, y=Ysp, z=Zsp,
    colorscale=[[0,'rgba(0,195,155,0.04)'],[1,'rgba(0,195,155,0.04)']],
    showscale=False, opacity=0.28, hoverinfo='skip',
    name=f'Campo Harmônico SEAL={SEAL:.3f}',
))
fig2.add_trace(go.Scatter3d(
    x=xw, y=yw, z=zw, mode='lines',
    line=dict(color='rgba(0,195,155,0.20)', width=1),
    showlegend=False, hoverinfo='skip',
))
fig2.add_trace(go.Scatter3d(
    x=xa_, y=ya_, z=za_, mode='lines',
    line=dict(color='rgba(255,150,80,0.10)', width=1),
    showlegend=False, hoverinfo='skip',
))
fig2.add_trace(go.Scatter3d(
    x=xr_, y=yr_, z=zr_, mode='lines',
    line=dict(color='rgba(180,175,230,0.18)', width=1.5),
    name='Redistribuição MPAP', hoverinfo='skip',
))
fig2.add_trace(go.Scatter3d(
    x=cx_, y=cy_, z=cz_, mode='markers',
    marker=dict(size=5+raw/raw.max()*9, color='rgba(255,80,55,0.90)',
                line=dict(color='rgba(255,130,100,0.45)', width=1)),
    name=f'Rede Conv  Coh={coh_r:.3f}',
    hovertemplate='N%{pointNumber}  ativ=%{customdata:.4f}<extra>Rede Conv</extra>',
    customdata=raw,
))
fig2.add_trace(go.Scatter3d(
    x=mx_, y=my_, z=mz_, mode='markers',
    marker=dict(
        size=4+geo/geo.max()*16, color=geo,
        colorscale=[[0,'rgba(10,50,160,0.55)'],[1,'rgba(75,200,255,0.95)']],
        cmin=0., cmax=float(geo.max()), showscale=False,
        line=dict(color='rgba(100,205,255,0.55)', width=1),
    ),
    name=f'MPAP (r=SEAL)  Coh={coh_m:.3f}',
    hovertemplate='N%{pointNumber}  geo=%{customdata:.4f}<extra>MPAP</extra>',
    customdata=geo,
))
fig2.add_trace(go.Scatter3d(
    x=[0.], y=[0.], z=[0.], mode='markers+text',
    marker=dict(size=17, color='rgba(255,215,55,0.95)',
                line=dict(color='gold', width=3)),
    text=['α'], textfont=dict(size=16, color='gold'),
    textposition='top center', name='α  r=0  âncora entrópica',
))

fig2.update_layout(
    title=dict(
        text=(f'Modelo Radial Alpha-Phi — Acoplamento MPAP<br>'
              f'α(r=0) âncora · Rede Conv dispersa (r<SEAL) · '
              f'Campo Harmônico SEAL={SEAL:.4f} · '
              f'MPAP concentra na superfície'),
        font=dict(size=13, color='#c5bdb0'), x=0.5,
    ),
    scene=dict(
        xaxis=dict(showgrid=True, gridcolor='#18182a', title='',
                   showbackground=True, backgroundcolor='#07070e', color='#444'),
        yaxis=dict(showgrid=True, gridcolor='#18182a', title='',
                   showbackground=True, backgroundcolor='#07070e', color='#444'),
        zaxis=dict(showgrid=True, gridcolor='#18182a', title='',
                   showbackground=True, backgroundcolor='#07070e', color='#444'),
        bgcolor='#04040a',
        camera=dict(eye=dict(x=1.9, y=1.7, z=1.2)),
        aspectmode='cube',
        annotations=[
            dict(x=0., y=0., z=SEAL*1.14,
                 text=f'Campo Harmônico  SEAL={SEAL:.3f}',
                 font=dict(color='#00d5b0', size=12), showarrow=False),
        ],
    ),
    paper_bgcolor='#05050b',
    legend=dict(font=dict(color='#aaa', size=11),
                bgcolor='rgba(8,8,18,0.92)',
                bordercolor='#252535', borderwidth=1,
                x=0.01, y=0.99),
    margin=dict(l=0, r=0, t=95, b=0),
    height=680,
)

print("\n── FIGURA 2: Modelo Radial (r=0 → r<SEAL → r=SEAL) ──────")
fig2.show()
