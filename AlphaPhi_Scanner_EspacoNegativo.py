# AlphaPhi_Scanner_EspacoNegativo.py
# Três abordagens para observar o espaço entre os picos —
# o que o Scanner padrão não registra como imagem.
#
# Abordagem 1 — Espaço Negativo: inverte o contraste (vales → montanhas)
# Abordagem 2 — Sub-frequência: STFT de alta resolução 0–880Hz como superfície principal
# Abordagem 3 — Segunda Derivada: ∇²(Sl) revela bordas e geometria oculta
#
# Vitor Edson Delavi · Florianópolis · 2026
# © CC BY-NC-ND 4.0

import numpy as np
from scipy.signal import stft
from scipy.ndimage import gaussian_filter, laplace
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly', '-q'])
    import plotly.graph_objects as go

# ─── Constantes irrevogáveis ──────────────────────────────────────────────────
PHI     = 1.6180339887
ALPHA   = 1 / 137.035999
SEAL    = 1 / PHI
THETA_R = np.arctan(2.0)
SR      = 44100
DUR     = 10.0          # 10s — suficiente para revelar padrões
N_SIG   = int(SR * DUR)
BASE    = 880.0
t       = np.linspace(0, DUR, N_SIG, endpoint=False)

print(f"φ={PHI}  α={ALPHA:.8f}  θ_R={np.degrees(THETA_R):.2f}°")

def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)

# ─── Sinal: EcoBIP ────────────────────────────────────────────────────────────
def gerar_ecobip():
    quad   = np.sign(np.sin(2 * np.pi * BASE * t))
    fm_phi = np.sin(2 * np.pi * BASE * t + PHI * np.sin(2 * np.pi * (BASE/4) * t))
    return _norm((1 - ALPHA) * quad + ALPHA * fm_phi)

print("Gerando EcoBIP...")
x = gerar_ecobip()
print("  Pronto.\n")

# ─── STFT padrão (φ-escalado) ─────────────────────────────────────────────────
def stft_phi(sig, f_max=5000):
    win = int(SR / BASE * 2 * PHI)
    win = max(512, min(win, 4096))
    hop = win // 4
    f, tv, Zxx = stft(sig, fs=SR, window='hann', nperseg=win, noverlap=win-hop)
    S    = np.abs(Zxx)**2
    mask = f <= f_max
    return f[mask], tv, S[mask]

# ─── STFT sub-frequência (alta resolução 0–880Hz) ────────────────────────────
def stft_sub(sig):
    win = int(SR / BASE * 8 * PHI)
    win = max(4096, min(win, 16384))
    hop = win // 8
    f, tv, Zxx = stft(sig, fs=SR, window='hann', nperseg=win, noverlap=win-hop)
    S    = np.abs(Zxx)**2
    mask = f <= BASE
    return f[mask], tv, S[mask]

# ─── Linha Grade R sobre qualquer superfície ─────────────────────────────────
def linha_grade_r(fv_d, tv_d, Sl_d):
    t_r = np.linspace(tv_d[0], tv_d[-1], 80)
    f_r = np.tan(THETA_R) * (t_r - tv_d[0]) / (tv_d[-1] - tv_d[0] + 1e-9) * fv_d[-1]
    f_r = np.clip(f_r, fv_d[0], fv_d[-1])
    z_r = [float(Sl_d[np.argmin(np.abs(fv_d-fi)), np.argmin(np.abs(tv_d-ti))]) + 0.06
           for ti, fi in zip(t_r, f_r)]
    return t_r, f_r, np.array(z_r)

# ─── Harmônicos φ verticais ───────────────────────────────────────────────────
def phi_harmonicos(fv_d, tv_d, zmin, dz=0.25):
    ph_x, ph_y, ph_z = [], [], []
    for k in range(-4, 9):
        fp = BASE * PHI**k
        if fv_d[0] < fp < fv_d[-1]:
            for ti in [tv_d[0], tv_d[-1]]:
                ph_x += [ti, ti, None]
                ph_y += [fp, fp, None]
                ph_z += [zmin, zmin+dz, None]
    return ph_x, ph_y, ph_z

# ─── Construir ambientes ──────────────────────────────────────────────────────
print("Computando STFTs...")
fv_p, tv_p, S_p = stft_phi(x)
fv_s, tv_s, S_s = stft_sub(x)
print("  STFTs prontas.\n")

fig       = go.Figure()
vis_map   = {}
trace_idx = 0

# ══════════════════════════════════════════════════════════════════════════════
# ABORDAGEM 1 — Espaço Negativo (inverte contraste: vales → montanhas)
# ══════════════════════════════════════════════════════════════════════════════
print("[1/3] Espaço Negativo...")
sf = max(1, len(fv_p)//140); st = max(1, len(tv_p)//90)
fv1, tv1 = fv_p[::sf], tv_p[::st]
Sl_pos   = np.log1p(S_p[::sf, ::st] * 100)
Sl_neg   = Sl_pos.max() - Sl_pos          # inversão: vales viram picos
T1, F1   = np.meshgrid(tv1, fv1)

fig.add_trace(go.Surface(
    x=T1, y=F1, z=Sl_neg,
    colorscale='ice',
    opacity=0.92, showscale=True,
    colorbar=dict(title=dict(text='Sl_max−logE', font=dict(color='#AAA',size=10)),
                  len=0.45, x=1.01, tickfont=dict(color='#AAA',size=8)),
    name='Espaço Negativo',
    hovertemplate='t=%{x:.2f}s · f=%{y:.0f}Hz · val=%{z:.3f}<extra>Espaço Negativo</extra>',
    visible=True,
))
t_r1, f_r1, z_r1 = linha_grade_r(fv1, tv1, Sl_neg)
fig.add_trace(go.Scatter3d(
    x=t_r1, y=f_r1, z=z_r1, mode='lines',
    line=dict(color='#00FF88', width=5),
    name='Grade R θ=63.4°', visible=True,
    hovertemplate='Grade R · t=%{x:.2f}s · f=%{y:.0f}Hz<extra></extra>',
))
ph_x1,ph_y1,ph_z1 = phi_harmonicos(fv1, tv1, Sl_neg.min())
fig.add_trace(go.Scatter3d(
    x=ph_x1, y=ph_y1, z=ph_z1, mode='lines',
    line=dict(color='rgba(255,200,50,0.5)',width=2),
    name='Harmônicos φ', visible=True,
    hovertemplate='φ-harmônico<extra></extra>',
))
vis_map[0] = list(range(trace_idx, trace_idx+3)); trace_idx += 3

# ══════════════════════════════════════════════════════════════════════════════
# ABORDAGEM 2 — Sub-frequência como superfície principal (0–880Hz, ~3Hz/bin)
# ══════════════════════════════════════════════════════════════════════════════
print("[2/3] Sub-frequência como superfície principal...")
sf2 = max(1, len(fv_s)//140); st2 = max(1, len(tv_s)//90)
fv2, tv2 = fv_s[::sf2], tv_s[::st2]
Sl2      = np.log1p(S_s[::sf2, ::st2] * 1000)   # ×1000: amplifica sub-freq fraca
T2, F2   = np.meshgrid(tv2, fv2)

fig.add_trace(go.Surface(
    x=T2, y=F2, z=Sl2,
    colorscale='plasma',
    opacity=0.92, showscale=False,
    name='Sub-freq 0–880Hz',
    hovertemplate='t=%{x:.2f}s · f=%{y:.1f}Hz · logE=%{z:.3f}<extra>Sub-frequência</extra>',
    visible=False,
))
# Grade R adaptada à faixa 0–880Hz
t_r2 = np.linspace(tv2[0], tv2[-1], 80)
f_r2 = np.tan(THETA_R) * (t_r2-tv2[0]) / (tv2[-1]-tv2[0]+1e-9) * fv2[-1]
f_r2 = np.clip(f_r2, fv2[0], fv2[-1])
z_r2 = [float(Sl2[np.argmin(np.abs(fv2-fi)), np.argmin(np.abs(tv2-ti))])+0.06
        for ti,fi in zip(t_r2, f_r2)]
fig.add_trace(go.Scatter3d(
    x=t_r2, y=f_r2, z=np.array(z_r2), mode='lines',
    line=dict(color='#00FF88', width=5),
    name='Grade R (0–880Hz)', visible=False,
    hovertemplate='Grade R · t=%{x:.2f}s · f=%{y:.1f}Hz<extra></extra>',
))
# Harmônicos φ sub-BASE
ph_x2,ph_y2,ph_z2=[],[],[]
for k in range(-8,0):
    fp = BASE * PHI**k
    if fv2[0] < fp < fv2[-1]:
        for ti in [tv2[0], tv2[-1]]:
            ph_x2+=[ti,ti,None]; ph_y2+=[fp,fp,None]
            ph_z2+=[Sl2.min(), Sl2.min()+0.3, None]
fig.add_trace(go.Scatter3d(
    x=ph_x2, y=ph_y2, z=ph_z2, mode='lines',
    line=dict(color='rgba(255,200,50,0.55)',width=2),
    name='Harmônicos φ sub-BASE', visible=False,
    hovertemplate='φ^-k harmônico<extra></extra>',
))
vis_map[1] = list(range(trace_idx, trace_idx+3)); trace_idx += 3

# ══════════════════════════════════════════════════════════════════════════════
# ABORDAGEM 3 — Segunda Derivada ∇²(Sl): geometria oculta nas bordas
# ══════════════════════════════════════════════════════════════════════════════
print("[3/3] Segunda Derivada ∇²(Sl)...")
sf3 = max(1, len(fv_p)//140); st3 = max(1, len(tv_p)//90)
fv3, tv3 = fv_p[::sf3], tv_p[::st3]
Sl3_base = np.log1p(S_p[::sf3, ::st3] * 100)
Sl3_sm   = gaussian_filter(Sl3_base, sigma=1.5)  # suaviza antes de derivar
Lap      = laplace(Sl3_sm)                         # ∇²
p97      = np.percentile(np.abs(Lap), 97)
Lap      = np.clip(Lap, -p97, p97)                 # escala simétrica
T3, F3   = np.meshgrid(tv3, fv3)

fig.add_trace(go.Surface(
    x=T3, y=F3, z=Lap,
    surfacecolor=Lap,
    colorscale='RdBu_r',
    cmin=-p97, cmax=p97,
    opacity=0.92, showscale=False,
    name='∇²(Sl) — bordas',
    hovertemplate='t=%{x:.2f}s · f=%{y:.0f}Hz · ∇²=%{z:.4f}<extra>Segunda Derivada</extra>',
    visible=False,
))
t_r3, f_r3, z_r3 = linha_grade_r(fv3, tv3, Lap)
fig.add_trace(go.Scatter3d(
    x=t_r3, y=f_r3, z=z_r3, mode='lines',
    line=dict(color='#00FF88', width=5),
    name='Grade R θ=63.4°', visible=False,
    hovertemplate='Grade R · t=%{x:.2f}s · f=%{y:.0f}Hz<extra></extra>',
))
ph_x3,ph_y3,ph_z3 = phi_harmonicos(fv3, tv3, Lap.min(), dz=0.15)
fig.add_trace(go.Scatter3d(
    x=ph_x3, y=ph_y3, z=ph_z3, mode='lines',
    line=dict(color='rgba(255,200,50,0.5)',width=2),
    name='Harmônicos φ', visible=False,
    hovertemplate='φ-harmônico<extra></extra>',
))
vis_map[2] = list(range(trace_idx, trace_idx+3)); trace_idx += 3

# ─── Dropdown ─────────────────────────────────────────────────────────────────
N_AMB = 3
descricoes = [
    ('Espaço Negativo  —  vales → montanhas',
     'Inverte log(E): o que era silêncio torna-se visível. Revela a topografia entre os picos.',
     'ice', dict(x=1.8,y=-1.8,z=0.90)),
    ('Sub-frequência  —  0–880Hz como superfície',
     'STFT de alta resolução (~3Hz/bin) abaixo de BASE. Revela estrutura sub-harmônica fraca.',
     'plasma', dict(x=1.8,y=-1.8,z=0.90)),
    ('Segunda Derivada  ∇²(Sl)  —  geometria oculta',
     'Laplaciano de log(E): destaca bordas e transições. Revela padrão geométrico nos vales.',
     'RdBu', dict(x=1.8,y=-1.8,z=0.90)),
]

buttons = []
for i, (nome, desc, _, cam) in enumerate(descricoes):
    vis = [False] * (N_AMB * 3)
    for idx in vis_map[i]: vis[idx] = True
    buttons.append(dict(
        label=nome,
        method='update',
        args=[{'visible': vis},
              {'title.text': (
                  f'<b>Scanner — Espaço Oculto · {nome}</b><br>'
                  f'<span style="font-size:10px;color:#AAAAAA">{desc}</span><br>'
                  f'<span style="font-size:10px;color:#00FF88">'
                  f'θ_R={np.degrees(THETA_R):.2f}°  φ={PHI:.7f}  α={ALPHA:.6f}</span>'
              ), 'scene.camera.eye': cam}],
    ))

# ─── Layout ───────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            '<b>Scanner — Espaço Oculto · Espaço Negativo  —  vales → montanhas</b><br>'
            '<span style="font-size:10px;color:#AAAAAA">'
            'Inverte log(E): o que era silêncio torna-se visível.</span><br>'
            f'<span style="font-size:10px;color:#00FF88">'
            f'θ_R={np.degrees(THETA_R):.2f}°  φ={PHI:.7f}  α={ALPHA:.6f}</span>'
        ),
        font=dict(color='white', size=12),
        x=0.5, xanchor='center',
    ),
    paper_bgcolor='#030308', plot_bgcolor='#030308',
    scene=dict(
        bgcolor='#06060F',
        xaxis=dict(backgroundcolor='#06060F', gridcolor='#1A1A33',
                   title=dict(text='Tempo (s)', font=dict(color='#AAA',size=10)),
                   tickfont=dict(color='#888',size=8)),
        yaxis=dict(backgroundcolor='#06060F', gridcolor='#1A1A33',
                   title=dict(text='Frequência (Hz)', font=dict(color='#AAA',size=10)),
                   tickfont=dict(color='#888',size=8)),
        zaxis=dict(backgroundcolor='#06060F', gridcolor='#1A1A33',
                   title=dict(text='Valor observado', font=dict(color='#AAA',size=10)),
                   tickfont=dict(color='#888',size=8)),
        camera=dict(eye=dict(x=1.8, y=-1.8, z=0.90)),
        aspectmode='manual', aspectratio=dict(x=2.0, y=1.4, z=0.85),
    ),
    updatemenus=[dict(
        type='dropdown', showactive=True, active=0,
        x=0.01, xanchor='left', y=1.10, yanchor='top',
        bgcolor='#0D0D1E', bordercolor='#334466',
        font=dict(color='#DDDDDD', size=11),
        buttons=buttons, pad=dict(r=8,t=8),
    )],
    legend=dict(bgcolor='rgba(8,8,22,0.88)', bordercolor='#223344',
                font=dict(color='#CCC',size=9), x=0.01, y=0.99),
    margin=dict(l=0,r=0,t=145,b=0), height=800,
    annotations=[dict(
        text=(
            '<b>1 — Espaço Negativo</b>: vales → montanhas  ·  '
            '<b>2 — Sub-frequência</b>: 0–880Hz alta resolução  ·  '
            '<b>3 — ∇²(Sl)</b>: geometria das bordas'
        ),
        xref='paper', yref='paper', x=0.5, y=-0.01,
        xanchor='center', yanchor='top', showarrow=False,
        font=dict(color='#666',size=9),
    )],
)

out = 'scanner_espaco_oculto.html'
fig.write_html(out, include_plotlyjs='cdn', full_html=True)
print(f"\nSalvo: {out}")
print("\nPara visualizar no Colab:")
print("  from IPython.display import HTML")
print(f"  with open('{out}') as f: display(HTML(f.read()))")
