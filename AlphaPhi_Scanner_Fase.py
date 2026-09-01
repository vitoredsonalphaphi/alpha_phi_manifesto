# AlphaPhi_Scanner_Fase.py
# Scanner de Fase — observar o eixo invisível do EcoBIP
# Três painéis: Amplitude · Fase Bruta · Coerência de Fase
# Vitor Edson Delavi · Florianópolis · 2026
# © CC BY-NC-ND 4.0

import numpy as np
from scipy.signal import stft
from scipy.ndimage import gaussian_filter, uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly', '-q'])
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI     = 1.6180339887
ALPHA   = 1 / 137.035999
SEAL    = 1 / PHI
THETA_R = np.arctan(2.0)
SR      = 44100
DUR     = 10.0        # janela maior para ver mais ciclos de alternância
N_SIG   = int(SR * DUR)
BASE    = 880.0
t       = np.linspace(0, DUR, N_SIG, endpoint=False)

print(f"φ={PHI}  α={ALPHA:.8f}  θ_R={np.degrees(THETA_R):.2f}°  DUR={DUR}s")

def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)

# ─── Sinal EcoBIP (Quadrada → Semente α-φ) ───────────────────────────────────
def gerar_ecobip():
    quad = _norm(np.sign(np.sin(2 * np.pi * BASE * t)))
    F    = np.fft.rfft(quad)
    mag  = np.abs(F)
    b0   = int(np.argmax(mag[1:]) + 1)
    ref  = mag[b0]
    for k in range(1, 9):
        w  = ALPHA * SEAL**k * ref
        ph = np.exp(1j * PHI * k)
        for bk in [int(round(b0 * PHI**k)), int(round(b0 / PHI**k))]:
            if 0 < bk < len(F):
                F[bk] += w * ph
    return _norm(np.fft.irfft(F, n=len(quad)))

print("Gerando EcoBIP...")
sig = gerar_ecobip()
print("  Pronto.")

# ─── STFT mantendo fase ───────────────────────────────────────────────────────
print("Calculando STFT complexa...")
win = int(SR / BASE * 2 * PHI)
win = max(512, min(win, 4096))
hop = win // 4

freqs, times, Zxx = stft(sig, fs=SR, window='hann', nperseg=win, noverlap=win-hop)

# Recortar até 5000Hz
fmask = freqs <= 5000
fv    = freqs[fmask]
Zc    = Zxx[fmask]          # complexo completo

# ── Amplitude ─────────────────────────────────────────────────────────────────
S      = np.abs(Zc)**2
Sl     = np.log1p(S * 100)
gradS  = Sl - gaussian_filter(Sl, sigma=2.0)

# ── Fase bruta (-π a π) ───────────────────────────────────────────────────────
phase_raw = np.angle(Zc)    # shape (f, t)

# ── Coerência de fase local (PLV sobre janela de tempo) ───────────────────────
# PLV(f, t) = |mean(exp(i*phi))| numa janela ±W frames
W = 12  # ~0.3s de janela local dependendo do hop
cos_m = uniform_filter1d(np.cos(phase_raw), size=2*W+1, axis=1)
sin_m = uniform_filter1d(np.sin(phase_raw), size=2*W+1, axis=1)
phase_coh = np.sqrt(cos_m**2 + sin_m**2)   # 0=incoerente, 1=coerente

print("  Fase e coerência calculadas.")

# ─── Vértices Grade R (por critério de amplitude + gradiente) ─────────────────
def grade_r_vertices_2d(fv, tv, Sl, gradS, theta=THETA_R, margin=0.07):
    thr_sl = Sl.mean() + 0.45 * Sl.std()
    thr_gs = 0.12
    f_exp  = np.tan(theta) * (tv - tv[0]) / (tv[-1] - tv[0] + 1e-9) * fv[-1]
    f_exp  = np.clip(f_exp, fv[0], fv[-1])
    vt, vf = [], []
    for ti_i, (ti, fe) in enumerate(zip(tv, f_exp)):
        bw   = fe * margin + 40.0
        fi_lo = int(np.searchsorted(fv, fe - bw))
        fi_hi = int(np.searchsorted(fv, fe + bw))
        for fi_i in range(fi_lo, min(fi_hi, len(fv))):
            if gradS[fi_i, ti_i] > thr_gs and Sl[fi_i, ti_i] > thr_sl:
                vt.append(float(ti))
                vf.append(float(fv[fi_i]))
    return vt, vf

print("Detectando vértices Grade R...")
# downsample para plot
sf = max(1, len(fv) // 200)
st = max(1, len(times) // 300)
fv_d   = fv[::sf];     tv_d   = times[::st]
Sl_d   = Sl[::sf, ::st]
gS_d   = gradS[::sf, ::st]
ph_d   = phase_raw[::sf, ::st]
coh_d  = phase_coh[::sf, ::st]

vt, vf = grade_r_vertices_2d(fv_d, tv_d, Sl_d, gS_d)
print(f"  {len(vt)} vértices Grade R detectados.")

# ─── Construir figura Plotly ───────────────────────────────────────────────────
print("Construindo figura...")
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[
        'Amplitude  log(E)  ·  eixo conhecido',
        'Fase Bruta  (−π … +π)  ·  eixo invisível',
        'Coerência de Fase  (PLV local)  ·  0=caos → 1=ordem',
    ]
)

common_hover = (
    'tempo = %{x:.3f} s<br>'
    'freq  = %{y:.0f} Hz<br>'
    'valor = %{z:.4f}<extra></extra>'
)

# ── Painel 1: Amplitude ───────────────────────────────────────────────────────
fig.add_trace(go.Heatmap(
    x=tv_d, y=fv_d, z=Sl_d,
    colorscale='Inferno',
    colorbar=dict(
        title=dict(text='log(E)', font=dict(color='#AAA', size=10)),
        len=0.28, y=0.88, tickfont=dict(color='#AAA', size=8),
    ),
    hovertemplate=common_hover,
    name='Amplitude',
    zsmooth='best',
), row=1, col=1)

# Grade R sobre amplitude
fig.add_trace(go.Scatter(
    x=vt, y=vf,
    mode='markers',
    marker=dict(size=4, color='#00FFFF', opacity=0.7,
                symbol='diamond', line=dict(color='white', width=0.5)),
    name='Vértice Grade R',
    hovertemplate='Grade R<br>t=%{x:.3f}s  f=%{y:.0f}Hz<extra></extra>',
), row=1, col=1)

# Linha θ_R
t_rl = np.linspace(tv_d[0], tv_d[-1], 200)
f_rl = np.tan(THETA_R) * (t_rl - tv_d[0]) / (tv_d[-1] - tv_d[0] + 1e-9) * fv_d[-1]
f_rl = np.clip(f_rl, fv_d[0], fv_d[-1])
fig.add_trace(go.Scatter(
    x=t_rl, y=f_rl,
    mode='lines',
    line=dict(color='#00FF88', width=1.5, dash='dot'),
    name='θ_R = 63.43°',
    hoverinfo='skip',
), row=1, col=1)

# ── Painel 2: Fase bruta ──────────────────────────────────────────────────────
fig.add_trace(go.Heatmap(
    x=tv_d, y=fv_d, z=ph_d,
    colorscale='RdBu',
    zmid=0,
    colorbar=dict(
        title=dict(text='fase (rad)', font=dict(color='#AAA', size=10)),
        len=0.28, y=0.50, tickfont=dict(color='#AAA', size=8),
    ),
    hovertemplate=common_hover,
    name='Fase',
    zsmooth='best',
    showscale=True,
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=vt, y=vf,
    mode='markers',
    marker=dict(size=4, color='#00FFFF', opacity=0.7,
                symbol='diamond', line=dict(color='white', width=0.5)),
    name='Vértice Grade R',
    showlegend=False,
    hovertemplate='Grade R<br>t=%{x:.3f}s  f=%{y:.0f}Hz<extra></extra>',
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=t_rl, y=f_rl,
    mode='lines',
    line=dict(color='#00FF88', width=1.5, dash='dot'),
    name='θ_R',
    showlegend=False,
    hoverinfo='skip',
), row=2, col=1)

# ── Painel 3: Coerência de fase ───────────────────────────────────────────────
fig.add_trace(go.Heatmap(
    x=tv_d, y=fv_d, z=coh_d,
    colorscale='Viridis',
    zmin=0, zmax=1,
    colorbar=dict(
        title=dict(text='PLV', font=dict(color='#AAA', size=10)),
        len=0.28, y=0.12, tickfont=dict(color='#AAA', size=8),
    ),
    hovertemplate=common_hover,
    name='Coerência',
    zsmooth='best',
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=vt, y=vf,
    mode='markers',
    marker=dict(size=4, color='#FF4400', opacity=0.85,
                symbol='diamond', line=dict(color='white', width=0.5)),
    name='Vértice Grade R',
    showlegend=False,
    hovertemplate='Grade R<br>t=%{x:.3f}s  f=%{y:.0f}Hz<extra></extra>',
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=t_rl, y=f_rl,
    mode='lines',
    line=dict(color='#00FF88', width=1.5, dash='dot'),
    name='θ_R',
    showlegend=False,
    hoverinfo='skip',
), row=3, col=1)

# ─── Layout ───────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            '<b>Scanner de Fase — EcoBIP α-φ</b><br>'
            f'<span style="font-size:11px;color:#00FF88">'
            f'BASE={BASE:.0f}Hz  ·  DUR={DUR}s  ·  '
            f'θ_R={np.degrees(THETA_R):.2f}°  ·  '
            f'φ={PHI:.6f}  ·  α={ALPHA:.6f}'
            f'</span>'
        ),
        font=dict(color='white', size=14),
        x=0.5, xanchor='center',
    ),
    paper_bgcolor='#030308',
    plot_bgcolor='#070712',
    font=dict(color='#CCCCCC'),
    height=1050,
    legend=dict(
        bgcolor='rgba(8,8,22,0.88)',
        bordercolor='#223344',
        font=dict(color='#CCCCCC', size=9),
        x=1.12, y=0.99,
    ),
    margin=dict(l=60, r=140, t=120, b=50),
)

for row in [1, 2, 3]:
    fig.update_yaxes(
        title_text='Freq (Hz)',
        title_font=dict(color='#AAAAAA', size=10),
        tickfont=dict(color='#555566', size=8),
        gridcolor='#111122',
        row=row, col=1,
    )

fig.update_xaxes(
    title_text='Tempo (s)',
    title_font=dict(color='#AAAAAA', size=10),
    tickfont=dict(color='#555566', size=8),
    gridcolor='#111122',
    row=3, col=1,
)

for ann in fig.layout.annotations:
    ann.font.color = '#AAAAAA'
    ann.font.size  = 11

# ─── Exportar ─────────────────────────────────────────────────────────────────
out = 'scanner_fase.html'
fig.write_html(
    out,
    include_plotlyjs='cdn',
    config={
        'displayModeBar': True,
        'scrollZoom': True,
        'displaylogo': False,
    }
)
print(f"\nSalvo: {out}")
print("\nColab:")
print("  from IPython.display import IFrame")
print(f"  display(IFrame('{out}', 1300, 1100))")

try:
    fig.show()
except Exception:
    pass
