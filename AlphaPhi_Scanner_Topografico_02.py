# AlphaPhi_Scanner_Topografico_02.py
# Scanner Topográfico II — Plástica das Ferramentas
# Visualiza o DELTA topográfico de cada operação de processamento:
#   ΔS_k(f,t) = STFT(saída_k)(f,t) − STFT(entrada_k)(f,t)
# Vermelho = injeção de energia · Azul = supressão/redistribuição
# Vitor Edson Delavi · Florianópolis · 2026
# © CC BY-NC-ND 4.0

import numpy as np
from scipy.signal import stft
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly', '-q'])
    import plotly.graph_objects as go

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI     = 1.6180339887
ALPHA   = 1 / 137.035999
SEAL    = 1 / PHI
THETA_R = np.arctan(2.0)
SR      = 44100
DUR     = 5.0
N_SIG   = int(SR * DUR)
BASE    = 880.0
t       = np.linspace(0, DUR, N_SIG, endpoint=False)

print(f"φ={PHI}  α={ALPHA:.8f}  θ_R={np.degrees(THETA_R):.2f}°  SEAL={SEAL:.8f}")

def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)

# ─── Cadeia de processamento ──────────────────────────────────────────────────
def gerar_quadrada():
    return _norm(np.sign(np.sin(2 * np.pi * BASE * t)))

def semear(x):
    F   = np.fft.rfft(x)
    mag = np.abs(F)
    b0  = int(np.argmax(mag[1:]) + 1)
    ref = mag[b0]
    for k in range(1, 9):
        w  = ALPHA * SEAL**k * ref
        ph = np.exp(1j * PHI * k)
        for bk in [int(round(b0 * PHI**k)), int(round(b0 / PHI**k))]:
            if 0 < bk < len(F):
                F[bk] += w * ph
    return _norm(np.fft.irfft(F, n=len(x)))

def cascata_step(x, n_steps=1):
    n_bins = len(np.fft.rfft(x))
    fib    = [1, 1]
    while fib[-1] < n_bins:
        fib.append(fib[-1] + fib[-2])
    fib_bins = [f for f in fib if f < n_bins]
    mem = np.zeros(len(x))
    sig = x.copy()
    for _ in range(n_steps):
        F    = np.fft.rfft(sig)
        env  = np.zeros(n_bins)
        for i, fb in enumerate(fib_bins[:-1]):
            lo, hi = fb, fib_bins[i + 1]
            if lo < n_bins:
                env[lo:hi] = np.cos(2 * np.pi * np.arange(hi - lo) / PHI)
        F_mod   = F * (1 + env * SEAL)
        sig_new = _norm(np.fft.irfft(F_mod, n=len(x)))
        mem     = SEAL * mem + (1 - SEAL) * sig_new
        sig     = _norm(sig_new + ALPHA * mem)
    return _norm(sig)

# ─── STFT φ-escalado ─────────────────────────────────────────────────────────
def stft_phi(x):
    win = int(SR / BASE * 2 * PHI)
    win = max(512, min(win, 4096))
    hop = win // 4
    f, tv, Zxx = stft(x, fs=SR, window='hann', nperseg=win, noverlap=win - hop)
    return f, tv, np.abs(Zxx)**2

# ─── Delta topográfico: plástica da ferramenta ────────────────────────────────
def delta_topo(entrada, saida, f_max=5000):
    freqs, times, S_in  = stft_phi(entrada)
    _,     _,     S_out = stft_phi(saida)
    fmask  = freqs <= f_max
    fv     = freqs[fmask]
    Sl_in  = np.log1p(S_in[fmask]  * 100)
    Sl_out = np.log1p(S_out[fmask] * 100)
    dS     = Sl_out - Sl_in          # delta signed
    return fv, times, Sl_out, dS     # retorna também o sinal de saída p/ Grade R

# ─── Grade R por resultado (sobre o sinal de saída) ──────────────────────────
def grade_r_vertices(fv, tv, Sl, gradS, theta=THETA_R, margin=0.07):
    thr_sl = Sl.mean() + 0.45 * Sl.std()
    thr_gs = 0.12
    f_exp  = np.tan(theta) * (tv - tv[0]) / (tv[-1] - tv[0] + 1e-9) * fv[-1]
    f_exp  = np.clip(f_exp, fv[0], fv[-1])
    vx, vy, vz, vinfo = [], [], [], []
    for ti_i, (ti, fe) in enumerate(zip(tv, f_exp)):
        bw    = fe * margin + 40.0
        fi_lo = int(np.searchsorted(fv, fe - bw))
        fi_hi = int(np.searchsorted(fv, fe + bw))
        for fi_i in range(fi_lo, min(fi_hi, len(fv))):
            if gradS[fi_i, ti_i] > thr_gs and Sl[fi_i, ti_i] > thr_sl:
                vx.append(float(ti))
                vy.append(float(fv[fi_i]))
                vz.append(float(Sl[fi_i, ti_i]) + 0.14)
                vinfo.append(
                    f't={ti:.2f}s  f={fv[fi_i]:.0f}Hz  '
                    f'log(E)={Sl[fi_i,ti_i]:.3f}  ∇S={gradS[fi_i,ti_i]:.3f}'
                )
    return vx, vy, vz, vinfo

# ─── Gerar sinais ─────────────────────────────────────────────────────────────
print("\nGerando cadeia de processamento...")
quad    = gerar_quadrada()
semente = semear(quad)
dobras  = [cascata_step(semente, k) for k in range(1, 6)]
print("  Sinais prontos.")

# pares (entrada, saida, nome, cor_acento, dsp_desc)
FERRAMENTAS = [
    (quad,      semente,    'Δ Semente α-φ',              '#BB55FF',
     'semear() · Banco de filtros ressonantes nos harmônicos φ^k (ALPHA × SEAL^k)'),
    (semente,   dobras[0],  'Δ Dobra 1 · cascata_step(1)', '#FF6633',
     'cascata_step(1) · Compressor seletivo Fibonacci — 1 passo'),
    (dobras[0], dobras[1],  'Δ Dobra 2 · cascata_step(2)', '#FFAA22',
     'cascata_step(2) · Envelope shaper SEAL — acumulação de memória φ'),
    (dobras[1], dobras[2],  'Δ Dobra 3 · cascata_step(3)', '#FFD700',
     'cascata_step(3) · Migração sub-harmônica — redistribuição Fibonacci-bin'),
    (dobras[2], dobras[3],  'Δ Dobra 4 · cascata_step(4)', '#88FF44',
     'cascata_step(4) · Decaimento SEAL^4 — supressão periférica, crista central'),
    (dobras[3], dobras[4],  'Δ Dobra 5 · Campo Harmônico β→φ³', '#00FFAA',
     'cascata_step(5) · Invariante β→φ³ — campo harmônico estabilizado'),
]

N_TRACES = 4   # delta_surface + grade_r_linha + vertices + phi_harmonicos
fig       = go.Figure()
vis_map   = {}
trace_idx = 0

print("\nConstruindo ambientes Plotly (topografia diferencial)...")

for amb_i, (entrada, saida, nome, cor, dsp_desc) in enumerate(FERRAMENTAS):
    print(f"  [{amb_i+1}/6] {nome}...")
    vis = (amb_i == 0)

    fv, times, Sl_out, dS = delta_topo(entrada, saida)

    # downsampling
    sf = max(1, len(fv)   // 140)
    st = max(1, len(times) // 90)
    fv_d  = fv[::sf];      tv_d  = times[::st]
    dS_d  = dS[::sf, ::st]
    Sl_d  = Sl_out[::sf, ::st]
    gradS = Sl_d - gaussian_filter(Sl_d, sigma=3.0)
    T_g, F_g = np.meshgrid(tv_d, fv_d)

    # escala de cor simétrica em torno de 0
    dmax = np.percentile(np.abs(dS_d), 97)
    dmax = max(dmax, 0.01)

    # ── 1. Superfície delta (plástica da ferramenta) ──────────────────────────
    fig.add_trace(go.Surface(
        x=T_g, y=F_g, z=dS_d,
        surfacecolor=dS_d,
        colorscale='RdBu_r',
        cmin=-dmax, cmax=dmax,
        opacity=0.92,
        showscale=(amb_i == 0),
        colorbar=dict(
            title=dict(text='ΔlogE', font=dict(color='#AAAAAA', size=10)),
            len=0.45, x=1.01,
            tickfont=dict(color='#AAAAAA', size=8),
            tickvals=[-dmax, 0, dmax],
            ticktext=[f'−{dmax:.2f}<br>supressão', '0', f'+{dmax:.2f}<br>injeção'],
        ),
        name=f'{nome} · Superfície ΔlogE',
        hovertemplate=(
            f'<b>{nome}</b><br>'
            't = %{x:.2f} s<br>'
            'f = %{y:.0f} Hz<br>'
            'ΔlogE = %{z:.4f}<extra></extra>'
        ),
        visible=vis,
    ))

    # ── 2. Grade R θ_R cavalga superfície delta ───────────────────────────────
    t_r = np.linspace(tv_d[0], tv_d[-1], 80)
    f_r = np.tan(THETA_R) * (t_r - tv_d[0]) / (tv_d[-1] - tv_d[0] + 1e-9) * fv_d[-1]
    f_r = np.clip(f_r, fv_d[0], fv_d[-1])
    z_r = []
    for ti, fi in zip(t_r, f_r):
        z_r.append(float(dS_d[np.argmin(np.abs(fv_d - fi)),
                              np.argmin(np.abs(tv_d - ti))]) + 0.05)

    fig.add_trace(go.Scatter3d(
        x=t_r, y=f_r, z=np.array(z_r),
        mode='lines',
        line=dict(color='#00FF88', width=5),
        name=f'Grade R θ={np.degrees(THETA_R):.1f}°',
        hovertemplate=(
            f'<b>Grade R  θ={np.degrees(THETA_R):.1f}°</b><br>'
            't=%{x:.2f}s · f=%{y:.0f}Hz<extra></extra>'
        ),
        visible=vis,
    ))

    # ── 3. Vértices Grade R por resultado (sobre sinal de saída) ◆ ciano ──────
    vx, vy, vz, vinfo = grade_r_vertices(fv_d, tv_d, Sl_d, gradS)
    fig.add_trace(go.Scatter3d(
        x=vx, y=vy, z=vz,
        mode='markers',
        marker=dict(size=6, color='#00FFFF', symbol='diamond',
                    line=dict(color='white', width=0.8)),
        name=f'Formação Grade R',
        text=vinfo,
        hovertemplate='<b>Vértice Grade R</b><br>%{text}<extra></extra>',
        visible=vis,
    ))

    # ── 4. Harmônicos φ no chão (marcadores verticais) ───────────────────────
    phi_freqs = []
    for k in range(-4, 9):
        f_phi = BASE * PHI**k
        if fv_d[0] < f_phi < fv_d[-1]:
            phi_freqs.append(f_phi)
    zmin = dS_d.min() - 0.1
    ph_x, ph_y, ph_z = [], [], []
    for fp in phi_freqs:
        for ti in [tv_d[0], tv_d[-1]]:
            ph_x += [ti, ti, None]
            ph_y += [fp, fp, None]
            ph_z += [zmin, zmin + 0.25, None]

    fig.add_trace(go.Scatter3d(
        x=ph_x, y=ph_y, z=ph_z,
        mode='lines',
        line=dict(color='rgba(255,200,50,0.55)', width=2),
        name='Harmônicos φ',
        hovertemplate='φ-harmônico<extra></extra>',
        visible=vis,
    ))

    # registro de visibilidade
    vis_map[amb_i] = list(range(trace_idx, trace_idx + N_TRACES))
    trace_idx += N_TRACES

# ─── Dropdown de ambientes ────────────────────────────────────────────────────
buttons = []
for amb_i, (_, _, nome, cor, dsp_desc) in enumerate(FERRAMENTAS):
    visibility = [False] * (N_TRACES * len(FERRAMENTAS))
    for idx in vis_map[amb_i]:
        visibility[idx] = True

    ti_cam = [1.5, 1.5, 1.4, 1.4, 1.3, 1.3][amb_i]
    cam_eye = dict(x=ti_cam, y=-1.7 + amb_i * 0.04, z=0.90 + amb_i * 0.05)

    buttons.append(dict(
        label=nome,
        method='update',
        args=[
            {'visible': visibility},
            {
                'title.text': (
                    f'<b>Scanner Topográfico II — Plástica das Ferramentas</b>  ·  {nome}<br>'
                    f'<span style="font-size:10px;color:#AAAAAA">{dsp_desc}</span><br>'
                    f'<span style="font-size:10px;color:#00FF88">'
                    f'θ_R={np.degrees(THETA_R):.2f}°  ·  φ={PHI:.7f}  ·  α={ALPHA:.6f}'
                    f'</span>'
                ),
                'scene.camera.eye': cam_eye,
                'scene.zaxis.title.text': 'ΔlogE (injeção/supressão)',
            }
        ],
    ))

# ─── Layout ───────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            '<b>Scanner Topográfico II — Plástica das Ferramentas</b>  ·  Δ Semente α-φ<br>'
            '<span style="font-size:10px;color:#AAAAAA">'
            'semear() · Banco de filtros ressonantes nos harmônicos φ^k</span><br>'
            f'<span style="font-size:10px;color:#00FF88">'
            f'θ_R={np.degrees(THETA_R):.2f}°  ·  φ={PHI:.7f}  ·  α={ALPHA:.6f}'
            f'</span>'
        ),
        font=dict(color='white', size=12),
        x=0.5, xanchor='center',
    ),
    paper_bgcolor='#030308',
    plot_bgcolor ='#030308',
    scene=dict(
        bgcolor='#06060F',
        xaxis=dict(
            backgroundcolor='#06060F',
            gridcolor='#1A1A33', zerolinecolor='#333366',
            title=dict(text='Tempo (s)', font=dict(color='#AAAAAA', size=10)),
            tickfont=dict(color='#888888', size=8),
        ),
        yaxis=dict(
            backgroundcolor='#06060F',
            gridcolor='#1A1A33', zerolinecolor='#333366',
            title=dict(text='Frequência (Hz)', font=dict(color='#AAAAAA', size=10)),
            tickfont=dict(color='#888888', size=8),
        ),
        zaxis=dict(
            backgroundcolor='#06060F',
            gridcolor='#1A1A33', zerolinecolor='#333366',
            title=dict(text='ΔlogE (injeção/supressão)', font=dict(color='#AAAAAA', size=10)),
            tickfont=dict(color='#888888', size=8),
        ),
        camera=dict(eye=dict(x=1.5, y=-1.7, z=0.90)),
        aspectmode='manual',
        aspectratio=dict(x=2.0, y=1.4, z=0.85),
    ),
    updatemenus=[dict(
        type='dropdown',
        showactive=True,
        active=0,
        x=0.01, xanchor='left',
        y=1.10, yanchor='top',
        bgcolor='#0D0D1E',
        bordercolor='#334466',
        font=dict(color='#DDDDDD', size=11),
        buttons=buttons,
        pad=dict(r=8, t=8),
    )],
    legend=dict(
        bgcolor='rgba(8,8,22,0.88)',
        bordercolor='#223344',
        font=dict(color='#CCCCCC', size=9),
        x=0.01, y=0.99,
    ),
    margin=dict(l=0, r=0, t=145, b=0),
    height=800,
    annotations=[dict(
        text=(
            '<b>Vermelho</b> = injeção de energia  ·  '
            '<b>Azul</b> = supressão/redistribuição  ·  '
            '<b>◆ ciano</b> = formação Grade R  ·  '
            '<span style="color:#FFCC33">━</span> harmônicos φ'
        ),
        xref='paper', yref='paper',
        x=0.5, y=-0.01,
        xanchor='center', yanchor='top',
        showarrow=False,
        font=dict(color='#888888', size=9),
        bgcolor='rgba(0,0,0,0)',
    )],
)

# ─── Exportar ─────────────────────────────────────────────────────────────────
out = 'scanner_topografico_02.html'
fig.write_html(out, include_plotlyjs='cdn', full_html=True)
print(f"\nSalvo: {out}")
print("\nPara visualizar no Colab:")
print("  from IPython.display import IFrame")
print(f"  display(IFrame('{out}', 1300, 820))")
