# AlphaPhi_Scanner_Interativo.py
# Scanner Topográfico Interativo — Plotly 3D
# Superfície espectral · Grade R · Sub-harmônicos · Navegação por ambiente
# Vitor Edson Delavi · Florianópolis · 2026
# © CC BY-NC-ND 4.0

import numpy as np
from scipy.signal import stft, butter, filtfilt
from scipy.ndimage import gaussian_filter
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
THETA_R = np.arctan(2.0)          # 63.43° — Grade Romboédrica
SR      = 44100
DUR     = 10.0
BASE    = 880.0

print(f"φ  = {PHI}")
print(f"α  = {ALPHA:.8f}")
print(f"θ_R = {np.degrees(THETA_R):.2f}°  (Grade Romboédrica)")

# ─── Geração de sinais ────────────────────────────────────────────────────────
def gerar_ecobeep(sr=SR, dur=DUR):
    t  = np.linspace(0, dur, int(sr*dur), endpoint=False)
    fm = np.sin(2*np.pi*220*t + PHI*np.sin(2*np.pi*(220/PHI)*t))
    sq = np.sign(np.sin(2*np.pi*BASE*t))
    x  = (1-ALPHA)*sq + ALPHA*fm
    return t, x / np.max(np.abs(x))

def gerar_quadrada(sr=SR, dur=DUR):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    return t, np.sign(np.sin(2*np.pi*BASE*t)).astype(float)

# ─── STFT φ-escalado ─────────────────────────────────────────────────────────
def stft_phi(x, sr=SR):
    win = int(sr / BASE * 2 * PHI)
    win = max(512, min(win, 4096))
    hop = win // 4
    f, t, Zxx = stft(x, fs=sr, window='hann', nperseg=win, noverlap=win-hop)
    return f, t, np.abs(Zxx)**2

def stft_subfreq(x, sr=SR):
    """Janela 8× maior — resolução ~3Hz/bin abaixo de 880Hz."""
    win = int(sr / BASE * 8 * PHI)
    win = max(4096, min(win, 32768))
    hop = win // 8
    f, t, Zxx = stft(x, fs=sr, window='hann', nperseg=win, noverlap=win-hop)
    return f, t, np.abs(Zxx)**2

# ─── Gradiente de entropia (∇S) ───────────────────────────────────────────────
def grad_entropia(T, sigma=2.5):
    return T - gaussian_filter(T, sigma=sigma)

# ─── Preparar dados de um ambiente ───────────────────────────────────────────
def preparar_ambiente(x, modo='full'):
    """
    Retorna arrays downsampled prontos para Plotly Surface.
    modo = 'full' (0-5000Hz) ou 'sub' (0-880Hz)
    """
    if modo == 'full':
        freqs, times, S = stft_phi(x)
        f_max = 5000.0
    else:
        freqs, times, S = stft_subfreq(x)
        f_max = BASE * 1.02

    fmask = freqs <= f_max
    fv, Sv = freqs[fmask], S[fmask]
    Sl = np.log1p(Sv * 100)
    gradS = grad_entropia(Sl, sigma=3.0)

    # Downsample para performance Plotly
    sf = max(1, len(fv) // 150)
    st = max(1, len(times) // 100)
    fv_d  = fv[::sf]
    tv_d  = times[::st]
    Sl_d  = Sl[::sf, ::st]
    gS_d  = gradS[::sf, ::st]

    return fv_d, tv_d, Sl_d, gS_d

def grade_r_linha(fv, tv, Sl_d):
    """Linha θ_R que cavalga a superfície espectral."""
    t_r = np.linspace(tv[0], tv[-1], 60)
    f_r = np.tan(THETA_R) * (t_r - tv[0]) / (tv[-1] - tv[0] + 1e-9) * fv[-1]
    f_r = np.clip(f_r, fv[0], fv[-1])

    z_r = []
    for ti, fi in zip(t_r, f_r):
        ti_idx = np.argmin(np.abs(tv - ti))
        fi_idx = np.argmin(np.abs(fv - fi))
        z_r.append(float(Sl_d[fi_idx, ti_idx]) + 0.05)
    return t_r, f_r, np.array(z_r)

def phi_harmonicos_linhas(fv, tv, Sl_d):
    """Linhas verticais nos φ-harmônicos, elevadas até a superfície."""
    linhas = []
    for k in range(-4, 6):
        fh = BASE * PHI**k
        if fv[0] < fh < fv[-1]:
            fi = np.argmin(np.abs(fv - fh))
            z_col = Sl_d[fi, :]
            linhas.append({
                'k': k, 'fh': fh,
                't': tv, 'f': np.full_like(tv, fh), 'z': z_col + 0.02
            })
    return linhas

def vertices_grade_r(fv, tv, Sl_d, gradS):
    """Pontos onde a Grade R cruza com vértices energéticos (∇S > 0)."""
    t_r = np.linspace(tv[0], tv[-1], 60)
    f_r = np.tan(THETA_R) * (t_r - tv[0]) / (tv[-1] - tv[0] + 1e-9) * fv[-1]
    f_r = np.clip(f_r, fv[0], fv[-1])

    vx, vy, vz = [], [], []
    for ti, fi in zip(t_r, f_r):
        ti_idx = np.argmin(np.abs(tv - ti))
        fi_idx = np.argmin(np.abs(fv - fi))
        gs_val = gradS[fi_idx, ti_idx]
        sl_val = Sl_d[fi_idx, ti_idx]
        if gs_val > 0.1:  # vértice energético
            vx.append(float(ti))
            vy.append(float(fi))
            vz.append(float(sl_val) + 0.15)
    return vx, vy, vz

# ─── Construir figura Plotly interativa ───────────────────────────────────────
def scanner_interativo():
    _, eco  = gerar_ecobeep()
    _, quad = gerar_quadrada()

    AMBIENTES = [
        ('EcoBIP 880Hz',  eco,  'full', '#FFD700', 'inferno'),
        ('EcoBIP — Sub-harmônico (0–880Hz)', eco,  'sub',  '#FFD700', 'plasma'),
        ('Quadrada Pura', quad, 'full', '#4488FF', 'viridis'),
        ('Quadrada — Sub-harmônico (0–880Hz)', quad, 'sub',  '#4488FF', 'cividis'),
    ]

    fig = go.Figure()
    N_TRACES_POR_AMB = 4  # surface + grade_r + phi_harmonicos + vertices

    # índices de visibilidade por ambiente
    vis_map = {}
    trace_idx = 0

    for amb_i, (nome, x, modo, cor, cmap) in enumerate(AMBIENTES):
        print(f"  Preparando: {nome}...")
        fv, tv, Sl_d, gS_d = preparar_ambiente(x, modo)

        T_g, F_g = np.meshgrid(tv, fv)

        # Colorscale customizada com cor do sinal
        norm = (Sl_d - Sl_d.min()) / (Sl_d.max() - Sl_d.min() + 1e-9)

        # ── 1: Superfície principal ───────────────────────────────────────────
        fig.add_trace(go.Surface(
            x=T_g, y=F_g, z=Sl_d,
            colorscale=cmap,
            opacity=0.92,
            showscale=(amb_i == 0),
            colorbar=dict(
                title='log(E)', len=0.5, x=1.02,
                tickfont=dict(color='#AAAAAA', size=9)
            ),
            name=f'{nome} · Superfície',
            hovertemplate=(
                '<b>Tempo</b>: %{x:.2f}s<br>'
                '<b>Freq</b>: %{y:.0f}Hz<br>'
                '<b>log(E)</b>: %{z:.3f}<extra></extra>'
            ),
            visible=(amb_i == 0),
        ))

        # ── 2: Linha Grade R (cavalga a superfície) ───────────────────────────
        t_r, f_r, z_r = grade_r_linha(fv, tv, Sl_d)
        fig.add_trace(go.Scatter3d(
            x=t_r, y=f_r, z=z_r,
            mode='lines+markers',
            line=dict(color='#00FF88', width=5),
            marker=dict(size=2, color='#00FF88'),
            name=f'Grade R θ={np.degrees(THETA_R):.1f}° · {nome}',
            hovertemplate=(
                f'<b>Grade R {np.degrees(THETA_R):.1f}°</b><br>'
                'Tempo: %{x:.2f}s<br>'
                'Freq: %{y:.0f}Hz<extra></extra>'
            ),
            visible=(amb_i == 0),
        ))

        # ── 3: φ-harmônicos (cristas sobre a superfície) ─────────────────────
        phi_linhas = phi_harmonicos_linhas(fv, tv, Sl_d)
        phi_t, phi_f, phi_z, phi_txt = [], [], [], []
        for ln in phi_linhas:
            phi_t.extend(list(ln['t']) + [None])
            phi_f.extend(list(ln['f']) + [None])
            phi_z.extend(list(ln['z']) + [None])
        fig.add_trace(go.Scatter3d(
            x=phi_t, y=phi_f, z=phi_z,
            mode='lines',
            line=dict(color='#AA88FF', width=2, dash='dot'),
            name=f'φ-Harmônicos · {nome}',
            hoverinfo='skip',
            visible=(amb_i == 0),
        ))

        # ── 4: Vértices da Grade R (estrelas nos cruzamentos energéticos) ─────
        vx, vy, vz = vertices_grade_r(fv, tv, Sl_d, gS_d)
        fig.add_trace(go.Scatter3d(
            x=vx, y=vy, z=vz,
            mode='markers',
            marker=dict(
                size=7, color='#FFD700',
                symbol='diamond',
                line=dict(color='white', width=1)
            ),
            name=f'Vértices Grade R · {nome}',
            hovertemplate=(
                '<b>Vértice Grade R</b><br>'
                'Tempo: %{x:.2f}s<br>'
                'Freq: %{y:.0f}Hz<extra></extra>'
            ),
            visible=(amb_i == 0),
        ))

        n_traces = 4
        vis_map[amb_i] = list(range(trace_idx, trace_idx + n_traces))
        trace_idx += n_traces

    total_traces = trace_idx

    # ── Botões de navegação entre ambientes ───────────────────────────────────
    buttons = []
    for amb_i, (nome, _, _, cor, _) in enumerate(AMBIENTES):
        vis = [False] * total_traces
        for idx in vis_map[amb_i]:
            vis[idx] = True

        modos_label = ['Espectro Completo (0–5kHz)', 'Sub-harmônico (0–880Hz)',
                       'Espectro Completo (0–5kHz)', 'Sub-harmônico (0–880Hz)']
        cameras = [
            dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            dict(eye=dict(x=1.4, y=-1.8, z=1.1)),
            dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            dict(eye=dict(x=1.4, y=-1.8, z=1.1)),
        ]

        buttons.append(dict(
            label=nome,
            method='update',
            args=[
                {'visible': vis},
                {
                    'title.text': (
                        f'<b>Scanner Topográfico Interativo</b>  ·  {nome}  ·  '
                        f'{modos_label[amb_i]}<br>'
                        f'<span style="font-size:11px;color:#00FF88">'
                        f'θ_R = {np.degrees(THETA_R):.2f}°  ·  '
                        f'φ = {PHI:.7f}  ·  α = {ALPHA:.6f}</span>'
                    ),
                    'scene.camera': cameras[amb_i],
                }
            ]
        ))

    # ── Layout 3D ─────────────────────────────────────────────────────────────
    nome0 = AMBIENTES[0][0]
    fig.update_layout(
        title=dict(
            text=(
                f'<b>Scanner Topográfico Interativo</b>  ·  {nome0}  ·  '
                f'Espectro Completo (0–5kHz)<br>'
                f'<span style="font-size:11px;color:#00FF88">'
                f'θ_R = {np.degrees(THETA_R):.2f}°  ·  '
                f'φ = {PHI:.7f}  ·  α = {ALPHA:.6f}</span>'
            ),
            font=dict(color='white', size=14),
            x=0.5, xanchor='center',
        ),
        paper_bgcolor='#030308',
        plot_bgcolor='#030308',
        scene=dict(
            bgcolor='#070712',
            xaxis=dict(
                title=dict(text='Tempo (s)', font=dict(color='#AAAAAA', size=11)),
                tickfont=dict(color='#777777', size=9),
                gridcolor='#1A1A2A', showbackground=True,
                backgroundcolor='#0A0A18',
                zerolinecolor='#333344',
            ),
            yaxis=dict(
                title=dict(text='Frequência (Hz)', font=dict(color='#AAAAAA', size=11)),
                tickfont=dict(color='#777777', size=9),
                gridcolor='#1A1A2A', showbackground=True,
                backgroundcolor='#0A0A18',
                zerolinecolor='#333344',
            ),
            zaxis=dict(
                title=dict(text='log(Energia)', font=dict(color='#AAAAAA', size=11)),
                tickfont=dict(color='#777777', size=9),
                gridcolor='#1A1A2A', showbackground=True,
                backgroundcolor='#0A0A18',
                zerolinecolor='#333344',
            ),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            aspectmode='manual',
            aspectratio=dict(x=2.0, y=1.5, z=0.7),
        ),
        updatemenus=[dict(
            type='buttons',
            direction='right',
            showactive=True,
            active=0,
            x=0.5, xanchor='center',
            y=1.05, yanchor='top',
            bgcolor='#0D0D1E',
            bordercolor='#333355',
            font=dict(color='#DDDDDD', size=11),
            buttons=buttons,
            pad=dict(r=8, t=8),
        )],
        legend=dict(
            bgcolor='#0A0A1A',
            bordercolor='#222244',
            font=dict(color='#CCCCCC', size=9),
            x=0.01, y=0.99,
        ),
        margin=dict(l=0, r=0, t=110, b=0),
        height=750,
    )

    # Salvar como HTML interativo
    fig.write_html(
        'scanner_interativo.html',
        include_plotlyjs='cdn',
        config={
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['orbitRotation'],
            'scrollZoom': True,
            'displaylogo': False,
        }
    )
    print("Salvo: scanner_interativo.html")
    print("Para abrir no Colab: display(IFrame('scanner_interativo.html', 1200, 800))")

    return fig


# ─── Execução ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  AlphaPhi · Scanner Topográfico Interativo")
    print("  Grade R · Sub-harmônicos · Plotly 3D")
    print("=" * 60)
    fig = scanner_interativo()
    try:
        fig.show()
    except Exception:
        print("Para visualizar: abrir scanner_interativo.html no navegador")
        print("No Colab:")
        print("  from IPython.display import IFrame")
        print("  display(IFrame('scanner_interativo.html', 1200, 800))")
