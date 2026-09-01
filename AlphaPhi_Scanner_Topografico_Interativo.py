# AlphaPhi_Scanner_Topografico_Interativo.py
# Scanner Topográfico Interativo — Progressão das Dobras
# Combina: interatividade Plotly 3D + sequência das 5 dobras +
#          Grade R por resultado + tapete sub-harmônico
# Vitor Edson Delavi · Florianópolis · 2026
# © CC BY-NC-ND 4.0

import numpy as np
from scipy.signal import stft
from scipy.ndimage import gaussian_filter, uniform_filter1d
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
THETA_R = np.arctan(2.0)   # 63.43° — Grade Romboédrica
SR      = 44100
DUR     = 5.0
N_SIG   = int(SR * DUR)
BASE    = 880.0
t       = np.linspace(0, DUR, N_SIG, endpoint=False)

print(f"φ  = {PHI}")
print(f"α  = {ALPHA:.8f}")
print(f"θ_R = {np.degrees(THETA_R):.2f}°")
print(f"SEAL = {SEAL:.8f}")

def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)

# ─── Geradores de sinal ───────────────────────────────────────────────────────
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
    n_bins  = len(np.fft.rfft(x))
    fib     = [1, 1]
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
    f, tv, Zxx = stft(x, fs=SR, window='hann', nperseg=win, noverlap=win-hop)
    return f, tv, np.abs(Zxx)**2

def stft_phi_complex(x):
    win = int(SR / BASE * 2 * PHI)
    win = max(512, min(win, 4096))
    hop = win // 4
    f, tv, Zxx = stft(x, fs=SR, window='hann', nperseg=win, noverlap=win-hop)
    return f, tv, Zxx  # complexo — fase preservada

def stft_subfreq(x):
    win = int(SR / BASE * 8 * PHI)
    win = max(4096, min(win, 16384))
    hop = win // 8
    f, tv, Zxx = stft(x, fs=SR, window='hann', nperseg=win, noverlap=win-hop)
    return f, tv, np.abs(Zxx)**2

# ─── Vértices da Grade R por resultado ───────────────────────────────────────
def grade_r_vertices(fv, tv, Sl, gradS, theta=THETA_R, margin=0.07):
    """
    Localiza ONDE a Grade R está se formando por resultado energético:
    proximidade à diagonal θ_R + vértice (∇S > 0) + energia acima da média.
    Retorna coordenadas (t, f, z) e texto para hover.
    """
    thr_sl = Sl.mean() + 0.45 * Sl.std()
    thr_gs = 0.12

    f_exp = np.tan(theta) * (tv - tv[0]) / (tv[-1] - tv[0] + 1e-9) * fv[-1]
    f_exp = np.clip(f_exp, fv[0], fv[-1])

    vx, vy, vz, vinfo = [], [], [], []
    for ti_i, (ti, fe) in enumerate(zip(tv, f_exp)):
        bw = fe * margin + 40.0
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

# ─── Gerar todos os sinais ────────────────────────────────────────────────────
print("\nGerando sinais (Quadrada → Semente → 5 Dobras)...")
quad    = gerar_quadrada()
semente = semear(quad)
dobras  = [cascata_step(semente, k) for k in range(1, 6)]
print("  Sinais prontos.")

AMBIENTES = [
    ('Base — Quadrada 880Hz',           quad,       '#888888', 'Greys',   dict(x=1.9, y=-1.9, z=0.75), 'amp'),
    ('Semente α-φ',                     semente,    '#BB55FF', 'Purples', dict(x=1.7, y=-1.7, z=0.85), 'amp'),
    ('Dobra 1',                         dobras[0],  '#FF6633', 'hot',     dict(x=1.6, y=-1.7, z=0.90), 'amp'),
    ('Dobra 2',                         dobras[1],  '#FFAA22', 'YlOrRd',  dict(x=1.5, y=-1.6, z=0.95), 'amp'),
    ('Dobra 3',                         dobras[2],  '#FFD700', 'inferno', dict(x=1.4, y=-1.6, z=1.00), 'amp'),
    ('Dobra 4',                         dobras[3],  '#88FF44', 'YlGn',    dict(x=1.4, y=-1.5, z=1.05), 'amp'),
    ('Dobra 5 — Campo Harmônico β→φ³',  dobras[4],  '#00FFAA', 'viridis', dict(x=1.3, y=-1.5, z=1.10), 'amp'),
    ('Fase Bruta — EcoBIP',             semente,    '#4488FF', 'RdBu',    dict(x=1.3, y=-1.5, z=1.10), 'phase_raw'),
    ('Coerência de Fase — EcoBIP',      semente,    '#44FF88', 'Viridis', dict(x=1.3, y=-1.5, z=1.10), 'phase_coh'),
    ('Acoplamento Fase Base↔Teto',      semente,    '#FF88FF', 'Viridis', dict(x=1.5, y=-1.8, z=1.20), 'phase_couple'),
    ('Base + Teto Unificados',          semente,    '#AAFFFF', 'ice',     dict(x=1.8, y=-2.2, z=2.00), 'unified'),
]

N_TRACES = 5  # por ambiente: superfície + grade_r_linha + vértices + tapete_sub + phi_harm
fig = go.Figure()
vis_map  = {}
trace_idx = 0

print("\nConstruindo ambientes Plotly...")

for amb_i, (nome, x, cor, cmap, cam_eye, mode) in enumerate(AMBIENTES):
    print(f"  [{amb_i+1}/{len(AMBIENTES)}] {nome}...")
    vis = (amb_i == 0)

    # STFT principal — amplitude ou fase
    plv_map       = None   # inicialização — sobrescrito nos modos de fase
    Sl_base       = None   # inicialização — sobrescrito no modo unified
    Sl_teto_color = None   # inicialização — sobrescrito no modo unified
    if mode == 'phase_raw':
        freqs, times, Zxx_c = stft_phi_complex(x)
        fmask = freqs <= 5000
        fv    = freqs[fmask]
        Zc    = Zxx_c[fmask]
        Sl    = np.angle(Zc)          # fase bruta -π … +π
        gradS = Sl - gaussian_filter(Sl, sigma=3.0)
        S     = np.abs(Zc)**2
        Sl_amp = np.log1p(S * 100)
        gradS_amp = Sl_amp - gaussian_filter(Sl_amp, sigma=3.0)
    elif mode in ('phase_coh', 'phase_couple'):
        # Amplitude para altura Z (teto visível)
        freqs, times, S_amp = stft_phi(x)
        fmask  = freqs <= 5000
        fv     = freqs[fmask]
        Sl     = np.log1p(S_amp[fmask] * 100)
        gradS  = Sl - gaussian_filter(Sl, sigma=3.0)
        Sl_amp = Sl
        gradS_amp = gradS
        # PLV para cor da superfície
        _, _, Zxx_c = stft_phi_complex(x)
        Zc    = Zxx_c[fmask]
        ph    = np.angle(Zc)
        W     = 12
        cos_m = uniform_filter1d(np.cos(ph), size=2*W+1, axis=1)
        sin_m = uniform_filter1d(np.sin(ph), size=2*W+1, axis=1)
        plv_map = np.sqrt(cos_m**2 + sin_m**2)   # PLV 0–1
        ph_raw  = ph   # fase bruta — tapete do acoplamento
    elif mode == 'unified':
        freqs, times, S = stft_phi(x)
        fmask  = freqs <= 5000
        fv     = freqs[fmask]
        Sl_pos_raw = np.log1p(S[fmask] * 100)
        Sl_neg_raw = Sl_pos_raw.max() - Sl_pos_raw   # Espaço Negativo
        Sl_base = Sl_pos_raw / (Sl_pos_raw.max() + 1e-9)   # piso: amplitude 0–1
        Sl_teto_color = Sl_neg_raw / (Sl_neg_raw.max() + 1e-9)  # cor do teto: Espaço Negativo 0–1
        # Teto Z deslocado 2+ acima do piso (0–1) para ser claramente visível
        Sl = Sl_teto_color + 2.0   # teto Z: 2.0 a 3.0
        gradS   = Sl - gaussian_filter(Sl, sigma=3.0)
        Sl_amp  = Sl_pos_raw
        gradS_amp = Sl_amp - gaussian_filter(Sl_amp, sigma=3.0)
        ph_raw  = None
    else:
        freqs, times, S = stft_phi(x)
        fmask = freqs <= 5000
        fv, Sv = freqs[fmask], S[fmask]
        Sl    = np.log1p(Sv * 100)
        gradS = Sl - gaussian_filter(Sl, sigma=3.0)
        Sl_amp = Sl
        gradS_amp = gradS
        ph_raw = None

    sf = max(1, len(fv) // 140)
    st = max(1, len(times) // 90)
    fv_d  = fv[::sf];    tv_d  = times[::st]
    Sl_d  = Sl[::sf, ::st]
    gS_d  = gradS[::sf, ::st]
    plv_d        = plv_map[::sf, ::st]       if mode in ('phase_coh', 'phase_couple') else None
    teto_color_d = Sl_teto_color[::sf, ::st] if mode == 'unified'                    else None
    T_g, F_g = np.meshgrid(tv_d, fv_d)

    # ── 1. Superfície espectral principal ─────────────────────────────────────
    if mode == 'unified':
        # Teto = Z deslocado 2–3 acima do piso; COR = Espaço Negativo (ice)
        fig.add_trace(go.Surface(
            x=T_g, y=F_g, z=Sl_d,
            surfacecolor=teto_color_d,
            colorscale='ice',
            cmin=0, cmax=1,
            opacity=0.93,
            showscale=True,
            colorbar=dict(
                title=dict(text='Teto (neg)', font=dict(color='#AAAAAA', size=10)),
                len=0.45, x=1.01,
                tickfont=dict(color='#AAAAAA', size=8),
            ),
            name=f'{nome} · Teto',
            hovertemplate=(
                f'<b>Teto — Espaço Negativo</b><br>'
                't = %{x:.2f} s<br>'
                'f = %{y:.0f} Hz<br>'
                'val = %{z:.3f}<extra></extra>'
            ),
            visible=vis,
        ))
    elif mode in ('phase_coh', 'phase_couple'):
        # Z = amplitude (teto visível), cor = PLV (coerência de fase)
        fig.add_trace(go.Surface(
            x=T_g, y=F_g, z=Sl_d,
            surfacecolor=plv_d,
            colorscale='Viridis',
            cmin=0, cmax=1,
            opacity=0.93,
            showscale=True,
            colorbar=dict(
                title=dict(text='PLV (coer.)', font=dict(color='#AAAAAA', size=10)),
                len=0.45, x=1.01,
                tickfont=dict(color='#AAAAAA', size=8),
            ),
            name=f'{nome} · Superfície',
            hovertemplate=(
                f'<b>{nome}</b><br>'
                't = %{x:.2f} s<br>'
                'f = %{y:.0f} Hz<br>'
                'log(E) = %{z:.3f}<extra></extra>'
            ),
            visible=vis,
        ))
    else:
        surf_extra = {}
        if mode == 'phase_raw':
            surf_extra = dict(cmin=-np.pi, cmax=np.pi, cmid=0)
        fig.add_trace(go.Surface(
            x=T_g, y=F_g, z=Sl_d,
            colorscale=cmap,
            **surf_extra,
            opacity=0.91,
            showscale=(amb_i == 0),
            colorbar=dict(
                title=dict(text='log(E)', font=dict(color='#AAAAAA', size=10)),
                len=0.45, x=1.01,
                tickfont=dict(color='#AAAAAA', size=8),
            ),
            name=f'{nome} · Superfície',
            hovertemplate=(
                f'<b>{nome}</b><br>'
                't = %{x:.2f} s<br>'
                'f = %{y:.0f} Hz<br>'
                'log(E) = %{z:.3f}<extra></extra>'
            ),
            visible=vis,
        ))

    # ── 2. Linha Grade R θ_R (cavalga superfície) ─────────────────────────────
    t_r  = np.linspace(tv_d[0], tv_d[-1], 80)
    f_r  = np.tan(THETA_R) * (t_r - tv_d[0]) / (tv_d[-1] - tv_d[0] + 1e-9) * fv_d[-1]
    f_r  = np.clip(f_r, fv_d[0], fv_d[-1])
    z_r  = []
    for ti, fi in zip(t_r, f_r):
        z_r.append(float(Sl_d[np.argmin(np.abs(fv_d-fi)),
                               np.argmin(np.abs(tv_d-ti))]) + 0.09)

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

    # ── 3. Vértices Grade R por resultado ◆ ciano ─────────────────────────────
    Sl_amp_d   = Sl_amp[::sf, ::st]
    gradS_amp_d = gradS_amp[::sf, ::st]
    vx, vy, vz, vinfo = grade_r_vertices(fv_d, tv_d, Sl_amp_d, gradS_amp_d)

    fig.add_trace(go.Scatter3d(
        x=vx, y=vy, z=vz,
        mode='markers',
        marker=dict(
            size=6, color='#00FFFF', symbol='diamond',
            line=dict(color='white', width=0.8),
        ),
        name=f'Formação Grade R · {nome}',
        text=vinfo,
        hovertemplate='<b>Vértice Grade R</b><br>%{text}<extra></extra>',
        visible=vis,
    ))

    # ── 4. Tapete no piso ─────────────────────────────────────────────────────
    if mode == 'unified':
        # Piso = amplitude base 0–1 (naturalmente abaixo do teto 2–3)
        Sl_base_d = Sl_base[::sf, ::st]
        T_s, F_s  = np.meshgrid(tv_d, fv_d)
        fig.add_trace(go.Surface(
            x=T_s, y=F_s, z=Sl_base_d,
            colorscale='hot',
            cmin=0, cmax=1,
            opacity=0.88,
            showscale=False,
            name=f'Base — Amplitude · {nome}',
            hovertemplate=(
                '<b>Base (amplitude)</b><br>'
                't=%{x:.2f}s<br>'
                'f=%{y:.0f}Hz<br>'
                'amp=%{z:.3f}<extra></extra>'
            ),
            visible=vis,
        ))
    elif mode == 'phase_couple':
        # Piso = fase bruta (-π … +π) — o "Base" da fase
        ph_raw_d = ph_raw[::sf, ::st]
        T_s, F_s = np.meshgrid(tv_d, fv_d)
        z_piso   = np.full_like(ph_raw_d, -0.4)
        fig.add_trace(go.Surface(
            x=T_s, y=F_s, z=z_piso,
            surfacecolor=ph_raw_d,
            colorscale='RdBu',
            cmin=-np.pi, cmax=np.pi,
            opacity=0.85,
            showscale=False,
            name=f'Fase Bruta (base) · {nome}',
            hovertemplate=(
                '<b>Fase Bruta — Base</b><br>'
                't=%{x:.2f}s<br>'
                'f=%{y:.0f}Hz<br>'
                'fase=%{customdata:.3f} rad<extra></extra>'
            ),
            customdata=ph_raw_d,
            visible=vis,
        ))
    else:
        f_sub, t_sub, S_sub = stft_subfreq(x)
        fm_s = f_sub <= BASE * 1.01
        fv_s = f_sub[fm_s]; Sv_s = S_sub[fm_s]
        Sl_s = np.log1p(Sv_s * 100)
        sf2 = max(1, len(fv_s) // 80)
        st2 = max(1, len(t_sub) // 70)
        fv_s2  = fv_s[::sf2];  tv_s2  = t_sub[::st2]
        Sl_s2  = Sl_s[::sf2, ::st2]
        T_s, F_s = np.meshgrid(tv_s2, fv_s2)
        z_piso   = np.full_like(Sl_s2, -0.4)
        fig.add_trace(go.Surface(
            x=T_s, y=F_s, z=z_piso,
            surfacecolor=Sl_s2,
            colorscale='Plasma',
            opacity=0.80,
            showscale=False,
            name=f'Sub-harmônicos 0–880Hz · {nome}',
            hovertemplate=(
                '<b>Sub-harmônico</b><br>'
                't=%{x:.2f}s<br>'
                'f=%{y:.0f}Hz<br>'
                'log(E)=%{customdata:.3f}<extra></extra>'
            ),
            customdata=Sl_s2,
            visible=vis,
        ))

    # ── 5. φ-harmônicos (cristas sobre superfície) ───────────────────────────
    phi_t, phi_f, phi_z = [], [], []
    for k in range(-3, 5):
        fh = BASE * PHI**k
        if fv_d[0] < fh < fv_d[-1]:
            fi_h = np.argmin(np.abs(fv_d - fh))
            phi_t.extend(list(tv_d) + [None])
            phi_f.extend([fh] * len(tv_d) + [None])
            phi_z.extend(list(Sl_d[fi_h, :] + 0.04) + [None])

    fig.add_trace(go.Scatter3d(
        x=phi_t, y=phi_f, z=phi_z,
        mode='lines',
        line=dict(color='#AA88FF', width=2, dash='dot'),
        name=f'φ-Harmônicos · {nome}',
        hoverinfo='skip',
        visible=vis,
    ))

    vis_map[amb_i] = list(range(trace_idx, trace_idx + N_TRACES))
    trace_idx += N_TRACES

total_traces = trace_idx

# ─── Botões de navegação ──────────────────────────────────────────────────────
buttons = []
for amb_i, (nome, _, cor, _, cam_eye, _mode) in enumerate(AMBIENTES):
    vis = [False] * total_traces
    for idx in vis_map[amb_i]: vis[idx] = True
    buttons.append(dict(
        label=nome,
        method='update',
        args=[
            {'visible': vis},
            {
                'title.text': (
                    f'<b>Scanner Topográfico Interativo</b>  ·  {nome}<br>'
                    f'<span style="font-size:11px;color:#00FF88">'
                    f'θ_R={np.degrees(THETA_R):.2f}°  ·  '
                    f'φ={PHI:.7f}  ·  α={ALPHA:.6f}  ·  SEAL={SEAL:.6f}'
                    f'</span>'
                ),
                'scene.camera': {'eye': cam_eye},
            }
        ]
    ))

# ─── Layout ───────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            '<b>Scanner Topográfico Interativo</b>  ·  Base — Quadrada 880Hz<br>'
            f'<span style="font-size:11px;color:#00FF88">'
            f'θ_R={np.degrees(THETA_R):.2f}°  ·  '
            f'φ={PHI:.7f}  ·  α={ALPHA:.6f}  ·  SEAL={SEAL:.6f}'
            f'</span>'
        ),
        font=dict(color='white', size=13),
        x=0.5, xanchor='center',
    ),
    paper_bgcolor='#030308',
    scene=dict(
        bgcolor='#070712',
        xaxis=dict(
            title=dict(text='Tempo (s)',        font=dict(color='#AAAAAA', size=10)),
            tickfont=dict(color='#555566', size=8),
            gridcolor='#111122', showbackground=True,
            backgroundcolor='#08080F', zerolinecolor='#1A1A2A',
        ),
        yaxis=dict(
            title=dict(text='Frequência (Hz)',  font=dict(color='#AAAAAA', size=10)),
            tickfont=dict(color='#555566', size=8),
            gridcolor='#111122', showbackground=True,
            backgroundcolor='#08080F', zerolinecolor='#1A1A2A',
        ),
        zaxis=dict(
            title=dict(text='log(Energia)',     font=dict(color='#AAAAAA', size=10)),
            tickfont=dict(color='#555566', size=8),
            gridcolor='#111122', showbackground=True,
            backgroundcolor='#08080F', zerolinecolor='#1A1A2A',
            range=[-0.6, None],
        ),
        camera=dict(eye=dict(x=1.9, y=-1.9, z=0.75)),
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
    margin=dict(l=0, r=0, t=125, b=0),
    height=800,
)

# ─── Exportar ─────────────────────────────────────────────────────────────────
out = 'scanner_topografico_interativo.html'
fig.write_html(
    out,
    include_plotlyjs='cdn',
    config={
        'displayModeBar': True,
        'scrollZoom': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['orbitRotation'],
    }
)
print(f"\nSalvo: {out}")
print("\nPara visualizar no Colab:")
print("  from IPython.display import IFrame")
print(f"  display(IFrame('{out}', 1300, 820))")

try:
    fig.show()
except Exception:
    pass
