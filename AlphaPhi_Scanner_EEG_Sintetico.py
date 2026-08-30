# AlphaPhi_Scanner_EEG_Sintetico.py
# Scanner Topográfico — Substrato EEG Sintético
# Testa formação da Grade R em domínio neurofisiológico (0–100 Hz)
# θ_R = arctan(2) permanece invariante — só a escala do substrato muda
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

# ─── Constantes irrevogáveis ──────────────────────────────────────────────────
PHI     = 1.6180339887
ALPHA   = 1 / 137.035999
SEAL    = 1 / PHI
THETA_R = np.arctan(2.0)   # 63.43° — Grade Romboédrica — invariante de substrato

# ─── Parâmetros do substrato EEG ──────────────────────────────────────────────
SR   = 1000          # Hz — taxa de amostragem EEG padrão
DUR  = 30.0          # s  — duração longa para capturar dinâmica EEG
BASE = 10.0          # Hz — banda alpha como frequência de referência
F_MAX = 100.0        # Hz — limite espectral EEG
N_SIG = int(SR * DUR)
t     = np.linspace(0, DUR, N_SIG, endpoint=False)

print(f"φ={PHI}  α={ALPHA:.8f}  θ_R={np.degrees(THETA_R):.2f}°  SEAL={SEAL:.8f}")
print(f"Substrato EEG: SR={SR}Hz  DUR={DUR}s  BASE={BASE}Hz  f_max={F_MAX}Hz")

def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)

# ─── Gerador de EEG sintético ─────────────────────────────────────────────────
def gerar_eeg():
    """
    EEG sintético realístico — soma de bandas com amplitudes e fases típicas.
    Amplitudes em μV (normalizadas ao final):
      delta  0.5–4 Hz  — sono profundo / oscilação lenta
      theta  4–8 Hz   — estado de repouso / meditação
      alpha  8–13 Hz  — vigília relaxada (banda dominante)
      beta   13–30 Hz — atenção ativa
      gamma  30–80 Hz — atividade cognitiva de alta frequência
    """
    rng = np.random.default_rng(2026)  # semente fixa para reprodutibilidade

    def banda(f, amp, fase):
        return amp * np.sin(2 * np.pi * f * t + fase)

    delta  = (banda(1.5, 35, rng.uniform(0, 2*np.pi)) +
              banda(3.0, 20, rng.uniform(0, 2*np.pi)))
    theta  = (banda(5.0, 15, rng.uniform(0, 2*np.pi)) +
              banda(7.0, 12, rng.uniform(0, 2*np.pi)))
    alpha  = (banda(9.0,  25, rng.uniform(0, 2*np.pi)) +
              banda(10.0, 30, rng.uniform(0, 2*np.pi)) +
              banda(11.5, 20, rng.uniform(0, 2*np.pi)))
    beta   = (banda(15.0, 8, rng.uniform(0, 2*np.pi)) +
              banda(20.0, 6, rng.uniform(0, 2*np.pi)) +
              banda(25.0, 5, rng.uniform(0, 2*np.pi)))
    gamma  = (banda(40.0, 3, rng.uniform(0, 2*np.pi)) +
              banda(60.0, 2, rng.uniform(0, 2*np.pi)))
    ruido  = rng.normal(0, 4, N_SIG)   # ruído de eletrodo (~4 μV)

    return _norm(delta + theta + alpha + beta + gamma + ruido)

# ─── semear() adaptado para substrato EEG (BASE=10Hz) ─────────────────────────
def semear_eeg(x):
    """
    Injeta semente α-φ nos harmônicos φ^k a partir da banda alpha (BASE=10Hz).
    Mesma lógica do semear() de áudio — só a frequência de referência muda.
    """
    F   = np.fft.rfft(x)
    mag = np.abs(F)
    # bin da banda alpha (~10Hz) como âncora
    b0  = int(BASE * DUR)   # bin correspondente a BASE Hz com resolução 1/DUR
    b0  = max(1, min(b0, len(F)-1))
    ref = mag[b0]
    for k in range(1, 7):
        w  = ALPHA * SEAL**k * ref
        ph = np.exp(1j * PHI * k)
        for bk in [int(round(b0 * PHI**k)), int(round(b0 / PHI**k))]:
            if 0 < bk < len(F):
                F[bk] += w * ph
    return _norm(np.fft.irfft(F, n=len(x)))

# ─── STFT φ-escalado para EEG ─────────────────────────────────────────────────
def stft_phi_eeg(x):
    win = int(SR / BASE * 2 * PHI)
    win = max(256, min(win, 8192))
    hop = win // 4
    f, tv, Zxx = stft(x, fs=SR, window='hann', nperseg=win, noverlap=win - hop)
    return f, tv, np.abs(Zxx)**2

# ─── Vértices Grade R por resultado ──────────────────────────────────────────
def grade_r_vertices(fv, tv, Sl, gradS, theta=THETA_R, margin=0.07):
    thr_sl = Sl.mean() + 0.45 * Sl.std()
    thr_gs = 0.12
    f_exp  = np.tan(theta) * (tv - tv[0]) / (tv[-1] - tv[0] + 1e-9) * fv[-1]
    f_exp  = np.clip(f_exp, fv[0], fv[-1])
    vx, vy, vz, vinfo = [], [], [], []
    for ti_i, (ti, fe) in enumerate(zip(tv, f_exp)):
        bw    = fe * margin + 1.0
        fi_lo = int(np.searchsorted(fv, fe - bw))
        fi_hi = int(np.searchsorted(fv, fe + bw))
        for fi_i in range(fi_lo, min(fi_hi, len(fv))):
            if gradS[fi_i, ti_i] > thr_gs and Sl[fi_i, ti_i] > thr_sl:
                vx.append(float(ti))
                vy.append(float(fv[fi_i]))
                vz.append(float(Sl[fi_i, ti_i]) + 0.14)
                vinfo.append(
                    f't={ti:.2f}s  f={fv[fi_i]:.1f}Hz  '
                    f'log(E)={Sl[fi_i,ti_i]:.3f}  ∇S={gradS[fi_i,ti_i]:.3f}'
                )
    return vx, vy, vz, vinfo

# ─── Gerar sinais ─────────────────────────────────────────────────────────────
print("\nGerando sinais EEG...")
eeg_bruto   = gerar_eeg()
eeg_semeado = semear_eeg(eeg_bruto)
print("  EEG bruto e EEG semeado α-φ prontos.")

# métricas rápidas para print
def entropia_espectral(x):
    freqs = np.fft.rfftfreq(len(x), 1/SR)
    psd   = np.abs(np.fft.rfft(x))**2
    mask  = freqs <= F_MAX
    p     = psd[mask] / (psd[mask].sum() + 1e-12)
    return -np.sum(p * np.log2(p + 1e-12))

def eso(x):
    freqs = np.fft.rfftfreq(len(x), 1/SR)
    psd   = np.abs(np.fft.rfft(x))**2
    sub   = psd[freqs < BASE].sum()
    tot   = psd[freqs <= F_MAX].sum()
    return sub / (tot + 1e-12)

print(f"\n  H(EEG bruto)    = {entropia_espectral(eeg_bruto):.4f}")
print(f"  H(EEG semeado)  = {entropia_espectral(eeg_semeado):.4f}")
print(f"  ESO(EEG bruto)  = {eso(eeg_bruto):.4f}")
print(f"  ESO(EEG semeado)= {eso(eeg_semeado):.4f}")

# ─── Ambientes ────────────────────────────────────────────────────────────────
AMBIENTES = [
    ('EEG Bruto',        eeg_bruto,   '#44AAFF', 'Blues',
     'EEG sintético — delta+theta+alpha+beta+gamma+ruído · sem semente α-φ'),
    ('EEG Semeado α-φ',  eeg_semeado, '#00FFAA', 'viridis',
     f'EEG com semente α-φ · BASE={BASE}Hz · ALPHA={ALPHA:.6f} · SEAL={SEAL:.6f}'),
]

N_TRACES = 4
fig       = go.Figure()
vis_map   = {}
trace_idx = 0

print("\nConstruindo ambientes Plotly...")

for amb_i, (nome, x, cor, cmap, desc) in enumerate(AMBIENTES):
    print(f"  [{amb_i+1}/2] {nome}...")
    vis = (amb_i == 0)

    freqs, times, S = stft_phi_eeg(x)
    fmask = freqs <= F_MAX
    fv, Sv = freqs[fmask], S[fmask]
    Sl     = np.log1p(Sv * 100)
    gradS  = Sl - gaussian_filter(Sl, sigma=3.0)

    sf = max(1, len(fv)   // 120)
    st = max(1, len(times) // 100)
    fv_d  = fv[::sf];     tv_d  = times[::st]
    Sl_d  = Sl[::sf, ::st]
    gS_d  = gradS[::sf, ::st]
    T_g, F_g = np.meshgrid(tv_d, fv_d)

    # ── 1. Superfície espectral ───────────────────────────────────────────────
    fig.add_trace(go.Surface(
        x=T_g, y=F_g, z=Sl_d,
        colorscale=cmap,
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
            'f = %{y:.1f} Hz<br>'
            'log(E) = %{z:.3f}<extra></extra>'
        ),
        visible=vis,
    ))

    # ── 2. Linha Grade R θ_R ─────────────────────────────────────────────────
    t_r = np.linspace(tv_d[0], tv_d[-1], 80)
    f_r = np.tan(THETA_R) * (t_r - tv_d[0]) / (tv_d[-1] - tv_d[0] + 1e-9) * fv_d[-1]
    f_r = np.clip(f_r, fv_d[0], fv_d[-1])
    z_r = []
    for ti, fi in zip(t_r, f_r):
        z_r.append(float(Sl_d[np.argmin(np.abs(fv_d - fi)),
                              np.argmin(np.abs(tv_d - ti))]) + 0.09)

    fig.add_trace(go.Scatter3d(
        x=t_r, y=f_r, z=np.array(z_r),
        mode='lines',
        line=dict(color='#00FF88', width=5),
        name=f'Grade R θ={np.degrees(THETA_R):.1f}°',
        hovertemplate=(
            f'<b>Grade R  θ={np.degrees(THETA_R):.1f}°</b><br>'
            't=%{x:.2f}s · f=%{y:.1f}Hz<extra></extra>'
        ),
        visible=vis,
    ))

    # ── 3. Vértices Grade R por resultado ◆ ciano ─────────────────────────────
    vx, vy, vz, vinfo = grade_r_vertices(fv_d, tv_d, Sl_d, gS_d)
    n_vertices = len(vx)

    fig.add_trace(go.Scatter3d(
        x=vx, y=vy, z=vz,
        mode='markers',
        marker=dict(size=6, color='#00FFFF', symbol='diamond',
                    line=dict(color='white', width=0.8)),
        name=f'Formação Grade R ({n_vertices} vértices)',
        text=vinfo,
        hovertemplate='<b>Vértice Grade R</b><br>%{text}<extra></extra>',
        visible=vis,
    ))

    # ── 4. Harmônicos φ a partir de BASE (linhas verticais) ──────────────────
    phi_freqs = []
    for k in range(-3, 6):
        f_phi = BASE * PHI**k
        if fv_d[0] < f_phi < fv_d[-1]:
            phi_freqs.append(f_phi)
    zmin = Sl_d.min() - 0.05
    ph_x, ph_y, ph_z = [], [], []
    for fp in phi_freqs:
        for ti in [tv_d[0], tv_d[-1]]:
            ph_x += [ti, ti, None]
            ph_y += [fp, fp, None]
            ph_z += [zmin, zmin + 0.3, None]

    fig.add_trace(go.Scatter3d(
        x=ph_x, y=ph_y, z=ph_z,
        mode='lines',
        line=dict(color='rgba(255,200,50,0.55)', width=2),
        name='Harmônicos φ (BASE=10Hz)',
        hovertemplate='φ-harmônico<extra></extra>',
        visible=vis,
    ))

    # métricas no nome para legenda
    h_val  = entropia_espectral(x)
    es_val = eso(x)
    print(f"     → {n_vertices} vértices Grade R · H={h_val:.4f} · ESO={es_val:.4f}")

    vis_map[amb_i] = list(range(trace_idx, trace_idx + N_TRACES))
    trace_idx += N_TRACES

# ─── Dropdown ─────────────────────────────────────────────────────────────────
buttons = []
for amb_i, (nome, _, cor, _, desc) in enumerate(AMBIENTES):
    visibility = [False] * (N_TRACES * len(AMBIENTES))
    for idx in vis_map[amb_i]:
        visibility[idx] = True

    buttons.append(dict(
        label=nome,
        method='update',
        args=[
            {'visible': visibility},
            {
                'title.text': (
                    f'<b>Scanner EEG Sintético — Substrato Neurofisiológico</b>  ·  {nome}<br>'
                    f'<span style="font-size:10px;color:#AAAAAA">{desc}</span><br>'
                    f'<span style="font-size:10px;color:#00FF88">'
                    f'θ_R={np.degrees(THETA_R):.2f}°  ·  φ={PHI:.7f}  ·  α={ALPHA:.6f}'
                    f'</span>'
                ),
            }
        ],
    ))

# ─── Layout ───────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            '<b>Scanner EEG Sintético — Substrato Neurofisiológico</b>  ·  EEG Bruto<br>'
            '<span style="font-size:10px;color:#AAAAAA">'
            'EEG sintético — delta+theta+alpha+beta+gamma+ruído · sem semente α-φ</span><br>'
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
            title=dict(text='Frequência (Hz)  [0–100 Hz EEG]',
                       font=dict(color='#AAAAAA', size=10)),
            tickfont=dict(color='#888888', size=8),
        ),
        zaxis=dict(
            backgroundcolor='#06060F',
            gridcolor='#1A1A33', zerolinecolor='#333366',
            title=dict(text='log(Energia)', font=dict(color='#AAAAAA', size=10)),
            tickfont=dict(color='#888888', size=8),
        ),
        camera=dict(eye=dict(x=1.8, y=-1.8, z=0.85)),
        aspectmode='manual',
        aspectratio=dict(x=2.2, y=1.4, z=0.80),
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
            '<b>◆ ciano</b> = vértice Grade R por resultado  ·  '
            '<b>━ verde</b> = linha θ_R=63.43°  ·  '
            '<span style="color:#FFCC33">━</span> harmônicos φ (BASE=10Hz)'
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
out = 'scanner_eeg_sintetico.html'
fig.write_html(out, include_plotlyjs='cdn', full_html=True)
print(f"\nSalvo: {out}")
print("\nPara visualizar no Colab:")
print("  from IPython.display import HTML")
print(f"  with open('{out}') as f: display(HTML(f.read()))")
