# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Scanner_Animado.py
Vitor Edson Delavi · Florianópolis · 2026

SCANNER TOPOGRÁFICO ANIMADO — Visualização do Fluxo

Janela de análise φ-escalada desliza sobre o sinal EcoBIP.
Cada frame da animação = uma posição da janela no tempo.

O que se observa em movimento:
  - Picos harmônicos da onda quadrada (880Hz, 2640Hz, 4400Hz...)
  - Sidebands FM_φ se expandindo e contraindo em torno da portadora
  - Grade R (θ_R = 63.43°) como referência fixa no campo
  - Fluxo de energia: de onde vem, para onde vai, como oscila

Dois modos de visualização:
  1. Superfície 3D animada — janela deslizante sobre o EcoBIP
  2. Espectrograma 2D completo — visão estática de referência

Analogia: o scanner estático é a foto. O animado é o filme.
O gradiente visto no filme revela a mecânica do fluxo.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import stft as scipy_stft

# ── Constantes ─────────────────────────────────────────────────────────────────
PHI      = (1 + np.sqrt(5)) / 2    # 1.6180339887
ALPHA    = 1 / 137.035999
ALPHA_OP = 1 / 3
SEAL     = 1 / PHI
THETA_R  = np.arctan(2)            # 63.43° em radianos
SR       = 44100

# ── Sinal EcoBIP (0.25 s — gerenciável no Colab) ───────────────────────────────
N   = int(SR * 0.25)
t   = np.linspace(0, 0.25, N, endpoint=False)
qd  = np.sign(np.sin(2 * np.pi * 880 * t))
fm  = np.sin(2 * np.pi * 880 * t + PHI * np.sin(2 * np.pi * 220 * t))
sig = (1 - ALPHA_OP) * qd + ALPHA_OP * fm

# ── Parâmetros STFT φ-escalados ────────────────────────────────────────────────
WIN    = int(SR / 880 * 2 * PHI)   # janela φ-escalada ≈ 102 → clampada
WIN    = max(128, min(WIN, 512))
HOP    = WIN // 4
N_BINS = 80                        # bins de frequência exibidos
N_ANIM = 28                        # frames da animação
COLORSCALE = [
    [0.0,  'rgb(0,0,25)'],
    [0.25, 'rgb(10,30,100)'],
    [0.55, 'rgb(160,100,0)'],
    [0.80, 'rgb(230,180,20)'],
    [1.0,  'rgb(255,240,80)'],
]

print(f"EcoBIP: {N} amostras  SR={SR}Hz  Duração=0.25s")
print(f"STFT: WIN={WIN}  HOP={HOP}  Bins={N_BINS}")
print(f"θ_R = {np.degrees(THETA_R):.4f}°")

# ── STFT completa ──────────────────────────────────────────────────────────────
f_full, tv_full, Zxx_full = scipy_stft(
    sig, fs=SR, window='hann', nperseg=WIN, noverlap=WIN - HOP
)
f_ax  = f_full[:N_BINS]                           # eixo de frequência
logE  = np.log1p(np.abs(Zxx_full[:N_BINS]) * 200) # energia log-comprimida
n_t   = logE.shape[1]                             # total de frames temporais

print(f"Espectrograma: {N_BINS} freq × {n_t} frames temporais")

# ── Janelas para animação ──────────────────────────────────────────────────────
W_JANELA = max(20, n_t // 4)   # tamanho de cada janela de análise
PASSO    = max(1, (n_t - W_JANELA) // (N_ANIM - 1))

starts = [min(i * PASSO, n_t - W_JANELA) for i in range(N_ANIM)]
starts = sorted(set(starts))
N_ANIM = len(starts)
print(f"Animação: {N_ANIM} frames  |  janela={W_JANELA} frames  |  passo={PASSO}")

# ── Grade R — linha de referência ──────────────────────────────────────────────
def linha_grade_r(t_janela, f_ax):
    """
    Linha Grade R em coordenadas (tempo, frequência).
    θ_R = arctan(2) → razão f/t (normalizada) = 2.
    Linha passa pelo centro da janela com inclinação tan(θ_R).
    """
    t_c = t_janela[len(t_janela) // 2]
    f_c = f_ax[len(f_ax) // 2]
    dt  = t_janela[-1] - t_janela[0]
    df  = dt * np.tan(THETA_R) * (f_ax[-1] - f_ax[0]) / (t_janela[-1] - t_janela[0] + 1e-9)
    t_gr = [t_janela[0],  t_janela[-1]]
    f_gr = [f_c - df / 2, f_c + df / 2]
    f_gr = [np.clip(v, f_ax[0], f_ax[-1]) for v in f_gr]
    z_gr = [logE[:, starts[0]:starts[0]+W_JANELA].max() * 1.05] * 2
    return t_gr, f_gr, z_gr

# ── Construir frames da animação ───────────────────────────────────────────────
frames = []
for k, s in enumerate(starts):
    e  = min(s + W_JANELA, n_t)
    Zf = logE[:, s:e]
    Tf = tv_full[s:e]

    # grade corrigida para esta janela
    t_gr, f_gr, z_gr = linha_grade_r(Tf, f_ax)

    T_grid, F_grid = np.meshgrid(Tf, f_ax)

    frame_data = [
        go.Surface(
            x=T_grid, y=F_grid, z=Zf,
            colorscale=COLORSCALE,
            showscale=False,
            name='EcoBIP'
        ),
        go.Scatter3d(
            x=t_gr, y=f_gr, z=z_gr,
            mode='lines',
            line=dict(color='lime', width=5),
            name=f'Grade R {np.degrees(THETA_R):.1f}°'
        ),
    ]
    frames.append(go.Frame(data=frame_data, name=str(k),
                           layout=go.Layout(title_text=f'EcoBIP — janela {k+1}/{N_ANIM} '
                                                       f'| t={Tf[0]*1000:.1f}–{Tf[-1]*1000:.1f} ms')))

# ── Figura inicial (primeiro frame) ───────────────────────────────────────────
s0 = starts[0]
Z0 = logE[:, s0:s0+W_JANELA]
T0 = tv_full[s0:s0+W_JANELA]
T0g, F0g = np.meshgrid(T0, f_ax)
t_gr0, f_gr0, z_gr0 = linha_grade_r(T0, f_ax)

fig_anim = go.Figure(
    data=[
        go.Surface(x=T0g, y=F0g, z=Z0,
                   colorscale=COLORSCALE, showscale=True,
                   colorbar=dict(title='logE', thickness=12, len=0.6)),
        go.Scatter3d(x=t_gr0, y=f_gr0, z=z_gr0,
                     mode='lines', line=dict(color='lime', width=5),
                     name=f'Grade R {np.degrees(THETA_R):.1f}°'),
    ],
    frames=frames
)

fig_anim.update_layout(
    title=dict(text='Scanner Topográfico Animado — EcoBIP · Fluxo em Movimento',
               font=dict(size=15)),
    template='plotly_dark',
    height=680,
    scene=dict(
        xaxis_title='Tempo (s)',
        yaxis_title='Frequência (Hz)',
        zaxis_title='Energia log',
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        bgcolor='rgb(5,5,15)',
    ),
    updatemenus=[dict(
        type='buttons', showactive=False,
        y=0.02, x=0.5, xanchor='center',
        buttons=[
            dict(label='▶ Play',
                 method='animate',
                 args=[None, dict(frame=dict(duration=220, redraw=True),
                                 fromcurrent=True, transition=dict(duration=80))]),
            dict(label='⏸ Pausa',
                 method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                   mode='immediate', transition=dict(duration=0))]),
        ]
    )],
    sliders=[dict(
        steps=[dict(method='animate', args=[[str(k)],
                    dict(mode='immediate', frame=dict(duration=220, redraw=True),
                         transition=dict(duration=80))],
                    label=f'{tv_full[s]*1000:.0f}ms')
               for k, s in enumerate(starts)],
        active=0,
        y=0.0, x=0.05, len=0.9,
        currentvalue=dict(prefix='Posição: ', font=dict(size=12)),
        transition=dict(duration=80),
    )]
)

fig_anim.show()

# ── Espectrograma 2D completo (referência estática) ────────────────────────────
fig_2d = go.Figure()

fig_2d.add_trace(go.Heatmap(
    x=tv_full,
    y=f_ax,
    z=logE,
    colorscale=COLORSCALE,
    colorbar=dict(title='logE'),
    name='Espectrograma'
))

# Grade R como linha sobre o espectrograma 2D
t_mid = tv_full[n_t // 2]
f_mid = f_ax[N_BINS // 2]
dt_span = (tv_full[-1] - tv_full[0]) * 0.3
df_span = dt_span * np.tan(THETA_R) * (f_ax[-1] - f_ax[0]) / (tv_full[-1] - tv_full[0])
fig_2d.add_trace(go.Scatter(
    x=[t_mid - dt_span, t_mid + dt_span],
    y=[f_mid - df_span, f_mid + df_span],
    mode='lines', line=dict(color='lime', width=2, dash='dot'),
    name=f'Grade R {np.degrees(THETA_R):.1f}°'
))

# Harmônicos φ
for n in range(1, 7, 2):   # ímpares: 880, 2640, 4400...
    fh = 880 * n
    if fh <= f_ax[-1]:
        fig_2d.add_hline(y=fh, line_color='rgba(255,200,0,0.35)',
                         line_dash='dot', line_width=1,
                         annotation_text=f'{fh}Hz', annotation_font_size=9)

fig_2d.update_layout(
    title='Espectrograma EcoBIP — Referência Estática',
    xaxis_title='Tempo (s)', yaxis_title='Frequência (Hz)',
    template='plotly_dark', height=420
)
fig_2d.show()

print("\nScanner animado — pronto.")
print(f"  {N_ANIM} frames  |  janela={W_JANELA} frames  |  Grade R = {np.degrees(THETA_R):.2f}°")
print("Próximo: Scanner de Gradiente (durante treinamento — Estágio III)")
