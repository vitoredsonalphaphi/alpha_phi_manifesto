# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Scanner_Fiel.py
Vitor Edson Delavi · Florianópolis · 2026

SCANNER TOPOGRÁFICO — Representação Fiel do Sinal no Espaço

Objetivo: mostrar a forma real do EcoBIP no espaço tempo-frequência,
com resolução adequada ao conteúdo do sinal.

O que estava inadequado na versão animada:
  WIN=162 → resolução 272Hz/bin → 80 bins cobrindo 0–21kHz
  → toda a energia do EcoBIP comprimida nos primeiros 3 bins
  → topografia achatada, infiel à estrutura real

O que está corrigido aqui:
  WIN=512 → resolução 86Hz/bin → foco em 0–5500Hz (64 bins)
  → EcoBIP a 880Hz claramente em bin ~10
  → harmônicos em bins ~30 (2640Hz), ~51 (4400Hz)
  → sidebands FM_φ visíveis em torno de cada harmônico
  → topografia fiel: montanhas nos harmônicos, vales no ruído

Grade R (θ_R = 63.43°) como referência de estrutura canônica.
Harmônicos φ marcados como linhas douradas.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import stft as scipy_stft

# ── Constantes ─────────────────────────────────────────────────────────────────
PHI      = (1 + np.sqrt(5)) / 2
ALPHA    = 1 / 137.035999
ALPHA_OP = 1 / 3
SEAL     = 1 / PHI
THETA_R  = np.arctan(2)          # 63.43°
SR       = 44100
BASE     = 880                   # frequência fundamental

COLORSCALE = [
    [0.00, 'rgb(0,0,20)'],
    [0.15, 'rgb(5,20,80)'],
    [0.40, 'rgb(80,50,0)'],
    [0.65, 'rgb(180,120,0)'],
    [0.85, 'rgb(230,190,20)'],
    [1.00, 'rgb(255,248,100)'],
]

# ── Sinal EcoBIP (0.5 s — riqueza temporal adequada) ───────────────────────────
DUR = 0.5
N   = int(SR * DUR)
t   = np.linspace(0, DUR, N, endpoint=False)
qd  = np.sign(np.sin(2 * np.pi * BASE * t))
fm  = np.sin(2 * np.pi * BASE * t + PHI * np.sin(2 * np.pi * (BASE/4) * t))
sig = (1 - ALPHA_OP) * qd + ALPHA_OP * fm

# ── STFT com resolução adequada ────────────────────────────────────────────────
WIN    = 512          # resolução: SR/WIN = 86 Hz por bin
HOP    = WIN // 4     # 128 amostras → 2.9ms por frame

f_full, tv_full, Zxx_full = scipy_stft(
    sig, fs=SR, window='hann', nperseg=WIN, noverlap=WIN - HOP
)

# ── Foco em 0–5500Hz onde o EcoBIP tem conteúdo real ──────────────────────────
F_MAX   = 5500                              # Hz máximo exibido
BIN_MAX = int(F_MAX / (SR / WIN)) + 1      # bin correspondente
f_ax    = f_full[:BIN_MAX]                 # eixo de frequência
logE    = np.log1p(np.abs(Zxx_full[:BIN_MAX]) * 500)  # energia

n_f = len(f_ax)
n_t = logE.shape[1]

print(f"EcoBIP: {DUR}s  SR={SR}Hz  N={N}")
print(f"STFT: WIN={WIN}  HOP={HOP}  Resolução={SR/WIN:.1f}Hz/bin")
print(f"Faixa exibida: 0–{F_MAX}Hz  ({n_f} bins × {n_t} frames)")
print()

# Localização dos harmônicos
for k in range(1, 8, 2):   # ímpares: 1, 3, 5, 7
    fh = BASE * k
    if fh <= F_MAX:
        bh = int(fh / (SR / WIN))
        print(f"  Harmônico {k}×: {fh}Hz  → bin {bh}  logE={logE[bh].mean():.3f}")

# ── Grade de exibição — downsample para superfície suave ──────────────────────
N_DISP_F = 80    # pontos na dimensão de frequência
N_DISP_T = 100   # pontos na dimensão de tempo

idx_f = np.linspace(0, n_f - 1, N_DISP_F).astype(int)
idx_t = np.linspace(0, n_t - 1, N_DISP_T).astype(int)

f_disp  = f_ax[idx_f]
tv_disp = tv_full[idx_t]
Z_disp  = logE[np.ix_(idx_f, idx_t)]

T_grid, F_grid = np.meshgrid(tv_disp, f_disp)

print(f"\nGrade de exibição: {N_DISP_F}×{N_DISP_T} pontos")
print(f"logE  min={Z_disp.min():.3f}  max={Z_disp.max():.3f}")

# ── Grade R — linha de referência no espaço fiel ──────────────────────────────
# θ_R = 63.43° no plano tempo-frequência normalizado.
# Em unidades reais: para cada Δt no eixo de tempo,
# Δf = tan(θ_R) × (f_range/t_range) × Δt

t_range = tv_disp[-1] - tv_disp[0]
f_range = f_disp[-1]  - f_disp[0]
t_c = tv_disp[N_DISP_T // 2]
f_c = f_disp[N_DISP_F // 2]

dt_gr = t_range * 0.35
df_gr = dt_gr * np.tan(THETA_R) * (f_range / t_range)

t_gr  = [t_c - dt_gr, t_c + dt_gr]
f_gr  = [f_c - df_gr, f_c + df_gr]
f_gr  = [np.clip(v, f_disp[0], f_disp[-1]) for v in f_gr]
z_gr  = [Z_disp.max() * 1.08] * 2   # acima da superfície

# ── Harmônicos φ como linhas horizontais ──────────────────────────────────────
harmonicos = []
for k in range(1, 8, 2):
    fh = BASE * k
    if fh <= F_MAX:
        harmonicos.append(fh)

# ── Superfície 3D principal ────────────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Surface(
    x=T_grid, y=F_grid, z=Z_disp,
    colorscale=COLORSCALE,
    showscale=True,
    colorbar=dict(title='logE', thickness=14, len=0.65,
                  tickfont=dict(size=11)),
    lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5,
                  specular=0.3, fresnel=0.2),
    lightposition=dict(x=100, y=200, z=300),
    name='EcoBIP'
))

# Grade R
fig.add_trace(go.Scatter3d(
    x=t_gr, y=f_gr, z=z_gr,
    mode='lines',
    line=dict(color='lime', width=6),
    name=f'Grade R  θ={np.degrees(THETA_R):.2f}°'
))

# Harmônicos φ como linhas verticais (planos de frequência fixa)
cores_harm = ['rgba(255,200,0,0.7)', 'rgba(255,160,0,0.5)',
              'rgba(255,120,0,0.4)', 'rgba(255,80,0,0.3)']
for fh, cor in zip(harmonicos, cores_harm):
    n_harm = BASE // BASE  # só para calcular múltiplo
    k_harm = round(fh / BASE)
    # linha ao longo do tempo nesta frequência
    z_harm = logE[int(fh / (SR / WIN)), idx_t]
    fig.add_trace(go.Scatter3d(
        x=tv_disp, y=[fh] * N_DISP_T, z=z_harm,
        mode='lines',
        line=dict(color=cor, width=3),
        name=f'{fh:.0f}Hz  ({k_harm}×)',
        showlegend=True
    ))

fig.update_layout(
    title=dict(
        text=f'Scanner Topográfico — EcoBIP Fiel · 0–{F_MAX}Hz · WIN={WIN} · θ_R={np.degrees(THETA_R):.2f}°',
        font=dict(size=14)
    ),
    template='plotly_dark',
    height=700,
    scene=dict(
        xaxis=dict(title='Tempo (s)', showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(title='Frequência (Hz)', showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        zaxis=dict(title='Energia log', showgrid=True, gridcolor='rgba(255,255,255,0.08)'),
        camera=dict(eye=dict(x=1.8, y=-1.4, z=0.8)),
        bgcolor='rgb(4,4,14)',
        aspectmode='manual',
        aspectratio=dict(x=2.0, y=1.0, z=0.7),
    ),
    legend=dict(x=0.01, y=0.95, font=dict(size=11)),
    margin=dict(l=0, r=0, t=50, b=0)
)

fig.show()

# ── Espectrograma 2D de referência ─────────────────────────────────────────────
fig2 = go.Figure()

fig2.add_trace(go.Heatmap(
    x=tv_disp, y=f_disp, z=Z_disp,
    colorscale=COLORSCALE,
    colorbar=dict(title='logE'),
    name='Espectrograma'
))

# Grade R no 2D
fig2.add_trace(go.Scatter(
    x=t_gr, y=f_gr,
    mode='lines', line=dict(color='lime', width=2),
    name=f'Grade R {np.degrees(THETA_R):.1f}°'
))

# Harmônicos no 2D
for fh, cor in zip(harmonicos, cores_harm):
    fig2.add_hline(
        y=fh,
        line_color=cor.replace('0.7','1').replace('0.5','0.8').replace('0.4','0.7').replace('0.3','0.6'),
        line_dash='dot', line_width=1.5,
        annotation_text=f'{fh:.0f}Hz',
        annotation_font_size=10,
        annotation_font_color='orange'
    )

fig2.update_layout(
    title=f'Espectrograma EcoBIP — 0–{F_MAX}Hz · Referência plana',
    xaxis_title='Tempo (s)', yaxis_title='Frequência (Hz)',
    template='plotly_dark', height=400
)
fig2.show()

print(f"\nScanner fiel — pronto.")
print(f"Harmônicos visíveis: {harmonicos}")
print(f"Grade R: t={[round(v,4) for v in t_gr]}s  f={[round(v,1) for v in f_gr]}Hz")
