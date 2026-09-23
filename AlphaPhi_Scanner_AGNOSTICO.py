"""
AlphaPhi_Scanner_AGNOSTICO.py
Scanner Topográfico Agnóstico

Aplica sobre QUALQUER sinal 1D numpy.
Sem dependências de ECO-BIP, phantom, Grade R ou PHI.
Cole numa única célula no Google Colab e execute.

© Vitor Edson Delavi · Florianópolis · setembro 2026
"""

import numpy as np
import plotly.graph_objects as go
from scipy.signal import stft as scipy_stft

# ══════════════════════════════════════════════════════════════════
#  PARÂMETROS — ajuste conforme o sinal
# ══════════════════════════════════════════════════════════════════
FS    = 44100     # taxa de amostragem (Hz)
F_MAX = 4000      # frequência máxima a exibir (Hz)
N_WIN = 2048      # janela STFT (amostras)
HOP   = N_WIN//4  # salto entre janelas

# ══════════════════════════════════════════════════════════════════
#  SINAL — substitua aqui por qualquer array numpy 1D
# ══════════════════════════════════════════════════════════════════
t         = np.linspace(0, 2.0, int(FS * 2), endpoint=False)
SINAL     = (np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*880*t)).astype(np.float64)
TITULO    = "Calibração"
SUBTITULO = "440 Hz + 880 Hz · agnóstico"

# ══════════════════════════════════════════════════════════════════
#  FUNÇÃO SCANNER
# ══════════════════════════════════════════════════════════════════
def scanner_agnostico(sig, titulo="Sinal", subtitulo="",
                      fs=FS, f_max=F_MAX, n_win=N_WIN, hop=HOP):
    """
    Scanner topográfico agnóstico.

    Parâmetros
    ----------
    sig      : np.ndarray  — sinal 1D
    titulo   : str
    subtitulo: str
    fs       : int   — taxa de amostragem (Hz)
    f_max    : float — frequência máxima a exibir (Hz)
    n_win    : int   — janela STFT
    hop      : int   — salto entre janelas
    """
    # STFT
    f_ax, t_ax, Zxx = scipy_stft(
        sig, fs=fs, nperseg=n_win, noverlap=n_win - hop, window='hann'
    )
    mask   = f_ax <= f_max
    f_plot = f_ax[mask]
    Z      = np.abs(Zxx[mask])
    Z_db   = 20 * np.log10(Z + 1e-10)

    # Grids
    T, F_g = np.meshgrid(t_ax, f_plot)

    # Subsampling para performance (plotly Surface ≤ 200×300 pontos ideal)
    sf = max(1, len(f_plot) // 150)
    st = max(1, len(t_ax)   // 250)
    T_s   = T[::sf, ::st]
    F_s   = F_g[::sf, ::st]
    Z_s   = Z_db[::sf, ::st]

    fig = go.Figure()

    # [SINAL] Campo espectral
    fig.add_trace(go.Surface(
        x=T_s, y=F_s, z=Z_s,
        colorscale='Viridis',
        opacity=0.93,
        showscale=True,
        colorbar=dict(title='dB', x=1.02, len=0.8, thickness=14),
        name='[SINAL] Campo espectral'
    ))

    # [COMPOSIÇÃO] Piso de referência visual — z = 35% abaixo do mínimo
    z_floor = float(Z_db.min()) - 0.35 * float(Z_db.max() - Z_db.min())
    sf2 = max(1, len(f_plot) // 30)
    st2 = max(1, len(t_ax)   // 50)
    fig.add_trace(go.Surface(
        x=T[::sf2, ::st2],
        y=F_g[::sf2, ::st2],
        z=np.full(T[::sf2, ::st2].shape, z_floor),
        colorscale='Plasma',
        opacity=0.18,
        showscale=False,
        name='[COMPOSIÇÃO] Piso — referência visual'
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{titulo}</b><br><sup>{subtitulo}</sup>",
            x=0.5
        ),
        scene=dict(
            xaxis_title='Tempo (s)',
            yaxis_title='Frequência (Hz)',
            zaxis_title='Amplitude (dB)',
            camera=dict(eye=dict(x=1.5, y=-1.7, z=1.0))
        ),
        width=1100, height=740,
        template='plotly_dark',
        margin=dict(l=0, r=0, t=60, b=0)
    )

    fig.show()
    dur = len(sig) / fs
    print(f"\n  Sinal : {len(sig):,} amostras · {dur:.2f}s · {fs} Hz")
    print(f"  STFT  : {len(t_ax)} frames × {len(f_plot)} bins · 0–{f_max:.0f} Hz")
    print(f"  Janela: N_WIN={n_win} · HOP={hop} · resolução={fs/n_win:.1f} Hz/bin")
    print(f"  Amp   : [{Z_db.min():.1f}, {Z_db.max():.1f}] dB")

# ══════════════════════════════════════════════════════════════════
#  EXECUTAR
# ══════════════════════════════════════════════════════════════════
scanner_agnostico(SINAL, TITULO, SUBTITULO)
