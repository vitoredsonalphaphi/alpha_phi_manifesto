# AlphaPhi_Scanner_Forensico.py
# Scanner Topográfico Forense — Verificação de Autenticidade Visual
# Derivado de AlphaPhi_Scanner_Topografico_Interativo.py
# Vitor Edson Delavi · Florianópolis · 2026
# © CC BY-NC-ND 4.0
#
# Uso:
#   python AlphaPhi_Scanner_Forensico.py imagem.jpg
#   (sem argumento: gera imagem de teste sintética)
#
# No Colab:
#   !python AlphaPhi_Scanner_Forensico.py imagem.jpg
#   from IPython.display import HTML
#   with open('scanner_forensico_interativo.html') as f:
#       display(HTML(f.read()))

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import stft as scipy_stft
import warnings, sys, os
warnings.filterwarnings('ignore')

try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow', '-q'])
    from PIL import Image

try:
    import plotly.graph_objects as go
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly', '-q'])
    import plotly.graph_objects as go

# ─── Constantes (herdadas do Scanner Topográfico) ─────────────────────────────
PHI     = 1.6180339887
ALPHA   = 1 / 137.035999
SEAL    = 1 / PHI
THETA_R = np.arctan(2.0)   # 63.43°

print(f"φ  = {PHI}")
print(f"α  = {ALPHA:.8f}")
print(f"θ_R = {np.degrees(THETA_R):.2f}°")
print(f"SEAL = {SEAL:.8f}")

# ─── Carregamento de imagem ───────────────────────────────────────────────────
def load_image(path=None):
    """Carrega imagem e converte para array float normalizado [0,1]."""
    if path and os.path.exists(path):
        img = Image.open(path).convert('L')
        print(f"  Imagem: {path} ({img.width}×{img.height}px)")
    else:
        print("  Imagem não fornecida — gerando padrão sintético (Bayer + ruído Poisson)...")
        W = H = 256
        x, y  = np.meshgrid(np.arange(W), np.arange(H))
        base  = 0.5 + 0.3 * np.sin(2*np.pi*x/20) * np.cos(2*np.pi*y/28)
        bayer = 0.04 * (np.cos(np.pi * x) + np.cos(np.pi * y))  # período 2px
        rng   = np.random.default_rng(42)
        noise = rng.poisson(np.clip(base * 200, 1, 255)) / 200.0 - base
        arr   = np.clip(base + bayer + noise * 0.3, 0, 1)
        img   = Image.fromarray((arr * 255).astype(np.uint8))
    return np.array(img, dtype=float) / 255.0

# ─── Análise 1 — STFT Espacial por faixas ────────────────────────────────────
def spatial_stft_map(img, win=64):
    """
    Trata faixas horizontais da imagem como sinais 1D.
    Retorna: (freqs, mapa_energia normalizado shape=(n_freqs, n_times))
    """
    H, W    = img.shape
    hop     = win // 4
    patch_h = max(8, H // 24)
    rows    = []
    freqs_out = None

    for y0 in range(0, H - patch_h + 1, patch_h):
        row = img[y0:y0 + patch_h, :].mean(axis=0)
        f, _, Zxx = scipy_stft(row, nperseg=win, noverlap=win - hop)
        rows.append(np.log1p(np.abs(Zxx)**2 * 100))
        if freqs_out is None:
            freqs_out = f

    E_mean = np.array(rows).mean(axis=0)
    return freqs_out, E_mean / (E_mean.max() + 1e-9)

# ─── Análise 2 — Micro-Cepstro 2D ────────────────────────────────────────────
def micro_cepstrum(img, region=48):
    """
    Cepstro 2D centrado: IFFT2(log|FFT2(I)|²)
    Região de baixa quefrência revela assinaturas periódicas físicas:
    - Câmera real: picos em quefrência 2px (padrão Bayer)
    - IA: sem picos — cepstro suave
    """
    F   = np.fft.fft2(img)
    LP  = np.log(np.abs(F)**2 + 1e-10)
    C   = np.abs(np.fft.ifft2(LP))
    Cs  = np.fft.fftshift(C)
    cy, cx = Cs.shape[0] // 2, Cs.shape[1] // 2
    r   = min(region, cy, cx)
    micro = Cs[cy - r:cy + r, cx - r:cx + r]
    return micro / (micro.max() + 1e-9)

# ─── Análise 3 — Mapa de Ruído de Sensor ─────────────────────────────────────
def noise_energy_map(img, sigma=3):
    """
    Energia local do ruído de alta frequência (img − GaussianBlur).
    Câmera real: distribuição Poisson/Gaussiana espacialmente uniforme.
    IA: ruído artificial com distribuição diferente — frequentemente mais suave.
    """
    noise  = img - gaussian_filter(img, sigma=sigma)
    energy = gaussian_filter(noise**2, sigma=sigma * 2)
    return energy / (energy.max() + 1e-9)

# ─── Análise 4 — Espectro Radial de Potência ─────────────────────────────────
def radial_spectrum(img, n_bins=96):
    """
    Perfil radial médio do espectro de potência 2D.
    Câmera real: decaimento tipo lei de potência 1/f² bem definido.
    IA: decaimento diferente — frequentemente demasiado suave ou estruturado.
    """
    F  = np.fft.fft2(img)
    P  = np.abs(np.fft.fftshift(F))**2
    H, W = P.shape
    cy, cx = H // 2, W // 2
    Y, X  = np.ogrid[:H, :W]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    r_max = min(cy, cx, n_bins)
    radial = np.zeros(r_max)
    counts = np.zeros(r_max)
    mask   = R < r_max
    np.add.at(radial, R[mask], P[mask])
    np.add.at(counts, R[mask], 1)
    radial /= (counts + 1e-10)
    return np.log1p(radial)

# ─── DFI — Digital Fingerprint Index ─────────────────────────────────────────
def compute_dfi(img, micro, radial):
    """
    Digital Fingerprint Index: 0 = provável IA  |  1 = provável câmera real.
    Composição de três indicadores independentes.
    """
    # 1. Bayer Score: pico no cepstro em quefrência ±2px (período Bayer = 2px)
    mid    = micro.shape[0] // 2
    window = micro[mid - 3:mid + 4, mid - 3:mid + 4].copy()
    window[3, :] = 0  # zera linha DC (centro da janela)
    window[:, 3] = 0  # zera coluna DC
    bayer  = float(window.max())

    # 2. Gaussianidade do ruído de alta freq (câmera real: kurtosis excess ≈ 0)
    noise  = (img - gaussian_filter(img, sigma=3)).flatten()
    std    = noise.std()
    if std > 1e-9:
        z             = noise / std
        kurt_excess   = float(np.mean(z**4) - 3.0)
        gauss         = float(np.exp(-abs(kurt_excess) * 0.15))
    else:
        gauss = 0.0

    # 3. Coerência do decaimento espectral (câmera real: lei de potência bem fit)
    r  = np.arange(1, len(radial))
    slope      = float(np.polyfit(np.log(r + 1), radial[1:], 1)[0])
    residuals  = radial[1:] - np.polyval(np.polyfit(np.log(r + 1), radial[1:], 1), np.log(r + 1))
    coherence  = float(np.exp(-np.std(residuals) * 2.0))

    dfi = float(np.clip(bayer * 0.35 + gauss * 0.35 + coherence * 0.30, 0, 1))
    return dfi, {
        'bayer': bayer, 'gauss': gauss,
        'coherence': coherence, 'spectral_k': slope
    }

# ─── Visualização Plotly 3D ───────────────────────────────────────────────────
def build_fig(img, dfi_val, dfi_c, freqs, E_stft, micro, noise_e, radial):
    H, W = img.shape
    fig  = go.Figure()
    traces_per = []

    # ── Ambiente 1: STFT Espacial
    nf, nt = E_stft.shape
    sf = max(1, nf // 60)
    st = max(1, nt // 60)
    Ed = E_stft[::sf, ::st]
    F_ax = (freqs[::sf] * W * 2)[:Ed.shape[0]]
    T_ax = np.linspace(0, W, Ed.shape[1])
    Tg, Fg = np.meshgrid(T_ax, F_ax)
    if Tg.shape != Ed.shape:
        Ed = Ed[:Fg.shape[0], :Tg.shape[1]]
    fig.add_trace(go.Surface(
        x=Tg, y=Fg, z=Ed, colorscale='Viridis', opacity=0.93,
        visible=True, name='STFT Espacial',
        hovertemplate='x=%{x:.0f}px  f=%{y:.1f}  E=%{z:.3f}<extra>STFT Espacial</extra>'
    ))
    traces_per.append([0])

    # ── Ambiente 2: Micro-Cepstro 2D
    r_m = micro.shape[0] // 2
    sf2 = max(1, micro.shape[0] // 60)
    Md  = micro[::sf2, ::sf2]
    qv  = np.linspace(-r_m, r_m, Md.shape[0])
    Qx, Qy = np.meshgrid(qv, qv)
    fig.add_trace(go.Surface(
        x=Qx, y=Qy, z=Md, colorscale='ice', opacity=0.93,
        visible=False, name='Micro-Cepstro 2D',
        hovertemplate='qx=%{x:.1f}  qy=%{y:.1f}  C=%{z:.4f}<extra>Micro-Cepstro</extra>'
    ))
    traces_per.append([1])

    # ── Ambiente 3: Mapa Ruído Sensor
    sf3 = max(1, H // 60)
    st3 = max(1, W // 60)
    Nd  = noise_e[::sf3, ::st3]
    px  = np.linspace(0, W, Nd.shape[1])
    py  = np.linspace(0, H, Nd.shape[0])
    Px, Py = np.meshgrid(px, py)
    fig.add_trace(go.Surface(
        x=Px, y=Py, z=Nd, colorscale='hot', opacity=0.93,
        visible=False, name='Mapa Ruído Sensor',
        hovertemplate='x=%{x:.0f}  y=%{y:.0f}  E=%{z:.4f}<extra>Ruído Sensor</extra>'
    ))
    traces_per.append([2])

    # ── Ambiente 4: Espectro Radial (superfície de revolução)
    n_r   = len(radial)
    theta = np.linspace(0, 2 * np.pi, 72)
    R_g, T_g2 = np.meshgrid(np.arange(n_r), theta)
    Z_g   = np.tile(radial, (72, 1))
    Z_n   = Z_g / (Z_g.max() + 1e-9)
    Xg    = R_g * np.cos(T_g2)
    Yg    = R_g * np.sin(T_g2)
    fig.add_trace(go.Surface(
        x=Xg, y=Yg, z=Z_n, colorscale='plasma', opacity=0.93,
        visible=False, name='Espectro Radial',
        customdata=R_g,
        hovertemplate='r=%{customdata:.0f}px  E=%{z:.3f}<extra>Espectro Radial</extra>'
    ))
    traces_per.append([3])

    # ── Dropdown
    AMBIENTES = ['STFT Espacial', 'Micro-Cepstro 2D',
                 'Mapa Ruído Sensor', 'Espectro Radial']
    total   = len(fig.data)
    buttons = []
    for i, nome in enumerate(AMBIENTES):
        vlist = [False] * total
        for ti in traces_per[i]:
            vlist[ti] = True
        buttons.append(dict(
            label=nome, method='update',
            args=[{'visible': vlist},
                  {'title.text': f'<b>Scanner Forense</b>  ·  {nome}'}]
        ))

    cor   = '#00FF88' if dfi_val > 0.6 else ('#FFAA00' if dfi_val > 0.35 else '#FF4444')
    verd  = 'CÂMERA REAL' if dfi_val > 0.6 else ('INCONCLUSIVO' if dfi_val > 0.35 else 'PROVÁVEL IA')

    fig.update_layout(
        title=dict(
            text=(f'<b>Scanner Topográfico Forense</b>  ·  STFT Espacial<br>'
                  f'<span style="color:{cor};font-size:13px">'
                  f'DFI = {dfi_val:.3f}  →  {verd}  │  '
                  f'Bayer={dfi_c["bayer"]:.3f}  '
                  f'Ruído={dfi_c["gauss"]:.3f}  '
                  f'Espectro={dfi_c["coherence"]:.3f}  '
                  f'k={dfi_c["spectral_k"]:.2f}'
                  f'</span>'),
            font=dict(size=13)
        ),
        updatemenus=[dict(
            buttons=buttons, direction='down',
            x=0.0, xanchor='left', y=1.14, yanchor='top',
            bgcolor='#1a1a2e', bordercolor='#444',
            font=dict(color='white', size=12)
        )],
        scene=dict(
            xaxis_title='Posição / Quefrência X',
            yaxis_title='Freq. Espacial / Quefrência Y',
            zaxis_title='Energia Normalizada',
            bgcolor='#0a0a0f',
            xaxis=dict(gridcolor='#333', color='#aaa', showbackground=False),
            yaxis=dict(gridcolor='#333', color='#aaa', showbackground=False),
            zaxis=dict(gridcolor='#333', color='#aaa', showbackground=False),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))
        ),
        paper_bgcolor='#0a0a0f', plot_bgcolor='#0a0a0f',
        font=dict(color='#cccccc'),
        margin=dict(l=0, r=0, b=20, t=95),
        height=720
    )
    return fig

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    out      = 'scanner_forensico_interativo.html'

    print("\n▶  Scanner Topográfico Forense  ·  AlphaPhi")
    print("=" * 52)

    img = load_image(img_path)

    print("\nComputando análises forenses...")
    freqs, E_stft = spatial_stft_map(img)
    micro         = micro_cepstrum(img)
    noise_e       = noise_energy_map(img)
    radial        = radial_spectrum(img)

    dfi_val, dfi_c = compute_dfi(img, micro, radial)
    verd = 'CÂMERA REAL' if dfi_val > 0.6 else ('INCONCLUSIVO' if dfi_val > 0.35 else 'PROVÁVEL IA')

    print(f"\n  ┌───────────────────────────────────┐")
    print(f"  │  DFI  = {dfi_val:.3f}  →  {verd:<12s}  │")
    print(f"  │  Bayer peak   : {dfi_c['bayer']:.3f}              │")
    print(f"  │  Ruído Gauss  : {dfi_c['gauss']:.3f}              │")
    print(f"  │  Espectro coh : {dfi_c['coherence']:.3f}  (k={dfi_c['spectral_k']:.2f})   │")
    print(f"  └───────────────────────────────────┘")

    print("\nConstruindo visualização...")
    fig = build_fig(img, dfi_val, dfi_c, freqs, E_stft, micro, noise_e, radial)

    fig.write_html(out, include_plotlyjs='cdn',
                   config={'displaylogo': False, 'scrollZoom': True,
                           'modeBarButtonsToRemove': ['toImage']})

    print(f"\n✓  Salvo: {out}")
    print("\nUso:")
    print("  python AlphaPhi_Scanner_Forensico.py imagem.jpg")
    print("\nNo Colab:")
    print("  !python AlphaPhi_Scanner_Forensico.py imagem.jpg")
    print("  from IPython.display import HTML")
    print("  with open('scanner_forensico_interativo.html') as f: display(HTML(f.read()))")
