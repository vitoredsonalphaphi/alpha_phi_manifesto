"""
AlphaPhi_SerialPhantom_SemSelagem_COLAB.py
Serial φ Phantom — SEM Selagem Hermética + Scanner Topográfico

Gera o Serial φ (10 cones ECO-BIP 880) sem a etapa selar_hermetico().
Visualiza a estrutura arquitetural do Phantom no scanner topográfico agnóstico.

Cole numa única célula no Google Colab e execute.

© Vitor Edson Delavi · Florianópolis · setembro 2026
"""

import numpy as np
import plotly.graph_objects as go
from scipy.signal import stft as scipy_stft

# ══════════════════════════════════════════════════════════════════
#  CONSTANTES ALPHA-PHI
# ══════════════════════════════════════════════════════════════════
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5
N_STEPS    = 5
N_CICLOS   = 20
DITHER_AMP = 1.0 / PHI**5
N_SINAL    = int(FS * DURACAO)
N_CONES    = 10
F_MAX_SCAN = 4000

print("=" * 62)
print("  Serial φ Phantom — SEM SELAGEM")
print("  Scanner Topográfico Agnóstico")
print("=" * 62)

# ══════════════════════════════════════════════════════════════════
#  ECO-BIP (funções internas — sem selar_hermetico)
# ══════════════════════════════════════════════════════════════════
def nrm(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def _bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max: break
        f = f_next
    return bandas

def _bins(bandas, n):
    return [(max(0, int(f_lo / (FS / n))),
             min(int(f_hi / (FS / n)) + 1, n // 2 + 1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

BANDAS   = _bandas_phi()
BINS_PHI = _bins(BANDAS, N_SINAL)

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N  = len(x)
    F  = np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi   = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb   = F[b_lo:b_hi]
        mag  = np.abs(Fb)
        phase = np.angle(Fb)
        an   = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh  = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        ce   = (wn * coh + wm * float(coh_mem[i])) if coh_mem is not None else coh
        gain = (bi * ce) ** 0.5
        F_out[b_lo:b_hi] = mag * gain * np.exp(1j * phase)
        cohs.append(ce)
    return np.fft.irfft(F_out, n=N), np.array(cohs)

def crossfade_phi(a, b):
    n_cf = int(N_SINAL / (PHI ** 3))
    env  = np.linspace(0, 1, n_cf)
    out  = np.concatenate([
        a[:-n_cf],
        a[-n_cf:] * (1 - env) + b[:n_cf] * env,
        b[n_cf:]
    ])
    return nrm(out)

def gerar_cone_sem_selagem(seed):
    """ECO-BIP 880 completo — SEM selar_hermetico."""
    rng  = np.random.default_rng(seed)
    t    = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
    beep = np.sin(2 * np.pi * F_BEEP * t)
    fm   = np.sin(2 * np.pi * F_M * t + BETA_FM * np.sin(2 * np.pi * F_ORG * t))
    dth  = rng.uniform(-1, 1, N_SINAL) * DITHER_AMP
    hyb  = nrm(ALPHA_STAR * beep + (1 - ALPHA_STAR) * fm + dth)

    bins_phi = _bins(BANDAS, N_SINAL)
    beta_c   = np.ones(len(bins_phi))
    coh_mem  = np.zeros(len(bins_phi))
    cas_c    = [hyb]

    for _ in range(N_STEPS):
        for _ in range(N_CICLOS):
            y, cohs = eco_eq(cas_c[-1], bins_phi, beta_c, coh_mem)
            coh_mem = cohs
            max_coh = cohs.max()
            beta_c  = np.clip(
                beta_c * (PHI ** 3 / max(max_coh, 1e-6)) ** 0.1,
                1.0, PHI ** 3
            )
        cas_c.append(nrm(y))

    return cas_c[-1], beta_c.max()   # ← sem selar_hermetico

def concatenar_phi(cones):
    out = cones[0]
    for c in cones[1:]:
        out = crossfade_phi(out, c)
    return out

# ══════════════════════════════════════════════════════════════════
#  GERAR SERIAL SEM SELAGEM
# ══════════════════════════════════════════════════════════════════
print(f"\n  Gerando {N_CONES} cones SEM selagem...")
cones, betas = [], []
for i in range(N_CONES):
    cone, bmax = gerar_cone_sem_selagem(i)
    cones.append(cone)
    betas.append(bmax)
    print(f"    Cone {i+1:02d}  β={bmax:.4f}")

SERIAL = concatenar_phi(cones)
dur    = len(SERIAL) / FS
print(f"\n  Serial: {len(SERIAL):,} amostras · {dur:.2f}s")
print(f"  β médio: {np.mean(betas):.6f} | variação: {np.max(betas)-np.min(betas):.6f}")

# Verificar entropia (critério da Terceira Estrutura)
from scipy.signal import stft as _stft
_, _, Zxx_chk = _stft(SERIAL, fs=FS, nperseg=2048, noverlap=1536, window='hann')
E_por_frame = np.abs(Zxx_chk)
entr_frames = []
for col in E_por_frame.T:
    p = col / (col.sum() + 1e-10)
    entr_frames.append(-np.sum(p * np.log(p + 1e-10)))
entr_media = np.mean(entr_frames)
limiar_ts  = 5.16
print(f"\n  Entropia média por quadro: {entr_media:.4f}")
print(f"  Limiar Terceira Estrutura: {limiar_ts}")
pct_abaixo = np.mean(np.array(entr_frames) < limiar_ts) * 100
print(f"  Quadros abaixo do limiar:  {pct_abaixo:.1f}%")
if pct_abaixo > 90:
    print("  → Terceira Estrutura CONFIRMADA")
else:
    print("  → Terceira Estrutura NÃO confirmada nesta métrica")

# ══════════════════════════════════════════════════════════════════
#  SCANNER TOPOGRÁFICO AGNÓSTICO
# ══════════════════════════════════════════════════════════════════
def scanner_agnostico(sig, titulo="Sinal", subtitulo="",
                      fs=FS, f_max=F_MAX_SCAN, n_win=2048, hop=512):
    f_ax, t_ax, Zxx = scipy_stft(
        sig, fs=fs, nperseg=n_win, noverlap=n_win - hop, window='hann'
    )
    mask   = f_ax <= f_max
    f_plot = f_ax[mask]
    Z      = np.abs(Zxx[mask])
    Z_db   = 20 * np.log10(Z + 1e-10)
    T, F_g = np.meshgrid(t_ax, f_plot)

    sf = max(1, len(f_plot) // 150)
    st = max(1, len(t_ax)   // 250)

    fig = go.Figure()

    # [SINAL] Campo espectral
    fig.add_trace(go.Surface(
        x=T[::sf, ::st],
        y=F_g[::sf, ::st],
        z=Z_db[::sf, ::st],
        colorscale='Viridis',
        opacity=0.93,
        showscale=True,
        colorbar=dict(title='dB', x=1.02, len=0.8, thickness=14),
        name='[SINAL] Campo espectral'
    ))

    # [COMPOSIÇÃO] Piso de referência visual
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
    print(f"\n  Scanner: {len(t_ax)} frames × {len(f_plot)} bins · 0–{f_max:.0f} Hz")
    print(f"  Amplitude: [{Z_db.min():.1f}, {Z_db.max():.1f}] dB")

# ══════════════════════════════════════════════════════════════════
#  EXECUTAR SCANNER
# ══════════════════════════════════════════════════════════════════
scanner_agnostico(
    SERIAL,
    titulo="Serial φ Phantom — SEM Selagem Hermética",
    subtitulo=f"10 cones ECO-BIP 880 · β médio={np.mean(betas):.6f} · {dur:.1f}s · estrutura ternária: {pct_abaixo:.0f}% quadros"
)
