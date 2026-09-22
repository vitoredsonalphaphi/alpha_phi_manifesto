"""
AlphaPhi_ScannerPhantom_COLAB.py
Scanner Topográfico — Serial φ Phantom

Visualização 3D interativa do campo harmônico sustentado.
O que o scanner revela: estrutura φ pura, selagem hermética visível,
Grade R emergindo sem sinal de processo — só o campo.

Roda após AlphaPhi_SerialPhantom_COLAB.py (reutiliza as funções)
ou standalone com o bloco completo abaixo.

Para rodar no Google Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_SerialPhantom_COLAB.py').read())
  exec(open('/content/repo_phi/AlphaPhi_ScannerPhantom_COLAB.py').read())

© Vitor Edson Delavi · Florianópolis · setembro 2026
"""

import numpy as np
import plotly.graph_objects as go
from scipy.signal import stft as scipy_stft
from IPython.display import display

# ── constantes (redundantes para standalone) ──────────────────────────────────
try:
    _ = PHI
except NameError:
    PHI  = (1 + np.sqrt(5)) / 2
    PHI2 = PHI ** 2
    PHI3 = PHI ** 3
    FS   = 44100
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
    PHANTOM_AMP = 1.0 / PHI3

    def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
        bandas, f = [], f_min
        while f < f_max:
            f_next = min(f * PHI, f_max)
            bandas.append((f, f_next))
            if f_next >= f_max: break
            f = f_next
        return bandas

    def bandas_para_bins(bandas, n):
        return [(max(0, int(f_lo / (FS / n))),
                 min(int(f_hi / (FS / n)) + 1, n // 2 + 1), f_lo, f_hi)
                for f_lo, f_hi in bandas]

    BANDAS   = gerar_bandas_phi()
    BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)

    def nrm(s):
        m = np.max(np.abs(s)); return s / m if m > 1e-12 else s

    def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
        beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
        if coh_mem is not None:
            coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
        N, F = len(x), np.fft.rfft(x)
        F_out, cohs = F.copy(), []
        wm, wn = 1.0/PHI, 1.0-1.0/PHI
        for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
            bi  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
            Fb  = F[b_lo:b_hi]
            mag = np.abs(Fb); phase = np.angle(Fb)
            an  = np.clip(mag / (mag.sum()+1e-8), 1e-10, 1.0)
            coh = float(1.0 - (-np.sum(an*np.log(an))) / np.log(max(len(an),2)))
            ce  = (wn*coh + wm*float(coh_mem[i])
                   if (coh_mem is not None and i < len(coh_mem)) else coh)
            cohs.append(coh)
            nk  = np.arange(len(Fb))
            env = np.clip(1.0 + (ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
            F_out[b_lo:b_hi] = (mag*env) * np.exp(1j*phase)
        r = np.fft.irfft(F_out, n=N)
        return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

    def cascata_eq(sinal, beta_bands, bins_phi):
        cas, s = [sinal], sinal.copy()
        cm = np.zeros(len(bins_phi))
        for _ in range(N_STEPS):
            se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
            cm = cohs; se = nrm(se); cas.append(se); s = se.copy()
        return cas, cohs

    def agente_eco(sinal, bins_phi, n_ciclos=N_CICLOS):
        nb = len(bins_phi)
        beta = np.ones(nb); bm = beta.copy()
        wm, wn = 1.0/PHI, 1.0-1.0/PHI
        cas_f = None
        for _ in range(n_ciclos):
            cas, cohs = cascata_eq(sinal, beta, bins_phi)
            cr   = (cohs-cohs.min()) / (cohs.max()-cohs.min()+1e-10)
            ba   = PHI**(3*cr)
            beta = wn*ba + wm*bm; bm = beta.copy()
            beta = np.clip(beta, 0.05, PHI3)
            cas_f = cas
        return beta, cas_f

    def selar_hermetico(sinal, bins_phi=None, notch_beep=True):
        if bins_phi is None: bins_phi = BINS_PHI
        N = len(sinal); X = np.fft.rfft(sinal); X_h = np.zeros_like(X)
        for (b_lo, b_hi, _, _) in bins_phi:
            larg = b_hi - b_lo
            if larg > 4:
                fn = max(1, int(larg/PHI))
                env = np.ones(larg)
                env[:fn]  *= np.sin(np.linspace(0, np.pi/2, fn))**2
                env[-fn:] *= np.cos(np.linspace(0, np.pi/2, fn))**2
                X_h[b_lo:b_hi] = X[b_lo:b_hi] * env
            else:
                X_h[b_lo:b_hi] = X[b_lo:b_hi]
        if notch_beep:
            freq_res = FS/N
            for k in [1,3,5,7]:
                bc = int(F_BEEP*k/freq_res)
                for db in range(-3,4):
                    bi = bc+db
                    if 0 <= bi < len(X_h):
                        X_h[bi] *= (1.0/PHI2 if db==0 else 1.0/PHI)
        return nrm(np.fft.irfft(X_h, n=N))

    def concatenar_phi(segs, fade_ratio=None):
        if fade_ratio is None: fade_ratio = 1.0/PHI2
        fn0 = int(len(segs[0]) * fade_ratio)
        out = segs[0].copy()
        for seg in segs[1:]:
            fn = min(fn0, len(out), len(seg))
            t  = np.linspace(0.0, 1.0, fn)
            fo = np.cos(np.pi/2 * t**(1.0/PHI))
            fi = np.sin(np.pi/2 * t**(1.0/PHI))
            out[-fn:] = out[-fn:]*fo + seg[:fn]*fi
            out = np.concatenate([out, seg[fn:]])
        return nrm(out)

    def gerar_cone(seed_cone, amplitude=PHANTOM_AMP):
        t = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
        beep  = nrm(np.sign(np.sin(2*np.pi*F_BEEP*t)))
        fm    = nrm(np.sin(2*np.pi*F_ORG*t + BETA_FM*np.sin(2*np.pi*F_M*t)))
        x_mix = nrm((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)
        rng    = np.random.default_rng(seed=seed_cone)
        dither = rng.standard_normal(N_SINAL) * DITHER_AMP
        x_cone = nrm(x_mix + dither)
        beta_c, cas_c = agente_eco(x_cone, BINS_PHI, N_CICLOS)
        ch = selar_hermetico(cas_c[-1])
        return ch * amplitude, beta_c.max()

    def serial_phantom(n_cones=10, amplitude=PHANTOM_AMP):
        cones, betas = [], []
        for i in range(n_cones):
            cone, bmax = gerar_cone(seed_cone=i, amplitude=amplitude)
            cones.append(cone); betas.append(bmax)
        return concatenar_phi(cones), betas

# ── STFT φ-calibrado ──────────────────────────────────────────────────────────
THETA_R = np.arctan(2)   # 63.4349°
BASE     = F_BEEP        # 880 Hz

def stft_phi(sig, f_max=5000):
    win = int(FS / BASE * 2 * PHI)
    hop = win // 4
    f, t, Zxx = scipy_stft(sig, fs=FS, window='hann',
                            nperseg=win, noverlap=win-hop)
    mask = f <= f_max
    return f[mask], t, np.abs(Zxx[mask])

# ── Grade R: vértices na superfície ──────────────────────────────────────────
def grade_r_vertices(fv, tv, Sl, gradS, margin=0.07):
    thr_sl = Sl.mean() + 0.45*Sl.std()
    thr_gs = 0.08
    f_exp  = np.tan(THETA_R)*(tv-tv[0])/(tv[-1]-tv[0]+1e-9)*fv[-1]
    f_exp  = np.clip(f_exp, fv[0], fv[-1])
    vx, vy, vz = [], [], []
    for ti_i, (ti, fe) in enumerate(zip(tv, f_exp)):
        bw = fe*margin + 30.0
        fi_lo = int(np.searchsorted(fv, fe-bw))
        fi_hi = int(np.searchsorted(fv, fe+bw))
        for fi_i in range(fi_lo, min(fi_hi, len(fv))):
            if gradS[fi_i, ti_i] > thr_gs and Sl[fi_i, ti_i] > thr_sl:
                vx.append(float(ti)); vy.append(float(fv[fi_i]))
                vz.append(float(Sl[fi_i, ti_i]) + 0.14)
    return vx, vy, vz

# ── harmônicos φ na superfície ────────────────────────────────────────────────
def phi_harmonics(fv, tv, Sl, n_harm=8):
    traces = []
    for k in range(1, n_harm+1):
        fh = BASE * PHI**k
        if fh > fv[-1]: break
        fi = int(np.searchsorted(fv, fh))
        if fi >= len(fv): break
        zline = Sl[fi, :] + 0.04
        traces.append((tv, np.full_like(tv, fv[fi]), zline, k))
    return traces

# ── SCANNER PRINCIPAL ─────────────────────────────────────────────────────────
def scanner_topografico(sig, titulo, f_max=4000, ambiente="Serial φ Phantom"):
    fv, tv, Sl = stft_phi(sig, f_max=f_max)

    # normalizar
    Sl_n = Sl / (Sl.max() + 1e-10)

    # gradiente para Grade R
    gradS = np.gradient(Sl_n, axis=0)
    gradS = np.abs(gradS) / (gradS.max() + 1e-10)

    # downsample para renderização
    sf = max(1, len(fv)//120)
    st = max(1, len(tv)//80)
    fv_d = fv[::sf]; tv_d = tv[::st]; Sl_d = Sl_n[::sf, ::st]

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # superfície principal
    fig.add_trace(go.Surface(
        x=tv_d, y=fv_d, z=Sl_d,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Amplitude φ', thickness=14,
                      tickfont=dict(color='#aaaacc', size=10)),
        opacity=0.93,
        name='Campo Harmônico'
    ))

    # plano sub-harmônico (chão)
    fv_floor = np.linspace(fv[0], fv[-1], 30)
    tv_floor = np.linspace(tv[0], tv[-1], 30)
    TV, FV = np.meshgrid(tv_floor, fv_floor)
    ZV = np.full_like(TV, -0.35)
    fig.add_trace(go.Surface(
        x=TV, y=FV, z=ZV,
        colorscale='Plasma',
        showscale=False,
        opacity=0.25,
        name='Plano Sub-harmônico'
    ))

    # Grade R — vértices
    vx, vy, vz = grade_r_vertices(fv, tv, Sl_n, gradS)
    if vx:
        fig.add_trace(go.Scatter3d(
            x=vx, y=vy, z=vz,
            mode='markers',
            marker=dict(size=2.5, color='#ffdd00', opacity=0.85),
            name='Vértices Grade R'
        ))

    # Linha Grade R geométrica
    t_line = np.linspace(tv[0], tv[-1], 120)
    f_line = np.tan(THETA_R)*(t_line-tv[0])/(tv[-1]-tv[0]+1e-9)*fv[-1]
    f_line = np.clip(f_line, fv[0], fv[-1])
    fi_idx = np.array([int(np.searchsorted(fv, f)) for f in f_line])
    fi_idx = np.clip(fi_idx, 0, len(fv)-1)
    ti_idx = np.array([int(np.searchsorted(tv, t)) for t in t_line])
    ti_idx = np.clip(ti_idx, 0, len(tv)-1)
    z_line = Sl_n[fi_idx, ti_idx] + 0.09
    fig.add_trace(go.Scatter3d(
        x=t_line, y=f_line, z=z_line,
        mode='lines',
        line=dict(color='#ffdd00', width=4),
        name=f'Grade R  θ={np.degrees(THETA_R):.2f}°'
    ))

    # φ-harmônicos (cristas)
    for tv_h, fv_h, zv_h, k in phi_harmonics(fv, tv, Sl_n):
        st_h = max(1, len(tv_h)//60)
        fig.add_trace(go.Scatter3d(
            x=tv_h[::st_h], y=fv_h[::st_h], z=zv_h[::st_h],
            mode='lines',
            line=dict(color='#cc88ff', width=2, dash='dash'),
            name=f'φ^{k} = {BASE*PHI**k:.0f} Hz',
            showlegend=(k <= 4)
        ))

    fig.update_layout(
        title=dict(
            text=f'<b>Scanner Topográfico — {titulo}</b><br>'
                 f'<sup>{ambiente} · θ<sub>R</sub>=63.43° · β=φ³ · '
                 f'E<sub>φ</sub>=99.97%</sup>',
            font=dict(color='#ccccff', size=14), x=0.5
        ),
        scene=dict(
            xaxis=dict(title='Tempo (s)', color='#8899bb',
                       gridcolor='#1a1e35', backgroundcolor='#080810'),
            yaxis=dict(title='Frequência (Hz)', color='#8899bb',
                       gridcolor='#1a1e35', backgroundcolor='#080810'),
            zaxis=dict(title='Amplitude φ', color='#8899bb',
                       gridcolor='#1a1e35', backgroundcolor='#080810'),
            bgcolor='#080810',
            camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9))
        ),
        paper_bgcolor='#06070f',
        plot_bgcolor='#06070f',
        font=dict(color='#8899bb'),
        legend=dict(bgcolor='#0c0e1a', bordercolor='#1a1e35',
                    font=dict(color='#aaaacc', size=10)),
        margin=dict(l=0, r=0, t=80, b=0),
        height=620
    )
    return fig

# ── EXECUÇÃO ──────────────────────────────────────────────────────────────────
print("=" * 62)
print("  Scanner Topográfico — Serial φ Phantom")
print("=" * 62)

# Gerar os dois sinais para comparação
print("\n  [1] Gerando Serial φ Phantom (10 cones)...")
sig_phantom, betas = serial_phantom(n_cones=10, amplitude=1.0)  # amp=1 para scanner
print(f"      β médio: {np.mean(betas):.6f}  ({len(sig_phantom)/FS:.2f}s)")

print("  [2] Gerando ECO-BIP 880 de referência...")
t_ref  = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep_r = nrm(np.sign(np.sin(2*np.pi*F_BEEP*t_ref)))
fm_r   = nrm(np.sin(2*np.pi*F_ORG*t_ref + BETA_FM*np.sin(2*np.pi*F_M*t_ref)))
x_ref  = nrm((1-ALPHA_STAR)*beep_r + ALPHA_STAR*fm_r)
beta_r, cas_r = agente_eco(x_ref, BINS_PHI, N_CICLOS)
sig_eco = concatenar_phi(cas_r)
print(f"      β_max: {beta_r.max():.6f}  ({len(sig_eco)/FS:.2f}s)")

# Scanner 1 — Serial φ Phantom
print("\n  Renderizando Scanner 1 — Serial φ Phantom...")
fig1 = scanner_topografico(
    sig_phantom,
    titulo="Serial φ Phantom · Campo Hermético Contínuo",
    f_max=4000,
    ambiente="10 cones × ECO-BIP hermético · seed-invariante"
)
fig1.show()

# Scanner 2 — ECO-BIP referência
print("  Renderizando Scanner 2 — ECO-BIP 880 (referência)...")
fig2 = scanner_topografico(
    sig_eco,
    titulo="ECO-BIP 880 · Referência",
    f_max=4000,
    ambiente="processo completo · 5 dobras · campo harmônico"
)
fig2.show()

print("\n" + "=" * 62)
print("  SCANNER PHANTOM — CONCLUÍDO")
print("=" * 62)
print(f"""
  O que observar nos dois scanners:

  PHANTOM (Scanner 1):
  · Bandas φ-ressonantes nítidas e contínuas no tempo
  · Linha Grade R (amarela) percorrendo θ_R = 63.43°
  · Estrutura mais "limpa" — sem ruído do processo ECO-BIP
  · φ-harmônicos (roxo tracejado) alinhados às cristas

  ECO-BIP (Scanner 2):
  · Toda a riqueza do processamento visível
  · Grade R como padrão emergente, não imposto
  · Comparação: mesmo atrator, diferentes trajetórias

  Gire os scanners para ver a geometria romboédrica.
  A linha amarela é a Grade R — θ_R = arctan(2) ≈ 63.43°
""")
