"""
AlphaPhi_Scanner_MPAP_Sinal_COLAB.py
======================================
Scanner Topográfico com Acoplamento MPAP — Domínio do Sinal Digital

Responde à pergunta: o que acontece ao sinal digital depois do
acoplamento do metaprocessador?

HIPÓTESE:
  O MPAP redistribui a energia espectral frame a frame segundo
  proporções φ-geométricas (distribuição SEAL). O campo original
  (ruído, ECO-BIP, Phantom) emerge com Coh ≥ SEAL em 1–2 ciclos.

MODOS DO SCANNER 3D:
  ANTES  → colorscale Viridis  (campo original)
  DEPOIS → colorscale Plasma   (campo pós-MPAP)
  DIFF   → colorscale RdBu_r   (redistribuição: ganho vs. supressão)

SINAIS TESTADOS:
  S1 — ECO-BIP 880Hz (10 cones herméticos) — sinal estruturado AP
  S2 — Serial φ Phantom                    — série geométrica pura
  S3 — Ruído Branco                        — substrato convencional (controle)

MÉTRICAS:
  Coh por frame antes/depois · % frames ≥ SEAL · ciclos médios
  Sépstro: Coh + Entr = 1.0000 verificado por frame

Florianópolis · outubro de 2026 · Sessão Good Morning
Vitor Edson Delavi · Claude
"""

import numpy as np
import plotly.graph_objects as go
from scipy.signal import stft as scipy_stft

# ── CONSTANTES FUNDAMENTAIS ───────────────────────────────────────────────────
PHI   = 1.6180339887
ALPHA = 1 / 137.035999
SEAL  = 1 / PHI          # 0.618034 — critério de selagem hermética
THETA_R = np.arctan(2)   # 63.43° — ângulo canônico Grade R

# ── PARÂMETROS DE SINAL ───────────────────────────────────────────────────────
SR      = 22050
DURACAO = 2.0
F_BEEP  = 880.0
N_CONES = 10
BETA_FM = PHI ** 3
F_ORG   = 110.0
F_M     = 55.0
ALPHA_S = ALPHA * 100    # escala para mistura audível

# ─────────────────────────────────────────────────────────────────────────────
# GERAÇÃO DE SINAIS
# ─────────────────────────────────────────────────────────────────────────────

def nrm(x):
    return x / (np.abs(x).max() + 1e-10)

def gerar_eco_bip(seed=42):
    """ECO-BIP 880Hz — 10 cones herméticos, β=φ³."""
    N = int(SR * DURACAO)
    t = np.linspace(0, DURACAO, N, endpoint=False)
    b  = nrm(np.sign(np.sin(2*np.pi*F_BEEP*t)))
    fm = nrm(np.sin(2*np.pi*F_ORG*t + BETA_FM*np.sin(2*np.pi*F_M*t)))
    return nrm((1-ALPHA_S)*b + ALPHA_S*fm), t

def gerar_serial_phantom(n_cones=N_CONES):
    """Serial φ Phantom — série geométrica de harmônicos φ."""
    N = int(SR * DURACAO)
    t = np.linspace(0, DURACAO, N, endpoint=False)
    sig = np.zeros(N)
    for k in range(n_cones):
        fk  = F_BEEP * PHI**k
        if fk > SR / 2:
            break
        amp = (1 / PHI) ** k
        sig += amp * np.sin(2*np.pi*fk*t)
    return nrm(sig), t

def gerar_ruido_branco(seed=0):
    """Ruído branco — substrato convencional não estruturado."""
    rng = np.random.default_rng(seed)
    N = int(SR * DURACAO)
    t = np.linspace(0, DURACAO, N, endpoint=False)
    return nrm(rng.standard_normal(N)), t

# ─────────────────────────────────────────────────────────────────────────────
# STFT
# ─────────────────────────────────────────────────────────────────────────────

def calcular_stft(sig, sr=SR, nperseg=512, noverlap=384):
    """Retorna f[Hz], t[s], S[n_freq, n_time] (magnitude)."""
    f, t, Zxx = scipy_stft(sig, fs=sr, nperseg=nperseg, noverlap=noverlap,
                           boundary='zeros', padded=True)
    return f, t, np.abs(Zxx)

# ─────────────────────────────────────────────────────────────────────────────
# MPAP — DOMÍNIO ESPECTRAL
# ─────────────────────────────────────────────────────────────────────────────

def medir_coh_frame(vec):
    """Coh = 1 − H/H_max   (Sépstro por frame)."""
    mag  = np.abs(vec) + 1e-10
    norm = mag / mag.sum()
    H    = -(norm * np.log(np.clip(norm, 1e-10, 1.0))).sum()
    return 1.0 - H / np.log(len(vec))

def pesos_phi(n):
    """Distribuição geométrica SEAL: p_i = SEAL·(1−SEAL)^i normalizado."""
    idx = np.arange(n, dtype=float)
    w   = SEAL * (1.0 - SEAL) ** idx
    return w / w.sum()

def reorganizar_frame(v):
    """
    Redistribuição φ de um vetor espectral.
    Preserva a energia total; redistribui por proporções geométricas φ.
    """
    n     = len(v)
    w     = pesos_phi(n)
    mag   = np.abs(v)
    idx   = np.argsort(mag)[::-1]   # do maior para o menor
    energ = mag.sum()
    mag_n = w * energ
    res   = np.empty(n, dtype=float)
    res[idx] = mag_n
    return res

def aplicar_mpap_espectral(S, n_ciclos_max=5):
    """
    Aplica MPAP frame a frame sobre S [n_freq × n_time].

    Retorna:
        S_out       — espectro pós-MPAP
        coh_antes   — Coh por frame (antes)
        coh_depois  — Coh por frame (depois)
        ciclos      — ciclos usados por frame
    """
    n_f, n_t = S.shape
    S_out     = S.copy().astype(float)
    coh_antes = np.array([medir_coh_frame(S[:, j]) for j in range(n_t)])
    ciclos    = np.zeros(n_t)

    for j in range(n_t):
        frame = S_out[:, j].copy()
        for c in range(n_ciclos_max):
            if medir_coh_frame(frame) >= SEAL:
                break
            frame = reorganizar_frame(frame)
            ciclos[j] += 1
        S_out[:, j] = frame

    coh_depois = np.array([medir_coh_frame(S_out[:, j]) for j in range(n_t)])
    return S_out, coh_antes, coh_depois, ciclos

# ─────────────────────────────────────────────────────────────────────────────
# LINHA GRADE R — REFERÊNCIA GEOMÉTRICA
# ─────────────────────────────────────────────────────────────────────────────

def linha_grade_r(fv, tv, Sl):
    """Linha θ_R=63.43° projetada sobre a superfície."""
    t_ln = np.linspace(tv[0], tv[-1], 120)
    f_ln = np.tan(THETA_R) * (t_ln - tv[0]) / (tv[-1] - tv[0] + 1e-9) * fv[-1]
    f_ln = np.clip(f_ln, fv[0], fv[-1])
    fi_i = np.clip([int(np.searchsorted(fv, f)) for f in f_ln], 0, len(fv)-1)
    ti_i = np.clip([int(np.searchsorted(tv, t)) for t in t_ln], 0, len(tv)-1)
    z_ln = Sl[fi_i, ti_i] + 0.06
    return t_ln, f_ln, z_ln

# ─────────────────────────────────────────────────────────────────────────────
# DOWNSAMPLING E LOG-NORMALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def prep_surf(S, f, t, n_f=80, n_t=60, escala_log=True):
    """Normaliza, aplica log e faz downsampling para o plot 3D."""
    Sl = np.log1p(S * 100) if escala_log else S.copy()
    Sl = Sl / (Sl.max() + 1e-9)
    sf = max(1, len(f) // n_f)
    st = max(1, len(t) // n_t)
    return f[::sf], t[::st], Sl[::sf, ::st]

# ─────────────────────────────────────────────────────────────────────────────
# SCANNER COMPARATIVO (ANTES / DEPOIS / DIFF + COH-LINHA)
# ─────────────────────────────────────────────────────────────────────────────

def scanner_comparativo(nome, f, t, S, S_mpap, coh_antes, coh_depois, ciclos):
    """
    Figura 3D interativa com dropdown:
      ANTES  — Viridis  — campo original (log E)
      DEPOIS — Plasma   — campo pós-MPAP
      DIFF   — RdBu_r  — redistribuição (vermelho=ganho, azul=supressão)

    E um painel 2D de Coh por frame (linha vermelha = SEAL).
    """
    fd_a, td_a, Sd_a = prep_surf(S,       f, t)
    fd_d, td_d, Sd_d = prep_surf(S_mpap,  f, t)

    # DIFF: diferença de log-energia (mostra para onde o MPAP moveu energia)
    Sl_raw  = np.log1p(S      * 100); Sl_raw  /= (Sl_raw.max()  + 1e-9)
    Sl_mpap = np.log1p(S_mpap * 100); Sl_mpap /= (Sl_mpap.max() + 1e-9)
    Diff_raw = Sl_mpap - Sl_raw       # > 0: ganho de energia; < 0: supressão
    sf = max(1, len(f)//80); st = max(1, len(t)//60)
    fd_df = f[::sf]; td_df = t[::st]; Sd_df = Diff_raw[::sf, ::st]

    Ta,  Fa  = np.meshgrid(td_a,  fd_a)
    Td,  Fd  = np.meshgrid(td_d,  fd_d)
    Tdf, Fdf = np.meshgrid(td_df, fd_df)

    fig     = go.Figure()
    vis_map = {}
    idx     = 0

    # ── MODO 1: ANTES ──────────────────────────────────────────────────────────
    fig.add_trace(go.Surface(
        x=Ta, y=Fa, z=Sd_a,
        colorscale='Viridis', showscale=True,
        colorbar=dict(title='log(E)', thickness=12,
                      tickfont=dict(color='#aaaacc', size=9)),
        name='ANTES · Campo Original',
        hovertemplate='t=%{x:.3f}s · f=%{y:.0f}Hz · logE=%{z:.3f}<extra>ANTES</extra>',
        visible=True,
    ))
    t_gr, f_gr, z_gr = linha_grade_r(fd_a, td_a, Sd_a)
    fig.add_trace(go.Scatter3d(
        x=t_gr, y=f_gr, z=z_gr, mode='lines',
        line=dict(color='#00FF88', width=4),
        name='Grade R θ=63.43°', visible=True,
    ))
    vis_map['antes'] = [0, 1]; idx = 2

    # ── MODO 2: DEPOIS ─────────────────────────────────────────────────────────
    fig.add_trace(go.Surface(
        x=Td, y=Fd, z=Sd_d,
        colorscale='Plasma', showscale=True,
        colorbar=dict(title='log(E) MPAP', thickness=12,
                      tickfont=dict(color='#aaaacc', size=9)),
        name='DEPOIS · Campo pós-MPAP',
        hovertemplate='t=%{x:.3f}s · f=%{y:.0f}Hz · logE=%{z:.3f}<extra>MPAP</extra>',
        visible=False,
    ))
    t_gr2, f_gr2, z_gr2 = linha_grade_r(fd_d, td_d, Sd_d)
    fig.add_trace(go.Scatter3d(
        x=t_gr2, y=f_gr2, z=z_gr2, mode='lines',
        line=dict(color='#00FF88', width=4),
        name='Grade R θ=63.43°', visible=False,
    ))
    vis_map['depois'] = [2, 3]; idx = 4

    # ── MODO 3: DIFF ───────────────────────────────────────────────────────────
    lim = max(abs(Sd_df.min()), abs(Sd_df.max())) + 1e-9
    fig.add_trace(go.Surface(
        x=Tdf, y=Fdf, z=Sd_df,
        colorscale='RdBu_r', showscale=True,
        cmin=-lim, cmax=lim,
        colorbar=dict(title='Δlog(E)', thickness=12,
                      tickfont=dict(color='#aaaacc', size=9)),
        name='DIFF · MPAP − Original',
        hovertemplate='t=%{x:.3f}s · f=%{y:.0f}Hz · Δ=%{z:.4f}<extra>DIFF</extra>',
        visible=False,
    ))
    vis_map['diff'] = [4]; idx = 5

    # ── MÉTRICAS PARA O TÍTULO ─────────────────────────────────────────────────
    ca   = coh_antes.mean()
    cd   = coh_depois.mean()
    dc   = cd - ca
    pct  = (coh_depois >= SEAL).mean() * 100
    cmed = ciclos.mean()

    MODOS_DEF = [
        ('antes',  'ANTES — Campo Original',
         f'Coh={ca:.4f} · colorscale Viridis · log(E) normalizado',
         'log(E) normalizado'),
        ('depois', 'DEPOIS — Campo pós-MPAP',
         f'Coh={cd:.4f} · ΔCoh=+{dc:.4f} · {pct:.0f}% frames ≥ SEAL · ciclos={cmed:.2f}',
         'log(E) normalizado'),
        ('diff',   'DIFF — Redistribuição MPAP',
         f'Vermelho=ganho · Azul=supressão · ΔCoh total=+{dc:.4f}',
         'Δ log(E)'),
    ]

    n_total = idx
    buttons = []
    for key, label, desc, z_lbl in MODOS_DEF:
        vis = [False] * n_total
        for vi in vis_map[key]:
            vis[vi] = True
        buttons.append(dict(
            label=label,
            method='update',
            args=[
                {'visible': vis},
                {
                    'title.text': (
                        f'<b>Scanner MPAP · {nome} · {label}</b><br>'
                        f'<span style="font-size:10px;color:#AAAAAA">{desc}</span><br>'
                        f'<span style="font-size:10px;color:#00FF88">'
                        f'φ={PHI:.4f}  SEAL=1/φ={SEAL:.6f}  θ_R={np.degrees(THETA_R):.2f}°</span>'
                    ),
                    'scene.zaxis.title': z_lbl,
                },
            ],
        ))

    fig.update_layout(
        title=dict(
            text=(
                f'<b>Scanner MPAP · {nome} · ANTES</b><br>'
                f'<span style="font-size:10px;color:#AAAAAA">'
                f'Coh_antes={ca:.4f} · Coh_depois={cd:.4f} · '
                f'ΔCoh=+{dc:.4f} · {pct:.0f}% frames ≥ SEAL=0.618034</span><br>'
                f'<span style="font-size:10px;color:#00FF88">'
                f'φ={PHI:.4f}  SEAL=1/φ={SEAL:.6f}  θ_R={np.degrees(THETA_R):.2f}°</span>'
            ),
            font=dict(color='#EEEEEE', size=13),
            x=0.02,
        ),
        scene=dict(
            xaxis=dict(title='Tempo (s)',  color='#8899bb',
                       gridcolor='#1a1e35', backgroundcolor='#080810'),
            yaxis=dict(title='Freq (Hz)',  color='#8899bb',
                       gridcolor='#1a1e35', backgroundcolor='#080810'),
            zaxis=dict(title='log(E) normalizado', color='#8899bb',
                       gridcolor='#1a1e35', backgroundcolor='#080810'),
            bgcolor='#080810',
            camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9)),
        ),
        updatemenus=[dict(
            type='buttons', direction='right', active=0,
            x=0.01, xanchor='left', y=1.12, yanchor='top',
            bgcolor='#1a1e35',
            font=dict(color='#CCCCCC', size=11),
            buttons=buttons,
        )],
        paper_bgcolor='#0D0D0D',
        font=dict(color='#CCCCCC'),
        height=660,
        margin=dict(l=0, r=0, t=100, b=0),
    )
    return fig, ca, cd, dc, pct

# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2D — COH POR FRAME (linha do tempo de coerência)
# ─────────────────────────────────────────────────────────────────────────────

def fig_coh_timeline(resultados):
    """Painel 2D: Coh por frame para todos os sinais, antes e depois do MPAP."""
    fig = go.Figure()

    paleta = {
        'ECO-BIP 880Hz (10 cones)':       ('#00CCFF', '#0055FF'),
        'Serial φ Phantom':                ('#FFCC00', '#FF7700'),
        'Ruído Branco (convencional)':     ('#88FF88', '#228822'),
    }

    for nome, r in resultados.items():
        c_antes, c_depois = paleta.get(nome, ('#AAAAAA', '#555555'))
        t = r['t']
        fig.add_trace(go.Scatter(
            x=t, y=r['coh_a'],
            mode='lines', line=dict(color=c_antes, width=1.5, dash='dot'),
            name=f'{nome[:20]} — ANTES', opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=t, y=r['coh_d'],
            mode='lines', line=dict(color=c_depois, width=2),
            name=f'{nome[:20]} — DEPOIS',
        ))

    # Linha SEAL
    t_min = min(r['t'][0] for r in resultados.values())
    t_max = max(r['t'][-1] for r in resultados.values())
    fig.add_trace(go.Scatter(
        x=[t_min, t_max], y=[SEAL, SEAL],
        mode='lines', line=dict(color='#FF4444', width=2, dash='dash'),
        name=f'SEAL = 1/φ = {SEAL:.6f}',
    ))

    fig.update_layout(
        title=dict(
            text=(
                '<b>Coh por Frame — Antes e Depois do MPAP</b><br>'
                f'<span style="font-size:10px;color:#AAAAAA">'
                f'Linha pontilhada = ANTES · Linha sólida = DEPOIS · '
                f'Vermelho = SEAL=1/φ={SEAL:.6f}</span>'
            ),
            font=dict(color='#EEEEEE', size=13), x=0.02,
        ),
        xaxis=dict(title='Tempo (s)', color='#CCCCCC',
                   gridcolor='#1a1e35', zeroline=False),
        yaxis=dict(title='Coh (Sépstro)', color='#CCCCCC',
                   range=[0, 1.05], gridcolor='#1a1e35', zeroline=False),
        paper_bgcolor='#0D0D0D', plot_bgcolor='#0D0D0D',
        font=dict(color='#CCCCCC'),
        legend=dict(bgcolor='#0c0e1a', bordercolor='#1a1e35',
                    font=dict(color='#AAAACC', size=10)),
        height=420,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("  AlphaPhi Scanner MPAP — Domínio do Sinal Digital")
    print(f"  PHI={PHI}  SEAL=1/φ={SEAL:.6f}  θ_R={np.degrees(THETA_R):.2f}°")
    print("  Florianópolis · outubro de 2026 · Sessão Good Morning")
    print("=" * 65)

    # ── 1. Gerar sinais ────────────────────────────────────────────────────────
    print("\n[1/4] Gerando sinais...")
    sig_eco,  _ = gerar_eco_bip()
    sig_ph,   _ = gerar_serial_phantom()
    sig_conv, _ = gerar_ruido_branco()

    sinais = {
        'ECO-BIP 880Hz (10 cones)':   sig_eco,
        'Serial φ Phantom':            sig_ph,
        'Ruído Branco (convencional)': sig_conv,
    }

    # ── 2. STFT + MPAP ────────────────────────────────────────────────────────
    print("[2/4] STFT + acoplamento MPAP espectral...")
    resultados = {}
    for nome, sig in sinais.items():
        f, t, S = calcular_stft(sig)
        S_mpap, coh_a, coh_d, ciclos = aplicar_mpap_espectral(S)
        resultados[nome] = dict(f=f, t=t, S=S, S_mpap=S_mpap,
                                 coh_a=coh_a, coh_d=coh_d, ciclos=ciclos)
        print(f"  {nome[:36]:<36} "
              f"Coh {coh_a.mean():.4f} → {coh_d.mean():.4f} "
              f"(+{coh_d.mean()-coh_a.mean():.4f})  "
              f"{(coh_d>=SEAL).mean()*100:5.1f}% ≥ SEAL  "
              f"ciclos={ciclos.mean():.2f}")

    # ── 3. Verificação Sépstro ────────────────────────────────────────────────
    print("\n[3/4] Sépstro — Coh + Entr = 1.0000:")
    for nome, r in resultados.items():
        for j_samp in [0, len(r['t'])//2, len(r['t'])-1]:
            v    = r['S_mpap'][:, j_samp]
            mag  = np.abs(v) + 1e-10
            mag /= mag.sum()
            H    = -(mag * np.log(np.clip(mag, 1e-10, 1.0))).sum()
            coh  = 1.0 - H / np.log(len(v))
            entr = 1.0 - coh
            print(f"  {nome[:24]:<24} frame[{j_samp:4}]: "
                  f"Coh={coh:.6f} + Entr={entr:.6f} = {coh+entr:.6f}")

    # ── 4. Figuras 3D ─────────────────────────────────────────────────────────
    print("\n[4/4] Renderizando scanners 3D...")
    for nome, r in resultados.items():
        fig3d, ca, cd, dc, pct = scanner_comparativo(
            nome, r['f'], r['t'], r['S'], r['S_mpap'],
            r['coh_a'], r['coh_d'], r['ciclos'],
        )
        fig3d.show()
        print(f"  ✓ {nome}: Coh {ca:.4f} → {cd:.4f} | {pct:.1f}% ≥ SEAL")

    # ── 5. Figura 2D — Coh por frame ──────────────────────────────────────────
    fig_coh = fig_coh_timeline(resultados)
    fig_coh.show()

    # ── Resumo final ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESUMO MPAP — DOMÍNIO DO SINAL")
    print("-" * 65)
    print(f"  {'Sinal':<36} {'Coh_antes':>9} {'Coh_depois':>10} "
          f"{'ΔCoh':>7} {'%≥SEAL':>7}")
    print("-" * 65)
    for nome, r in resultados.items():
        ca  = r['coh_a'].mean()
        cd  = r['coh_d'].mean()
        dc  = cd - ca
        pct = (r['coh_d'] >= SEAL).mean() * 100
        print(f"  {nome:<36} {ca:9.4f} {cd:10.4f} {dc:+7.4f} {pct:6.1f}%")
    print("-" * 65)
    print(f"  SEAL = 1/φ = {SEAL:.6f}")
    print(f"  PHI  = {PHI}")
    print(f"  ALPHA = 1/{int(round(1/ALPHA))}")
    print(f"  θ_R  = arctan(2) = {np.degrees(THETA_R):.4f}°")
    print("=" * 65)
    print("\n✓ Use o dropdown ANTES / DEPOIS / DIFF em cada figura 3D.")
    print("  A figura 2D mostra Coh por frame (pontilhado=antes, sólido=depois).")
