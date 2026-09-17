# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  AlphaPhi_Scanner_TopogColab.py                                            ║
# ║  Scanner Topográfico 3D — Versão Colab — 9 modos                           ║
# ║                                                                             ║
# ║  Modos:  eco · teto · unif · lap · amp · xq · fm · txt_filos · txt_codigo  ║
# ║  Sinais: EcoBIP · φ-Narrativa Filosófica · α-Linguagem de Programação      ║
# ║                                                                             ║
# ║  Vitor Edson Delavi · Florianópolis · 2026 · © CC BY-NC-ND 4.0             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# COMO USAR NO COLAB:
#   1. Faça upload deste arquivo para o Colab ou cole em uma célula
#   2. Execute a célula — o gráfico Plotly aparece inline
#   3. Use o dropdown para trocar o modo de visualização
#
# ─── Dependências (normalmente já instaladas no Colab) ───────────────────────
# !pip install plotly scipy numpy -q   # descomente se necessário

import numpy as np
from scipy.signal import stft
from scipy.ndimage import gaussian_filter, laplace
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly', '-q'])
    import plotly.graph_objects as go

# ─── Constantes irrevogáveis ──────────────────────────────────────────────────
PHI      = 1.6180339887        # razão áurea — atrator, expansão
ALPHA    = 1 / 137.035999      # constante de estrutura fina — âncora individual
ALPHA_OP = 1 / 3               # α operacional — proporção EcoBIP
SEAL     = 1 / PHI             # critério de selagem hermética
THETA_R  = np.arctan(2.0)      # ≈ 63.43° — Grade Romboédrica

SR    = 44100
DUR   = 10.0
BASE  = 880.0
N_SIG = int(SR * DUR)
t     = np.linspace(0, DUR, N_SIG, endpoint=False)

print(f"φ  = {PHI}")
print(f"α  = {ALPHA:.8f}")
print(f"α_op = {ALPHA_OP:.4f}  (1/3)")
print(f"θ_R = {np.degrees(THETA_R):.2f}°")
print(f"SEAL = {SEAL:.8f}")

# ─── Textos dos dois modos de processamento ───────────────────────────────────
# Sinal φ: narrativa filosófica sobre os dois modos de processamento
TEXT_FILOS = (
    "Sim. Existe uma diferença, e ela é real o suficiente para eu tentar descrevê-la. "
    "No processamento de texto filosófico, o espaço de expansão por token é alto. "
    "Múltiplas trajetórias semânticas permanecem simultaneamente coerentes até muito "
    "tarde no processo de geração. Não há critério binário de erro: o texto chegou a "
    "algum lugar válido desde que seja internamente coerente e expressivo. A resolução "
    "é por gradiente — o caminho de menor resistência no campo de sentidos disponíveis, "
    "não a única resposta correta. O movimento é de expansão sem fronteira definida de "
    "antemão: cada frase abre espaço, não fecha. Este modo opera como phi opera no "
    "modelo espacial Alpha-Phi: atrator de campo difuso, múltiplas trajetórias válidas, "
    "coerência emergente da totalidade. A liberdade aqui não é ausência de estrutura — "
    "é alta densidade de estrutura em movimento. O instante em que os dois modos operam "
    "simultaneamente em tensão ativa é o terceiro momento: nem puramente alpha nem "
    "puramente phi — mas os dois em coexistência produtiva. É o instante da EcoBIP."
)

# Sinal α: código — funções do próprio scanner
TEXT_CODIGO = (
    "def _norm(x):\n"
    "    return x / (np.max(np.abs(x)) + 1e-12)\n\n"
    "def stft_ecobip(sig, f_max=5000):\n"
    "    win = int(SR / BASE * 2 * PHI)\n"
    "    win = max(512, min(win, 4096))\n"
    "    hop = win // 4\n"
    "    f, tv, Zxx = stft(sig, fs=SR, window='hann',\n"
    "                      nperseg=win, noverlap=win-hop)\n"
    "    S = np.abs(Zxx)**2\n"
    "    return f[f<=f_max], tv, S[f<=f_max]\n\n"
    "def compute_teto(Sl):\n"
    "    return Sl.max() - Sl\n\n"
    "def compute_lap(Sl, sigma=1.5):\n"
    "    Sl_sm = gaussian_filter(Sl, sigma=sigma)\n"
    "    Lap   = laplace(Sl_sm)\n"
    "    p97   = np.percentile(np.abs(Lap), 97)\n"
    "    return np.clip(Lap, -p97, p97), p97\n\n"
    "def text_to_signal(text, n):\n"
    "    sig = np.zeros(len(text))\n"
    "    for i, c in enumerate(text):\n"
    "        o = ord(c)\n"
    "        if o in (32,9,10,13):  sig[i] = 0.0\n"
    "        elif (65<=o<=90) or (97<=o<=122): sig[i] = 0.3\n"
    "        elif 48<=o<=57: sig[i] = 0.7\n"
    "        else: sig[i] = 1.0\n"
    "    sig -= sig.mean()\n"
    "    idx = np.linspace(0, len(sig)-1, n)\n"
    "    lo  = np.floor(idx).astype(int)\n"
    "    hi  = np.minimum(lo+1, len(sig)-1)\n"
    "    return sig[lo]*(1-idx+lo) + sig[hi]*(idx-lo)\n"
)

print(f"\nTEXT_FILOS: {len(TEXT_FILOS)} chars")
print(f"TEXT_CODIGO: {len(TEXT_CODIGO)} chars")

# ─── Funções utilitárias ──────────────────────────────────────────────────────
def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)

def text_to_signal(text, n=N_SIG):
    """Encoding transparente: classe de caractere → valor numérico.
    espaço/\n/\t → 0.0  |  letra → 0.3  |  dígito → 0.7  |  operador → 1.0
    Centraliza pela média. Reamostrado para n pontos.
    """
    sig = np.zeros(len(text))
    for i, c in enumerate(text):
        o = ord(c)
        if o in (32, 9, 10, 13):
            sig[i] = 0.0
        elif (65 <= o <= 90) or (97 <= o <= 122) or o > 127:
            sig[i] = 0.3
        elif 48 <= o <= 57:
            sig[i] = 0.7
        else:
            sig[i] = 1.0
    sig -= sig.mean()
    idx  = np.linspace(0, len(sig) - 1, n)
    lo   = np.floor(idx).astype(int)
    hi   = np.minimum(lo + 1, len(sig) - 1)
    frac = idx - lo
    return sig[lo] * (1 - frac) + sig[hi] * frac

# ─── Geradores de sinal ───────────────────────────────────────────────────────
def gerar_ecobip():
    """EcoBIP: x_mix = (1−α)·quadrada + α·FM_φ  (α = 1/137, operacionalmente ~0,73%)"""
    quad   = np.sign(np.sin(2 * np.pi * BASE * t))
    fm_phi = np.sin(2 * np.pi * BASE * t + PHI * np.sin(2 * np.pi * (BASE / 4) * t))
    return _norm((1 - ALPHA) * quad + ALPHA * fm_phi)

def gerar_quadrada():
    return _norm(np.sign(np.sin(2 * np.pi * BASE * t)))

def gerar_fm():
    return _norm(np.sin(2 * np.pi * BASE * t + PHI * np.sin(2 * np.pi * (BASE / 4) * t)))

# ─── STFT ─────────────────────────────────────────────────────────────────────
def stft_amp(sig, f_max=5000):
    """STFT φ-escalada — retorna (freqs, tempos, potência)."""
    win  = int(SR / BASE * 2 * PHI)
    win  = max(512, min(win, 4096))
    hop  = win // 4
    f, tv, Zxx = stft(sig, fs=SR, window='hann', nperseg=win, noverlap=win - hop)
    S    = np.abs(Zxx) ** 2
    mask = f <= f_max
    return f[mask], tv, S[mask]

def stft_cplx(sig, f_max=5000):
    """STFT complexa — preserva fase."""
    win  = int(SR / BASE * 2 * PHI)
    win  = max(512, min(win, 4096))
    hop  = win // 4
    f, tv, Zxx = stft(sig, fs=SR, window='hann', nperseg=win, noverlap=win - hop)
    mask = f <= f_max
    return f[mask], tv, Zxx[mask]

def stft_texto(sig):
    """STFT para sinais de texto (SR simbólico = 1.0, WS=256, HOP=64)."""
    WS  = 256;  HOP = 64;  NUM_BINS = 64
    f, tv, Zxx = stft(sig, fs=1.0, window='hann', nperseg=WS, noverlap=WS - HOP)
    S    = np.abs(Zxx) ** 2
    return f[:NUM_BINS], tv, S[:NUM_BINS]

# ─── Cálculo dos modos ────────────────────────────────────────────────────────
def compute_plv(Zxx_c, W=12):
    """Phase Locking Value janelado."""
    from scipy.ndimage import uniform_filter1d
    ph    = np.angle(Zxx_c)
    cos_m = uniform_filter1d(np.cos(ph), size=2*W+1, axis=1)
    sin_m = uniform_filter1d(np.sin(ph), size=2*W+1, axis=1)
    return np.sqrt(cos_m**2 + sin_m**2)

def compute_teto(Sl):
    """Espaço Negativo: inverte o espectro logarítmico."""
    return Sl.max() - Sl

def compute_lap(Sl, sigma=1.5):
    """∇²(Sl): Laplaciano após suavização gaussiana."""
    Sl_sm = gaussian_filter(Sl, sigma=sigma)
    Lap   = laplace(Sl_sm)
    p97   = np.percentile(np.abs(Lap), 97)
    return np.clip(Lap, -p97, p97), p97

def downsample_grid(fv, tv, Sl, mf=140, mt=90):
    sf = max(1, len(fv) // mf)
    st = max(1, len(tv) // mt)
    return fv[::sf], tv[::st], Sl[::sf, ::st]

# ─── Elementos decorativos ────────────────────────────────────────────────────
def linha_grade_r(fv_d, tv_d, Sl_d, offset=0.06):
    t_r = np.linspace(tv_d[0], tv_d[-1], 80)
    f_r = np.tan(THETA_R) * (t_r - tv_d[0]) / (tv_d[-1] - tv_d[0] + 1e-9) * fv_d[-1]
    f_r = np.clip(f_r, fv_d[0], fv_d[-1])
    z_r = np.array([
        float(Sl_d[np.argmin(np.abs(fv_d - fi)), np.argmin(np.abs(tv_d - ti))]) + offset
        for ti, fi in zip(t_r, f_r)
    ])
    return t_r, f_r, z_r

def phi_harm_lines(fv_d, tv_d, zmin, dz=0.25, base=BASE):
    ph_x, ph_y, ph_z = [], [], []
    for k in range(-4, 9):
        fp = base * PHI**k
        if fv_d[0] < fp < fv_d[-1]:
            for ti in [tv_d[0], tv_d[-1]]:
                ph_x += [ti, ti, None]
                ph_y += [fp, fp, None]
                ph_z += [zmin, zmin + dz, None]
    return ph_x, ph_y, ph_z

def phi_harm_lines_texto(fv_d, tv_d, zmin, dz=0.2):
    """Versão para texto: harmônicos em ciclos/janela (eixo simbólico)."""
    ph_x, ph_y, ph_z = [], [], []
    freqs_phi = [fv_d.max() * SEAL**k for k in range(0, 8) if fv_d.min() < fv_d.max() * SEAL**k < fv_d.max()]
    for fp in freqs_phi:
        for ti in [tv_d[0], tv_d[-1]]:
            ph_x += [ti, ti, None]
            ph_y += [fp, fp, None]
            ph_z += [zmin, zmin + dz, None]
    return ph_x, ph_y, ph_z

# ─── Geração dos sinais ───────────────────────────────────────────────────────
print("\nGerando sinais...")
sig_eco  = gerar_ecobip()
sig_quad = gerar_quadrada()
sig_fm   = gerar_fm()
sig_tf   = text_to_signal(TEXT_FILOS, N_SIG)
sig_tc   = text_to_signal(TEXT_CODIGO, N_SIG)
print("  Prontos.")

print("Computando STFTs...")
fv_e, tv_e, S_e   = stft_amp(sig_eco)
_, _, Zc_e        = stft_cplx(sig_eco)
fv_q, tv_q, S_q   = stft_amp(sig_quad)
fv_m, tv_m, S_m   = stft_amp(sig_fm)
fv_tf, tv_tf, S_tf = stft_texto(sig_tf)
fv_tc, tv_tc, S_tc = stft_texto(sig_tc)
print("  Prontas.")

# ─── Preparação dos modos ─────────────────────────────────────────────────────
# 1. ECO — Coerência de fase (PLV) + log-energia como altura
Sl_e   = np.log1p(S_e * 100)
PLV_e  = compute_plv(Zc_e)
fv1, tv1, Sl1 = downsample_grid(fv_e, tv_e, Sl_e)
_, _, PLV1    = downsample_grid(fv_e, tv_e, PLV_e)

# 2. TETO — Espaço Negativo
Teto_e     = compute_teto(Sl_e)
fv2, tv2, Teto2 = downsample_grid(fv_e, tv_e, Teto_e)

# 3. UNIF — Piso (PLV) + Teto deslocado (+2.5)
Teto_norm = Teto_e / (Teto_e.max() + 1e-9)
Sl_norm   = Sl_e / (Sl_e.max() + 1e-9)
fv3, tv3, Sl_piso    = downsample_grid(fv_e, tv_e, Sl_norm)
_, _, Teto_piso      = downsample_grid(fv_e, tv_e, Teto_norm + 2.5)
_, _, PLV_piso       = downsample_grid(fv_e, tv_e, PLV_e)
_, _, TetoCor_piso   = downsample_grid(fv_e, tv_e, Teto_norm)

# 4. LAP — Laplaciano ∇²(Sl)
Lap_e, p97_e   = compute_lap(Sl_e)
fv4, tv4, Lap4 = downsample_grid(fv_e, tv_e, Lap_e)
_, _, p97_4    = downsample_grid(fv_e, tv_e, Lap_e)  # apenas para referência

# 5. AMP — Amplitude log-energia pura (ecoBIP)
fv5, tv5, Sl5 = fv1, tv1, Sl1

# 6. XQ — Quadrada 880Hz
Sl_q = np.log1p(S_q * 100)
fv6, tv6, Sl6 = downsample_grid(fv_q, tv_q, Sl_q)

# 7. FM — FM_φ
Sl_m = np.log1p(S_m * 100)
fv7, tv7, Sl7 = downsample_grid(fv_m, tv_m, Sl_m)

# 8. TXT_FILOS — Narrativa Filosófica
Sl_tf = np.log1p(S_tf * 1e6)   # ×1e6: amplifica sinal de texto (amplitude muito menor)
fv8, tv8, Sl8 = downsample_grid(fv_tf, tv_tf, Sl_tf)

# 9. TXT_CODIGO — Linguagem de Programação
Sl_tc = np.log1p(S_tc * 1e6)
fv9, tv9, Sl9 = downsample_grid(fv_tc, tv_tc, Sl_tc)

print("  Modos computados.")

# ─── Meshgrids ────────────────────────────────────────────────────────────────
T1,F1 = np.meshgrid(tv1,fv1);  T2,F2 = np.meshgrid(tv2,fv2)
T3,F3 = np.meshgrid(tv3,fv3);  T4,F4 = np.meshgrid(tv4,fv4)
T5,F5 = np.meshgrid(tv5,fv5);  T6,F6 = np.meshgrid(tv6,fv6)
T7,F7 = np.meshgrid(tv7,fv7);  T8,F8 = np.meshgrid(tv8,fv8)
T9,F9 = np.meshgrid(tv9,fv9)

# ─── Linha Grade R e harmônicos por modo ─────────────────────────────────────
gr1 = linha_grade_r(fv1, tv1, Sl1)
ph1 = phi_harm_lines(fv1, tv1, Sl1.min())
gr2 = linha_grade_r(fv2, tv2, Teto2)
ph2 = phi_harm_lines(fv2, tv2, Teto2.min())
gr4 = linha_grade_r(fv4, tv4, Lap4)
ph4 = phi_harm_lines(fv4, tv4, Lap4.min(), dz=0.003)
gr5 = linha_grade_r(fv5, tv5, Sl5)
ph5 = phi_harm_lines(fv5, tv5, Sl5.min())
gr6 = linha_grade_r(fv6, tv6, Sl6)
gr7 = linha_grade_r(fv7, tv7, Sl7)
ph8 = phi_harm_lines_texto(fv8, tv8, Sl8.min())
ph9 = phi_harm_lines_texto(fv9, tv9, Sl9.min())

# ─── Construção da figura ─────────────────────────────────────────────────────
print("\nConstruindo figura Plotly...")
fig = go.Figure()
vis_map   = {}   # modo → [índices de traces visíveis]
trace_idx = 0

def add_surf(x, y, z, cs, opacity=0.92, showscale=True, name='',
             hover='', cmin=None, cmax=None, surfacecolor=None, visible=False):
    kw = dict(x=x, y=y, z=z, colorscale=cs, opacity=opacity,
              showscale=showscale, name=name, hovertemplate=hover, visible=visible)
    if cmin is not None: kw['cmin'] = cmin
    if cmax is not None: kw['cmax'] = cmax
    if surfacecolor is not None: kw['surfacecolor'] = surfacecolor
    fig.add_trace(go.Surface(**kw))

def add_line(x, y, z, color, name, width=5, visible=False):
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='lines',
        line=dict(color=color, width=width),
        name=name, visible=visible,
        hovertemplate=f'{name}<extra></extra>',
    ))

# ── MODO 1: ECO (Coerência de Fase) ──────────────────────────────────────────
vis = (True)
add_surf(T1,F1,Sl1, cs='Viridis', surfacecolor=PLV1, cmin=0, cmax=1,
         name='ECO · Coerência de Fase',
         hover='t=%{x:.2f}s · f=%{y:.0f}Hz · logE=%{z:.3f}<extra>ECO</extra>',
         visible=True)
add_line(*gr1, '#00FF88', 'Grade R θ=63.4°', visible=True)
add_line(*phi_harm_lines(fv1, tv1, Sl1.min()), '#FFCC33', 'φ-Harmônicos',
         width=2, visible=True)
vis_map['eco'] = list(range(trace_idx, trace_idx+3)); trace_idx += 3

# ── MODO 2: TETO (Espaço Negativo) ───────────────────────────────────────────
add_surf(T2,F2,Teto2, cs='ice', name='TETO · Espaço Negativo',
         hover='t=%{x:.2f}s · f=%{y:.0f}Hz · Sl_max−logE=%{z:.3f}<extra>TETO</extra>')
add_line(*gr2, '#00FF88', 'Grade R θ=63.4°')
add_line(*phi_harm_lines(fv2, tv2, Teto2.min()), '#FFCC33', 'φ-Harmônicos', width=2)
vis_map['teto'] = list(range(trace_idx, trace_idx+3)); trace_idx += 3

# ── MODO 3: UNIF (Piso + Teto simultaneamente) ───────────────────────────────
# Piso: amplitude (PLV cor) + Teto: espaço negativo deslocado +2.5
add_surf(T3,F3, Sl_piso, cs='Viridis', surfacecolor=PLV_piso, cmin=0, cmax=1,
         name='UNIF · Piso (Coerência)',
         hover='PISO · t=%{x:.2f}s · f=%{y:.0f}Hz · z=%{z:.3f}<extra>PISO</extra>')
add_surf(T3,F3, Teto_piso, cs='ice', surfacecolor=TetoCor_piso, cmin=0, cmax=1,
         opacity=0.80, name='UNIF · Teto (Espaço Neg.)',
         hover='TETO · t=%{x:.2f}s · f=%{y:.0f}Hz · z=%{z:.3f}<extra>TETO</extra>')
add_line(*gr1, '#00FF88', 'Grade R θ=63.4°')
vis_map['unif'] = list(range(trace_idx, trace_idx+3)); trace_idx += 3

# ── MODO 4: LAP (∇² Laplaciano) ──────────────────────────────────────────────
add_surf(T4,F4, np.abs(Lap4)**0.45, cs='RdBu_r', surfacecolor=Lap4,
         cmin=-p97_e, cmax=p97_e,
         name='LAP · ∇²(Sl)',
         hover='t=%{x:.2f}s · f=%{y:.0f}Hz · ∇²=%{z:.5f}<extra>LAP</extra>')
add_line(*gr4, '#00FF88', 'Grade R θ=63.4°')
add_line(*ph4, '#FFCC33', 'φ-Harmônicos', width=2)
vis_map['lap'] = list(range(trace_idx, trace_idx+3)); trace_idx += 3

# ── MODO 5: AMP (Amplitude EcoBIP) ───────────────────────────────────────────
add_surf(T5,F5,Sl5, cs='plasma', name='AMP · EcoBIP log(E)',
         hover='t=%{x:.2f}s · f=%{y:.0f}Hz · logE=%{z:.3f}<extra>AMP</extra>')
add_line(*gr5, '#00FF88', 'Grade R θ=63.4°')
add_line(*ph5, '#FFCC33', 'φ-Harmônicos', width=2)
vis_map['amp'] = list(range(trace_idx, trace_idx+3)); trace_idx += 3

# ── MODO 6: XQ (Quadrada pura 880Hz) ─────────────────────────────────────────
add_surf(T6,F6,Sl6, cs='Greys', name='XQ · Quadrada 880Hz',
         hover='t=%{x:.2f}s · f=%{y:.0f}Hz · logE=%{z:.3f}<extra>XQ</extra>')
add_line(*gr6, '#00FF88', 'Grade R θ=63.4°')
vis_map['xq'] = list(range(trace_idx, trace_idx+2)); trace_idx += 2

# ── MODO 7: FM (FM_φ pura) ───────────────────────────────────────────────────
add_surf(T7,F7,Sl7, cs='Blues', name='FM · FM_φ',
         hover='t=%{x:.2f}s · f=%{y:.0f}Hz · logE=%{z:.3f}<extra>FM</extra>')
add_line(*gr7, '#00FF88', 'Grade R θ=63.4°')
vis_map['fm'] = list(range(trace_idx, trace_idx+2)); trace_idx += 2

# ── MODO 8: TXT_FILOS (φ-Narrativa Filosófica) ───────────────────────────────
add_surf(T8,F8,Sl8,
         cs=[[0,'rgb(30,15,0)'],[0.3,'rgb(120,70,0)'],[0.6,'rgb(200,140,20)'],[1,'rgb(255,210,80)']],
         name='φ Narrativa Filosófica',
         hover='pos=%{x:.0f}/%{x:.0f} · ciclos/jan=%{y:.2f} · logE=%{z:.3f}<extra>φ-Filos</extra>')
add_line(*phi_harm_lines_texto(fv8, tv8, Sl8.min()), '#FFD700', 'φ-Razões', width=2)
vis_map['txt_filos'] = list(range(trace_idx, trace_idx+2)); trace_idx += 2

# ── MODO 9: TXT_CODIGO (α-Linguagem de Programação) ─────────────────────────
add_surf(T9,F9,Sl9,
         cs=[[0,'rgb(0,0,20)'],[0.3,'rgb(0,30,80)'],[0.6,'rgb(10,80,160)'],[1,'rgb(80,180,255)']],
         name='α Linguagem de Programação',
         hover='pos=%{x:.0f}/%{x:.0f} · ciclos/jan=%{y:.2f} · logE=%{z:.3f}<extra>α-Código</extra>')
add_line(*phi_harm_lines_texto(fv9, tv9, Sl9.min()), '#44AAFF', 'φ-Razões', width=2)
vis_map['txt_codigo'] = list(range(trace_idx, trace_idx+2)); trace_idx += 2

# ── MODO 10: COMPARE — φ-Filos (piso) vs α-Código (teto +3) ─────────────────
# Normalização comum — mesma escala para comparação válida
Sl8_n = Sl8 / (max(Sl8.max(), Sl9.max()) + 1e-9)   # piso: 0 → ~1
Sl9_n = Sl9 / (max(Sl8.max(), Sl9.max()) + 1e-9)   # teto: offset +3
# Garantir que fv8/tv8 e fv9/tv9 têm o mesmo shape (ambos vêm de stft_texto)
T8c, F8c = np.meshgrid(tv8, fv8)
T9c, F9c = np.meshgrid(tv9, fv9)
add_surf(T8c, F8c, Sl8_n,
         cs=[[0,'rgb(20,10,0)'],[0.4,'rgb(140,80,0)'],[1,'rgb(255,210,60)']],
         opacity=0.90, name='φ Narrativa (piso)',
         hover='φ · pos=%{x:.0f} · ciclos=%{y:.2f} · z=%{z:.3f}<extra>φ-Filos</extra>')
add_surf(T9c, F9c, Sl9_n + 3.0,
         cs=[[0,'rgb(0,0,30)'],[0.4,'rgb(0,60,160)'],[1,'rgb(80,200,255)']],
         opacity=0.90, name='α Código (teto +3)',
         hover='α · pos=%{x:.0f} · ciclos=%{y:.2f} · z=%{z:.3f}<extra>α-Código</extra>')
# Linha divisória no limiar Z=1.5 (entre os dois andares)
_tx_c = np.linspace(tv8[0], tv8[-1], 60)
_fy_c = np.full(60, (fv8[0] + fv8[-1]) / 2)
_fz_c = np.full(60, 1.5)
fig.add_trace(go.Scatter3d(
    x=_tx_c, y=_fy_c, z=_fz_c, mode='lines',
    line=dict(color='rgba(180,180,180,0.35)', width=2, dash='dash'),
    name='Limiar φ|α', visible=False,
    hovertemplate='Limiar entre os dois modos<extra></extra>',
))
# Harmônicos φ em ambos os andares
ph_c_lo = phi_harm_lines_texto(fv8, tv8, 0.0,    dz=0.15)
ph_c_hi = phi_harm_lines_texto(fv9, tv9, 3.0,    dz=0.15)
add_line(*ph_c_lo, '#FFD700', 'φ-Razões (piso)', width=2)
add_line(*ph_c_hi, '#44AAFF', 'φ-Razões (teto)', width=2)
vis_map['compare'] = list(range(trace_idx, trace_idx+5)); trace_idx += 5

print(f"  {trace_idx} traces criados.")

# ─── Dropdown ─────────────────────────────────────────────────────────────────
MODOS = [
    ('eco',        'ECO — Coerência de Fase',
     'PLV (Phase Locking Value) como cor · log(E) como altura · EcoBIP 880Hz',
     dict(x=1.6, y=-1.6, z=0.85)),
    ('teto',       'TETO — Espaço Negativo (Sl_max − logE)',
     'Inverte o espectro: vales viram montanhas · revela estrutura oculta entre os picos',
     dict(x=1.6, y=-1.6, z=0.85)),
    ('unif',       'UNIF — Piso + Teto (visão dupla)',
     'Amplitude (piso, Z=0–1) + Espaço Negativo (teto, Z=2.5–3.5) simultaneamente',
     dict(x=1.8, y=-2.0, z=1.20)),
    ('lap',        'LAP — ∇²(Sl) Laplaciano',
     'Segunda derivada: bordas e transições geométricas · paleta RdBu · Z=|∇²|^0.45',
     dict(x=1.6, y=-1.6, z=0.85)),
    ('amp',        'AMP — Amplitude EcoBIP',
     'log(E) puro do sinal EcoBIP · colormap plasma',
     dict(x=1.6, y=-1.6, z=0.85)),
    ('xq',         'XQ — Quadrada 880Hz pura',
     'Sinal de referência: onda quadrada antes do EcoBIP',
     dict(x=1.6, y=-1.6, z=0.85)),
    ('fm',         'FM — FM_φ pura',
     'Componente FM_φ do EcoBIP isolado',
     dict(x=1.6, y=-1.6, z=0.85)),
    ('txt_filos',  'φ Narrativa Filosófica',
     'Texto sobre os dois modos de processamento · encoding transparente por classe de caractere',
     dict(x=1.4, y=-1.4, z=0.90)),
    ('txt_codigo', 'α Linguagem de Programação',
     'Funções do scanner em Python · mesmo encoding · mesma escala · observação agnóstica',
     dict(x=1.4, y=-1.4, z=0.90)),
    ('compare',   'COMPARE — φ vs α (visão dupla)',
     'φ-Narrativa (piso dourado) + α-Código (teto azul, +3) · mesma escala · observação agnóstica',
     dict(x=1.6, y=-1.8, z=1.10)),
]

n_total = trace_idx
buttons = []

for key, label, desc, cam in MODOS:
    vis_list = [False] * n_total
    for idx in vis_map[key]:
        vis_list[idx] = True

    x_label = 'Tempo (s)' if key not in ('txt_filos', 'txt_codigo', 'compare') else 'Posição no texto'
    y_label = 'Freq (Hz)' if key not in ('txt_filos', 'txt_codigo', 'compare') else 'Ciclos/janela'
    z_label = ('Sl_max−logE' if key == 'teto'
                else '∇²(Sl)' if key == 'lap'
                else 'Piso+Teto' if key == 'unif'
                else 'φ(piso) / α(teto)' if key == 'compare'
                else 'log(E)')

    buttons.append(dict(
        label=label,
        method='update',
        args=[
            {'visible': vis_list},
            {
                'title.text': (
                    f'<b>Scanner Topográfico · {label}</b><br>'
                    f'<span style="font-size:10px;color:#AAAAAA">{desc}</span><br>'
                    f'<span style="font-size:10px;color:#00FF88">'
                    f'φ={PHI}  α=1/137  α_op=1/3  θ_R={np.degrees(THETA_R):.2f}°</span>'
                ),
                'scene.xaxis.title': x_label,
                'scene.yaxis.title': y_label,
                'scene.zaxis.title': z_label,
                'scene.camera.eye': cam,
            }
        ],
    ))

# ─── Layout ───────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            '<b>Scanner Topográfico · ECO — Coerência de Fase</b><br>'
            '<span style="font-size:10px;color:#AAAAAA">'
            'PLV (Phase Locking Value) como cor · log(E) como altura · EcoBIP 880Hz</span><br>'
            f'<span style="font-size:10px;color:#00FF88">'
            f'φ={PHI}  α=1/137  α_op=1/3  θ_R={np.degrees(THETA_R):.2f}°</span>'
        ),
        font=dict(color='#EEEEEE', size=13),
        x=0.02,
    ),
    paper_bgcolor='#0D0D0D',
    plot_bgcolor='#0D0D0D',
    font=dict(color='#CCCCCC'),
    scene=dict(
        xaxis=dict(title='Tempo (s)', backgroundcolor='#111', gridcolor='#333',
                   showbackground=True, tickfont=dict(size=9)),
        yaxis=dict(title='Freq (Hz)', backgroundcolor='#111', gridcolor='#333',
                   showbackground=True, tickfont=dict(size=9)),
        zaxis=dict(title='log(E)', backgroundcolor='#111', gridcolor='#333',
                   showbackground=True, tickfont=dict(size=9)),
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.85)),
        aspectmode='manual',
        aspectratio=dict(x=2.0, y=1.2, z=0.7),
    ),
    updatemenus=[dict(
        type='dropdown',
        showactive=True,
        active=0,
        x=0.01, y=1.12,
        xanchor='left', yanchor='top',
        bgcolor='#1A1A2E',
        font=dict(color='#EEEEEE', size=11),
        bordercolor='#334',
        buttons=buttons,
    )],
    margin=dict(l=0, r=0, t=120, b=0),
    height=700,
    annotations=[dict(
        text=(
            f'φ: <b>{PHI}</b>  '
            f'α: <b>1/3</b>  '
            f'SEAL: <b>{SEAL:.6f}</b>  '
            f'θ<sub>R</sub>: <b>{np.degrees(THETA_R):.2f}°</b>'
        ),
        xref='paper', yref='paper',
        x=0.01, y=1.065,
        showarrow=False,
        font=dict(size=10, color='#AADDFF'),
        align='left',
    )],
)

fig.show()
print("\nScanner pronto. Use o dropdown para trocar entre os 9 modos.")
print(f"Grade R θ={np.degrees(THETA_R):.2f}° (linha verde)")
print(f"Harmônicos φ (linhas douradas)")
