# AlphaPhi_Medicao_Shannon.py
# Medição Shannon — Comparação EcoBIP × FM Convencional × Quadrada
# Métricas: H (entropia espectral), ESO (energia sub-harmônica),
#            DGR (densidade Grade R), ICφ (coerência áurea)
# Saída: tabela numérica + gráfico comparativo PNG
# Vitor Edson Delavi · Florianópolis · 2026
# © CC BY-NC-ND 4.0

import numpy as np
from scipy.signal import stft
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib', '-q'])
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

# ─── Constantes irrevogáveis ──────────────────────────────────────────────────
PHI     = 1.6180339887
ALPHA   = 1 / 137.035999
SEAL    = 1 / PHI
THETA_R = np.arctan(2.0)
SR      = 44100
DUR     = 30.0
N_SIG   = int(SR * DUR)
BASE    = 880.0
F_MAX   = 5000.0
t       = np.linspace(0, DUR, N_SIG, endpoint=False)

print(f"φ={PHI}  α={ALPHA:.8f}  θ_R={np.degrees(THETA_R):.2f}°")
print(f"SR={SR}Hz  DUR={DUR}s  BASE={BASE}Hz  F_MAX={F_MAX}Hz\n")

def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)

# ─── Três sinais ──────────────────────────────────────────────────────────────
def gerar_quadrada():
    return _norm(np.sign(np.sin(2 * np.pi * BASE * t)))

def gerar_fm_convencional():
    # FM padrão: carrier 880Hz, modulador 220Hz, índice β=2.5 — sem α, sem φ
    beta = 2.5
    mod  = np.sin(2 * np.pi * (BASE / 4) * t)
    return _norm(np.sin(2 * np.pi * BASE * t + beta * mod))

def gerar_ecobip():
    # EcoBIP INVARIANTE: (1-α)×quadrada + α×FM_φ_220
    quad      = np.sign(np.sin(2 * np.pi * BASE * t))
    fm_phi    = np.sin(2 * np.pi * BASE * t + PHI * np.sin(2 * np.pi * (BASE / 4) * t))
    return _norm((1 - ALPHA) * quad + ALPHA * fm_phi)

print("Gerando sinais (30s cada)...")
sinais = {
    'Quadrada':       gerar_quadrada(),
    'FM Convencional': gerar_fm_convencional(),
    'EcoBIP':         gerar_ecobip(),
}
print("  Sinais prontos.\n")

# ─── Métricas ─────────────────────────────────────────────────────────────────

def H_espectral(x):
    """Entropia espectral de Shannon — menor H = espectro mais organizado."""
    freqs = np.fft.rfftfreq(N_SIG, 1 / SR)
    psd   = np.abs(np.fft.rfft(x))**2
    mask  = (freqs > 0) & (freqs <= F_MAX)
    p     = psd[mask] / (psd[mask].sum() + 1e-12)
    return float(-np.sum(p * np.log2(p + 1e-12)))

def ESO(x):
    """Energia Sub-harmônica Organizada — E(f < BASE) / E(f <= F_MAX)."""
    freqs = np.fft.rfftfreq(N_SIG, 1 / SR)
    psd   = np.abs(np.fft.rfft(x))**2
    sub   = psd[(freqs > 0) & (freqs < BASE)].sum()
    tot   = psd[(freqs > 0) & (freqs <= F_MAX)].sum()
    return float(sub / (tot + 1e-12))

def ICphi(x):
    """Índice de Coerência φ — energia nos harmônicos φ^k × BASE / E_total."""
    freqs      = np.fft.rfftfreq(N_SIG, 1 / SR)
    psd        = np.abs(np.fft.rfft(x))**2
    bw_bins    = max(2, int(30 / (SR / N_SIG)))  # ±30Hz ao redor de cada harmônico
    phi_energy = 0.0
    for k in range(-5, 9):
        f_phi = BASE * PHI**k
        if 0 < f_phi <= F_MAX:
            idx = int(np.argmin(np.abs(freqs - f_phi)))
            lo  = max(0, idx - bw_bins)
            hi  = min(len(psd), idx + bw_bins + 1)
            phi_energy += psd[lo:hi].sum()
    tot = psd[(freqs > 0) & (freqs <= F_MAX)].sum()
    return float(phi_energy / (tot + 1e-12))

def DGR(x):
    """Densidade Grade R — vértices por segundo."""
    win = int(SR / BASE * 2 * PHI)
    win = max(512, min(win, 4096))
    hop = win // 4
    freqs, times, Zxx = stft(x, fs=SR, window='hann',
                              nperseg=win, noverlap=win - hop)
    S     = np.abs(Zxx)**2
    fmask = freqs <= F_MAX
    fv, Sv = freqs[fmask], S[fmask]
    Sl    = np.log1p(Sv * 100)
    gradS = Sl - gaussian_filter(Sl, sigma=2.0)

    sf = max(1, len(fv) // 120)
    st = max(1, len(times) // 100)
    fv_d, tv_d = fv[::sf], times[::st]
    Sl_d  = Sl[::sf, ::st]
    gS_d  = gradS[::sf, ::st]

    thr_sl = Sl_d.mean() + 0.45 * Sl_d.std()
    thr_gs = 0.12
    f_exp  = np.tan(THETA_R) * (tv_d - tv_d[0]) / (tv_d[-1] - tv_d[0] + 1e-9) * fv_d[-1]
    f_exp  = np.clip(f_exp, fv_d[0], fv_d[-1])

    n_vertices = 0
    for ti_i, (ti, fe) in enumerate(zip(tv_d, f_exp)):
        bw    = fe * 0.07 + 40.0
        fi_lo = int(np.searchsorted(fv_d, fe - bw))
        fi_hi = int(np.searchsorted(fv_d, fe + bw))
        for fi_i in range(fi_lo, min(fi_hi, len(fv_d))):
            if gS_d[fi_i, ti_i] > thr_gs and Sl_d[fi_i, ti_i] > thr_sl:
                n_vertices += 1

    return float(n_vertices / DUR)

# ─── Calcular todas as métricas ───────────────────────────────────────────────
print("Calculando métricas...\n")
resultados = {}
for nome, x in sinais.items():
    print(f"  {nome}...")
    resultados[nome] = {
        'H':    H_espectral(x),
        'ESO':  ESO(x),
        'ICφ':  ICphi(x),
        'DGR':  DGR(x),
    }

# ─── Tabela de resultados ─────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'MEDIÇÃO SHANNON — Alpha-Phi':^65}")
print(f"{'EcoBIP × FM Convencional × Quadrada':^65}")
print("="*65)
print(f"{'Métrica':<22} {'Quadrada':>12} {'FM Convenc.':>12} {'EcoBIP':>12}")
print("-"*65)

metricas = [
    ('H (entropia)',  'H',   '{:.4f}', 'menor = mais organizado'),
    ('ESO (sub-harm)', 'ESO', '{:.4f}', 'maior = mais volatilização'),
    ('ICφ (coer. áurea)', 'ICφ', '{:.6f}', 'maior = mais estrutura φ'),
    ('DGR (Grade R/s)', 'DGR', '{:.4f}', 'maior = mais formação Grade R'),
]

for label, chave, fmt, desc in metricas:
    q  = resultados['Quadrada'][chave]
    fm = resultados['FM Convencional'][chave]
    eb = resultados['EcoBIP'][chave]
    print(f"{label:<22} {fmt.format(q):>12} {fmt.format(fm):>12} {fmt.format(eb):>12}   ← {desc}")

print("="*65)

# ─── Análise comparativa ──────────────────────────────────────────────────────
print("\nANÁLISE COMPARATIVA:")
eb = resultados['EcoBIP']
fm = resultados['FM Convencional']
q  = resultados['Quadrada']

print(f"\n  H: EcoBIP {'<' if eb['H'] < fm['H'] else '>'} FM Convencional "
      f"({'−' if eb['H'] < fm['H'] else '+'}{abs(eb['H']-fm['H']):.4f})")
print(f"     Hipótese Shannon: {'SUPORTADA' if eb['H'] < fm['H'] else 'NÃO SUPORTADA'} "
      f"— EcoBIP {'tem menor' if eb['H'] < fm['H'] else 'NÃO tem menor'} entropia")

print(f"\n  ESO: EcoBIP {'>' if eb['ESO'] > fm['ESO'] else '<'} FM Convencional "
      f"({'+'  if eb['ESO'] > fm['ESO'] else '−'}{abs(eb['ESO']-fm['ESO']):.4f})")
print(f"     Volatilização: {'CONFIRMADA' if eb['ESO'] > fm['ESO'] else 'NÃO CONFIRMADA'} "
      f"— EcoBIP {'concentra mais' if eb['ESO'] > fm['ESO'] else 'NÃO concentra mais'} energia sub-harmônica")

print(f"\n  DGR: EcoBIP {'>' if eb['DGR'] > fm['DGR'] else '<'} FM Convencional "
      f"({'+'  if eb['DGR'] > fm['DGR'] else '−'}{abs(eb['DGR']-fm['DGR']):.4f} vért/s)")
print(f"     Grade R: {'DIFERENCIADA' if eb['DGR'] > fm['DGR'] else 'NÃO DIFERENCIADA'} "
      f"— EcoBIP {'forma mais' if eb['DGR'] > fm['DGR'] else 'NÃO forma mais'} Grade R")

# ─── Gráfico comparativo ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8), facecolor='#0A0A14')
fig.suptitle('Medição Shannon — EcoBIP × FM Convencional × Quadrada\n'
             f'θ_R={np.degrees(THETA_R):.2f}°  ·  φ={PHI:.7f}  ·  α={ALPHA:.6f}',
             color='white', fontsize=12, y=0.98)

gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                         left=0.08, right=0.96, top=0.90, bottom=0.08)
axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]

cores  = ['#888888', '#44AAFF', '#00FFAA']
nomes  = ['Quadrada', 'FM Conv.', 'EcoBIP']
chaves = ['H', 'ESO', 'ICφ', 'DGR']
titulos = [
    'H — Entropia Espectral\n(↓ menor = mais organizado)',
    'ESO — Energia Sub-harmônica\n(↑ maior = mais volatilização)',
    'ICφ — Coerência Áurea\n(↑ maior = mais estrutura φ)',
    'DGR — Densidade Grade R (vért/s)\n(↑ maior = mais formação Grade R)',
]

for ax, chave, titulo in zip(axes, chaves, titulos):
    vals = [resultados['Quadrada'][chave],
            resultados['FM Convencional'][chave],
            resultados['EcoBIP'][chave]]
    bars = ax.bar(nomes, vals, color=cores, width=0.55,
                  edgecolor='#333355', linewidth=0.8)

    # destacar EcoBIP
    bars[2].set_edgecolor('#00FFAA')
    bars[2].set_linewidth(2.0)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{val:.4f}', ha='center', va='bottom',
                color='white', fontsize=8)

    ax.set_title(titulo, color='#CCCCCC', fontsize=9, pad=6)
    ax.set_facecolor('#06060F')
    ax.tick_params(colors='#888888', labelsize=8)
    ax.spines['bottom'].set_color('#333355')
    ax.spines['left'].set_color('#333355')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('#888888')

out_png = 'medicao_shannon.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='#0A0A14')
print(f"\nGráfico salvo: {out_png}")
plt.close()
print("\nPara visualizar no Colab:")
print("  from IPython.display import Image")
print(f"  display(Image('{out_png}'))")
