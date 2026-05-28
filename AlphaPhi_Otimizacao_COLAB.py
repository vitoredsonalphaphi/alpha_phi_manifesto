"""
AlphaPhi_Otimizacao_COLAB.py
Métricas de Otimização Digital do ECO BEEP 880

Mede quantitativamente o quanto o ECO BEEP 880 otimiza um sinal digital
em múltiplas dimensões comparado ao sinal bruto.

Cinco estados comparados:
  [1] Beep puro         onda quadrada 880Hz — entropia máxima
  [2] x_mix             beep + fm — sinal de entrada do ECO
  [3] ECO BEEP 880      campo harmônico final
  [4] Serial φ          5 cones herméticos
  [5] ECO sobre Serial  campo de segunda ordem (β_ini=√5)

Métricas calculadas:
  H_espectral    Entropia de Shannon espectral (bits/coeficiente)
                 Medida direta de compressibilidade
  Compressão     Fator de compressão estimado vs beep puro
  Planura        Planura espectral (Wiener entropy): 0=tonal, 1=ruído
  BW_95          Largura de banda efetiva (95% da energia)
  Gini           Concentração de energia (Gini espectral): 1=máx concentração
  E_φ / E_¬φ    Energia dentro/fora das bandas φ
  PAPR           Peak-to-Average Power Ratio (eficiência de transmissão)
  Autocorr       Periodicidade (pico de autocorrelação normalizada)

Para rodar no Google Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_Otimizacao_COLAB.py').read())

© Vitor Edson Delavi · Florianópolis · maio 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")

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
FADE_S     = 0.15
N_SINAL    = int(FS * DURACAO)
N_CONES    = 5
DITHER_AMP = 1.0 / (PHI ** 5)

print("=" * 68)
print("  AlphaPhi · Métricas de Otimização Digital — ECO BEEP 880")
print("=" * 68)
print(f"  φ = {PHI:.7f}  /  φ³ = {PHI**3:.6f}")

# ── funções core ──────────────────────────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

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
             min(int(f_hi / (FS / n)) + 1, n // 2 + 1),
             f_lo, f_hi)
            for f_lo, f_hi in bandas]

BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb  = F[b_lo:b_hi]
        mag = np.abs(Fb); phase = np.angle(Fb)
        an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        ce  = (wn * coh + wm * float(coh_mem[i])
               if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce * PHI**bi) * np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)
    r = np.fft.irfft(F_out, n=N)
    return r / (np.max(np.abs(r)) + 1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    cm = np.zeros(len(bins_phi))
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se); cas.append(se); s = se.copy()
    return cas, cohs

def agente_eco(sinal, bins_phi, n_ciclos=N_CICLOS):
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    cas_f = None
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn * ba + wm * bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        cas_f = cas
    return beta, cas_f

def selar_hermetico(sinal, bins_phi=None):
    if bins_phi is None: bins_phi = BINS_PHI
    N = len(sinal); X = np.fft.rfft(sinal); X_h = np.zeros_like(X)
    for (b_lo, b_hi, _, _) in bins_phi:
        larg = b_hi - b_lo
        if larg > 4:
            fn = max(1, int(larg / PHI)); env = np.ones(larg)
            env[:fn]  *= np.sin(np.linspace(0, np.pi/2, fn))**2
            env[-fn:] *= np.cos(np.linspace(0, np.pi/2, fn))**2
            X_h[b_lo:b_hi] = X[b_lo:b_hi] * env
        else:
            X_h[b_lo:b_hi] = X[b_lo:b_hi]
    freq_res = FS / N
    for k in [1, 3, 5, 7]:
        bc = int(F_BEEP * k / freq_res)
        for db in range(-3, 4):
            bi = bc + db
            if 0 <= bi < len(X_h):
                X_h[bi] *= (1/PHI**2) if db == 0 else (1/PHI)
    return normalizar(np.fft.irfft(X_h, n=N))

def concatenar_phi(segs):
    fade_n = int(len(segs[0]) / PHI**2)
    out = segs[0].copy()
    for seg in segs[1:]:
        fn = min(fade_n, len(out), len(seg))
        t  = np.linspace(0.0, 1.0, fn)
        fo = np.cos(np.pi/2 * t**(1/PHI))
        fi = np.sin(np.pi/2 * t**(1/PHI))
        out[-fn:] = out[-fn:] * fo + seg[:fn] * fi
        out = np.concatenate([out, seg[fn:]])
    return normalizar(out)

# ── MÉTRICAS DE OTIMIZAÇÃO ────────────────────────────────────────────────────
def calcular_metricas(sinal, bins_phi=None, label=""):
    """Calcula todas as métricas de otimização para um sinal."""
    if bins_phi is None: bins_phi = BINS_PHI
    N   = len(sinal)
    X   = np.fft.rfft(sinal)
    mag = np.abs(X)
    pow_spec = mag ** 2

    # ── 1. Entropia de Shannon espectral (bits/coeficiente) ──────────────────
    p   = pow_spec / (pow_spec.sum() + 1e-10)
    p   = np.clip(p, 1e-12, 1.0)
    H   = float(-np.sum(p * np.log2(p)))  # bits
    H_max = np.log2(len(p))               # entropia máxima (uniforme)
    H_norm = H / H_max                    # 0=máx organização, 1=máx entropia

    # ── 2. Planura espectral (Wiener entropy) ────────────────────────────────
    # 0 = sinal tonal puro, 1 = ruído branco
    log_mean = np.exp(np.mean(np.log(pow_spec + 1e-12)))
    arith_mean = np.mean(pow_spec)
    planura = float(log_mean / (arith_mean + 1e-12))
    planura = np.clip(planura, 0.0, 1.0)

    # ── 3. Largura de banda efetiva (95% da energia) ─────────────────────────
    cumsum = np.cumsum(pow_spec)
    total  = cumsum[-1]
    idx_bw = np.searchsorted(cumsum, total * 0.95)
    bw_hz  = idx_bw * (FS / N)
    bw_frac = bw_hz / (FS / 2)  # fração do espectro total utilizado

    # ── 4. Coeficiente de Gini espectral ─────────────────────────────────────
    # Mede concentração: 0=uniforme, 1=toda energia num único bin
    s = np.sort(pow_spec)
    n = len(s)
    idx = np.arange(1, n + 1)
    gini = float(2 * np.sum(idx * s) / (n * s.sum() + 1e-12) - (n + 1) / n)
    gini = np.clip(gini, 0.0, 1.0)

    # ── 5. E_φ / E_¬φ ────────────────────────────────────────────────────────
    Xn    = mag / (mag.sum() + 1e-10)
    e_phi = sum(Xn[b_lo:b_hi].sum() for b_lo, b_hi, _, _ in bins_phi)
    e_fora = max(0.0, 1.0 - e_phi)

    # ── 6. PAPR (Peak-to-Average Power Ratio) ────────────────────────────────
    papr_linear = float(np.max(pow_spec) / (np.mean(pow_spec) + 1e-12))
    papr_db     = 10 * np.log10(papr_linear + 1e-12)

    # ── 7. Periodicidade (autocorrelação normalizada) ────────────────────────
    sinal_norm = sinal - sinal.mean()
    autocorr   = np.correlate(sinal_norm, sinal_norm, mode='full')
    autocorr   = autocorr[len(autocorr)//2:]
    autocorr_n = autocorr / (autocorr[0] + 1e-12)
    # pico máximo excluindo lag=0
    periodo_peak = float(np.max(autocorr_n[1:min(int(FS/F_BEEP)*5, len(autocorr_n))]))

    # ── 8. Fator de compressão estimado ──────────────────────────────────────
    # Baseado na entropia: quantos bits por amostra na representação φ-otimizada
    bits_por_amostra_original = H_norm * np.log2(N)
    # Taxa de compressão relativa à entropia máxima
    taxa_compressao = H_max / (H + 1e-6)

    return {
        'H':           H,
        'H_norm':      H_norm,
        'H_max':       H_max,
        'planura':     planura,
        'bw_hz':       bw_hz,
        'bw_frac':     bw_frac,
        'gini':        gini,
        'e_phi':       e_phi,
        'e_fora':      e_fora,
        'papr_db':     papr_db,
        'autocorr':    periodo_peak,
        'compressao':  taxa_compressao,
    }

# ── GERAR SINAIS ──────────────────────────────────────────────────────────────
print("\n  Gerando sinais...")
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep_q = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_sig)))  # onda quadrada pura
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_sig + BETA_FM*np.sin(2*np.pi*F_M*t_sig)))
x_mix = normalizar((1 - ALPHA_STAR) * beep_q + ALPHA_STAR * fm)

print("  [1] Beep puro...")
print("  [2] x_mix (já gerado)...")

print("  [3] ECO BEEP 880...")
beta_eco, cas_eco = agente_eco(x_mix, BINS_PHI, N_CICLOS)
eco_campo = cas_eco[-1]

print("  [4] Serial φ (5 cones)...")
rng = np.random.default_rng(seed=42)
cones_s = []
for i in range(N_CONES):
    d = rng.standard_normal(N_SINAL) * DITHER_AMP
    x = normalizar(x_mix + d * (1/PHI**i))
    _, cas = agente_eco(x, BINS_PHI, N_CICLOS)
    cones_s.append(selar_hermetico(cas[-1], BINS_PHI))
serial_phi = concatenar_phi(cones_s)
# usar o cone 5 (campo hermético individual) para métricas comparáveis
serial_cone5 = cones_s[-1]

print("  [5] ECO sobre Serial...")
N_SER    = len(serial_phi)
BINS_SER = bandas_para_bins(BANDAS, N_SER)
_, cas_es = agente_eco(serial_phi, BINS_SER, N_CICLOS)
eco_sobre_serial = cas_es[-1][:N_SINAL]  # primeiro segmento para comparação uniforme

# ── CALCULAR MÉTRICAS ────────────────────────────────────────────────────────
print("\n  Calculando métricas...")
estados = [
    ('[1] Beep puro',        beep_q,           '#888888'),
    ('[2] x_mix',            x_mix,            '#aaaaaa'),
    ('[3] ECO BEEP 880',     eco_campo,        '#00aaff'),
    ('[4] Serial φ (C5)',    serial_cone5,     '#00ffaa'),
    ('[5] ECO s/ Serial',    eco_sobre_serial, '#ffaa00'),
]

metricas = {}
for label, sig, cor in estados:
    m = calcular_metricas(sig, BINS_PHI, label)
    metricas[label] = (m, cor)

# ── TABELA PRINCIPAL ──────────────────────────────────────────────────────────
print("\n" + "="*68)
print("  MÉTRICAS DE OTIMIZAÇÃO DIGITAL — ECO BEEP 880")
print("="*68)

ref_H = metricas['[1] Beep puro'][0]['H']
ref_bw = metricas['[1] Beep puro'][0]['bw_hz']

print(f"\n  1. ARMAZENAMENTO — Entropia de Shannon espectral (bits/coef)")
print(f"     Quanto menor, mais compressível o sinal.")
print(f"  {'-'*60}")
for label, sig, cor in estados:
    m = metricas[label][0]
    ganho = (1 - m['H']/ref_H) * 100 if ref_H > 0 else 0
    bar = '█' * int((1 - m['H_norm']) * 30)
    sinal_g = f"+{ganho:.1f}%" if ganho >= 0 else f"{ganho:.1f}%"
    print(f"  {label:<22}  H={m['H']:6.2f}b  ({sinal_g:>7} vs bruto)  {bar}")

print(f"\n  2. COMPRESSÃO ESTIMADA (fator relativo ao beep puro)")
print(f"     Quanto maior, mais compressível vs beep bruto.")
print(f"  {'-'*60}")
ref_comp = metricas['[1] Beep puro'][0]['compressao']
for label, sig, cor in estados:
    m = metricas[label][0]
    fator = m['compressao'] / ref_comp
    bar = '█' * min(40, int(fator * 10))
    print(f"  {label:<22}  ×{fator:5.2f}  {bar}")

print(f"\n  3. TRÁFEGO / LARGURA DE BANDA EFETIVA (95% da energia)")
print(f"     Quanto menor, menos canal necessário para transmissão.")
print(f"  {'-'*60}")
for label, sig, cor in estados:
    m = metricas[label][0]
    reducao = (1 - m['bw_hz']/ref_bw) * 100 if ref_bw > 0 else 0
    bar = '█' * int((1 - m['bw_frac']) * 30)
    print(f"  {label:<22}  BW={m['bw_hz']:7.0f}Hz  ({reducao:+.1f}%)  {bar}")

print(f"\n  4. PLANURA ESPECTRAL (0=tonal puro, 1=ruído branco)")
print(f"     Quanto menor, mais organizado e mais eficiente para codificação.")
print(f"  {'-'*60}")
for label, sig, cor in estados:
    m = metricas[label][0]
    bar = '█' * int((1 - m['planura']) * 30)
    print(f"  {label:<22}  {m['planura']:.6f}  {bar}")

print(f"\n  5. CONCENTRAÇÃO DE ENERGIA (Gini espectral, 0=uniforme, 1=máx)")
print(f"     Quanto maior, mais energia concentrada = mais compressível.")
print(f"  {'-'*60}")
for label, sig, cor in estados:
    m = metricas[label][0]
    bar = '█' * int(m['gini'] * 30)
    print(f"  {label:<22}  {m['gini']:.6f}  {bar}")

print(f"\n  6. TRANSMISSÃO — PAPR (dB)  [menor = mais eficiente]")
print(f"  {'-'*60}")
for label, sig, cor in estados:
    m = metricas[label][0]
    print(f"  {label:<22}  {m['papr_db']:6.2f} dB")

print(f"\n  7. PERIODICIDADE (autocorrelação normalizada, 0-1)")
print(f"     Quanto maior, mais previsível = mais compressível.")
print(f"  {'-'*60}")
for label, sig, cor in estados:
    m = metricas[label][0]
    bar = '█' * int(m['autocorr'] * 30)
    print(f"  {label:<22}  {m['autocorr']:.6f}  {bar}")

print(f"\n  8. E_φ / E_¬φ (energia nas bandas φ)")
print(f"  {'-'*60}")
for label, sig, cor in estados:
    m = metricas[label][0]
    selo = "✓ hermético" if m['e_fora'] < 0.001 else f"  {m['e_fora']:.4f}"
    print(f"  {label:<22}  E_φ={m['e_phi']:.4f}  E_¬φ={m['e_fora']:.4f}  {selo}")

# ── SÍNTESE ───────────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("  SÍNTESE — GANHOS DO ECO BEEP 880 vs BEEP PURO")
print("="*68)
m_bruto = metricas['[1] Beep puro'][0]
m_eco   = metricas['[3] ECO BEEP 880'][0]
m_ser   = metricas['[4] Serial φ (C5)'][0]
m_ecs   = metricas['[5] ECO s/ Serial'][0]

print(f"""
  Armazenamento (entropia):
    ECO BEEP 880:    {(1 - m_eco['H']/m_bruto['H'])*100:+.1f}% de redução de entropia
    Serial φ:        {(1 - m_ser['H']/m_bruto['H'])*100:+.1f}% de redução de entropia
    ECO s/ Serial:   {(1 - m_ecs['H']/m_bruto['H'])*100:+.1f}% de redução de entropia

  Largura de banda (tráfego):
    ECO BEEP 880:    {(1 - m_eco['bw_hz']/m_bruto['bw_hz'])*100:+.1f}% de redução de BW efetiva
    Serial φ:        {(1 - m_ser['bw_hz']/m_bruto['bw_hz'])*100:+.1f}% de redução de BW efetiva
    ECO s/ Serial:   {(1 - m_ecs['bw_hz']/m_bruto['bw_hz'])*100:+.1f}% de redução de BW efetiva

  Concentração espectral (Gini):
    Beep puro:       {m_bruto['gini']:.4f}
    ECO BEEP 880:    {m_eco['gini']:.4f}  (+{(m_eco['gini']-m_bruto['gini']):.4f})
    Serial φ:        {m_ser['gini']:.4f}  (+{(m_ser['gini']-m_bruto['gini']):.4f})
    ECO s/ Serial:   {m_ecs['gini']:.4f}  (+{(m_ecs['gini']-m_bruto['gini']):.4f})

  Planura espectral (menor = mais organizado):
    Beep puro:       {m_bruto['planura']:.6f}
    ECO BEEP 880:    {m_eco['planura']:.6f}
    Serial φ:        {m_ser['planura']:.6f}
    ECO s/ Serial:   {m_ecs['planura']:.6f}
""")

# ── VISUALIZAÇÃO ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14), facecolor='#0a0a0a')
fig.suptitle('AlphaPhi · Métricas de Otimização Digital — ECO BEEP 880',
             color='white', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

nomes  = [l for l, s, c in estados]
cores  = [c for l, s, c in estados]
labels_curtos = ['Beep\npuro', 'x_mix', 'ECO\nBEEP', 'Serial\nφ', 'ECO s/\nSerial']

def bar_ax(ax, vals, titulo, ylabel, inv=False):
    ax.set_facecolor('#111111')
    vals_plot = [(1-v) if inv else v for v in vals]
    bars = ax.bar(range(len(nomes)), vals_plot, color=cores, alpha=0.85, width=0.6)
    ax.set_xticks(range(len(nomes)))
    ax.set_xticklabels(labels_curtos, color='#888888', fontsize=7)
    ax.set_title(titulo, color='white', fontsize=8.5)
    ax.set_ylabel(ylabel, color='#888888', fontsize=7)
    ax.tick_params(colors='#888888')
    for sp in ax.spines.values(): sp.set_color('#333333')
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.3f}', ha='center', va='bottom', color='white', fontsize=6.5)

H_vals      = [metricas[l][0]['H']        for l,s,c in estados]
comp_vals   = [metricas[l][0]['compressao']/ref_comp for l,s,c in estados]
bw_vals     = [metricas[l][0]['bw_frac']  for l,s,c in estados]
planura_vals= [metricas[l][0]['planura']  for l,s,c in estados]
gini_vals   = [metricas[l][0]['gini']     for l,s,c in estados]
papr_vals   = [metricas[l][0]['papr_db']  for l,s,c in estados]
autocorr_v  = [metricas[l][0]['autocorr'] for l,s,c in estados]
e_fora_vals = [metricas[l][0]['e_fora']   for l,s,c in estados]
e_phi_vals  = [min(metricas[l][0]['e_phi'],2.0) for l,s,c in estados]

bar_ax(fig.add_subplot(gs[0,0]), H_vals,       'Entropia Shannon (bits)\nmenor = mais compressível', 'bits', False)
bar_ax(fig.add_subplot(gs[0,1]), comp_vals,     'Fator de Compressão\nmaior = melhor vs bruto', '×', False)
bar_ax(fig.add_subplot(gs[0,2]), bw_vals,       'Largura de Banda Efetiva\nmenor = menos canal necessário', 'fração BW', False)
bar_ax(fig.add_subplot(gs[1,0]), planura_vals,  'Planura Espectral\n0=tonal, 1=ruído', 'planura', False)
bar_ax(fig.add_subplot(gs[1,1]), gini_vals,     'Gini Espectral\nmaior = mais concentrado', 'Gini', False)
bar_ax(fig.add_subplot(gs[1,2]), papr_vals,     'PAPR (dB)\nmenor = transmissão eficiente', 'dB', False)
bar_ax(fig.add_subplot(gs[2,0]), autocorr_v,   'Periodicidade\nmaior = mais previsível', 'autocorr', False)
bar_ax(fig.add_subplot(gs[2,1]), e_fora_vals,   'E_¬φ (energia fora φ)\nmenor = mais hermético', 'E_¬φ', False)
bar_ax(fig.add_subplot(gs[2,2]), e_phi_vals,    'E_φ (energia em bandas φ)\nmaior = mais organizado', 'E_φ', False)

plt.savefig('otimizacao.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
print("\n  Gráfico: otimizacao.png")
print("\n  Script de otimização concluído.")
