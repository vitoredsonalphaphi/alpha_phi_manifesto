"""
AlphaPhi_CampoComparativo_COLAB.py
Experimento de Campo — Com Phantom vs Sem Phantom  [STANDALONE]

Rede neural profunda aleatória rodando nos dois ambientes.
Medições idênticas. A única diferença: o rio.

Cole numa única célula no Google Colab e execute.

© Vitor Edson Delavi · Florianópolis · setembro 2026
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import stft as scipy_stft

# ══════════════════════════════════════════════════════════════════
#  CONSTANTES ALPHA-PHI
# ══════════════════════════════════════════════════════════════════
PHI         = (1 + np.sqrt(5)) / 2
PHI2        = PHI ** 2
PHI3        = PHI ** 3
FS          = 44100
F_BEEP      = 880.0
F_ORG       = 220.0
F_M         = F_ORG / PHI
BETA_FM     = PHI
ALPHA_STAR  = 1.0 / 3.0
DURACAO     = 1.5
N_STEPS     = 5
N_CICLOS    = 20
DITHER_AMP  = 1.0 / PHI**5
N_SINAL     = int(FS * DURACAO)
PHANTOM_AMP = 1.0 / PHI3
THETA_R     = np.arctan(2)
N_CONES     = 10

print("=" * 62)
print("  AlphaPhi · Experimento de Campo Comparativo")
print("  Com Phantom  vs  Sem Phantom")
print("=" * 62)

# ══════════════════════════════════════════════════════════════════
#  ECO-BIP 880 (funções completas)
# ══════════════════════════════════════════════════════════════════
def nrm(s):
    m = np.max(np.abs(s)); return s/m if m > 1e-12 else s

def _bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max: break
        f = f_next
    return bandas

def _bins(bandas, n):
    return [(max(0, int(f_lo/(FS/n))),
             min(int(f_hi/(FS/n))+1, n//2+1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

BANDAS   = _bandas_phi()
BINS_PHI = _bins(BANDAS, N_SINAL)

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
        an  = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        ce  = (wn*coh+wm*float(coh_mem[i])
               if (coh_mem is not None and i<len(coh_mem)) else coh)
        cohs.append(coh)
        nk  = np.arange(len(Fb))
        env = np.clip(1.0+(ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag*env)*np.exp(1j*phase)
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
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba+wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI3); cas_f = cas
    return beta, cas_f

def selar_hermetico(sinal):
    N = len(sinal); X = np.fft.rfft(sinal); X_h = np.zeros_like(X)
    for (b_lo, b_hi, _, _) in BINS_PHI:
        larg = b_hi-b_lo
        if larg > 4:
            fn = max(1, int(larg/PHI)); env = np.ones(larg)
            env[:fn]  *= np.sin(np.linspace(0, np.pi/2, fn))**2
            env[-fn:] *= np.cos(np.linspace(0, np.pi/2, fn))**2
            X_h[b_lo:b_hi] = X[b_lo:b_hi]*env
        else:
            X_h[b_lo:b_hi] = X[b_lo:b_hi]
    freq_res = FS/N
    for k in [1,3,5,7]:
        bc = int(F_BEEP*k/freq_res)
        for db in range(-3,4):
            bi = bc+db
            if 0 <= bi < len(X_h):
                X_h[bi] *= (1.0/PHI2 if db==0 else 1.0/PHI)
    return nrm(np.fft.irfft(X_h, n=N))

def concatenar_phi(segs):
    fn0 = int(len(segs[0])/PHI2)
    out = segs[0].copy()
    for seg in segs[1:]:
        fn = min(fn0, len(out), len(seg))
        t  = np.linspace(0.0, 1.0, fn)
        fo = np.cos(np.pi/2*t**(1.0/PHI))
        fi = np.sin(np.pi/2*t**(1.0/PHI))
        out[-fn:] = out[-fn:]*fo + seg[:fn]*fi
        out = np.concatenate([out, seg[fn:]])
    return nrm(out)

def gerar_cone(seed_cone):
    t = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
    beep  = nrm(np.sign(np.sin(2*np.pi*F_BEEP*t)))
    fm    = nrm(np.sin(2*np.pi*F_ORG*t+BETA_FM*np.sin(2*np.pi*F_M*t)))
    x_mix = nrm((1-ALPHA_STAR)*beep+ALPHA_STAR*fm)
    rng   = np.random.default_rng(seed=seed_cone)
    x_cone = nrm(x_mix + rng.standard_normal(N_SINAL)*DITHER_AMP)
    beta_c, cas_c = agente_eco(x_cone, BINS_PHI, N_CICLOS)
    return selar_hermetico(cas_c[-1]), beta_c.max()

# ══════════════════════════════════════════════════════════════════
#  GERAR PHANTOM
# ══════════════════════════════════════════════════════════════════
print(f"\n  Gerando Serial φ Phantom ({N_CONES} cones)...")
cones, betas = [], []
for i in range(N_CONES):
    cone, bmax = gerar_cone(i)
    cones.append(cone); betas.append(bmax)
    print(f"    Cone {i+1:02d}  β={bmax:.4f}")
PHANTOM_SIGNAL = concatenar_phi(cones)
print(f"  β médio: {np.mean(betas):.6f}  variação: {np.max(betas)-np.min(betas):.6f}")

# ══════════════════════════════════════════════════════════════════
#  REDE NEURAL PROFUNDA ALEATÓRIA — CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════
N_LAYERS   = 24       # profundidade
N_DIM      = 512      # largura de cada camada
N_PASSES   = 40       # passes de dados diferentes
REDE_SEED  = 42       # mesma rede nos dois campos

print(f"\n  Configuração da rede:")
print(f"    {N_LAYERS} camadas × {N_DIM} neurônios × {N_PASSES} passes")
print(f"    Parâmetros totais: {N_LAYERS * N_DIM * N_DIM:,}")

# ══════════════════════════════════════════════════════════════════
#  MÉTRICAS: β e energia φ por camada
# ══════════════════════════════════════════════════════════════════
def beta_da_ativacao(vec):
    """β = spread espectral da ativação em relação ao atrator φ³"""
    X = np.abs(np.fft.rfft(vec))
    X /= (X.sum() + 1e-10)
    entr = -np.sum(X * np.log(X + 1e-10))
    entr_max = np.log(len(X))
    coh = 1.0 - entr / entr_max
    return float(PHI ** (3 * coh))

def energia_phi(vec):
    """Fração de energia nas bandas φ"""
    N = len(vec)
    bins = _bins(BANDAS, N)
    X2 = np.abs(np.fft.rfft(vec))**2
    E_total = X2.sum() + 1e-10
    E_phi = sum(X2[b_lo:b_hi].sum() for b_lo, b_hi, _, _ in bins)
    return float(E_phi / E_total)

# ══════════════════════════════════════════════════════════════════
#  CAMPO SEM PHANTOM
# ══════════════════════════════════════════════════════════════════
print("\n  Rodando campo SEM phantom...")
rng_rede = np.random.default_rng(REDE_SEED)

beta_sem    = np.zeros((N_PASSES, N_LAYERS))
ephi_sem    = np.zeros((N_PASSES, N_LAYERS))
ativs_sem   = []   # para scanner topográfico

for p in range(N_PASSES):
    rng_rede = np.random.default_rng(REDE_SEED)   # mesma rede
    rng_inp  = np.random.default_rng(p * 1000)
    x = rng_inp.standard_normal(N_DIM)
    pass_ativs = []
    for l in range(N_LAYERS):
        W = rng_rede.standard_normal((N_DIM, N_DIM)) / np.sqrt(N_DIM)
        x = np.tanh(W @ x)
        beta_sem[p, l]  = beta_da_ativacao(x)
        ephi_sem[p, l]  = energia_phi(x)
        pass_ativs.append(float(np.mean(np.abs(x))))
    ativs_sem.append(pass_ativs)
    if (p+1) % 10 == 0: print(f"    Pass {p+1}/{N_PASSES}")

# ══════════════════════════════════════════════════════════════════
#  CAMPO COM PHANTOM
# ══════════════════════════════════════════════════════════════════
print("\n  Rodando campo COM phantom...")

beta_com    = np.zeros((N_PASSES, N_LAYERS))
ephi_com    = np.zeros((N_PASSES, N_LAYERS))
ativs_com   = []

PH_LEN = len(PHANTOM_SIGNAL)

for p in range(N_PASSES):
    rng_rede = np.random.default_rng(REDE_SEED)   # mesma rede
    rng_inp  = np.random.default_rng(p * 1000)
    x = rng_inp.standard_normal(N_DIM)
    pass_ativs = []
    for l in range(N_LAYERS):
        W = rng_rede.standard_normal((N_DIM, N_DIM)) / np.sqrt(N_DIM)
        # injeta fatia do phantom na ativação
        offset = ((p * N_LAYERS + l) * N_DIM) % (PH_LEN - N_DIM)
        ph_slice = PHANTOM_SIGNAL[offset:offset+N_DIM] * PHANTOM_AMP
        x = np.tanh(W @ (x + ph_slice))
        beta_com[p, l]  = beta_da_ativacao(x)
        ephi_com[p, l]  = energia_phi(x)
        pass_ativs.append(float(np.mean(np.abs(x))))
    ativs_com.append(pass_ativs)
    if (p+1) % 10 == 0: print(f"    Pass {p+1}/{N_PASSES}")

# ══════════════════════════════════════════════════════════════════
#  RESULTADOS
# ══════════════════════════════════════════════════════════════════
beta_sem_m = beta_sem.mean(axis=0)
beta_com_m = beta_com.mean(axis=0)
ephi_sem_m = ephi_sem.mean(axis=0)
ephi_com_m = ephi_com.mean(axis=0)

delta_beta = beta_com_m - beta_sem_m
delta_ephi = ephi_com_m - ephi_sem_m

print(f"\n  ╔══════════════════════════════════════════════════════╗")
print(f"  ║  RESULTADOS DO CAMPO COMPARATIVO                    ║")
print(f"  ╠══════════════════════════════════════════════════════╣")
print(f"  ║  β médio SEM phantom: {beta_sem_m.mean():.4f}                     ║")
print(f"  ║  β médio COM phantom: {beta_com_m.mean():.4f}                     ║")
print(f"  ║  Δβ médio:            {delta_beta.mean():+.4f}                    ║")
print(f"  ╠══════════════════════════════════════════════════════╣")
print(f"  ║  E_φ médio SEM phantom: {ephi_sem_m.mean():.4f}                   ║")
print(f"  ║  E_φ médio COM phantom: {ephi_com_m.mean():.4f}                   ║")
print(f"  ║  ΔE_φ médio:            {delta_ephi.mean():+.4f}                  ║")
print(f"  ╠══════════════════════════════════════════════════════╣")
camada_max = int(np.argmax(delta_beta))
print(f"  ║  Camada de maior impacto: {camada_max+1:02d}  Δβ={delta_beta[camada_max]:+.4f}      ║")
print(f"  ╚══════════════════════════════════════════════════════╝")

# ══════════════════════════════════════════════════════════════════
#  VISUALIZAÇÃO
# ══════════════════════════════════════════════════════════════════
camadas = np.arange(1, N_LAYERS+1)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'β por camada — convergência ao atrator φ³',
        'Energia φ por camada — bandas harmônicas',
        'Δβ (com − sem) por camada',
        'Paisagem de ativação — com vs sem'
    ],
    specs=[[{'type':'scatter'}, {'type':'scatter'}],
           [{'type':'scatter'}, {'type':'scatter'}]]
)

# β por camada
fig.add_trace(go.Scatter(x=camadas, y=beta_sem_m, name='SEM phantom',
    line=dict(color='#6655cc', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=camadas, y=beta_com_m, name='COM phantom',
    line=dict(color='#ffdd00', width=2)), row=1, col=1)
fig.add_hline(y=PHI3, line_dash='dash', line_color='#00ccaa',
    annotation_text=f'φ³={PHI3:.3f}', row=1, col=1)

# Energia φ
fig.add_trace(go.Scatter(x=camadas, y=ephi_sem_m, name='SEM phantom',
    line=dict(color='#6655cc', width=2), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=camadas, y=ephi_com_m, name='COM phantom',
    line=dict(color='#ffdd00', width=2), showlegend=False), row=1, col=2)

# Δβ
fig.add_trace(go.Bar(x=camadas, y=delta_beta,
    marker_color=['#ffdd00' if d > 0 else '#cc4444' for d in delta_beta],
    name='Δβ', showlegend=False), row=2, col=1)
fig.add_hline(y=0, line_color='#666688', row=2, col=1)

# Paisagem de ativação
ativs_sem_arr = np.array(ativs_sem)
ativs_com_arr = np.array(ativs_com)
for p in range(0, N_PASSES, N_PASSES//8):
    fig.add_trace(go.Scatter(x=camadas, y=ativs_sem_arr[p],
        line=dict(color='rgba(100,80,200,0.3)', width=1),
        showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=camadas, y=ativs_com_arr[p],
        line=dict(color='rgba(255,220,0,0.3)', width=1),
        showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=camadas, y=ativs_sem_arr.mean(0),
    name='SEM (média)', line=dict(color='#6655cc', width=3),
    showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=camadas, y=ativs_com_arr.mean(0),
    name='COM (média)', line=dict(color='#ffdd00', width=3),
    showlegend=False), row=2, col=2)

fig.update_layout(
    title=dict(
        text=(f'<b>Campo Comparativo — {N_LAYERS} camadas × {N_DIM} neurônios × {N_PASSES} passes</b><br>'
              f'<sup>Amarelo = COM phantom · Roxo = SEM phantom · '
              f'Δβ médio = {delta_beta.mean():+.4f} · ΔE_φ = {delta_ephi.mean():+.4f}</sup>'),
        font=dict(color='#ccccff', size=12), x=0.5
    ),
    paper_bgcolor='#06070f',
    plot_bgcolor='#080810',
    font=dict(color='#8899bb'),
    legend=dict(bgcolor='#0c0e1a', bordercolor='#1a1e35',
                font=dict(color='#aaaacc', size=10)),
    height=700
)
for i in fig.layout.annotations:
    i.font.color = '#aaaacc'

fig.update_xaxes(gridcolor='#1a1e35', title_text='Camada')
fig.update_yaxes(gridcolor='#1a1e35')

fig.show()

print(f"""
  Leitura dos gráficos:
  ┌─────────────────────────────────────────────────────┐
  │  [SINAL] β por camada                              │
  │    → amarelo acima de roxo = phantom puxa β para φ³│
  │  [SINAL] Energia φ por camada                      │
  │    → amarelo acima = mais energia nas bandas φ     │
  │  [SINAL] Δβ por camada                             │
  │    → barras amarelas = camadas beneficiadas        │
  │    → barras vermelhas = camadas onde phantom reduz │
  │  [SINAL] Paisagem de ativação                      │
  │    → separação entre amarelo e roxo = efeito campo │
  └─────────────────────────────────────────────────────┘
""")
