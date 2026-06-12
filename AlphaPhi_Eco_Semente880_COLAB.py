# AlphaPhi_Eco_Semente880_COLAB.py
# Vitor Edson Delavi · Florianopolis · 2026
#
# ECO-RESSONANTE 880Hz + SEMENTE ALPHA/PHI
#
# Integra os 5 modulos identificados nos dialogos:
#
#   M1  semente_alpha_banda(mag)
#         an = clip(mag/sum, ALPHA_FINE, 1.0)
#         H_alpha = H / log(137)
#         coh_alpha = 1 - H_alpha
#         (mede entropia em alpha-nats — regua irrevogavel)
#
#   M2  eco_entropico_scale(H_alpha, H_ema)
#         delta = H_alpha - H_ema
#         E (periferica):  scale = clip(1 + delta*phi, 1, phi^2)   ancora
#         X (nuclear):     scale = clip(1 - |delta|/phi, 0.1, 1)   expande
#         (feedback sobre a POSICAO da entropia no triangulo)
#
#   M3  triangulacao_vertex(H_alpha, beta)
#         identifica em qual vertice o sistema esta:
#         "campo" (beta>=phi^3) | "periferica" (H>0.6) | "nuclear" (H<=0.6)
#
#   M4  eco_eq_semente — passo principal
#         usa coh_alpha (nativo-alpha) como driver do envelope
#         substitui coh_raw do eco_eq_passo original
#
#   M5  alpha_estrela emergente
#         H_alpha_equilibrio por substrato — onde H_ema para de oscilar
#         (registrado no final de cada config)
#
# ABLACAO:
#   AA: coh_raw  + sem eco_entropico  (baseline classico — regressao)
#   BB: coh_alpha + sem eco_entropico  (semente so medindo)
#   CC: coh_alpha + eco_entropico      (feedback de entropia ativo)
#   DD: coh_alpha + eco_entropico + roldana N=137
#
# PERGUNTAS:
#   (1) R_natural ainda converge a phi em AA? (regressao)
#   (2) BB vs AA: coh_alpha vs coh_raw muda R_natural?
#   (3) CC: eco_entropico acelera a formacao do campo?
#   (4) H_alpha_equilibrio do substrato 880Hz (alpha* no eixo de entropia)
#   (5) DD: roldana + eco_entropico — sinergia?

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
ALPHA_FINE = 1 / 137.035999084
LOG_ALPHA  = np.log(1.0 / ALPHA_FINE)   # log(137) ~= 4.920
N_BASE     = int(round(1.0 / ALPHA_FINE))  # 137

FS       = 44100
F_BEEP   = 880.0
F_ORG    = 220.0
F_M      = F_ORG / PHI
BETA_FM  = PHI
DURACAO  = 1.5
N_STEPS  = 5
N_CICLOS = 20
N_OBS    = 10
LIMIAR   = 0.99 * PHI**3

wm = 1.0 / PHI
wn = 1.0 - 1.0 / PHI

print(f"phi          = {PHI:.10f}")
print(f"alpha        = {ALPHA_FINE:.10f}  (1/{N_BASE})")
print(f"log(1/alpha) = {LOG_ALPHA:.6f}  (regua de entropia)")
print(f"phi^3        = {PHI**3:.6f}  (limiar campo harmonico)")
print(f"\nModulos acoplados:")
print(f"  M1 semente_alpha_banda  — H_alpha em alpha-nats")
print(f"  M2 eco_entropico_scale  — feedback sobre posicao da entropia")
print(f"  M3 triangulacao_vertex  — diagnostico: campo/periferica/nuclear")
print(f"  M4 eco_eq_semente       — coh_alpha como driver do envelope")
print(f"  M5 alpha_estrela        — H_alpha_equilibrio por substrato")
print("=" * 65)

# ─── Funcoes base ─────────────────────────────────────────────────────────────
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
             min(int(f_hi / (FS / n)) + 1, n // 2 + 1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

def gerar_beep(duracao, alpha_sinal=1.0 / 3.0):
    N = int(FS * duracao)
    t = np.linspace(0, duracao, N, endpoint=False)
    beep = np.sign(np.sin(2 * np.pi * F_BEEP * t))
    fm   = np.sin(2 * np.pi * F_M * t + BETA_FM * np.sin(2 * np.pi * F_M * t / PHI))
    return normalizar(alpha_sinal * beep + (1 - alpha_sinal) * fm)

def medir_R(cohs):
    c      = np.asarray(cohs, dtype=float)
    E_geom = float(np.mean((1.0 - c)**2))
    E_coer = float(np.mean(c**2))
    return E_geom / (E_coer + 1e-10)

# ─── M1: Semente alpha ────────────────────────────────────────────────────────
def semente_alpha_banda(mag):
    """Mede entropia em alpha-nats. Retorna (H_alpha, coh_alpha)."""
    an = np.clip(mag / (mag.sum() + 1e-8), ALPHA_FINE, 1.0)
    an = an / an.sum()
    H  = -np.sum(an * np.log(an + 1e-15))
    H_alpha   = float(np.clip(H / LOG_ALPHA, 0.0, 1.0))
    coh_alpha = 1.0 - H_alpha
    return H_alpha, coh_alpha

def coh_raw_banda(mag):
    """Coerencia classica: H / log(N_bins). Usada em AA."""
    an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
    an  = an / an.sum()
    H   = -np.sum(an * np.log(an + 1e-15))
    coh = float(np.clip(1.0 - H / np.log(max(len(an), 2)), 0.0, 1.0))
    return coh

# ─── M2: Eco-entropico ────────────────────────────────────────────────────────
def eco_entropico_scale(H_alpha, H_ema):
    """
    Feedback sobre a posicao da entropia no triangulo alpha/phi.
    H_alpha > H_ema: entropia periferica → ancora (E)
    H_alpha < H_ema: entropia nuclear    → expande (X)
    """
    delta = H_alpha - H_ema
    if delta > 0:
        scale = float(np.clip(1.0 + delta * PHI, 1.0, PHI**2))
    else:
        scale = float(np.clip(1.0 - abs(delta) / PHI, 0.1, 1.0))
    return scale

# ─── M3: Triangulacao ────────────────────────────────────────────────────────
def triangulacao_vertex(H_alpha, beta):
    """Qual vertice do triangulo alpha-phi?"""
    if beta >= PHI**3:
        return "campo"
    elif H_alpha > 0.6:
        return "periferica"
    else:
        return "nuclear"

# ─── M4: Passo principal — eco_eq_semente ────────────────────────────────────
def eco_eq_semente(x, bins_phi, beta_bands, coh_mem, H_ema_mem,
                   usar_eco=False, ce_scale_ext=1.0, usar_coh_raw=False):
    """
    Passo do equalizador com semente alpha.

    usar_coh_raw=True : AA baseline (coh_raw, sem semente)
    usar_eco=False    : BB (coh_alpha so medindo, ce_scale fixo)
    usar_eco=True     : CC/DD (eco_entropico ativo)
    ce_scale_ext      : roldana phi-alternada externa (DD)

    Retorna: (sinal, cohs, H_alphas, H_ema_novo)
    """
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    coh_mem    = np.atleast_1d(np.asarray(coh_mem,    dtype=float))
    H_ema_mem  = np.atleast_1d(np.asarray(H_ema_mem,  dtype=float))

    N, F  = len(x), np.fft.rfft(x)
    F_out = F.copy()
    cohs, H_alphas, H_emas_new = [], [], []

    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi    = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb    = F[b_lo:b_hi]
        mag   = np.abs(Fb)
        phase = np.angle(Fb)

        if usar_coh_raw:
            # AA: coerencia classica
            coh   = coh_raw_banda(mag)
            H_a   = 1.0 - coh
            H_ema = float(H_ema_mem[i]) if i < len(H_ema_mem) else 0.5
            ce_eco = 1.0
        else:
            # BB/CC/DD: semente alpha
            H_a, coh = semente_alpha_banda(mag)
            H_ema = float(H_ema_mem[i]) if i < len(H_ema_mem) else H_a
            ce_eco = eco_entropico_scale(H_a, H_ema) if usar_eco else 1.0

        # EMA de H_alpha (memoria do estado entropico)
        H_ema_new = wn * H_a + wm * H_ema

        # Memoria de coerencia
        cm = float(coh_mem[i]) if i < len(coh_mem) else coh
        ce = wn * coh + wm * cm

        # ce_scale total: eco_entropico * roldana externa
        ce_total = float(np.clip(ce * ce_eco * ce_scale_ext, 0.0, PHI**2))

        cohs.append(coh)
        H_alphas.append(H_a)
        H_emas_new.append(H_ema_new)

        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce_total * PHI**bi) * np.cos(2*np.pi*nk/PHI),
                      0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)

    r = np.fft.irfft(F_out, n=N)
    return (r / (np.max(np.abs(r)) + 1e-10),
            np.array(cohs), np.array(H_alphas), np.array(H_emas_new))

# ─── Cascata phi-alternada com semente ───────────────────────────────────────
def cascata_semente(sinal, beta_bands, bins_phi, coh_mem_init, H_ema_init,
                    N_base_rol=0, usar_eco=False, usar_coh_raw=False):
    s      = sinal.copy()
    cm     = coh_mem_init.copy()
    H_ema  = H_ema_init.copy()

    cohs_final = cm.copy()
    H_alphas_final = H_ema.copy()

    for k in range(N_STEPS):
        if N_base_rol == 0:
            ce_ext = 1.0
        else:
            Nk    = float(N_base_rol) / (PHI**k)
            delta = Nk * ALPHA_FINE
            if k % 2 == 0:
                ce_ext = float(np.clip(1.0 + delta, 1.0, PHI**2))
            else:
                ce_ext = float(np.clip(1.0 - delta / PHI, 0.1, 1.0))

        se, cohs, H_alphas, H_ema_new = eco_eq_semente(
            s, bins_phi, beta_bands, cm, H_ema,
            usar_eco=usar_eco, ce_scale_ext=ce_ext,
            usar_coh_raw=usar_coh_raw
        )
        cm     = cohs
        H_ema  = H_ema_new
        s      = normalizar(se)
        cohs_final    = cohs
        H_alphas_final = H_alphas

    return s, cohs_final, H_alphas_final, H_ema

# ─── Substrato e bandas ───────────────────────────────────────────────────────
sinal_base = gerar_beep(DURACAO, alpha_sinal=1.0 / 3.0)
BANDAS     = gerar_bandas_phi()
BINS       = bandas_para_bins(BANDAS, len(sinal_base))
nb         = len(BINS)

# H_alpha inicial do substrato 880Hz
F0   = np.fft.rfft(sinal_base)
mag0 = np.array([np.abs(F0[b_lo:b_hi]).mean() for b_lo, b_hi, _, _ in BINS])
H0_alphas = []
for b_lo, b_hi, _, _ in BINS:
    mag_b = np.abs(F0[b_lo:b_hi])
    if len(mag_b) > 0 and mag_b.sum() > 1e-12:
        H_a, _ = semente_alpha_banda(mag_b)
        H0_alphas.append(H_a)
H0_alpha_med = float(np.mean(H0_alphas)) if H0_alphas else 0.5

print(f"\nSubstrato 880Hz:")
print(f"  H_alpha inicial (medio) = {H0_alpha_med:.4f}")
print(f"  Bandas phi: {nb}  |  Amostras: {len(sinal_base)}")
print()

# ─── Configs de ablacao ───────────────────────────────────────────────────────
CONFIGS = {
    "AA — coh_raw,   sem eco_entropico  (baseline classico)": {
        "usar_coh_raw": True,  "usar_eco": False, "N_base_rol": 0,
        "color": "#8B949E"},
    "BB — coh_alpha, sem eco_entropico  (semente medindo)   ": {
        "usar_coh_raw": False, "usar_eco": False, "N_base_rol": 0,
        "color": "#00BFFF"},
    "CC — coh_alpha, eco_entropico      (feedback entropia) ": {
        "usar_coh_raw": False, "usar_eco": True,  "N_base_rol": 0,
        "color": "#00FF88"},
    "DD — coh_alpha, eco_entropico+rol  (N_base=137)        ": {
        "usar_coh_raw": False, "usar_eco": True,  "N_base_rol": N_BASE,
        "color": "#FF9944"},
}

# ─── Loop principal ───────────────────────────────────────────────────────────
resultados = {}
print("=" * 65)

for cfg_nome, cfg in CONFIGS.items():
    print(f"\n{cfg_nome}")

    beta     = np.ones(nb)
    bm       = beta.copy()
    coh_mem  = np.zeros(nb)
    H_ema    = np.full(nb, H0_alpha_med)

    R_ema   = 1.669
    R_prev  = R_ema

    hist_R    = []
    hist_beta = []
    hist_coh  = []
    hist_H    = []
    hist_vert = []

    ciclo_conv  = N_CICLOS
    sinal_final = sinal_base.copy()

    for ciclo in range(N_CICLOS):
        s_out, cohs, H_alphas, H_ema = cascata_semente(
            sinal_base, beta, BINS, coh_mem, H_ema,
            N_base_rol  = cfg["N_base_rol"],
            usar_eco    = cfg["usar_eco"],
            usar_coh_raw= cfg["usar_coh_raw"]
        )
        coh_mem = cohs.copy()

        # Atualiza beta
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn * ba + wm * bm
        bm   = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)

        R_ciclo = medir_R(cohs)
        R_prev  = R_ema
        R_ema   = ALPHA_FINE * R_ciclo + (1.0 - ALPHA_FINE) * R_ema

        beta_max  = float(beta.max())
        coh_med   = float(cohs.mean())
        H_alpha_med = float(H_alphas.mean())
        H_ema_med = float(H_ema.mean())

        vertex = triangulacao_vertex(H_alpha_med, beta_max)

        hist_R.append(R_ema)
        hist_beta.append(beta_max)
        hist_coh.append(coh_med)
        hist_H.append(H_alpha_med)
        hist_vert.append(vertex)

        sinal_final = s_out

        if beta_max >= LIMIAR and ciclo_conv == N_CICLOS:
            ciclo_conv = ciclo + 1
            print(f"  CAMPO HARMONICO -> ciclo {ciclo_conv:2d}  beta={beta_max:.4f}")

        if (ciclo + 1) % 5 == 0:
            print(f"  ciclo {ciclo+1:2d}"
                  f"  beta={beta_max:.4f}"
                  f"  R_ema={R_ema:.5f}({R_ema/ALPHA_FINE:.1f}a)"
                  f"  H_alpha={H_alpha_med:.4f}"
                  f"  H_ema={H_ema_med:.4f}"
                  f"  [{vertex}]")

    campo_ok    = ciclo_conv <= N_CICLOS
    R_final     = hist_R[-1]
    H_eq        = float(np.mean(hist_H[-5:]))   # H_alpha equilibrio (ultimos 5 ciclos)
    vert_final  = hist_vert[-1]

    print(f"  {'CAMPO ciclo '+str(ciclo_conv) if campo_ok else 'campo NAO formado'}")
    print(f"  R_final  = {R_final:.5f} = {R_final/ALPHA_FINE:.1f}*alpha"
          f"  (phi={PHI:.4f}  ratio={R_final/PHI:.4f})")
    print(f"  H_alpha_equilibrio = {H_eq:.4f}  (alpha* entropia do substrato)")
    print(f"  Vertice final: {vert_final}")

    resultados[cfg_nome] = {
        "hist_R": hist_R, "hist_beta": hist_beta,
        "hist_coh": hist_coh, "hist_H": hist_H,
        "ciclo_conv": ciclo_conv, "R_final": R_final,
        "H_eq": H_eq, "campo": campo_ok,
        "color": cfg["color"]
    }

# ─── Sintese ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SINTESE — Eco Semente 880Hz")
print("=" * 65)
print(f"\n{'Config':<52} {'Campo':>6} {'Ciclo':>6} {'R_final':>12} {'R/phi':>8} {'H_eq':>7}")
print("-" * 95)
for nome, res in resultados.items():
    campo_str = f"c{res['ciclo_conv']}" if res["campo"] else "NAO"
    print(f"{nome:<52} {'SIM':>6} {campo_str:>6}"
          f"  {res['R_final']:.5f}={res['R_final']/ALPHA_FINE:.1f}a"
          f"  {res['R_final']/PHI:.4f}"
          f"  {res['H_eq']:.4f}")

print(f"\nReferencia: R_natural phi = {PHI:.5f}"
      f"  (resultado historico eco-ressonante 880Hz)")
print(f"Perguntas:")
print(f"  (1) AA R_final/phi ~= 1.0?  (regressao)")
print(f"  (2) BB != AA?  (coh_alpha vs coh_raw)")
print(f"  (3) CC campo mais cedo que AA? (eco_entropico)")
print(f"  (4) H_eq por config = alpha* no eixo entropico?")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("AlphaPhi Eco Semente 880Hz\nα como semente + φ como atrator",
             fontsize=13, fontweight="bold")

ciclos = np.arange(1, N_CICLOS + 1)

ax = axes[0, 0]
for nome, res in resultados.items():
    ax.plot(ciclos, res["hist_R"], color=res["color"],
            label=nome[:30].strip(), linewidth=1.8)
ax.axhline(PHI, color="gold", linestyle="--", linewidth=1.5, label=f"φ = {PHI:.4f}")
ax.axhline(ALPHA_FINE, color="red", linestyle=":", linewidth=1.2, label=f"α = {ALPHA_FINE:.5f}")
ax.set_title("R_ema por ciclo"); ax.set_xlabel("Ciclo"); ax.legend(fontsize=7)

ax = axes[0, 1]
for nome, res in resultados.items():
    ax.plot(ciclos, res["hist_beta"], color=res["color"],
            label=nome[:30].strip(), linewidth=1.8)
ax.axhline(PHI**3, color="gold", linestyle="--", linewidth=1.5, label=f"φ³ = {PHI**3:.4f}")
ax.set_title("beta_max por ciclo"); ax.set_xlabel("Ciclo"); ax.legend(fontsize=7)

ax = axes[1, 0]
for nome, res in resultados.items():
    ax.plot(ciclos, res["hist_H"], color=res["color"],
            label=nome[:30].strip(), linewidth=1.8)
ax.axhline(H0_alpha_med, color="white", linestyle=":", linewidth=1.2,
           label=f"H0 inicial = {H0_alpha_med:.4f}")
ax.set_title("H_alpha medio por ciclo"); ax.set_xlabel("Ciclo"); ax.legend(fontsize=7)

ax = axes[1, 1]
for nome, res in resultados.items():
    ax.plot(ciclos, res["hist_coh"], color=res["color"],
            label=nome[:30].strip(), linewidth=1.8)
ax.set_title("Coerencia media por ciclo"); ax.set_xlabel("Ciclo"); ax.legend(fontsize=7)

for a in axes.flat:
    a.set_facecolor("#0d1117"); a.grid(alpha=0.2)
fig.patch.set_facecolor("#0d1117")
for a in axes.flat:
    for spine in a.spines.values(): spine.set_edgecolor("#444")
    a.tick_params(colors="#ccc"); a.xaxis.label.set_color("#ccc")
    a.yaxis.label.set_color("#ccc"); a.title.set_color("#eee")

plt.tight_layout()
plt.savefig("alphaphi_semente880.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("\nalpha-phi | eco semente 880Hz | alphaphi_semente880.png")
