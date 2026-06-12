# AlphaPhi_Eco_Alpha_Entropia_COLAB.py
# Vitor Edson Delavi · Florianopolis · 2026
#
# INTEGRACAO TRIPLA DE ALPHA NA ENTROPIA — ECO RESSONANTE 880Hz
#
# Tres camadas:
#
#   Camada 1 — GRAO (pre-entropia):
#     original: an = clip(mag/sum, 1e-10, 1.0)        piso arbitrario
#     alpha:    an = clip(mag/sum, ALPHA_FINE, 1.0)   alpha como quantum de desordem
#
#   Camada 2 — ESCALA (a propria entropia):
#     original: H_raw   = H / log(N_bins)   fracao do maximo por banda (varia)
#               coh_raw = 1 - H_raw
#     alpha:    H_alpha = H / log(1/alpha)   em alpha-nats  [log(137) ~= 4.919]
#               coh_alpha = 1 - H_alpha
#               MAX H_alpha = 1.0 quando H = log(137): 137 estados equiprovaveis
#               alpha e literalmente a regua do espaco de entropia
#
#   Camada 3 — FORCA (pos-entropia):
#     roldanas phi-alternadas agindo sobre coh_alpha (espaco alpha-nativo)
#     N_k = N_base / phi^k
#     E (k par):   ce_scale = 1 + N_k * alpha   (concentra distribuicao)
#     X (k impar): ce_scale = 1 - N_k * alpha / phi  (expande, mais fraco)
#
# ABLACAO:
#   AA: grao=1e-10  escala=nats    roldana=nao  (baseline original)
#   BB: grao=alpha  escala=nats    roldana=nao  (so grao alpha)
#   CC: grao=alpha  escala=alpha   roldana=nao  (grao + escala alpha)
#   DD: grao=alpha  escala=alpha   roldana=sim  (integracao tripla)
#
# DIAGNOSTICO PRINCIPAL:
#   R_raw   = E_geom / E_coer em coh_raw   (espaco original)
#   R_alpha = E_geom / E_coer em coh_alpha  (espaco alpha-nativo)
#   Pergunta: R_alpha converge para algum valor especial (1.0, phi, alpha)?
#   Beta sempre usa coh_raw — dinamica de campo harmonico preservada e comparavel
#
# Contexto (camara/marchas — ALPHA_vs_ALPHA_ESTRELA.md):
#   alpha = 1/137 e o cambio. Nao e a marcha.
#   H_alpha = H / log(137): o cambio como regua da propria entropia.
#   R_alpha = 1.0: equilibrio exato no espaco alpha-nativo.

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
ALPHA_FINE = 1 / 137.035999084
LOG_ALPHA  = np.log(1.0 / ALPHA_FINE)    # log(137) ~= 4.919  (max H em alpha-nats)
C_PHI      = 1.0 / PHI**2

FS        = 44100
F_BEEP    = 880.0
F_ORG     = 220.0
F_M       = F_ORG / PHI
BETA_FM   = PHI
DURACAO   = 1.5
N_STEPS   = 5
N_CICLOS  = 20
N_OBS     = 10
LIMIAR    = 0.99 * PHI**3
N_BASE    = int(round(1.0 / ALPHA_FINE))   # 137

print(f"phi              = {PHI:.10f}")
print(f"alpha            = {ALPHA_FINE:.10f}  (1/{N_BASE})")
print(f"log(1/alpha)     = {LOG_ALPHA:.6f}  (regua de entropia em alpha-nats)")
print(f"phi^3 (limiar)   = {PHI**3:.6f}")
print(f"\nPhi-roldanas DD (N_base={N_BASE}):")
for k in range(N_STEPS):
    Nk  = N_BASE / PHI**k
    d   = Nk * ALPHA_FINE
    tag = "E"; sc = min(1.0 + d, PHI**2)
    if k % 2 != 0:
        tag = "X"; sc = max(1.0 - d / PHI, 0.1)
    print(f"  passo {k} [{tag}]: N={Nk:.1f}  delta={d:.4f}  ce_scale={sc:.4f}")
print("=" * 65)

# ─── Funcoes auxiliares ───────────────────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max:
            break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    return [(max(0, int(f_lo / (FS / n))),
             min(int(f_hi / (FS / n)) + 1, n // 2 + 1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

def gerar_beep(duracao, alpha_sinal=1.0 / 3.0):
    N    = int(FS * duracao)
    t    = np.linspace(0, duracao, N, endpoint=False)
    beep = np.sign(np.sin(2 * np.pi * F_BEEP * t))
    fm   = np.sin(2 * np.pi * F_M * t + BETA_FM * np.sin(2 * np.pi * F_M * t / PHI))
    return normalizar(alpha_sinal * beep + (1 - alpha_sinal) * fm)

def calcular_coerencias(mag, usar_alpha_grain, n_bins_band):
    """
    Retorna (coh_raw, coh_alpha) para um vetor de magnitudes.
    coh_raw:   normalizado por log(N_bins)   — fracao do maximo da banda
    coh_alpha: normalizado por log(1/alpha)  — fracao do maximo em alpha-nats
    """
    floor = ALPHA_FINE if usar_alpha_grain else 1e-10
    an    = np.clip(mag / (mag.sum() + 1e-8), floor, 1.0)
    an    = an / an.sum()
    H     = -np.sum(an * np.log(an + 1e-15))

    coh_raw   = float(np.clip(1.0 - H / np.log(max(n_bins_band, 2)), 0.0, 1.0))
    coh_alpha = float(np.clip(1.0 - H / LOG_ALPHA, 0.0, 1.0))
    return coh_raw, coh_alpha

def medir_R(cohs):
    c      = np.asarray(cohs, dtype=float)
    E_geom = float(np.mean((1.0 - c)**2))
    E_coer = float(np.mean(c**2))
    return E_geom / (E_coer + 1e-10)

# ─── Passo do equalizador — arquitetura tripla alpha ─────────────────────────
def eco_eq_passo_v2(x, bins_phi, beta_bands,
                    coh_mem_raw, coh_mem_alpha,
                    usar_alpha_grain=False,
                    usar_alpha_scale=False,
                    ce_scale=1.0):
    """
    usar_alpha_grain: True -> piso da distribuicao = ALPHA_FINE  (Camada 1)
    usar_alpha_scale: True -> envelope conduzido por coh_alpha   (Camada 2)
                     False -> envelope conduzido por coh_raw
    ce_scale: multiplicador phi-alternado da roldana             (Camada 3)

    Beta usa coh_raw em todos os casos — campo harmonico comparavel.
    """
    beta_bands    = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    coh_mem_raw   = np.atleast_1d(np.asarray(coh_mem_raw, dtype=float))
    coh_mem_alpha = np.atleast_1d(np.asarray(coh_mem_alpha, dtype=float))

    N, F   = len(x), np.fft.rfft(x)
    F_out  = F.copy()
    cohs_raw_out   = []
    cohs_alpha_out = []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI

    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi    = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb    = F[b_lo:b_hi]
        mag   = np.abs(Fb)
        phase = np.angle(Fb)
        nb    = len(Fb)

        coh_raw, coh_alpha = calcular_coerencias(mag, usar_alpha_grain, nb)

        # Memoria raw (base de comparacao identica para todos)
        cm_r  = float(coh_mem_raw[i]) if i < len(coh_mem_raw) else coh_raw
        ce_raw = wn * coh_raw + wm * cm_r

        # Driver do envelope
        if usar_alpha_scale:
            cm_a     = float(coh_mem_alpha[i]) if i < len(coh_mem_alpha) else coh_alpha
            ce_driver = wn * coh_alpha + wm * cm_a
        else:
            ce_driver = ce_raw

        ce_eff = float(np.clip(ce_driver * ce_scale, 0.0, PHI**2))

        cohs_raw_out.append(coh_raw)
        cohs_alpha_out.append(coh_alpha)

        nk  = np.arange(nb)
        env = np.clip(1.0 + (ce_eff * PHI**bi) * np.cos(2 * np.pi * nk / PHI),
                      0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)

    r = np.fft.irfft(F_out, n=N)
    return r / (np.max(np.abs(r)) + 1e-10), np.array(cohs_raw_out), np.array(cohs_alpha_out)

# ─── Cascata phi-alternada v2 ─────────────────────────────────────────────────
def cascata_v2(sinal, beta_bands, bins_phi,
               cm_raw_init, cm_alpha_init,
               usar_alpha_grain, usar_alpha_scale,
               N_base_rol):
    """
    N_STEPS passos phi-alternados.
    N_base_rol=0: sem roldana (ce_scale=1.0 em todos os passos).
    N_base_rol>0: alternancia E/X com N_k = N_base/phi^k.
    Retorna coerencias do ultimo passo (raw e alpha).
    """
    s      = sinal.copy()
    cm_raw = cm_raw_init.copy()
    cm_alp = cm_alpha_init.copy()
    passos = []

    for k in range(N_STEPS):
        if N_base_rol == 0:
            ce_scale = 1.0
        else:
            Nk = float(N_base_rol) / (PHI**k)
            d  = Nk * ALPHA_FINE
            if k % 2 == 0:
                ce_scale = float(np.clip(1.0 + d, 1.0, PHI**2))
            else:
                ce_scale = float(np.clip(1.0 - d / PHI, 0.1, 1.0))

        se, cr, ca = eco_eq_passo_v2(
            s, bins_phi, beta_bands, cm_raw, cm_alp,
            usar_alpha_grain=usar_alpha_grain,
            usar_alpha_scale=usar_alpha_scale,
            ce_scale=ce_scale
        )
        cm_raw = cr
        cm_alp = ca
        se     = normalizar(se)
        passos.append(se.copy())
        s = se.copy()

    return passos, cm_raw, cm_alp   # ultimo passo: cohs_raw, cohs_alpha

# ─── Configuracoes de ablacao ─────────────────────────────────────────────────
CONFIGS = {
    "AA — grao=1e-10  escala=nats    roldana=nao ": {
        "use_grain": False, "use_scale": False, "N_rol": 0,
        "color": "#8B949E", "label": "AA baseline"},
    "BB — grao=alpha  escala=nats    roldana=nao ": {
        "use_grain": True,  "use_scale": False, "N_rol": 0,
        "color": "#00BFFF", "label": "BB grao"},
    "CC — grao=alpha  escala=alpha   roldana=nao ": {
        "use_grain": True,  "use_scale": True,  "N_rol": 0,
        "color": "#00FF88", "label": "CC grao+escala"},
    "DD — grao=alpha  escala=alpha   roldana=sim ": {
        "use_grain": True,  "use_scale": True,  "N_rol": N_BASE,
        "color": "#FF9944", "label": "DD tripla"},
}

# ─── Sinal base ───────────────────────────────────────────────────────────────
sinal_base = gerar_beep(DURACAO, alpha_sinal=1.0 / 3.0)
BANDAS     = gerar_bandas_phi()
BINS       = bandas_para_bins(BANDAS, len(sinal_base))
nb         = len(BINS)
print(f"\nBandas phi: {len(BANDAS)}  |  Sinal: {len(sinal_base)} amostras")

# ─── Loop principal ───────────────────────────────────────────────────────────
resultados = {}

print("\n" + "=" * 65)
for cfg_nome, cfg in CONFIGS.items():
    print(f"\n{cfg_nome}")

    beta   = np.ones(nb)
    bm     = beta.copy()
    wm_b   = 1.0 / PHI
    wn_b   = 1.0 - 1.0 / PHI

    cm_raw = np.zeros(nb)
    cm_alp = np.zeros(nb)

    R_ema_raw = 1.669   # valor observado nos experimentos anteriores
    R_ema_alp = 1.669

    hist_R_raw   = []
    hist_R_alpha = []
    hist_coh_raw = []
    hist_coh_alp = []
    hist_beta    = []

    ciclo_conv  = N_CICLOS
    sinal_final = sinal_base.copy()

    for ciclo in range(N_CICLOS):
        N_rol_ativo = cfg["N_rol"] if ciclo >= N_OBS else 0

        passos, cohs_raw, cohs_alp = cascata_v2(
            sinal_base, beta, BINS, cm_raw, cm_alp,
            usar_alpha_grain=cfg["use_grain"],
            usar_alpha_scale=cfg["use_scale"],
            N_base_rol=N_rol_ativo
        )

        cm_raw = cohs_raw.copy()
        cm_alp = cohs_alp.copy()

        # Beta usa coh_raw — dinamica de campo preservada
        cr    = ((cohs_raw - cohs_raw.min()) /
                 (cohs_raw.max() - cohs_raw.min() + 1e-10))
        ba    = PHI**(3 * cr)
        beta  = wn_b * ba + wm_b * bm
        bm    = beta.copy()
        beta  = np.clip(beta, 0.05, PHI**3)

        R_raw_c  = medir_R(cohs_raw)
        R_alp_c  = medir_R(cohs_alp)

        # EMA com alpha como taxa de suavizacao (meta: alpha suaviza R_alpha)
        R_ema_raw = ALPHA_FINE * R_raw_c  + (1.0 - ALPHA_FINE) * R_ema_raw
        R_ema_alp = ALPHA_FINE * R_alp_c  + (1.0 - ALPHA_FINE) * R_ema_alp

        beta_max = float(beta.max())

        hist_R_raw.append(R_ema_raw)
        hist_R_alpha.append(R_ema_alp)
        hist_coh_raw.append(float(cohs_raw.mean()))
        hist_coh_alp.append(float(cohs_alp.mean()))
        hist_beta.append(beta_max)

        sinal_final = passos[-1]

        if beta_max >= LIMIAR and ciclo_conv == N_CICLOS:
            ciclo_conv = ciclo + 1
            print(f"  CAMPO HARMONICO -> ciclo {ciclo_conv:2d}"
                  f"  beta_max={beta_max:.4f}")

        if (ciclo + 1) % 5 == 0:
            fase = "OBS" if ciclo < N_OBS else f"ROL(N={N_rol_ativo})"
            print(f"  ciclo {ciclo+1:2d} [{fase:>12}]"
                  f"  beta={beta_max:.4f}"
                  f"  R_raw={R_ema_raw/ALPHA_FINE:6.1f}a"
                  f"  R_alpha={R_ema_alp:.5f}"
                  f"  coh_raw={hist_coh_raw[-1]:.3f}"
                  f"  coh_alp={hist_coh_alp[-1]:.3f}")

    campo_ok     = ciclo_conv < N_CICLOS
    R_raw_final  = hist_R_raw[-1]
    R_alp_final  = hist_R_alpha[-1]

    print(f"  {'CAMPO: ciclo '+str(ciclo_conv) if campo_ok else 'campo NAO formado'}")
    print(f"  R_raw_final   = {R_raw_final:.5f} = {R_raw_final/ALPHA_FINE:.1f}*alpha")
    print(f"  R_alpha_final = {R_alp_final:.5f}"
          f"  (dist para 1.0: {abs(R_alp_final-1.0):.5f})")

    resultados[cfg_nome] = {
        "hist_R_raw"   : hist_R_raw,
        "hist_R_alpha" : hist_R_alpha,
        "hist_coh_raw" : hist_coh_raw,
        "hist_coh_alp" : hist_coh_alp,
        "hist_beta"    : hist_beta,
        "ciclo_conv"   : ciclo_conv,
        "campo_ok"     : campo_ok,
        "R_raw_final"  : R_raw_final,
        "R_alp_final"  : R_alp_final,
        "sinal_final"  : sinal_final,
        "label"        : cfg["label"],
        "color"        : cfg["color"],
    }

    try:
        from scipy.io import wavfile as wf
        tag   = cfg["label"].split()[0].lower()
        fname = f"alphaphi_alpha_entropia_{tag}.wav"
        wf.write(fname, FS, (sinal_final * 32767).astype(np.int16))
        print(f"  -> {fname}")
    except Exception:
        pass

# ─── Sintese ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SINTESE — Integracao tripla de alpha na entropia")
print("=" * 65)
print(f"\n{'Config':<48} {'Campo':>6} {'Cic':>4}"
      f" {'R_raw/a':>8} {'R_alpha':>8} {'coh_raw':>8} {'coh_alp':>8}")
print("-" * 100)
for cfg_nome, res in resultados.items():
    c  = "ok" if res["campo_ok"] else "--"
    ci = str(res["ciclo_conv"]) if res["campo_ok"] else "--"
    print(f"{cfg_nome:<48} {c:>6} {ci:>4}"
          f" {res['R_raw_final']/ALPHA_FINE:>8.1f}"
          f" {res['R_alp_final']:>8.5f}"
          f" {float(np.mean(res['hist_coh_raw'])):>8.3f}"
          f" {float(np.mean(res['hist_coh_alp'])):>8.3f}")

aa_alp = resultados["AA — grao=1e-10  escala=nats    roldana=nao "]["R_alp_final"]
print(f"\nR_alpha natural (AA) = {aa_alp:.5f}")
print(f"R_alpha = 1.0 seria: equilibrio exato no espaco de alpha-nats")
print(f"R_alpha = {PHI:.4f} (phi) seria: atrator phi no espaco alpha-nativo")

# ─── Visualizacao ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.patch.set_facecolor("#0d1117")
ciclos_eixo = list(range(1, N_CICLOS + 1))

# Graf 1 — R_raw por ciclo (espaco original)
ax = axes[0, 0]
ax.set_facecolor("#161b22")
ax.set_title("R_raw por ciclo\n(espaco original — nats)",
             color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "DD" in cfg_nome or "CC" in cfg_nome else 1.5
    ax.plot(ciclos_eixo, [r / ALPHA_FINE for r in res["hist_R_raw"]],
            color=res["color"], linewidth=lw, label=res["label"],
            marker="o", markersize=3)
ax.axhline(1.0, color="#FF4466", linestyle="--", linewidth=2,
           label="alpha (1.0)", alpha=0.9)
ax.axhline(PHI / ALPHA_FINE, color="#DAA520", linestyle=":", linewidth=1,
           label=f"phi/alpha={PHI/ALPHA_FINE:.0f}", alpha=0.5)
ax.axvline(N_OBS + 0.5, color="#888888", linestyle="--", alpha=0.4, label="fase 2")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("R_raw / alpha", color="#8B949E")
ax.grid(True, alpha=0.15)

# Graf 2 — R_alpha por ciclo (CHAVE — espaco alpha-nativo)
ax = axes[0, 1]
ax.set_facecolor("#161b22")
ax.set_title("R_ALPHA por ciclo\n(espaco alpha-nativo — CHAVE)",
             color="#FF9944", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "DD" in cfg_nome or "CC" in cfg_nome else 1.5
    ax.plot(ciclos_eixo, res["hist_R_alpha"],
            color=res["color"], linewidth=lw, label=res["label"],
            marker="o", markersize=3)
ax.axhline(1.0, color="#FF4466", linestyle="--", linewidth=2.5,
           label="R_alpha=1.0 (equilibrio alpha)", alpha=0.9)
ax.axhline(PHI, color="#DAA520", linestyle=":", linewidth=1,
           label=f"phi={PHI:.3f}", alpha=0.5)
ax.axhline(aa_alp, color="#8B949E", linestyle=":", linewidth=1,
           label=f"AA natural={aa_alp:.3f}", alpha=0.7)
ax.axvline(N_OBS + 0.5, color="#888888", linestyle="--", alpha=0.4, label="fase 2")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("R_alpha", color="#8B949E")
ax.grid(True, alpha=0.15)

# Graf 3 — beta_max por ciclo
ax = axes[0, 2]
ax.set_facecolor("#161b22")
ax.set_title("beta_max por ciclo\n(campo harmonico preservado?)",
             color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "DD" in cfg_nome or "CC" in cfg_nome else 1.5
    ax.plot(ciclos_eixo, res["hist_beta"],
            color=res["color"], linewidth=lw, label=res["label"],
            marker="o", markersize=3)
ax.axhline(PHI**3, color="#DAA520", linestyle="--", linewidth=2,
           label=f"phi^3={PHI**3:.3f}")
ax.axvline(N_OBS + 0.5, color="#888888", linestyle="--", alpha=0.4)
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("beta_max", color="#8B949E")
ax.grid(True, alpha=0.15)

# Graf 4 — coh_raw vs coh_alpha por ciclo
ax = axes[1, 0]
ax.set_facecolor("#161b22")
ax.set_title("coh_raw (solido) vs coh_alpha (tracejado)\nimpacto da escala alpha",
             color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.0 if "CC" in cfg_nome or "DD" in cfg_nome else 1.0
    ax.plot(ciclos_eixo, res["hist_coh_raw"],
            color=res["color"], linewidth=lw, linestyle="-",
            label=f"{res['label']} raw")
    ax.plot(ciclos_eixo, res["hist_coh_alp"],
            color=res["color"], linewidth=lw, linestyle="--",
            label=f"{res['label']} alp", alpha=0.7)
ax.axhline(0.5, color="#FF4466", linestyle=":", linewidth=1,
           label="coh=0.5 (R_alpha=1.0)", alpha=0.7)
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("Coerencia media", color="#8B949E")
ax.grid(True, alpha=0.15)

# Graf 5 — R_alpha final por config (barras)
ax = axes[1, 1]
ax.set_facecolor("#161b22")
ax.set_title("R_alpha final por config\n(quanto mais proximo de 1.0, melhor)",
             color="#DAA520", fontweight="bold", fontsize=10)
nomes_p = [res["label"] for res in resultados.values()]
vals_p  = [res["R_alp_final"] for res in resultados.values()]
cores_p = [res["color"] for res in resultados.values()]
bars = ax.bar(nomes_p, vals_p, color=cores_p, alpha=0.8)
ax.axhline(1.0, color="#FF4466", linestyle="--", linewidth=2.5,
           label="R_alpha=1.0 (equilibrio alpha)")
ax.axhline(aa_alp, color="#8B949E", linestyle=":", linewidth=1.5,
           label=f"AA natural={aa_alp:.3f}", alpha=0.8)
for bar, val in zip(bars, vals_p):
    ax.text(bar.get_x() + bar.get_width() / 2.,
            bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom",
            color="#E6EDF3", fontsize=9)
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=9)
ax.tick_params(colors="#8B949E", axis='x', rotation=15)
ax.tick_params(colors="#8B949E", axis='y')
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_ylabel("R_alpha", color="#8B949E")
ax.grid(True, alpha=0.15, axis='y')

# Graf 6 — R_raw final por config (em multiplos de alpha)
ax = axes[1, 2]
ax.set_facecolor("#161b22")
ax.set_title("R_raw final por config\n(em multiplos de alpha)",
             color="#DAA520", fontweight="bold", fontsize=10)
vals_raw = [res["R_raw_final"] / ALPHA_FINE for res in resultados.values()]
bars2 = ax.bar(nomes_p, vals_raw, color=cores_p, alpha=0.8)
ax.axhline(1.0, color="#FF4466", linestyle="--", linewidth=2, label="alpha")
ax.axhline(PHI / ALPHA_FINE, color="#DAA520", linestyle=":",
           alpha=0.5, label=f"phi/alpha={PHI/ALPHA_FINE:.0f}")
for bar, val in zip(bars2, vals_raw):
    ax.text(bar.get_x() + bar.get_width() / 2.,
            bar.get_height() + 0.5,
            f"{val:.1f}a", ha="center", va="bottom",
            color="#E6EDF3", fontsize=9)
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=9)
ax.tick_params(colors="#8B949E", axis='x', rotation=15)
ax.tick_params(colors="#8B949E", axis='y')
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_ylabel("R_raw / alpha", color="#8B949E")
ax.grid(True, alpha=0.15, axis='y')

fig.suptitle(
    f"AlphaPhi — Integracao tripla de alpha na entropia  |  Eco Ressonante 880Hz\n"
    f"Camada 1: grao=alpha  |  Camada 2: H em alpha-nats [log(137)={LOG_ALPHA:.3f}]"
    f"  |  Camada 3: roldanas phi sobre coh_alpha\n"
    f"phi={PHI:.4f}  alpha=1/{N_BASE}  log(1/alpha)={LOG_ALPHA:.3f}  |  Florianopolis 2026",
    color="#DAA520", fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig("alphaphi_eco_alpha_entropia.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Grafico: alphaphi_eco_alpha_entropia.png")

# ─── Conclusao ────────────────────────────────────────────────────────────────
aa = resultados["AA — grao=1e-10  escala=nats    roldana=nao "]
bb = resultados["BB — grao=alpha  escala=nats    roldana=nao "]
cc = resultados["CC — grao=alpha  escala=alpha   roldana=nao "]
dd = resultados["DD — grao=alpha  escala=alpha   roldana=sim "]

print("\n" + "=" * 65)
print("CONCLUSAO — Integracao tripla alpha na entropia")
print("=" * 65)
print(f"""
Camada 0 — original sem alpha (AA):
  R_raw_final   = {aa['R_raw_final']/ALPHA_FINE:.1f}*alpha  (atrator phi)
  R_alpha_final = {aa['R_alp_final']:.5f}  (natural no espaco alpha)
  Campo: {'sim c'+str(aa['ciclo_conv']) if aa['campo_ok'] else 'nao'}

Camada 1 — grao alpha (BB):
  R_raw_final   = {bb['R_raw_final']/ALPHA_FINE:.1f}*alpha
  R_alpha_final = {bb['R_alp_final']:.5f}  (delta vs AA: {bb['R_alp_final']-aa['R_alp_final']:+.5f})
  Campo: {'sim c'+str(bb['ciclo_conv']) if bb['campo_ok'] else 'nao'}

Camada 1+2 — grao + escala alpha (CC):
  R_raw_final   = {cc['R_raw_final']/ALPHA_FINE:.1f}*alpha
  R_alpha_final = {cc['R_alp_final']:.5f}  (delta vs AA: {cc['R_alp_final']-aa['R_alp_final']:+.5f})
  Campo: {'sim c'+str(cc['ciclo_conv']) if cc['campo_ok'] else 'nao'}

Camada 1+2+3 — integracao tripla (DD):
  R_raw_final   = {dd['R_raw_final']/ALPHA_FINE:.1f}*alpha
  R_alpha_final = {dd['R_alp_final']:.5f}  (delta vs AA: {dd['R_alp_final']-aa['R_alp_final']:+.5f})
  Campo: {'sim c'+str(dd['ciclo_conv']) if dd['campo_ok'] else 'nao'}

Referencias:
  R_alpha=1.0  -> equilibrio exato no espaco alpha-nats (coh_alpha=0.5)
  R_alpha=phi  -> atrator phi no espaco alpha-nativo
  R_alpha natural (AA) = {aa['R_alp_final']:.5f}

Interpretacao:
  H_alpha = H / log(137) e a entropia medida na regua de alpha.
  log(137) e a entropia maxima do sistema nativo de alpha:
    exatamente 1/alpha = 137 estados equiprovaveis.
  O campo phi (beta >= phi^3) preservado indica que phi ainda e o atrator.
  A questao e: alpha, como regua da entropia, revela alguma nova geometria?
  R_alpha > R_alpha_natural: as camadas alpha abriram o espaco de desordem.
  R_alpha < R_alpha_natural: as camadas alpha comprimiram o espaco de desordem.
  R_alpha = 1.0: ponto de equiibrio exato no espaco alpha-nativo.
""")
print("alpha-phi  |  cambio + marchas  |  tres camadas")
