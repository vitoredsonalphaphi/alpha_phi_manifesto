# AlphaPhi_Eco_Roldana_Phi_COLAB.py
# Vitor Edson Delavi · Florianopolis · 2026
#
# SISTEMA DE ROLDANAS COM ALTERNANCIA PHI PARA ALPHA
#
# Diagnostico anterior:
#   R_natural = 228*alpha ~= phi (atrator forte)
#   Roldana direta N=137: apenas -0.5% em 10 ciclos (mecanismo fraco)
#   Ponto de grip: ciclo 1 (antes da formacao do campo)
#
# NOVO PRINCIPIO — ALTERNANCIA PHI:
#   N_STEPS=5 passos por ciclo, forcas alternadas E/X com amortecimento phi:
#
#   Passo 0 (Entropia E): ce_scale = 1 + N0*alpha   N0 = N_base
#   Passo 1 (Expansao X): ce_scale = 1 - N1*alpha   N1 = N0/phi
#   Passo 2 (Entropia E): ce_scale = 1 + N2*alpha   N2 = N1/phi
#   Passo 3 (Expansao X): ce_scale = 1 - N3*alpha   N3 = N2/phi
#   Passo 4 (Entropia E): ce_scale = 1 + N4*alpha   N4 = N3/phi
#
#   Ultimo passo sempre Entropia → bias liquido para coerencia
#   Amortecimento phi: N_k = N_base / phi^k → serie de Fibonacci decrescente
#
# POR QUE NAO TODOS NO MESMO SENTIDO:
#   Sistema governa por phi (atrator forte) → empurra diretamente = resistencia maxima
#   Alternancia = "linguagem do proprio sistema" → phi fala, phi responde
#   Expansao permite que o sistema "respire" → cria janela para proxima compressao
#   Ratchet: compressao mais forte que expansao → progresso liquido
#
# MECANISMO — SHARPENING ESPECTRAL (mais direto que ce_eff):
#   Entropia (E): mag_eff = mag * (1 + delta) onde delta = N_k * alpha
#     Amplifica bins dominantes em relacao aos fracos → concentra distribuicao
#     Entropia H diminui → coh aumenta → R desce
#   Expansao (X): mag_eff = mag / (1 + delta/phi)
#     Atenua bins dominantes levemente → distribui energia → H sobe ligeiramente
#     Forca de expansao = delta/phi (mais fraca que compressao)
#
# OBSERVACAO DENTRO DA ALTERNANCIA (eco ressonante como observer):
#   Mede R_step apos cada um dos 5 passos
#   Identifica: qual passo tem R mais proximo de alpha?
#   Esse e o AMBIENTE FAVORAVEL para alpha (fase de acoplamento)
#
# ABLACAO:
#   AA: sem roldana (baseline, N_base=0)
#   BB: phi-alternado N_base=14   (moderado, 14 roldanas)
#   CC: phi-alternado N_base=137  (maximo, 137 roldanas)
#   DD: phi-alternado N_base=137, roldana ativa desde ciclo 1 (sem fase de obs)
#
# DIAGNOSTICO PRINCIPAL:
#   (1) R_step por passo: qual fase (E/X) tem R mais baixo? (ambiente de alpha)
#   (2) R_ciclo por ciclo: mais rapido que direto N=137 (-0.5% em 10 ciclos)?
#   (3) Campo harmonico preservado (beta_max >= phi^3)?
#   (4) Net sharpening > 1.0? (acumulado dos 5 passos)

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# -- Constantes --
PHI        = (1 + np.sqrt(5)) / 2
ALPHA_FINE = 1 / 137.035999084
C_PHI      = 1.0 / PHI**2

FS       = 44100
F_BEEP   = 880.0
F_ORG    = 220.0
F_M      = F_ORG / PHI
BETA_FM  = PHI
DURACAO  = 1.5
N_STEPS  = 5
N_CICLOS = 20
N_OBS    = 10       # ciclos antes de ligar a roldana (exceto DD)
LIMIAR   = 0.99 * PHI**3
TARGET_R = ALPHA_FINE

N_BASE_MOD = 14
N_BASE_MAX = int(round(1.0 / ALPHA_FINE))  # 137

print(f"phi           = {PHI:.10f}")
print(f"alpha         = {ALPHA_FINE:.10f}  (1/{1/ALPHA_FINE:.3f})")
print(f"phi^3 (limiar)= {PHI**3:.6f}")
print(f"\nAlternancia phi para N_base=137:")
for k in range(N_STEPS):
    N_k    = N_BASE_MAX / (PHI ** k)
    sinal  = "E (entropia)" if k % 2 == 0 else "X (expansao)"
    delta  = N_k * ALPHA_FINE
    escala = 1.0 + delta if k % 2 == 0 else 1.0 - delta / PHI
    print(f"  Passo {k} [{sinal}]: N={N_k:.1f}  delta={delta:.4f}  ce_scale={escala:.4f}")

# Net liquido: produto das escalas
escala_net = 1.0
for k in range(N_STEPS):
    N_k   = N_BASE_MAX / (PHI ** k)
    delta = N_k * ALPHA_FINE
    if k % 2 == 0:
        escala_net *= (1.0 + delta)
    else:
        escala_net *= max(1.0 - delta / PHI, 0.1)
print(f"\n  Net ce_scale (produto dos 5 passos): {escala_net:.4f}")
print(f"  (>1 = bias para coerencia; <1 = bias para expansao)")
print("=" * 65)
print("AlphaPhi -- Roldanas phi-alternadas")
print("Obs: mede R por passo para identificar ambiente de alpha")
print("=" * 65)

# -- Funcoes base --
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
    N = int(FS * duracao)
    t = np.linspace(0, duracao, N, endpoint=False)
    beep = np.sign(np.sin(2 * np.pi * F_BEEP * t))
    fm   = np.sin(2 * np.pi * F_M * t + BETA_FM * np.sin(2 * np.pi * F_M * t / PHI))
    return normalizar(alpha_sinal * beep + (1 - alpha_sinal) * fm)

def medir_R_audio(cohs):
    cohs   = np.asarray(cohs, dtype=float)
    E_geom = float(np.mean((1.0 - cohs) ** 2))
    E_coer = float(np.mean(cohs ** 2))
    return E_geom / (E_coer + 1e-10), E_geom, E_coer

# -- Um passo do equalizador com sharpening espectral --
def eco_eq_passo(x, bins_phi, beta_bands, coh_mem, ce_scale=1.0):
    """
    Um passo do equalizador com ce escalado.

    ce_scale > 1.0 (Entropia E): amplifica ce → mais coerencia → R desce
    ce_scale < 1.0 (Expansao X): atenua ce  → menos coerencia → R sobe
    ce_scale = 1.0: passo original sem roldana

    Diferente de eco_eq anterior: ce_scale multiplica a COERENCIA PONDERADA
    antes do calculo do envelope, nao apenas o fator de amplificacao.
    """
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))

    N, F   = len(x), np.fft.rfft(x)
    F_out  = F.copy()
    cohs   = []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI

    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb  = F[b_lo:b_hi]
        mag = np.abs(Fb)
        phase = np.angle(Fb)

        # Entropia da banda
        an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        an  = an / an.sum()
        H   = -np.sum(an * np.log(an))
        coh = float(1.0 - H / np.log(max(len(an), 2)))

        # Memoria de coerencia
        ce = (wn * coh + wm * float(coh_mem[i])
              if (coh_mem is not None and i < len(coh_mem)) else coh)

        # Roldana: escala ce
        ce_eff = float(np.clip(ce * ce_scale, 0.0, PHI ** 2))

        cohs.append(coh)

        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce_eff * PHI ** bi) * np.cos(2 * np.pi * nk / PHI),
                      0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)

    r = np.fft.irfft(F_out, n=N)
    return r / (np.max(np.abs(r)) + 1e-10), np.array(cohs)

# -- Cascata com alternancia phi --
def cascata_phi_alternada(sinal, beta_bands, bins_phi,
                          coh_mem_init=None, N_base=0):
    """
    N_STEPS passos com alternancia phi.

    Passo k:
      N_k = N_base / phi^k
      delta_k = N_k * alpha
      E (k par):  ce_scale = 1 + delta_k   (concentra)
      X (k impar): ce_scale = max(0.1, 1 - delta_k/phi)  (expande, mais fraco)

    Retorna:
      passos: lista de sinais por passo
      cohs_por_passo: coherencias por passo
      R_por_passo: R_audio medido apos cada passo
      ce_scales: escalas aplicadas em cada passo
    """
    s  = sinal.copy()
    cm = (coh_mem_init.copy() if coh_mem_init is not None
          else np.zeros(len(bins_phi)))

    passos        = []
    cohs_por_passo = []
    R_por_passo   = []
    ce_scales     = []

    for k in range(N_STEPS):
        if N_base == 0:
            ce_scale = 1.0
        else:
            N_k    = float(N_base) / (PHI ** k)
            delta_k = N_k * ALPHA_FINE
            if k % 2 == 0:
                # Entropia: concentra
                ce_scale = float(np.clip(1.0 + delta_k, 1.0, PHI ** 2))
            else:
                # Expansao: expande (forcas menores por phi)
                ce_scale = float(np.clip(1.0 - delta_k / PHI, 0.1, 1.0))

        se, cohs = eco_eq_passo(s, bins_phi, beta_bands, cm, ce_scale)
        R_k, _, _ = medir_R_audio(cohs)

        cm = cohs
        se = normalizar(se)
        passos.append(se.copy())
        cohs_por_passo.append(cohs.copy())
        R_por_passo.append(R_k)
        ce_scales.append(ce_scale)
        s = se.copy()

    return passos, cohs_por_passo[-1], R_por_passo, ce_scales

# -- Configuracoes de ablacao --
CONFIGS = {
    "AA — sem roldana    (N_base=0)         ": {
        "N_base": 0, "N_obs": N_OBS,
        "color": "#8B949E", "label": "AA N=0"},
    "BB — phi-alt N=14   (moderado)         ": {
        "N_base": N_BASE_MOD, "N_obs": N_OBS,
        "color": "#00BFFF", "label": "BB N=14"},
    "CC — phi-alt N=137  (maximo, obs+rol)  ": {
        "N_base": N_BASE_MAX, "N_obs": N_OBS,
        "color": "#00FF88", "label": "CC N=137"},
    "DD — phi-alt N=137  (desde ciclo 1)    ": {
        "N_base": N_BASE_MAX, "N_obs": 0,
        "color": "#FF9944", "label": "DD N=137 desde c1"},
}

# -- Sinal base --
sinal_base = gerar_beep(DURACAO, alpha_sinal=1.0 / 3.0)
BANDAS     = gerar_bandas_phi()
BINS       = bandas_para_bins(BANDAS, len(sinal_base))
nb         = len(BINS)
print(f"\nBandas phi: {len(BANDAS)}  |  Sinal: {len(sinal_base)} amostras")

# -- Loop principal --
resultados = {}

print("\n" + "=" * 65)
for cfg_nome, cfg in CONFIGS.items():
    N_base_cfg = cfg["N_base"]
    N_obs_cfg  = cfg["N_obs"]

    print(f"\n{cfg_nome}")
    print(f"  N_base={N_base_cfg}  N_obs={N_obs_cfg}")

    # Mostra escala de cada passo para este N_base
    if N_base_cfg > 0:
        scales_info = []
        for k in range(N_STEPS):
            N_k    = float(N_base_cfg) / (PHI ** k)
            delta_k = N_k * ALPHA_FINE
            if k % 2 == 0:
                sc = np.clip(1.0 + delta_k, 1.0, PHI**2)
                tag = "E"
            else:
                sc = np.clip(1.0 - delta_k/PHI, 0.1, 1.0)
                tag = "X"
            scales_info.append(f"p{k}[{tag}]:{sc:.3f}")
        print(f"  ce_scales: {' | '.join(scales_info)}")

    beta  = np.ones(nb)
    bm    = beta.copy()
    wm_b  = 1.0 / PHI
    wn_b  = 1.0 - 1.0 / PHI

    R_ema    = 1.669   # valor observado
    R_prev   = R_ema

    hist_R        = []
    hist_coh      = []
    hist_beta     = []
    hist_R_steps  = []   # R_audio por passo, por ciclo
    hist_ce_scales = []
    hist_sens     = []

    ciclo_conv = N_CICLOS
    sinal_final = sinal_base.copy()

    for ciclo in range(N_CICLOS):
        em_obs = (ciclo < N_obs_cfg)
        N_ativo = 0 if em_obs else N_base_cfg

        passos, cohs, R_steps, ce_scales = cascata_phi_alternada(
            sinal_base, beta, BINS,
            N_base=N_ativo
        )

        # Atualiza beta
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn_b * ba + wm_b * bm
        bm   = beta.copy()
        beta = np.clip(beta, 0.05, PHI ** 3)

        R_ciclo, _, _ = medir_R_audio(cohs)
        R_prev = R_ema
        R_ema  = ALPHA_FINE * R_ciclo + (1.0 - ALPHA_FINE) * R_ema

        beta_max = float(beta.max())
        coh_med  = float(cohs.mean())
        sens     = abs(R_ema - R_prev) / (abs(R_prev) + 1e-10)

        hist_R.append(R_ema)
        hist_coh.append(coh_med)
        hist_beta.append(beta_max)
        hist_R_steps.append(R_steps[:])
        hist_ce_scales.append(ce_scales[:])
        hist_sens.append(sens)

        sinal_final = passos[-1]

        if beta_max >= LIMIAR and ciclo_conv == N_CICLOS:
            ciclo_conv = ciclo + 1
            print(f"  CAMPO HARMONICO -> ciclo {ciclo_conv:2d}"
                  f"  beta_max={beta_max:.4f}")

        if (ciclo + 1) % 5 == 0:
            fase = "OBS" if em_obs else f"ROL(N={N_ativo})"
            # R_steps: min e max no ciclo atual
            r_min = min(R_steps); r_max = max(R_steps)
            r_min_step = R_steps.index(r_min)
            print(f"  ciclo {ciclo+1:2d} [{fase:>12}]"
                  f"  beta={beta_max:.4f}"
                  f"  R_ema={R_ema:.5f}({R_ema/ALPHA_FINE:.1f}a)"
                  f"  R_min_passo={r_min:.5f}(p{r_min_step})"
                  f"  R_max={r_max:.5f}")

    campo_ok = ciclo_conv < N_CICLOS
    R_final  = hist_R[-1]
    R_fase1  = float(np.mean(hist_R[:N_obs_cfg])) if N_obs_cfg > 0 else float(hist_R[0])
    R_fase2  = float(np.mean(hist_R[N_obs_cfg:]))

    # Analise por passo: media de R_step k ao longo de todos os ciclos
    R_por_passo_medio = np.zeros(N_STEPS)
    for ciclo_idx in range(N_CICLOS):
        for k in range(N_STEPS):
            R_por_passo_medio[k] += hist_R_steps[ciclo_idx][k]
    R_por_passo_medio /= N_CICLOS

    passo_min_R = int(np.argmin(R_por_passo_medio))
    fase_min    = "Entropia(E)" if passo_min_R % 2 == 0 else "Expansao(X)"

    print(f"  {'CAMPO FORMADO ciclo '+str(ciclo_conv) if campo_ok else 'campo NAO formado'}")
    print(f"  R_fase1={R_fase1:.5f}={R_fase1/ALPHA_FINE:.1f}a  "
          f"R_fase2={R_fase2:.5f}={R_fase2/ALPHA_FINE:.1f}a  "
          f"R_final={R_final:.5f}={R_final/ALPHA_FINE:.1f}a")
    print(f"  R medio por passo: "
          + "  ".join([f"p{k}({'E' if k%2==0 else 'X'})={R_por_passo_medio[k]:.4f}"
                       for k in range(N_STEPS)]))
    print(f"  -> AMBIENTE FAVORAVEL alpha: passo {passo_min_R} ({fase_min})"
          f"  R_min={R_por_passo_medio[passo_min_R]:.5f}"
          f"={R_por_passo_medio[passo_min_R]/ALPHA_FINE:.1f}a")

    resultados[cfg_nome] = {
        "hist_R"            : hist_R,
        "hist_coh"          : hist_coh,
        "hist_beta"         : hist_beta,
        "hist_R_steps"      : hist_R_steps,
        "hist_sens"         : hist_sens,
        "R_por_passo_medio" : R_por_passo_medio.tolist(),
        "passo_min_R"       : passo_min_R,
        "fase_min"          : fase_min,
        "ciclo_conv"        : ciclo_conv,
        "campo_ok"          : campo_ok,
        "R_fase1"           : R_fase1,
        "R_fase2"           : R_fase2,
        "R_final"           : R_final,
        "sinal_final"       : sinal_final,
        "label"             : cfg["label"],
        "color"             : cfg["color"],
        "N_base"            : N_base_cfg,
    }

    try:
        from scipy.io import wavfile as wf
        fname = f"alphaphi_roldana_phi_{cfg['label'].split()[0].lower()}.wav"
        wf.write(fname, FS, (sinal_final * 32767).astype(np.int16))
        print(f"  -> {fname}")
    except Exception:
        pass

# -- Sintese --
print("\n" + "=" * 65)
print("SINTESE — Roldanas phi-alternadas")
print("=" * 65)
print(f"\n{'Config':<47} {'Campo':>6} {'Cic':>4} {'R_f1/a':>7} {'R_f2/a':>7}"
      f" {'R_fin/a':>8} {'p_min_R':>8} {'fase_min':>12}")
print("-" * 110)
for cfg_nome, res in resultados.items():
    c  = "ok" if res["campo_ok"] else "--"
    ci = str(res["ciclo_conv"]) if res["campo_ok"] else "--"
    print(f"{cfg_nome:<47} {c:>6} {ci:>4}"
          f" {res['R_fase1']/ALPHA_FINE:>7.1f} {res['R_fase2']/ALPHA_FINE:>7.1f}"
          f" {res['R_final']/ALPHA_FINE:>8.1f}"
          f" p{res['passo_min_R']:>5}    {res['fase_min']:>12}")

# -- Ambiente favoravel para alpha --
print("\n" + "-" * 65)
print("AMBIENTE FAVORAVEL PARA ALPHA (R mais proximo de alpha):")
for cfg_nome, res in resultados.items():
    pm = res["passo_min_R"]
    print(f"  {res['label']:20s}: passo {pm} ({res['fase_min'][:1]})"
          f"  R_med={res['R_por_passo_medio'][pm]:.5f}"
          f"={res['R_por_passo_medio'][pm]/ALPHA_FINE:.1f}a")

# -- Eficacia vs referencia direta --
print("\nEFICACA vs roldana direta anterior (DD N=137: R_fase2=227.4*alpha):")
ref_R_fase2 = 227.4 * ALPHA_FINE
for cfg_nome, res in resultados.items():
    if "AA" in cfg_nome:
        continue
    delta = res["R_fase2"] - ref_R_fase2
    print(f"  {res['label']:20s}: R_fase2={res['R_fase2']/ALPHA_FINE:.1f}*alpha"
          f"  delta_vs_ref={delta/ALPHA_FINE:+.1f}*alpha")

# -- Visualizacao --
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.patch.set_facecolor("#0d1117")
ciclos_eixo = list(range(1, N_CICLOS + 1))

# Graf 1 — R_ema por ciclo
ax = axes[0, 0]
ax.set_facecolor("#161b22")
ax.set_title("R_ema por ciclo\n(phi-alternado vs direto)",
             color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "CC" in cfg_nome or "DD" in cfg_nome else 1.5 if "AA" in cfg_nome else 1.2
    ax.plot(ciclos_eixo, res["hist_R"],
            color=res["color"], linewidth=lw, label=res["label"],
            marker="o", markersize=3)
ax.axhline(ALPHA_FINE, color="#FF4466", linestyle="--", linewidth=2,
           label=f"alpha={ALPHA_FINE:.5f}", alpha=0.9)
ax.axhline(PHI, color="#DAA520", linestyle=":", linewidth=1,
           label=f"phi={PHI:.3f}", alpha=0.5)
ax.axvline(N_OBS + 0.5, color="#888888", linestyle="--", alpha=0.4)
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("R_ema", color="#8B949E")
ax.grid(True, alpha=0.15)

# Graf 2 — R por passo (media sobre todos ciclos) — CC
ax = axes[0, 1]
ax.set_facecolor("#161b22")
ax.set_title("R medio por passo (CC phi-alternado)\nAmbiente favoravel de alpha",
             color="#DAA520", fontweight="bold", fontsize=10)
passos_eixo = list(range(N_STEPS))
for cfg_nome, res in resultados.items():
    if "AA" in cfg_nome:
        continue
    ax.plot(passos_eixo, res["R_por_passo_medio"],
            color=res["color"], linewidth=2, label=res["label"],
            marker="o", markersize=6)
ax.axhline(ALPHA_FINE, color="#FF4466", linestyle="--", linewidth=2,
           label=f"alpha", alpha=0.9)
ax.axhline(PHI, color="#DAA520", linestyle=":", linewidth=1,
           label=f"phi", alpha=0.5)
cc_pmin = resultados["CC — phi-alt N=137  (maximo, obs+rol)  "]["passo_min_R"]
ax.axvline(cc_pmin, color="#00FF88", linestyle="--", alpha=0.7,
           label=f"min R: passo {cc_pmin}")
for k in range(N_STEPS):
    tag = "E" if k % 2 == 0 else "X"
    cor = "#00FF88" if k % 2 == 0 else "#FF9944"
    ax.text(k, ax.get_ylim()[0] if len(ax.lines) > 0 else 0,
            tag, ha="center", color=cor, fontsize=10, fontweight="bold",
            transform=ax.get_xaxis_transform())
ax.set_xticks(passos_eixo)
ax.set_xticklabels([f"p{k}\n{'E' if k%2==0 else 'X'}" for k in passos_eixo],
                   color="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Passo (E=entropia, X=expansao)", color="#8B949E")
ax.set_ylabel("R medio", color="#8B949E")
ax.grid(True, alpha=0.15)

# Graf 3 — beta_max por ciclo
ax = axes[0, 2]
ax.set_facecolor("#161b22")
ax.set_title("beta_max por ciclo\n(campo harmonico preservado?)",
             color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "CC" in cfg_nome else 1.5 if "AA" in cfg_nome else 1.2
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

# Graf 4 — R por passo por ciclo (heatmap) CC
ax = axes[1, 0]
ax.set_facecolor("#161b22")
ax.set_title("R por passo por ciclo — CC (heatmap)\nBranco=R baixo (favoravel alpha)",
             color="#DAA520", fontweight="bold", fontsize=10)
cc_steps = resultados["CC — phi-alt N=137  (maximo, obs+rol)  "]["hist_R_steps"]
mat_cc   = np.array(cc_steps).T  # shape (N_STEPS, N_CICLOS)
im = ax.imshow(mat_cc, aspect="auto", cmap="hot_r",
               extent=[0.5, N_CICLOS + 0.5, -0.5, N_STEPS - 0.5],
               origin="lower")
plt.colorbar(im, ax=ax, label="R_audio")
ax.axvline(N_OBS + 0.5, color="white", linestyle="--", alpha=0.7,
           label="Fase 2")
ax.set_yticks(range(N_STEPS))
ax.set_yticklabels([f"p{k} {'E' if k%2==0 else 'X'}" for k in range(N_STEPS)],
                   color="#8B949E")
ax.tick_params(colors="#8B949E")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("Passo", color="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")

# Graf 5 — R por passo por ciclo (heatmap) DD
ax = axes[1, 1]
ax.set_facecolor("#161b22")
ax.set_title("R por passo por ciclo — DD (desde ciclo 1)\nAlternancia desde o inicio",
             color="#DAA520", fontweight="bold", fontsize=10)
dd_steps = resultados["DD — phi-alt N=137  (desde ciclo 1)    "]["hist_R_steps"]
mat_dd   = np.array(dd_steps).T
im2 = ax.imshow(mat_dd, aspect="auto", cmap="hot_r",
                extent=[0.5, N_CICLOS + 0.5, -0.5, N_STEPS - 0.5],
                origin="lower")
plt.colorbar(im2, ax=ax, label="R_audio")
ax.set_yticks(range(N_STEPS))
ax.set_yticklabels([f"p{k} {'E' if k%2==0 else 'X'}" for k in range(N_STEPS)],
                   color="#8B949E")
ax.tick_params(colors="#8B949E")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("Passo", color="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")

# Graf 6 — comparacao R_fase2
ax = axes[1, 2]
ax.set_facecolor("#161b22")
ax.set_title("R_fase2 / alpha por config\n(menor = mais proximo de alpha)",
             color="#DAA520", fontweight="bold", fontsize=10)
nomes_plot = [res["label"] for res in resultados.values()]
vals_plot  = [res["R_fase2"] / ALPHA_FINE for res in resultados.values()]
cores_plot = [res["color"] for res in resultados.values()]
bars = ax.bar(nomes_plot, vals_plot, color=cores_plot, alpha=0.8)
ax.axhline(1.0, color="#FF4466", linestyle="--", linewidth=2, label="target=alpha")
ax.axhline(PHI / ALPHA_FINE, color="#DAA520", linestyle=":", alpha=0.5,
           label=f"phi/alpha={PHI/ALPHA_FINE:.0f}")
for bar, val in zip(bars, vals_plot):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f"{val:.1f}a", ha="center", va="bottom", color="#E6EDF3", fontsize=8)
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E", axis='x', rotation=15)
ax.tick_params(colors="#8B949E", axis='y')
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_ylabel("R_fase2 / alpha", color="#8B949E")
ax.grid(True, alpha=0.15, axis='y')

fig.suptitle(
    f"AlphaPhi -- Roldanas phi-alternadas  |  E/X amortecido por phi\n"
    f"Passos E(0,2,4) comprimem, X(1,3) expandem  |  N_k = N_base/phi^k"
    f"  |  phi={PHI:.4f}  alpha=1/{N_BASE_MAX}  |  Florianopolis 2026",
    color="#DAA520", fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig("alphaphi_eco_roldana_phi.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Grafico: alphaphi_eco_roldana_phi.png")

# -- Sintese final --
cc = resultados["CC — phi-alt N=137  (maximo, obs+rol)  "]
dd = resultados["DD — phi-alt N=137  (desde ciclo 1)    "]
aa = resultados["AA — sem roldana    (N_base=0)         "]

print("\n" + "=" * 65)
print("AMBIENTE FAVORAVEL PARA ALPHA — conclusao")
print("=" * 65)
print(f"""
R_natural = {aa['R_fase2']:.4f} = {aa['R_fase2']/ALPHA_FINE:.1f}*alpha ~= phi

Alternancia phi:
  CC (N=137, fase 2): R_fase2 = {cc['R_fase2']:.4f} = {cc['R_fase2']/ALPHA_FINE:.1f}*alpha
  DD (N=137, c1):     R_fase2 = {dd['R_fase2']:.4f} = {dd['R_fase2']/ALPHA_FINE:.1f}*alpha

Passo com menor R (mais favoravel para alpha):
  CC: passo {cc['passo_min_R']} ({cc['fase_min']})  R={cc['R_por_passo_medio'][cc['passo_min_R']]:.4f} = {cc['R_por_passo_medio'][cc['passo_min_R']]/ALPHA_FINE:.1f}*alpha
  DD: passo {dd['passo_min_R']} ({dd['fase_min']})  R={dd['R_por_passo_medio'][dd['passo_min_R']]:.4f} = {dd['R_por_passo_medio'][dd['passo_min_R']]/ALPHA_FINE:.1f}*alpha

Interpretacao:
  O passo de menor R e a FASE NATURAL DE ACOPLAMENTO de alpha.
  Nessa fase, o sistema e mais receptivo a perturbacoes na escala de alpha.
  O campo harmonico foi preservado? {'SIM' if cc['campo_ok'] and dd['campo_ok'] else 'VERIFICAR'}.
""")
print("alpha-phi")
