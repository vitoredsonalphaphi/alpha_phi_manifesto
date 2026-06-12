# AlphaPhi_Eco_Roldana_COLAB.py
# Vitor Edson Delavi · Florianópolis · 2026
#
# SISTEMA DE ROLDANAS PARA α — ECO RESSONANTE
#
# Observacao anterior (AlphaPhi_Eco_Ressonante_Alpha_COLAB.py):
#   R_audio_natural ~= 228*alpha ~= phi   (nao ~= alpha como no SST2)
#   R desce naturalmente ~0.18*alpha/ciclo
#   Para R->alpha sem roldana: ~1256 ciclos
#   Com n=137 roldanas (= 1/alpha): ~9 ciclos dentro da janela de 20
#
#   coh_natural ~= 0.42 (moderada)
#   Campo harmonico forma no ciclo 10 (robusto a qualquer config de alpha)
#   Janela de maxima sensibilidade: ciclos 1-9 (pre-formacao do campo)
#
# PRINCIPIO DAS ROLDANAS (mecanica adaptativa):
#
#   Uma roldana: forca_efetiva = alpha * 1   = 0.0073
#   N roldanas:  forca_efetiva = alpha * N
#   N = 1/alpha = 137: forca_efetiva = 1.0  (ganho maximo natural)
#
#   Tendencia alinhada com target -> N=1 (guia, nao empurra)
#   Tendencia oposta ao target   -> N proporcional a |erro| (forca necessaria)
#
# MECANISMO — torque via ce_eff no eco_eq:
#   R = E_geom/E_coer = (1-coh)^2/coh^2
#   Para reduzir R: aumentar coh
#   coh aumenta quando entropia H diminui (espectro mais concentrado)
#   H diminui quando ce (coerencia ponderada) aumenta
#   ce_eff = ce * (1 + N * alpha * sign(target - R))
#     N*alpha = 0: sem torque (original)
#     N*alpha = 1: dobra ce (ganho maximo)
#
# FASES:
#   Fase 1 (ciclos 1-N_obs):  observacao pura
#     - Mede trajetoria natural de R, coh, beta
#     - Calcula sensibilidade = |dR/dt| / R (onde alfa tem mais alavancagem)
#     - Identifica "ponto de grip": ciclo de maxima sensibilidade
#   Fase 2 (ciclos N_obs+1-N_CICLOS): roldana ativa
#     - Calcula N_roldanas baseado em tendencia e erro
#     - Aplica torque via ce_eff
#     - Monitora se campo harmonico e preservado
#
# ABLACAO:
#   AA: sem roldana  (observacao pura, 20 ciclos)
#   BB: roldana N=1  (ganho=alpha, pressao minima)
#   CC: roldana adaptativa N in [1, 1/alpha] (baseado em tendencia)
#   DD: roldana N=1/alpha=137 (ganho=1.0, pressao maxima)
#
# DIAGNOSTICO PRINCIPAL:
#   (1) R_audio se aproxima de alpha em fase 2?
#   (2) Campo harmonico (beta_max >= phi^3) preservado com roldana?
#   (3) N_roldanas adaptativo -- quando usa mais/menos forca?
#   (4) Ciclo de maxima sensibilidade confirma janela 1-9?

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
N_OBS    = 10          # ciclos de observacao pura antes da roldana
LIMIAR   = 0.99 * PHI**3

TARGET_R  = ALPHA_FINE                  # objetivo: R -> alpha
N_MAX_ROL = int(round(1.0 / ALPHA_FINE))  # = 137 roldanas maximas

print(f"phi           = {PHI:.10f}")
print(f"alpha         = {ALPHA_FINE:.10f}  (1/{1/ALPHA_FINE:.3f})")
print(f"phi^3 (limiar)= {PHI**3:.6f}")
print(f"N_MAX_ROL     = 1/alpha = {N_MAX_ROL}")
print(f"TARGET_R      = alpha = {TARGET_R:.8f}")
print(f"N_OBS         = {N_OBS} ciclos de observacao")
print(f"\nAnalise de forca por ciclo:")
print(f"  N=1   (1 roldana):  forca = {1*ALPHA_FINE:.5f} = alpha")
print(f"  N=14  (14 roldanas): forca = {14*ALPHA_FINE:.5f} = 14*alpha = ~0.1")
print(f"  N=137 (max):        forca = {137*ALPHA_FINE:.5f} = 1/alpha^2 * alpha = 1.0")
print(f"\nTaxa natural de descida de R: ~0.18*alpha/ciclo")
print(f"  N=1:   {1*0.18:.3f}*alpha/ciclo -> {1256:.0f} ciclos para R=alpha")
print(f"  N=14:  {14*0.18:.3f}*alpha/ciclo -> {1256//14:.0f} ciclos")
print(f"  N=137: {137*0.18:.3f}*alpha/ciclo -> {1256//137:.0f} ciclos")
print("=" * 65)
print("AlphaPhi -- Eco Roldana  |  Forca Adaptativa para alfa")
print("Fase1: observacao  |  Fase2: torque via N*alpha em ce")
print("=" * 65)

# -- Funcoes auxiliares --
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

# -- Sistema de Roldanas --
def calcular_roldanas(R_atual, R_prev, target, alpha=ALPHA_FINE, n_max=None):
    """
    Calcula o numero de roldanas necessario.

    Tendencia: positiva = R sobe; negativa = R desce
    Erro:      positivo = R acima do target; negativo = abaixo

    Tendencia e erro mesmo sinal  -> R se afasta do target -> mais roldanas
    Tendencia e erro sinais opostos -> R se aproxima -> menos roldanas (guia)

    N in [1, n_max]  onde n_max = 1/alpha = 137
    """
    if n_max is None:
        n_max = int(round(1.0 / alpha))

    tendencia = R_atual - R_prev  # velocidade atual de R
    erro      = R_atual - target  # distancia ao objetivo

    if abs(erro) < alpha:
        # Ja chegou perto o suficiente: minima forca
        return 1

    # Alinhamento: tendencia * erro
    # < 0 -> indo na direcao certa (tendencia oposta ao erro)
    # > 0 -> indo na direcao errada
    if tendencia * erro <= 0:
        # Indo na direcao certa -- guia com 1 roldana
        n = 1
    else:
        # Indo na direcao errada -- proporcional ao erro em unidades de alpha
        n = int(np.clip(abs(erro) / alpha, 1, n_max))

    return n

def eco_eq_roldana(x, bins_phi, beta_bands, coh_mem=None,
                   n_roldanas=0, R_ema=None, target=TARGET_R):
    """
    Equalizador eco-phi com torque de roldana.

    n_roldanas=0: original, sem torque
    n_roldanas>0: ce_eff = ce * (1 + n*alpha * sign(target - R_ema))

    Mecanismo:
      sign(target - R) = -1 quando R > target (caso normal: R ~= 228*alpha)
      torque > 0 -> amplifica ce -> espectro mais concentrado -> coh sobe -> R desce
    """
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))

    N, F   = len(x), np.fft.rfft(x)
    F_out  = F.copy()
    cohs   = []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI

    # Torque de roldana: ganho mecanico = n * alpha
    # Sinal do torque: queremos REDUZIR R -> AUMENTAR coh -> AUMENTAR ce
    torque_gain = float(n_roldanas) * ALPHA_FINE   # in [0, 1.0]
    if R_ema is not None and target is not None:
        # Aplica torque na direcao que reduz o erro
        torque_dir = float(np.sign(target - R_ema))  # -1 quando R > target
        # Para R > target queremos aumentar coh, o que requer ce maior
        # Logo: quando R > target (torque_dir=-1), amplificamos ce mesmo assim
        # A direcao aqui e: sempre amplificar ce para puxar R para baixo
        torque = torque_gain * abs(torque_dir)       # sempre positivo
    else:
        torque = 0.0

    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb  = F[b_lo:b_hi]
        mag = np.abs(Fb)
        phase = np.angle(Fb)

        # Entropia espectral (original: clip 1e-10)
        an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        an  = an / an.sum()
        H   = -np.sum(an * np.log(an))
        coh = float(1.0 - H / np.log(max(len(an), 2)))

        ce = (wn * coh + wm * float(coh_mem[i])
              if (coh_mem is not None and i < len(coh_mem)) else coh)

        # Roldana: amplifica ce pelo torque
        ce_eff = float(np.clip(ce * (1.0 + torque), 0.0, PHI**2))

        cohs.append(coh)

        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce_eff * PHI ** bi) * np.cos(2 * np.pi * nk / PHI),
                      0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)

    r = np.fft.irfft(F_out, n=N)
    return r / (np.max(np.abs(r)) + 1e-10), np.array(cohs)

def cascata_roldana(sinal, beta_bands, bins_phi, coh_mem_init=None,
                    n_roldanas=0, R_ema=None, target=TARGET_R):
    s  = sinal.copy()
    cm = (coh_mem_init.copy() if coh_mem_init is not None
          else np.zeros(len(bins_phi)))
    passos = []
    for _ in range(N_STEPS):
        se, cohs = eco_eq_roldana(s, bins_phi, beta_bands, cm,
                                   n_roldanas=n_roldanas,
                                   R_ema=R_ema, target=target)
        cm = cohs
        se = normalizar(se)
        passos.append(se.copy())
        s = se.copy()
    return passos, cohs

# -- Configuracoes de ablacao --
CONFIGS = {
    "AA — sem roldana    (N=0)         ": {
        "modo": "fixo", "N_fixo": 0,    "color": "#8B949E", "label": "AA N=0"},
    "BB — roldana N=1    (ganho=alpha) ": {
        "modo": "fixo", "N_fixo": 1,    "color": "#00BFFF", "label": "BB N=1"},
    "CC — roldana adapt  (N=1..137)    ": {
        "modo": "adapt","N_fixo": None, "color": "#00FF88", "label": "CC adaptativo"},
    "DD — roldana N=137  (ganho=1.0)   ": {
        "modo": "fixo", "N_fixo": N_MAX_ROL, "color": "#FF9944", "label": "DD N=137"},
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
    print(f"\n{cfg_nome}")

    beta     = np.ones(nb)
    bm       = beta.copy()
    wm_b     = 1.0 / PHI
    wn_b     = 1.0 - 1.0 / PHI

    R_ema        = 1.667   # valor observado no experimento anterior
    R_prev       = R_ema

    hist_R       = []
    hist_coh     = []
    hist_beta    = []
    hist_N_rol   = []
    hist_sens    = []   # sensibilidade = |dR/dt| / R

    ciclo_conv   = N_CICLOS
    sinal_final  = sinal_base.copy()
    coh_final    = np.zeros(nb)

    for ciclo in range(N_CICLOS):
        # Decide modo: observacao ou roldana
        em_observacao = (ciclo < N_OBS)

        if em_observacao or cfg["modo"] == "fixo" and cfg["N_fixo"] == 0:
            n_rol = 0
        elif cfg["modo"] == "fixo":
            n_rol = cfg["N_fixo"]
        else:
            # Adaptativo: so na fase 2
            if em_observacao:
                n_rol = 0
            else:
                n_rol = calcular_roldanas(R_ema, R_prev, TARGET_R)

        passos, cohs = cascata_roldana(
            sinal_base, beta, BINS,
            n_roldanas=n_rol,
            R_ema=R_ema,
            target=TARGET_R
        )

        # Atualiza beta (identico ao BEEP880 original)
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn_b * ba + wm_b * bm
        bm   = beta.copy()
        beta = np.clip(beta, 0.05, PHI ** 3)

        # Mede R_audio do ciclo
        R_ciclo, E_geom, E_coer = medir_R_audio(cohs)

        # EMA de R com alpha (como Exp 6)
        R_prev = R_ema
        R_ema  = ALPHA_FINE * R_ciclo + (1.0 - ALPHA_FINE) * R_ema

        beta_max = float(beta.max())
        coh_med  = float(cohs.mean())

        # Sensibilidade: variacao relativa de R neste ciclo
        sens = abs(R_ema - R_prev) / (abs(R_prev) + 1e-10)

        hist_R.append(R_ema)
        hist_coh.append(coh_med)
        hist_beta.append(beta_max)
        hist_N_rol.append(n_rol)
        hist_sens.append(sens)

        sinal_final = passos[-1]
        coh_final   = cohs

        if beta_max >= LIMIAR and ciclo_conv == N_CICLOS:
            ciclo_conv = ciclo + 1
            print(f"  CAMPO HARMONICO -> ciclo {ciclo_conv:2d}"
                  f"  beta_max={beta_max:.4f}")

        if (ciclo + 1) % 5 == 0:
            fase = "OBS" if em_observacao else f"ROL(N={n_rol})"
            print(f"  ciclo {ciclo+1:2d} [{fase:>10}]"
                  f"  beta={beta_max:.4f}"
                  f"  coh={coh_med:.4f}"
                  f"  R={R_ema:.5f} ({R_ema/ALPHA_FINE:.1f}*alpha)"
                  f"  sens={sens:.6f}")

    # Identificar ponto de grip (ciclo de maxima sensibilidade)
    grip_ciclo = int(np.argmax(hist_sens)) + 1

    campo_ok = ciclo_conv < N_CICLOS
    R_final  = hist_R[-1]
    R_fase1  = float(np.mean(hist_R[:N_OBS]))
    R_fase2  = float(np.mean(hist_R[N_OBS:]))

    print(f"  {'CAMPO FORMADO ciclo ' + str(ciclo_conv) if campo_ok else 'campo NAO formado'}")
    print(f"  R fase1 (obs):  {R_fase1:.5f} = {R_fase1/ALPHA_FINE:.1f}*alpha")
    print(f"  R fase2 (rol):  {R_fase2:.5f} = {R_fase2/ALPHA_FINE:.1f}*alpha")
    print(f"  R final:        {R_final:.5f} = {R_final/ALPHA_FINE:.1f}*alpha")
    print(f"  Ponto de grip (max sensibilidade): ciclo {grip_ciclo}")

    resultados[cfg_nome] = {
        "hist_R"     : hist_R,
        "hist_coh"   : hist_coh,
        "hist_beta"  : hist_beta,
        "hist_N_rol" : hist_N_rol,
        "hist_sens"  : hist_sens,
        "ciclo_conv" : ciclo_conv,
        "campo_ok"   : campo_ok,
        "R_fase1"    : R_fase1,
        "R_fase2"    : R_fase2,
        "R_final"    : R_final,
        "grip_ciclo" : grip_ciclo,
        "sinal_final": sinal_final,
        "label"      : cfg["label"],
        "color"      : cfg["color"],
    }

    try:
        from scipy.io import wavfile as wf
        fname = f"alphaphi_roldana_{cfg['label'].split()[0].lower()}.wav"
        wf.write(fname, FS, (sinal_final * 32767).astype(np.int16))
        print(f"  -> {fname}")
    except Exception:
        pass

# -- Sintese --
print("\n" + "=" * 65)
print("SINTESE — Sistema de Roldanas para alpha")
print("=" * 65)
print(f"\n{'Config':<44} {'Campo':>6} {'Ciclo':>6} {'R_f1':>8} {'R_f2':>8}"
      f" {'R_f1/a':>7} {'R_f2/a':>7} {'Grip':>5}")
print("-" * 100)
for cfg_nome, res in resultados.items():
    c = "ok" if res["campo_ok"] else "--"
    ci = str(res["ciclo_conv"]) if res["campo_ok"] else "--"
    print(f"{cfg_nome:<44} {c:>6} {ci:>6}"
          f" {res['R_fase1']:>8.4f} {res['R_fase2']:>8.4f}"
          f" {res['R_fase1']/ALPHA_FINE:>7.1f} {res['R_fase2']/ALPHA_FINE:>7.1f}"
          f" {res['grip_ciclo']:>5}")

# -- Analise do ponto de grip --
print("\nPONTO DE GRIP — ambiente favoravel para alpha:")
aa_sens = resultados["AA — sem roldana    (N=0)         "]["hist_sens"]
grip    = resultados["AA — sem roldana    (N=0)         "]["grip_ciclo"]
print(f"  Ciclo de maxima sensibilidade (AA): {grip}")
print(f"  Sensibilidade maxima: {max(aa_sens):.6f}")
print(f"  Sensibilidade media ciclos 1-{N_OBS}: {np.mean(aa_sens[:N_OBS]):.6f}")
print(f"  Sensibilidade media ciclos {N_OBS+1}-{N_CICLOS}: {np.mean(aa_sens[N_OBS:]):.6f}")
if grip <= N_OBS:
    print(f"  -> CONFIRMADO: maxima sensibilidade na fase pre-formacao (ciclos 1-{N_OBS})")
    print(f"     Roldana mais eficaz ANTES do campo se formar")
else:
    print(f"  -> Sensibilidade maxima pos-formacao (ciclo {grip})")
    print(f"     Roldana mais eficaz apos campo estabilizar")

# -- Eficacia das roldanas --
print("\nEFICACA DAS ROLDANAS — reducao de R (aproximacao a alpha):")
aa_R2 = resultados["AA — sem roldana    (N=0)         "]["R_fase2"]
for cfg_nome, res in resultados.items():
    if "AA" in cfg_nome:
        continue
    delta = res["R_fase2"] - aa_R2
    pct   = delta / aa_R2 * 100
    print(f"  {res['label']}: R_fase2={res['R_fase2']:.4f}  "
          f"delta vs AA = {delta:+.4f} ({pct:+.1f}%)  "
          f"{'reduz R' if delta < 0 else 'aumenta R'}")

# -- Visualizacao --
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor("#0d1117")

ciclos_eixo = list(range(1, N_CICLOS + 1))

# Graf 1 — R_audio por ciclo (questao central)
ax = axes[0, 0]
ax.set_facecolor("#161b22")
ax.set_title("R_audio por ciclo\n(roldana ativa a partir do ciclo N_OBS+1)",
             color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "CC" in cfg_nome else 1.5 if "AA" in cfg_nome else 1.2
    ax.plot(ciclos_eixo, res["hist_R"],
            color=res["color"], linewidth=lw, label=res["label"],
            marker="o", markersize=3)
ax.axhline(ALPHA_FINE, color="#FF4466", linestyle="--", linewidth=2,
           label=f"target = alpha = {ALPHA_FINE:.5f}", alpha=0.9)
ax.axhline(PHI, color="#DAA520", linestyle=":", linewidth=1,
           label=f"phi = {PHI:.4f}", alpha=0.6)
ax.axvline(N_OBS + 0.5, color="#888888", linestyle="--", alpha=0.5,
           label=f"Fase 2 inicia (ciclo {N_OBS+1})")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("R_audio (EMA)", color="#8B949E")
ax.grid(True, alpha=0.15)

# Graf 2 — N_roldanas por ciclo (forca aplicada)
ax = axes[0, 1]
ax.set_facecolor("#161b22")
ax.set_title("N_roldanas por ciclo\n(forca mecanica de alpha = N * alpha)",
             color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "CC" in cfg_nome else 1.2
    ax.plot(ciclos_eixo, res["hist_N_rol"],
            color=res["color"], linewidth=lw, label=res["label"],
            marker="o", markersize=3)
ax.axhline(N_MAX_ROL, color="#DAA520", linestyle=":", alpha=0.5,
           label=f"N_max = 1/alpha = {N_MAX_ROL}")
ax.axvline(N_OBS + 0.5, color="#888888", linestyle="--", alpha=0.5)
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("N roldanas", color="#8B949E")
ax.grid(True, alpha=0.15)

# Graf 3 — beta_max por ciclo
ax = axes[1, 0]
ax.set_facecolor("#161b22")
ax.set_title("beta_max por ciclo\n(campo harmonico preservado?)",
             color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "CC" in cfg_nome else 1.5 if "AA" in cfg_nome else 1.2
    ax.plot(ciclos_eixo, res["hist_beta"],
            color=res["color"], linewidth=lw, label=res["label"],
            marker="o", markersize=3)
ax.axhline(PHI**3, color="#DAA520", linestyle="--", linewidth=2,
           label=f"phi^3 = {PHI**3:.3f}")
ax.axvline(N_OBS + 0.5, color="#888888", linestyle="--", alpha=0.5)
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("beta_max", color="#8B949E")
ax.grid(True, alpha=0.15)

# Graf 4 — sensibilidade (ponto de grip)
ax = axes[1, 1]
ax.set_facecolor("#161b22")
ax.set_title("Sensibilidade |dR/dt|/R por ciclo (AA)\n(ponto de grip = max alavancagem para alpha)",
             color="#DAA520", fontweight="bold", fontsize=10)
aa_s = resultados["AA — sem roldana    (N=0)         "]["hist_sens"]
aa_g = resultados["AA — sem roldana    (N=0)         "]["grip_ciclo"]
ax.plot(ciclos_eixo, aa_s, color="#8B949E", linewidth=2,
        marker="o", markersize=4, label="sensibilidade AA")
ax.axvline(aa_g, color="#FF4466", linestyle="--", linewidth=2,
           label=f"ponto de grip: ciclo {aa_g}")
ax.axvline(N_OBS + 0.5, color="#888888", linestyle="--", alpha=0.5,
           label=f"inicio fase 2 (ciclo {N_OBS+1})")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("sensibilidade", color="#8B949E")
ax.grid(True, alpha=0.15)

fig.suptitle(
    f"AlphaPhi -- Eco Roldana  |  Forca Adaptativa N * alpha para eco ressonante\n"
    f"AA: sem roldana  |  BB: N=1  |  CC: adaptativo  |  DD: N=137"
    f"  |  phi={PHI:.4f}  alpha=1/{N_MAX_ROL}  |  Florianopolis 2026",
    color="#DAA520", fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig("alphaphi_eco_roldana.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Grafico: alphaphi_eco_roldana.png")

# -- Sintese final --
print("\n" + "=" * 65)
print("RESPOSTA -- ambiente favoravel para surgimento de alpha")
print("=" * 65)
aa_R   = resultados["AA — sem roldana    (N=0)         "]
cc_R   = resultados["CC — roldana adapt  (N=1..137)    "]
dd_R   = resultados["DD — roldana N=137  (ganho=1.0)   "]

print(f"""
Observacao (Fase 1, ciclos 1-{N_OBS}):
  R natural = {aa_R['R_fase1']:.4f} = {aa_R['R_fase1']/ALPHA_FINE:.1f}*alpha ~= phi
  Ponto de grip: ciclo {aa_R['grip_ciclo']} (maxima sensibilidade a perturbacao)
  Taxa descida natural: ~0.18*alpha/ciclo (muito lenta para alpha sem roldana)

Intervencao (Fase 2, ciclos {N_OBS+1}-{N_CICLOS}):
  CC adaptativo R_fase2 = {cc_R['R_fase2']:.4f} = {cc_R['R_fase2']/ALPHA_FINE:.1f}*alpha
  DD N=137       R_fase2 = {dd_R['R_fase2']:.4f} = {dd_R['R_fase2']/ALPHA_FINE:.1f}*alpha
  AA sem roldana R_fase2 = {aa_R['R_fase2']:.4f} = {aa_R['R_fase2']/ALPHA_FINE:.1f}*alpha

Campo harmonico:
  {'TODOS preservaram beta_max >= phi^3' if all(r['campo_ok'] for r in resultados.values()) else 'Algum nao preservou -- ver tabela'}

Conclusao:
  O ambiente favoravel para alpha e o periodo pre-formacao do campo (ciclos 1-{aa_R['grip_ciclo']}).
  Nesse janela o sistema esta em transicao -- maxima sensibilidade a perturbacao.
  Roldanas permitem que alpha exerca forca sem destruir o campo?
  Resposta nos dados acima -- verificar R_fase2 vs R_fase1.
""")
print("alpha-phi")
