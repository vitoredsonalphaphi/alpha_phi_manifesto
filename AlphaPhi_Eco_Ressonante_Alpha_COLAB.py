# AlphaPhi_Eco_Ressonante_Alpha_COLAB.py
# © Vitor Edson Delavi · Florianópolis · 2026
#
# TRANSFERÊNCIA DOS SEIS EXPERIMENTOS SST2 → ECO RESSONANTE / BEEP 880
#
# Diagnóstico acumulado (Exps 1–6, SST2):
#   Todos os null results partilham o mesmo padrão:
#     α injetado como multiplicador externo → efeito = α × grande ≈ ínfimo
#   O que funcionou:
#     norm_h ≈ φ − 7α   (Exp 3): α como UNIDADE de distância ao atrator φ
#     H_campo baixo      (Exp 4): grão mínimo = α — menor perturbação real
#     R = E_φ/E_CE ≈ α  (Exp 6): α emerge como equilíbrio natural, não injetado
#
# O QUE NUNCA FOI TESTADO no eco ressonante:
#   eco_ressonante() — só φ, α ausente (confirmado em utils_phi.py)
#   phi_spectral_modulator_v2 — tentou α, mas phi*alpha*(1/alpha) = phi (α some)
#   BEEP880 — usa ALPHA=1/3 (ponto de emergência acústico), não α=1/137
#
# PRINCÍPIO DESTA EXPERIÊNCIA (lição dos 6 experimentos):
#   NÃO injetar α como multiplicador de fase — produziria o mesmo null result
#   MEDIR R_audio = E_geom/E_coer — se ≈ α: α já é equilíbrio (como no SST2)
#   TESTAR α como grão mínimo de entropia em eco_eq() — menor mudança possível
#
# R_audio = E_geom / E_coer
#   E_geom = mean((1 − coh)²)   — desvio de coerência máxima (análogo E_φ SST2)
#   E_coer = mean(coh²)          — energia coerente    (análogo E_CE SST2)
#   Se coh ≈ 0.93 → R_audio ≈ α — ponto de equilíbrio α é atingível
#
# ABLAÇÃO:
#   AA: original           α_grain=1e-10  α_sinal=1/3    (histórico)
#   BB: α como grão        α_grain=α_FINE α_sinal=1/3    (α como entropia)
#   CC: α no sinal         α_grain=1e-10  α_sinal=α_FINE (controle: beep fraco)
#   DD: α em ambos         α_grain=α_FINE α_sinal=α_FINE (controle combinado)
#
# DIAGNÓSTICO PRINCIPAL:
#   (1) Campo harmônico se forma em AA e BB? β_max ≥ φ³?
#   (2) R_audio ≈ α naturalmente em AA? (transferência Exp 6)
#   (3) BB altera velocidade ou qualidade da convergência?
#
# Célula única para Google Colab.
# Arquivos gerados: alphaphi_eco_alpha_AA.wav, BB.wav, CC.wav, DD.wav

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# ── Constantes ────────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
ALPHA_FINE = 1 / 137.035999084        # constante de estrutura fina
C_PHI      = 1.0 / PHI**2

FS       = 44100
F_BEEP   = 880.0
F_ORG    = 220.0
F_M      = F_ORG / PHI
BETA_FM  = PHI
DURACAO  = 1.5
N_STEPS  = 5
N_CICLOS = 20
LIMIAR   = 0.99 * PHI**3

print(f"φ          = {PHI:.10f}")
print(f"α          = {ALPHA_FINE:.10f}  (1/{1/ALPHA_FINE:.3f})")
print(f"φ³ (limiar)= {PHI**3:.6f}")
print(f"grão α     = {ALPHA_FINE:.8f}  vs  1e-10 = {1e-10}")
print(f"\nSe coh ≈ 0.930: R_audio = (0.070)²/(0.930)² = {(0.070)**2/(0.930)**2:.6f}"
      f"  = {(0.070)**2/(0.930)**2 / ALPHA_FINE:.2f}α")
print("=" * 65)
print("AlphaPhi — Eco Ressonante + α  |  Transfer Exps 1–6")
print("R_audio = E_geom/E_coer — α emerge como equilíbrio?")
print("=" * 65)

# ── Funções auxiliares ────────────────────────────────────────────────────────
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

# ── Medição R_audio ───────────────────────────────────────────────────────────
def medir_R_audio(cohs):
    """
    Transferência de Exp 6 (R = E_φ/E_CE) para o domínio acústico:

    R_audio = E_geom / E_coer
      E_geom = mean((1 − coh)²)  — desvio de coerência máxima  (≡ E_φ)
      E_coer = mean(coh²)         — energia de coerência         (≡ E_CE)

    Se coh_med ≈ 0.93 → R_audio ≈ α (equilíbrio de estrutura fina).
    Se R_audio ≈ α sem nenhuma injeção → α é equilíbrio natural do campo.
    """
    cohs   = np.asarray(cohs, dtype=float)
    E_geom = float(np.mean((1.0 - cohs) ** 2))
    E_coer = float(np.mean(cohs ** 2))
    R      = E_geom / (E_coer + 1e-10)
    return R, E_geom, E_coer

# ── Eco equalizador φ com α como grão mínimo de entropia ─────────────────────
def eco_eq(x, bins_phi, beta_bands, coh_mem=None, alpha_grain=1e-10):
    """
    Equalizador eco-φ.

    alpha_grain — probabilidade mínima no cálculo de entropia espectral:
      1e-10      : original (grão arbitrário — sem significado físico)
      ALPHA_FINE : α como grão mínimo (menor perturbação real — Exp 4 transfer)
                   abaixo de α = ruído, não contribui para a entropia do campo

    A única mudança em BB vs AA está em dois caracteres: 1e-10 → ALPHA_FINE.
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

        # α como grão mínimo: abaixo de alpha_grain → invisível à entropia
        an  = np.clip(mag / (mag.sum() + 1e-8), alpha_grain, 1.0)
        an  = an / an.sum()   # renormalizar após clip
        H   = -np.sum(an * np.log(an))
        coh = float(1.0 - H / np.log(max(len(an), 2)))

        ce  = (wn * coh + wm * float(coh_mem[i])
               if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)

        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce * PHI ** bi) * np.cos(2 * np.pi * nk / PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)

    r = np.fft.irfft(F_out, n=N)
    return r / (np.max(np.abs(r)) + 1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi, coh_mem_init=None, alpha_grain=1e-10):
    s  = sinal.copy()
    cm = (coh_mem_init.copy() if coh_mem_init is not None
          else np.zeros(len(bins_phi)))
    passos = []
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm, alpha_grain=alpha_grain)
        cm = cohs
        se = normalizar(se)
        passos.append(se.copy())
        s = se.copy()
    return passos, cohs

# ── Configurações de ablação ──────────────────────────────────────────────────
CONFIGS = {
    "AA — original    (grain=1e-10, sinal=1/3)": {
        "alpha_grain": 1e-10,
        "alpha_sinal": 1.0 / 3.0,
        "label": "AA original",
        "color": "#8B949E",
    },
    "BB — α-grão      (grain=α_FINE, sinal=1/3)": {
        "alpha_grain": ALPHA_FINE,
        "alpha_sinal": 1.0 / 3.0,
        "label": "BB α-grão",
        "color": "#00FF88",
    },
    "CC — α-sinal     (grain=1e-10, sinal=α_FINE)": {
        "alpha_grain": 1e-10,
        "alpha_sinal": ALPHA_FINE,
        "label": "CC α-sinal",
        "color": "#00BFFF",
    },
    "DD — α-grão+sinal(grain=α_FINE, sinal=α_FINE)": {
        "alpha_grain": ALPHA_FINE,
        "alpha_sinal": ALPHA_FINE,
        "label": "DD grão+sinal",
        "color": "#FF9944",
    },
}

# ── Banda de referência φ ─────────────────────────────────────────────────────
BANDAS = gerar_bandas_phi()
print(f"\nBandas φ: {len(BANDAS)}  |  N_CICLOS={N_CICLOS}  |  N_STEPS={N_STEPS}")
print(f"Sinal: F_BEEP={F_BEEP}Hz  F_M={F_M:.2f}Hz  DURACAO={DURACAO}s")

# ── Loop principal ────────────────────────────────────────────────────────────
resultados = {}

print("\n" + "=" * 65)
for cfg_nome, cfg in CONFIGS.items():
    ag  = cfg["alpha_grain"]
    as_ = cfg["alpha_sinal"]

    print(f"\n{cfg_nome}")
    print(f"  grain={ag:.2e}  sinal={as_:.6f}"
          f"  ({'α_FINE' if ag == ALPHA_FINE else '1e-10'},"
          f" {'α_FINE' if as_ == ALPHA_FINE else '1/3'})")

    sinal_base = gerar_beep(DURACAO, alpha_sinal=as_)
    BINS       = bandas_para_bins(BANDAS, len(sinal_base))
    nb         = len(BINS)

    beta  = np.ones(nb)
    bm    = beta.copy()
    wm_b  = 1.0 / PHI
    wn_b  = 1.0 - 1.0 / PHI

    hist_beta_max  = []
    hist_coh_med   = []
    hist_R_audio   = []
    ciclo_conv     = N_CICLOS   # sentinela: campo não formado
    sinal_final    = sinal_base.copy()
    coh_final      = np.zeros(nb)

    for ciclo in range(N_CICLOS):
        passos, cohs = cascata_eq(sinal_base, beta, BINS, alpha_grain=ag)

        # Atualização β — idêntica ao BEEP880 original
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn_b * ba + wm_b * bm
        bm   = beta.copy()
        beta = np.clip(beta, 0.05, PHI ** 3)

        beta_max  = float(beta.max())
        coh_med   = float(cohs.mean())
        R_audio, E_geom, E_coer = medir_R_audio(cohs)

        hist_beta_max.append(beta_max)
        hist_coh_med.append(coh_med)
        hist_R_audio.append(R_audio)

        sinal_final = passos[-1]
        coh_final   = cohs

        if beta_max >= LIMIAR and ciclo_conv == N_CICLOS:
            ciclo_conv = ciclo + 1
            print(f"  ✓ Campo harmônico → ciclo {ciclo_conv:2d}/{N_CICLOS}"
                  f"  β_max={beta_max:.4f}  φ³={PHI**3:.4f}")

        if (ciclo + 1) % 5 == 0:
            print(f"  ciclo {ciclo+1:2d}"
                  f"  β_max={beta_max:.4f}"
                  f"  coh={coh_med:.4f}"
                  f"  R={R_audio:.6f}"
                  f"  R/α={R_audio/ALPHA_FINE:.3f}")

    campo_ok = ciclo_conv < N_CICLOS
    R_med    = float(np.mean(hist_R_audio))
    R_fin    = hist_R_audio[-1]
    print(f"  {'✓ CAMPO FORMADO' if campo_ok else '✗ campo NÃO formado'}"
          f"  β_max={hist_beta_max[-1]:.4f}")
    print(f"  R_audio médio={R_med:.6f} = {R_med/ALPHA_FINE:.3f}α"
          f"  |  R_final={R_fin:.6f} = {R_fin/ALPHA_FINE:.3f}α")

    resultados[cfg_nome] = {
        "hist_beta_max" : hist_beta_max,
        "hist_coh_med"  : hist_coh_med,
        "hist_R_audio"  : hist_R_audio,
        "ciclo_conv"    : ciclo_conv,
        "campo_ok"      : campo_ok,
        "beta_max_final": hist_beta_max[-1],
        "coh_final"     : float(coh_final.mean()),
        "R_medio"       : R_med,
        "R_final"       : R_fin,
        "sinal_final"   : sinal_final,
        "label"         : cfg["label"],
        "color"         : cfg["color"],
    }

    # Salvar áudio
    label_arquivo = cfg["label"].split()[0].lower()
    wavfile_path  = f"alphaphi_eco_alpha_{label_arquivo}.wav"
    try:
        from scipy.io import wavfile as wf
        wf.write(wavfile_path,
                 FS,
                 (sinal_final * 32767).astype(np.int16))
        print(f"  → {wavfile_path}")
    except Exception:
        pass

# ── Análise final ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SÍNTESE — α como entropia no eco ressonante / beep 880")
print("=" * 65)
print(f"\n{'Config':<44} {'Campo?':>7} {'Ciclo':>6} {'β_max':>7}"
      f" {'R_med':>9} {'R_med/α':>8} {'coh_fin':>8}")
print("-" * 100)

for cfg_nome, res in resultados.items():
    campo = "✓" if res["campo_ok"] else "✗"
    ciclo = str(res["ciclo_conv"]) if res["campo_ok"] else "—"
    print(f"{cfg_nome:<44} {campo:>7} {ciclo:>6} {res['beta_max_final']:>7.4f}"
          f" {res['R_medio']:>9.6f} {res['R_medio']/ALPHA_FINE:>8.3f}"
          f" {res['coh_final']:>8.4f}")

# ── Questão central: R_natural ≈ α? ──────────────────────────────────────────
print("\n" + "─" * 65)
print("QUESTÃO 1 — R_audio ≈ α no baseline AA? (transfer Exp 6)")
aa_R   = resultados["AA — original    (grain=1e-10, sinal=1/3)"]["R_medio"]
aa_coh = resultados["AA — original    (grain=1e-10, sinal=1/3)"]["coh_final"]
print(f"  AA R_audio médio = {aa_R:.6f}  = {aa_R/ALPHA_FINE:.3f}α")
print(f"  AA coh_final     = {aa_coh:.4f}")
if 0.5 <= aa_R / ALPHA_FINE <= 2.0:
    print("  → CONFIRMADO: R_natural ∈ [0.5α, 2α]")
    print("    α é o equilíbrio natural do eco ressonante — como no SST2.")
    print("    Próximo passo: atrator R_audio → α (Exp 6 transfer no áudio).")
elif aa_R / ALPHA_FINE < 0.5:
    print(f"  → R_natural < 0.5α — equilíbrio está abaixo de α")
    print(f"    α grão (BB) pode puxar R para cima — verificar tendência BB.")
else:
    print(f"  → R_natural > 2α — equilíbrio acima de α")
    print(f"    α grão pode comprimir entropia — verificar BB.")

print("\nQUESTÃO 2 — α como grão (BB) preserva o campo harmônico?")
aa_ok = resultados["AA — original    (grain=1e-10, sinal=1/3)"]["campo_ok"]
bb_ok = resultados["BB — α-grão      (grain=α_FINE, sinal=1/3)"]["campo_ok"]
aa_ciclo = resultados["AA — original    (grain=1e-10, sinal=1/3)"]["ciclo_conv"]
bb_ciclo = resultados["BB — α-grão      (grain=α_FINE, sinal=1/3)"]["ciclo_conv"]
if aa_ok and bb_ok:
    dif = bb_ciclo - aa_ciclo
    if dif < 0:
        print(f"  → BB converge {abs(dif)} ciclos MAIS RÁPIDO que AA — α acelera o campo")
    elif dif > 0:
        print(f"  → BB converge {dif} ciclos mais lento que AA — α desacelera levemente")
    else:
        print(f"  → AA e BB convergem no mesmo ciclo {aa_ciclo} — α neutro na velocidade")
    print("  → Campo harmônico preservado com α como grão de entropia ✓")
elif aa_ok and not bb_ok:
    print("  → AA forma o campo, BB não — α como grão perturba a convergência")
    print("  → Revisar: clip ALPHA_FINE pode ser grande demais para esta escala")
elif not aa_ok and not bb_ok:
    print("  → Nenhum formou o campo em 20 ciclos — verificar sinal e parâmetros")

print("\nQUESTÃO 3 — CC (α_sinal=1/137): beep quase ausente")
cc_ok  = resultados["CC — α-sinal     (grain=1e-10, sinal=α_FINE)"]["campo_ok"]
cc_bm  = resultados["CC — α-sinal     (grain=1e-10, sinal=α_FINE)"]["beta_max_final"]
print(f"  → {'Campo formado' if cc_ok else 'Campo NÃO formado'}  β_max={cc_bm:.4f}")
print(f"  → α_sinal=1/3 é necessário para o campo harmônico?"
      f"  {'SIM' if aa_ok and not cc_ok else 'NÃO' if aa_ok and cc_ok else '—'}")

# ── Visualização ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor("#0d1117")

ciclos_eixo = list(range(1, N_CICLOS + 1))

# Gráfico 1 — β_max por ciclo
ax = axes[0]
ax.set_facecolor("#161b22")
ax.set_title("β_max por ciclo\n(Campo harmônico: β ≥ φ³)",
             color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "BB" in cfg_nome else 1.5 if "AA" in cfg_nome else 1.0
    ax.plot(ciclos_eixo, res["hist_beta_max"],
            color=res["color"], linewidth=lw, label=res["label"],
            marker="o", markersize=3)
ax.axhline(PHI**3, color="#DAA520", linestyle="--", alpha=0.7,
           label=f"φ³={PHI**3:.3f}")
ax.axhline(ALPHA_FINE, color="#FF4466", linestyle=":", alpha=0.4,
           label=f"α={ALPHA_FINE:.4f}")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("β_max", color="#8B949E")
ax.grid(True, alpha=0.15)

# Gráfico 2 — R_audio por ciclo (questão central)
ax = axes[1]
ax.set_facecolor("#161b22")
ax.set_title(
    "R_audio = E_geom/E_coer por ciclo\n"
    "Transfer Exp 6: R_natural ≈ α?",
    color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "BB" in cfg_nome else 1.5 if "AA" in cfg_nome else 1.0
    ax.plot(ciclos_eixo, res["hist_R_audio"],
            color=res["color"], linewidth=lw, label=res["label"],
            marker="o", markersize=3)
ax.axhline(ALPHA_FINE, color="#00FF88", linestyle="--", linewidth=2,
           label=f"α = {ALPHA_FINE:.5f}", alpha=0.8)
ax.axhline(ALPHA_FINE * 0.7, color="#00FF88", linestyle=":", alpha=0.4,
           label=f"0.7α")
ax.axhline(ALPHA_FINE * 2.0, color="#00FF88", linestyle=":", alpha=0.4,
           label=f"2α")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("R_audio", color="#8B949E")
ax.grid(True, alpha=0.15)

# Gráfico 3 — coh_med por ciclo
ax = axes[2]
ax.set_facecolor("#161b22")
ax.set_title("Coerência média por ciclo\n(alto = campo ordenado)",
             color="#DAA520", fontweight="bold", fontsize=10)
for cfg_nome, res in resultados.items():
    lw = 2.5 if "BB" in cfg_nome else 1.5 if "AA" in cfg_nome else 1.0
    ax.plot(ciclos_eixo, res["hist_coh_med"],
            color=res["color"], linewidth=lw, label=res["label"],
            marker="o", markersize=3)
coh_para_alpha = 1.0 - np.sqrt(ALPHA_FINE * (1.0 - ALPHA_FINE**0.5))
ax.axhline(coh_para_alpha, color="#00FF88", linestyle="--", alpha=0.6,
           label=f"coh para R≈α ≈ {coh_para_alpha:.3f}")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
ax.tick_params(colors="#8B949E")
for sp in ax.spines.values(): sp.set_color("#30363d")
ax.set_xlabel("Ciclo", color="#8B949E")
ax.set_ylabel("coh_med", color="#8B949E")
ax.grid(True, alpha=0.15)

fig.suptitle(
    f"AlphaPhi — Eco Ressonante + α como Entropia  |  Transfer Exps 1–6 → Beep 880\n"
    f"AA: original  |  BB: α-grão  |  CC: α-sinal  |  DD: ambos"
    f"  |  φ={PHI:.4f}  α=1/{round(1/ALPHA_FINE)}  |  Florianópolis 2026",
    color="#DAA520", fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig("alphaphi_eco_ressonante_alpha.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Gráfico: alphaphi_eco_ressonante_alpha.png")

# ── Áudio comparativo ─────────────────────────────────────────────────────────
print("\nAudioPlay:")
for cfg_nome, res in resultados.items():
    if res["campo_ok"]:
        print(f"  {res['label']} — campo formado ciclo {res['ciclo_conv']}")
        display(Audio(res["sinal_final"], rate=FS))

# ── Síntese final ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SÍNTESE — Seis experimentos SST2 → Eco Ressonante → Beep 880")
print("=" * 65)
print(f"""
Exps 1–5 (SST2): α INJETADO como multiplicador externo → null em todos

Exp 6 (SST2): R = E_φ/E_CE ≈ 0.7α emergente — α é equilíbrio natural

Este experimento — mesma pergunta no domínio acústico:
  R_audio = E_geom/E_coer — α emerge como equilíbrio no eco beep 880?

  AA (sem α): R_audio médio = {resultados["AA — original    (grain=1e-10, sinal=1/3)"]["R_medio"]:.6f} = {resultados["AA — original    (grain=1e-10, sinal=1/3)"]["R_medio"]/ALPHA_FINE:.3f}α
  BB (α-grão): R_audio médio = {resultados["BB — α-grão      (grain=α_FINE, sinal=1/3)"]["R_medio"]:.6f} = {resultados["BB — α-grão      (grain=α_FINE, sinal=1/3)"]["R_medio"]/ALPHA_FINE:.3f}α

Trajetória validada: eco_ressonante_alpha → eco_beep_880_alpha
  {'✓ Campo harmônico preservado com α como grão de entropia' if resultados["BB — α-grão      (grain=α_FINE, sinal=1/3)"]["campo_ok"] else '✗ Campo afetado — revisar implementação de α'}
  {'✓ R_audio natural ≈ α — confirma equilíbrio (transfer Exp 6)' if 0.5 <= resultados["AA — original    (grain=1e-10, sinal=1/3)"]["R_medio"]/ALPHA_FINE <= 2.0 else '→ R_audio ≠ α — equilíbrio em diferente escala'}
""")
print("alpha-phi")
