"""
AlphaPhi Audio — Beep de Interface
Célula única para Google Colab.

Testa se α=1/137 emerge como ponto de organização máxima
quando o sinal digital é um beep real de interface (880Hz / 440Hz)
em vez do sinal quadrado sintético dos experimentos anteriores.

Pergunta central: o princípio generaliza?
Se α=1/137 emerge aqui também — com sinal de interface real —
a hipótese da modulação ergonômica tem fundamento para qualquer
emissão digital, não apenas para sinais de teste.

Sinais testados:
  - Beep 880Hz  (pitch típico de notificação — tom agudo)
  - Beep 440Hz  (pitch de alerta — tom médio)
  - FM-φ        (sinal orgânico de referência — mesmo de antes)

Método: varredura_hibrida idêntica ao Audio_Hibrido —
  x_mix(α) = (1-α)·beep + α·FM-φ
  para cada α, mede Δentropia e coerência após eco-φ.
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

PHI     = (1 + np.sqrt(5)) / 2
FS      = 44100
DURACAO = 1.5
N_STEPS = 5
ALPHA_F = 1.0 / 137.035999084

F_ORG   = 220.0          # frequência base do FM-φ (mantida igual)
F_M     = F_ORG / PHI    # frequência de modulação FM

F_BEEP_ALTO  = 880.0     # notificação — tom agudo
F_BEEP_MEDIO = 440.0     # alerta — tom médio

# ── bandas e bins ─────────────────────────────────────────────────────────────
def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max:
            break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n_samples):
    resultado = []
    for f_low, f_high in bandas:
        b_low  = max(int(np.floor(f_low  * n_samples / FS)), 0)
        b_high = min(int(np.ceil(f_high  * n_samples / FS)), n_samples // 2 + 1)
        if b_high - b_low >= 3:
            resultado.append((b_low, b_high, f_low, f_high))
    return resultado

# ── eco EQ ────────────────────────────────────────────────────────────────────
def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N     = len(x)
    F     = np.fft.rfft(x)
    F_out = F.copy()
    cohs  = []
    w_mem = 1.0 / PHI
    w_now = 1.0 - w_mem
    for i, (b_low, b_high, _, _) in enumerate(bins_phi):
        beta_i = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        F_band = F[b_low:b_high]
        mag    = np.abs(F_band)
        phase  = np.angle(F_band)
        an     = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh    = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        if coh_mem is not None and i < len(coh_mem):
            coh_ef = w_now * coh + w_mem * float(coh_mem[i])
        else:
            coh_ef = coh
        cohs.append(coh)
        n_idx = np.arange(len(F_band))
        env   = np.clip(
            1.0 + (coh_ef * PHI**beta_i) * np.cos(2.0 * np.pi * n_idx / PHI),
            0.05, None
        )
        F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase)
    resultado = np.fft.irfft(F_out, n=N)
    return resultado / (np.max(np.abs(resultado)) + 1e-10), np.array(cohs, dtype=float)

def cascata_eq(sinal_entrada, beta_bands, bins_phi):
    cas        = [sinal_entrada]
    s          = sinal_entrada.copy()
    n_bandas   = len(bins_phi)
    coh_mem    = np.zeros(n_bandas, dtype=float)
    cohs_final = np.zeros(n_bandas, dtype=float)
    for _ in range(N_STEPS):
        s_e, cohs  = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem    = cohs
        cohs_final = cohs
        s_e        = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()
    return cas, cohs_final

def agente_eco(sinal_entrada, bins_phi, n_ciclos=20):
    n_bandas  = len(bins_phi)
    beta      = np.ones(n_bandas, dtype=float)
    beta_mem  = beta.copy()
    w_mem     = 1.0 / PHI
    w_now     = 1.0 - w_mem
    cas_final = None
    for _ in range(n_ciclos):
        cas, cohs  = cascata_eq(sinal_entrada, beta, bins_phi)
        coh_rel    = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo  = PHI ** (3 * coh_rel)
        beta       = w_now * beta_alvo + w_mem * beta_mem
        beta_mem   = beta.copy()
        beta       = np.clip(beta, 0.05, PHI**3)
        cas_final  = cas
    return beta, cas_final

# ── métricas ──────────────────────────────────────────────────────────────────
def entropia_espectral(sig):
    F_sig = np.abs(np.fft.rfft(sig))
    F_sig = np.clip(F_sig / (F_sig.sum() + 1e-10), 1e-10, 1.0)
    return float(-np.sum(F_sig * np.log(F_sig)))

def coerencia_media(bins_phi, sig):
    F    = np.fft.rfft(sig)
    cohs = []
    for b_low, b_high, _, _ in bins_phi:
        F_band = F[b_low:b_high]
        mag    = np.abs(F_band)
        an     = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh    = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        cohs.append(coh)
    return float(np.mean(cohs))

def crest_factor(sig):
    """Razão pico/RMS — menor = mais orgânico, menos abrupto."""
    rms = float(np.sqrt(np.mean(sig**2)))
    return float(np.max(np.abs(sig))) / (rms + 1e-10)

# ── síntese ───────────────────────────────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_beep(freq, duracao=DURACAO):
    """Onda quadrada com envelope abrupto — simula beep de interface."""
    t   = np.linspace(0, duracao, int(FS * duracao), endpoint=False)
    sig = np.sign(np.sin(2 * np.pi * freq * t))
    return normalizar(sig)

def gerar_fm_phi():
    t   = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    sig = np.sin(2 * np.pi * F_ORG * t + PHI * np.sin(2 * np.pi * F_M * t))
    return normalizar(sig)

def crossfade(a, b, fade=2000):
    t = np.linspace(0, 1, fade)
    return np.concatenate([a[:-fade], a[-fade:] * (1-t) + b[:fade] * t, b[fade:]])

def concatenar(seq):
    out = seq[0].copy()
    for sig in seq[1:]:
        out = crossfade(out, sig)
    return normalizar(out)

def salvar_wav(s, nome):
    wavfile.write(nome, FS, np.int16(np.clip(s, -1.0, 1.0) * 32767))

# ── varredura híbrida ─────────────────────────────────────────────────────────
def varredura_hibrida(sinal_org, sinal_dig, bins_phi, label_dig, n_alpha=25):
    alphas_base = [float(i) / float(n_alpha - 1) for i in range(n_alpha)]
    alphas_esp  = [float(ALPHA_F), float(1.0/PHI**2), float(1.0/PHI), 0.5]
    alphas      = sorted(set([round(a, 8) for a in alphas_base + alphas_esp]))

    resultados = []
    for alpha in alphas:
        x_mix     = normalizar((1.0 - alpha) * sinal_dig + alpha * sinal_org)
        ent_ini   = entropia_espectral(x_mix)
        coh_ini   = coerencia_media(bins_phi, x_mix)
        beta_mix, cas_mix = agente_eco(x_mix, bins_phi, n_ciclos=20)
        sig_eco   = cas_mix[-1]
        ent_fin   = entropia_espectral(sig_eco)
        coh_fin   = coerencia_media(bins_phi, sig_eco)
        delta_ent = ent_ini - ent_fin
        cf_ini    = crest_factor(x_mix)
        cf_fin    = crest_factor(sig_eco)

        i_max     = int(np.argmax(beta_mix))
        _, _, f_low_max, f_high_max = bins_phi[i_max]
        f_atrator = (f_low_max + f_high_max) / 2
        direcao   = "GRAVE" if f_atrator < 400 else "AGUDO"

        especial = ""
        if abs(alpha - ALPHA_F) < 1e-5:
            especial = " ← α=1/137"
        elif abs(alpha - 1.0/PHI**2) < 1e-5:
            especial = " ← α=1/φ²"
        elif abs(alpha - 1.0/PHI) < 1e-5:
            especial = " ← α=1/φ"
        elif abs(alpha - 0.5) < 1e-5:
            especial = " ← α=1/2"

        resultados.append({
            "alpha": alpha, "ent_ini": ent_ini, "ent_fin": ent_fin,
            "delta": delta_ent, "coh_ini": coh_ini, "coh_fin": coh_fin,
            "cf_ini": cf_ini, "cf_fin": cf_fin,
            "beta": beta_mix.copy(), "cas": cas_mix,
            "f_atrator": f_atrator, "direcao": direcao,
            "especial": especial, "x_mix": x_mix,
        })
    return resultados

def imprimir_resultado(resultados, label):
    deltas   = [r["delta"]   for r in resultados]
    cohs_fin = [r["coh_fin"] for r in resultados]
    i_max_d  = int(np.argmax(deltas))
    i_max_c  = int(np.argmax(cohs_fin))
    alpha_d  = resultados[i_max_d]["alpha"]
    alpha_c  = resultados[i_max_c]["alpha"]
    r_em     = resultados[i_max_d]

    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    print(f"  {'α':>10}  {'ent_ini':>8}  {'ent_fin':>8}  {'Δent':>7}  "
          f"{'coh_fin':>8}  {'CF_ini':>6}  {'CF_fin':>6}  nota")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}  "
          f"{'─'*8}  {'─'*6}  {'─'*6}  {'─'*15}")
    for r in resultados:
        print(f"  {r['alpha']:>10.6f}  {r['ent_ini']:>8.4f}  {r['ent_fin']:>8.4f}  "
              f"{r['delta']:>+7.4f}  {r['coh_fin']:>8.4f}  "
              f"{r['cf_ini']:>6.2f}  {r['cf_fin']:>6.2f}  {r['especial']}")

    print(f"\n── Ponto de emergência ──────────────────────────────────")
    print(f"  α* Δentropia : {alpha_d:.8f}  (Δ={r_em['delta']:+.4f})")
    print(f"  α* coerência : {alpha_c:.8f}  (coh={resultados[i_max_c]['coh_fin']:.4f})")
    print(f"  Atrator      : {r_em['f_atrator']:.0f}Hz  ({r_em['direcao']})")
    print(f"  Entropia beep puro  : {entropia_espectral(sinal_dig_atual):.4f}")
    print(f"  Entropia FM-φ puro  : {entropia_espectral(sinal_org):.4f}")
    print(f"  Entropia híbrido α* : {r_em['ent_fin']:.4f}")
    print(f"  Crest factor beep   : {crest_factor(sinal_dig_atual):.4f}")
    print(f"  Crest factor eco α* : {r_em['cf_fin']:.4f}")

    if r_em['ent_fin'] < min(entropia_espectral(sinal_org),
                             entropia_espectral(sinal_dig_atual)):
        print(f"\n  ★ Híbrido em α* é MAIS organizado que qualquer componente puro.")
        print(f"    Terceira estrutura emergiu.")
    elif r_em['delta'] > 0:
        print(f"\n  → Eco organizou o híbrido (Δ positivo).")

    return r_em, alpha_d

# ── configuração ──────────────────────────────────────────────────────────────
N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

print(f"Bandas φ: {N_BANDAS}  |  α=1/137 = {ALPHA_F:.8f}\n")

sinal_org = gerar_fm_phi()

# ── experimento 1: beep 880Hz (notificação aguda) ─────────────────────────────
print("Gerando beep 880Hz...")
sinal_dig_atual = gerar_beep(F_BEEP_ALTO)
print(f"Varredura α — Beep {F_BEEP_ALTO:.0f}Hz vs FM-φ...")
res_880 = varredura_hibrida(sinal_org, sinal_dig_atual, BINS_PHI,
                            f"Beep {F_BEEP_ALTO:.0f}Hz (notificação aguda)")
r_em_880, alpha_880 = imprimir_resultado(res_880, f"Beep {F_BEEP_ALTO:.0f}Hz")

# ── experimento 2: beep 440Hz (alerta médio) ──────────────────────────────────
print(f"\n\nGerando beep 440Hz...")
sinal_dig_atual = gerar_beep(F_BEEP_MEDIO)
print(f"Varredura α — Beep {F_BEEP_MEDIO:.0f}Hz vs FM-φ...")
res_440 = varredura_hibrida(sinal_org, sinal_dig_atual, BINS_PHI,
                            f"Beep {F_BEEP_MEDIO:.0f}Hz (alerta médio)")
r_em_440, alpha_440 = imprimir_resultado(res_440, f"Beep {F_BEEP_MEDIO:.0f}Hz")

# ── comparação entre os dois experimentos ─────────────────────────────────────
print(f"\n\n{'═'*60}")
print(f"  COMPARAÇÃO — 880Hz vs 440Hz vs experimento anterior")
print(f"{'═'*60}")
print(f"  {'sinal':>20}  {'α* Δent':>12}  {'Δent em α*':>12}  {'ent final':>10}")
print(f"  {'─'*20}  {'─'*12}  {'─'*12}  {'─'*10}")
print(f"  {'Beep 880Hz':>20}  {alpha_880:>12.8f}  "
      f"{res_880[next(i for i,r in enumerate(res_880) if abs(r['alpha']-alpha_880)<1e-9)]['delta']:>+12.4f}  "
      f"{r_em_880['ent_fin']:>10.4f}")
print(f"  {'Beep 440Hz':>20}  {alpha_440:>12.8f}  "
      f"{res_440[next(i for i,r in enumerate(res_440) if abs(r['alpha']-alpha_440)<1e-9)]['delta']:>+12.4f}  "
      f"{r_em_440['ent_fin']:>10.4f}")
print(f"  {'Quadrada 220Hz*':>20}  {'0.00729735':>12}  {'+6.5806':>12}  {'1.0521':>10}")
print(f"\n  * referência do experimento Audio_Hibrido anterior")

alpha_igual_880 = abs(alpha_880 - ALPHA_F) < 1e-5
alpha_igual_440 = abs(alpha_440 - ALPHA_F) < 1e-5
if alpha_igual_880 and alpha_igual_440:
    print(f"\n  ★★ α=1/137 emerge nos DOIS beeps — princípio generaliza.")
elif alpha_igual_880 or alpha_igual_440:
    print(f"\n  ◆ α=1/137 emerge em um dos beeps — generalização parcial.")
else:
    print(f"\n  → α=1/137 não emerge como α* — ponto de emergência diferente.")
    print(f"    Resultado a analisar.")

# ── áudio ─────────────────────────────────────────────────────────────────────
print(f"\n── Áudio ────────────────────────────────────────────────────")

wavs = [
    ("beep_org_puro.wav",      sinal_org,                       "FM-φ puro (orgânico)"),
    ("beep_880_puro.wav",      gerar_beep(F_BEEP_ALTO),         "Beep 880Hz puro (digital)"),
    ("beep_880_eco.wav",       concatenar(r_em_880["cas"]),     f"Beep 880Hz eco α*={alpha_880:.4f}"),
    ("beep_440_puro.wav",      gerar_beep(F_BEEP_MEDIO),        "Beep 440Hz puro (digital)"),
    ("beep_440_eco.wav",       concatenar(r_em_440["cas"]),     f"Beep 440Hz eco α*={alpha_440:.4f}"),
]

for nome, sig, _ in wavs:
    salvar_wav(sig, nome)

for nome, _, label in wavs:
    print(f"\n{label}")
    display(Audio(nome))
