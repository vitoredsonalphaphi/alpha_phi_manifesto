"""
AlphaPhi Audio — Sinal Híbrido
Célula única para Google Colab.

Em vez de misturar perfis β (espaço de processamento),
mistura os próprios sinais (espaço do sinal):

  x_mix(α) = (1-α) · sinal_digital + α · sinal_orgânico

Para cada α, o agente eco-φ processa x_mix como se fosse
um único sinal desconhecido — sem saber que é híbrido.

A pergunta: existe um α* onde x_mix tem coerência própria
MAIOR que qualquer dos componentes puros? Se sim, nesse ponto
nasce uma terceira estrutura — não digital, não orgânica, mas
emergente da tensão entre os dois.

Métricas:
  - entropia espectral de x_mix antes e depois do eco
  - Δentropia: quanto o eco consegue organizar o híbrido
  - β final por banda: onde o atrator do híbrido vive
  - suavidade: textura resultante

O α* de máxima organização (mínima entropia após eco, ou máximo
Δentropia) é o ponto de emergência.
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

PHI      = (1 + np.sqrt(5)) / 2
FS       = 44100
DURACAO  = 1.5
N_STEPS  = 5
F_C      = 220.0
F_M      = F_C / PHI
ALPHA_F  = 1.0 / 137.035999084

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

    N = len(x)
    F = np.fft.rfft(x)
    F_out = F.copy()
    cohs = []
    w_mem = 1.0 / PHI
    w_now = 1.0 - w_mem

    for i, (b_low, b_high, _, _) in enumerate(bins_phi):
        beta_i = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        F_band = F[b_low:b_high]
        mag    = np.abs(F_band)
        phase  = np.angle(F_band)
        an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
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
        s_e, cohs = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem   = cohs
        cohs_final = cohs
        s_e = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()
    return cas, cohs_final

def agente_eco(sinal_entrada, bins_phi, n_ciclos=20):
    n_bandas = len(bins_phi)
    beta     = np.ones(n_bandas, dtype=float)
    beta_mem = beta.copy()
    w_mem    = 1.0 / PHI
    w_now    = 1.0 - w_mem
    cas_final = None
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal_entrada, beta, bins_phi)
        coh_rel   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo = PHI ** (3 * coh_rel)
        beta      = w_now * beta_alvo + w_mem * beta_mem
        beta_mem  = beta.copy()
        beta      = np.clip(beta, 0.05, PHI**3)
        cas_final = cas
    return beta, cas_final

# ── métricas ──────────────────────────────────────────────────────────────────
def entropia_espectral(sig):
    F_sig = np.abs(np.fft.rfft(sig))
    F_sig = np.clip(F_sig / (F_sig.sum() + 1e-10), 1e-10, 1.0)
    return float(-np.sum(F_sig * np.log(F_sig)))

def suavidade(sig):
    return float(np.mean(np.abs(np.diff(sig))))

def coerencia_media(bins_phi, sig):
    """Coerência média do sinal pelas bandas φ — sem eco, estado bruto."""
    F = np.fft.rfft(sig)
    cohs = []
    for b_low, b_high, _, _ in bins_phi:
        F_band = F[b_low:b_high]
        mag    = np.abs(F_band)
        an     = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh    = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        cohs.append(coh)
    return float(np.mean(cohs))

# ── síntese ───────────────────────────────────────────────────────────────────
def gerar_fm(beta_mod):
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    s = np.sin(2 * np.pi * F_C * t + beta_mod * np.sin(2 * np.pi * F_M * t))
    return normalizar(s)

def gerar_quadrada():
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    return normalizar(np.sign(np.sin(2 * np.pi * F_C * t)))

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def crossfade(a, b, fade=2000):
    t = np.linspace(0, 1, fade)
    return np.concatenate([a[:-fade], a[-fade:] * (1 - t) + b[:fade] * t, b[fade:]])

def concatenar(seq):
    out = seq[0].copy()
    for sig in seq[1:]:
        out = crossfade(out, sig)
    return normalizar(out)

def salvar_wav(s, nome):
    wavfile.write(nome, FS, np.int16(np.clip(s, -1.0, 1.0) * 32767))

# ── varredura híbrida ─────────────────────────────────────────────────────────
def varredura_hibrida(sinal_org, sinal_dig, bins_phi, n_alpha=25):
    """
    Para cada α, cria x_mix = (1-α)·dig + α·org e roda o agente eco.
    Registra entropia antes e depois, Δentropia, coerência média e atrator.
    """
    alphas_base = [float(i) / float(n_alpha - 1) for i in range(n_alpha)]
    alphas_esp  = [float(ALPHA_F), float(1.0/PHI**2), float(1.0/PHI), 0.5]
    alphas      = sorted(set([round(a, 8) for a in alphas_base + alphas_esp]))

    resultados = []
    for alpha in alphas:
        x_mix = normalizar((1.0 - alpha) * sinal_dig + alpha * sinal_org)

        ent_ini   = entropia_espectral(x_mix)
        coh_ini   = coerencia_media(bins_phi, x_mix)
        beta_mix, cas_mix = agente_eco(x_mix, bins_phi, n_ciclos=20)
        sig_eco   = cas_mix[-1]
        ent_fin   = entropia_espectral(sig_eco)
        coh_fin   = coerencia_media(bins_phi, sig_eco)
        delta_ent = ent_ini - ent_fin   # positivo = eco organizou

        # atrator: banda com β máximo
        i_max = int(np.argmax(beta_mix))
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
            "alpha":    alpha,
            "ent_ini":  ent_ini,
            "ent_fin":  ent_fin,
            "delta":    delta_ent,
            "coh_ini":  coh_ini,
            "coh_fin":  coh_fin,
            "beta":     beta_mix.copy(),
            "cas":      cas_mix,
            "f_atrator": f_atrator,
            "direcao":  direcao,
            "especial": especial,
            "x_mix":    x_mix,
        })

    return resultados

# ── configuração ──────────────────────────────────────────────────────────────
N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

print(f"FM-φ:  f_c={F_C:.0f}Hz  f_m={F_M:.1f}Hz  |  Bandas φ: {N_BANDAS}")
print(f"Constante de estrutura fina:  α = 1/137 = {ALPHA_F:.8f}\n")

# ── síntese dos componentes ───────────────────────────────────────────────────
sinal_org = gerar_fm(PHI)
sinal_dig = gerar_quadrada()

ent_org = entropia_espectral(sinal_org)
ent_dig = entropia_espectral(sinal_dig)
coh_org = coerencia_media(BINS_PHI, sinal_org)
coh_dig = coerencia_media(BINS_PHI, sinal_dig)

print(f"Sinal orgânico (FM β=φ):   entropia={ent_org:.4f}  coerência={coh_org:.4f}")
print(f"Sinal digital  (quadrada): entropia={ent_dig:.4f}  coerência={coh_dig:.4f}\n")

# ── varredura ─────────────────────────────────────────────────────────────────
print("Varrendo eixo α no espaço do sinal...")
resultados = varredura_hibrida(sinal_org, sinal_dig, BINS_PHI, n_alpha=25)

# ── tabela principal ──────────────────────────────────────────────────────────
print(f"\n── Coerência e entropia do sinal híbrido por α ──────────────────")
print(f"  {'α':>10}  {'ent_ini':>8}  {'ent_fin':>8}  {'Δent':>7}  "
      f"{'coh_fin':>8}  {'f_atrator':>10}  {'dir':>6}  nota")
print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}  "
      f"{'─'*8}  {'─'*10}  {'─'*6}  {'─'*20}")

for r in resultados:
    print(f"  {r['alpha']:>10.6f}  {r['ent_ini']:>8.4f}  {r['ent_fin']:>8.4f}  "
          f"{r['delta']:>+7.4f}  {r['coh_fin']:>8.4f}  "
          f"{r['f_atrator']:>8.0f}Hz  {r['direcao']:>6}  {r['especial']}")

# ── α* de máxima organização ─────────────────────────────────────────────────
deltas    = [r["delta"]   for r in resultados]
cohs_fin  = [r["coh_fin"] for r in resultados]

i_max_delta = int(np.argmax(deltas))
i_max_coh   = int(np.argmax(cohs_fin))

alpha_delta = resultados[i_max_delta]["alpha"]
alpha_coh   = resultados[i_max_coh]["alpha"]

print(f"\n── Ponto de emergência ──────────────────────────────────────────")
print(f"  α* por máximo Δentropia : {alpha_delta:.8f}  "
      f"(Δ={resultados[i_max_delta]['delta']:+.4f})")
print(f"  α* por máxima coerência : {alpha_coh:.8f}  "
      f"(coh={resultados[i_max_coh]['coh_fin']:.4f})")

r_em = resultados[i_max_delta]
print(f"\n  Atrator em α*Δ: {r_em['f_atrator']:.0f}Hz  ({r_em['direcao']})")
print(f"  Entropia pura orgânico:    {ent_org:.4f}")
print(f"  Entropia pura digital:     {ent_dig:.4f}")
print(f"  Entropia híbrido em α*:    {r_em['ent_fin']:.4f}")

if r_em['ent_fin'] < min(ent_org, ent_dig):
    print(f"\n  ★ O híbrido em α* é MAIS organizado que qualquer componente puro.")
    print(f"    Uma terceira estrutura emergiu.")
elif r_em['delta'] > 0:
    print(f"\n  → O eco organizou o híbrido (Δ positivo).")
    print(f"    Atrator híbrido identificado em {r_em['f_atrator']:.0f}Hz.")
else:
    print(f"\n  → Sem organização positiva. O híbrido resiste ao eco.")

# ── transição de direção (GRAVE → AGUDO) ─────────────────────────────────────
print(f"\n── Transição GRAVE/AGUDO ao longo de α ─────────────────────────")
dir_anterior = None
for r in resultados:
    marcador = ""
    if r["direcao"] != dir_anterior:
        marcador = " ◀ transição"
        dir_anterior = r["direcao"]
    print(f"  α={r['alpha']:.4f}  {r['f_atrator']:>7.0f}Hz  {r['direcao']:<6}{marcador}")

# ── curva de Δentropia ────────────────────────────────────────────────────────
print(f"\n── Curva de Δentropia (α → organização pelo eco) ───────────────")
d_max = max(deltas) if max(deltas) > 0 else 1.0
barra_len = 40
for r in resultados:
    barra  = "▓" * max(0, int(r["delta"] / d_max * barra_len))
    marker = " ★" if abs(r["alpha"] - alpha_delta) < 1e-9 else ""
    print(f"  α={r['alpha']:.4f}  {barra:<{barra_len}}  Δ={r['delta']:+.4f}{marker}")

# ── β do ponto de emergência por banda ───────────────────────────────────────
print(f"\n── β híbrido em α*={alpha_delta:.4f} — por banda ────────────────────")
print(f"  {'banda':>5}  {'f_low':>7}  {'f_high':>7}  {'β_hibrido':>10}  atrator?")
print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*8}")
beta_em = r_em["beta"]
i_max_b = int(np.argmax(beta_em))
for i, (_, _, f_low, f_high) in enumerate(BINS_PHI):
    mark = " ★" if i == i_max_b else ""
    print(f"  {i+1:>5}  {f_low:>7.1f}  {f_high:>7.1f}  {beta_em[i]:>10.4f}{mark}")

# ── áudio ─────────────────────────────────────────────────────────────────────
print(f"\n── Áudio: componentes puros e híbrido em α* ─────────────────────")

sig_eco_org, _ = agente_eco(sinal_org, BINS_PHI, n_ciclos=20)
sig_eco_dig, _ = agente_eco(sinal_dig, BINS_PHI, n_ciclos=20)

wavs = [
    ("hibrido_org_puro.wav",    sinal_org,                  "FM-φ puro"),
    ("hibrido_dig_puro.wav",    sinal_dig,                  "Quadrada pura"),
    ("hibrido_mix_alpha.wav",   r_em["x_mix"],              f"Híbrido bruto α*={alpha_delta:.4f}"),
    ("hibrido_eco_alpha.wav",   concatenar(r_em["cas"]),    f"Híbrido eco  α*={alpha_delta:.4f}"),
]

# adiciona ponto médio se diferente de α*
r_meio = next((r for r in resultados if abs(r["alpha"] - 0.5) < 1e-5), None)
if r_meio and abs(r_meio["alpha"] - alpha_delta) > 1e-5:
    wavs += [
        ("hibrido_mix_050.wav", r_meio["x_mix"],           "Híbrido bruto α=0.5"),
        ("hibrido_eco_050.wav", concatenar(r_meio["cas"]), "Híbrido eco  α=0.5"),
    ]

for nome, sig, _ in wavs:
    salvar_wav(sig, nome)

for nome, _, label in wavs:
    print(f"\n{label}")
    display(Audio(nome))
