"""
AlphaPhi Audio — Convergência Bidirecional
Célula única para Google Colab.

Dois sinais em tensão:
  sinal_org  →  FM-φ  (β=φ)   orgânico / contínuo
  sinal_dig  →  onda quadrada  digital  / discreto

O agente bidirecional mistura os perfis β de ambos ao longo de um
eixo α ∈ [0, 1]:

  β_mix(α) = (1-α) · β_dig + α · β_org

Para cada α, processa os dois sinais com β_mix e mede a distância
de coerência entre eles.  O α* onde a distância é mínima é o ponto
de afinação emergente — o instante em que orgânico e digital se
aproximam ao máximo sem se fundir.

Marcadores especiais:
  α = 1/137  →  constante de estrutura fina  (acoplamento eletromagnético)
  α = 1/φ²   →  sub-φ quadrático
  α = 1/φ    →  sub-φ linear
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
ALPHA_F  = 1.0 / 137.035999084   # constante de estrutura fina

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
    # garantia 1-D para evitar IndexError em arrays 0-D
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
    cas       = [sinal_entrada]
    s         = sinal_entrada.copy()
    n_bandas  = len(bins_phi)
    coh_mem   = np.zeros(n_bandas, dtype=float)
    cohs_final = np.zeros(n_bandas, dtype=float)

    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem   = cohs
        cohs_final = cohs
        s_e = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()

    return cas, cohs_final

# ── agente eco-ressonante (preservado) ───────────────────────────────────────
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

def distancia_coerencia(coh_a, coh_b):
    a = np.atleast_1d(np.asarray(coh_a, dtype=float))
    b = np.atleast_1d(np.asarray(coh_b, dtype=float))
    return float(np.sqrt(np.sum((a - b) ** 2)))

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

# ── agente bidirecional ───────────────────────────────────────────────────────
def agente_bidirecional(sinal_org, sinal_dig, beta_org, beta_dig, bins_phi, n_alpha=15):
    """
    Varre α de 0 → 1 misturando os perfis β dos dois sinais.
    Para cada α mede distância de coerência entre os dois sinais processados.
    Retorna tabela de resultados e α* (mínimo da distância).
    """
    beta_org = np.atleast_1d(np.asarray(beta_org, dtype=float))
    beta_dig = np.atleast_1d(np.asarray(beta_dig, dtype=float))

    # constrói lista de alphas: grade uniforme + especiais
    alphas_base   = [float(i) / float(n_alpha - 1) for i in range(n_alpha)]
    alphas_esp    = [float(ALPHA_F), float(1.0 / PHI**2), float(1.0 / PHI)]
    alphas        = sorted(set([round(a, 8) for a in alphas_base + alphas_esp]))

    resultados = []
    for alpha in alphas:
        beta_mix = (1.0 - alpha) * beta_dig + alpha * beta_org
        beta_mix = np.clip(beta_mix, 0.05, PHI**3)

        _, cohs_org = cascata_eq(sinal_org, beta_mix, bins_phi)
        _, cohs_dig = cascata_eq(sinal_dig, beta_mix, bins_phi)

        dist = distancia_coerencia(cohs_org, cohs_dig)
        ent_org = entropia_espectral(sinal_org)
        ent_dig = entropia_espectral(sinal_dig)

        especial = ""
        if abs(alpha - ALPHA_F) < 1e-5:
            especial = " ← α=1/137 (estrutura fina)"
        elif abs(alpha - 1.0 / PHI**2) < 1e-5:
            especial = " ← α=1/φ²"
        elif abs(alpha - 1.0 / PHI) < 1e-5:
            especial = " ← α=1/φ"

        resultados.append({
            "alpha":    alpha,
            "dist":     dist,
            "ent_org":  ent_org,
            "ent_dig":  ent_dig,
            "beta_mix": beta_mix.copy(),
            "especial": especial,
        })

    # α* = mínimo da distância de coerência
    i_min  = int(np.argmin([r["dist"] for r in resultados]))
    alpha_star = resultados[i_min]["alpha"]

    return resultados, alpha_star, i_min

# ── configuração ──────────────────────────────────────────────────────────────
N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

print(f"FM-φ:  f_c={F_C:.0f}Hz  f_m={F_M:.1f}Hz  |  Bandas φ: {N_BANDAS}")
print(f"Constante de estrutura fina:  α = 1/137 = {ALPHA_F:.8f}\n")

# ── síntese dos dois sinais ───────────────────────────────────────────────────
sinal_org = gerar_fm(PHI)         # FM-φ  (orgânico)
sinal_dig = gerar_quadrada()      # onda quadrada (digital)

print(f"Sinal orgânico (FM β=φ):  entropia = {entropia_espectral(sinal_org):.4f}")
print(f"Sinal digital  (quadrada): entropia = {entropia_espectral(sinal_dig):.4f}\n")

# ── agente eco em cada sinal separado ────────────────────────────────────────
print("Calculando perfis β individuais (agente eco)...")
beta_org, cas_org = agente_eco(sinal_org, BINS_PHI, n_ciclos=20)
beta_dig, cas_dig = agente_eco(sinal_dig, BINS_PHI, n_ciclos=20)

print(f"  β_org: min={beta_org.min():.4f}  max={beta_org.max():.4f}  "
      f"mean={beta_org.mean():.4f}")
print(f"  β_dig: min={beta_dig.min():.4f}  max={beta_dig.max():.4f}  "
      f"mean={beta_dig.mean():.4f}\n")

# ── varredura bidirecional ────────────────────────────────────────────────────
print("Varrendo eixo α  (0=digital puro → 1=orgânico puro)...")
resultados, alpha_star, i_min = agente_bidirecional(
    sinal_org, sinal_dig, beta_org, beta_dig, BINS_PHI, n_alpha=21
)

# ── tabela de resultados ──────────────────────────────────────────────────────
print(f"\n── Distância de coerência por α ─────────────────────────────────")
print(f"  {'α':>10}  {'dist_coh':>10}  nota")
print(f"  {'─'*10}  {'─'*10}  {'─'*30}")
for r in resultados:
    marker = " ★" if abs(r["alpha"] - alpha_star) < 1e-9 else ""
    print(f"  {r['alpha']:>10.6f}  {r['dist']:>10.6f}  {r['especial']}{marker}")

# ── ponto de afinação emergente ───────────────────────────────────────────────
r_star = resultados[i_min]
print(f"\n── Ponto de afinação emergente ──────────────────────────────────")
print(f"  α*  = {alpha_star:.8f}")
print(f"  α_fina_estrutura = {ALPHA_F:.8f}  (Δ = {abs(alpha_star - ALPHA_F):.8f})")
print(f"  distância mínima = {r_star['dist']:.6f}")

razao = alpha_star / ALPHA_F if ALPHA_F > 0 else float('inf')
print(f"  α* / α_137 = {razao:.4f}")

if razao < 1.5:
    print(f"\n  → α* próximo à constante de estrutura fina.")
    print(f"    O acoplamento orgânico/digital emerge perto do ponto")
    print(f"    onde elétrons acoplam à luz — φ e α como co-organizadores.")
else:
    print(f"\n  → α* difere de α_137 por fator {razao:.2f}.")
    print(f"    O acoplamento emerge em posição distinta.")

# ── β do ponto de afinação por banda ─────────────────────────────────────────
print(f"\n── β_mix no ponto α* — por banda ────────────────────────────────")
print(f"  {'banda':>5}  {'f_low':>7}  {'f_high':>7}  {'β_dig':>8}  {'β_org':>8}  {'β_mix*':>8}")
print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}")
beta_mix_star = r_star["beta_mix"]
for i, (_, _, f_low, f_high) in enumerate(BINS_PHI):
    print(f"  {i+1:>5}  {f_low:>7.1f}  {f_high:>7.1f}  "
          f"{beta_dig[i]:>8.4f}  {beta_org[i]:>8.4f}  {beta_mix_star[i]:>8.4f}")

# ── curva de distância ────────────────────────────────────────────────────────
print(f"\n── Curva de distância (α → dist_coh) ────────────────────────────")
alphas_plot = [r["alpha"] for r in resultados]
dists_plot  = [r["dist"]  for r in resultados]
d_max = max(dists_plot) if dists_plot else 1.0
barra_len = 40
for r in resultados:
    barra = "▓" * int(r["dist"] / d_max * barra_len)
    marker = " ★" if abs(r["alpha"] - alpha_star) < 1e-9 else ""
    print(f"  α={r['alpha']:.4f}  {barra:<{barra_len}}  {r['dist']:.4f}{marker}")

# ── áudio: sinal original, eco individual e eco no ponto α* ──────────────────
print(f"\n── Áudio: sinais e convergência em α* ───────────────────────────")

# processa os dois sinais com β_mix do ponto α*
cas_mix_org, _ = cascata_eq(sinal_org, beta_mix_star, BINS_PHI)
cas_mix_dig, _ = cascata_eq(sinal_dig, beta_mix_star, BINS_PHI)

wavs = [
    ("bidirecional_org_orig.wav",   sinal_org,             "FM-φ original"),
    ("bidirecional_org_eco.wav",    concatenar(cas_org),   "FM-φ eco individual"),
    ("bidirecional_org_mix.wav",    concatenar(cas_mix_org), f"FM-φ em α*={alpha_star:.4f}"),
    ("bidirecional_dig_orig.wav",   sinal_dig,             "Quadrada original"),
    ("bidirecional_dig_eco.wav",    concatenar(cas_dig),   "Quadrada eco individual"),
    ("bidirecional_dig_mix.wav",    concatenar(cas_mix_dig), f"Quadrada em α*={alpha_star:.4f}"),
]

for nome, sig, label in wavs:
    salvar_wav(sig, nome)

for nome, _, label in wavs:
    print(f"\n{label}")
    display(Audio(nome))
