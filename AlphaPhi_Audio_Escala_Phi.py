"""
AlphaPhi Audio — Escala φ de Extensão Espectral
Célula única para Google Colab.

Progressão do concentrado ao estendido via índice de modulação FM:

  β_FM = 0      →  senoide pura         (entropia mínima — concentrado)
  β_FM = 1/φ    →  FM sub-φ            (levemente estendido)
  β_FM = 1      →  FM neutro
  β_FM = φ      →  FM-φ do projeto  ★  (ponto de equilíbrio natural)
  β_FM = φ²     →  FM estendido
  β_FM = φ³     →  FM máximo            (orgânico complexo)
  + onda quadrada                        (referência digital)

Para cada sinal:
  1. Mede entropia espectral inicial
  2. Aplica agente eco-φ (20 ciclos, sem objetivo externo)
  3. Revela o atrator φ: qual banda recebeu β máximo
  4. Mostra direção de convergência: grave ou agudo

A entropia é a régua da progressão concentrado → orgânico.
O eco-φ revela onde a coerência vive em cada nível de extensão.
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

PHI     = (1 + np.sqrt(5)) / 2
K_MIN   = np.sqrt(2)
FS      = 44100
DURACAO = 1.5
N_STEPS = 5
F_C     = 220.0
F_M     = F_C / PHI

# ── bandas e bins ─────────────────────────────────────────────────────────────
def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max: break
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
    N = len(x); F = np.fft.rfft(x); F_out = F.copy()
    cohs = []; w_mem = 1.0 / PHI; w_now = 1.0 - w_mem
    for i, (b_low, b_high, _, _) in enumerate(bins_phi):
        F_band = F[b_low:b_high]; mag = np.abs(F_band); phase = np.angle(F_band)
        an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        coh_ef = w_now * coh + w_mem * coh_mem[i] if coh_mem is not None else coh
        cohs.append(coh)
        n_idx = np.arange(len(F_band))
        env = np.clip(1.0 + (coh_ef * PHI**beta_bands[i]) * np.cos(2.0 * np.pi * n_idx / PHI), 0.05, None)
        F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase)
    resultado = np.fft.irfft(F_out, n=N)
    return resultado / (np.max(np.abs(resultado)) + 1e-10), np.array(cohs)

def cascata_eq(sinal_entrada, beta_bands):
    cas = [sinal_entrada]; s = sinal_entrada.copy()
    coh_mem = np.zeros(len(BINS_PHI)); cohs_final = np.zeros(len(BINS_PHI))
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, BINS_PHI, beta_bands, coh_mem)
        coh_mem = cohs; cohs_final = cohs
        s_e = normalizar(s_e); cas.append(s_e); s = s_e.copy()
    return cas, cohs_final

# ── métricas ──────────────────────────────────────────────────────────────────
def entropia_espectral(sig):
    F_sig = np.abs(np.fft.rfft(sig))
    F_sig = np.clip(F_sig / (F_sig.sum() + 1e-10), 1e-10, 1.0)
    return float(-np.sum(F_sig * np.log(F_sig)))

def suavidade(sig):
    return float(np.mean(np.abs(np.diff(sig))))

# ── agente eco-ressonante ─────────────────────────────────────────────────────
def agente_eco(sinal_entrada, n_ciclos=20):
    n_bandas = len(BINS_PHI)
    beta = np.ones(n_bandas); beta_mem = beta.copy()
    w_mem = 1.0 / PHI; w_now = 1.0 - w_mem
    cas_final = None
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal_entrada, beta)
        coh_rel  = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo = PHI ** (3 * coh_rel)
        beta     = w_now * beta_alvo + w_mem * beta_mem
        beta_mem = beta.copy()
        beta     = np.clip(beta, 0.05, PHI**3)
        cas_final = cas
    return beta, cas_final

# ── síntese ───────────────────────────────────────────────────────────────────
def gerar_fm(beta_mod):
    """FM com índice de modulação variável."""
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    s = np.sin(2*np.pi*F_C*t + beta_mod * np.sin(2*np.pi*F_M*t))
    return normalizar(s)

def gerar_quadrada():
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    return normalizar(np.sign(np.sin(2 * np.pi * F_C * t)))

def normalizar(s):
    m = np.max(np.abs(s)); return s / m if m > 1e-12 else s

def crossfade(a, b, fade=2000):
    t = np.linspace(0, 1, fade)
    return np.concatenate([a[:-fade], a[-fade:]*(1-t) + b[:fade]*t, b[fade:]])

def concatenar(seq):
    out = seq[0].copy()
    for sig in seq[1:]: out = crossfade(out, sig)
    return normalizar(out)

def salvar_wav(s, nome):
    wavfile.write(nome, FS, np.int16(np.clip(s, -1.0, 1.0) * 32767))

# ── configuração ──────────────────────────────────────────────────────────────
N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

# escala de modulação: 7 pontos φ-proporcionais + quadrada
ESCALA = [
    (0,        "senoide    β=0  "),
    (1/PHI,    "FM  β=1/φ       "),
    (1,        "FM  β=1         "),
    (PHI,      "FM  β=φ   ★     "),
    (PHI**2,   "FM  β=φ²        "),
    (PHI**3,   "FM  β=φ³        "),
    (None,     "quadrada        "),
]

print(f"FM-φ:  f_c={F_C:.0f}Hz  f_m={F_M:.1f}Hz  |  Bandas φ: {N_BANDAS}")
print(f"\nEscala φ de extensão espectral")
print(f"  {'sinal':<20}  {'β_FM':>6}  {'entropia':>9}  {'suavidade':>10}")
print(f"  {'─'*20}  {'─'*6}  {'─'*9}  {'─'*10}")

sinais = []
for beta_mod, label in ESCALA:
    sig = gerar_fm(beta_mod) if beta_mod is not None else gerar_quadrada()
    ent = entropia_espectral(sig)
    suav = suavidade(sig)
    beta_str = f"{beta_mod:.3f}" if beta_mod is not None else "—"
    print(f"  {label:<20}  {beta_str:>6}  {ent:>9.4f}  {suav:>10.6f}")
    sinais.append((sig, label.strip(), beta_mod))

# ── agente eco em cada nível da escala ───────────────────────────────────────
print(f"\n── Atratores φ por nível da escala ──────────────────────────────")
print(f"  {'sinal':<20}  {'ent_ini':>8}  {'ent_fin':>8}  {'Δent':>7}  {'β_max banda':>12}  {'f_atrator':>10}  direção")
print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*10}  {'─'*10}")

resultados = []
for sig, label, beta_mod in sinais:
    ent_ini = entropia_espectral(sig)
    beta_final, cas = agente_eco(sig, n_ciclos=20)

    sig_eco = cas[-1]
    ent_fin = entropia_espectral(sig_eco)

    # atrator: banda com β máximo
    i_max = int(np.argmax(beta_final))
    _, _, f_low_max, f_high_max = BINS_PHI[i_max]
    f_atrator = (f_low_max + f_high_max) / 2

    # direção: grave (<400Hz) ou agudo (>400Hz)
    direcao = "GRAVE" if f_atrator < 400 else "AGUDO"

    delta = ent_fin - ent_ini
    print(f"  {label:<20}  {ent_ini:>8.4f}  {ent_fin:>8.4f}  {delta:>+7.4f}  "
          f"banda {i_max+1:>2} ({f_low_max:.0f}–{f_high_max:.0f}Hz)  "
          f"{f_atrator:>8.0f}Hz  {direcao}")

    resultados.append((sig, label, beta_final, cas, f_atrator, direcao))

# ── β por banda — comparação da escala completa ───────────────────────────────
print(f"\n── β final por banda — escala completa ──────────────────────────")
header = f"  {'banda':>5}  {'f_low':>7}  {'f_high':>7}"
for _, label, _, _, _, _ in resultados:
    header += f"  {label[:8]:>8}"
print(header)
for i, (_, _, f_low, f_high) in enumerate(BINS_PHI):
    linha = f"  {i+1:>5}  {f_low:>7.1f}  {f_high:>7.1f}"
    for _, _, beta_final, _, _, _ in resultados:
        linha += f"  {beta_final[i]:>8.4f}"
    print(linha)

# ── linha de limiar grave/agudo ───────────────────────────────────────────────
print(f"\n── Limiar grave/agudo na escala φ ───────────────────────────────")
print(f"  β_FM cresce → atrator migra de GRAVE para AGUDO")
print(f"  O ponto φ (β=φ) é o equilíbrio — FM-φ do projeto")
for _, label, _, _, f_atrator, direcao in resultados:
    barra = "▓" * int(f_atrator / 400)
    print(f"  {label:<20}  {f_atrator:>7.0f}Hz  {direcao:<6}  {barra}")

# ── áudio da progressão ───────────────────────────────────────────────────────
wavs = []
for sig, label, _, cas, _, _ in resultados:
    nome_orig = f"escala_{label[:8].strip().replace(' ','_')}_orig.wav"
    nome_eco  = f"escala_{label[:8].strip().replace(' ','_')}_eco.wav"
    salvar_wav(sig,             nome_orig)
    salvar_wav(concatenar(cas), nome_eco)
    wavs.append((nome_orig, nome_eco, label))

print(f"\n── Progressão sonora: concentrado → estendido ───────────────────")
for nome_orig, nome_eco, label in wavs:
    print(f"\n{label.strip()}")
    print(f"  Original:"); display(Audio(nome_orig))
    print(f"  Eco-φ:");    display(Audio(nome_eco))
