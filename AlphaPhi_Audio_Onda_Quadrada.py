"""
AlphaPhi Audio — Agente Eco-Ressonante: Onda Quadrada → Onda Orgânica
Célula única para Google Colab.

Questão central:
  O agente eco-φ, sem objetivo externo, transforma uma onda quadrada
  (artificial, digital) em algo mais orgânico?

O agente não recebe instrução de "suavizar".
Ele lê a coerência φ de cada banda e ajusta β com memória 1/φ.
O que emerge é a "interpretação φ" da onda quadrada.

Métricas de disparidade (quadrada vs orgânica):
  entropia espectral — quão dispersa está a energia (alta=quadrada)
  suavidade          — derivada média (alta=quadrada, transições bruscas)
  THD                — distorção harmônica total (alta=quadrada)
  coerência φ        — média das coerências por banda (alta=orgânica)

Comparação com 3 sinais de entrada:
  onda quadrada  — artificial, digital
  FM-φ           — orgânica, referência do projeto
  senoide pura   — orgânica, mínima complexidade
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
def eco_eq(x, bins_phi, beta_bands, coh_mem=None, modo_fase='amp'):
    N = len(x); F = np.fft.rfft(x); F_out = F.copy()
    cohs_atuais = []; w_mem = 1.0 / PHI; w_atual = 1.0 - w_mem
    for i, (b_low, b_high, _, _) in enumerate(bins_phi):
        F_band = F[b_low:b_high]; mag = np.abs(F_band); phase = np.angle(F_band)
        an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        coh_ef = w_atual * coh + w_mem * coh_mem[i] if coh_mem is not None else coh
        cohs_atuais.append(coh)
        k = K_MIN + (PHI - K_MIN) * coh_ef
        n_idx = np.arange(len(F_band))
        env = np.clip(1.0 + (coh_ef * PHI**beta_bands[i]) * np.cos(2.0 * np.pi * n_idx / PHI), 0.05, None)
        if modo_fase == 'amp':
            F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase)
        else:
            F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase * k)
    resultado = np.fft.irfft(F_out, n=N)
    return resultado / (np.max(np.abs(resultado)) + 1e-10), np.array(cohs_atuais)

def cascata_eq(sinal_entrada, beta_bands, modo_fase='amp'):
    """Cascata a partir de qualquer sinal de entrada; retorna cascata + coerências finais."""
    cas = [sinal_entrada]; s = sinal_entrada.copy()
    coh_mem = np.zeros(len(BINS_PHI)); cohs_final = np.zeros(len(BINS_PHI))
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, BINS_PHI, beta_bands, coh_mem, modo_fase)
        coh_mem = cohs; cohs_final = cohs
        s_e = normalizar(s_e); cas.append(s_e); s = s_e.copy()
    return cas, cohs_final

# ── métricas de disparidade ───────────────────────────────────────────────────
def entropia_espectral(sig):
    """Alta para espectro disperso (quadrada); baixa para concentrado (orgânica)."""
    F_sig = np.abs(np.fft.rfft(sig))
    F_sig = np.clip(F_sig / (F_sig.sum() + 1e-10), 1e-10, 1.0)
    return float(-np.sum(F_sig * np.log(F_sig)))

def suavidade(sig):
    """Derivada média: alta = transições bruscas (quadrada); baixa = suave (orgânica)."""
    return float(np.mean(np.abs(np.diff(sig))))

def thd(sig, f0=F_C):
    """Distorção harmônica total: energia nos harmônicos / fundamental."""
    F_sig = np.abs(np.fft.rfft(sig))
    b0 = round(f0 * N_SINAL / FS)
    e0 = F_sig[b0] ** 2
    e_harm = sum(F_sig[round(n * f0 * N_SINAL / FS)] ** 2
                 for n in range(2, 20)
                 if round(n * f0 * N_SINAL / FS) < len(F_sig))
    return float(np.sqrt(e_harm / (e0 + 1e-12)))

def metricas(sig):
    return entropia_espectral(sig), suavidade(sig), thd(sig)

# ── agente eco-ressonante para qualquer onda ──────────────────────────────────
def agente_eco_onda(sinal_entrada, n_ciclos=20, label=""):
    """
    Mesma mecânica do agente eco-ressonante, aplicada a qualquer onda de entrada.
    β evolui por coerência do eco; sem objetivo externo.
    """
    n_bandas = len(BINS_PHI)
    beta     = np.ones(n_bandas)
    beta_mem = beta.copy()
    w_mem    = 1.0 / PHI
    w_now    = 1.0 - w_mem

    hist = []
    print(f"\n── {label} ─────────────────────────────────────────")
    print(f"  {'ciclo':>5}  {'entropia':>9}  {'suavidade':>10}  {'THD':>8}  {'coh_med':>8}  {'β_med':>7}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}")

    cas_final = None
    for ciclo in range(n_ciclos):
        cas, cohs = cascata_eq(sinal_entrada, beta, 'amp')
        sig5 = cas[-1]

        coh_min  = cohs.min(); coh_max = cohs.max()
        coh_rel  = (cohs - coh_min) / (coh_max - coh_min + 1e-10)
        beta_alvo = PHI ** (3 * coh_rel)

        beta     = w_now * beta_alvo + w_mem * beta_mem
        beta_mem = beta.copy()
        beta     = np.clip(beta, 0.05, PHI**3)

        ent, suav, thd_val = metricas(sig5)
        hist.append((ent, suav, thd_val, beta.copy(), cohs.copy()))
        cas_final = cas

        print(f"  {ciclo+1:>5}  {ent:>9.4f}  {suav:>10.6f}  {thd_val:>8.4f}  "
              f"{cohs.mean():>8.4f}  {beta.mean():>7.4f}")

    ent0, suav0, thd0 = metricas(sinal_entrada)
    ent_f, suav_f, thd_f = hist[-1][0], hist[-1][1], hist[-1][2]
    print(f"\n  Entropia  : {ent0:.4f} → {ent_f:.4f}  ({ent_f-ent0:+.4f})")
    print(f"  Suavidade : {suav0:.6f} → {suav_f:.6f}  ({suav_f-suav0:+.6f})")
    print(f"  THD       : {thd0:.4f} → {thd_f:.4f}  ({thd_f-thd0:+.4f})")
    print(f"  Coerência : {hist[-1][4].mean():.4f}")

    return hist[-1][3], hist, cas_final

# ── síntese ───────────────────────────────────────────────────────────────────
def gerar_quadrada():
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    return normalizar(np.sign(np.sin(2 * np.pi * F_C * t)))

def gerar_fm():
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    s = np.sin(2*np.pi*F_C*t + PHI * np.sin(2*np.pi*F_M*t))
    return normalizar(s)

def gerar_senoide():
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    return normalizar(np.sin(2 * np.pi * F_C * t))

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

sinal_quad = gerar_quadrada()
sinal_fm   = gerar_fm()
sinal_sen  = gerar_senoide()

print(f"FM-φ:  f_c={F_C:.0f}Hz  f_m={F_M:.1f}Hz  |  Bandas φ: {N_BANDAS}")
print(f"\nMétricas iniciais:")
print(f"  {'sinal':>12}  {'entropia':>9}  {'suavidade':>10}  {'THD':>8}")
for sig, nome in [(sinal_quad,"quadrada"), (sinal_fm,"FM-φ"), (sinal_sen,"senoide")]:
    e, s, t = metricas(sig)
    print(f"  {nome:>12}  {e:>9.4f}  {s:>10.6f}  {t:>8.4f}")

# ── agente eco aplicado às três ondas ─────────────────────────────────────────
beta_q, hist_q, cas_q = agente_eco_onda(sinal_quad, n_ciclos=20, label="Onda quadrada → eco-φ")
beta_f, hist_f, cas_f = agente_eco_onda(sinal_fm,   n_ciclos=20, label="FM-φ → eco-φ (referência orgânica)")
beta_s, hist_s, cas_s = agente_eco_onda(sinal_sen,  n_ciclos=20, label="Senoide → eco-φ (referência mínima)")

# ── β final por banda — comparação ────────────────────────────────────────────
print(f"\n── β final por banda ─────────────────────────────────────────────")
print(f"  {'banda':>5}  {'f_low':>8}  {'f_high':>8}  {'β quad':>8}  {'β FM-φ':>8}  {'β sen':>8}")
for i, (_, _, f_low, f_high) in enumerate(BINS_PHI):
    print(f"  {i+1:>5}  {f_low:>8.1f}  {f_high:>8.1f}  "
          f"{beta_q[i]:>8.4f}  {beta_f[i]:>8.4f}  {beta_s[i]:>8.4f}")

# ── resumo de convergência ────────────────────────────────────────────────────
print(f"\n── Convergência das métricas — entrada vs eco×5 ciclo 20 ─────────")
print(f"  {'sinal':>12}  {'ΔEntrop':>9}  {'ΔSuavid':>10}  {'ΔTHD':>8}  direção")
for hist, nome in [(hist_q,"quadrada"), (hist_f,"FM-φ"), (hist_s,"senoide")]:
    de = hist[-1][0] - hist[0][0]
    ds = hist[-1][1] - hist[0][1]
    dt = hist[-1][2] - hist[0][2]
    direcao = "→ orgânica" if de < 0 and ds < 0 and dt < 0 else \
              "→ mais quadrada" if de > 0 and ds > 0 else "→ mista"
    print(f"  {nome:>12}  {de:>+9.4f}  {ds:>+10.6f}  {dt:>+8.4f}  {direcao}")

# ── áudio ─────────────────────────────────────────────────────────────────────
salvar_wav(sinal_quad,                   "onda_quad_original.wav")
salvar_wav(concatenar(cas_q),            "onda_quad_eco_desc.wav")
salvar_wav(concatenar(cas_q[::-1]),      "onda_quad_eco_asc.wav")
salvar_wav(sinal_fm,                     "onda_fm_original.wav")
salvar_wav(concatenar(cas_f),            "onda_fm_eco_desc.wav")
salvar_wav(sinal_sen,                    "onda_sen_original.wav")
salvar_wav(concatenar(cas_s),            "onda_sen_eco_desc.wav")

print("\n── Onda quadrada original ───────────────────────────────────────")
display(Audio("onda_quad_original.wav"))
print("\n── Onda quadrada → eco-φ (descendente) ─────────────────────────")
display(Audio("onda_quad_eco_desc.wav"))
print("\n── Onda quadrada → eco-φ (ascendente) ──────────────────────────")
display(Audio("onda_quad_eco_asc.wav"))
print("\n── FM-φ original (referência orgânica) ─────────────────────────")
display(Audio("onda_fm_original.wav"))
print("\n── FM-φ → eco-φ ─────────────────────────────────────────────────")
display(Audio("onda_fm_eco_desc.wav"))
print("\n── Senoide original ─────────────────────────────────────────────")
display(Audio("onda_sen_original.wav"))
print("\n── Senoide → eco-φ ──────────────────────────────────────────────")
display(Audio("onda_sen_eco_desc.wav"))
