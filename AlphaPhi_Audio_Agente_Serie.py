"""
AlphaPhi Audio — Agente φ com Série Geométrica Centrada
Célula única para Google Colab.

Objetivo corrigido: preservar a série φ completa, simétrica, centrada em f_c.

Série φ centrada em f_c = 220 Hz:
  f_c / φ³  =  51.9 Hz   (sub-grave)
  f_c / φ²  =  84.0 Hz
  f_c / φ¹  = 136.0 Hz   (f_m, a moduladora)
  f_c       = 220.0 Hz   ← centro — α — deve ser preservado
  f_c × φ¹  = 356.0 Hz
  f_c × φ²  = 576.0 Hz
  f_c × φ³  = 932.0 Hz   (limite superior)

Critério — média geométrica:
  Se qualquer membro vai a zero → objetivo vai a zero.
  Preserva α (tensão do centro) enquanto busca φ (coerência da série).

Comparação:
  agente_ratio  — objetivo anterior: maximiza 356/220 → destroi centro
  agente_serie  — objetivo corrigido: maximiza série completa → preserva centro
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

SERIE_PHI = [F_C / PHI**3, F_C / PHI**2, F_C / PHI,
             F_C,
             F_C * PHI,   F_C * PHI**2,  F_C * PHI**3]

NOMES_SERIE = ["f_c/φ³", "f_c/φ²", "f_c/φ (f_m)",
               "f_c ★",
               "f_c×φ",  "f_c×φ²",  "f_c×φ³"]

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

# ── eco EQ (3 camadas) ────────────────────────────────────────────────────────
def eco_eq(x, bins_phi, beta_bands, coh_mem=None, modo_fase='amp'):
    N = len(x); F = np.fft.rfft(x); F_out = F.copy()
    cohs_atuais = []; w_mem = 1.0 / PHI; w_atual = 1.0 - w_mem
    for i, (b_low, b_high, _, _) in enumerate(bins_phi):
        F_band = F[b_low:b_high]; mag = np.abs(F_band); phase = np.angle(F_band)
        an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        coh_ef = w_atual * coh + w_mem * coh_mem[i] if coh_mem is not None else coh
        cohs_atuais.append(coh)
        k = K_MIN + (PHI - K_MIN) * coh_ef; beta = beta_bands[i]
        n_idx = np.arange(len(F_band))
        env = np.clip(1.0 + (coh_ef * PHI**beta) * np.cos(2.0 * np.pi * n_idx / PHI), 0.05, None)
        if modo_fase == 'amp':
            F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase)
        elif modo_fase == 'phase':
            F_out[b_low:b_high] = mag * np.exp(1j * phase * k)
        else:
            F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase * k)
    resultado = np.fft.irfft(F_out, n=N)
    return resultado / (np.max(np.abs(resultado)) + 1e-10), np.array(cohs_atuais)

def cascata_eq(beta_bands, modo_fase='amp', silencioso=False, label=""):
    cas = [sinal_fm]; s = sinal_fm.copy(); coh_mem = np.zeros(len(BINS_PHI))
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, BINS_PHI, beta_bands, coh_mem, modo_fase)
        coh_mem = cohs; s_e = normalizar(s_e); cas.append(s_e); s = s_e.copy()
    if not silencioso:
        print(f"  {label}")
    return cas

# ── objetivos ─────────────────────────────────────────────────────────────────
def energias_serie(sig):
    """Energia relativa (ao original) de cada membro da série φ."""
    F_sig = np.fft.rfft(sig)
    out = {}
    for f, nome in zip(SERIE_PHI, NOMES_SERIE):
        b = round(f * N_SINAL / FS)
        if b < len(F_sig):
            out[f] = np.abs(F_sig[b]) / (ref[b] + 1e-12)
        else:
            out[f] = 0.0
    return out

def objetivo_ratio(sig):
    """Antigo: razão 356/220 — destroi o centro."""
    e = energias_serie(sig)
    return e[F_C * PHI] / (e[F_C] + 1e-12)

def objetivo_serie_geom(sig):
    """Novo: média geométrica da série completa — preserva o centro."""
    e = energias_serie(sig)
    vals = [max(v, 1e-10) for v in e.values()]
    return float(np.exp(np.mean(np.log(vals))))

# ── agentes ───────────────────────────────────────────────────────────────────
def agente_ratio(n_ciclos=12, lr=0.3):
    """Agente anterior — objetivo: maximizar razão 356/220."""
    n_bandas = len(BINS_PHI); beta = np.ones(n_bandas); hist = []
    print(f"  {'ciclo':>5}  {'obj(ratio)':>11}  {'220Hz':>8}  {'356Hz':>8}  {'51.9Hz':>8}")
    print(f"  {'─'*5}  {'─'*11}  {'─'*8}  {'─'*8}  {'─'*8}")
    for ciclo in range(n_ciclos):
        cas = cascata_eq(beta, 'amp', silencioso=True)
        sig5 = cas[-1]; obj = objetivo_ratio(sig5); e = energias_serie(sig5)
        F5 = np.fft.rfft(sig5)
        cohs = []
        for b_low, b_high, _, _ in BINS_PHI:
            F_band = F5[b_low:b_high]; mag = np.abs(F_band)
            an = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
            cohs.append(float(1.0 - (-np.sum(an*np.log(an))) / np.log(max(len(an),2))))
        cohs = np.array(cohs); media = cohs.mean()
        for b in range(n_bandas):
            beta[b] += lr * PHI if cohs[b] > media else -lr / PHI
        beta = np.clip(beta, 0.0, PHI**3)
        hist.append((obj, beta.copy()))
        print(f"  {ciclo+1:>5}  {obj:>11.4f}  {e[F_C]:>8.4f}  {e[F_C*PHI]:>8.4f}  {e[F_C/PHI**3]:>8.4f}")
    print(f"  Obj final: {hist[-1][0]:.4f}  (centro: {energias_serie(cas[-1])[F_C]:.4f})")
    return hist[-1][1], hist, cas

def agente_serie(n_ciclos=15, lr=0.2):
    """
    Agente corrigido — objetivo: média geométrica da série φ completa.
    Adaptação dirigida: cada banda que hospeda um membro fraco recebe β↑.
    Bandas sem membro da série: β↓ progressivo.
    """
    n_bandas = len(BINS_PHI)
    beta = np.ones(n_bandas)

    # mapa: qual banda contém cada membro da série
    banda_de = {}
    for f in SERIE_PHI:
        for i, (_, _, f_low, f_high) in enumerate(BINS_PHI):
            if f_low <= f < f_high:
                banda_de[i] = f
                break

    hist = []
    print(f"  {'ciclo':>5}  {'obj(série)':>11}  {'220★':>8}  {'356Hz':>8}  {'51.9Hz':>8}  {'β_med':>7}")
    print(f"  {'─'*5}  {'─'*11}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")

    for ciclo in range(n_ciclos):
        cas = cascata_eq(beta, 'amp', silencioso=True)
        sig5 = cas[-1]
        obj  = objetivo_serie_geom(sig5)
        e    = energias_serie(sig5)
        e_media = np.mean(list(e.values()))

        for i in range(n_bandas):
            if i in banda_de:
                f_membro = banda_de[i]
                e_membro = e.get(f_membro, 0.0)
                if e_membro < e_media:
                    beta[i] += lr * PHI      # membro fraco → amplifica
                elif e_membro > e_media * PHI:
                    beta[i] -= lr / PHI      # membro muito forte → equilibra
            else:
                beta[i] -= lr / PHI          # sem membro → libera espaço
        beta = np.clip(beta, 0.0, PHI**3)

        hist.append((obj, beta.copy()))
        print(f"  {ciclo+1:>5}  {obj:>11.4f}  {e[F_C]:>8.4f}  {e[F_C*PHI]:>8.4f}  {e[F_C/PHI**3]:>8.4f}  {beta.mean():>7.4f}")

    print(f"\n  Obj inicial : {hist[0][0]:.4f}")
    print(f"  Obj final   : {hist[-1][0]:.4f}")
    print(f"  Ganho       : ×{hist[-1][0]/hist[0][0]:.2f}")
    print(f"  Centro 220Hz: {energias_serie(cas[-1])[F_C]:.4f}  (preservado)")
    return hist[-1][1], hist, cas

# ── síntese e utilitários ─────────────────────────────────────────────────────
def gerar_fm():
    t = np.linspace(0, DURACAO, int(FS * DURACAO), endpoint=False)
    s = np.sin(2*np.pi*F_C*t + PHI * np.sin(2*np.pi*F_M*t))
    return s / np.max(np.abs(s))

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
sinal_fm = gerar_fm()
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
ref      = np.abs(np.fft.rfft(sinal_fm))
N_BANDAS = len(BINS_PHI)

print(f"FM-φ:  f_c={F_C:.0f}Hz  f_m={F_M:.1f}Hz")
print(f"\nSérie φ centrada em {F_C:.0f}Hz:")
for f, nome in zip(SERIE_PHI, NOMES_SERIE):
    centro = "  ← α (centro)" if f == F_C else ""
    print(f"  {f:>8.1f} Hz  {nome}{centro}")

# ── executar os dois agentes ──────────────────────────────────────────────────
print("\n── Agente anterior (objetivo: razão 356/220) ────────────────────")
beta_r, hist_r, cas_r = agente_ratio(n_ciclos=12)

print("\n── Agente série φ (objetivo: média geométrica da série) ─────────")
beta_s, hist_s, cas_s = agente_serie(n_ciclos=15)

# ── comparação de β por banda ─────────────────────────────────────────────────
print("\n── β por banda — comparação dos dois agentes ────────────────────")
print(f"  {'banda':>5}  {'f_low':>8}  {'f_high':>8}  {'β ratio':>9}  {'β série':>9}  conteúdo")
for i, (_, _, f_low, f_high) in enumerate(BINS_PHI):
    membros = [nome for f, nome in zip(SERIE_PHI, NOMES_SERIE) if f_low <= f < f_high]
    mark = "  " + ", ".join(membros) if membros else ""
    print(f"  {i+1:>5}  {f_low:>8.1f}  {f_high:>8.1f}  {beta_r[i]:>9.4f}  {beta_s[i]:>9.4f}{mark}")

# ── energia na série φ — comparação final ─────────────────────────────────────
print("\n── Energia na série φ — eco×5 ───────────────────────────────────")
print(f"  {'membro':>14}  {'Hz':>8}  {'agente ratio':>14}  {'agente série':>14}")
e_r = energias_serie(cas_r[-1])
e_s = energias_serie(cas_s[-1])
for f, nome in zip(SERIE_PHI, NOMES_SERIE):
    centro = " ★" if f == F_C else "  "
    print(f"  {nome:>14}  {f:>8.1f}  {e_r[f]:>14.4f}  {e_s[f]:>14.4f}{centro}")

# ── áudio ─────────────────────────────────────────────────────────────────────
salvar_wav(concatenar(cas_r),         "serie_ratio_desc.wav")
salvar_wav(concatenar(cas_r[::-1]),   "serie_ratio_asc.wav")
salvar_wav(concatenar(cas_s),         "serie_geom_desc.wav")
salvar_wav(concatenar(cas_s[::-1]),   "serie_geom_asc.wav")

print("\n── Agente ratio (centro destruído) ──────────────────────────────")
print("Descendente:"); display(Audio("serie_ratio_desc.wav"))
print("Ascendente:");  display(Audio("serie_ratio_asc.wav"))
print("\n── Agente série φ (centro preservado) ───────────────────────────")
print("Descendente:"); display(Audio("serie_geom_desc.wav"))
print("Ascendente:");  display(Audio("serie_geom_asc.wav"))
