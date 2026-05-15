"""
AlphaPhi Audio — Agente Eco-Ressonante φ
Célula única para Google Colab.

Agente com mecânica idêntica ao eco-φ:

  Update rule (espelha eco_eq):
    β_alvo[i] = PHI^(3 × coh_rel[i])              — coerência → escala β
    β_new[i]  = (1-1/φ)×β_alvo[i] + (1/φ)×β_old  — memória 1/φ

Diferença fundamental dos agentes anteriores:
  - Sem objetivo externo (não mede 356/220, não mede triângulo)
  - A coerência que o eco JÁ produz é o único sinal de adaptação
  - β evolui com o mesmo decaimento que coh_mem no eco_eq
  - O triângulo é observado como resultado — não imposto como meta

O agente não sabe o que é 220Hz. Ele ressoa com o que o campo contém.
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

TRIANGULO = [F_C / PHI**3, F_C, F_C * PHI]
NOMES_T   = ["f_c/φ³  51.9Hz", "f_c ★  220Hz", "f_c×φ  356Hz"]

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

def cascata_eq(beta_bands, modo_fase='amp'):
    """Retorna cascata + coerências finais por banda (eco×N_STEPS)."""
    cas = [sinal_fm]; s = sinal_fm.copy()
    coh_mem = np.zeros(len(BINS_PHI)); cohs_final = np.zeros(len(BINS_PHI))
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, BINS_PHI, beta_bands, coh_mem, modo_fase)
        coh_mem = cohs; cohs_final = cohs
        s_e = normalizar(s_e); cas.append(s_e); s = s_e.copy()
    return cas, cohs_final

# ── observação do triângulo (leitura, sem interferência) ─────────────────────
def energias_triangulo(sig):
    F_sig = np.fft.rfft(sig)
    out = {}
    for f in TRIANGULO:
        b = round(f * N_SINAL / FS)
        out[f] = float(np.abs(F_sig[b]) / (ref[b] + 1e-12)) if b < len(F_sig) else 0.0
    return out

def objetivo_triangulo(sig):
    e = energias_triangulo(sig)
    vals = [max(v, 1e-10) for v in e.values()]
    return float(np.exp(np.mean(np.log(vals))))

# ── agente eco-ressonante ─────────────────────────────────────────────────────
def agente_eco(n_ciclos=20):
    """
    Agente cujo update rule é a mesma equação do eco-φ.

    Lê coerência por banda da cascata (o eco já produz isso).
    Mapeia coerência relativa para escala β via PHI^3.
    Atualiza β com memória 1/φ — idêntico ao eco_eq interno.

    Nenhum objetivo externo. Nenhum conhecimento de frequências-alvo.
    O triângulo φ é observado como fenômeno emergente.
    """
    n_bandas = len(BINS_PHI)
    beta     = np.ones(n_bandas)
    beta_mem = beta.copy()
    w_mem    = 1.0 / PHI          # mesmo peso do eco
    w_now    = 1.0 - w_mem

    hist = []
    print(f"  {'ciclo':>5}  {'obj':>10}  {'51.9Hz':>9}  {'220Hz★':>9}  {'356Hz':>9}  {'coh_med':>8}  {'β_med':>7}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*7}")

    for ciclo in range(n_ciclos):
        # 1. cascata — as coerências já são o sinal de adaptação
        cas, cohs = cascata_eq(beta, 'amp')
        sig5 = cas[-1]

        # 2. coerência relativa → escala β  (mesma lógica do eco para k)
        coh_min = cohs.min(); coh_max = cohs.max()
        coh_rel  = (cohs - coh_min) / (coh_max - coh_min + 1e-10)
        beta_alvo = PHI ** (3 * coh_rel)          # [PHI^0=1 , PHI^3=4.236]

        # 3. memória 1/φ — MESMA EQUAÇÃO DO eco_eq
        beta     = w_now * beta_alvo + w_mem * beta_mem
        beta_mem = beta.copy()
        beta     = np.clip(beta, 0.05, PHI**3)

        # 4. observar triângulo (leitura pura — não altera o próximo ciclo)
        e   = energias_triangulo(sig5)
        obj = objetivo_triangulo(sig5)
        hist.append((obj, beta.copy(), e.copy(), cohs.copy()))

        print(f"  {ciclo+1:>5}  {obj:>10.5f}  "
              f"{e[F_C/PHI**3]:>9.4f}  {e[F_C]:>9.4f}  {e[F_C*PHI]:>9.4f}  "
              f"{cohs.mean():>8.4f}  {beta.mean():>7.4f}")

    obj_final = hist[-1][0]; e_final = hist[-1][2]
    print(f"\n  Obj inicial : {hist[0][0]:.5f}")
    print(f"  Obj final   : {obj_final:.5f}")
    if hist[0][0] > 0:
        print(f"  Ganho       : ×{obj_final/hist[0][0]:.3f}")
    bal = min(e_final.values()) / (max(e_final.values()) + 1e-12)
    print(f"  Equilíbrio  : {bal:.4f}  (1.0=vértices iguais, 0=dominante)")

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
print(f"Bandas φ: {N_BANDAS}\n")

# ── referência β=1 ────────────────────────────────────────────────────────────
beta_ref = np.ones(N_BANDAS)
cas_ref, cohs_ref = cascata_eq(beta_ref, 'amp')
e_ref = energias_triangulo(cas_ref[-1])
obj_ref = objetivo_triangulo(cas_ref[-1])
print(f"Referência (β=1 uniforme):  obj={obj_ref:.5f}  "
      f"51.9={e_ref[F_C/PHI**3]:.3f}  220={e_ref[F_C]:.3f}  356={e_ref[F_C*PHI]:.3f}")

# ── coerências iniciais por banda ─────────────────────────────────────────────
print(f"\nCoerência inicial por banda (cascata β=1):")
for i, (_, _, f_low, f_high) in enumerate(BINS_PHI):
    bar   = "█" * int(cohs_ref[i] * 20)
    verts = [nome for f, nome in zip(TRIANGULO, NOMES_T) if f_low <= f < f_high]
    mark  = "  ← " + ", ".join(verts) if verts else ""
    print(f"  banda {i+1:>2}  {f_low:>7.1f}–{f_high:>7.1f}Hz  coh={cohs_ref[i]:.4f}  {bar}{mark}")

# ── agente eco-ressonante ─────────────────────────────────────────────────────
print(f"\n── Agente Eco-Ressonante φ ──────────────────────────────────────")
print(f"  Sem objetivo externo — β evolui pela coerência do eco\n")
beta_eco, hist_eco, cas_eco = agente_eco(n_ciclos=20)

# ── β final por banda ─────────────────────────────────────────────────────────
print(f"\nβ final por banda (agente eco-ressonante):")
cohs_finais = hist_eco[-1][3]
for i, (_, _, f_low, f_high) in enumerate(BINS_PHI):
    bar   = "█" * int(beta_eco[i] / PHI**3 * 24)
    verts = [nome for f, nome in zip(TRIANGULO, NOMES_T) if f_low <= f < f_high]
    mark  = "  ← " + ", ".join(verts) if verts else ""
    print(f"  banda {i+1:>2}  {f_low:>7.1f}–{f_high:>7.1f}Hz  "
          f"β={beta_eco[i]:.4f}  coh={cohs_finais[i]:.4f}  {bar}{mark}")

# ── perfil espectral comparado ────────────────────────────────────────────────
print(f"\n── Perfil espectral — referência × agente eco ───────────────────")
print(f"  {'freq':>8}  {'referência':>12}  {'agente_eco':>12}  {'Δ':>8}")
F_ref_s = np.fft.rfft(cas_ref[-1])
F_eco_s = np.fft.rfft(cas_eco[-1])
for f, nome in zip(TRIANGULO, NOMES_T):
    b  = round(f * N_SINAL / FS)
    r1 = np.abs(F_ref_s[b]) / (ref[b] + 1e-12)
    r2 = np.abs(F_eco_s[b]) / (ref[b] + 1e-12)
    mark = " ★" if f == F_C else "  "
    print(f"  {f:>8.1f}  {r1:>12.4f}  {r2:>12.4f}  {r2-r1:>+8.4f}  {nome}{mark}")

# ── áudio ─────────────────────────────────────────────────────────────────────
salvar_wav(concatenar(cas_ref),         "eco_ref_desc.wav")
salvar_wav(concatenar(cas_ref[::-1]),   "eco_ref_asc.wav")
salvar_wav(concatenar(cas_eco),         "eco_agente_desc.wav")
salvar_wav(concatenar(cas_eco[::-1]),   "eco_agente_asc.wav")

print("\n── Referência β=1 uniforme ──────────────────────────────────────")
print("Descendente:"); display(Audio("eco_ref_desc.wav"))
print("Ascendente:");  display(Audio("eco_ref_asc.wav"))
print("\n── Agente Eco-Ressonante φ ──────────────────────────────────────")
print("Descendente:"); display(Audio("eco_agente_desc.wav"))
print("Ascendente:");  display(Audio("eco_agente_asc.wav"))
