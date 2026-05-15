"""
AlphaPhi Audio — EQ φ Completo (3 camadas) + Agente Adaptativo
Célula única para Google Colab.

Camadas EQ adaptadas para eco-φ:
  1. Bandas φ-proporcionais com β por banda  (resolução + controle)
  2. Memória com decaimento 1/φ              (inércia temporal)
  3. Modo fase: 'full' | 'amp' | 'phase'    (profundidade)

Agente adaptativo:
  Observa coerência por banda após cada cascata.
  Ajusta β por banda em direção à máxima coerência.
  Usa φ como critério E como taxa de adaptação.
  Objetivo padrão: maximizar razão 356Hz/220Hz no eco×5.
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

# ── bandas φ-proporcionais ────────────────────────────────────────────────────
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

# ── eco EQ completo (3 camadas) ───────────────────────────────────────────────
def eco_eq(x, bins_phi,
           beta_bands,        # array: β por banda
           coh_mem=None,      # array: memória de coerência por banda (camada 2)
           modo_fase='full'): # 'full' | 'amp' | 'phase'          (camada 3)
    """
    Camada 1: β independente por banda
    Camada 2: coerência efetiva = (1-1/φ)×atual + (1/φ)×memória
    Camada 3: modo_fase controla o que é modulado
      'full'  → amplitude + fase (padrão)
      'amp'   → só amplitude, fase preservada
      'phase' → só fase, amplitude preservada
    """
    N = len(x)
    F = np.fft.rfft(x)
    F_out = F.copy()
    cohs_atuais = []
    w_mem  = 1.0 / PHI          # peso da memória ≈ 0.618
    w_atual = 1.0 - w_mem       # peso do presente ≈ 0.382

    for i, (b_low, b_high, _, _) in enumerate(bins_phi):
        F_band = F[b_low:b_high]
        mag    = np.abs(F_band)
        phase  = np.angle(F_band)

        # coerência atual da banda
        an   = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        e    = -np.sum(an * np.log(an))
        coh  = float(1.0 - e / np.log(max(len(an), 2)))

        # camada 2: coerência efetiva com memória
        if coh_mem is not None:
            coh_ef = w_atual * coh + w_mem * coh_mem[i]
        else:
            coh_ef = coh
        cohs_atuais.append(coh)

        k     = K_MIN + (PHI - K_MIN) * coh_ef
        beta  = beta_bands[i]
        n_idx = np.arange(len(F_band))
        env   = np.clip(1.0 + (coh_ef * PHI**beta) * np.cos(2.0 * np.pi * n_idx / PHI), 0.05, None)

        # camada 3: modo fase
        if modo_fase == 'amp':
            F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase)
        elif modo_fase == 'phase':
            F_out[b_low:b_high] = mag * np.exp(1j * phase * k)
        else:  # 'full'
            F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase * k)

    resultado = np.fft.irfft(F_out, n=N)
    return (resultado / (np.max(np.abs(resultado)) + 1e-10),
            np.array(cohs_atuais))

# ── cascata com memória ───────────────────────────────────────────────────────
def cascata_eq(beta_bands, modo_fase='full', label="", silencioso=False):
    """Cascata N_STEPS com memória acumulada por banda."""
    cas       = [sinal_fm]
    s         = sinal_fm.copy()
    coh_mem   = np.zeros(len(BINS_PHI))  # inicia sem memória

    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, BINS_PHI, beta_bands, coh_mem, modo_fase)
        coh_mem   = cohs                 # atualiza memória
        s_e       = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()

    razao = medir_razao_par(cas[-1])
    if not silencioso:
        b_label = f"β=[{beta_bands[4]:.1f},{beta_bands[5]:.1f}]"  # bandas do par φ
        print(f"  {label:<32} modo={modo_fase}  356/220={razao:.3f}  {b_label}")
    return cas, razao

def medir_razao_par(sig):
    """Razão 356Hz/220Hz — objetivo do agente."""
    F_sig = np.fft.rfft(sig)
    b220  = round(220.0 * N_SINAL / FS)
    b356  = round(356.0 * N_SINAL / FS)
    r220  = np.abs(F_sig[b220]) / (ref[b220] + 1e-12)
    r356  = np.abs(F_sig[b356]) / (ref[b356] + 1e-12)
    return r356 / (r220 + 1e-12)

# ── agente adaptativo ─────────────────────────────────────────────────────────
def agente_phi(n_ciclos=12, lr=0.3, modo_fase='full'):
    """
    Agente que ajusta β por banda buscando máxima razão 356/220.

    A cada ciclo:
      1. Roda cascata com β atual
      2. Mede coerência por banda no eco×5
      3. Atualiza β: bandas acima da média recebem +lr×φ, abaixo -lr/φ
      4. Clippa β em [0, φ³] para manter dentro da escala φ

    A taxa de aprendizado lr também é φ-escalada:
      acima da média → impulso = lr × φ
      abaixo da média → redução = lr / φ
    """
    n_bandas  = len(BINS_PHI)
    beta      = np.ones(n_bandas)      # começa com β=1 em todas as bandas
    historico = []

    print(f"\n  {'ciclo':>5}  {'razão 356/220':>14}  {'β_banda4':>9}  {'β_banda5':>9}  {'β_med':>7}")
    print(f"  {'─'*5}  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*7}")

    for ciclo in range(n_ciclos):
        cas, razao = cascata_eq(beta, modo_fase, silencioso=True)

        # mede coerência do eco×5 por banda
        sig5 = cas[-1]
        F5   = np.fft.rfft(sig5)
        cohs = []
        for b_low, b_high, _, _ in BINS_PHI:
            F_band = F5[b_low:b_high]
            mag    = np.abs(F_band)
            an     = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
            e      = -np.sum(an * np.log(an))
            cohs.append(float(1.0 - e / np.log(max(len(an), 2))))
        cohs = np.array(cohs)

        # atualização φ-escalada
        media = cohs.mean()
        for b in range(n_bandas):
            if cohs[b] > media:
                beta[b] += lr * PHI          # impulso φ para cima
            else:
                beta[b] -= lr / PHI          # redução suave para baixo
        beta = np.clip(beta, 0.0, PHI**3)   # escala máxima = φ³

        historico.append((razao, beta.copy()))
        print(f"  {ciclo+1:>5}  {razao:>14.4f}  {beta[4]:>9.4f}  {beta[5]:>9.4f}  {beta.mean():>7.4f}")

    razao_final = historico[-1][0]
    beta_final  = historico[-1][1]
    print(f"\n  Razão inicial : {historico[0][0]:.4f}")
    print(f"  Razão final   : {razao_final:.4f}")
    print(f"  Ganho         : ×{razao_final/historico[0][0]:.2f}")
    return beta_final, historico

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
print(f"FM-φ:  f_c={F_C:.0f}Hz  f_m={F_M:.1f}Hz  β=φ={PHI:.3f}")
print(f"Bandas φ: {N_BANDAS}  |  par φ em bandas 5 (220★) e 6 (356★)\n")

# ── comparação das 3 camadas ──────────────────────────────────────────────────
print("── Comparação dos modos ─────────────────────────────────────────")
beta_uniform = np.ones(N_BANDAS)       # β=1 em todas as bandas
beta_grave   = np.array([2.0 if i < 4 else 1.0 for i in range(N_BANDAS)])
beta_par_phi = np.array([2.0 if i in (4,5) else 0.5 for i in range(N_BANDAS)])

cas_full,  _ = cascata_eq(beta_uniform, 'full',  "Uniforme  β=1  full")
cas_amp,   _ = cascata_eq(beta_uniform, 'amp',   "Uniforme  β=1  amp")
cas_phase, _ = cascata_eq(beta_uniform, 'phase', "Uniforme  β=1  phase")
cas_grave, _ = cascata_eq(beta_grave,   'full',  "Grave β=2  agudo β=1")
cas_par,   _ = cascata_eq(beta_par_phi, 'full',  "Par φ β=2  resto β=0.5")

# ── agente adaptativo ─────────────────────────────────────────────────────────
print("\n── Agente adaptativo φ ──────────────────────────────────────────")
print("  Objetivo: maximizar razão 356Hz/220Hz")
print("  Adaptação: β↑ para bandas acima da média | β↓ abaixo\n")
beta_agente, hist = agente_phi(n_ciclos=12)
cas_agente, razao_ag = cascata_eq(beta_agente, 'full', "Agente (β adaptado)")

# ── tabela de parciais — comparação final ─────────────────────────────────────
vistos, parciais = set(), []
for n in range(6):
    for f in [abs(F_C + n*F_M), abs(F_C - n*F_M)]:
        f = round(f, 1)
        if 20 < f < FS/2 and f not in vistos:
            vistos.add(f); parciais.append(f)
parciais = sorted(parciais)[:8]

def tabela(cas, titulo):
    print(f"\n{titulo}")
    print(f"{'':>6}", end="")
    for f in parciais: print(f"  {f:>6.1f}Hz", end="")
    print()
    for i, sig in enumerate(cas):
        F_sig = np.fft.rfft(sig)
        label = "orig  " if i == 0 else f"eco×{i} "
        print(label, end="")
        for f in parciais:
            b = round(f * N_SINAL / FS)
            r = np.abs(F_sig[b]) / (ref[b] + 1e-12) if b < len(F_sig) else 0.0
            print(f"  {r:>6.3f}", end="")
        print()

tabela(cas_full,   "── EQ full (amplitude+fase) ─────────────────────────")
tabela(cas_amp,    "── EQ amp  (só amplitude) ───────────────────────────")
tabela(cas_agente, "── Agente adaptativo ────────────────────────────────")

# β final do agente por banda
print("\nβ por banda após adaptação do agente:")
for i, (_, _, f_low, f_high) in enumerate(BINS_PHI):
    bar  = "█" * int(beta_agente[i] / PHI**3 * 20)
    mark = " ← 220★" if i==4 else (" ← 356★" if i==5 else "")
    print(f"  banda {i+1:>2}  {f_low:>8.1f}–{f_high:>8.1f} Hz  β={beta_agente[i]:.4f}  {bar}{mark}")

# ── áudio ─────────────────────────────────────────────────────────────────────
salvar_wav(concatenar(cas_full),          "eq_full.wav")
salvar_wav(concatenar(cas_amp),           "eq_amp.wav")
salvar_wav(concatenar(cas_phase),         "eq_phase.wav")
salvar_wav(concatenar(cas_par),           "eq_par_phi.wav")
salvar_wav(concatenar(cas_agente),        "eq_agente_desc.wav")
salvar_wav(concatenar(cas_agente[::-1]),  "eq_agente_asc.wav")

print("\n── EQ full (referência) ─────────────────────────────")
display(Audio("eq_full.wav"))
print("\n── EQ amp (só amplitude, fase preservada) ───────────")
display(Audio("eq_amp.wav"))
print("\n── EQ par φ (β=2 nas bandas 220★ e 356★) ───────────")
display(Audio("eq_par_phi.wav"))
print("\n── Agente adaptativo φ ──────────────────────────────")
print("Descendente:"); display(Audio("eq_agente_desc.wav"))
print("Ascendente:");  display(Audio("eq_agente_asc.wav"))
