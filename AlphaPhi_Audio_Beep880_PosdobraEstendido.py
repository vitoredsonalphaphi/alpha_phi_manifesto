"""
AlphaPhi Audio — Beep 880Hz Pós-Dobra Estendido (2 minutos)
Célula única para Google Colab.

O experimento Beep Interface revelou 3 pontos de dobra no áudio
"Beep 880Hz eco α*=0.3333" entre os segundos 4 e 8:

  ~4s  → primeiro nulo: harmônicos agudos da quadrada colapsam
  ~6s  → segundo nulo:  880Hz residual se extingue
  ~7s  → terceiro nulo: sistema reorganiza em 42Hz (atrator)

A partir do 5º segundo, apenas o atrator de 42Hz persiste.
O 880Hz digital desapareceu — emergiu a terceira estrutura.

Problema do código anterior (Estendido 3min):
  Regenerava o beep do zero a cada janela → 880Hz re-entrava sempre.

Solução aqui:
  1. Reproduz o experimento original (agente_eco, 20 ciclos)
     para obter β_final e a cascata pós-dobra
  2. Toma a saída do passo 3 da cascata como ponto de partida
     (já após a primeira dobra, 880Hz ausente)
  3. Estende para 2 minutos alimentando a SAÍDA de cada janela
     como entrada da próxima — o 880Hz nunca re-entra
  4. β_final permanece fixo: o campo já aprendeu onde viver (42Hz)

Arquivos gerados:
  beep880_posdobra_referencia.wav   — cascata original (8s) para comparação
  beep880_posdobra_2min.wav         — extensão a partir da dobra (2 min)
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

# ── constantes (idênticas ao Beep Interface original) ─────────────────────
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
DURACAO    = 1.5
N_STEPS    = 5
ALPHA_STAR = 1.0 / 3.0
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI
ALPHA_F    = 1.0 / 137.035999084

DURACAO_EXT     = 120.0   # 2 minutos
N_FADE          = 4410    # crossfade 0.1s
N_FADEOUT_SEG   = 15      # fade-out final

# ── bandas φ-proporcionais (idêntico ao original) ─────────────────────────
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
    result = []
    for f_lo, f_hi in bandas:
        b_lo = int(f_lo / (FS / n))
        b_hi = int(f_hi / (FS / n)) + 1
        b_lo = max(0, b_lo)
        b_hi = min(b_hi, n // 2 + 1)
        result.append((b_lo, b_hi, f_lo, f_hi))
    return result

# ── eco_eq (idêntico ao Beep Interface original) ──────────────────────────
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
            1.0 + (coh_ef * PHI ** beta_i) * np.cos(2.0 * np.pi * n_idx / PHI),
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
    for ciclo in range(n_ciclos):
        cas, cohs  = cascata_eq(sinal_entrada, beta, bins_phi)
        coh_rel    = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo  = PHI ** (3 * coh_rel)
        beta       = w_now * beta_alvo + w_mem * beta_mem
        beta_mem   = beta.copy()
        beta       = np.clip(beta, 0.05, PHI**3)
        cas_final  = cas
    return beta, cas_final

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_beep(freq, duracao=DURACAO):
    t   = np.linspace(0, duracao, int(FS * duracao), endpoint=False)
    sig = np.sign(np.sin(2 * np.pi * freq * t))
    return normalizar(sig)

def gerar_fm_phi(duracao=DURACAO):
    t   = np.linspace(0, duracao, int(FS * duracao), endpoint=False)
    sig = np.sin(2 * np.pi * F_ORG * t + PHI * np.sin(2 * np.pi * F_M * t))
    return normalizar(sig)

def crossfade(a, b, fade=N_FADE):
    fade  = min(fade, len(a), len(b))
    t     = np.linspace(0, 1, fade)
    saida = a.copy()
    saida[-fade:] = a[-fade:] * (1 - t) + b[:fade] * t
    return np.concatenate([saida, b[fade:]])

def concatenar(seq):
    out = seq[0].copy()
    for sig in seq[1:]:
        out = crossfade(out, sig, fade=2000)
    return normalizar(out)

def salvar_wav(sinal, nome):
    s16 = np.int16(np.clip(normalizar(sinal), -1.0, 1.0) * 32767)
    wavfile.write(nome, FS, s16)
    print(f"  → {nome}  ({len(sinal)/FS:.1f}s)")

# ── passo 1: reproduzir experimento original ──────────────────────────────
N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

print(f"Bandas φ: {N_BANDAS}  |  α* = {ALPHA_STAR:.5f}  |  Beep {F_BEEP:.0f}Hz")
print("\n[1/3] Rodando agente_eco original (20 ciclos) para obter β_final...")

sinal_org = gerar_fm_phi()
sinal_dig = gerar_beep(F_BEEP)
x_mix     = normalizar((1.0 - ALPHA_STAR) * sinal_dig + ALPHA_STAR * sinal_org)

beta_final, cas_final = agente_eco(x_mix, BINS_PHI, n_ciclos=20)

# ── identificar banda do atrator ──────────────────────────────────────────
idx_atrator = int(np.argmax(beta_final))
f_lo_at, f_hi_at = BANDAS[idx_atrator]
f_atrator = (f_lo_at + f_hi_at) / 2
print(f"  β_final máximo: {beta_final[idx_atrator]:.3f}  →  atrator: {f_atrator:.0f}Hz")
print(f"  β_final mínimo: {beta_final.min():.3f}")

# ── passo 2: gerar referência (cascata original completa) ─────────────────
print("\n[2/3] Gerando referência (cascata completa — 8s com as dobras)...")
ref_audio = concatenar(cas_final)
salvar_wav(ref_audio, "beep880_posdobra_referencia.wav")

# ── passo 3: extensão a partir da primeira dobra ──────────────────────────
# cas_final = [passo0, passo1, passo2, passo3, passo4, passo5]
# A primeira dobra ocorre em ~passo 2-3. Partimos do passo 3.
PASSO_PARTIDA = 3
sinal_semente = cas_final[PASSO_PARTIDA]

print(f"\n[3/3] Estendendo a partir do passo {PASSO_PARTIDA} (pós-dobra) por {DURACAO_EXT:.0f}s...")

N_JANELAS = int(DURACAO_EXT / DURACAO)
segmentos  = [sinal_semente]
sinal_atual = sinal_semente.copy()

# β_final fixo — o campo já aprendeu; não re-adaptamos
# Rodamos apenas 1 passo de eco por janela (não 5) para evitar over-processing
# e preservar a textura da dobra
n_bandas  = len(BINS_PHI)
coh_mem   = np.zeros(n_bandas, dtype=float)

for j in range(N_JANELAS):
    # 1 passo de eco com β fixo pós-adaptação
    s_eco, cohs = eco_eq(sinal_atual, BINS_PHI, beta_final, coh_mem)
    coh_mem     = cohs
    s_eco       = normalizar(s_eco)

    # a saída vira a nova entrada — 880Hz nunca re-entra
    sinal_atual = s_eco.copy()
    segmentos.append(s_eco)

    if (j + 1) % 20 == 0 or j == N_JANELAS - 1:
        beta_max_idx = int(np.argmax(beta_final))
        f_lo_dbg, _  = BANDAS[beta_max_idx]
        print(f"  janela {j+1:3d}/{N_JANELAS}  coh_atrator={cohs[beta_max_idx]:.3f}  "
              f"coh_med={cohs.mean():.3f}  atrator={f_lo_dbg:.0f}Hz")

# ── montagem ──────────────────────────────────────────────────────────────
print("\nMontando...")
audio = segmentos[0]
for seg in segmentos[1:]:
    audio = crossfade(audio, seg, fade=N_FADE)

# fade-in suave (primeiros 3s)
n_fi = int(3 * FS)
if len(audio) > n_fi:
    audio[:n_fi] *= np.linspace(0.0, 1.0, n_fi)

# fade-out final
n_fo = int(N_FADEOUT_SEG * FS)
if len(audio) > n_fo:
    audio[-n_fo:] *= np.linspace(1.0, 0.0, n_fo)

print(f"Duração total: {len(audio)/FS:.1f}s\n")
salvar_wav(audio, "beep880_posdobra_2min.wav")

# ── playback ──────────────────────────────────────────────────────────────
print("\nPlayback — cascata original (8s) — ouça as 3 dobras:")
display(Audio("beep880_posdobra_referencia.wav"))

print(f"\nPlayback — pós-dobra estendido (2 min) — apenas o atrator {f_atrator:.0f}Hz:")
display(Audio("beep880_posdobra_2min.wav"))

print("\nConcluído.")
