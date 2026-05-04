"""
AlphaPhi Audio — Beep 880Hz Ergonômico (domínio 0.25×)
Célula única para Google Colab.

Descoberta (Vitor Edson, maio 2026):
  Tocar o áudio "Beep 880Hz eco α*=0.333" a 0.25× de velocidade
  revelou sensação ergonômica notória — especialmente nos pontos de
  dobra do 4º segundo em diante. Dois ritmos consonantes emergem.

Por que 0.25× funciona:
  Frequências originais → ÷4 → domínio perceptível

  Beep 880Hz      → 220Hz  (portadora do FM — consonância perfeita)
  FM portadora 220Hz → 55Hz   (Fibonacci — Lá1)
  FM moduladora 136Hz → 34Hz   (Fibonacci)
  Atrator  42Hz   → 10.5Hz (banda theta/alfa do EEG)
  Razão dos dois ritmos: 55/34 ≈ φ  ← proporção áurea audível

  Batimentos laterais do FM:
    55 − 34 = 21Hz  (Fibonacci)
    55 + 34 = 89Hz  (Fibonacci)
  Sequência completa: 21 · 34 · 55 · 89 — Fibonacci puro.

Este código gera o áudio NATIVAMENTE no domínio 0.25×:
  F_BEEP = 880/4 = 220Hz
  F_ORG  = 220/4 =  55Hz
  F_M    =  55/φ ≈  34Hz
  DURACAO = 1.5 × 4 = 6s por janela

Resultado: mesma sensação ergonômica sem alterar velocidade de reprodução.
Inclui análise dos dois ritmos e verificação da razão φ.

Arquivos gerados:
  beep880_erg_referencia.wav  — cascata 0.25× com dobras (32s)
  beep880_erg_posdobra.wav    — pós-dobra estendido (2 min)
  beep880_erg_analise.wav     — espectrograma dos dois ritmos
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

# ── constantes no domínio 0.25× ────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100

# frequências divididas por 4
F_BEEP     = 880.0 / 4        #  220Hz — Lá3 (consonante com FM)
F_ORG      = 220.0 / 4        #   55Hz — Lá1 (Fibonacci)
F_M        = F_ORG / PHI      #   ≈34Hz — (Fibonacci)
BETA_FM    = PHI              # índice de modulação

# durações × 4 (tempo esticado nativamente)
DURACAO    = 1.5 * 4          #    6s por janela
N_STEPS    = 5                # cascata de eco
ALPHA_STAR = 1.0 / 3.0        # ponto de emergência 880Hz

DURACAO_EXT   = 120.0         # 2 minutos de extensão pós-dobra
N_FADE        = int(0.5 * FS) # crossfade 0.5s
N_FADEOUT_SEG = 20
PASSO_PARTIDA = 3             # pós-dobra: passo 3 da cascata

print(f"Domínio 0.25×:")
print(f"  F_BEEP = {F_BEEP:.1f}Hz  (880/4)")
print(f"  F_ORG  = {F_ORG:.1f}Hz  (220/4 — Fibonacci)")
print(f"  F_M    = {F_M:.2f}Hz  (55/φ — Fibonacci)")
print(f"  Razão F_ORG/F_M = {F_ORG/F_M:.5f}  (φ = {PHI:.5f})")
print(f"  Batimentos: {F_ORG-F_M:.0f}Hz e {F_ORG+F_M:.0f}Hz (ambos Fibonacci)\n")

# ── bandas φ-proporcionais ─────────────────────────────────────────────────
def gerar_bandas_phi(f_min=5.0, f_max=22050.0):
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

def cascata_eq(sinal, beta_bands, bins_phi):
    cas     = [sinal]
    s       = sinal.copy()
    n_b     = len(bins_phi)
    coh_mem = np.zeros(n_b, dtype=float)
    cohs_f  = np.zeros(n_b, dtype=float)
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem   = cohs
        cohs_f    = cohs
        s_e       = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()
    return cas, cohs_f

def agente_eco(sinal, bins_phi, n_ciclos=20):
    n_b      = len(bins_phi)
    beta     = np.ones(n_b, dtype=float)
    beta_mem = beta.copy()
    w_mem    = 1.0 / PHI
    w_now    = 1.0 - w_mem
    cas_f    = None
    for _ in range(n_ciclos):
        cas, cohs  = cascata_eq(sinal, beta, bins_phi)
        coh_rel    = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo  = PHI ** (3 * coh_rel)
        beta       = w_now * beta_alvo + w_mem * beta_mem
        beta_mem   = beta.copy()
        beta       = np.clip(beta, 0.05, PHI**3)
        cas_f      = cas
    return beta, cas_f

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_beep(freq=F_BEEP, dur=DURACAO):
    t   = np.linspace(0, dur, int(FS * dur), endpoint=False)
    sig = np.sign(np.sin(2 * np.pi * freq * t))
    return normalizar(sig)

def gerar_fm(dur=DURACAO):
    t   = np.linspace(0, dur, int(FS * dur), endpoint=False)
    sig = np.sin(2 * np.pi * F_ORG * t + BETA_FM * np.sin(2 * np.pi * F_M * t))
    return normalizar(sig)

def crossfade(a, b, fade=N_FADE):
    fade  = min(fade, len(a), len(b))
    t     = np.linspace(0.0, 1.0, fade)
    saida = a.copy()
    saida[-fade:] = a[-fade:] * (1.0 - t) + b[:fade] * t
    return np.concatenate([saida, b[fade:]])

def concatenar(seq, fade=2000):
    out = seq[0].copy()
    for s in seq[1:]:
        out = crossfade(out, s, fade=fade)
    return normalizar(out)

def salvar_wav(sinal, nome):
    s16 = np.int16(np.clip(normalizar(sinal), -1.0, 1.0) * 32767)
    wavfile.write(nome, FS, s16)
    print(f"  → {nome}  ({len(sinal)/FS:.1f}s)")

# ── análise dos dois ritmos ─────────────────────────────────────────────────
def analisar_ritmos(sig, label=""):
    N    = len(sig)
    F    = np.abs(np.fft.rfft(sig))
    freq = np.fft.rfftfreq(N, d=1.0/FS)

    # picos espectrais (excluir DC)
    mask = (freq > 5.0) & (freq < 300.0)
    F_m  = F.copy()
    F_m[~mask] = 0
    idx_sorted = np.argsort(F_m)[::-1]

    picos = []
    for idx in idx_sorted:
        f_pico = freq[idx]
        if all(abs(f_pico - p) > 3.0 for p in picos):
            picos.append(f_pico)
        if len(picos) >= 4:
            break

    print(f"  {label}")
    for i, f_p in enumerate(picos[:4]):
        print(f"    pico {i+1}: {f_p:.2f}Hz")

    if len(picos) >= 2:
        r = max(picos[0], picos[1]) / min(picos[0], picos[1])
        print(f"    razão picos 1/2: {r:.5f}  (φ={PHI:.5f}  φ²={PHI**2:.5f})")
        if abs(r - PHI) < 0.05:
            print(f"    ★ Razão ≈ φ — ritmo áureo confirmado")
        elif abs(r - PHI**2) < 0.1:
            print(f"    ★ Razão ≈ φ² — oitava áurea")
    print()

# ── pipeline ───────────────────────────────────────────────────────────────
N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

print(f"Bandas φ: {N_BANDAS}  |  janela: {DURACAO}s  |  α*={ALPHA_STAR:.4f}")
print(f"\n[1/4] Gerando sinais no domínio 0.25×...")

sinal_org = gerar_fm()
sinal_dig = gerar_beep()
x_mix     = normalizar((1.0 - ALPHA_STAR) * sinal_dig + ALPHA_STAR * sinal_org)

print(f"\n[2/4] Rodando agente_eco (20 ciclos)...")
beta_final, cas_final = agente_eco(x_mix, BINS_PHI, n_ciclos=20)

idx_at    = int(np.argmax(beta_final))
f_lo_at   = BANDAS[idx_at][0]
f_hi_at   = BANDAS[idx_at][1]
f_atrator = (f_lo_at + f_hi_at) / 2
print(f"  Atrator: {f_atrator:.1f}Hz  |  β_max={beta_final[idx_at]:.3f}")

# ── referência: cascata completa com as dobras ─────────────────────────────
print(f"\n[3/4] Referência — cascata com dobras (~{DURACAO*(N_STEPS+1):.0f}s)...")
ref = concatenar(cas_final, fade=int(0.15*FS))
salvar_wav(ref, "beep880_erg_referencia.wav")

# análise dos ritmos na referência
print("  Análise espectral:")
analisar_ritmos(cas_final[0], "  passo 0 (híbrido puro):")
analisar_ritmos(cas_final[PASSO_PARTIDA], f"  passo {PASSO_PARTIDA} (pós-dobra):")
analisar_ritmos(cas_final[-1], "  passo final:")

# ── extensão pós-dobra: 2 minutos ──────────────────────────────────────────
sinal_semente = cas_final[PASSO_PARTIDA]
N_JANELAS     = int(DURACAO_EXT / DURACAO)

print(f"[4/4] Extensão pós-dobra ({DURACAO_EXT:.0f}s = {N_JANELAS} janelas de {DURACAO}s)...")

segmentos   = [sinal_semente]
sinal_atual = sinal_semente.copy()
coh_mem     = np.zeros(N_BANDAS, dtype=float)

for j in range(N_JANELAS):
    s_eco, cohs = eco_eq(sinal_atual, BINS_PHI, beta_final, coh_mem)
    coh_mem     = cohs
    s_eco       = normalizar(s_eco)
    sinal_atual = s_eco.copy()
    segmentos.append(s_eco)

    if (j + 1) % 10 == 0 or j == N_JANELAS - 1:
        print(f"  janela {j+1:3d}/{N_JANELAS}  "
              f"coh_at={cohs[idx_at]:.3f}  "
              f"coh_med={cohs.mean():.3f}")

# ── montagem ──────────────────────────────────────────────────────────────
print("\nMontando...")
audio = segmentos[0]
for seg in segmentos[1:]:
    audio = crossfade(audio, seg, fade=N_FADE)

# fade-in (5s)
n_fi = int(5 * FS)
if len(audio) > n_fi:
    audio[:n_fi] *= np.linspace(0.0, 1.0, n_fi)

# fade-out
n_fo = int(N_FADEOUT_SEG * FS)
if len(audio) > n_fo:
    audio[-n_fo:] *= np.linspace(1.0, 0.0, n_fo)

print(f"Duração total: {len(audio)/FS:.1f}s\n")
salvar_wav(audio, "beep880_erg_posdobra.wav")

# ── análise final dos ritmos no áudio estendido ────────────────────────────
print("\nAnálise dos ritmos no pós-dobra estendido:")
amostra = audio[int(10*FS):int(30*FS)]  # janela de 20s após estabilização
analisar_ritmos(amostra, "  trecho 10–30s:")

# ── playback ──────────────────────────────────────────────────────────────
print("\nPlayback — referência com dobras (~30s) — ouça as transições:")
display(Audio("beep880_erg_referencia.wav"))

print(f"\nPlayback — pós-dobra ergonômico (2 min) — atrator {f_atrator:.1f}Hz + ritmo φ:")
display(Audio("beep880_erg_posdobra.wav"))

print("\nConcluído.")
print(f"\n  F_ORG={F_ORG:.1f}Hz  F_M={F_M:.2f}Hz  razão={F_ORG/F_M:.5f}  φ={PHI:.5f}")
print(f"  Fibonacci: {int(F_ORG-F_M)}, {int(round(F_M))}, {int(F_ORG)}, {int(F_ORG+F_M)}")
