"""
AlphaPhi Audio — Beep 880Hz Eco α*=0.333 Estendido (3 minutos)
Célula única para Google Colab.

Ponto de emergência identificado:
  α* = 1/3 ≈ 0.333  →  Δentropia máximo (+5.4258)
  Atrator: 42Hz (GRAVE) — energia migra para sub-grave ao longo do tempo
  Entropia híbrido α*: 2.0166  (mais organizado que qualquer componente puro)

Estrutura temporal:
  0:00–0:20  intro — híbrido puro sem eco (beep+FM-φ fundidos)
  0:20–2:40  corpo — agente eco-φ com estado persistido entre janelas
             β por banda evolui com memória 1/φ → atrator 42Hz emerge
  2:40–3:00  fade-out suave

Arquivos gerados:
  beep880_eco_alpha333_3min.wav   — áudio principal (3 min)
  beep880_puro_15s.wav            — referência: híbrido sem eco (15s)
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display

# ── constantes ─────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
K_MIN      = np.sqrt(2)
FS         = 44100

ALPHA_STAR = 1.0 / 3.0        # ponto de emergência para 880Hz
F_BEEP     = 880.0             # beep de interface
F_ORG      = 220.0             # FM-φ: portadora
F_M        = F_ORG / PHI       # FM-φ: moduladora ≈ 135.9Hz
BETA_FM    = PHI               # FM-φ: índice de modulação

DURACAO_TOTAL   = 180.0        # segundos (3 minutos)
JANELA_SEG      = 1.5          # cada janela = 1.5s
N_STEPS_ECO     = 5            # cascata de eco por janela
N_FADE          = 4410         # crossfade entre janelas (0.1s)
N_INTRO_JANELAS = 9            # ~13.5s de intro sem eco
N_FADEOUT_SEG   = 20           # fade-out final em segundos

# ── bandas φ-proporcionais ─────────────────────────────────────────────────
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
        result.append((b_lo, min(b_hi, n // 2)))
    return result

# ── eco_eq: núcleo do processamento por banda ───────────────────────────────
def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    n = len(x)
    F   = np.fft.fft(x)
    amp = np.abs(F)
    ang = np.angle(F)

    amp_out  = amp.copy()
    new_coh  = []

    for i, (b_lo, b_hi) in enumerate(bins_phi):
        if b_lo >= b_hi:
            new_coh.append(coh_mem[i] if coh_mem else 0.0)
            continue
        a_b  = amp[b_lo:b_hi]
        s    = a_b.sum()
        if s < 1e-12:
            new_coh.append(0.0)
            continue
        a_bn = np.clip(a_b / s, 1e-10, 1.0)
        e_b  = -np.sum(a_bn * np.log(a_bn))
        coh_b = float(1.0 - e_b / np.log(max(len(a_b), 2)))
        if coh_mem:
            coh_b = (1 - 1/PHI) * coh_b + (1/PHI) * coh_mem[i]
        new_coh.append(coh_b)

        env = PHI ** (beta_bands[i] * coh_b)
        amp_out[b_lo:b_hi]     *= env
        amp_out[n-b_hi:n-b_lo] *= env

    F_out = amp_out * np.exp(1j * ang)
    return np.real(np.fft.ifft(F_out)), new_coh

def cascata_eq(x, bins_phi, beta_bands, n_steps=N_STEPS_ECO):
    s, coh_mem = x.copy(), None
    for _ in range(n_steps):
        s, coh_mem = eco_eq(s, bins_phi, beta_bands, coh_mem)
    return s, coh_mem

# ── agente: update rule espelhando o eco (sem objetivo externo) ─────────────
def agente_update(beta_bands, bins_phi, sig):
    F   = np.fft.fft(sig)
    amp = np.abs(F)
    cohs = []
    for b_lo, b_hi in bins_phi:
        if b_lo >= b_hi:
            cohs.append(0.0)
            continue
        a_b = amp[b_lo:b_hi]
        s   = a_b.sum()
        if s < 1e-12:
            cohs.append(0.0)
            continue
        a_bn = np.clip(a_b / s, 1e-10, 1.0)
        e_b  = -np.sum(a_bn * np.log(a_bn))
        cohs.append(float(1.0 - e_b / np.log(max(len(a_b), 2))))

    coh_max = max(cohs) if max(cohs) > 1e-9 else 1.0
    new_beta = []
    for coh, b_old in zip(cohs, beta_bands):
        coh_rel = coh / coh_max
        b_alvo  = PHI ** (3 * coh_rel)
        b_new   = (1 - 1/PHI) * b_alvo + (1/PHI) * b_old
        new_beta.append(b_new)
    return new_beta

# ── geradores de sinal base ─────────────────────────────────────────────────
def gerar_beep(duracao):
    t = np.linspace(0, duracao, int(FS * duracao), endpoint=False)
    return np.sin(2 * np.pi * F_BEEP * t)

def gerar_fm(duracao):
    t   = np.linspace(0, duracao, int(FS * duracao), endpoint=False)
    mod = np.sin(2 * np.pi * F_M * t)
    return np.sin(2 * np.pi * F_ORG * t + BETA_FM * mod)

def normalizar(s, headroom=0.92):
    m = np.max(np.abs(s))
    return s / m * headroom if m > 1e-9 else s

def crossfade(a, b, fade=N_FADE):
    fade = min(fade, len(a), len(b))
    env_out = np.linspace(1.0, 0.0, fade)
    env_in  = np.linspace(0.0, 1.0, fade)
    saida   = a.copy()
    saida[-fade:] = a[-fade:] * env_out + b[:fade] * env_in
    return np.concatenate([saida, b[fade:]])

def salvar_wav(sinal, nome):
    s16 = np.int16(normalizar(sinal) * 32767)
    wavfile.write(nome, FS, s16)
    print(f"  → {nome}  ({len(sinal)/FS:.1f}s)")

# ── pipeline principal ──────────────────────────────────────────────────────
N_JANELAS  = int(DURACAO_TOTAL / JANELA_SEG)
n_janela   = int(FS * JANELA_SEG)

bandas   = gerar_bandas_phi()
bins_phi = bandas_para_bins(bandas, n_janela)
n_bandas = len(bandas)

print(f"Bandas φ: {n_bandas}")
print(f"α* = {ALPHA_STAR:.5f}  |  Beep {F_BEEP:.0f}Hz  |  FM-φ {F_ORG:.0f}Hz")
print(f"Janelas: {N_JANELAS} × {JANELA_SEG}s = {N_JANELAS * JANELA_SEG:.0f}s\n")

# estado do agente — evolui continuamente entre janelas
beta_bands = [1.0] * n_bandas   # β inicial uniforme
segmentos  = []

for j in range(N_JANELAS):
    # gera janela de sinal base
    beep_j = gerar_beep(JANELA_SEG)
    fm_j   = gerar_fm(JANELA_SEG)

    # híbrido no ponto de emergência α*=1/3
    x_mix = (1.0 - ALPHA_STAR) * beep_j + ALPHA_STAR * fm_j

    if j < N_INTRO_JANELAS:
        # intro: híbrido puro sem eco
        seg = x_mix
    else:
        # corpo: eco-φ com estado persistido
        seg, _ = cascata_eq(x_mix, bins_phi, beta_bands, n_steps=N_STEPS_ECO)
        beta_bands = agente_update(beta_bands, bins_phi, seg)

    segmentos.append(normalizar(seg))

    if (j + 1) % 15 == 0 or j == N_JANELAS - 1:
        beta_max = max(beta_bands)
        beta_min = min(beta_bands)
        idx_max  = beta_bands.index(beta_max)
        f_lo, _  = bandas[idx_max]
        print(f"  janela {j+1:3d}/{N_JANELAS}  β_max={beta_max:.3f} ({f_lo:.0f}Hz)  β_min={beta_min:.3f}")

# ── montagem ────────────────────────────────────────────────────────────────
print("\nMontando áudio...")
audio = segmentos[0]
for seg in segmentos[1:]:
    audio = crossfade(audio, seg, fade=N_FADE)

# fade-out final
n_fo = int(N_FADEOUT_SEG * FS)
if len(audio) > n_fo:
    env = np.ones(len(audio))
    env[-n_fo:] = np.linspace(1.0, 0.0, n_fo)
    audio *= env

print(f"Duração total: {len(audio)/FS:.1f}s\n")

# ── exportar ────────────────────────────────────────────────────────────────
# referência: 15s de híbrido puro (antes do eco)
ref = np.concatenate(segmentos[:10])
salvar_wav(ref[:int(15*FS)], "beep880_puro_15s.wav")

# principal: 3 minutos com eco evolutivo
salvar_wav(audio, "beep880_eco_alpha333_3min.wav")

print("\nPlayback — híbrido puro (referência 15s):")
display(Audio("beep880_puro_15s.wav", rate=FS))

print("\nPlayback — eco α*=0.333 estendido (3 min):")
display(Audio("beep880_eco_alpha333_3min.wav", rate=FS))

print("\nConcluído.")
