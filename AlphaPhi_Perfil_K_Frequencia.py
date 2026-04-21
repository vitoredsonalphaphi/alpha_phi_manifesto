# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0

"""
AlphaPhi_Perfil_K_Frequencia.py
Vitor Edson Delavi · Florianópolis · 2026

Questão: o mesmo princípio que produziu k ≈ √2 globalmente (sem ser
programado) pode ser aplicado BANDA A BANDA para mapear onde um sinal
é coerente no espectro — e detectar a FREQUÊNCIA DE TRANSIÇÃO entre
regimes sem instrução explícita?

O k global emergiu porque:
    medir_campo(X) calcula a entropia do espectro coletivo → coerência
    → k = √2 + (φ - √2) * coerência

Aqui fazemos o mesmo por banda de frequência:
    para cada banda b → medir_campo(X_b) → k_b

O perfil k(f) é uma assinatura espectral do sinal.
O ponto de maior gradiente em k(f) é a frequência de transição.

Experimento:
    Três classes: Alpha (8-13 Hz), Beta (13-30 Hz), Ruído (1/f)
    fs = 256 Hz, N = 256 amostras (1 segundo)
    Bandas: 32 bandas × 4 Hz cada (0 a 128 Hz)

    Hipótese: sem instrução, o perfil k revela automaticamente
    onde cada classe é coerente — e a transição Alpha→Beta aparece
    como gradiente máximo em torno de 13 Hz.
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils_phi import PHI

# ── Constantes ─────────────────────────────────────────────────────────────
FS      = 256       # Hz
N       = 256       # amostras (1 segundo)
N_SINAIS = 200      # por classe
N_BANDAS = 64       # bandas de 2 Hz cada (0-128 Hz)
K_MIN   = np.sqrt(2)

print("Experimento: Perfil k(f) por banda de frequência")
print(f"fs={FS} Hz | N={N} | {N_BANDAS} bandas × {(FS//2)//N_BANDAS} Hz cada")
print(f"k_min=√2={K_MIN:.5f} | k_max=φ={PHI:.5f}\n")

# ── Geração de sinais ──────────────────────────────────────────────────────

def pink_noise(n, rng, amplitude=0.3):
    white = rng.normal(0, 1, n * 4)
    freqs = np.fft.rfft(white)
    f = np.arange(1, len(freqs) + 1, dtype=float)
    freqs = freqs / np.sqrt(f)
    pink = np.fft.irfft(freqs)[:n]
    return amplitude * pink / (np.std(pink) + 1e-8)

def gerar_alpha(n_sinais, rng):
    """Sinais com dominância Alpha: 8-13 Hz, fundo rosa reduzido."""
    t = np.arange(N) / FS
    X = np.zeros((n_sinais, N))
    for i in range(n_sinais):
        f_dom = rng.uniform(8.0, 13.0)
        amp   = rng.uniform(5.0, 8.0)          # SNR alto
        fase  = rng.uniform(0, 2 * np.pi)
        sinal = amp * np.sin(2 * np.pi * f_dom * t + fase)
        # harmônico dentro da banda Alpha
        sinal += rng.uniform(0.15, 0.3) * amp * np.sin(4 * np.pi * f_dom * t + rng.uniform(0, 2*np.pi))
        sinal += pink_noise(N, rng, amplitude=0.15)   # ruído fraco
        X[i] = sinal / (np.std(sinal) + 1e-8)
    return X

def gerar_beta(n_sinais, rng):
    """Sinais com dominância Beta: 15-25 Hz, alta amplitude, ruído fraco."""
    t = np.arange(N) / FS
    X = np.zeros((n_sinais, N))
    for i in range(n_sinais):
        f_dom = rng.uniform(15.0, 25.0)        # Beta central, sem overlap com Alpha
        amp   = rng.uniform(5.0, 8.0)
        fase  = rng.uniform(0, 2 * np.pi)
        sinal = amp * np.sin(2 * np.pi * f_dom * t + fase)
        sinal += rng.uniform(0.2, 0.4) * amp * np.sin(2 * np.pi * f_dom * 1.5 * t + rng.uniform(0, 2*np.pi))
        sinal += pink_noise(N, rng, amplitude=0.15)
        X[i] = sinal / (np.std(sinal) + 1e-8)
    return X

def gerar_ruido(n_sinais, rng):
    """Ruído 1/f puro — sem frequência dominante."""
    X = np.zeros((n_sinais, N))
    for i in range(n_sinais):
        X[i] = pink_noise(N, rng, amplitude=1.0)
        X[i] = X[i] / (np.std(X[i]) + 1e-8)
    return X

# ── Campo coletivo por banda ──────────────────────────────────────────────

def perfil_k_por_banda(X, n_bandas=N_BANDAS):
    """
    Calcula o campo coletivo k para cada banda de frequência.

    Para cada banda b:
        1. Extrai os bins FFT daquela banda do batch X
        2. Calcula amplitude média por bin dentro da banda
        3. Normaliza → distribuição de probabilidade da energia dentro da banda
        4. Calcula entropia → coerência intra-banda → k_b

    Retorna:
        k_perfil: array [n_bandas] com k de cada banda
        freq_hz:  array [n_bandas] com frequência central de cada banda (Hz)
    """
    freq_batch = np.fft.fft(X, axis=-1)           # (n_sinais, N) complexo
    N_freq     = N // 2                            # bins positivos (Nyquist)
    bsize      = N_freq // n_bandas                # bins por banda

    k_perfil = np.zeros(n_bandas)
    freq_hz  = np.zeros(n_bandas)

    for b in range(n_bandas):
        start = b * bsize
        end   = start + bsize
        # amplitude média nessa banda, sobre todo o batch
        amp_banda = np.abs(freq_batch[:, start:end]).mean(axis=0)
        amp_norm  = np.clip(amp_banda / (amp_banda.sum() + 1e-8), 1e-10, 1.0)
        ent       = -np.sum(amp_norm * np.log(amp_norm))
        coh       = float(np.clip(1.0 - ent / np.log(bsize + 1e-8), 0.0, 1.0))
        k_b       = K_MIN + (PHI - K_MIN) * coh
        k_perfil[b] = k_b
        freq_hz[b]  = (start + bsize / 2) * FS / N   # frequência central (Hz)

    return k_perfil, freq_hz

def ponto_transicao(k_perfil, freq_hz):
    """
    Frequência onde o gradiente de k é máximo — transição entre regimes.
    """
    grad = np.abs(np.diff(k_perfil))
    idx  = np.argmax(grad)
    f_tr = (freq_hz[idx] + freq_hz[idx + 1]) / 2
    return f_tr, grad

# ── Execução ───────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
X_alpha = gerar_alpha(N_SINAIS, rng)
X_beta  = gerar_beta(N_SINAIS,  rng)
X_ruido = gerar_ruido(N_SINAIS, rng)

print("Calculando perfil k(f) para cada classe...")
k_alpha, freq_hz = perfil_k_por_banda(X_alpha)
k_beta,  _       = perfil_k_por_banda(X_beta)
k_ruido, _       = perfil_k_por_banda(X_ruido)

# Verificar k global (comparar com valor esperado ≈ √2)
def k_global(X):
    freq_batch = np.fft.fft(X, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)
    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    ent        = -np.sum(amp_norm * np.log(amp_norm))
    coh        = float(1.0 - ent / np.log(N))
    return K_MIN + (PHI - K_MIN) * coh

print(f"\n  k global Alpha: {k_global(X_alpha):.5f}")
print(f"  k global Beta:  {k_global(X_beta):.5f}")
print(f"  k global Ruído: {k_global(X_ruido):.5f}")
print(f"  √2 = {K_MIN:.5f}  φ = {PHI:.5f}")

# Transições
f_tr_alpha, grad_alpha = ponto_transicao(k_alpha, freq_hz)
f_tr_beta,  grad_beta  = ponto_transicao(k_beta,  freq_hz)

print(f"\n  Transição detectada (max gradiente k):")
print(f"    Alpha: {f_tr_alpha:.1f} Hz  (esperado: pico em 8-13 Hz)")
print(f"    Beta:  {f_tr_beta:.1f} Hz  (esperado: pico em 13-30 Hz)")

# Frequência de pico de k por classe
print(f"\n  Frequência com k máximo (mais coerente):")
print(f"    Alpha: {freq_hz[np.argmax(k_alpha)]:.1f} Hz — k={k_alpha.max():.5f}")
print(f"    Beta:  {freq_hz[np.argmax(k_beta)]:.1f} Hz  — k={k_beta.max():.5f}")
print(f"    Ruído: {freq_hz[np.argmax(k_ruido)]:.1f} Hz — k={k_ruido.max():.5f}")

# ── Visualização ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#0d1117")
COLORS = {"alpha": "#4FC3F7", "beta": "#FF8C00", "ruido": "#888888",
          "text": "#E6EDF3", "sub": "#8B949E", "panel": "#161b22",
          "grid": "#21262d", "sqrt2": "#2ECC71", "phi": "#DAA520"}

# 1. Perfil k(f) por classe
ax = axes[0, 0]
ax.set_facecolor(COLORS["panel"])
ax.plot(freq_hz, k_alpha, color=COLORS["alpha"], linewidth=2.5,
        label=f"Alpha (8-13 Hz)", marker='o', markersize=4)
ax.plot(freq_hz, k_beta,  color=COLORS["beta"],  linewidth=2.5,
        label=f"Beta  (13-30 Hz)", marker='s', markersize=4)
ax.plot(freq_hz, k_ruido, color=COLORS["ruido"], linewidth=1.5,
        label="Ruído (1/f)", linestyle="--", alpha=0.7)
ax.axhline(K_MIN, color=COLORS["sqrt2"], linestyle=":", linewidth=1.5,
           alpha=0.7, label=f"√2 = {K_MIN:.4f}")
ax.axhline(PHI,   color=COLORS["phi"],   linestyle=":", linewidth=1.5,
           alpha=0.7, label=f"φ  = {PHI:.4f}")
ax.axvspan(8, 13,  alpha=0.08, color=COLORS["alpha"])
ax.axvspan(13, 30, alpha=0.08, color=COLORS["beta"])
ax.set_title("Perfil k(f) por banda — cada classe", color=COLORS["text"], fontsize=11)
ax.set_xlabel("Frequência (Hz)", color=COLORS["sub"])
ax.set_ylabel("k_campo", color=COLORS["sub"])
ax.legend(facecolor=COLORS["panel"], labelcolor=COLORS["text"], fontsize=8)
ax.set_xlim(0, 60)
ax.tick_params(colors=COLORS["sub"])
ax.grid(color=COLORS["grid"], linewidth=0.5)
for spine in ax.spines.values(): spine.set_edgecolor("#30363d")

# 2. Gradiente de k(f) — onde muda mais rápido
ax = axes[0, 1]
ax.set_facecolor(COLORS["panel"])
f_meio = (freq_hz[:-1] + freq_hz[1:]) / 2
ax.plot(f_meio, grad_alpha, color=COLORS["alpha"], linewidth=2, label="Alpha")
ax.plot(f_meio, grad_beta,  color=COLORS["beta"],  linewidth=2, label="Beta")
ax.axvline(f_tr_alpha, color=COLORS["alpha"], linestyle="--", alpha=0.6,
           label=f"Transição Alpha: {f_tr_alpha:.1f} Hz")
ax.axvline(f_tr_beta,  color=COLORS["beta"],  linestyle="--", alpha=0.6,
           label=f"Transição Beta:  {f_tr_beta:.1f} Hz")
ax.axvline(13, color="#ffffff", linestyle=":", alpha=0.3, linewidth=1)
ax.set_title("Gradiente |Δk/Δf| — ponto de transição", color=COLORS["text"], fontsize=11)
ax.set_xlabel("Frequência (Hz)", color=COLORS["sub"])
ax.set_ylabel("|Δk|", color=COLORS["sub"])
ax.legend(facecolor=COLORS["panel"], labelcolor=COLORS["text"], fontsize=8)
ax.set_xlim(0, 60)
ax.tick_params(colors=COLORS["sub"])
ax.grid(color=COLORS["grid"], linewidth=0.5)
for spine in ax.spines.values(): spine.set_edgecolor("#30363d")

# 3. Espectro médio por classe
ax = axes[1, 0]
ax.set_facecolor(COLORS["panel"])
def espectro_medio(X):
    f = np.fft.rfft(X, axis=-1)
    amp = np.abs(f).mean(axis=0)
    freqs_hz = np.fft.rfftfreq(N, 1.0/FS)
    return freqs_hz, amp

fa, sa = espectro_medio(X_alpha)
fb, sb = espectro_medio(X_beta)
fr, sr = espectro_medio(X_ruido)
ax.plot(fa, sa, color=COLORS["alpha"], linewidth=2,   label="Alpha",  alpha=0.9)
ax.plot(fb, sb, color=COLORS["beta"],  linewidth=2,   label="Beta",   alpha=0.9)
ax.plot(fr, sr, color=COLORS["ruido"], linewidth=1.5, label="Ruído",  alpha=0.6, linestyle="--")
ax.axvspan(8, 13,  alpha=0.1, color=COLORS["alpha"])
ax.axvspan(13, 30, alpha=0.1, color=COLORS["beta"])
ax.set_title("Espectro médio por classe (referência)", color=COLORS["text"], fontsize=11)
ax.set_xlabel("Frequência (Hz)", color=COLORS["sub"])
ax.set_ylabel("Amplitude média", color=COLORS["sub"])
ax.set_xlim(0, 60)
ax.legend(facecolor=COLORS["panel"], labelcolor=COLORS["text"], fontsize=8)
ax.tick_params(colors=COLORS["sub"])
ax.grid(color=COLORS["grid"], linewidth=0.5)
for spine in ax.spines.values(): spine.set_edgecolor("#30363d")

# 4. k_max por classe — separação sem classificador
ax = axes[1, 1]
ax.set_facecolor(COLORS["panel"])
classes  = ["Alpha\n(8-13 Hz)", "Beta\n(13-30 Hz)", "Ruído\n(1/f)"]
k_maxes  = [k_alpha.max(), k_beta.max(), k_ruido.max()]
f_picos  = [freq_hz[np.argmax(k_alpha)], freq_hz[np.argmax(k_beta)], freq_hz[np.argmax(k_ruido)]]
bars = ax.bar(classes, k_maxes,
              color=[COLORS["alpha"], COLORS["beta"], COLORS["ruido"]],
              alpha=0.8, edgecolor="#30363d")
ax.axhline(K_MIN, color=COLORS["sqrt2"], linestyle=":", linewidth=1.5,
           alpha=0.8, label=f"√2 = {K_MIN:.4f}")
ax.axhline(PHI,   color=COLORS["phi"],   linestyle=":", linewidth=1.5,
           alpha=0.8, label=f"φ  = {PHI:.4f}")
for bar, km, fp in zip(bars, k_maxes, f_picos):
    ax.text(bar.get_x() + bar.get_width()/2, km + 0.002,
            f"k={km:.4f}\n@ {fp:.0f}Hz",
            ha='center', va='bottom', color=COLORS["text"], fontsize=8)
ax.set_title("k máximo por classe — coerência de pico", color=COLORS["text"], fontsize=11)
ax.set_ylabel("k máximo", color=COLORS["sub"])
ax.legend(facecolor=COLORS["panel"], labelcolor=COLORS["text"], fontsize=8)
ax.tick_params(colors=COLORS["sub"])
ax.set_ylim(K_MIN * 0.998, PHI * 1.01)
for spine in ax.spines.values(): spine.set_edgecolor("#30363d")

plt.suptitle("Campo Coletivo por Banda — k(f) emerge sem instrução explícita",
             color=COLORS["text"], fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("perfil_k_frequencia.png", dpi=140, bbox_inches="tight",
            facecolor="#0d1117")
print("\nGráfico salvo: perfil_k_frequencia.png")

# ── Resultados ─────────────────────────────────────────────────────────────

# Separabilidade sem classificador: distância entre perfis
sep_alpha_beta = float(np.linalg.norm(k_alpha - k_beta))
sep_alpha_ruido = float(np.linalg.norm(k_alpha - k_ruido))
sep_beta_ruido  = float(np.linalg.norm(k_beta  - k_ruido))

print(f"\n  Separabilidade dos perfis k(f) (distância L2):")
print(f"    Alpha vs Beta:  {sep_alpha_beta:.5f}")
print(f"    Alpha vs Ruído: {sep_alpha_ruido:.5f}")
print(f"    Beta  vs Ruído: {sep_beta_ruido:.5f}")
print(f"\n  → Perfis distintos sem treinamento, sem parâmetros ajustados.")

resultados = {
    "k_global": {
        "alpha": float(k_global(X_alpha)),
        "beta":  float(k_global(X_beta)),
        "ruido": float(k_global(X_ruido)),
        "sqrt2": float(K_MIN), "phi": float(PHI)
    },
    "k_pico_banda": {
        "alpha": {"k": float(k_alpha.max()), "freq_hz": float(freq_hz[np.argmax(k_alpha)])},
        "beta":  {"k": float(k_beta.max()),  "freq_hz": float(freq_hz[np.argmax(k_beta)])},
        "ruido": {"k": float(k_ruido.max()), "freq_hz": float(freq_hz[np.argmax(k_ruido)])},
    },
    "transicao_detectada_hz": {
        "alpha": float(f_tr_alpha),
        "beta":  float(f_tr_beta),
    },
    "separabilidade_L2": {
        "alpha_vs_beta":  sep_alpha_beta,
        "alpha_vs_ruido": sep_alpha_ruido,
        "beta_vs_ruido":  sep_beta_ruido,
    },
    "k_perfil_alpha": k_alpha.tolist(),
    "k_perfil_beta":  k_beta.tolist(),
    "k_perfil_ruido": k_ruido.tolist(),
    "freq_hz":        freq_hz.tolist(),
}

with open("perfil_k_frequencia.json", "w") as f:
    json.dump(resultados, f, indent=2)
print("Resultados: perfil_k_frequencia.json")
