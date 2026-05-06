"""
AlphaPhi_Eco_Hiperbolico.py
Eco Ressonante no Espaço Hiperbólico (bola de Poincaré, c = C_PHI = 1/φ²)
Célula única para Google Colab.

© Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto

Hipótese (Entrada 38 do Journal):
  O espaço hiperbólico comprime amplitudes grandes via tanh(√c · ‖v‖).
  Com c = C_PHI = 1/φ², a curvatura é a proporção áurea inversa ao quadrado.
  A bola de Poincaré tem raio máximo 1/√C_PHI = φ.
  Estruturas de baixa amplitude (ruído, resíduos estruturais) ficam
  relativamente amplificadas no interior da bola — o eco ressonante
  aplicado nesse espaço "vê" o sinal de forma diferente e pode revelar
  atratores ocultos.

Pipeline:
  1. Gerar x_mix  (α*=0.333, domínio 0.25×, mesmo que Ergonômico)
  2. Segmentar em janelas de CHUNK_SIZE amostras (vetores em R^CHUNK_SIZE)
  3. expmap0(x_chunks, c=C_PHI)  → espaço hiperbólico (bola de Poincaré)
  4. eco_ressonante(x_hyp)        → ressonância φ no espaço hiperbólico
  5. logmap0(x_hyp_eco, c=C_PHI) → retorno ao Euclidiano
  6. agente_eco(x_pre)            → cascata eco-φ padrão
  7. Comparar β, coh_med vs. resultado Euclidiano puro
  8. Gerar áudio para comparação sensorial

Arquivos gerados:
  beep880_euclid.wav    — pipeline Euclidiano (baseline)
  beep880_hiperbolico.wav — pipeline Hiperbólico
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── importar utilitários φ ─────────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
sys.path.insert(0, _dir)
from utils_phi import C_PHI, expmap0, logmap0, eco_ressonante

# ── constantes (domínio 0.25× — mesmo que Ergonômico) ─────────────────────
PHI        = (1 + np.sqrt(5)) / 2    # 1.6180...
FS         = 44100
F_BEEP     = 880.0 / 4               #  220Hz
F_ORG      = 220.0 / 4               #   55Hz
F_M        = F_ORG / PHI             #   ≈34Hz
BETA_FM    = PHI
DURACAO    = 1.5 * 4                 #    6s por janela
N_STEPS    = 5
ALPHA_STAR = 1.0 / 3.0              # ponto de emergência 880Hz

CHUNK_SIZE = 2048                    # janela hiperbólica (≈46ms)

print("=" * 58)
print("  AlphaPhi · Eco Hiperbólico · c = C_PHI = 1/φ²")
print("=" * 58)
print(f"\n  PHI    = {PHI:.6f}")
print(f"  C_PHI  = {C_PHI:.6f}  (1/φ² = curvatura natural)")
print(f"  Raio bola Poincaré = 1/√C_PHI = {1/np.sqrt(C_PHI):.6f} ≈ φ")
print(f"  CHUNK_SIZE = {CHUNK_SIZE}  ({CHUNK_SIZE/FS*1000:.1f}ms)")

# ── funções auxiliares ─────────────────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

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
        b_lo = max(0, int(f_lo / (FS / n)))
        b_hi = min(int(f_hi / (FS / n)) + 1, n // 2 + 1)
        result.append((b_lo, b_hi, f_lo, f_hi))
    return result

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    w_mem = 1.0 / PHI
    w_now = 1.0 - w_mem
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        beta_i = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        F_band = F[b_lo:b_hi]
        mag    = np.abs(F_band)
        phase  = np.angle(F_band)
        an     = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh    = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        if coh_mem is not None and i < len(coh_mem):
            coh_ef = w_now * coh + w_mem * float(coh_mem[i])
        else:
            coh_ef = coh
        cohs.append(coh)
        n_idx  = np.arange(len(F_band))
        env    = np.clip(
            1.0 + (coh_ef * PHI ** beta_i) * np.cos(2.0 * np.pi * n_idx / PHI),
            0.05, None
        )
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)
    result = np.fft.irfft(F_out, n=N)
    return result / (np.max(np.abs(result)) + 1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    n_b, coh_mem = len(bins_phi), np.zeros(len(bins_phi))
    cohs_f = np.zeros(n_b)
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem, cohs_f = cohs, cohs
        s_e = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()
    return cas, cohs_f

def agente_eco(sinal, bins_phi, n_ciclos=20):
    n_b   = len(bins_phi)
    beta  = np.ones(n_b)
    beta_mem = beta.copy()
    w_mem = 1.0 / PHI
    w_now = 1.0 - w_mem
    cas_f = None
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        coh_rel   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo = PHI ** (3 * coh_rel)
        beta      = w_now * beta_alvo + w_mem * beta_mem
        beta_mem  = beta.copy()
        beta      = np.clip(beta, 0.05, PHI**3)
        cas_f     = cas
    return beta, cas_f

def medir_coh_sinal(sinal, bins_phi):
    _, cohs = eco_eq(sinal, bins_phi, np.ones(len(bins_phi)))
    return float(cohs.mean())

def salvar_wav(sinal, nome):
    s16 = np.int16(np.clip(normalizar(sinal), -1.0, 1.0) * 32767)
    wavfile.write(nome, FS, s16)
    print(f"  → {nome}  ({len(sinal)/FS:.1f}s)")

def concatenar(cas, fade=2000):
    out = cas[0].copy()
    for s in cas[1:]:
        fade = min(fade, len(out), len(s))
        t    = np.linspace(0.0, 1.0, fade)
        out[-fade:] = out[-fade:] * (1.0 - t) + s[:fade] * t
        out  = np.concatenate([out, s[fade:]])
    return normalizar(out)

# ── gerar sinal base ───────────────────────────────────────────────────────
N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

t        = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep     = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t)))
fm       = normalizar(np.sin(2 * np.pi * F_ORG * t + BETA_FM * np.sin(2 * np.pi * F_M * t)))
x_mix    = normalizar((1.0 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

print(f"\n  Sinal x_mix: {N_SINAL} amostras  ({DURACAO:.1f}s)")
print(f"  Bandas φ: {N_BANDAS}")

# ══════════════════════════════════════════════════════════════════════════
# BLOCO 1 — BASELINE EUCLIDIANO
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 58)
print("  BLOCO 1 — Euclidiano (baseline)")
print("─" * 58)

beta_euclid, cas_euclid = agente_eco(x_mix, BINS_PHI, n_ciclos=20)

coh_euclid   = medir_coh_sinal(cas_euclid[-1], BINS_PHI)
beta_med_e   = float(beta_euclid.mean())
beta_max_e   = float(beta_euclid.max())
idx_at_e     = int(np.argmax(beta_euclid))
f_at_e       = (BANDAS[idx_at_e][0] + BANDAS[idx_at_e][1]) / 2

print(f"  β_med   = {beta_med_e:.4f}  (φ³ = {PHI**3:.4f})")
print(f"  β_max   = {beta_max_e:.4f}  @ {f_at_e:.1f}Hz")
print(f"  coh_med = {coh_euclid:.4f}")

ref_euclid = concatenar(cas_euclid, fade=int(0.15 * FS))
salvar_wav(ref_euclid, "beep880_euclid.wav")

# ══════════════════════════════════════════════════════════════════════════
# BLOCO 2 — ECO HIPERBÓLICO
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 58)
print("  BLOCO 2 — Hiperbólico (expmap0 → eco → logmap0)")
print("─" * 58)

# 2a. Segmentar em chunks (vetores em R^CHUNK_SIZE)
n_chunks = len(x_mix) // CHUNK_SIZE
x_trim   = x_mix[:n_chunks * CHUNK_SIZE]
x_chunks = x_trim.reshape(n_chunks, CHUNK_SIZE)

norma_e_antes = float(np.linalg.norm(x_chunks, axis=-1).mean())
print(f"\n  Segmentos: {n_chunks} × {CHUNK_SIZE}  ({CHUNK_SIZE/FS*1000:.1f}ms)")
print(f"  Norma média (Euclidiana):  {norma_e_antes:.4f}")

# 2b. Euclidiano → Hiperbólico (bola de Poincaré, c = C_PHI)
# expmap0: v → tanh(√c · ‖v‖) / (√c · ‖v‖) · v
# Raio da bola: 1/√C_PHI = φ ≈ 1.6180
x_hyp      = expmap0(x_chunks, c=C_PHI)
norma_hyp  = float(np.linalg.norm(x_hyp, axis=-1).mean())
raio_ball  = 1.0 / np.sqrt(C_PHI)
print(f"  Norma média (Hiperbólica): {norma_hyp:.4f}  "
      f"(raio bola = {raio_ball:.4f} = φ)")
print(f"  Compressão: {norma_hyp/norma_e_antes:.4f}  "
      f"(tanh atua como filtro de curvatura)")

# 2c. Eco ressonante em espaço hiperbólico (3 ciclos)
# eco_ressonante opera ao longo de axis=-1 (chunk_size) — FFT por chunk
# No espaço hiperbólico: a fase é rotacionada por φ no espaço comprimido
x_hyp_eco = eco_ressonante(x_hyp, phi=PHI, n_eco=3)
norma_eco  = float(np.linalg.norm(x_hyp_eco, axis=-1).mean())
print(f"\n  Eco ressonante (3 ciclos) aplicado no espaço hiperbólico")
print(f"  Norma pós-eco (Hiperbólica): {norma_eco:.4f}")

# 2d. Hiperbólico → Euclidiano (logmap0)
# logmap0: y → arctanh(√c · ‖y‖) / (√c · ‖y‖) · y
x_euclid_eco = logmap0(x_hyp_eco, c=C_PHI)
norma_e_pos  = float(np.linalg.norm(x_euclid_eco, axis=-1).mean())
print(f"  Norma pós-retorno (Euclidiana): {norma_e_pos:.4f}")

# 2e. Reconstruir sinal 1D e preencher extremidade
x_pre = normalizar(x_euclid_eco.flatten())
if len(x_pre) < len(x_mix):
    pad   = len(x_mix) - len(x_pre)
    x_pre = np.concatenate([x_pre, x_mix[-pad:]])
elif len(x_pre) > len(x_mix):
    x_pre = x_pre[:len(x_mix)]

print(f"\n  Sinal pré-processado reconstruído: {len(x_pre)} amostras")

# 2f. agente_eco no sinal pré-processado
print("  Rodando agente_eco (20 ciclos) no sinal hiperbólico...")
beta_hyp, cas_hyp = agente_eco(x_pre, BINS_PHI, n_ciclos=20)

coh_hyp   = medir_coh_sinal(cas_hyp[-1], BINS_PHI)
beta_med_h = float(beta_hyp.mean())
beta_max_h = float(beta_hyp.max())
idx_at_h   = int(np.argmax(beta_hyp))
f_at_h     = (BANDAS[idx_at_h][0] + BANDAS[idx_at_h][1]) / 2

print(f"  β_med   = {beta_med_h:.4f}  (φ³ = {PHI**3:.4f})")
print(f"  β_max   = {beta_max_h:.4f}  @ {f_at_h:.1f}Hz")
print(f"  coh_med = {coh_hyp:.4f}")

ref_hyp = concatenar(cas_hyp, fade=int(0.15 * FS))
salvar_wav(ref_hyp, "beep880_hiperbolico.wav")

# ══════════════════════════════════════════════════════════════════════════
# COMPARAÇÃO
# ══════════════════════════════════════════════════════════════════════════
delta_coh  = (coh_hyp - coh_euclid) / (coh_euclid + 1e-10) * 100
delta_beta = (beta_med_h - beta_med_e) / (beta_med_e + 1e-10) * 100

print("\n" + "═" * 58)
print("  RESULTADO — Euclidiano × Hiperbólico")
print("═" * 58)
print(f"  Métrica        Euclidiano     Hiperbólico")
print(f"  ─────────────  ───────────    ───────────")
print(f"  β_med          {beta_med_e:.4f}        {beta_med_h:.4f}")
print(f"  β_max          {beta_max_e:.4f}        {beta_max_h:.4f}")
print(f"  coh_med        {coh_euclid:.4f}        {coh_hyp:.4f}")
print(f"  f_atrator      {f_at_e:.2f}Hz       {f_at_h:.2f}Hz")
print(f"  Δβ_med         {delta_beta:+.2f}%")
print(f"  Δcoh_med       {delta_coh:+.2f}%")
print(f"  φ³             {PHI**3:.4f}")
print(f"  φ²             {PHI**2:.4f}")
print("═" * 58)

if coh_hyp > coh_euclid:
    print(f"\n  ★ Hiperbólico MAIS coerente (+{delta_coh:.2f}%)")
    print(f"    A curvatura C_PHI={C_PHI:.4f} amplificou a estrutura oculta.")
elif coh_hyp < coh_euclid:
    print(f"\n  → Hiperbólico MENOS coerente ({delta_coh:.2f}%)")
    print(f"    O espaço hiperbólico dispersou coerência — resultado inesperado.")
else:
    print(f"\n  → Coerências idênticas — espaço hiperbólico neutro.")

if abs(f_at_h - f_at_e) < 2.0:
    print(f"  ★ Atrator preservado: {f_at_h:.2f}Hz (invariante à curvatura)")
else:
    print(f"  → Atrator deslocado: {f_at_e:.2f}Hz → {f_at_h:.2f}Hz")
    if abs(f_at_h / f_at_e - PHI) < 0.05:
        print(f"    ★ Deslocamento ≈ φ! ({f_at_h/f_at_e:.4f} ≈ {PHI:.4f})")
    elif abs(f_at_h / f_at_e - 1.0/PHI) < 0.05:
        print(f"    ★ Deslocamento ≈ 1/φ! ({f_at_h/f_at_e:.4f} ≈ {1/PHI:.4f})")

# ══════════════════════════════════════════════════════════════════════════
# PLOT — espectros comparativos
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
fig.patch.set_facecolor('#0d1117')

PLOT_COLORS = {
    'gold': '#DAA520', 'blue': '#4169E1', 'red': '#FF4466',
    'text': '#8B949E', 'title': '#E6EDF3', 'panel': '#161b22',
}

def espectro(sinal, label, color, ax, f_max=300):
    N  = len(sinal)
    F  = np.abs(np.fft.rfft(sinal))
    fr = np.fft.rfftfreq(N, 1.0 / FS)
    m  = fr < f_max
    ax.fill_between(fr[m], F[m], alpha=0.3, color=color)
    ax.plot(fr[m], F[m], color=color, lw=1.2, label=label)

for ax in axes:
    ax.set_facecolor(PLOT_COLORS['panel'])
    ax.tick_params(colors=PLOT_COLORS['text'])
    for sp in ax.spines.values():
        sp.set_color('#30363d')

# Painel 1 — sinal inicial vs pré-processado hiperbólico
espectro(x_mix, 'x_mix (original)', PLOT_COLORS['text'], axes[0])
espectro(x_pre, 'x_pre (pós-hiperbólico)', PLOT_COLORS['blue'], axes[0])
axes[0].set_title('Sinal antes e depois do pré-processamento hiperbólico',
                  color=PLOT_COLORS['title'], fontsize=11)
axes[0].legend(facecolor='#161b22', labelcolor=PLOT_COLORS['text'], fontsize=9)
axes[0].set_ylabel('Amplitude', color=PLOT_COLORS['text'])

# Painel 2 — β por banda
x_band = range(N_BANDAS)
axes[1].bar(x_band, beta_euclid, alpha=0.5, color=PLOT_COLORS['text'],
            label=f'β Euclidiano (med={beta_med_e:.3f})')
axes[1].bar(x_band, beta_hyp, alpha=0.6, color=PLOT_COLORS['gold'],
            label=f'β Hiperbólico (med={beta_med_h:.3f})')
axes[1].axhline(PHI**3, color=PLOT_COLORS['red'], lw=1.2, ls='--',
                label=f'φ³ = {PHI**3:.4f}')
axes[1].axhline(PHI**2, color=PLOT_COLORS['blue'], lw=0.8, ls=':',
                label=f'φ² = {PHI**2:.4f}')
axes[1].set_title('β por banda φ — comparação Euclidiano × Hiperbólico',
                  color=PLOT_COLORS['title'], fontsize=11)
axes[1].legend(facecolor='#161b22', labelcolor=PLOT_COLORS['text'], fontsize=9)
axes[1].set_ylabel('β', color=PLOT_COLORS['text'])

# Painel 3 — sinal final euclidiano vs hiperbólico (primeiros 0.2s)
n_plot = int(0.2 * FS)
tempo  = np.arange(n_plot) / FS * 1000  # ms
axes[2].plot(tempo, cas_euclid[-1][:n_plot],
             color=PLOT_COLORS['text'], lw=0.8, alpha=0.7,
             label=f'Euclidiano  coh={coh_euclid:.4f}')
axes[2].plot(tempo, cas_hyp[-1][:n_plot],
             color=PLOT_COLORS['gold'], lw=0.8, alpha=0.9,
             label=f'Hiperbólico  coh={coh_hyp:.4f}')
axes[2].set_title('Sinal final — primeiros 200ms',
                  color=PLOT_COLORS['title'], fontsize=11)
axes[2].legend(facecolor='#161b22', labelcolor=PLOT_COLORS['text'], fontsize=9)
axes[2].set_xlabel('Tempo (ms)', color=PLOT_COLORS['text'])
axes[2].set_ylabel('Amplitude', color=PLOT_COLORS['text'])

plt.suptitle(
    f'AlphaPhi · Eco Hiperbólico · c=C_PHI={C_PHI:.4f} · raio bola=φ={PHI:.4f}',
    color=PLOT_COLORS['title'], fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig('eco_hiperbolico_comparacao.png', dpi=120, bbox_inches='tight',
            facecolor='#0d1117')
print("\n  Gráfico: eco_hiperbolico_comparacao.png")

# ══════════════════════════════════════════════════════════════════════════
# PLAYBACK
# ══════════════════════════════════════════════════════════════════════════
print("\nPlayback — Euclidiano (baseline):")
display(Audio("beep880_euclid.wav"))

print(f"\nPlayback — Hiperbólico (c=C_PHI={C_PHI:.4f}, raio bola=φ):")
display(Audio("beep880_hiperbolico.wav"))

print("\nConcluído.")
print(f"\n  C_PHI = 1/φ² = {C_PHI:.6f}")
print(f"  Raio da bola de Poincaré = 1/√C_PHI = φ = {PHI:.6f}")
print(f"  φ³ = {PHI**3:.6f}  (atrator β)")
print(f"  Entrada 38 do Journal: eco retorna ao campo — ciclo completo.")
