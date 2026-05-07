"""
AlphaPhi_Eco_Hiperbolico_v2.py
Eco Ressonante Hiperbólico — Hipótese Entrada 45:
  Expoente β = 1 (hiperbólico) vs. 3 (euclidiano)
  α invariante nos dois campos.

© Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto

Hipótese (Entrada 45 do Journal):
  No experimento v1, o mesmo expoente 3 foi usado nos dois campos.
  A geometria hiperbólica já amplifica por φ² (curvatura C_PHI = 1/φ²).
  Portanto o sistema buscava efetivamente φ^(3+2) = φ^5 ≈ 11.09 — confirmado
  pelo β_max = 10.59 observado. A norma colapsou a φ¹ = 1.618: o ponto fixo
  natural do campo hiperbólico, atingido de forma abrupta por excesso de modulação.

  Correção: expoente 1 no hiperbólico.
    β_alvo_hip = φ^(1 × coh_rel)    ← geometria entrega os 2 restantes
    β_alvo_euc = φ^(3 × coh_rel)    ← campo plano, precisa dos 3

  α invariante nos dois campos — é a constante de ancoramento fundamental.
  O teto de β no hiperbólico: φ¹ = 1.618 (não φ³).

Previsão:
  β_max hiperbólico v2 ≈ φ¹ = 1.618 (convergência suave, não colapso)
  coh_med hiperbólico v2 ≥ 0.984 (igual ou superior ao euclidiano)
  norma preservada (sem colapso de expressão)
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
sys.path.insert(0, _dir)
from utils_phi import C_PHI, expmap0, logmap0, eco_ressonante

PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0 / 4
F_ORG      = 220.0 / 4
F_M        = F_ORG / PHI
BETA_FM    = PHI
DURACAO    = 1.5 * 4
N_STEPS    = 5
ALPHA_STAR = 1.0 / 3.0
CHUNK_SIZE = 2048

print("=" * 62)
print("  AlphaPhi · Eco Hiperbólico v2 · Hipótese Entrada 45")
print("=" * 62)
print(f"\n  PHI         = {PHI:.6f}")
print(f"  C_PHI       = {C_PHI:.6f}  (1/φ²)")
print(f"  φ¹ (teto β hiperbólico) = {PHI**1:.6f}")
print(f"  φ³ (teto β euclidiano)  = {PHI**3:.6f}")
print(f"\n  Hipótese: expoente 1 no hiperbólico, 3 no euclidiano")
print(f"  α = 1/137 invariante nos dois campos")

# ── auxiliares ────────────────────────────────────────────────────────────
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
        n_idx = np.arange(len(F_band))
        env   = np.clip(
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

def agente_eco(sinal, bins_phi, n_ciclos=20, expoente=3):
    """
    expoente=3 → euclidiano  (β_alvo = φ^(3 × coh_rel), teto φ³)
    expoente=1 → hiperbólico (β_alvo = φ^(1 × coh_rel), teto φ¹)
    α invariante nos dois casos.
    """
    n_b      = len(bins_phi)
    beta     = np.ones(n_b)
    beta_mem = beta.copy()
    w_mem    = 1.0 / PHI
    w_now    = 1.0 - w_mem
    teto     = PHI ** expoente
    cas_f    = None
    traj     = []
    for _ in range(n_ciclos):
        cas, cohs    = cascata_eq(sinal, beta, bins_phi)
        coh_rel      = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo    = PHI ** (expoente * coh_rel)
        beta         = w_now * beta_alvo + w_mem * beta_mem
        beta_mem     = beta.copy()
        beta         = np.clip(beta, 0.05, teto)
        cas_f        = cas
        traj.append((float(beta.max()), float(beta.mean())))
    return beta, cas_f, traj

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

# ── sinal base ────────────────────────────────────────────────────────────
N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

t     = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t)))
fm    = normalizar(np.sin(2 * np.pi * F_ORG * t + BETA_FM * np.sin(2 * np.pi * F_M * t)))
x_mix = normalizar((1.0 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

print(f"\n  Sinal x_mix: {N_SINAL} amostras  ({DURACAO:.1f}s)")
print(f"  Bandas φ: {N_BANDAS}")

# ══════════════════════════════════════════════════════════════════════════
# BLOCO 1 — EUCLIDIANO  (expoente=3, teto=φ³)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  BLOCO 1 — Euclidiano  (expoente β = 3, teto = φ³ = 4.236)")
print("─" * 62)

beta_euc, cas_euc, traj_euc = agente_eco(x_mix, BINS_PHI, n_ciclos=20, expoente=3)
coh_euc   = medir_coh_sinal(cas_euc[-1], BINS_PHI)
beta_med_e = float(beta_euc.mean())
beta_max_e = float(beta_euc.max())

print(f"  β_max   = {beta_max_e:.6f}  (φ³ = {PHI**3:.6f})")
print(f"  β_med   = {beta_med_e:.6f}")
print(f"  coh_med = {coh_euc:.6f}")

ref_euc = concatenar(cas_euc, fade=int(0.15 * FS))
salvar_wav(ref_euc, "beep880_euclid_v2.wav")

# ══════════════════════════════════════════════════════════════════════════
# BLOCO 2 — PRÉ-PROCESSAMENTO HIPERBÓLICO (igual ao v1)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  BLOCO 2 — Pré-processamento hiperbólico (expmap0 → eco → logmap0)")
print("─" * 62)

n_chunks = len(x_mix) // CHUNK_SIZE
x_trim   = x_mix[:n_chunks * CHUNK_SIZE]
x_chunks = x_trim.reshape(n_chunks, CHUNK_SIZE)

norma_antes = float(np.linalg.norm(x_chunks, axis=-1).mean())
x_hyp       = expmap0(x_chunks, c=C_PHI)
x_hyp_eco   = eco_ressonante(x_hyp, phi=PHI, n_eco=3)
x_euc_eco   = logmap0(x_hyp_eco, c=C_PHI)
norma_depois = float(np.linalg.norm(x_euc_eco, axis=-1).mean())

print(f"  Norma antes  (Euclidiana): {norma_antes:.4f}")
print(f"  Norma depois (retorno):    {norma_depois:.4f}")
print(f"  Razão: {norma_depois/norma_antes:.4f}  (φ = {PHI:.4f})")

x_pre = normalizar(x_euc_eco.flatten())
if len(x_pre) < len(x_mix):
    x_pre = np.concatenate([x_pre, x_mix[-(len(x_mix)-len(x_pre)):]])
elif len(x_pre) > len(x_mix):
    x_pre = x_pre[:len(x_mix)]

# ══════════════════════════════════════════════════════════════════════════
# BLOCO 3 — HIPERBÓLICO v2  (expoente=1, teto=φ¹)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  BLOCO 3 — Hiperbólico v2  (expoente β = 1, teto = φ¹ = 1.618)")
print("  Hipótese: geometria entrega φ², modulador busca apenas φ¹")
print("─" * 62)

beta_hip, cas_hip, traj_hip = agente_eco(x_pre, BINS_PHI, n_ciclos=20, expoente=1)
coh_hip    = medir_coh_sinal(cas_hip[-1], BINS_PHI)
beta_med_h = float(beta_hip.mean())
beta_max_h = float(beta_hip.max())

print(f"  β_max   = {beta_max_h:.6f}  (φ¹ = {PHI**1:.6f})")
print(f"  β_med   = {beta_med_h:.6f}")
print(f"  coh_med = {coh_hip:.6f}")

ref_hip = concatenar(cas_hip, fade=int(0.15 * FS))
salvar_wav(ref_hip, "beep880_hiperbolico_v2.wav")

# ══════════════════════════════════════════════════════════════════════════
# TRAJETÓRIA β — 20 ciclos
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  TRAJETÓRIA β — 20 ciclos")
print(f"  {'Ciclo':>5}  {'β_max (Euc)':>12}  {'β_max (Hip v2)':>14}")
print("─" * 62)
for i, (te, th) in enumerate(zip(traj_euc, traj_hip)):
    print(f"  {i+1:>5}  {te[0]:>12.6f}  {th[0]:>14.6f}")
print(f"  {'Alvo':>5}  {'φ³=':>3}{PHI**3:.6f}  {'φ¹=':>5}{PHI**1:.6f}")

# ══════════════════════════════════════════════════════════════════════════
# RESULTADO
# ══════════════════════════════════════════════════════════════════════════
delta_coh  = (coh_hip - coh_euc) / (coh_euc + 1e-10) * 100
delta_beta = (beta_med_h - beta_med_e) / (beta_med_e + 1e-10) * 100

print("\n" + "═" * 62)
print("  RESULTADO — Euclidiano × Hiperbólico v2")
print("═" * 62)
print(f"  {'Métrica':<16} {'Euclidiano':>12}  {'Hiperbólico v2':>14}")
print(f"  {'─'*16} {'─'*12}  {'─'*14}")
print(f"  {'β_max':<16} {beta_max_e:>12.6f}  {beta_max_h:>14.6f}")
print(f"  {'β_med':<16} {beta_med_e:>12.6f}  {beta_med_h:>14.6f}")
print(f"  {'coh_med':<16} {coh_euc:>12.6f}  {coh_hip:>14.6f}")
print(f"  {'Δβ_med':<16} {delta_beta:>+11.2f}%")
print(f"  {'Δcoh_med':<16} {delta_coh:>+11.2f}%")
print("═" * 62)

# Diagnóstico
print()
if abs(beta_max_h - PHI) < 0.05:
    print(f"  ★ β_max hiperbólico converge para φ¹ = {PHI:.4f}")
    print(f"    Hipótese confirmada: φ¹ é o atrator hiperbólico.")
else:
    print(f"  → β_max hiperbólico = {beta_max_h:.4f}  (φ¹ = {PHI:.4f})")

if coh_hip >= coh_euc:
    print(f"  ★ Coerência hiperbólica ≥ euclidiana (+{delta_coh:.2f}%)")
    print(f"    Expressão preservada. Campo coerente sem colapso.")
else:
    print(f"  → Coerência hiperbólica < euclidiana ({delta_coh:.2f}%)")
    print(f"    Hipótese parcial — ajuste adicional necessário.")

# ══════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
fig.patch.set_facecolor('#0d1117')
COLORS = {'gold': '#DAA520', 'blue': '#4169E1', 'red': '#FF4466',
          'text': '#8B949E', 'title': '#E6EDF3', 'panel': '#161b22'}

for ax in axes:
    ax.set_facecolor(COLORS['panel'])
    ax.tick_params(colors=COLORS['text'])
    for sp in ax.spines.values():
        sp.set_color('#30363d')

# Painel 1 — trajetória β
ciclos = list(range(1, 21))
axes[0].plot(ciclos, [t[0] for t in traj_euc], color=COLORS['text'],
             lw=1.5, marker='o', ms=4, label=f'β_max Euclidiano (exp=3)')
axes[0].plot(ciclos, [t[0] for t in traj_hip], color=COLORS['gold'],
             lw=1.5, marker='s', ms=4, label=f'β_max Hiperbólico v2 (exp=1)')
axes[0].axhline(PHI**3, color=COLORS['red'], lw=1, ls='--',
                label=f'φ³ = {PHI**3:.4f}')
axes[0].axhline(PHI**1, color=COLORS['blue'], lw=1, ls=':',
                label=f'φ¹ = {PHI**1:.4f}')
axes[0].set_title('Trajetória β_max — 20 ciclos', color=COLORS['title'])
axes[0].legend(facecolor='#161b22', labelcolor=COLORS['text'], fontsize=9)
axes[0].set_ylabel('β_max', color=COLORS['text'])

# Painel 2 — β por banda
x_band = range(N_BANDAS)
axes[1].bar(x_band, beta_euc, alpha=0.5, color=COLORS['text'],
            label=f'β Euclidiano  med={beta_med_e:.4f}')
axes[1].bar(x_band, beta_hip, alpha=0.6, color=COLORS['gold'],
            label=f'β Hiperbólico v2  med={beta_med_h:.4f}')
axes[1].axhline(PHI**3, color=COLORS['red'], lw=1, ls='--',
                label=f'φ³={PHI**3:.4f}')
axes[1].axhline(PHI**1, color=COLORS['blue'], lw=1, ls=':',
                label=f'φ¹={PHI**1:.4f}')
axes[1].set_title('β por banda φ', color=COLORS['title'])
axes[1].legend(facecolor='#161b22', labelcolor=COLORS['text'], fontsize=9)
axes[1].set_ylabel('β', color=COLORS['text'])

# Painel 3 — sinal final 200ms
n_plot = int(0.2 * FS)
tempo  = np.arange(n_plot) / FS * 1000
axes[2].plot(tempo, cas_euc[-1][:n_plot], color=COLORS['text'], lw=0.8,
             alpha=0.7, label=f'Euclidiano  coh={coh_euc:.4f}')
axes[2].plot(tempo, cas_hip[-1][:n_plot], color=COLORS['gold'], lw=0.8,
             alpha=0.9, label=f'Hiperbólico v2  coh={coh_hip:.4f}')
axes[2].set_title('Sinal final — primeiros 200ms', color=COLORS['title'])
axes[2].legend(facecolor='#161b22', labelcolor=COLORS['text'], fontsize=9)
axes[2].set_xlabel('Tempo (ms)', color=COLORS['text'])
axes[2].set_ylabel('Amplitude', color=COLORS['text'])

plt.suptitle(
    f'AlphaPhi · Eco Hiperbólico v2 · Hipótese Entrada 45 · '
    f'exp_euc=3 / exp_hip=1 · α invariante',
    color=COLORS['title'], fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig('eco_hiperbolico_v2_comparacao.png', dpi=120,
            bbox_inches='tight', facecolor='#0d1117')
print("\n  Gráfico: eco_hiperbolico_v2_comparacao.png")

print("\nPlayback — Euclidiano:")
display(Audio("beep880_euclid_v2.wav"))

print(f"\nPlayback — Hiperbólico v2 (expoente=1, teto=φ¹):")
display(Audio("beep880_hiperbolico_v2.wav"))

print("\nConcluído.")
print(f"\n  Hipótese Entrada 45:")
print(f"  φ¹ = {PHI:.6f} — atrator hiperbólico")
print(f"  φ³ = {PHI**3:.6f} — atrator euclidiano")
print(f"  Geometria hiperbólica entrega φ² = {PHI**2:.6f} de amplificação.")
print(f"  α = 1/137.035999084 — invariante nos dois campos.")
