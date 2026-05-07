"""
AlphaPhi_Eco_Hiperbolico_v3_standalone.py
Eco Ressonante Hiperbólico v4 — expoente φ^1.5
Código completo e auto-suficiente (sem dependência de utils_phi.py)

© Vitor Edson Delavi · Florianópolis · 2026
Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
"""

import numpy as np
from scipy.io import wavfile
from IPython.display import Audio, display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── constantes ────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2       # 1.6180339887...
C_PHI      = 1.0 / PHI**2               # 0.3820... curvatura hiperbólica
FS         = 44100
F_BEEP     = 880.0 / 4                  # 220Hz (domínio 0.25×)
F_ORG      = 220.0 / 4                  # 55Hz
F_M        = F_ORG / PHI
BETA_FM    = PHI
DURACAO    = 1.5 * 4                    # 6s
N_STEPS    = 5
ALPHA_STAR = 1.0 / 3.0
CHUNK_SIZE = 2048

print("=" * 62)
print("  AlphaPhi · Eco Hiperbólico v4 · geometria pura + 1 agente")
print("=" * 62)
print(f"\n  PHI           = {PHI:.6f}")
print(f"  C_PHI         = {C_PHI:.6f}  (1/φ²)")
print(f"  φ^1.5 (teto hiperbólico) = {PHI**1.5:.6f}")
print(f"  φ³    (teto euclidiano)  = {PHI**3:.6f}")
print(f"\n  v4: expmap0 → logmap0 apenas (sem eco_ressonante no meio)")
print(f"  1 agente único: agente_eco (20 ciclos)")
print(f"  Hipótese: curvatura hiperbólica sozinha organiza o campo")

# ── funções hiperbólicas ───────────────────────────────────────────────────
def expmap0(v, c=C_PHI):
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.clip(v_norm, 1e-8, None)
    tanh_v = np.tanh(np.clip(np.sqrt(c) * v_norm, -15, 15))
    return tanh_v * v / (np.sqrt(c) * v_norm)

def logmap0(y, c=C_PHI):
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    max_norm = (1.0 / np.sqrt(c)) - 1e-5
    y_norm = np.clip(y_norm, 1e-8, max_norm)
    return np.arctanh(np.clip(np.sqrt(c) * y_norm, -1+1e-8, 1-1e-8)) * y / (np.sqrt(c) * y_norm)

def eco_ressonante(x, phi=PHI, n_eco=3):
    x = np.asarray(x, dtype=float)
    sinal = x.copy()
    for _ in range(n_eco):
        freq      = np.fft.fft(sinal, axis=-1)
        amplitude = np.abs(freq)
        fase      = np.angle(freq)
        nova_fase = fase * phi
        reflexao  = np.real(np.fft.ifft(amplitude * np.exp(1j * nova_fase), axis=-1))
        sinal     = sinal + (reflexao - x) / phi
    return sinal

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
    coh_mem = np.zeros(len(bins_phi))
    cohs_f  = np.zeros(len(bins_phi))
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem, cohs_f = cohs, cohs
        s_e = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()
    return cas, cohs_f

def agente_eco(sinal, bins_phi, n_ciclos=20, expoente=3):
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

print(f"\n  Sinal: {N_SINAL} amostras ({DURACAO:.1f}s)  |  Bandas φ: {N_BANDAS}")

# ══════════════════════════════════════════════════════════════════════════
# BLOCO 1 — EUCLIDIANO (expoente=3)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  BLOCO 1 — Euclidiano  (expoente=3, teto=φ³=4.236)")
print("─" * 62)

beta_euc, cas_euc, traj_euc = agente_eco(x_mix, BINS_PHI, n_ciclos=20, expoente=3)
coh_euc    = medir_coh_sinal(cas_euc[-1], BINS_PHI)
beta_max_e = float(beta_euc.max())
beta_med_e = float(beta_euc.mean())
print(f"  β_max   = {beta_max_e:.6f}  (φ³ = {PHI**3:.6f})")
print(f"  β_med   = {beta_med_e:.6f}")
print(f"  coh_med = {coh_euc:.6f}")
salvar_wav(concatenar(cas_euc, fade=int(0.15*FS)), "beep880_euclid.wav")

# ══════════════════════════════════════════════════════════════════════════
# BLOCO 2 — GEOMETRIA HIPERBÓLICA PURA (sem eco_ressonante)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  BLOCO 2 — Geometria hiperbólica pura: expmap0 → logmap0")
print("  (eco_ressonante removido — 1 agente único: agente_eco)")
print("─" * 62)

n_chunks    = len(x_mix) // CHUNK_SIZE
x_chunks    = x_mix[:n_chunks * CHUNK_SIZE].reshape(n_chunks, CHUNK_SIZE)
norma_antes = float(np.linalg.norm(x_chunks, axis=-1).mean())
x_hyp       = expmap0(x_chunks, c=C_PHI)   # Euclidiano → Hiperbólico
x_euc_geo   = logmap0(x_hyp, c=C_PHI)      # Hiperbólico → Euclidiano (sem eco)
norma_depois = float(np.linalg.norm(x_euc_geo, axis=-1).mean())
print(f"  Norma antes: {norma_antes:.4f}  →  depois: {norma_depois:.4f}  (razão: {norma_depois/norma_antes:.4f})")
print(f"  (v3: ratio=0.3056 com eco — agora sem eco)")

x_pre = normalizar(x_euc_geo.flatten())
if len(x_pre) < len(x_mix):
    x_pre = np.concatenate([x_pre, x_mix[-(len(x_mix)-len(x_pre)):]])
elif len(x_pre) > len(x_mix):
    x_pre = x_pre[:len(x_mix)]

# ══════════════════════════════════════════════════════════════════════════
# BLOCO 3 — HIPERBÓLICO v4 (geometria pura + expoente=1.5)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  BLOCO 3 — Hiperbólico v4  (geometria pura, expoente=1.5)")
print("─" * 62)

beta_hip, cas_hip, traj_hip = agente_eco(x_pre, BINS_PHI, n_ciclos=20, expoente=1.5)
coh_hip    = medir_coh_sinal(cas_hip[-1], BINS_PHI)
beta_max_h = float(beta_hip.max())
beta_med_h = float(beta_hip.mean())
print(f"  β_max   = {beta_max_h:.6f}  (φ^1.5 = {PHI**1.5:.6f})")
print(f"  β_med   = {beta_med_h:.6f}")
print(f"  coh_med = {coh_hip:.6f}")
salvar_wav(concatenar(cas_hip, fade=int(0.15*FS)), "beep880_hiperbolico_v4.wav")

# ══════════════════════════════════════════════════════════════════════════
# TRAJETÓRIA β — 20 ciclos
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print(f"  {'Ciclo':>5}  {'β_max Euc (exp=3)':>18}  {'β_max Hip v3 (exp=1.5)':>22}")
print("─" * 62)
for i, (te, th) in enumerate(zip(traj_euc, traj_hip)):
    print(f"  {i+1:>5}  {te[0]:>18.6f}  {th[0]:>22.6f}")
print(f"  {'Alvo':>5}  {'φ³=4.236068':>18}  {'φ^1.5=2.058171':>22}")

# ══════════════════════════════════════════════════════════════════════════
# RESULTADO
# ══════════════════════════════════════════════════════════════════════════
delta_coh = (coh_hip - coh_euc) / (coh_euc + 1e-10) * 100
print("\n" + "═" * 62)
print("  RESULTADO — Euclidiano × Hiperbólico v4")
print("═" * 62)
print(f"  {'Métrica':<16} {'Euclidiano':>12}  {'Hiperbólico v4':>14}")
print(f"  {'─'*16} {'─'*12}  {'─'*14}")
print(f"  {'β_max':<16} {beta_max_e:>12.6f}  {beta_max_h:>14.6f}")
print(f"  {'β_med':<16} {beta_med_e:>12.6f}  {beta_med_h:>14.6f}")
print(f"  {'coh_med':<16} {coh_euc:>12.6f}  {coh_hip:>14.6f}")
print(f"  {'Δcoh_med':<16} {delta_coh:>+11.2f}%")
print("═" * 62)
print()
if abs(beta_max_h - PHI**1.5) < 0.05:
    print(f"  ★ β_max converge para φ^1.5 = {PHI**1.5:.4f}")
if coh_hip > coh_euc:
    print(f"  ★ Coerência hiperbólica SUPERIOR (+{delta_coh:.2f}%)")
elif coh_hip > 0.087694:
    print(f"  ★ Coerência melhorou em relação ao v2 (0.0877 → {coh_hip:.4f})")
else:
    print(f"  → Coerência: {coh_hip:.6f} — ajuste adicional necessário")

# ══════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
fig.patch.set_facecolor('#0d1117')
C = {'gold':'#DAA520','blue':'#4169E1','red':'#FF4466','text':'#8B949E',
     'title':'#E6EDF3','panel':'#161b22'}
for ax in axes:
    ax.set_facecolor(C['panel'])
    ax.tick_params(colors=C['text'])
    for sp in ax.spines.values(): sp.set_color('#30363d')

ciclos = list(range(1, 21))
axes[0].plot(ciclos, [t[0] for t in traj_euc], color=C['text'], lw=1.5,
             marker='o', ms=4, label='β_max Euclidiano (exp=3)')
axes[0].plot(ciclos, [t[0] for t in traj_hip], color=C['gold'], lw=1.5,
             marker='s', ms=4, label='β_max Hiperbólico v4 (exp=1.5)')
axes[0].axhline(PHI**3,   color=C['red'],  lw=1, ls='--', label=f'φ³={PHI**3:.4f}')
axes[0].axhline(PHI**1.5, color=C['blue'], lw=1, ls=':',  label=f'φ^1.5={PHI**1.5:.4f}')
axes[0].set_title('Trajetória β_max — 20 ciclos', color=C['title'])
axes[0].legend(facecolor='#161b22', labelcolor=C['text'], fontsize=9)
axes[0].set_ylabel('β_max', color=C['text'])

x_band = range(N_BANDAS)
axes[1].bar(x_band, beta_euc, alpha=0.5, color=C['text'],  label=f'β Euc  med={beta_med_e:.4f}')
axes[1].bar(x_band, beta_hip, alpha=0.6, color=C['gold'],  label=f'β Hip v3  med={beta_med_h:.4f}')
axes[1].axhline(PHI**3,   color=C['red'],  lw=1, ls='--', label=f'φ³={PHI**3:.4f}')
axes[1].axhline(PHI**1.5, color=C['blue'], lw=1, ls=':',  label=f'φ^1.5={PHI**1.5:.4f}')
axes[1].set_title('β por banda φ', color=C['title'])
axes[1].legend(facecolor='#161b22', labelcolor=C['text'], fontsize=9)
axes[1].set_ylabel('β', color=C['text'])

n_plot = int(0.2 * FS)
tempo  = np.arange(n_plot) / FS * 1000
axes[2].plot(tempo, cas_euc[-1][:n_plot], color=C['text'], lw=0.8, alpha=0.7,
             label=f'Euclidiano  coh={coh_euc:.4f}')
axes[2].plot(tempo, cas_hip[-1][:n_plot], color=C['gold'], lw=0.8, alpha=0.9,
             label=f'Hiperbólico v4  coh={coh_hip:.4f}')
axes[2].set_title('Sinal final — primeiros 200ms', color=C['title'])
axes[2].legend(facecolor='#161b22', labelcolor=C['text'], fontsize=9)
axes[2].set_xlabel('Tempo (ms)', color=C['text'])
axes[2].set_ylabel('Amplitude', color=C['text'])

plt.suptitle(f'AlphaPhi · Eco Hiperbólico v4 · exp=1.5 · teto=φ^1.5={PHI**1.5:.4f}',
             color=C['title'], fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('eco_hiperbolico_v4.png', dpi=120, bbox_inches='tight', facecolor='#0d1117')
print("\n  Gráfico: eco_hiperbolico_v4.png")

print("\nPlayback — Euclidiano:")
display(Audio("beep880_euclid.wav"))
print("\nPlayback — Hiperbólico v4 (exp=1.5):")
display(Audio("beep880_hiperbolico_v4.wav"))
print("\nConcluído.")
