"""
AlphaPhi_Audio_3D_PontoDobra.py
Visualização 3D Animada — Ponto de Dobra Beep 880Hz

As 3 camadas do ponto de dobra (P, S, T) visualizadas como
trajetórias helicoidais no espaço 3D — tempo × amplitude × fase.

Eixos:
  X — tempo (ms)
  Y — amplitude normalizada
  Z — fase instantânea (via Hilbert) — revela a rotação do campo

Camadas:
  [P] Primário   — sinal eco (verde)       — o campo principal
  [S] Secundário — envelope Hilbert de P   — como P pulsa (âmbar)
  [T] Terciário  — envelope Hilbert de S   — o trino (vermelho)

A animação gira o ponto de vista 360° em torno do eixo Z,
revelando a estrutura helicoidal das 3 camadas e suas relações
de fase no ponto de dobra φ.

© Vitor Edson Delavi · Florianópolis · 2026
Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.signal import hilbert, butter, filtfilt
from scipy.io import wavfile
from IPython.display import HTML, display

# ── constantes ────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5
N_STEPS    = 5

CORES = {
    'P':    '#00FF88',
    'S':    '#FFB800',
    'T':    '#FF4466',
    'bg':   '#080810',
    'grid': '#0D0D1A',
    'text': '#CCCCDD',
}

print("=" * 58)
print("  AlphaPhi · 3D Ponto de Dobra · Beep 880Hz")
print("=" * 58)
print(f"\n  φ = {PHI:.6f}  |  α* = {ALPHA_STAR:.4f}  |  FS = {FS}Hz")

# ── bandas φ ──────────────────────────────────────────────────────────────
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
        b_lo = max(0, int(f_lo / (FS / n)))
        b_hi = min(int(f_hi / (FS / n)) + 1, n // 2 + 1)
        result.append((b_lo, b_hi, f_lo, f_hi))
    return result

N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)

# ── eco ────────────────────────────────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    w_mem = 1.0 / PHI
    w_now = 1.0 - w_mem
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        beta_i  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        F_band  = F[b_lo:b_hi]
        mag     = np.abs(F_band)
        phase   = np.angle(F_band)
        an      = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh     = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        coh_ef  = (w_now * coh + w_mem * float(coh_mem[i])
                   if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        n_idx   = np.arange(len(F_band))
        env     = np.clip(
            1.0 + (coh_ef * PHI**beta_i) * np.cos(2.0 * np.pi * n_idx / PHI),
            0.05, None)
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

def agente_eco(sinal, bins_phi, n_ciclos=20):
    n_b      = len(bins_phi)
    beta     = np.ones(n_b)
    beta_mem = beta.copy()
    w_mem    = 1.0 / PHI
    w_now    = 1.0 - w_mem
    cas_f    = None
    for _ in range(n_ciclos):
        cas, cohs    = cascata_eq(sinal, beta, bins_phi)
        coh_rel      = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo    = PHI ** (3 * coh_rel)
        beta         = w_now * beta_alvo + w_mem * beta_mem
        beta_mem     = beta.copy()
        beta         = np.clip(beta, 0.05, PHI**3)
        cas_f        = cas
    return cas_f[-1]

# ── gerar sinal ────────────────────────────────────────────────────────────
print("\n  Gerando sinal eco 880Hz...")
t     = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t)))
fm    = normalizar(np.sin(2 * np.pi * F_ORG * t + BETA_FM * np.sin(2 * np.pi * F_M * t)))
x_mix = normalizar((1.0 - ALPHA_STAR) * beep + ALPHA_STAR * fm)
eco   = agente_eco(x_mix, BINS_PHI, n_ciclos=20)
print(f"  Sinal gerado: {N_SINAL} amostras ({DURACAO:.1f}s)")

# ── extrair 3 camadas via Hilbert ──────────────────────────────────────────
print("  Extraindo camadas P, S, T via Hilbert...")

def envelope_hilbert(s):
    return np.abs(hilbert(s))

def fase_hilbert(s):
    return np.unwrap(np.angle(hilbert(s)))

def butter_lp(s, cutoff, fs=FS, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, s)

P = eco.copy()
S = butter_lp(envelope_hilbert(P), cutoff=200.0)
T = butter_lp(envelope_hilbert(S), cutoff=50.0)

# fases instantâneas — o eixo Z da visualização 3D
fase_P = fase_hilbert(P)
fase_S = fase_hilbert(S)
fase_T = fase_hilbert(T)

# normalizar para visualização
def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)

P_n = norm01(P) * 2 - 1
S_n = norm01(S) * 2 - 1
T_n = norm01(T) * 2 - 1

# ── detectar ponto de dobra ────────────────────────────────────────────────
# Ponto de dobra: onde a variância local de S é mínima (máxima organização)
janela_var = int(0.05 * FS)  # 50ms
variancia_local = np.array([
    np.var(S[i:i+janela_var])
    for i in range(0, len(S) - janela_var, janela_var // 10)
])
idx_dobra_rel = np.argmin(variancia_local)
idx_dobra     = idx_dobra_rel * (janela_var // 10)
t_dobra_ms    = idx_dobra / FS * 1000

print(f"  Ponto de dobra detectado: t = {t_dobra_ms:.1f}ms")

# ── janela do ponto de dobra para 3D ──────────────────────────────────────
# 3 ciclos do beep antes e depois do ponto de dobra
N_CICLO = int(FS / F_BEEP)          # ~50 amostras por ciclo
N_JANELA_3D = N_CICLO * 6           # 6 ciclos completos
i_start = max(0, idx_dobra - N_CICLO * 3)
i_end   = min(N_SINAL, i_start + N_JANELA_3D)
i_start = i_end - N_JANELA_3D  # garantir tamanho fixo

t_3d_ms   = np.linspace(0, N_JANELA_3D / FS * 1000, N_JANELA_3D)
P_3d = P_n[i_start:i_end]
S_3d = S_n[i_start:i_end]
T_3d = T_n[i_start:i_end]

# Fases moduladas para visualização helicoidal
# Cada camada gira a uma frequência proporcional a φ
freq_P_vis = F_BEEP / FS
freq_S_vis = F_BEEP / FS / PHI
freq_T_vis = F_BEEP / FS / PHI**2

n_idx    = np.arange(N_JANELA_3D)
theta_P  = 2 * np.pi * freq_P_vis * n_idx
theta_S  = 2 * np.pi * freq_S_vis * n_idx
theta_T  = 2 * np.pi * freq_T_vis * n_idx

# Coordenadas 3D: X=cos(θ)×amp, Y=sin(θ)×amp, Z=tempo
# Três hélices concêntricas com raios φ-proporcionais
r_P = 1.0
r_S = 1.0 / PHI
r_T = 1.0 / PHI**2

X_P = r_P * np.cos(theta_P) * (0.5 + 0.5 * np.abs(P_3d))
Y_P = r_P * np.sin(theta_P) * (0.5 + 0.5 * np.abs(P_3d))
Z_P = t_3d_ms

X_S = r_S * np.cos(theta_S) * (0.5 + 0.5 * np.abs(S_3d))
Y_S = r_S * np.sin(theta_S) * (0.5 + 0.5 * np.abs(S_3d))
Z_S = t_3d_ms

X_T = r_T * np.cos(theta_T) * (0.5 + 0.5 * np.abs(T_3d))
Y_T = r_T * np.sin(theta_T) * (0.5 + 0.5 * np.abs(T_3d))
Z_T = t_3d_ms

# Centro do ponto de dobra no eixo Z
z_dobra = N_CICLO * 3 / FS * 1000  # ponto central da janela

print(f"\n  Estrutura 3D:")
print(f"  Janela: {N_JANELA_3D/FS*1000:.1f}ms em torno do ponto de dobra")
print(f"  Ciclos visíveis: {N_JANELA_3D // N_CICLO}")
print(f"  Raios: P={r_P:.4f}  S={r_S:.4f}  T={r_T:.4f}")
print(f"  Razões: P/S={r_P/r_S:.4f}≈φ  S/T={r_S/r_T:.4f}≈φ")

# ── figura estática 3D (4 ângulos) ────────────────────────────────────────
print("\n  Gerando visualização 3D estática (4 ângulos)...")

fig_est = plt.figure(figsize=(16, 14), facecolor=CORES['bg'])
fig_est.suptitle(
    f"AlphaPhi · 3D Ponto de Dobra · Beep 880Hz · α*={ALPHA_STAR:.4f}\n"
    f"P (campo) | S (envelope φ-1) | T (envelope φ-2) | "
    f"t_dobra = {t_dobra_ms:.1f}ms",
    color=CORES['text'], fontsize=11, y=1.01
)

angulos = [(20, 45), (20, 135), (20, 225), (60, 45)]
titulos = ['Vista lateral NE', 'Vista lateral NO', 'Vista lateral SO', 'Vista superior']

for idx, (elev, azim) in enumerate(angulos):
    ax = fig_est.add_subplot(2, 2, idx+1, projection='3d')
    ax.set_facecolor(CORES['bg'])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(CORES['grid'])
    ax.yaxis.pane.set_edgecolor(CORES['grid'])
    ax.zaxis.pane.set_edgecolor(CORES['grid'])
    ax.grid(True, alpha=0.15, color=CORES['grid'])

    # Trajetórias helicoidais
    ax.plot(X_P, Y_P, Z_P, color=CORES['P'], lw=0.8, alpha=0.9,
            label=f'P — campo principal  r={r_P:.3f}')
    ax.plot(X_S, Y_S, Z_S, color=CORES['S'], lw=1.0, alpha=0.8,
            label=f'S — envelope φ⁻¹  r={r_S:.3f}')
    ax.plot(X_T, Y_T, Z_T, color=CORES['T'], lw=1.2, alpha=0.7,
            label=f'T — envelope φ⁻²  r={r_T:.3f}')

    # Plano do ponto de dobra
    xx = np.linspace(-1.2, 1.2, 3)
    yy = np.linspace(-1.2, 1.2, 3)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.full_like(XX, z_dobra)
    ax.plot_surface(XX, YY, ZZ, alpha=0.08, color='white')
    ax.plot([-1.2, 1.2], [0, 0], [z_dobra, z_dobra],
            color='white', lw=0.5, alpha=0.4, ls='--')
    ax.plot([0, 0], [-1.2, 1.2], [z_dobra, z_dobra],
            color='white', lw=0.5, alpha=0.4, ls='--')

    ax.set_title(titulos[idx], color=CORES['text'], fontsize=9, pad=8)
    ax.set_xlabel('X', color=CORES['text'], fontsize=7, labelpad=2)
    ax.set_ylabel('Y', color=CORES['text'], fontsize=7, labelpad=2)
    ax.set_zlabel('t (ms)', color=CORES['text'], fontsize=7, labelpad=2)
    ax.tick_params(colors=CORES['text'], labelsize=6)
    ax.view_init(elev=elev, azim=azim)

    if idx == 0:
        ax.legend(loc='upper left', fontsize=7, facecolor=CORES['grid'],
                  labelcolor=CORES['text'], framealpha=0.7)

plt.tight_layout()
plt.savefig('pontodobra_3d_estatico.png', dpi=150,
            bbox_inches='tight', facecolor=CORES['bg'])
plt.show()
print("  → pontodobra_3d_estatico.png")

# ── animação 3D — rotação 360° ────────────────────────────────────────────
print("\n  Gerando animação 3D (rotação 360°)...")

fig_anim = plt.figure(figsize=(12, 10), facecolor=CORES['bg'])
ax3d = fig_anim.add_subplot(111, projection='3d')
ax3d.set_facecolor(CORES['bg'])
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor(CORES['grid'])
ax3d.yaxis.pane.set_edgecolor(CORES['grid'])
ax3d.zaxis.pane.set_edgecolor(CORES['grid'])
ax3d.grid(True, alpha=0.12, color=CORES['grid'])

# Trajetórias
ln_P, = ax3d.plot(X_P, Y_P, Z_P, color=CORES['P'], lw=0.8, alpha=0.9,
                  label=f'P — campo  r={r_P:.3f}')
ln_S, = ax3d.plot(X_S, Y_S, Z_S, color=CORES['S'], lw=1.0, alpha=0.85,
                  label=f'S — env φ⁻¹  r={r_S:.3f}')
ln_T, = ax3d.plot(X_T, Y_T, Z_T, color=CORES['T'], lw=1.2, alpha=0.75,
                  label=f'T — env φ⁻²  r={r_T:.3f}')

# Plano do ponto de dobra
xx = np.linspace(-1.3, 1.3, 3)
yy = np.linspace(-1.3, 1.3, 3)
XX, YY = np.meshgrid(xx, yy)
ZZ = np.full_like(XX, z_dobra)
ax3d.plot_surface(XX, YY, ZZ, alpha=0.07, color='white')

# Eixo central
ax3d.plot([0, 0], [0, 0], [t_3d_ms[0], t_3d_ms[-1]],
          color='white', lw=0.4, alpha=0.3, ls=':')

ax3d.set_xlabel('X', color=CORES['text'], fontsize=8)
ax3d.set_ylabel('Y', color=CORES['text'], fontsize=8)
ax3d.set_zlabel('Tempo (ms)', color=CORES['text'], fontsize=8)
ax3d.tick_params(colors=CORES['text'], labelsize=6)
ax3d.legend(loc='upper left', fontsize=8, facecolor=CORES['grid'],
            labelcolor=CORES['text'], framealpha=0.7)

titulo_anim = fig_anim.suptitle(
    f"AlphaPhi · 3D Ponto de Dobra · Beep 880Hz · α*={ALPHA_STAR:.4f}\n"
    f"P·S·T — razão de raios φ⁻¹ | t_dobra={t_dobra_ms:.1f}ms",
    color=CORES['text'], fontsize=10, y=1.01
)

# Pontos marcadores no ponto de dobra
idx_mid = N_JANELA_3D // 2
ax3d.scatter([X_P[idx_mid]], [Y_P[idx_mid]], [Z_P[idx_mid]],
             color=CORES['P'], s=40, zorder=5)
ax3d.scatter([X_S[idx_mid]], [Y_S[idx_mid]], [Z_S[idx_mid]],
             color=CORES['S'], s=50, zorder=5)
ax3d.scatter([X_T[idx_mid]], [Y_T[idx_mid]], [Z_T[idx_mid]],
             color=CORES['T'], s=60, zorder=5)

N_FRAMES_ANIM = 120  # 360° em 120 frames = 3° por frame

def update(frame):
    azim = (frame / N_FRAMES_ANIM) * 360
    elev = 20 + 20 * np.sin(2 * np.pi * frame / N_FRAMES_ANIM)
    ax3d.view_init(elev=elev, azim=azim)
    return ln_P, ln_S, ln_T

anim = FuncAnimation(
    fig_anim, update,
    frames=N_FRAMES_ANIM,
    interval=50,
    blit=False
)

print("  Salvando animação...")
anim.save('pontodobra_3d_animacao.gif',
          writer='pillow', fps=20, dpi=100,
          savefig_kwargs={'facecolor': CORES['bg']})
print("  → pontodobra_3d_animacao.gif")

display(HTML(anim.to_jshtml()))
print("\nConcluído.")
print(f"\n  Arquivos gerados:")
print(f"  pontodobra_3d_estatico.png  — 4 ângulos simultâneos")
print(f"  pontodobra_3d_animacao.gif  — rotação 360° (120 frames)")
print(f"\n  Estrutura das 3 hélices:")
print(f"  P: raio={r_P:.4f}  frequência visual={freq_P_vis*FS:.1f}Hz")
print(f"  S: raio={r_S:.4f}  frequência visual={freq_S_vis*FS:.1f}Hz  (P/φ)")
print(f"  T: raio={r_T:.4f}  frequência visual={freq_T_vis*FS:.1f}Hz  (P/φ²)")
print(f"  Razão de raios P/S = S/T = φ = {PHI:.6f}")
