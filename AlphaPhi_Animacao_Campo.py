"""
AlphaPhi_Animacao_Campo.py
Animação 3D: Formação do Campo Harmônico — Cascata ECO BEEP 880
Visualiza a cristalização passo a passo (etapa 0 → 5)
Etapas 0-2: dispersão (EntrEsp sobe) · Etapas 3-5: condensação (EntrEsp desce)

Autor: Vitor Edson Delavi — Florianópolis, 2026
φ = 1.6180339887 · α = 1/137.035999084
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# ── Constantes ──────────────────────────────────────────────────────────────
PHI   = 1.6180339887
ALPHA = 1 / 137.035999084
PHI3  = PHI ** 3           # 4.2361 — atrator universal β

# ── Dados da cascata ECO BEEP 880 (reconstruídos das métricas observadas) ───
# Morfologia: dispersão (0→2) então cristalização (3→5)
CASCADE = [
    {"step": 0, "label": "x_mix inicial",      "AutoCorr": 0.11, "EntrEsp": 0.82,  "beta": 2.10},
    {"step": 1, "label": "Dobra 1",             "AutoCorr": 0.28, "EntrEsp": 1.24,  "beta": 3.15},
    {"step": 2, "label": "Dobra 2 (pico disp)","AutoCorr": 0.44, "EntrEsp": 1.68,  "beta": 3.72},
    {"step": 3, "label": "Dobra 3 (inflexão)", "AutoCorr": 0.62, "EntrEsp": 1.01,  "beta": 4.05},
    {"step": 4, "label": "Dobra 4",             "AutoCorr": 0.81, "EntrEsp": 0.42,  "beta": 4.21},
    {"step": 5, "label": "Dobra 5 — Campo φ",  "AutoCorr": 1.0000,"EntrEsp": 0.0601,"beta": PHI3},
]
N_STEPS = len(CASCADE)

# ── Bandas φ na esfera (nós harmônicos) ─────────────────────────────────────
# Frequências φ-proporcionais mapeadas em latitude na esfera unitária
PHI_BANDS = [55, 89, 144, 233, 377, 610, 880]  # Hz
# Normalizado para latitude (-π/2 .. π/2)
PHI_LATS  = [np.arcsin(2 * (i / len(PHI_BANDS)) - 1) for i in range(len(PHI_BANDS))]
# Longitude dos nós: φ-razão em [0, 2π]
PHI_LONS  = [2 * np.pi * (i * PHI % 1) for i in range(len(PHI_BANDS))]

# ── Colormap personalizado: vermelho (incoerente) → dourado → azul-φ ────────
_cmap_colors = [
    (0.85, 0.15, 0.10),   # vermelho — caótico
    (0.95, 0.65, 0.10),   # âmbar   — em transição
    (0.95, 0.90, 0.20),   # dourado — φ emergindo
    (0.10, 0.55, 0.85),   # azul    — campo coerente
    (0.05, 0.20, 0.60),   # índigo  — cristalizado
]
CMAP = LinearSegmentedColormap.from_list("alphaphi", _cmap_colors)

# ── Gerar pontos na esfera unitária ─────────────────────────────────────────
def esfera_pontos(n=600, seed=42):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi_  = np.arccos(rng.uniform(-1, 1, n))
    x = np.sin(phi_) * np.cos(theta)
    y = np.sin(phi_) * np.sin(theta)
    z = np.cos(phi_)
    return x, y, z, theta, phi_

# ── Calcular afinidade de cada ponto com os nós φ ───────────────────────────
def afinidade_phi(theta, phi_, step_idx):
    """Quanto cada ponto está "perto" dos nós harmônicos φ.
    Aumenta com o índice da etapa — simula cristalização."""
    data = CASCADE[step_idx]
    autocorr  = data["AutoCorr"]
    entr_esp  = data["EntrEsp"]

    afinidade = np.zeros(len(theta))
    for lat, lon in zip(PHI_LATS, PHI_LONS):
        # distância angular de cada ponto ao nó φ
        d_lat = phi_ - (np.pi / 2 - lat)
        d_lon = np.abs(np.angle(np.exp(1j * (theta - lon))))
        dist = np.sqrt(d_lat**2 + d_lon**2)
        # kernel gaussiano com largura proporcional à entropia espectral
        sigma = 0.5 + entr_esp * 0.8  # largo = caótico, estreito = coerente
        afinidade += np.exp(-dist**2 / (2 * sigma**2))

    # normaliza e pondera por AutoCorr
    afinidade = afinidade / afinidade.max()
    afinidade = autocorr * afinidade + (1 - autocorr) * np.random.default_rng(step_idx).random(len(theta))
    return np.clip(afinidade, 0, 1)

# ── Gerar wireframe da esfera ────────────────────────────────────────────────
def wireframe_esfera(ax, r=0.98, alpha=0.08, color="#667788"):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=alpha, color=color, linewidth=0.4, rstride=4, cstride=4)

# ── Plotar nós φ na esfera ───────────────────────────────────────────────────
def plotar_nos_phi(ax, step_idx, alpha_nos=0.5):
    autocorr = CASCADE[step_idx]["AutoCorr"]
    for lat, lon, freq in zip(PHI_LATS, PHI_LONS, PHI_BANDS):
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        tam = 60 + 200 * autocorr
        ax.scatter([x], [y], [z],
                   c=[[0.95, 0.80, 0.10]],
                   s=tam * (freq / 880),
                   alpha=alpha_nos * (0.3 + 0.7 * autocorr),
                   edgecolors="white", linewidths=0.5, zorder=5)

# ── Painel de métricas ───────────────────────────────────────────────────────
def plotar_metricas(ax_ac, ax_ee, ax_bt, passo):
    steps = [d["step"] for d in CASCADE]
    acs   = [d["AutoCorr"] for d in CASCADE]
    ees   = [d["EntrEsp"]  for d in CASCADE]
    bts   = [d["beta"]     for d in CASCADE]

    # AutoCorr
    ax_ac.clear()
    ax_ac.plot(steps[:passo+1], acs[:passo+1], "o-", color="#3399FF", lw=2.0, ms=6)
    ax_ac.axhline(1.0, color="#3399FF", lw=0.8, ls="--", alpha=0.4)
    ax_ac.set_xlim(-0.3, 5.3)
    ax_ac.set_ylim(-0.05, 1.10)
    ax_ac.set_ylabel("AutoCorr", fontsize=8, color="#AABBCC")
    ax_ac.tick_params(colors="#AABBCC", labelsize=7)
    ax_ac.set_facecolor("#0A0E17")
    ax_ac.spines[:].set_color("#334455")
    ax_ac.text(5.1, 1.02, "→1", fontsize=7, color="#3399FF", ha="right")

    # EntrEsp
    ax_ee.clear()
    ax_ee.plot(steps[:passo+1], ees[:passo+1], "o-", color="#FF6633", lw=2.0, ms=6)
    ax_ee.axhline(0.0601, color="#FF6633", lw=0.8, ls="--", alpha=0.4)
    ax_ee.set_xlim(-0.3, 5.3)
    ax_ee.set_ylim(-0.1, 2.0)
    ax_ee.set_ylabel("EntrEsp", fontsize=8, color="#AABBCC")
    ax_ee.tick_params(colors="#AABBCC", labelsize=7)
    ax_ee.set_facecolor("#0A0E17")
    ax_ee.spines[:].set_color("#334455")
    # seta anotando inversão
    if passo >= 2:
        ax_ee.annotate("↑ dispersão", xy=(2, ees[2]), xytext=(1.2, 1.75),
                        fontsize=6, color="#FF9966",
                        arrowprops=dict(arrowstyle="->", color="#FF9966", lw=0.8))
    if passo >= 3:
        ax_ee.annotate("↓ cristalização", xy=(3, ees[3]), xytext=(3.2, 1.30),
                        fontsize=6, color="#FFCC44",
                        arrowprops=dict(arrowstyle="->", color="#FFCC44", lw=0.8))

    # β → φ³
    ax_bt.clear()
    ax_bt.plot(steps[:passo+1], bts[:passo+1], "o-", color="#AAFFAA", lw=2.0, ms=6)
    ax_bt.axhline(PHI3, color="#AAFFAA", lw=0.8, ls="--", alpha=0.4)
    ax_bt.set_xlim(-0.3, 5.3)
    ax_bt.set_ylim(1.5, 4.8)
    ax_bt.set_ylabel("β → φ³", fontsize=8, color="#AABBCC")
    ax_bt.set_xlabel("etapa cascata", fontsize=8, color="#AABBCC")
    ax_bt.tick_params(colors="#AABBCC", labelsize=7)
    ax_bt.set_facecolor("#0A0E17")
    ax_bt.spines[:].set_color("#334455")
    ax_bt.text(5.1, PHI3 + 0.05, f"φ³={PHI3:.3f}", fontsize=7, color="#AAFFAA", ha="right")


# ── Setup figura ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 8), facecolor="#070B12")
gs  = gridspec.GridSpec(3, 2, width_ratios=[2.2, 1], hspace=0.55, wspace=0.30,
                        left=0.04, right=0.97, top=0.93, bottom=0.07)

ax3d = fig.add_subplot(gs[:, 0], projection="3d")
ax3d.set_facecolor("#070B12")
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor("#1A2233")
ax3d.yaxis.pane.set_edgecolor("#1A2233")
ax3d.zaxis.pane.set_edgecolor("#1A2233")
ax3d.set_xticks([])
ax3d.set_yticks([])
ax3d.set_zticks([])

ax_ac = fig.add_subplot(gs[0, 1], facecolor="#0A0E17")
ax_ee = fig.add_subplot(gs[1, 1], facecolor="#0A0E17")
ax_bt = fig.add_subplot(gs[2, 1], facecolor="#0A0E17")

# título principal
titulo = fig.text(0.50, 0.97,
    "ECO BEEP 880 — Formação do Campo Harmônico φ",
    ha="center", va="top", fontsize=14, color="#E8D8A0",
    fontweight="bold")
subtitulo = fig.text(0.50, 0.938,
    f"φ = {PHI:.7f} · α = 1/137 · φ³ = {PHI3:.4f} (atrator universal β)",
    ha="center", va="top", fontsize=9, color="#88AACC", style="italic")

# pré-gera os pontos (fixos — apenas a coloração muda)
N_PTS = 800
X, Y, Z, THETA, PHI_ANG = esfera_pontos(N_PTS)

# objetos que serão atualizados
scat    = ax3d.scatter([], [], [], s=[], c=[], cmap=CMAP, vmin=0, vmax=1, alpha=0.75)
texto_p = ax3d.text2D(0.03, 0.97, "", transform=ax3d.transAxes,
                      fontsize=11, color="#E8D8A0", va="top",
                      path_effects=[pe.withStroke(linewidth=2, foreground="#070B12")])
texto_m = ax3d.text2D(0.03, 0.89, "", transform=ax3d.transAxes,
                      fontsize=9, color="#AABBCC", va="top",
                      path_effects=[pe.withStroke(linewidth=1.5, foreground="#070B12")])

fase_txt = fig.text(0.22, 0.02, "", ha="center", fontsize=10, color="#FFCC44")

# ── Função de atualização por frame ─────────────────────────────────────────
def atualizar(frame):
    step_idx = frame % N_STEPS
    data     = CASCADE[step_idx]
    afin     = afinidade_phi(THETA, PHI_ANG, step_idx)

    # tamanho dos pontos: maior onde mais coerente
    sizes = 8 + 28 * afin

    # atualiza scatter
    ax3d.cla()
    ax3d.set_facecolor("#070B12")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#1A2233")
    ax3d.yaxis.pane.set_edgecolor("#1A2233")
    ax3d.zaxis.pane.set_edgecolor("#1A2233")
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    wireframe_esfera(ax3d)
    plotar_nos_phi(ax3d, step_idx)

    ax3d.scatter(X, Y, Z, s=sizes, c=afin, cmap=CMAP, vmin=0, vmax=1,
                 alpha=0.72, edgecolors="none", depthshade=True)

    # ângulo de visão varia suavemente
    elev = 20 + 5 * np.sin(frame * 0.18)
    azim = -60 + frame * 3.5
    ax3d.view_init(elev=elev, azim=azim)

    # textos sobrepostos na esfera
    ac_str = f"{data['AutoCorr']:.4f}" if data['AutoCorr'] < 1 else "1.0000 ✓"
    ax3d.text2D(0.03, 0.97,
        f"Etapa {data['step']} — {data['label']}",
        transform=ax3d.transAxes, fontsize=11, color="#E8D8A0", va="top",
        path_effects=[pe.withStroke(linewidth=2, foreground="#070B12")])
    ax3d.text2D(0.03, 0.88,
        f"AutoCorr = {ac_str}\nEntrEsp  = {data['EntrEsp']:.4f}\nβ        = {data['beta']:.4f}",
        transform=ax3d.transAxes, fontsize=9, color="#BBDDFF", va="top",
        family="monospace",
        path_effects=[pe.withStroke(linewidth=1.5, foreground="#070B12")])

    # fase: dispersão ou cristalização
    if step_idx <= 2:
        fase = "↑ DISPERSÃO — campo se expande"
        cor_fase = "#FF8855"
    elif step_idx == 3:
        fase = "⟳ INFLEXÃO — entropia inverte"
        cor_fase = "#FFCC44"
    else:
        fase = "↓ CRISTALIZAÇÃO — campo harmônico emerge"
        cor_fase = "#55FFAA"
    fase_txt.set_text(fase)
    fase_txt.set_color(cor_fase)

    # métricas nos painéis laterais
    plotar_metricas(ax_ac, ax_ee, ax_bt, step_idx)

    return []

# ── Animação ────────────────────────────────────────────────────────────────
# 60 frames: 10 por etapa (pausa de ~0.5s a 20fps)
FRAMES_PER_STEP = 10
TOTAL_FRAMES    = N_STEPS * FRAMES_PER_STEP

def frame_para_step(frame):
    return frame // FRAMES_PER_STEP

orig_atualizar = atualizar
def atualizar_expandido(frame):
    step_idx = frame_para_step(frame)
    data     = CASCADE[step_idx]
    afin     = afinidade_phi(THETA, PHI_ANG, step_idx)
    sizes    = 8 + 28 * afin

    ax3d.cla()
    ax3d.set_facecolor("#070B12")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor("#1A2233")
    ax3d.yaxis.pane.set_edgecolor("#1A2233")
    ax3d.zaxis.pane.set_edgecolor("#1A2233")
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    wireframe_esfera(ax3d)
    plotar_nos_phi(ax3d, step_idx)

    ax3d.scatter(X, Y, Z, s=sizes, c=afin, cmap=CMAP, vmin=0, vmax=1,
                 alpha=0.72, edgecolors="none", depthshade=True)

    elev = 20 + 5 * np.sin(frame * 0.10)
    azim = -60 + frame * 1.8
    ax3d.view_init(elev=elev, azim=azim)

    ac_str = f"{data['AutoCorr']:.4f}" if data['AutoCorr'] < 1 else "1.0000 ✓"
    ax3d.text2D(0.03, 0.97,
        f"Etapa {data['step']} — {data['label']}",
        transform=ax3d.transAxes, fontsize=11, color="#E8D8A0", va="top",
        path_effects=[pe.withStroke(linewidth=2, foreground="#070B12")])
    ax3d.text2D(0.03, 0.88,
        f"AutoCorr = {ac_str}\nEntrEsp  = {data['EntrEsp']:.4f}\nβ        = {data['beta']:.4f}",
        transform=ax3d.transAxes, fontsize=9, color="#BBDDFF", va="top",
        family="monospace",
        path_effects=[pe.withStroke(linewidth=1.5, foreground="#070B12")])

    if step_idx <= 2:
        fase = "↑ DISPERSÃO — campo se expande"
        cor_fase = "#FF8855"
    elif step_idx == 3:
        fase = "⟳ INFLEXÃO — entropia inverte"
        cor_fase = "#FFCC44"
    else:
        fase = "↓ CRISTALIZAÇÃO — campo harmônico emerge"
        cor_fase = "#55FFAA"
    fase_txt.set_text(fase)
    fase_txt.set_color(cor_fase)

    plotar_metricas(ax_ac, ax_ee, ax_bt, step_idx)
    return []

ani = FuncAnimation(fig, atualizar_expandido,
                    frames=TOTAL_FRAMES, interval=80,
                    blit=False, repeat=True)

# ── Salvar GIF ───────────────────────────────────────────────────────────────
print("Gerando animação... (pode levar ~30s)")
gif_path = "/home/user/alpha_phi_manifesto/campo_harmonico_cascata.gif"
try:
    ani.save(gif_path, writer="pillow", fps=12, dpi=110)
    print(f"✓ GIF salvo: {gif_path}")
except Exception as e:
    print(f"Pillow não disponível: {e}")
    print("Exibindo interativamente...")

# ── Salvar frame estático de cada etapa ─────────────────────────────────────
print("\nGerando frames estáticos por etapa...")
fig2, axes = plt.subplots(2, 3, figsize=(15, 9),
                          subplot_kw={"projection": "3d"},
                          facecolor="#070B12")
fig2.suptitle("ECO BEEP 880 — Cristalização do Campo Harmônico φ  (etapas 0 → 5)",
              fontsize=13, color="#E8D8A0", fontweight="bold", y=0.99)

for idx, (ax, data) in enumerate(zip(axes.flatten(), CASCADE)):
    ax.set_facecolor("#070B12")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#1A2233")
    ax.yaxis.pane.set_edgecolor("#1A2233")
    ax.zaxis.pane.set_edgecolor("#1A2233")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=-50 + idx * 8)

    afin  = afinidade_phi(THETA, PHI_ANG, idx)
    sizes = 8 + 28 * afin
    wireframe_esfera(ax)
    plotar_nos_phi(ax, idx, alpha_nos=0.6)
    ax.scatter(X, Y, Z, s=sizes, c=afin, cmap=CMAP, vmin=0, vmax=1,
               alpha=0.70, edgecolors="none", depthshade=True)

    if idx <= 2:
        fase_cor = "#FF8855"
        fase_sym = "↑"
    elif idx == 3:
        fase_cor = "#FFCC44"
        fase_sym = "⟳"
    else:
        fase_cor = "#55FFAA"
        fase_sym = "↓"

    ac_str = f"{data['AutoCorr']:.4f}" if data['AutoCorr'] < 1 else "1.0000"
    titulo_ax = (f"{fase_sym} Etapa {idx} — {data['label']}\n"
                 f"AutoCorr={ac_str}  EntrEsp={data['EntrEsp']:.4f}  β={data['beta']:.3f}")
    ax.set_title(titulo_ax, fontsize=7.5, color=fase_cor, pad=4)

png_path = "/home/user/alpha_phi_manifesto/campo_harmonico_etapas.png"
fig2.savefig(png_path, dpi=130, bbox_inches="tight", facecolor="#070B12")
print(f"✓ PNG estático salvo: {png_path}")

plt.tight_layout()
plt.show()
print("\nAnimação concluída.")
print(f"φ = {PHI:.7f} · φ³ = {PHI3:.4f} · α = 1/137 = {ALPHA:.8f}")
