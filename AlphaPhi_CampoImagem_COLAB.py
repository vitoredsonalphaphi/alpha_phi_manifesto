# ALPHA PHI — φ-Rendering de Contornos
# Vitor Edson Delavi · Florianópolis · 2026
#
# O que este experimento faz:
# ─────────────────────────────────────────────────────────────────
# BIP 880Hz → aplicou estrutura φ ao sinal de áudio.
# Eco_text  → aplicou BIP a cada caractere (o campo existe, mas
#             o pixel euclidiano bloqueia a expressão visual).
# Aqui      → aplica transição φ-ponderada nos pixels de borda
#             dos contornos orgânicos da imagem.
#
# O pixel continua retangular. O que muda é a DISTRIBUIÇÃO
# DE INTENSIDADES nos pixels de borda — seguindo proporção φ.
#
# Pergunta: a transição φ é mais ergonômica (menos serrilhada)
# do que a Gaussiana convencional?
# ─────────────────────────────────────────────────────────────────

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "Pillow", "requests", "scipy"], check=True)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from scipy.ndimage import sobel, convolve1d

PHI   = (1 + np.sqrt(5)) / 2
C_PHI = 1.0 / PHI**2

print(f"φ         = {PHI:.6f}")
print(f"1/φ       = {1/PHI:.6f}  (decaimento por passo)")
print(f"1/φ²      = {C_PHI:.6f}  (curvatura)")
print("=" * 60)
print("ALPHA PHI — φ-Rendering de Contornos Orgânicos")
print("=" * 60)

# ── Kernels de transição ───────────────────────────────────────────────────────

def kernel_phi(raio=6):
    """
    Kernel φ-ponderado: w_n = 1/φ^n
    Mesmo princípio do BIP — cada passo de distância
    recebe peso 1/φ^n (decaimento natural de Fibonacci).
    """
    pesos = np.array([1.0 / PHI**n for n in range(raio + 1)])
    k = np.concatenate([pesos[1:][::-1], pesos])
    return k / k.sum()

def kernel_gaussiano(raio=6):
    """Gaussiana equivalente — anti-aliasing convencional."""
    sigma = raio / 2.5
    x = np.arange(-raio, raio + 1)
    k = np.exp(-x**2 / (2 * sigma**2))
    return k / k.sum()

def kernel_linear(raio=6):
    """Transição linear simples — referência mais básica."""
    x = np.arange(-raio, raio + 1)
    k = raio + 1 - np.abs(x)
    return k / k.sum()

def aplica_kernel_2d(img, kernel):
    """Aplica kernel 1D separável: horizontal depois vertical."""
    r = convolve1d(img.astype(float), kernel, axis=1, mode='reflect')
    r = convolve1d(r,                  kernel, axis=0, mode='reflect')
    return np.clip(r, 0, 255)

# ── Detecção de bordas e blending ──────────────────────────────────────────────

def detecta_bordas_sobel(img):
    """Mapa de bordas normalizado [0,1]."""
    gx = sobel(img.astype(float), axis=1)
    gy = sobel(img.astype(float), axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    return mag / (mag.max() + 1e-10)

def renderiza_com_kernel(img, kernel, forca_borda=0.7):
    """
    Aplica kernel APENAS nas regiões de borda.
    Fora das bordas: mantém o pixel original.
    Nas bordas: mistura original + suavizado pelo kernel.
    """
    bordas   = detecta_bordas_sobel(img)
    suavizado = aplica_kernel_2d(img, kernel)
    mascara  = np.clip(bordas * (1 / forca_borda), 0, 1)
    resultado = img.astype(float) * (1 - mascara) + suavizado * mascara
    return np.clip(resultado, 0, 255).astype(np.uint8), bordas

# ── Carrega imagem ─────────────────────────────────────────────────────────────

URLS = [
    "https://raw.githubusercontent.com/vitoredsonalphaphi/alpha_phi_manifesto/claude/good-morning-N6f3S/1775593114753.png",
    "https://raw.githubusercontent.com/vitoredsonalphaphi/alpha_phi_manifesto/main/1775593114753.png",
]

img_pil = None
for url in URLS:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            img_pil = Image.open(BytesIO(r.content))
            print(f"Imagem carregada: {img_pil.size}  modo={img_pil.mode}")
            break
    except Exception as e:
        print(f"  {e}")

if img_pil is None:
    raise RuntimeError("Não foi possível carregar a imagem.")

# Trabalha em escala real (até 800px)
MAX_DIM = 800
w, h = img_pil.size
escala = min(MAX_DIM / w, MAX_DIM / h, 1.0)
if escala < 1.0:
    img_pil = img_pil.resize((int(w * escala), int(h * escala)), Image.LANCZOS)
    print(f"Redimensionada: {img_pil.size}")

img_rgb  = np.array(img_pil.convert('RGB'))
img_gray = np.array(img_pil.convert('L'))
H, W     = img_gray.shape

# ── Processa com os três kernels ───────────────────────────────────────────────

RAIO = 5

k_phi   = kernel_phi(RAIO)
k_gauss = kernel_gaussiano(RAIO)
k_lin   = kernel_linear(RAIO)

print(f"\nRaio do kernel : {RAIO} pixels")
print(f"Kernel φ (pesos): {np.round(k_phi,   3)}")
print(f"Kernel G (pesos): {np.round(k_gauss, 3)}")
print(f"Kernel L (pesos): {np.round(k_lin,   3)}")

print("\nAplicando kernels...")
img_lin,   bordas = renderiza_com_kernel(img_gray, k_lin,   forca_borda=0.6)
img_gauss, _      = renderiza_com_kernel(img_gray, k_gauss, forca_borda=0.6)
img_phi,   _      = renderiza_com_kernel(img_gray, k_phi,   forca_borda=0.6)

# ── Seleciona região de zoom no contorno orgânico ─────────────────────────────

# Encontra centro de massa das bordas — provavelmente o círculo do logo
gy, gx   = np.where(bordas > 0.5)
if len(gy) > 0:
    cy, cx = int(gy.mean()), int(gx.mean())
else:
    cy, cx = H // 2, W // 2

# Zoom de 60px ao redor do contorno mais forte
ZOOM = 60
y1 = max(0, cy - ZOOM); y2 = min(H, cy + ZOOM)
x1 = max(0, cx - ZOOM); x2 = min(W, cx + ZOOM)

def z(arr):
    return arr[y1:y2, x1:x2]

# ── Perfil de intensidade cruzando borda ──────────────────────────────────────

# Linha que tem mais bordas
linha_idx = int(np.argmax(bordas.sum(axis=1)))
# Segmento que cruza a região mais ativa
col_ativo = bordas[linha_idx, :].argmax()
seg_w = 80
cs = max(0, col_ativo - seg_w); ce = min(W, col_ativo + seg_w)

p_orig  = img_gray[linha_idx, cs:ce].astype(float)
p_lin   = img_lin [linha_idx, cs:ce].astype(float)
p_gauss = img_gauss[linha_idx, cs:ce].astype(float)
p_phi   = img_phi [linha_idx, cs:ce].astype(float)

# ── Visualização ───────────────────────────────────────────────────────────────

GOLD  = "#DAA520"
CYAN  = "#00BFFF"
GREEN = "#00FF88"
WHITE = "#E6EDF3"

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#0d1117")
gs  = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.18)

def ax_img(ax, img, cmap, titulo, cor=GOLD, interp='bilinear'):
    ax.set_facecolor("#0d1117")
    ax.imshow(img, cmap=cmap, interpolation=interp)
    ax.set_title(titulo, color=cor, fontsize=10, fontweight='bold', pad=5)
    ax.axis('off')

def ax_plot(ax, titulo):
    ax.set_facecolor("#161b22")
    ax.set_title(titulo, color=GOLD, fontsize=10, fontweight='bold', pad=5)
    ax.tick_params(colors="#8B949E")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.grid(True, alpha=0.15)

# Linha 0 — imagens completas
ax_img(fig.add_subplot(gs[0,0]), img_gray,  'gray',    "Original\n(pixel euclidiano)")
ax_img(fig.add_subplot(gs[0,1]), img_lin,   'gray',    "Transição linear\n(referência básica)",     CYAN)
ax_img(fig.add_subplot(gs[0,2]), img_gauss, 'gray',    "Transição Gaussiana\n(anti-aliasing padrão)", CYAN)
ax_img(fig.add_subplot(gs[0,3]), img_phi,   'gray',    "Transição φ  (1/φⁿ)\n(anti-aliasing φ-ponderado)", GREEN)

# Linha 1 — zoom no contorno orgânico (NEAREST para ver os pixels)
ax_img(fig.add_subplot(gs[1,0]), z(img_gray),  'gray', f"ZOOM original\n(contorno orgânico — pixel a pixel)", interp='nearest')
ax_img(fig.add_subplot(gs[1,1]), z(img_lin),   'gray', f"ZOOM linear",   CYAN,  interp='nearest')
ax_img(fig.add_subplot(gs[1,2]), z(img_gauss), 'gray', f"ZOOM Gaussiana", CYAN,  interp='nearest')
ax_img(fig.add_subplot(gs[1,3]), z(img_phi),   'gray', f"ZOOM φ",         GREEN, interp='nearest')

# Linha 2 — perfil de intensidade e kernels
ax_p = fig.add_subplot(gs[2, :2]); ax_plot(ax_p, f"Perfil de intensidade — linha {linha_idx}  (cruzando contorno orgânico)")
x_p  = np.arange(len(p_orig))
ax_p.plot(x_p, p_orig,  color=GOLD,  lw=1.5, label="Original",        alpha=0.9)
ax_p.plot(x_p, p_lin,   color=CYAN,  lw=1.5, label="Linear",          alpha=0.8, ls=':')
ax_p.plot(x_p, p_gauss, color=CYAN,  lw=2.0, label="Gaussiana",       alpha=0.9, ls='--')
ax_p.plot(x_p, p_phi,   color=GREEN, lw=2.5, label="φ (1/φⁿ)",        alpha=1.0)
ax_p.legend(facecolor="#161b22", labelcolor=WHITE, fontsize=9)
ax_p.set_ylabel("Intensidade", color="#8B949E")

ax_k = fig.add_subplot(gs[2, 2:]); ax_plot(ax_k, "Kernel de transição de borda: Linear vs Gaussiana vs φ")
x_k  = np.arange(len(k_phi)) - len(k_phi) // 2
ax_k.plot(x_k, k_lin,   color=CYAN,  lw=1.5, marker='o', ms=4, label="Linear")
ax_k.plot(x_k, k_gauss, color=CYAN,  lw=2.0, marker='s', ms=5, label="Gaussiana",  ls='--')
ax_k.plot(x_k, k_phi,   color=GREEN, lw=2.5, marker='^', ms=5, label="φ  (1/φⁿ)")
# Marca posições φ^n
for n in range(1, 4):
    pos = int(round(PHI**n))
    if pos <= RAIO:
        ax_k.axvline( pos, color=GOLD, lw=0.8, ls=':', alpha=0.6)
        ax_k.axvline(-pos, color=GOLD, lw=0.8, ls=':', alpha=0.6)
        ax_k.text(pos + 0.1, k_phi[len(k_phi)//2 + pos] + 0.002,
                  f"φ^{n}", color=GOLD, fontsize=7, alpha=0.8)
ax_k.legend(facecolor="#161b22", labelcolor=WHITE, fontsize=9)
ax_k.set_xlabel("Distância do pixel de borda (pixels)", color="#8B949E")
ax_k.set_ylabel("Peso de intensidade", color="#8B949E")

fig.suptitle(
    f"ALPHA PHI — φ-Rendering  |  kernel: 1/φⁿ  |  φ={PHI:.4f}  |  raio={RAIO}px  |  Florianópolis 2026",
    color=GOLD, fontsize=12, fontweight='bold'
)

plt.savefig("alphaphi_rendering_phi.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.show()

print("\nGráfico salvo: alphaphi_rendering_phi.png")
print(f"\nO que o kernel φ faz diferente da Gaussiana:")
print(f"  Pixel 0 (borda):      φ =  {k_phi[RAIO]:.4f}  G = {k_gauss[RAIO]:.4f}")
print(f"  Pixel ±1:             φ =  {k_phi[RAIO+1]:.4f}  G = {k_gauss[RAIO+1]:.4f}")
print(f"  Pixel ±φ¹ ({int(round(PHI)):}px):      φ =  {k_phi[RAIO+int(round(PHI))]:.4f}  G = {k_gauss[RAIO+int(round(PHI))]:.4f}")
print(f"  Pixel ±φ² ({int(round(PHI**2)):}px):      φ =  {k_phi[RAIO+int(round(PHI**2))]:.4f}  G = {k_gauss[RAIO+int(round(PHI**2))]:.4f}")
print("\nalpha-phi")
