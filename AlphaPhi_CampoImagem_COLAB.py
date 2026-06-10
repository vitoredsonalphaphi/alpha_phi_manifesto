# ALPHA PHI — Sub-pixel φ no Contorno Orgânico
# Vitor Edson Delavi · Florianópolis · 2026
#
# HIPÓTESE (dos diálogos MICROPIXEL_reformulacao.md):
# Cada pixel tem 3 sub-emissores físicos: R (esquerda), G (centro), B (direita).
# ClearType (Microsoft) já usa essa estrutura para triplicar resolução horizontal.
# Aqui: distribui intensidade em proporção φ²:φ¹:φ⁰ em vez de linear/igual.
#
# BORDA ESQUERDA (escuro→claro, gx > 0):
#   sub-pixel R = I × 1/φ²   (esquerda — ainda na região escura)
#   sub-pixel G = I × 1/φ    (centro)
#   sub-pixel B = I × 1      (direita — mais dentro da região clara)
#   → centro de massa do pixel desloca φ-proporcionalmente para a direita
#
# BORDA DIREITA (claro→escuro, gx < 0):
#   sub-pixel R = I × 1      (esquerda — mais dentro da região clara)
#   sub-pixel G = I × 1/φ    (centro)
#   sub-pixel B = I × 1/φ²   (direita — ainda na região escura)
#   → centro de massa do pixel desloca φ-proporcionalmente para a esquerda
#
# REFERÊNCIA LINEAR (ClearType convencional):
#   Borda esquerda: R=1/3  G=2/3  B=1     (decay linear)
#   Borda direita:  R=1    G=2/3  B=1/3   (decay linear reverso)
#
# COMPARAÇÃO: Original | Linear (ClearType-like) | φ (AlphaPhi)

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "Pillow", "requests", "scipy"], check=True)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from scipy.ndimage import sobel, gaussian_filter

PHI   = (1 + np.sqrt(5)) / 2
C_PHI = 1.0 / PHI**2

print(f"φ   = {PHI:.6f}")
print(f"1/φ = {1/PHI:.6f}")
print(f"Pesos φ sub-pixel: R={1:.3f}  G={1/PHI:.3f}  B={1/PHI**2:.3f}  (borda direita)")
print(f"Pesos φ sub-pixel: R={1/PHI**2:.3f}  G={1/PHI:.3f}  B={1:.3f}  (borda esquerda)")
print("=" * 62)
print("Sub-pixel φ no Contorno Orgânico")
print("=" * 62)

# ── Renderização sub-pixel ────────────────────────────────────────────────────

def subpixel_phi(img_rgb, limiar=0.1):
    """
    Aplica pesos φ²:φ¹:φ⁰ nos canais R,G,B dos pixels de borda.
    Força proporcional ao gradiente horizontal normalizado.
    """
    gray  = img_rgb.mean(axis=2)
    gx    = sobel(gray.astype(float), axis=1)
    gx_sm = gaussian_filter(gx, sigma=0.5)   # remove ruído pontual

    gmax = np.abs(gx_sm).max() + 1e-10
    gx_n = gx_sm / gmax                       # [-1, 1]

    result = img_rgb.copy().astype(float)

    # Pesos φ normalizados (máximo = 1, mínimo = 1/φ²)
    w_max = 1.0
    w_mid = 1.0 / PHI       # ≈ 0.618
    w_min = 1.0 / PHI**2    # ≈ 0.382

    # Força do efeito: só age onde há gradiente acima do limiar
    forca = np.clip((np.abs(gx_n) - limiar) / (1.0 - limiar), 0, 1)

    for ch in range(3):
        I = img_rgb[:, :, ch].astype(float)

        # Borda direita (gx < 0): R=w_max, G=w_mid, B=w_min
        # Pesos por canal: ch0=w_max, ch1=w_mid, ch2=w_min
        pesos_dir = [w_max, w_mid, w_min]
        fator_dir = pesos_dir[ch]   # fração a aplicar na borda direita

        # Borda esquerda (gx > 0): R=w_min, G=w_mid, B=w_max (reverso)
        pesos_esq = [w_min, w_mid, w_max]
        fator_esq = pesos_esq[ch]

        # Gradiente positivo = borda esquerda, negativo = borda direita
        pos = np.clip( gx_n, 0, 1)   # borda esquerda
        neg = np.clip(-gx_n, 0, 1)   # borda direita

        fator_final = (1.0
                       - pos * forca * (1.0 - fator_esq)
                       - neg * forca * (1.0 - fator_dir))

        result[:, :, ch] = np.clip(I * fator_final, 0, 255)

    return result.astype(np.uint8)

def subpixel_linear(img_rgb, limiar=0.1):
    """
    Referência linear (análogo ao ClearType).
    Pesos 1:2/3:1/3 — decay linear em vez de φ.
    """
    gray  = img_rgb.mean(axis=2)
    gx    = sobel(gray.astype(float), axis=1)
    gx_sm = gaussian_filter(gx, sigma=0.5)
    gmax  = np.abs(gx_sm).max() + 1e-10
    gx_n  = gx_sm / gmax

    result = img_rgb.copy().astype(float)
    forca  = np.clip((np.abs(gx_n) - limiar) / (1.0 - limiar), 0, 1)

    pesos_dir = [1.0,  2/3,  1/3]   # R fica, G reduz, B quase vai
    pesos_esq = [1/3,  2/3,  1.0]   # espelho

    for ch in range(3):
        I = img_rgb[:, :, ch].astype(float)
        pos = np.clip( gx_n, 0, 1)
        neg = np.clip(-gx_n, 0, 1)
        fator = (1.0
                 - pos * forca * (1.0 - pesos_esq[ch])
                 - neg * forca * (1.0 - pesos_dir[ch]))
        result[:, :, ch] = np.clip(I * fator, 0, 255)

    return result.astype(np.uint8)

def gradiente_borda(img, mascara):
    """Gradiente médio nos pixels de borda — métrica de suavidade."""
    gray = img.mean(axis=2) if img.ndim == 3 else img
    gx   = np.abs(sobel(gray.astype(float), axis=1))
    gy   = np.abs(sobel(gray.astype(float), axis=0))
    mag  = np.sqrt(gx**2 + gy**2)
    return float(mag[mascara].mean()) if mascara.sum() > 0 else float(mag.mean())

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
            print(f"Imagem: {img_pil.size}  modo={img_pil.mode}")
            break
    except Exception as e:
        print(f"  {e}")
if img_pil is None:
    raise RuntimeError("Imagem não carregada.")

MAX_DIM = 700
w, h = img_pil.size
esc  = min(MAX_DIM/w, MAX_DIM/h, 1.0)
if esc < 1.0:
    img_pil = img_pil.resize((int(w*esc), int(h*esc)), Image.LANCZOS)
    print(f"Redimensionada: {img_pil.size}")

img_rgb  = np.array(img_pil.convert('RGB'))
img_gray = img_rgb.mean(axis=2)
H, W     = img_gray.shape

# ── Aplica versões ─────────────────────────────────────────────────────────────

print("\nAplicando sub-pixel linear (ClearType-like)...")
img_lin = subpixel_linear(img_rgb)

print("Aplicando sub-pixel φ (AlphaPhi)...")
img_phi = subpixel_phi(img_rgb)

# Máscara de borda para métrica
bx      = sobel(img_gray, axis=1)
by      = sobel(img_gray, axis=0)
bordas  = np.sqrt(bx**2 + by**2)
mascara = bordas > bordas.max() * 0.35

m_orig = gradiente_borda(img_rgb,  mascara)
m_lin  = gradiente_borda(img_lin,  mascara)
m_phi  = gradiente_borda(img_phi,  mascara)

print(f"\nMétrica — gradiente médio na borda:")
print(f"  Original              : {m_orig:.4f}")
print(f"  Linear (ClearType)    : {m_lin:.4f}")
print(f"  φ (AlphaPhi)          : {m_phi:.4f}  ← φ")

# ── Região de zoom ─────────────────────────────────────────────────────────────

iy, ix = np.where(bordas > bordas.max() * 0.45)
cy = int(iy.mean()) if len(iy) else H//2
cx = int(ix.mean()) if len(ix) else W//2

Z = 55
y1,y2 = max(0,cy-Z), min(H,cy+Z)
x1,x2 = max(0,cx-Z), min(W,cx+Z)

def zc(arr):
    return arr[y1:y2, x1:x2]

# Linha de perfil pelos canais R, G, B
linha_pico = bordas[y1:y2, :].sum(axis=1).argmax() + y1
s = 70
cs,ce = max(0,cx-s), min(W,cx+s)

# ── Visualização ───────────────────────────────────────────────────────────────

GOLD  = "#DAA520"
CYAN  = "#00BFFF"
GREEN = "#00FF88"
RED   = "#FF4466"
WHITE = "#E6EDF3"

fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor("#0d1117")
gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.12)

def ai(ax, im, titulo, cor=GOLD, interp='bilinear'):
    ax.set_facecolor("#0d1117")
    ax.imshow(im if im.ndim==3 else im, interpolation=interp)
    ax.set_title(titulo, color=cor, fontsize=11, fontweight='bold', pad=5)
    ax.axis('off')

def ap(ax, titulo):
    ax.set_facecolor("#161b22")
    ax.set_title(titulo, color=GOLD, fontsize=10, fontweight='bold', pad=5)
    ax.tick_params(colors="#8B949E")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.grid(True, alpha=0.15)

# Linha 1 — imagens completas
ai(fig.add_subplot(gs[0,0]), img_rgb, f"Original\ngrad borda={m_orig:.1f}")
ai(fig.add_subplot(gs[0,1]), img_lin, f"Linear (ClearType-like)\n1:2/3:1/3    grad={m_lin:.1f}", CYAN)
ai(fig.add_subplot(gs[0,2]), img_phi, f"φ-sub-pixel (AlphaPhi)\n1:1/φ:1/φ²  grad={m_phi:.1f}", GREEN)

# Linha 2 — ZOOM pixel-a-pixel (nearest = ver cada pixel individualmente)
ai(fig.add_subplot(gs[1,0]), zc(img_rgb), "ZOOM original\n(serrilhado — pixel retangular)",      GOLD,  'nearest')
ai(fig.add_subplot(gs[1,1]), zc(img_lin), "ZOOM linear\n(franja cromática linear)",              CYAN,  'nearest')
ai(fig.add_subplot(gs[1,2]), zc(img_phi), "ZOOM φ-sub-pixel\n(franja cromática φ-ponderada)",    GREEN, 'nearest')

# Linha 3 — Perfis dos canais R, G, B na linha de borda
ax_orig = fig.add_subplot(gs[2,0]); ap(ax_orig, f"Perfil canais — Original  (linha {linha_pico})")
ax_lin  = fig.add_subplot(gs[2,1]); ap(ax_lin,  "Perfil canais — Linear")
ax_phi  = fig.add_subplot(gs[2,2]); ap(ax_phi,  "Perfil canais — φ")

for ax, img, titulo in [(ax_orig,img_rgb,''),
                         (ax_lin, img_lin,''),
                         (ax_phi, img_phi,'')]:
    x_p = np.arange(ce - cs)
    ax.plot(x_p, img[linha_pico, cs:ce, 0].astype(float), color='#FF5555', lw=1.8, label='R')
    ax.plot(x_p, img[linha_pico, cs:ce, 1].astype(float), color='#55FF55', lw=1.8, label='G')
    ax.plot(x_p, img[linha_pico, cs:ce, 2].astype(float), color='#5599FF', lw=1.8, label='B')
    ax.legend(facecolor="#161b22", labelcolor=WHITE, fontsize=8, loc='upper right')
    ax.set_ylabel("Intensidade", color="#8B949E")
    ax.set_xlabel("Posição (pixels)", color="#8B949E")
    ax.set_ylim(-5, 270)

# Anotação: pesos φ
fig.text(0.67, 0.03,
    f"Pesos φ borda direita:  R={1:.3f}  G={1/PHI:.3f}  B={1/PHI**2:.3f}\n"
    f"Pesos φ borda esquerda: R={1/PHI**2:.3f}  G={1/PHI:.3f}  B={1:.3f}",
    ha='center', color=GREEN, fontsize=9, family='monospace')

fig.text(0.33, 0.03,
    f"Pesos lineares borda direita:  R=1.000  G=0.667  B=0.333\n"
    f"Pesos lineares borda esquerda: R=0.333  G=0.667  B=1.000",
    ha='center', color=CYAN, fontsize=9, family='monospace')

fig.suptitle(
    f"ALPHA PHI — Sub-pixel φ  |  pesos R:G:B = 1 : 1/φ : 1/φ²  |  φ={PHI:.4f}  |  Florianópolis 2026",
    color=GOLD, fontsize=12, fontweight='bold'
)

plt.savefig("alphaphi_subpixel_phi.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.show()

print("\nGráfico salvo: alphaphi_subpixel_phi.png")
print("\nO que observar no zoom (linha 2):")
print("  Original : borda em degraus de 1 pixel — quadradinho puro")
print("  Linear   : franja cromática R/G/B em cada pixel de borda")
print("  φ        : franja cromática com pesos φ — R:G:B = 1:0.618:0.382")
print("\nO que observar nos perfis (linha 3):")
print("  Original : R=G=B em todos os pixels (cinza puro)")
print("  Linear   : R≠G≠B nos pixels de borda (franja cromática linear)")
print("  φ        : R≠G≠B nos pixels de borda (franja cromática φ)")
print("\nalpha-phi")
