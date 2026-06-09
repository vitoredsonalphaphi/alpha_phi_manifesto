# ALPHA PHI — Comparativo de Decay: φ vs outros valores
# Vitor Edson Delavi · Florianópolis · 2026
#
# PERGUNTA:
# O decay 1/φ = 0.618 produz resultado visualmente distinto
# de outros valores de decay no filtro de eco unidirecional?
# Ou o efeito de suavização de contorno é genérico?
#
# MÉTODO:
# - Mesma estrutura de filtro para todas as versões
# - Mesmos delays: [2, 3, 4, 7]px  (posições round(φ^n))
# - Só o decay varia: 0.50 / 0.55 / 0.618(φ) / 0.65 / 0.70
# - Referência: Gaussiana simétrica (blur padrão)
# - Zoom pixel-a-pixel no contorno circular do símbolo
# - Métrica: gradiente médio na borda (menor = mais suave)

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "Pillow", "requests", "scipy"], check=True)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from scipy.ndimage import sobel, gaussian_filter

PHI = (1 + np.sqrt(5)) / 2
print(f"φ = {PHI:.6f}   1/φ = {1/PHI:.6f}")
print("=" * 60)
print("Comparativo de Decay: φ vs outros valores")
print("=" * 60)

# ── Filtro eco unidirecional com decay variável ───────────────────────────────

DELAYS = [2, 3, 4, 7]   # round(φ¹), round(φ²), round(φ³), round(φ⁴) — fixos

def eco_decay(sinal, d):
    """
    Eco unidirecional com decay `d`.
    Delays fixos em [2,3,4,7]px; ganho do estágio n = d^n.
    Compara: mesmo delay, decay diferente.
    """
    out = sinal.copy().astype(float)
    for n, delay in enumerate(DELAYS, start=1):
        ganho = d**n
        if delay < len(sinal):
            out[delay:] += ganho * sinal[:-delay]
    mx = out.max()
    if mx > 1e-10:
        out = out / mx * sinal.max()
    return out

def aplica_eco_imagem(img, d):
    """Aplica eco com decay d por linha e depois por coluna."""
    r = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        r[i, :] = eco_decay(img[i, :].astype(float), d)
    s = r.copy()
    for j in range(r.shape[1]):
        s[:, j] = eco_decay(r[:, j], d)
    return np.clip(s, 0, 255).astype(np.uint8)

def aplica_gaussiana(img, sigma=1.2):
    """Gaussiana simétrica — referência convencional."""
    return np.clip(gaussian_filter(img.astype(float), sigma=sigma), 0, 255).astype(np.uint8)

def gradiente_medio_borda(img, mascara_borda):
    """Métrica: gradiente médio nos pixels de borda (menor = mais suave)."""
    gx = np.abs(np.diff(img.astype(float), axis=1, prepend=img[:,:1]))
    gy = np.abs(np.diff(img.astype(float), axis=0, prepend=img[:1,:]))
    mag = np.sqrt(gx**2 + gy**2)
    if mascara_borda.sum() == 0:
        return float(mag.mean())
    return float(mag[mascara_borda].mean())

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

MAX_DIM = 600
w, h = img_pil.size
esc = min(MAX_DIM/w, MAX_DIM/h, 1.0)
if esc < 1.0:
    img_pil = img_pil.resize((int(w*esc), int(h*esc)), Image.LANCZOS)
    print(f"Redimensionada: {img_pil.size}")

img_gray = np.array(img_pil.convert('L'))
H, W     = img_gray.shape

# ── Aplica todas as versões ────────────────────────────────────────────────────

DECAYS = [0.50, 0.55, 1/PHI, 0.65, 0.70]
NOMES  = ['d=0.50', 'd=0.55', f'd=1/φ\n(0.618)', 'd=0.65', 'd=0.70']
CORES  = ['#4488FF', '#88BBFF', '#00FF88', '#FFBB44', '#FF6644']

print("\nProcessando versões...")
versoes     = {}
versoes['Original']  = img_gray.copy()
versoes['Gaussiana'] = aplica_gaussiana(img_gray, sigma=1.2)
for d, nome in zip(DECAYS, NOMES):
    chave = nome.replace('\n','')
    versoes[chave] = aplica_eco_imagem(img_gray, d)
    print(f"  {chave:<18}  ganhos: {[round(d**n,4) for n in range(1,5)]}")

# ── Detecção de bordas e zoom ──────────────────────────────────────────────────

bx     = sobel(img_gray.astype(float), axis=1)
by     = sobel(img_gray.astype(float), axis=0)
bordas = np.sqrt(bx**2 + by**2)

# Centro do contorno orgânico (círculo do logo)
iy, ix = np.where(bordas > bordas.max() * 0.45)
cy = int(iy.mean()) if len(iy) else H//2
cx = int(ix.mean()) if len(ix) else W//2

# Mascara de borda para métrica
mascara_borda = bordas > bordas.max() * 0.35

# Zoom region — 70px ao redor do centro do círculo
Z = 70
y1,y2 = max(0,cy-Z), min(H,cy+Z)
x1,x2 = max(0,cx-Z), min(W,cx+Z)

def zc(arr):
    return arr[y1:y2, x1:x2]

# Linha de perfil que cruza o contorno
linha_pico = bordas[y1:y2, :].sum(axis=1).argmax() + y1
seg = 90
cs  = max(0, cx - seg)
ce  = min(W, cx + seg)

# ── Métricas ───────────────────────────────────────────────────────────────────

print(f"\nMétrica — gradiente médio nos pixels de borda (menor = mais suave):")
metricas = {}
for nome, img in versoes.items():
    m = gradiente_medio_borda(img, mascara_borda)
    metricas[nome] = m
    phi_tag = "  ← φ" if '1/φ' in nome or '0.618' in nome else ""
    print(f"  {nome:<22} {m:.4f}{phi_tag}")

# ── Visualização ───────────────────────────────────────────────────────────────

GOLD  = "#DAA520"
WHITE = "#E6EDF3"

# 7 colunas: Original + 5 decay + Gaussiana
CHAVES_ORD = ['Original', 'd=0.50', 'd=0.55', 'd=1/φ(0.618)', 'd=0.65', 'd=0.70', 'Gaussiana']
CORES_ORD  = [GOLD, '#4488FF', '#88BBFF', '#00FF88', '#FFBB44', '#FF6644', '#AAAAAA']

fig = plt.figure(figsize=(22, 15))
fig.patch.set_facecolor("#0d1117")
gs  = fig.add_gridspec(3, 7, hspace=0.42, wspace=0.10)

def ai(ax, im, titulo, cor=GOLD, interp='bilinear'):
    ax.set_facecolor("#0d1117")
    ax.imshow(im, cmap='gray', interpolation=interp, vmin=0, vmax=255)
    ax.set_title(titulo, color=cor, fontsize=9, fontweight='bold', pad=4)
    ax.axis('off')

# Linha 1 — imagem completa
for col, (chave, cor) in enumerate(zip(CHAVES_ORD, CORES_ORD)):
    ax = fig.add_subplot(gs[0, col])
    img = versoes.get(chave, img_gray)
    m   = metricas.get(chave, 0)
    tag = " ←φ" if '1/φ' in chave else ""
    ai(ax, img, f"{chave}{tag}\ngrad={m:.2f}", cor)

# Linha 2 — ZOOM pixel-a-pixel no contorno circular
for col, (chave, cor) in enumerate(zip(CHAVES_ORD, CORES_ORD)):
    ax = fig.add_subplot(gs[1, col])
    img = versoes.get(chave, img_gray)
    tag = " ←φ" if '1/φ' in chave else ""
    ai(ax, zc(img), f"ZOOM{tag}", cor, interp='nearest')
    ax.axhline(linha_pico - y1, color='white', lw=0.7, ls='--', alpha=0.5)

# Linha 3 — Perfil de intensidade (esquerda) + Kernels (direita)
ax_p = fig.add_subplot(gs[2, :4])
ax_p.set_facecolor("#161b22")
ax_p.set_title(f"Perfil de intensidade — linha {linha_pico}  (cruza o contorno do círculo)",
               color=GOLD, fontsize=10, fontweight='bold', pad=5)
ax_p.tick_params(colors="#8B949E")
for sp in ax_p.spines.values(): sp.set_color("#30363d")
ax_p.grid(True, alpha=0.15)

cores_perf = [GOLD, '#4488FF', '#88BBFF', '#00FF88', '#FFBB44', '#FF6644', '#AAAAAA']
for (chave, cor) in zip(CHAVES_ORD, cores_perf):
    img  = versoes.get(chave, img_gray)
    perf = img[linha_pico, cs:ce].astype(float)
    lw   = 3.0 if '1/φ' in chave else (2.0 if chave == 'Original' else 1.2)
    ls   = '-'  if '1/φ' in chave or chave == 'Original' else '--'
    tag  = " (φ)" if '1/φ' in chave else ""
    ax_p.plot(np.arange(len(perf)), perf, color=cor, lw=lw, ls=ls,
              label=f"{chave}{tag}", alpha=0.92)
ax_p.legend(facecolor="#161b22", labelcolor=WHITE, fontsize=8, ncol=2)
ax_p.set_ylabel("Intensidade", color="#8B949E")
ax_p.set_xlabel("Posição (pixels)", color="#8B949E")

# Kernels: mostra os ganhos por estágio para cada decay
ax_k = fig.add_subplot(gs[2, 4:])
ax_k.set_facecolor("#161b22")
ax_k.set_title("Ganhos por estágio de eco  (delays fixos: 2, 3, 4, 7 px)",
               color=GOLD, fontsize=10, fontweight='bold', pad=5)
ax_k.tick_params(colors="#8B949E")
for sp in ax_k.spines.values(): sp.set_color("#30363d")
ax_k.grid(True, alpha=0.15)

estagios = np.arange(1, len(DELAYS)+1)
for d, nome, cor in zip(DECAYS, NOMES, CORES):
    ganhos = [d**n for n in estagios]
    lw  = 3.0 if abs(d - 1/PHI) < 0.001 else 1.5
    tag = " ← φ" if abs(d - 1/PHI) < 0.001 else ""
    ax_k.plot(estagios, ganhos, color=cor, lw=lw, marker='o', ms=6,
              label=f"d={d:.3f}{tag}")
ax_k.set_xticks(estagios)
ax_k.set_xticklabels([f"φ^{n}\n({DELAYS[n-1]}px)" for n in estagios], color="#8B949E")
ax_k.set_ylabel("Ganho do eco", color="#8B949E")
ax_k.legend(facecolor="#161b22", labelcolor=WHITE, fontsize=9)

fig.suptitle(
    f"ALPHA PHI — Comparativo de Decay  |  delays fixos {DELAYS}px  |  φ={PHI:.4f}  |  Florianópolis 2026",
    color=GOLD, fontsize=12, fontweight='bold'
)

plt.savefig("alphaphi_comparativo_decay.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.show()

print("\nGráfico salvo: alphaphi_comparativo_decay.png")
print("\nalpha-phi")
