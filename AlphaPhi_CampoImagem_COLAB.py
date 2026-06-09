# ALPHA PHI — EcoBIP no Contorno Orgânico
# Vitor Edson Delavi · Florianópolis · 2026
#
# PERGUNTA DO EXPERIMENTO:
# ──────────────────────────────────────────────────────────────────
# O pixel é sempre euclidiano (retangular). Um contorno orgânico
# (círculo, curva) representado por pixels quadradinhos produz
# inevitavelmente serrilhado — independente do grau de minimização.
#
# O EcoBIP aplicado a cada linha e coluna distribui intensidade
# nos pixels de borda em proporção φ (decaimento 1/φ^n por passo).
#
# Análogo ao que foi feito no áudio (BIP 880Hz) e no texto
# (eco em cada caractere): a estrutura φ de Fibonacci aplicada
# ao sinal — aqui o sinal é a linha de pixels do contorno.
#
# O que acontece com o contorno do símbolo?
# A distribuição φ de intensidade nos pixels de borda alcança
# ergonomia mais otimizada do que o pixel bruto original?
# ──────────────────────────────────────────────────────────────────

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "Pillow", "requests", "scipy"], check=True)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from scipy.ndimage import sobel

PHI   = (1 + np.sqrt(5)) / 2
C_PHI = 1.0 / PHI**2

print(f"φ         = {PHI:.6f}")
print(f"1/φ       = {1/PHI:.6f}  (decaimento por passo no EcoBIP)")
print(f"1/φ²      = {C_PHI:.6f}  (curvatura)")
print("=" * 62)
print("ALPHA PHI — EcoBIP no Contorno Orgânico")
print("=" * 62)

# ── EcoBIP φ (mesmo princípio do áudio) ──────────────────────────────────────

def ecobip_linha(sinal, n_estagios=5):
    """
    5 estágios de eco φ — mesmo princípio do BIP 880Hz.
    Cada estágio: delay = round(φ^n), ganho = 1/φ^n
    A intensidade dos pixels vizinhos decai como Fibonacci.
    """
    out = sinal.copy().astype(float)
    for s in range(1, n_estagios):
        delay = max(1, int(round(PHI**s)))
        ganho = 1.0 / PHI**s
        if delay < len(sinal):
            out[delay:] += ganho * sinal[:-delay]
    mx = out.max()
    if mx > 1e-10:
        out = out / mx * sinal.max()
    return out

def ecobip_imagem(img):
    """Aplica EcoBIP a cada linha (horizontal) e depois a cada coluna (vertical)."""
    img_f = img.astype(float)
    # Passa horizontal
    r = np.zeros_like(img_f)
    for i in range(img_f.shape[0]):
        r[i, :] = ecobip_linha(img_f[i, :])
    # Passa vertical no resultado
    s = r.copy()
    for j in range(r.shape[1]):
        s[:, j] = ecobip_linha(r[:, j])
    return np.clip(s, 0, 255).astype(np.uint8)

# ── Campo φ — coerência das relações entre pixels ─────────────────────────────

def campo_phi(img, raios=None):
    """
    Para cada pixel: mede o quanto a relação de intensidade
    com vizinhos a distâncias φ^n ressoa com a proporção áurea.
    """
    if raios is None:
        raios = [max(1, int(round(PHI**n))) for n in range(1, 5)]
    img_f = img.astype(float)
    campo = np.zeros_like(img_f)
    for r in raios:
        for dy, dx in [(0,1),(1,0),(1,1),(1,-1)]:
            viz      = np.roll(np.roll(img_f, dy*r, 0), dx*r, 1)
            razao    = (img_f + 1.0) / (viz + 1.0)
            distlog  = np.abs(np.log(np.abs(razao) + 1e-10) - np.log(PHI))
            campo   += np.exp(-distlog**2 / 0.07)
    return campo / (campo.max() + 1e-10)

# ── Poincaré (especulação: o que seria o pixel hiperbólico?) ──────────────────

def expmap0_img(img, c=C_PHI):
    """
    Mapeia intensidades para a bola de Poincaré com curvatura c=1/φ².
    Especulação: como seria a imagem se os pixels fossem hiperbólicos?
    """
    v    = img.astype(float) / 127.5 - 1.0     # normaliza [-1, 1]
    norma = np.abs(v) + 1e-10
    mapeado = np.tanh(np.sqrt(c) * norma) * v / (np.sqrt(c) * norma)
    return ((mapeado + 1.0) / 2.0 * 255).astype(np.uint8)

# ── Carrega imagem ─────────────────────────────────────────────────────────────

URLS = [
    "https://raw.githubusercontent.com/vitoredsonalphaphi/alpha_phi_manifesto/claude/good-morning-N6f3S/1775593114753.png",
    "https://raw.githubusercontent.com/vitoredsonalphaphi/alpha_phi_manifesto/main/1775593114753.png",
]

img_pil = None
for url in URLS:
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            img_pil = Image.open(BytesIO(resp.content))
            print(f"Imagem carregada: {img_pil.size}  modo={img_pil.mode}")
            break
    except Exception as e:
        print(f"  falhou: {e}")

if img_pil is None:
    raise RuntimeError("Imagem não carregada.")

MAX_DIM = 600
w, h = img_pil.size
esc  = min(MAX_DIM / w, MAX_DIM / h, 1.0)
if esc < 1.0:
    img_pil = img_pil.resize((int(w*esc), int(h*esc)), Image.LANCZOS)
    print(f"Redimensionada: {img_pil.size}")

img_rgb  = np.array(img_pil.convert('RGB'))
img_gray = np.array(img_pil.convert('L'))
H, W     = img_gray.shape

# ── Processa ───────────────────────────────────────────────────────────────────

print("EcoBIP...")
img_bip   = ecobip_imagem(img_gray)

print("Campo φ...")
campo     = campo_phi(img_gray)

print("Poincaré...")
img_hyp   = expmap0_img(img_gray)

# Δ BIP — o que o eco adiciona
delta_bip = np.abs(img_bip.astype(float) - img_gray.astype(float))
delta_vis = (delta_bip / (delta_bip.max() + 1e-10) * 255).astype(np.uint8)

# Overlay campo φ em dourado sobre original
overlay = img_rgb.copy().astype(float)
m = campo
overlay[:,:,0] = np.clip(overlay[:,:,0] + m * 110, 0, 255)
overlay[:,:,1] = np.clip(overlay[:,:,1] + m *  70, 0, 255)
overlay[:,:,2] = np.clip(overlay[:,:,2] - m *  60, 0, 255)
overlay = overlay.astype(np.uint8)

# ── Região de zoom: contorno circular do logo ─────────────────────────────────

# Sobel para encontrar as bordas
bx = sobel(img_gray.astype(float), axis=1)
by = sobel(img_gray.astype(float), axis=0)
bordas = np.sqrt(bx**2 + by**2)

# Centro de massa das bordas fortes = região do círculo do logo
iy, ix = np.where(bordas > bordas.max() * 0.4)
cy = int(iy.mean()) if len(iy) > 0 else H // 2
cx = int(ix.mean()) if len(ix) > 0 else W // 2

ZOOM = 55  # pixels ao redor do centro do círculo
y1,y2 = max(0, cy-ZOOM), min(H, cy+ZOOM)
x1,x2 = max(0, cx-ZOOM), min(W, cx+ZOOM)

def zc(arr):
    return arr[y1:y2, x1:x2]

# Perfil: linha que cruza o contorno circular
linha_pico = bordas[y1:y2, :].sum(axis=1).argmax() + y1
p_orig = img_gray[linha_pico, x1-20:x2+20].astype(float)
p_bip  = img_bip [linha_pico, x1-20:x2+20].astype(float)
p_delt = delta_bip[linha_pico, x1-20:x2+20]

# ── Visualização ───────────────────────────────────────────────────────────────

GOLD  = "#DAA520"
CYAN  = "#00BFFF"
GREEN = "#00FF88"
WHITE = "#E6EDF3"

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0d1117")
gs  = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.16)

def ai(ax, im, cmap, titulo, cor=GOLD, interp='bilinear'):
    ax.set_facecolor("#0d1117")
    ax.imshow(im, cmap=cmap, interpolation=interp)
    ax.set_title(titulo, color=cor, fontsize=10.5, fontweight='bold', pad=6)
    ax.axis('off')

def ap(ax, titulo):
    ax.set_facecolor("#161b22")
    ax.set_title(titulo, color=GOLD, fontsize=10, fontweight='bold', pad=5)
    ax.tick_params(colors="#8B949E")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.grid(True, alpha=0.15)

# ─── LINHA 1: Visão geral — as 6 perspectivas ────────────────────────────────

ai(fig.add_subplot(gs[0,0]), img_rgb,   None,    "① Original\n(pixel euclidiano)")
ai(fig.add_subplot(gs[0,1]), campo,     'inferno',"② Campo φ — Coerência\n(onde a imagem já ressoa em φ)")
ai(fig.add_subplot(gs[0,2]), overlay,   None,    "③ Campo φ sobreposto\n(dourado = alta ressonância φ)")

# ─── LINHA 2: EcoBIP + Δ + Poincaré ─────────────────────────────────────────

ai(fig.add_subplot(gs[1,0]), img_bip,   'gray',  "④ EcoBIP φ (5 estágios)\n(eco por linha e coluna)", GREEN)
ai(fig.add_subplot(gs[1,1]), delta_vis, 'hot',   "⑤ Δ BIP — o que o eco φ adiciona\n(campo harmônico emergente)")
ai(fig.add_subplot(gs[1,2]), img_hyp,   'gray',  f"⑥ Poincaré  c=1/φ²={C_PHI:.4f}\n(especulação: pixel hiperbólico)")

# ─── LINHA 3: ZOOM no contorno circular + perfil ─────────────────────────────

ax_zo = fig.add_subplot(gs[2,0])
ax_zb = fig.add_subplot(gs[2,1])
ax_pf = fig.add_subplot(gs[2,2])

# Zoom contorno — NEAREST para ver cada pixel individualmente
ai(ax_zo, zc(img_gray), 'gray', f"ZOOM contorno circular\nOriginal — serrilhado visível pixel a pixel", interp='nearest')
ai(ax_zb, zc(img_bip),  'gray', f"ZOOM contorno circular\nApós EcoBIP φ — distribuição 1/φⁿ", GREEN, interp='nearest')

# Linha de corte visualizada nos zooms
for ax in [ax_zo, ax_zb]:
    ax.axis('off')
    lrel = linha_pico - y1
    if 0 <= lrel < (y2-y1):
        ax.axhline(lrel, color='white', lw=0.9, ls='--', alpha=0.6)

# Perfil de intensidade cruzando o contorno
ap(ax_pf, f"Perfil — linha {linha_pico}  (cruza o círculo do símbolo)")
x_p = np.arange(len(p_orig))
ax_pf.plot(x_p, p_orig, color=GOLD,  lw=2.0, label="Original (serrilhado)")
ax_pf.plot(x_p, p_bip,  color=GREEN, lw=2.0, label="EcoBIP φ (1/φⁿ)")
ax_pf.fill_between(x_p, p_orig, p_bip, alpha=0.15, color=GREEN, label="Δ adicionado")
# Marca posições φ^n a partir do centro
centro_perf = len(p_orig) // 2
for n in range(1, 5):
    d = int(round(PHI**n))
    for sinal in [+1, -1]:
        pos = centro_perf + sinal*d
        if 0 <= pos < len(p_orig):
            ax_pf.axvline(pos, color=GOLD, lw=0.7, ls=':', alpha=0.5)
ax_pf.text(0.02, 0.95, f"↑ posições φⁿ marcadas em dourado", transform=ax_pf.transAxes,
           color=GOLD, fontsize=7, alpha=0.7, va='top')
ax_pf.legend(facecolor="#161b22", labelcolor=WHITE, fontsize=9)
ax_pf.set_ylabel("Intensidade pixel", color="#8B949E")
ax_pf.set_xlabel("Posição (pixels)", color="#8B949E")

# ─── Título e rodapé ──────────────────────────────────────────────────────────

fig.text(0.5, 0.01,
    "PERGUNTA: O EcoBIP distribui a intensidade dos pixels de borda em proporção φ — "
    "o contorno orgânico do símbolo alcança ergonomia mais otimizada?",
    ha='center', color=GOLD, fontsize=9.5, style='italic', alpha=0.85)

fig.suptitle(
    f"ALPHA PHI — EcoBIP no Contorno Orgânico  |  φ={PHI:.4f}  c=1/φ²={C_PHI:.4f}  |  Florianópolis 2026",
    color=GOLD, fontsize=12, fontweight='bold'
)

plt.savefig("alphaphi_ecobip_contorno.png", dpi=150,
            bbox_inches='tight', facecolor="#0d1117")
plt.show()

print(f"\nGráfico salvo: alphaphi_ecobip_contorno.png")
print(f"Região de zoom: y=[{y1}:{y2}]  x=[{x1}:{x2}]  (centro do círculo)")
print(f"Linha de perfil: {linha_pico}")
print(f"Delays do EcoBIP: {[max(1,int(round(PHI**s))) for s in range(1,5)]}px")
print(f"Ganhos do EcoBIP: {[round(1/PHI**s,4) for s in range(1,5)]}")
print("\nalpha-phi")
