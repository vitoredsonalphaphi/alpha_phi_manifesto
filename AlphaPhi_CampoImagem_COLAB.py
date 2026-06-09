# ALPHA PHI — Campo φ em Imagem Digital
# Vitor Edson Delavi · Florianópolis · 2026
#
# O que este experimento faz:
# Aplica análise de campo φ ao símbolo AlphaPhi (imagem digital, 100% euclidiana).
# Pergunta: onde a imagem já ressoa em φ? O que o eco BIP revela?
#
# O pixel é sempre euclidiano. Mas as RELAÇÕES entre pixels podem ter estrutura φ.
# Este experimento torna isso visível.

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "Pillow", "requests"], check=True)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

PHI   = (1 + np.sqrt(5)) / 2
C_PHI = 1.0 / PHI**2

print(f"φ         = {PHI:.10f}")
print(f"c = 1/φ²  = {C_PHI:.10f}")
print("=" * 60)
print("ALPHA PHI — Campo φ em Imagem Digital")
print("=" * 60)

# ── Funções do Campo φ ────────────────────────────────────────────────────────

def campo_phi_coerencia(img, raios=None):
    """
    Para cada pixel, mede o quanto sua relação de intensidade
    com vizinhos a distâncias φ^n ressoa com a proporção áurea.
    Alto = o pixel está numa estrutura com proporção φ.
    """
    if raios is None:
        # Distâncias φ¹, φ², φ³, φ⁴ arredondadas
        raios = [max(1, int(round(PHI**n))) for n in range(1, 5)]

    h, w  = img.shape
    campo = np.zeros((h, w), dtype=float)
    peso  = 0.0

    for n, r in enumerate(raios):
        # Peso decrescente por φ: raio mais próximo = mais peso
        w_n = 1.0 / PHI**n
        for dy, dx in [(0,1),(1,0),(1,1),(1,-1)]:
            viz   = np.roll(np.roll(img, dy*r, axis=0), dx*r, axis=1)
            razao = (img + 1.0) / (viz + 1.0)
            # Distância log da razão ao log(φ): zero = razão exata de φ
            distancia = np.abs(np.log(razao) - np.log(PHI))
            campo    += w_n * np.exp(-distancia**2 / 0.08)
            peso     += w_n

    return campo / (peso + 1e-10)

def bip_eco(sinal, n_estagios=5):
    """
    Eco φ em 5 estágios — análogo ao BIP do áudio.
    Cada estágio adiciona eco com delay φ^n e ganho 1/φ^n.
    """
    out = sinal.copy().astype(float)
    for s in range(1, n_estagios):
        delay = max(1, int(round(PHI**s)))
        ganho = 1.0 / PHI**s
        if delay < len(sinal):
            out[delay:] += ganho * sinal[:-delay]
    # Normaliza preservando a escala original
    mx = out.max()
    if mx > 1e-10:
        out = out / mx * sinal.max()
    return out

def bip_imagem(img):
    """Aplica eco BIP φ por linha e depois por coluna."""
    resultado = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        resultado[i, :] = bip_eco(img[i, :].astype(float))
    temp = resultado.copy()
    for j in range(img.shape[1]):
        resultado[:, j] = bip_eco(temp[:, j])
    return np.clip(resultado, 0, 255).astype(np.uint8)

def expmap0(v, c=C_PHI):
    """Mapeia cada pixel para a bola de Poincaré."""
    norma = np.abs(v) + 1e-10
    return np.tanh(np.sqrt(c) * norma) * v / (np.sqrt(c) * norma)

# ── Carrega a imagem ──────────────────────────────────────────────────────────

URL_BRANCH = "https://raw.githubusercontent.com/vitoredsonalphaphi/alpha_phi_manifesto/claude/good-morning-N6f3S/1775593114753.png"
URL_MAIN   = "https://raw.githubusercontent.com/vitoredsonalphaphi/alpha_phi_manifesto/main/1775593114753.png"

img_pil = None
for url in [URL_BRANCH, URL_MAIN]:
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            img_pil = Image.open(BytesIO(resp.content))
            print(f"Imagem carregada: {img_pil.size} modo={img_pil.mode}")
            break
    except Exception as e:
        print(f"Tentativa falhou: {e}")

if img_pil is None:
    raise RuntimeError("Não foi possível carregar a imagem do repositório.")

# Redimensiona para processamento rápido
MAX_DIM = 500
w_orig, h_orig = img_pil.size
escala = min(MAX_DIM / w_orig, MAX_DIM / h_orig, 1.0)
if escala < 1.0:
    img_pil = img_pil.resize((int(w_orig*escala), int(h_orig*escala)), Image.LANCZOS)
    print(f"Redimensionada para: {img_pil.size}")

img_rgb  = np.array(img_pil.convert('RGB'))
img_gray = np.array(img_pil.convert('L')).astype(float)

# ── Processa ──────────────────────────────────────────────────────────────────

print("Calculando campo φ...")
campo = campo_phi_coerencia(img_gray)

print("Aplicando eco BIP φ...")
img_bip = bip_imagem(img_gray)

print("Mapeando para bola de Poincaré...")
img_norm = img_gray / 127.5 - 1.0             # normaliza para [-1, 1]
img_hyp  = expmap0(img_norm)                  # mapeia para bola de Poincaré
img_hyp_v = ((img_hyp + 1) / 2 * 255).astype(np.uint8)

# Diferença BIP vs original
diff = np.abs(img_bip.astype(float) - img_gray)
diff = (diff / diff.max() * 255).astype(np.uint8)

# Overlay: campo φ em dourado sobre original
overlay = img_rgb.copy().astype(float)
mascara = campo / (campo.max() + 1e-10)
overlay[:,:,0] = np.clip(overlay[:,:,0] + mascara * 120, 0, 255)  # R+
overlay[:,:,1] = np.clip(overlay[:,:,1] + mascara *  80, 0, 255)  # G+
overlay[:,:,2] = np.clip(overlay[:,:,2] - mascara *  60, 0, 255)  # B-
overlay = overlay.astype(np.uint8)

# Perfil horizontal: intensidade média × campo φ médio por coluna
perfil_orig  = img_gray.mean(axis=0)
perfil_campo = campo.mean(axis=0)
perfil_bip   = img_bip.astype(float).mean(axis=0)

# ── Visualização ──────────────────────────────────────────────────────────────

GOLD  = "#DAA520"
CYAN  = "#00BFFF"
GREEN = "#00FF88"

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0d1117")

# Grade: 2 linhas × 3 colunas (imagens) + 1 linha perfis
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.15)

def estiliza(ax, titulo, cor=GOLD):
    ax.set_facecolor("#0d1117")
    ax.set_title(titulo, color=cor, fontsize=10, fontweight='bold', pad=6)
    ax.axis('off')

ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img_rgb);      estiliza(ax1, "Original\n(pixel euclidiano)")
ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(campo, cmap='inferno'); estiliza(ax2, "Campo φ — Coerência\n(onde a imagem já ressoa em φ)")
ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(overlay);      estiliza(ax3, "Campo φ sobreposto\n(dourado = alta ressonância φ)")

ax4 = fig.add_subplot(gs[1, 0]); ax4.imshow(img_bip, cmap='gray');   estiliza(ax4, "Eco BIP φ\n(5 estágios por linha e coluna)")
ax5 = fig.add_subplot(gs[1, 1]); ax5.imshow(diff, cmap='hot');        estiliza(ax5, "Δ BIP — O que o eco φ adiciona\n(campo harmônico emergente)")
ax6 = fig.add_subplot(gs[1, 2]); ax6.imshow(img_hyp_v, cmap='gray'); estiliza(ax6, f"expmap0  c=1/φ²={C_PHI:.4f}\n(mapeado para bola de Poincaré)")

# Perfis horizontais
ax7 = fig.add_subplot(gs[2, :])
ax7.set_facecolor("#161b22")
cols = np.arange(len(perfil_orig))
ax7.plot(cols, perfil_orig  / perfil_orig.max(),  color=GOLD,  lw=1.5, label="Intensidade original")
ax7.plot(cols, perfil_campo / perfil_campo.max(), color=CYAN,  lw=1.5, label="Campo φ (coerência média)")
ax7.plot(cols, perfil_bip   / perfil_bip.max(),   color=GREEN, lw=1.0, alpha=0.7, label="Eco BIP φ")
ax7.axhline(1/PHI, color='white', lw=0.7, ls='--', alpha=0.4, label=f"1/φ = {1/PHI:.3f}")
ax7.axhline(C_PHI, color='red',   lw=0.7, ls='--', alpha=0.4, label=f"c = 1/φ² = {C_PHI:.3f}")
ax7.set_title("Perfil horizontal — Intensidade × Campo φ × Eco BIP", color=GOLD, fontsize=10, fontweight='bold')
ax7.tick_params(colors="#8B949E")
for sp in ax7.spines.values(): sp.set_color("#30363d")
ax7.set_facecolor("#161b22")
ax7.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8, loc='upper right')
ax7.grid(True, alpha=0.15)

# Estatísticas
media_campo = campo.mean()
pico_campo  = campo.max()
print(f"\nCoerência φ média : {media_campo:.4f}")
print(f"Coerência φ pico  : {pico_campo:.4f}")
print(f"Δ BIP médio       : {diff.mean():.2f} / 255")
raios = [max(1, int(round(PHI**n))) for n in range(1, 5)]
print(f"Raios φ usados    : {raios}  (φ¹={PHI:.2f} → φ⁴={PHI**4:.2f})")

fig.suptitle(
    f"ALPHA PHI — Campo φ em Imagem Digital  c=1/φ²={C_PHI:.4f}  φ={PHI:.4f}  Florianópolis 2026",
    color=GOLD, fontsize=12, fontweight='bold'
)

plt.savefig("alphaphi_campo_imagem.png", dpi=150, bbox_inches='tight', facecolor="#0d1117")
plt.show()
print("\nGráfico salvo: alphaphi_campo_imagem.png")
print("alpha-phi")
