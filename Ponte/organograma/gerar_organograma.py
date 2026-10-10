"""Gera as imagens do organograma a partir de organograma.yaml.
Uso:  python3 gerar_organograma.py            (usa a data do arquivo)
Saída: imagens/organograma_visao_geral_<data>.png e imagens/ramo_<id>_<data>.png
As versões antigas ficam guardadas (nome com data)."""
import sys, textwrap, pathlib, collections
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

AQUI = pathlib.Path(__file__).parent
CORES = {"demonstrado": "#2e9e5b", "em_uso": "#2a9db0", "reauditar": "#b9b02c", "parcial": "#d99a2b",
         "hipotese": "#4a78c9", "nao_demonstrado": "#d9622b", "refutado": "#c0392b", "aberto": "#8b8f97"}
ROTULO = {"demonstrado": "demonstrado", "em_uso": "em uso", "reauditar": "a reauditar", "parcial": "parcial",
          "hipotese": "hipótese", "nao_demonstrado": "não demonstrado", "refutado": "refutado", "aberto": "em aberto"}
FORMAS = {"mediador": "forma 1 · mediador", "terceira_via": "forma 2 · terceira via", "dialetica": "forma 3 · dialética", "ciclo": "forma 4 · ciclo"}
BG, FG, MUTE = "#10131a", "#e8eaf0", "#9aa3b5"

def carregar():
    d = yaml.safe_load(open(AQUI / "organograma.yaml", encoding="utf-8"))
    return d["meta"], {n["id"]: n for n in d["nos"]}

def filhos(nos, pid): return [n for n in nos.values() if n.get("pai") == pid]

def descendentes(nos, pid):
    for f in filhos(nos, pid):
        yield f
        yield from descendentes(nos, f["id"])

def linhas_no(n, prof, larg_t=44):
    titulo = textwrap.wrap(n["titulo"], max(24, larg_t - prof * 4)) or [""]
    extra = []
    if n.get("forma"): extra.append(FORMAS[n["forma"]])
    if n.get("fonte"): extra.append("fonte: " + n["fonte"])
    if n.get("nota"): extra.append(n["nota"])
    nota = textwrap.wrap(" · ".join(extra), max(30, larg_t + 12 - prof * 4)) if extra else []
    return titulo, nota

def desenhar_ramo(nos, raiz_id, saida, meta, mostrar_raiz=True):
    W = 7.4
    itens = [(nos[raiz_id], 0)]
    def rec(pid, prof):
        for f in filhos(nos, pid):
            itens.append((f, prof)); rec(f["id"], prof + 1)
    rec(raiz_id, 1)
    alturas = []
    for n, prof in itens:
        t, nt = linhas_no(n, prof)
        alturas.append(0.20 + 0.235 * len(t) + 0.17 * len(nt))
    H = 0.95 + sum(alturas) + 0.55
    fig = plt.figure(figsize=(W, H), dpi=140, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.text(0.2, 0.32, f"{meta['titulo']} — {nos[raiz_id]['titulo'][:48] if raiz_id != 'raiz' else meta['subtitulo']}", color=FG, fontsize=11.5, fontweight="bold", va="center")
    ax.text(0.2, 0.62, f"{meta['versao']} · {meta['data']} · cada nó tem a cor do seu status", color=MUTE, fontsize=8, va="center")
    y = 0.95; ypos = {}
    for (n, prof), h in zip(itens, alturas):
        x0 = 0.25 + prof * 0.30
        cor = CORES[n["status"]]
        # conector
        if prof > 0 and n.get("pai") in ypos:
            px, py = ypos[n["pai"]]
            ax.plot([px, px, x0 - 0.02], [py, y + h / 2, y + h / 2], color="#3a4155", lw=1.0, zorder=1)
        # chip de status
        ax.add_patch(FancyBboxPatch((x0, y + 0.04), 1.02, 0.25, boxstyle="round,pad=0.01,rounding_size=0.06", fc=cor, ec="none", zorder=2))
        ax.text(x0 + 0.51, y + 0.165, ROTULO[n["status"]], color="#0b0d12", fontsize=7.4, fontweight="bold", ha="center", va="center", zorder=3)
        t, nt = linhas_no(n, prof)
        tx = x0 + 1.12
        yy = y + 0.17
        for l in t:
            ax.text(tx, yy, l, color=FG, fontsize=9.6, fontweight="bold" if prof <= 1 else "normal", va="center"); yy += 0.235
        yy -= 0.235 - 0.19
        for l in nt:
            ax.text(tx, yy, l, color=MUTE, fontsize=7.6, va="center"); yy += 0.17
        ypos[n["id"]] = (x0 + 0.02, y + h / 2)
        y += h
    # legenda
    x = 0.2; ly = H - 0.28
    for k, c in CORES.items():
        ax.add_patch(FancyBboxPatch((x, ly - 0.07), 0.14, 0.14, boxstyle="round,pad=0.005,rounding_size=0.03", fc=c, ec="none"))
        ax.text(x + 0.19, ly, ROTULO[k], color=MUTE, fontsize=6.6, va="center"); x += 0.19 + 0.075 * len(ROTULO[k]) + 0.16
        if x > W - 0.9: x = 0.2; ly += 0.2
    fig.savefig(saida, facecolor=BG); plt.close(fig)

def desenhar_visao_geral(nos, saida, meta):
    ramos = filhos(nos, "raiz")
    W, H = 7.4, 1.5 + 1.05 * len(ramos) + 1.05
    fig = plt.figure(figsize=(W, H), dpi=140, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.text(W / 2, 0.38, meta["titulo"] + "  ·  organograma", color=FG, fontsize=13, fontweight="bold", ha="center", va="center")
    ax.text(W / 2, 0.70, meta["subtitulo"], color=MUTE, fontsize=8.2, ha="center", va="center")
    ax.text(W / 2, 0.98, f"{meta['versao']} · {meta['data']}", color=MUTE, fontsize=7.4, ha="center", va="center")
    cx = W / 2; y0 = 1.25
    ax.add_patch(FancyBboxPatch((cx - 1.05, y0), 2.1, 0.5, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#1b2030", ec=CORES["hipotese"], lw=1.6))
    ax.text(cx, y0 + 0.25, "RAIZ  ·  a Ponte [f ○ c]", color=FG, fontsize=10.5, fontweight="bold", ha="center", va="center")
    y = y0 + 0.5 + 0.35
    ax.plot([cx, cx], [y0 + 0.5, y + 1.05 * len(ramos) - 0.55], color="#3a4155", lw=1.2, zorder=1)
    for r in ramos:
        cont = collections.Counter(d["status"] for d in descendentes(nos, r["id"]))
        n_tot = sum(cont.values())
        ax.plot([cx, cx - 0.35], [y + 0.32, y + 0.32], color="#3a4155", lw=1.2)
        ax.add_patch(FancyBboxPatch((0.25, y), W - 0.5, 0.88, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#171b27", ec="#2b3247", lw=1.0, zorder=2))
        ax.text(0.45, y + 0.25, r["titulo"], color=FG, fontsize=10, fontweight="bold", va="center", zorder=3)
        xs = 0.45
        for st in ["demonstrado", "em_uso", "reauditar", "parcial", "hipotese", "nao_demonstrado", "refutado", "aberto"]:
            if cont.get(st):
                rot = f"{cont[st]} {ROTULO[st]}"; wchip = 0.30 + 0.066 * len(rot)
                ax.add_patch(FancyBboxPatch((xs, y + 0.46), wchip, 0.26, boxstyle="round,pad=0.01,rounding_size=0.06", fc=CORES[st], ec="none", zorder=3))
                ax.text(xs + wchip / 2, y + 0.59, rot, color="#0b0d12", fontsize=6.3, fontweight="bold", ha="center", va="center", zorder=4)
                xs += wchip + 0.07
        ax.text(W - 0.45, y + 0.25, f"{n_tot} nós", color=MUTE, fontsize=7.6, ha="right", va="center", zorder=3)
        y += 1.05
    fig.savefig(saida, facecolor=BG); plt.close(fig)

if __name__ == "__main__":
    meta, nos = carregar(); data = meta["data"]
    pasta = AQUI / "imagens"; pasta.mkdir(exist_ok=True)
    desenhar_visao_geral(nos, pasta / f"organograma_visao_geral_{data}.png", meta)
    for r in filhos(nos, "raiz"):
        desenhar_ramo(nos, r["id"], pasta / f"ramo_{r['id']}_{data}.png", meta)
    print("Geradas:", *sorted(p.name for p in pasta.glob(f"*{data}.png")), sep="\n  ")
