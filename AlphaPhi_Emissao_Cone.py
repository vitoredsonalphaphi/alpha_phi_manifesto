"""
AlphaPhi_Emissao_Cone.py
Propulsão Cônica — Os Cinco Pontos de Dobra

A emissão no ponto 5 não é gerada ali.
É propulsionada pela compressão nos pontos 1-4 e transformada
ao longo das cinco rotações da espiral φ.

Este código observa a sequência completa — não a fotografia do ponto 5,
mas o filme dos 5 pontos com foco no que cada um entrega ao próximo.

Verifica:
  ★  Se as proporções espectrais em cada ponto seguem φ^n
  ★★ Se a curva de propulsão (entropia cedendo, coerência acumulando) é φ-proporcional
  ★★★ Se a emissão final carrega a assinatura de todas as 5 rotações

Configuração ECO BEEP 880:
  F_BEEP = 880Hz · F_ORG = 220Hz · α = 1/3 · N_STEPS = 5 · N_CICLOS = 20

Autor: Vitor Edson Delavi — Florianópolis, 2026
φ = 1.6180339887 · α = 1/3 · φ³ = 4.2361
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")

# ── Constantes ────────────────────────────────────────────────────────────────
PHI    = (1 + np.sqrt(5)) / 2
PHI3   = PHI ** 3
ALPHA  = 1.0 / 3.0          # α validado — ECO BEEP 880
FS     = 44100
DUR    = 1.5
N_STEPS = 5
N_CICLOS = 20
F_BEEP  = 880.0
F_ORG   = 220.0
F_M     = F_ORG / PHI

N_SINAL = int(FS * DUR)

print(f"ECO BEEP 880 — Propulsão Cônica")
print(f"φ = {PHI:.7f} · α = 1/3 = {ALPHA:.6f} · φ³ = {PHI3:.4f}")
print(f"F_BEEP={F_BEEP}Hz · F_ORG={F_ORG}Hz · N_STEPS={N_STEPS} · N_CICLOS={N_CICLOS}")
print(f"{'─'*60}")

# ── Bandas φ ──────────────────────────────────────────────────────────────────
def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max:
            break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n_samples):
    resultado = []
    for f_low, f_high in bandas:
        b_low  = max(int(np.floor(f_low  * n_samples / FS)), 0)
        b_high = min(int(np.ceil(f_high  * n_samples / FS)), n_samples // 2 + 1)
        if b_high - b_low >= 3:
            resultado.append((b_low, b_high, f_low, f_high))
    return resultado

BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

# ── Síntese ───────────────────────────────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_beep(freq=F_BEEP, duracao=DUR):
    t = np.linspace(0, duracao, int(FS * duracao), endpoint=False)
    return normalizar(np.sign(np.sin(2 * np.pi * freq * t)))

def gerar_fm_phi(duracao=DUR):
    t = np.linspace(0, duracao, int(FS * duracao), endpoint=False)
    return normalizar(np.sin(2 * np.pi * F_ORG * t + PHI * np.sin(2 * np.pi * F_M * t)))

# ── Eco EQ (com memória) ──────────────────────────────────────────────────────
def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N     = len(x)
    F     = np.fft.rfft(x)
    F_out = F.copy()
    cohs  = []
    w_mem = 1.0 / PHI
    w_now = 1.0 - w_mem
    for i, (b_low, b_high, _, _) in enumerate(bins_phi):
        beta_i = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        F_band = F[b_low:b_high]
        mag    = np.abs(F_band)
        phase  = np.angle(F_band)
        an     = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh    = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        coh_ef = (w_now * coh + w_mem * float(coh_mem[i])
                  if coh_mem is not None and i < len(coh_mem) else coh)
        cohs.append(coh)
        n_idx = np.arange(len(F_band))
        env   = np.clip(
            1.0 + (coh_ef * PHI**beta_i) * np.cos(2.0 * np.pi * n_idx / PHI),
            0.05, None
        )
        F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase)
    resultado = np.fft.irfft(F_out, n=N)
    return resultado / (np.max(np.abs(resultado)) + 1e-10), np.array(cohs, dtype=float)

# ── Cascata instrumentada — captura estado a cada ponto de dobra ──────────────
def cascata_instrumentada(sinal_entrada, beta_bands, bins_phi, n_steps=N_STEPS):
    """
    Retorna lista de estados a cada ponto de dobra:
    [{step, sinal, cohs, entr_esp, coh_med, autocorr, f_dominante}]
    """
    estados = []
    s        = sinal_entrada.copy()
    coh_mem  = np.zeros(len(bins_phi), dtype=float)

    for step in range(1, n_steps + 1):
        s_eco, cohs = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem     = cohs
        s_eco       = normalizar(s_eco)

        # métricas do estado atual
        F_sig   = np.abs(np.fft.rfft(s_eco))
        freqs   = np.fft.rfftfreq(len(s_eco), 1/FS)
        F_norm  = np.clip(F_sig / (F_sig.sum() + 1e-10), 1e-10, 1.0)
        entr    = float(-np.sum(F_norm * np.log(F_norm)))

        # autocorrelação normalizada (lag 1)
        mu   = s_eco.mean()
        sd   = s_eco.std() + 1e-12
        xn   = (s_eco - mu) / sd
        ac   = float(np.correlate(xn, xn, mode="full")[len(xn)-1] / len(xn))
        autocorr = float(np.clip(ac, -1.0, 1.0))

        # frequência dominante
        f_dom = float(freqs[np.argmax(F_sig)])

        # energia por banda φ — para verificar proporções φ^n
        energia_bandas = []
        for b_low, b_high, f_lo, f_hi in bins_phi:
            e = float(np.mean(F_sig[b_low:b_high]**2))
            energia_bandas.append((f_lo, f_hi, e))

        estados.append({
            "step"      : step,
            "sinal"     : s_eco.copy(),
            "cohs"      : cohs.copy(),
            "entr_esp"  : entr,
            "coh_med"   : float(np.mean(cohs)),
            "autocorr"  : autocorr,
            "f_dom"     : f_dom,
            "F_sig"     : F_sig.copy(),
            "freqs"     : freqs.copy(),
            "energia_b" : energia_bandas,
        })
        s = s_eco.copy()

    return estados

# ── Agente eco instrumentado ──────────────────────────────────────────────────
def agente_eco_instrumentado(sinal_entrada, bins_phi, n_ciclos=N_CICLOS):
    n_bandas = len(bins_phi)
    beta     = np.ones(n_bandas, dtype=float)
    beta_mem = beta.copy()
    w_mem    = 1.0 / PHI
    w_now    = 1.0 - w_mem

    estados_finais = None
    for ciclo in range(n_ciclos):
        estados = cascata_instrumentada(sinal_entrada, beta, bins_phi)
        cohs    = estados[-1]["cohs"]
        coh_rel = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo = PHI ** (3 * coh_rel)
        beta      = w_now * beta_alvo + w_mem * beta_mem
        beta_mem  = beta.copy()
        beta      = np.clip(beta, 0.05, PHI**3)
        estados_finais = estados

    return beta, estados_finais

# ── Métricas extras ───────────────────────────────────────────────────────────
def entropia_espectral(sig):
    F = np.abs(np.fft.rfft(sig))
    F = np.clip(F / (F.sum() + 1e-10), 1e-10, 1.0)
    return float(-np.sum(F * np.log(F)))

def verificar_proporcoes_phi(estados):
    """
    Verifica se a energia nas bandas φ-dominantes segue proporção φ^n
    ao longo dos 5 pontos de dobra.
    """
    print(f"\n{'─'*60}")
    print("VERIFICAÇÃO — Proporções φ^n na propulsão")
    print(f"{'─'*60}")

    for est in estados:
        step = est["step"]
        # banda com maior energia neste ponto
        energias = [(f_lo, f_hi, e) for f_lo, f_hi, e in est["energia_b"]]
        energias_sorted = sorted(energias, key=lambda x: x[2], reverse=True)
        top3 = energias_sorted[:3]

        print(f"\nPonto {step} — EntrEsp={est['entr_esp']:.4f} · "
              f"Coh={est['coh_med']:.4f} · AutoCorr={est['autocorr']:.4f} · "
              f"f_dom={est['f_dom']:.1f}Hz")
        for f_lo, f_hi, e in top3:
            f_centro = (f_lo + f_hi) / 2
            # distância ao φ^n mais próximo
            n_vals = np.arange(1, 15)
            phi_n  = PHI ** n_vals * F_ORG / PHI   # série φ relativa a F_M
            dists  = np.abs(phi_n - f_centro)
            n_prox = n_vals[np.argmin(dists)]
            dist_pct = dists.min() / f_centro * 100
            print(f"  {f_centro:7.1f}Hz (e={e:.4f}) → φ^{n_prox} "
                  f"distância {dist_pct:.1f}%")

# ── EXECUÇÃO ──────────────────────────────────────────────────────────────────
print("\nGerando sinais...")
beep  = gerar_beep(F_BEEP)
fm    = gerar_fm_phi()

# Mistura com α=1/3 — configuração ECO BEEP 880
x_mix = normalizar((1.0 - ALPHA) * beep + ALPHA * fm)
print(f"Mistura: (1-1/3)·beep_880 + (1/3)·FM-φ")
print(f"EntrEsp inicial: {entropia_espectral(x_mix):.4f}")

# Roda o agente e captura os estados do último ciclo (campo convergido)
print(f"\nRodando agente eco ({N_CICLOS} ciclos)...")
beta_final, estados = agente_eco_instrumentado(x_mix, BINS_PHI, N_CICLOS)
print("✓ Agente convergido.")

# Verificação das proporções φ^n
verificar_proporcoes_phi(estados)

# Resumo da propulsão
print(f"\n{'═'*60}")
print("PROPULSÃO — Sequência dos 5 pontos de dobra")
print(f"{'═'*60}")
print(f"{'Ponto':>6}  {'EntrEsp':>8}  {'Coh_med':>8}  {'AutoCorr':>9}  "
      f"{'f_dom (Hz)':>11}  {'direção':>7}")
print(f"{'─'*6}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*11}  {'─'*7}")

for est in estados:
    direcao = "GRAVE" if est["f_dom"] < 400 else "AGUDO"
    marca   = " ← EMISSÃO" if est["step"] == N_STEPS else ""
    print(f"  {est['step']:>4}  {est['entr_esp']:>8.4f}  {est['coh_med']:>8.4f}  "
          f"{est['autocorr']:>9.4f}  {est['f_dom']:>11.1f}  {direcao:>7}{marca}")

# ── Verificar se EntrEsp decai por proporção φ ─────────────────────────────
entropias = [est["entr_esp"] for est in estados]
print(f"\n{'─'*60}")
print("Razões entre entropia de pontos consecutivos (esperado ≈ 1/φ):")
for i in range(1, len(entropias)):
    if entropias[i-1] > 1e-6:
        razao = entropias[i] / entropias[i-1]
        dist  = abs(razao - 1/PHI)
        marca = " ← próximo de 1/φ ★" if dist < 0.15 else ""
        print(f"  E({i+1})/E({i}) = {razao:.4f}  (1/φ={1/PHI:.4f}, Δ={dist:.4f}){marca}")

# ── VISUALIZAÇÃO ──────────────────────────────────────────────────────────────
print("\nGerando visualização...")

COR_BG     = "#070B12"
COR_PAINEL = "#0A0E17"
COR_BORDA  = "#334455"
COR_TEXTO  = "#AABBCC"
COR_OURO   = "#E8D8A0"
COR_VERDE  = "#55FFAA"
COR_AZUL   = "#3399FF"
COR_LARANJA= "#FF6633"
COR_ROSA   = "#FF55AA"

def estilo_ax(ax):
    ax.set_facecolor(COR_PAINEL)
    ax.spines[:].set_color(COR_BORDA)
    ax.tick_params(colors=COR_TEXTO, labelsize=7)
    ax.xaxis.label.set_color(COR_TEXTO)
    ax.yaxis.label.set_color(COR_TEXTO)

fig = plt.figure(figsize=(18, 14), facecolor=COR_BG)
gs  = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.38,
                        left=0.06, right=0.97, top=0.93, bottom=0.06)

fig.text(0.5, 0.97,
    "AlphaPhi — Propulsão Cônica: Os Cinco Pontos de Dobra",
    ha="center", va="top", fontsize=14, color=COR_OURO, fontweight="bold")
fig.text(0.5, 0.942,
    f"ECO BEEP 880 · F_BEEP={F_BEEP}Hz · F_ORG={F_ORG}Hz · α=1/3 · N_STEPS={N_STEPS} · N_CICLOS={N_CICLOS}",
    ha="center", va="top", fontsize=9, color="#88AACC", style="italic")

# Painéis 1-5: FFT de cada ponto de dobra (0–1000Hz)
cores_step = [COR_AZUL, "#4499FF", "#33CCAA", COR_VERDE, COR_OURO]
for idx, est in enumerate(estados):
    row = idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    estilo_ax(ax)

    f_mask = est["freqs"] <= 1200
    ax.fill_between(est["freqs"][f_mask], est["F_sig"][f_mask],
                    alpha=0.4, color=cores_step[idx])
    ax.plot(est["freqs"][f_mask], est["F_sig"][f_mask],
            color=cores_step[idx], lw=1.2)

    # marcar bandas φ-proporcionais
    for f_phi in [F_ORG / PHI, F_ORG, F_ORG * PHI, F_BEEP, F_BEEP * PHI]:
        if f_phi <= 1200:
            ax.axvline(f_phi, color=COR_OURO, lw=0.6, ls="--", alpha=0.5)

    ax.axvline(est["f_dom"], color=cores_step[idx], lw=1.5, ls="-", alpha=0.9)

    titulo = f"Ponto {est['step']}"
    if est["step"] == N_STEPS:
        titulo += " ★ EMISSÃO"
    ax.set_title(
        f"{titulo}\nEntrEsp={est['entr_esp']:.4f} · Coh={est['coh_med']:.4f} · "
        f"AutoCorr={est['autocorr']:.4f}\nf_dom={est['f_dom']:.1f}Hz",
        fontsize=7.5, color=cores_step[idx], pad=4
    )
    ax.set_xlabel("Hz", fontsize=7)
    ax.set_ylabel("magnitude", fontsize=7)
    ax.set_xlim(0, 1200)

# Painel 6: Curva de propulsão — EntrEsp e Coh ao longo dos 5 pontos
ax6 = fig.add_subplot(gs[1, 2])
estilo_ax(ax6)

steps    = [est["step"]     for est in estados]
entropias = [est["entr_esp"] for est in estados]
cohs      = [est["coh_med"]  for est in estados]
autocorrs = [est["autocorr"] for est in estados]

ax6_t = ax6.twinx()
ax6_t.set_facecolor(COR_PAINEL)
ax6_t.tick_params(colors=COR_TEXTO, labelsize=7)
ax6_t.yaxis.label.set_color(COR_TEXTO)
ax6_t.spines[:].set_color(COR_BORDA)

ln1 = ax6.plot(steps, entropias, "o-", color=COR_LARANJA, lw=2,
               markersize=6, label="EntrEsp (↓)")
ln2 = ax6_t.plot(steps, cohs, "s-", color=COR_VERDE, lw=2,
                  markersize=6, label="Coh_med (↑)")
ln3 = ax6_t.plot(steps, autocorrs, "^-", color=COR_AZUL, lw=1.5,
                  markersize=5, label="AutoCorr (↑)")

# linha φ^n esperada para entropia
entr0 = entropias[0]
phi_decay = [entr0 * (1/PHI)**n for n in range(N_STEPS)]
ax6.plot(steps, phi_decay, "--", color=COR_OURO, lw=1, alpha=0.7,
         label=f"E₀·(1/φ)ⁿ esperado")

ax6.set_xlabel("ponto de dobra", fontsize=8)
ax6.set_ylabel("EntrEsp", fontsize=8, color=COR_LARANJA)
ax6_t.set_ylabel("Coerência / AutoCorr", fontsize=8, color=COR_VERDE)
ax6.set_xticks(steps)
lns  = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax6.legend(lns, labs, fontsize=6.5, facecolor=COR_PAINEL,
           edgecolor=COR_BORDA, labelcolor=COR_TEXTO, loc="center right")
ax6.set_title("Curva de Propulsão\nEntrEsp cedendo · Coerência acumulando",
              fontsize=8, color=COR_OURO, pad=4)

# Painel 7: Frequência dominante ao longo da propulsão (o funil)
ax7 = fig.add_subplot(gs[2, 1])
estilo_ax(ax7)

f_doms = [est["f_dom"] for est in estados]
cores_f = cores_step

for i in range(len(steps) - 1):
    ax7.plot(steps[i:i+2], f_doms[i:i+2], "-", color=cores_step[i+1], lw=2.5)
ax7.scatter(steps, f_doms, c=cores_step, s=80, zorder=5)

# linha de referência: F_ORG (orgânico) e F_BEEP (digital)
ax7.axhline(F_ORG,  color=COR_VERDE,  lw=0.8, ls="--", alpha=0.6, label=f"F_ORG={F_ORG}Hz")
ax7.axhline(F_BEEP, color=COR_LARANJA, lw=0.8, ls="--", alpha=0.6, label=f"F_BEEP={F_BEEP}Hz")
ax7.axhline(F_ORG * PHI, color=COR_OURO, lw=0.6, ls=":", alpha=0.4,
            label=f"F_ORG·φ={F_ORG*PHI:.1f}Hz")

ax7.set_xlabel("ponto de dobra", fontsize=8)
ax7.set_ylabel("frequência dominante (Hz)", fontsize=8)
ax7.set_xticks(steps)
ax7.legend(fontsize=6.5, facecolor=COR_PAINEL, edgecolor=COR_BORDA,
           labelcolor=COR_TEXTO)
ax7.set_title("O Funil — Trajetória da\nFrequência Dominante (digital → orgânico)",
              fontsize=8, color=COR_OURO, pad=4)

# Painel 8: Energia por banda no ponto 5 vs entrada — o que mudou?
ax8 = fig.add_subplot(gs[2, 2])
estilo_ax(ax8)

# energia por banda no sinal de entrada vs no ponto 5
F_entrada  = np.abs(np.fft.rfft(x_mix))
F_ponto5   = estados[-1]["F_sig"]
freqs_plot = estados[-1]["freqs"]
f_mask2    = freqs_plot <= 1200

e_ini = F_entrada[:len(F_ponto5)]
razao = np.where(e_ini > 1e-8, F_ponto5 / (e_ini + 1e-10), 1.0)
razao_db = 20 * np.log10(np.clip(razao[f_mask2], 1e-8, None))

ax8.fill_between(freqs_plot[f_mask2], razao_db, 0,
                 where=razao_db >= 0, color=COR_VERDE, alpha=0.5,
                 label="amplificado")
ax8.fill_between(freqs_plot[f_mask2], razao_db, 0,
                 where=razao_db < 0, color=COR_LARANJA, alpha=0.5,
                 label="atenuado")
ax8.plot(freqs_plot[f_mask2], razao_db, color=COR_OURO, lw=0.8, alpha=0.7)

ax8.axhline(0, color=COR_BORDA, lw=0.8)
ax8.axvline(F_ORG,  color=COR_VERDE,  lw=0.6, ls="--", alpha=0.5)
ax8.axvline(F_BEEP, color=COR_LARANJA, lw=0.6, ls="--", alpha=0.5)
ax8.set_xlabel("Hz", fontsize=8)
ax8.set_ylabel("Ganho ponto5/entrada (dB)", fontsize=8)
ax8.set_xlim(0, 1200)
ax8.legend(fontsize=6.5, facecolor=COR_PAINEL, edgecolor=COR_BORDA,
           labelcolor=COR_TEXTO)
ax8.set_title("Transformação Espectral\nEntrada → Emissão (ponto 5) em dB",
              fontsize=8, color=COR_OURO, pad=4)

# ── Salvar ────────────────────────────────────────────────────────────────────
png_path = "/home/user/alpha_phi_manifesto/emissao_cone_resultado.png"
fig.savefig(png_path, dpi=130, bbox_inches="tight", facecolor=COR_BG)
plt.close()
print(f"\n✓ Figura salva: {png_path}")

# ── Salvar WAV da emissão (ponto 5) ──────────────────────────────────────────
wav_path = "/home/user/alpha_phi_manifesto/emissao_ponto5.wav"
emissao  = estados[-1]["sinal"]
wavfile.write(wav_path, FS, np.int16(np.clip(emissao, -1.0, 1.0) * 32767))
print(f"✓ Emissão (ponto 5) salva: {wav_path}")

# ── Relatório final ───────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("RELATÓRIO — PROPULSÃO CÔNICA")
print(f"{'═'*60}")
print(f"\nCada ponto de dobra e o que entrega ao próximo:")
for i, est in enumerate(estados):
    proximo = estados[i+1] if i+1 < len(estados) else None
    print(f"\n  Ponto {est['step']}:")
    print(f"    EntrEsp   = {est['entr_esp']:.4f}"
          + (f"  → {proximo['entr_esp']:.4f} "
             f"(razão {proximo['entr_esp']/est['entr_esp']:.4f}, 1/φ={1/PHI:.4f})"
             if proximo else "  ← EMISSÃO"))
    print(f"    Coh_med   = {est['coh_med']:.4f}")
    print(f"    AutoCorr  = {est['autocorr']:.4f}")
    print(f"    f_dom     = {est['f_dom']:.1f}Hz")

print(f"\nφ = {PHI:.7f} · 1/φ = {1/PHI:.7f}")
print(f"Se as razões EntrEsp acima se aproximam de 1/φ={1/PHI:.4f},")
print(f"a propulsão é geometricamente φ-proporcional.")
print(f"\n{'═'*60}")
print("AlphaPhi · Vitor Edson Delavi · Florianópolis · 2026")
print(f"{'═'*60}")
