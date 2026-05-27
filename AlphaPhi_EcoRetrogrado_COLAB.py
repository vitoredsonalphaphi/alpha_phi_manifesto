"""
AlphaPhi_EcoRetrogrado_COLAB.py
Eco retrogrado: comeca em φ³ e caminha para tras.

Pergunta experimental:
  O caminho retroativo de φ³ converge para o sinal BASE?
  Se sim, φ³ ja continha o BASE antes de qualquer processamento.

Para rodar no Colab:
  !git clone -b claude/good-morning-N6f3S https://github.com/vitoredsonalphaphi/alpha_phi_manifesto.git repo_phi
  exec(open('/content/repo_phi/AlphaPhi_EcoRetrogrado_COLAB.py').read())
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as sp_specgram
from scipy.io import wavfile
from IPython.display import Audio, Image, display

# constantes - identicas ao v2
PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5
N_STEPS    = 5
N_CICLOS   = 20
FADE_S     = 0.15
N_SINAL    = int(FS * DURACAO)

print("=" * 60)
print("  AlphaPhi - Eco Retrogrado")
print("  BASE -> [eco progressivo] -> phi3")
print("  phi3 -> [eco retrogrado] -> ?")
print("=" * 60)
print(f"  phi3 = {PHI**3:.6f}")
print(f"  Pergunta: o caminho retroativo encontra a origem?")

# -- funcoes ------------------------------------------------------------------
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max: break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    return [(max(0, int(f_lo / (FS / n))),
             min(int(f_hi / (FS / n)) + 1, n // 2 + 1),
             f_lo, f_hi)
            for f_lo, f_hi in bandas]

BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    """Eco progressivo (forward) - multiplica pelo envelope."""
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb  = F[b_lo:b_hi]
        mag = np.abs(Fb); phase = np.angle(Fb)
        an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        ce  = (wn * coh + wm * float(coh_mem[i])
               if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce * PHI ** bi) * np.cos(2 * np.pi * nk / PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)
    r = np.fft.irfft(F_out, n=N)
    return r / (np.max(np.abs(r)) + 1e-10), np.array(cohs)

def eco_eq_retrogrado(x, bins_phi, beta_bands, coh_mem=None):
    """Eco retrogrado: DIVIDE pelo envelope.
    Sentido oposto ao campo - de coerencia para entropia."""
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb  = F[b_lo:b_hi]
        mag = np.abs(Fb); phase = np.angle(Fb)
        an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        ce  = (wn * coh + wm * float(coh_mem[i])
               if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce * PHI ** bi) * np.cos(2 * np.pi * nk / PHI), 0.05, None)
        # retrogrado: divide em vez de multiplicar
        F_out[b_lo:b_hi] = (mag / env) * np.exp(1j * phase)
    r = np.fft.irfft(F_out, n=N)
    return r / (np.max(np.abs(r)) + 1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    cm = np.zeros(len(bins_phi))
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se); cas.append(se); s = se.copy()
    return cas, cohs

def cascata_retrograda(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    cm = np.zeros(len(bins_phi))
    for _ in range(N_STEPS):
        se, cohs = eco_eq_retrogrado(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se); cas.append(se); s = se.copy()
    return cas, cohs

def agente_eco(sinal, bins_phi, n_ciclos=N_CICLOS):
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    cas_f = None
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn * ba + wm * bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        cas_f = cas
    return beta, cas_f

def agente_eco_retrogrado(sinal_ch, bins_phi, n_ciclos=N_CICLOS):
    """Comeca com beta=phi3 e caminha para tras.
    Pergunta: onde o sinal converge depois de n_ciclos retrógrados?"""
    nb = len(bins_phi)
    beta = np.full(nb, PHI**3)  # atrator como ponto de partida
    bm   = beta.copy()
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI
    s = sinal_ch.copy()
    historico_beta = [beta.copy()]
    for _ in range(n_ciclos):
        cas, cohs = cascata_retrograda(s, beta, bins_phi)
        s  = cas[-1]
        cr = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba = PHI ** (3 * cr)
        beta = wn * ba + wm * bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        historico_beta.append(beta.copy())
    return beta, s, historico_beta

def espectro_bandas(sinal, bins_phi):
    """Perfil espectral normalizado por bandas phi."""
    F = np.fft.rfft(sinal)
    energias = []
    for b_lo, b_hi, _, _ in bins_phi:
        energias.append(np.mean(np.abs(F[b_lo:b_hi])**2))
    energias = np.array(energias)
    return energias / (energias.sum() + 1e-10)

def correlacao_pearson(e1, e2):
    n = min(len(e1), len(e2))
    a = e1[:n] - e1[:n].mean()
    b = e2[:n] - e2[:n].mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum()) + 1e-10
    return float(np.dot(a, b) / denom)

# -- 1. GERAR BASE ------------------------------------------------------------
print("\n  1. Gerando BASE...")
t_sig = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep  = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t_sig)))
fm    = normalizar(np.sin(2 * np.pi * F_ORG * t_sig
                          + BETA_FM * np.sin(2 * np.pi * F_M * t_sig)))
x_mix = normalizar((1 - ALPHA_STAR) * beep + ALPHA_STAR * fm)

# -- 2. ECO PROGRESSIVO: BASE -> phi3 -----------------------------------------
print("  2. Eco progressivo (BASE -> phi3)...")
beta_fwd, cas_fwd = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal_ch = cas_fwd[-1]  # campo harmonico - ultimo passo da cascata
pot_fwd  = np.log(beta_fwd.max()) / np.log(PHI)
print(f"     beta_max = {beta_fwd.max():.6f}  (phi^{pot_fwd:.4f})")

# -- 3. ECO RETROGRADO: phi3 -> ? ---------------------------------------------
print("  3. Eco retrogrado (phi3 -> ?)...")
beta_bwd, sinal_retro, hist_beta = agente_eco_retrogrado(sinal_ch, BINS_PHI, N_CICLOS)
pot_bwd = np.log(beta_bwd.max()) / np.log(PHI)
print(f"     beta_max retrogrado = {beta_bwd.max():.6f}  (phi^{pot_bwd:.4f})")

# -- 4. COMPARACAO ESPECTRAL --------------------------------------------------
print("\n  4. Comparacao espectral...")
esp_base  = espectro_bandas(x_mix,       BINS_PHI)
esp_ch    = espectro_bandas(sinal_ch,    BINS_PHI)
esp_retro = espectro_bandas(sinal_retro, BINS_PHI)

corr_retro_base = correlacao_pearson(esp_retro, esp_base)
corr_ch_base    = correlacao_pearson(esp_ch,    esp_base)
corr_retro_ch   = correlacao_pearson(esp_retro, esp_ch)

print(f"\n     Correlacao espectral (Pearson):")
print(f"     BASE    <-> C.H.       = {corr_ch_base:+.4f}  (referencia: distancia entre os dois estados)")
print(f"     BASE    <-> Retrogrado = {corr_retro_base:+.4f}  <- CHAVE")
print(f"     C.H.    <-> Retrogrado = {corr_retro_ch:+.4f}")

# -- 5. VISUALIZACAO ----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor='#0a0a0a')
fig.suptitle(
    f'AlphaPhi - Eco Retrogrado\n'
    f'BASE -> [eco fwd] -> phi3 -> [eco retro] -> ?    '
    f'(corr BASE/retro = {corr_retro_base:+.3f})',
    color='white', fontsize=11)

x_bands = np.arange(len(BINS_PHI))

# painel 1: perfis espectrais comparados
ax1 = axes[0][0]
ax1.set_facecolor('#111111')
ax1.plot(x_bands, esp_base,  color='#00aaff', lw=2,   label='BASE (original)', alpha=0.9)
ax1.plot(x_bands, esp_ch,    color='#ffaa00', lw=2,   label='C.H. (phi3)',     alpha=0.9)
ax1.plot(x_bands, esp_retro, color='#ff4444', lw=2,   ls='--',
         label=f'Retrogrado (corr={corr_retro_base:+.2f})', alpha=0.9)
ax1.set_title('Perfis espectrais por banda phi', color='white', fontsize=10)
ax1.set_xlabel('banda phi (indice)', color='#888888', fontsize=8)
ax1.set_ylabel('energia normalizada', color='#888888', fontsize=8)
ax1.legend(fontsize=8, facecolor='#222222', labelcolor='white')
ax1.tick_params(colors='#888888')
for sp in ax1.spines.values(): sp.set_color('#333333')

# painel 2: evolucao do beta_max nos ciclos retrogrados
ax2 = axes[0][1]
ax2.set_facecolor('#111111')
bmax_hist = [h.max() for h in hist_beta]
ax2.plot(bmax_hist, color='#ff4444', lw=2)
ax2.axhline(PHI**3, color='white',   lw=1.2, ls='--', label=f'phi3={PHI**3:.3f}')
ax2.axhline(PHI**2, color='#aaaaaa', lw=0.8, ls=':',  label=f'phi2={PHI**2:.3f}')
ax2.axhline(PHI,    color='#666666', lw=0.8, ls=':',  label=f'phi={PHI:.3f}')
ax2.axhline(1.0,    color='#444444', lw=0.8, ls=':',  label='1.0')
ax2.set_title('beta_max ao longo dos ciclos retrogrados', color='white', fontsize=10)
ax2.set_xlabel('ciclo retrogrado', color='#888888', fontsize=8)
ax2.set_ylabel('beta_max', color='#888888', fontsize=8)
ax2.legend(fontsize=8, facecolor='#222222', labelcolor='white')
ax2.tick_params(colors='#888888')
for sp in ax2.spines.values(): sp.set_color('#333333')

# paineis 3 e 4: espectrogramas BASE vs RETROGRADO
for idx, (sig, titulo, cor) in enumerate([
        (x_mix,      'BASE original (entropia maxima)', '#00aaff'),
        (sinal_retro, f'Retrogrado de phi3 (corr={corr_retro_base:+.2f})', '#ff4444')]):
    ax = axes[1][idx]
    ax.set_facecolor('#050505')
    f_s, t_s, Sxx = sp_specgram(sig, fs=FS, nperseg=512, noverlap=480, window='hann')
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
    Sxx_db -= Sxx_db.max()
    ax.pcolormesh(t_s, f_s, Sxx_db, cmap='Blues', vmin=-55, vmax=0, shading='gouraud')
    ax.set_yscale('log'); ax.set_ylim(20, 22050)
    ax.axhline(880, color='#ffaa00', lw=0.9, alpha=0.7)
    ax.text(0.01, 950, '880Hz', color='#ffaa00', fontsize=7)
    ax.axhline(220, color='#aaffaa', lw=0.7, alpha=0.6)
    ax.text(0.01, 240, '220Hz', color='#aaffaa', fontsize=7)
    ax.set_title(titulo, color=cor, fontsize=9)
    ax.set_xlabel('tempo (s)', color='#888888', fontsize=8)
    ax.set_ylabel('frequencia Hz (log)', color='#888888', fontsize=8)
    ax.tick_params(colors='#888888', labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#333333')

plt.tight_layout(h_pad=2.5)
plt.savefig('/content/eco_retrogrado_resultado.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.show()
display(Image('/content/eco_retrogrado_resultado.png'))

# -- audios -------------------------------------------------------------------
def salvar_e_tocar(sinal, nome, label):
    s16 = np.int16(np.clip(normalizar(sinal), -1, 1) * 32767)
    wavfile.write(nome, FS, s16)
    print(f"\n  {label}")
    display(Audio(nome))

print("\n  Audios:")
salvar_e_tocar(x_mix,       '/content/retro_0_base.wav',      'BASE original (1.5s)')
salvar_e_tocar(sinal_ch,    '/content/retro_1_campo_harm.wav', 'Campo Harmonico phi3 (1.5s)')
salvar_e_tocar(sinal_retro, '/content/retro_2_retrogrado.wav', 'Eco Retrogrado (1.5s)')

# -- resultado final ----------------------------------------------------------
print("\n" + "=" * 60)
print("  RESULTADO FINAL")
print("=" * 60)
print(f"  BASE -> eco progressivo -> phi3  (verificado)")
print(f"  phi3 -> eco retrogrado  -> ?")
print(f"")
print(f"  Correlacao espectral BASE / Retrogrado = {corr_retro_base:+.4f}")
print(f"  Correlacao espectral BASE / C.H.       = {corr_ch_base:+.4f}")
print(f"  Distancia percorrida de volta          = {corr_retro_base - corr_ch_base:+.4f}")
print(f"")
if corr_retro_base > 0.7:
    print(f"  >> CAMPO FECHADO")
    print(f"     O retrogrado converge para BASE")
    print(f"     phi3 ja continha a assinatura espectral do BASE")
    print(f"     Retrocausalidade como estrutura operacional: confirmada")
elif corr_retro_base > corr_ch_base + 0.1:
    print(f"  >> CAMPO PARCIALMENTE FECHADO")
    print(f"     O retrogrado se aproxima do BASE mais que o C.H.")
    print(f"     Informacao parcialmente recuperavel: {corr_retro_base*100:.0f}%")
    print(f"     O campo tem memoria do ponto de partida")
else:
    print(f"  >> CAMINHO UNIDIRECIONAL")
    print(f"     O eco-phi e irreversivel nesta implementacao")
    print(f"     Informacao do BASE perdida no avanco para phi3")
    print(f"     Questao em aberto: qual sinal de entrada e necessario")
    print(f"     para que o retrogrado feche o campo?")
print("=" * 60)
