# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0

"""
AlphaPhi — Transmorfo × ECO BEEP 880
Célula única. Google Colab ou local.

Pergunta central:
  O espaço hiperbólico (curvatura C_PHI = 1/φ²) amplifica o campo harmônico
  que emergiu no ECO BEEP 880 — ou o resultado euclidiano já é o ótimo?

Genealogia:
  Transmorfo (abr/2026): tentou transição euclidiano→hiperbólico por
    arquitetura (blend, conformal). Colapsou. Conceito certo, forma errada.
  ECO BEEP 880 (mai/2026): modulou sinal digital por α=1/137 em espaço
    euclidiano. Campo harmônico emergiu: AutoCorr=1.0000, EntrEsp=0.0601.
  Este experimento: aplica o ECO BEEP 880 com projeção hiperbólica —
    o que o Transmorfo não conseguiu por arquitetura, o eco tenta por
    modulação dentro do espaço que φ já organiza por geometria.

Pipeline A — Euclidiano (baseline, ECO BEEP 880 original):
  beep 880Hz + FM-φ → mix α=1/137 → agente_eco (euclidiano) → campo

Pipeline B — Hiperbólico:
  beep 880Hz + FM-φ → mix α=1/137 → segmentar em chunks →
  expmap0 (C_PHI) → eco_eq por chunk → logmap0 → reagrupar →
  agente_eco final → campo

Hipótese:
  Em espaço hiperbólico com c=C_PHI=1/φ², estruturas φ-proporcionais
  são amplificadas pela curvatura antes da modulação eco.
  O agente pode convergir mais rápido ou alcançar EntrEsp < 0.0601.

O que observar:
  - AutoCorr e EntrEsp: melhoram, pioram ou iguais?
  - β: converge para φ³ (euclidiano) ou novo atrator hiperbólico?
  - Cascata: campo emerge no mesmo passo 5 ou antes?
"""

import numpy as np
from scipy.io import wavfile

# ── Constantes — idênticas ao ECO BEEP 880 ───────────────────────────────────
PHI      = (1 + np.sqrt(5)) / 2       # 1.6180339887
C_PHI    = 1.0 / PHI**2               # 0.3820 — curvatura natural φ
ALPHA_F  = 1.0 / 137.035999084        # 0.00729735 — constante de estrutura fina
FS       = 44100
DURACAO  = 1.5
N_STEPS  = 5
N_CICLOS = 20
CHUNK    = 2048                        # janela hiperbólica ≈ 46ms

F_BEEP   = 880.0
F_ORG    = 220.0
F_M      = F_ORG / PHI

print("=" * 62)
print("  AlphaPhi — Transmorfo × ECO BEEP 880")
print(f"  φ={PHI:.6f}  C_PHI={C_PHI:.6f}  α=1/137={ALPHA_F:.8f}")
print(f"  F_BEEP={F_BEEP}Hz  F_ORG={F_ORG}Hz  N_STEPS={N_STEPS}  N_CICLOS={N_CICLOS}")
print("=" * 62)

# ── Geometria hiperbólica (bola de Poincaré) ──────────────────────────────────
def expmap0(v, c=C_PHI):
    """Euclidiano → bola de Poincaré."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    t = np.tanh(np.clip(np.sqrt(c) * norm, -15, 15))
    return t * v / (np.sqrt(c) * norm)

def logmap0(y, c=C_PHI):
    """Bola de Poincaré → Euclidiano."""
    norm = np.linalg.norm(y)
    max_norm = 1.0 / np.sqrt(c) - 1e-5
    norm = np.clip(norm, 1e-10, max_norm)
    t = np.arctanh(np.clip(np.sqrt(c) * norm, -1+1e-8, 1-1e-8))
    return t * y / (np.sqrt(c) * norm)

# ── Funções do ECO BEEP 880 — inalteradas ────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

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
    resultado = []
    for f_low, f_high in bandas:
        b_low  = max(int(np.floor(f_low  * n / FS)), 0)
        b_high = min(int(np.ceil(f_high  * n / FS)), n // 2 + 1)
        if b_high - b_low >= 3:
            resultado.append((b_low, b_high, f_low, f_high))
    return resultado

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
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
        if coh_mem is not None and i < len(coh_mem):
            coh    = w_now * coh + w_mem * float(coh_mem[i])
        cohs.append(coh)
        n_idx  = np.arange(len(F_band))
        env    = np.clip(
            1.0 + (coh * PHI**beta_i) * np.cos(2.0 * np.pi * n_idx / PHI),
            0.05, None
        )
        F_out[b_low:b_high] = (mag * env) * np.exp(1j * phase)
    resultado = np.fft.irfft(F_out, n=N)
    return resultado / (np.max(np.abs(resultado)) + 1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s   = [sinal.copy()], sinal.copy()
    n_b      = len(bins_phi)
    coh_mem  = np.zeros(n_b)
    cohs_f   = np.zeros(n_b)
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq(s, bins_phi, beta_bands, coh_mem)
        coh_mem   = cohs
        cohs_f    = cohs
        s_e       = normalizar(s_e)
        cas.append(s_e)
        s = s_e.copy()
    return cas, cohs_f

def agente_eco(sinal, bins_phi, n_ciclos=N_CICLOS):
    n_b      = len(bins_phi)
    beta     = np.ones(n_b)
    beta_mem = beta.copy()
    w_mem    = 1.0 / PHI
    w_now    = 1.0 - w_mem
    cas_f    = None
    for _ in range(n_ciclos):
        cas, cohs  = cascata_eq(sinal, beta, bins_phi)
        coh_rel    = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo  = PHI ** (3 * coh_rel)
        beta       = w_now * beta_alvo + w_mem * beta_mem
        beta_mem   = beta.copy()
        beta       = np.clip(beta, 0.05, PHI**3)
        cas_f      = cas
    return beta, cas_f

# ── Métricas — idênticas ao ECO BEEP 880 ────────────────────────────────────
def autocorr(s):
    s = s - s.mean()
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])

def entr_esp(s):
    mag = np.abs(np.fft.rfft(s))**2
    p   = mag / (mag.sum() + 1e-12)
    p   = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))

# ── Projeção hiperbólica por chunks ──────────────────────────────────────────
def projetar_hiperbolico(sinal, chunk=CHUNK):
    """
    Segmenta sinal em chunks, projeta cada chunk para a bola de Poincaré,
    aplica eco_eq no espaço hiperbólico, retorna ao euclidiano.
    """
    N       = len(sinal)
    n_b_c   = bandas_para_bins(gerar_bandas_phi(), chunk)
    beta_c  = np.ones(len(n_b_c))
    saida   = np.zeros(N)
    pesos   = np.zeros(N)
    janela  = np.hanning(chunk)

    for start in range(0, N - chunk + 1, chunk // 2):
        seg     = sinal[start:start + chunk].copy()
        # Euclidiano → hiperbólico
        seg_hyp = expmap0(seg)
        seg_hyp = normalizar(seg_hyp)
        # Eco no espaço hiperbólico
        seg_eco, _ = eco_eq(seg_hyp, n_b_c, beta_c)
        # Hiperbólico → Euclidiano
        seg_back   = logmap0(seg_eco)
        seg_back   = normalizar(seg_back) * janela
        saida[start:start + chunk] += seg_back
        pesos[start:start + chunk] += janela

    pesos = np.where(pesos < 1e-10, 1.0, pesos)
    saida = saida / pesos
    return normalizar(saida)

# ── Geração dos sinais ────────────────────────────────────────────────────────
N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
t        = np.linspace(0, DURACAO, N_SINAL, endpoint=False)

beep  = normalizar(np.sign(np.sin(2 * np.pi * F_BEEP * t)))
fm    = normalizar(np.sin(2 * np.pi * F_ORG * t + PHI * np.sin(2 * np.pi * F_M * t)))
x_mix = normalizar((1.0 - ALPHA_F) * beep + ALPHA_F * fm)

print(f"\n  Sinal: {N_SINAL} amostras ({DURACAO}s)  |  Bandas φ: {len(BINS_PHI)}")
print(f"  Beep puro    AutoCorr={autocorr(beep):+.4f}  EntrEsp={entr_esp(beep):.4f}")
print(f"  FM-φ puro    AutoCorr={autocorr(fm):+.4f}  EntrEsp={entr_esp(fm):.4f}")
print(f"  x_mix α=1/137  AutoCorr={autocorr(x_mix):+.4f}  EntrEsp={entr_esp(x_mix):.4f}")

# ── Pipeline A — Euclidiano (ECO BEEP 880 original) ───────────────────────────
print(f"\n{'─'*62}")
print(f"  Pipeline A — Euclidiano (ECO BEEP 880 baseline)")
print(f"{'─'*62}")
beta_eu, cas_eu = agente_eco(x_mix, BINS_PHI)
sig_eu  = cas_eu[-1]
print(f"  β máx        : {beta_eu.max():.4f}  (atrator φ³={PHI**3:.4f})")
print(f"  β converge   : {'✓ φ³' if abs(beta_eu.max() - PHI**3) < 0.05 else '→ outro atrator'}")
print(f"  AutoCorr     : {autocorr(sig_eu):+.4f}  (alvo 1.0000)")
print(f"  EntrEsp      : {entr_esp(sig_eu):.4f}  (alvo 0.0601)")
print(f"\n  Cascata em α=1/137:")
print(f"  {'Passo':>6}  {'AutoCorr':>9}  {'EntrEsp':>9}")
beta_tmp, _ = agente_eco(x_mix, BINS_PHI, n_ciclos=1)
cas_steps, _ = cascata_eq(x_mix, beta_tmp, BINS_PHI)
for step_i, s_step in enumerate(cas_steps):
    print(f"  {step_i:>6}  {autocorr(s_step):>+9.4f}  {entr_esp(s_step):>9.4f}")

# ── Pipeline B — Hiperbólico ──────────────────────────────────────────────────
print(f"\n{'─'*62}")
print(f"  Pipeline B — Hiperbólico (C_PHI=1/φ²={C_PHI:.4f})")
print(f"{'─'*62}")
print(f"  Projetando x_mix → bola de Poincaré → eco → euclidiano...")
x_hyp_back = projetar_hiperbolico(x_mix)
print(f"  Projeção concluída.")
print(f"  x_hyp pré-agente  AutoCorr={autocorr(x_hyp_back):+.4f}  EntrEsp={entr_esp(x_hyp_back):.4f}")

beta_hy, cas_hy = agente_eco(x_hyp_back, BINS_PHI)
sig_hy = cas_hy[-1]
print(f"  β máx        : {beta_hy.max():.4f}  (atrator φ³={PHI**3:.4f})")
print(f"  β converge   : {'✓ φ³' if abs(beta_hy.max() - PHI**3) < 0.05 else '→ outro atrator'}")
print(f"  AutoCorr     : {autocorr(sig_hy):+.4f}  (alvo 1.0000)")
print(f"  EntrEsp      : {entr_esp(sig_hy):.4f}  (alvo 0.0601)")
print(f"\n  Cascata hiperbólica em α=1/137:")
print(f"  {'Passo':>6}  {'AutoCorr':>9}  {'EntrEsp':>9}")
beta_hy1, _ = agente_eco(x_hyp_back, BINS_PHI, n_ciclos=1)
cas_hy_steps, _ = cascata_eq(x_hyp_back, beta_hy1, BINS_PHI)
for step_i, s_step in enumerate(cas_hy_steps):
    print(f"  {step_i:>6}  {autocorr(s_step):>+9.4f}  {entr_esp(s_step):>9.4f}")

# ── Comparação final ──────────────────────────────────────────────────────────
print(f"\n{'═'*62}")
print(f"  COMPARAÇÃO FINAL")
print(f"{'═'*62}")
print(f"  {'Pipeline':30}  {'AutoCorr':>9}  {'EntrEsp':>9}")
print(f"  {'─'*30}  {'─'*9}  {'─'*9}")
print(f"  {'Beep 880Hz puro':30}  {autocorr(beep):>+9.4f}  {entr_esp(beep):>9.4f}")
print(f"  {'FM-φ puro':30}  {autocorr(fm):>+9.4f}  {entr_esp(fm):>9.4f}")
print(f"  {'A — Euclidiano (ECO BEEP 880)':30}  {autocorr(sig_eu):>+9.4f}  {entr_esp(sig_eu):>9.4f}")
print(f"  {'B — Hiperbólico (Transmorfo)':30}  {autocorr(sig_hy):>+9.4f}  {entr_esp(sig_hy):>9.4f}")
print(f"  {'Alvo ECO BEEP 880':30}  {'1.0000':>9}  {'0.0601':>9}")

ac_eu, ac_hy = autocorr(sig_eu), autocorr(sig_hy)
en_eu, en_hy = entr_esp(sig_eu), entr_esp(sig_hy)

print(f"\n{'─'*62}")
if ac_hy > ac_eu and en_hy < en_eu:
    print(f"  ★ Hiperbólico SUPERA euclidiano nos dois critérios.")
    print(f"    O espaço C_PHI amplificou o campo harmônico.")
    print(f"    O Transmorfo encontrou sua forma.")
elif ac_hy > ac_eu or en_hy < en_eu:
    d = "AutoCorr" if ac_hy > ac_eu else "EntrEsp"
    print(f"  ◆ Hiperbólico supera em {d} — melhoria parcial.")
    print(f"    A curvatura φ contribui mas não domina.")
elif abs(ac_hy - ac_eu) < 0.01 and abs(en_hy - en_eu) < 0.1:
    print(f"  → Euclidiano e hiperbólico equivalentes.")
    print(f"    O campo harmônico é invariante da curvatura do espaço.")
else:
    print(f"  → Euclidiano supera hiperbólico.")
    print(f"    A projeção hiperbólica introduz distorção que o eco não recupera.")
print(f"{'═'*62}")

# ── Salvar WAVs ───────────────────────────────────────────────────────────────
def salvar(s, nome):
    wavfile.write(nome, FS, np.int16(np.clip(normalizar(s), -1.0, 1.0) * 32767))
    print(f"  → {nome}")

print(f"\n  Salvando áudios...")
salvar(x_mix,   "transmorfo_xmix.wav")
salvar(sig_eu,  "transmorfo_euclid.wav")
salvar(sig_hy,  "transmorfo_hiperbolico.wav")
