# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi — Fusão Diapasão <> ECO BEEP 880
Célula única. Google Colab ou local.

Hipótese central:
  O campo harmônico (AutoCorr→1.0000, EntrEsp→0.0601) que emergiu
  no ECO BEEP 880 em substrato sonoro emerge também em substrato
  de embeddings neurais — se o princípio é invariante de substrato.

Genealogia:
  Geração 0 — Diapasão (mar/2026): FFT como instrumento de medição.
              Substrato texto (SST-2). Resultado +0.09%. Pergunta certa,
              substrato errado.
  Geração 6 — ECO BEEP 880 (mai/2026): mesma pergunta, substrato sonoro.
              AutoCorr=1.0000, EntrEsp=0.0601. Campo harmônico emergiu.
  Esta fusão — retorno ao substrato do Diapasão com a arquitetura
              completa do ECO BEEP: mixing α=1/137 + eco cascata + agente.
              O campo harmônico fecha o círculo?

Estrutura:
  sinal_digital  → embedding normalizado (alta entropia no espaço de features)
  sinal_orgânico → referência FM-φ no espaço de embedding (D dimensões)
  mixing α=1/137 → x_mix = (1-α)·embedding + α·ref_phi
  eco cascata    → mesma lógica do ECO BEEP, aplicada sobre dims do embedding
  terceira struct → AutoCorr e EntrEsp abaixo de qualquer componente puro?

Substrato: embeddings reais (SST-2 via sentence-transformers) com
           fallback para embeddings sintéticos φ-estruturados.
"""

import numpy as np

# ── Constantes ────────────────────────────────────────────────────────────────
PHI     = (1 + np.sqrt(5)) / 2       # 1.6180339887
ALPHA_F = 1.0 / 137.035999084        # 0.00729735 — constante de estrutura fina
N_STEPS = 5                           # passos da cascata (quinto ponto de dobra)
N_CICLOS = 20                         # ciclos do agente eco
FS_REF  = 44100.0                     # frequência de referência do ECO BEEP (para escala)
F_ORG   = 220.0                       # portadora FM-φ original
F_M_ORG = F_ORG / PHI                 # 135.97 Hz — moduladora FM

# ── Geração de dados ──────────────────────────────────────────────────────────
def carregar_embeddings():
    """Carrega embeddings SST-2 reais. Fallback sintético se indisponível."""
    try:
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
        print("Carregando SST-2 e gerando embeddings reais...")
        ds  = load_dataset("glue", "sst2", split="validation[:200]")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(ds["sentence"], show_progress_bar=False,
                            convert_to_numpy=True)
        print(f"  Embeddings reais: {embs.shape}  (substrato SST-2)")
        return embs, "SST-2 real (MiniLM-L6)"
    except Exception as e:
        print(f"  sentence-transformers indisponível ({type(e).__name__}) — usando sintético.")
        return gerar_embeddings_sinteticos(), "sintético φ-estruturado"

def gerar_embeddings_sinteticos(N=200, D=384, seed=42):
    """
    Embeddings sintéticos como o sinal_digital do ECO BEEP:
    alta entropia, sem estrutura φ prévia.
    Mistura ruído gaussiano com componente aleatória — análogo à onda quadrada.
    """
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((N, D))
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    return embs

# ── Referência orgânica FM-φ no espaço de embeddings ─────────────────────────
def gerar_ref_phi(D):
    """
    Referência FM-φ no espaço de D dimensões.
    Traduz a lógica do FM-φ sonoro para o domínio de features:
      ref[i] = sin(2π · k_org · i/D + φ · sin(2π · k_m · i/D))
    onde k_org preserva a proporção F_ORG/FS do experimento sonoro.
    """
    k_org = max(2, round(D * F_ORG / FS_REF))  # proporção conservada
    k_m   = k_org / PHI
    i = np.arange(D, dtype=float)
    ref = np.sin(2 * np.pi * k_org * i / D + PHI * np.sin(2 * np.pi * k_m * i / D))
    return ref / (np.max(np.abs(ref)) + 1e-10)

# ── Bandas φ no espaço de bins de embedding ───────────────────────────────────
def gerar_bandas_bins(D):
    """
    Bandas φ-proporcionais sobre os bins FFT do embedding.
    Traduz gerar_bandas_phi() do ECO BEEP (Hz → bins sobre D dims).
    """
    n_bins = D // 2 + 1
    bandas, b = [], 1.0
    while b < n_bins:
        b_next = min(b * PHI, float(n_bins))
        b_low  = int(np.floor(b))
        b_high = int(np.ceil(b_next))
        if b_high - b_low >= 2:
            bandas.append((b_low, b_high))
        if b_next >= n_bins:
            break
        b = b_next
    return bandas

# ── ECO EQ no espaço de embeddings ────────────────────────────────────────────
def eco_eq_emb(x_batch, bandas, beta_bands, coh_mem=None):
    """
    Eco EQ aplicado sobre dimensões do embedding (batch, D).
    Mesma lógica do eco_eq sonoro — FFT sobre D, φ-bandas, amplificação.
    """
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    N, D = x_batch.shape
    F = np.fft.rfft(x_batch, axis=-1)          # (N, D//2+1)
    F_out = F.copy()
    w_mem = 1.0 / PHI
    w_now = 1.0 - w_mem
    cohs = []
    for i, (b_low, b_high) in enumerate(bandas):
        beta_i  = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        F_band  = F[:, b_low:b_high]            # (N, banda)
        mag     = np.abs(F_band)
        phase   = np.angle(F_band)
        # coerência por banda — média sobre batch
        mag_sum = mag.sum(axis=-1, keepdims=True) + 1e-8
        an      = np.clip(mag / mag_sum, 1e-10, 1.0)
        H       = -np.sum(an * np.log(an), axis=-1)         # (N,)
        n_b     = max(b_high - b_low, 2)
        coh_b   = float(np.mean(1.0 - H / np.log(n_b)))
        if coh_mem is not None and i < len(coh_mem):
            coh_b = w_now * coh_b + w_mem * float(coh_mem[i])
        cohs.append(coh_b)
        n_idx = np.arange(b_high - b_low, dtype=float)
        env   = np.clip(
            1.0 + (coh_b * PHI**beta_i) * np.cos(2.0 * np.pi * n_idx / PHI),
            0.05, None
        )
        F_out[:, b_low:b_high] = (mag * env) * np.exp(1j * phase)
    resultado = np.fft.irfft(F_out, n=D, axis=-1)           # (N, D)
    norms = np.max(np.abs(resultado), axis=-1, keepdims=True) + 1e-10
    return resultado / norms, np.array(cohs, dtype=float)

def cascata_eq_emb(x_batch, beta_bands, bandas):
    cas      = [x_batch.copy()]
    s        = x_batch.copy()
    n_bandas = len(bandas)
    coh_mem  = np.zeros(n_bandas, dtype=float)
    for _ in range(N_STEPS):
        s_e, cohs = eco_eq_emb(s, bandas, beta_bands, coh_mem)
        coh_mem   = cohs
        cas.append(s_e)
        s = s_e.copy()
    return cas, cohs

def agente_eco_emb(x_batch, bandas):
    n_bandas = len(bandas)
    beta     = np.ones(n_bandas, dtype=float)
    beta_mem = beta.copy()
    w_mem    = 1.0 / PHI
    w_now    = 1.0 - w_mem
    cas_final = None
    for _ in range(N_CICLOS):
        cas, cohs  = cascata_eq_emb(x_batch, beta, bandas)
        coh_rel    = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        beta_alvo  = PHI ** (3 * coh_rel)
        beta       = w_now * beta_alvo + w_mem * beta_mem
        beta_mem   = beta.copy()
        beta       = np.clip(beta, 0.05, PHI**3)
        cas_final  = cas
    return beta, cas_final

# ── Métricas — mesmas do ECO BEEP 880 ────────────────────────────────────────
def autocorr(x_batch):
    """Lag-1 autocorrelação sobre dimensões do embedding — média do batch."""
    vals = []
    for s in x_batch:
        s = s - s.mean()
        std = s.std()
        if std < 1e-10:
            continue
        c = float(np.corrcoef(s[:-1], s[1:])[0, 1])
        if not np.isnan(c):
            vals.append(c)
    return float(np.mean(vals)) if vals else 0.0

def entr_esp(x_batch):
    """Entropia espectral sobre dims do embedding — média do batch."""
    vals = []
    for s in x_batch:
        mag = np.abs(np.fft.rfft(s))**2
        tot = mag.sum() + 1e-12
        p   = mag / tot
        p   = p[p > 1e-15]
        vals.append(float(-np.sum(p * np.log2(p))))
    return float(np.mean(vals)) if vals else 0.0

# ── Varredura α — coração do experimento ─────────────────────────────────────
def varredura_alpha(embs_norm, ref_phi, bandas, n_alpha=25):
    """
    Varre α de 0 a 1 e encontra α* de máxima organização.
    Mesma lógica da varredura_hibrida do ECO BEEP 880.
    """
    alphas_base = [float(i) / float(n_alpha - 1) for i in range(n_alpha)]
    alphas_esp  = [ALPHA_F, 1.0/PHI**2, 1.0/PHI, 0.5]
    alphas      = sorted(set(round(a, 8) for a in alphas_base + alphas_esp))

    resultados = []
    for alpha in alphas:
        x_mix = (1.0 - alpha) * embs_norm + alpha * ref_phi  # broadcasting: ref_phi (D,) → (N,D)
        norms = np.max(np.abs(x_mix), axis=-1, keepdims=True) + 1e-10
        x_mix = x_mix / norms

        ent_ini = entr_esp(x_mix)
        ac_ini  = autocorr(x_mix)

        _, cas = agente_eco_emb(x_mix, bandas)
        sig_eco = cas[-1]

        ent_fin = entr_esp(sig_eco)
        ac_fin  = autocorr(sig_eco)
        delta   = ent_ini - ent_fin

        nota = ""
        if abs(alpha - ALPHA_F) < 1e-5:    nota = " ← α=1/137  ★"
        elif abs(alpha - 1.0/PHI**2) < 1e-5: nota = " ← α=1/φ²"
        elif abs(alpha - 1.0/PHI) < 1e-5:  nota = " ← α=1/φ"
        elif abs(alpha - 0.5) < 1e-5:      nota = " ← α=1/2"

        resultados.append({
            "alpha": alpha, "ent_ini": ent_ini, "ent_fin": ent_fin,
            "delta": delta, "ac_ini": ac_ini, "ac_fin": ac_fin,
            "nota": nota, "eco": sig_eco,
        })
    return resultados

# ── Main ──────────────────────────────────────────────────────────────────────
print("="*62)
print("  AlphaPhi — Fusão Diapasão <> ECO BEEP 880")
print(f"  φ = {PHI:.10f}  |  α = {ALPHA_F:.10f}")
print(f"  N_STEPS={N_STEPS}  |  N_CICLOS={N_CICLOS}")
print("="*62)

embs_raw, fonte = carregar_embeddings()
N, D = embs_raw.shape
print(f"\n  Fonte       : {fonte}")
print(f"  Shape       : {N} amostras × {D} dimensões")

# Normaliza embedding → sinal_digital (análogo ao beep quadrado)
norms = np.max(np.abs(embs_raw), axis=-1, keepdims=True) + 1e-10
embs_norm = embs_raw / norms

# Gera referência orgânica FM-φ no espaço de D dimensões
ref_phi = gerar_ref_phi(D)
print(f"  k_org (FM-φ): {max(2, round(D * F_ORG / FS_REF))}  "
      f"(preserva proporção F_ORG/FS = {F_ORG/FS_REF:.5f})")

bandas = gerar_bandas_bins(D)
print(f"  Bandas φ    : {len(bandas)}")

# Métricas dos componentes puros (linha de base)
ent_digital = entr_esp(embs_norm)
ac_digital  = autocorr(embs_norm)
ent_organico = entr_esp(ref_phi[np.newaxis, :])
ac_organico  = autocorr(ref_phi[np.newaxis, :])

print(f"\n{'─'*62}")
print(f"  Componentes puros:")
print(f"  {'sinal_digital (embedding)':30}  AutoCorr={ac_digital:+.4f}  "
      f"EntrEsp={ent_digital:.4f}")
print(f"  {'sinal_orgânico (FM-φ ref)':30}  AutoCorr={ac_organico:+.4f}  "
      f"EntrEsp={ent_organico:.4f}")
print(f"{'─'*62}")

# Varredura α
print(f"\n  Varredura α — {N} embeddings × {D} dims × {N_CICLOS} ciclos...")
print(f"\n  {'α':>10}  {'ent_ini':>8}  {'ent_fin':>8}  {'Δent':>7}  "
      f"{'AC_ini':>7}  {'AC_fin':>7}  nota")
print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*15}")

resultados = varredura_alpha(embs_norm, ref_phi, bandas)

for r in resultados:
    print(f"  {r['alpha']:>10.6f}  {r['ent_ini']:>8.4f}  {r['ent_fin']:>8.4f}  "
          f"{r['delta']:>+7.4f}  {r['ac_ini']:>+7.4f}  {r['ac_fin']:>+7.4f}  {r['nota']}")

# Ponto de emergência
deltas   = [r["delta"] for r in resultados]
acs_fin  = [r["ac_fin"] for r in resultados]
i_max_d  = int(np.argmax(deltas))
i_max_ac = int(np.argmax(acs_fin))
r_em     = resultados[i_max_d]

print(f"\n{'═'*62}")
print(f"  PONTO DE EMERGÊNCIA")
print(f"{'═'*62}")
print(f"  α* (max Δentropia) : {r_em['alpha']:.8f}  (Δ={r_em['delta']:+.4f})")
print(f"  α* (max AutoCorr)  : {resultados[i_max_ac]['alpha']:.8f}  "
      f"(AC={resultados[i_max_ac]['ac_fin']:+.4f})")
print(f"\n  {'':30}  AutoCorr    EntrEsp")
print(f"  {'─'*30}  {'─'*9}   {'─'*9}")
print(f"  {'digital puro':30}  {ac_digital:>+9.4f}  {ent_digital:>9.4f}")
print(f"  {'orgânico puro (FM-φ)':30}  {ac_organico:>+9.4f}  {ent_organico:>9.4f}")
print(f"  {'híbrido α* (eco)':30}  {r_em['ac_fin']:>+9.4f}  {r_em['ent_fin']:>9.4f}")
print(f"  {'Alvo ECO BEEP 880':30}  {'1.0000':>9}  {'0.0601':>9}")

# Teste da terceira estrutura
ent_min_puro = min(ent_digital, ent_organico)
if r_em["ent_fin"] < ent_min_puro:
    print(f"\n  ★ Terceira estrutura emergiu.")
    print(f"    EntrEsp híbrido ({r_em['ent_fin']:.4f}) < qualquer componente puro "
          f"(mín={ent_min_puro:.4f}).")
    print(f"    O campo harmônico atravessou o substrato.")
elif r_em["delta"] > 0:
    print(f"\n  → Eco organizou o híbrido (Δ positivo).")
    print(f"    Terceira estrutura não emergiu — mas organização ocorreu.")
else:
    print(f"\n  → Eco não organizou. Substrato resiste.")
    print(f"    Hipótese: estrutura φ já presente ou dimensão insuficiente.")

# Convergência β
_, cas = agente_eco_emb(
    (1 - ALPHA_F) * embs_norm + ALPHA_F * ref_phi,
    bandas
)
print(f"\n{'─'*62}")
print(f"  Cascata em α=1/137 — quinto ponto de dobra:")
print(f"  {'Passo':>6}  {'AutoCorr':>9}  {'EntrEsp':>9}")
for step_i, s_step in enumerate(cas):
    print(f"  {step_i:>6}  {autocorr(s_step):>+9.4f}  {entr_esp(s_step):>9.4f}")
print(f"  {'Alvo':>6}  {'1.0000':>9}  {'0.0601':>9}")
print(f"{'═'*62}")
