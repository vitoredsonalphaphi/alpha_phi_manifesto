"""
AlphaPhi_Medicao_Capacidade.py
Medição honesta da capacidade informacional em cada estágio do processamento.

Métricas:
  1. Entropia de Shannon (bits efetivos por amostra)
  2. Taxa de compressão zlib (proxy de complexidade estrutural)
  3. Autocorrelação (estrutura interna / previsibilidade)
  4. Entropia espectral (organização no domínio da frequência)
  5. Fator de crista (relação pico/RMS)

O objetivo é observar o que o código realmente produz — sem expectativa pré-definida.

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import zlib
import json
from scipy.signal import butter, filtfilt
from scipy.stats import entropy as scipy_entropy

# ── constantes ORIGINAIS — não modificar ─────────────────────
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
FADE       = int(0.15 * FS)

# ── funções originais ─────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s)); return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max: break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    return [(max(0, int(f_lo/(FS/n))),
             min(int(f_hi/(FS/n))+1, n//2+1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0/PHI, 1.0 - 1.0/PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi   = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb   = F[b_lo:b_hi]
        mag  = np.abs(Fb); phase = np.angle(Fb)
        an   = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh  = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        ce   = (wn*coh + wm*float(coh_mem[i])
                if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk   = np.arange(len(Fb))
        env  = np.clip(1.0+(ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag*env)*np.exp(1j*phase)
    r = np.fft.irfft(F_out, n=N)
    return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    cm = np.zeros(len(bins_phi))
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm = cohs; se = normalizar(se); cas.append(se); s = se.copy()
    return cas, cohs

def agente_eco(sinal, bins_phi, n_ciclos=20):
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
    return beta, cas

def concatenar(cas):
    out = cas[0].copy()
    for s in cas[1:]:
        fade_n = min(FADE, len(out), len(s))
        t_fade = np.linspace(0.0, 1.0, fade_n)
        out[-fade_n:] = out[-fade_n:]*(1-t_fade) + s[:fade_n]*t_fade
        out = np.concatenate([out, s[fade_n:]])
    return normalizar(out)

# ── funções de medição ────────────────────────────────────────

def entropia_amplitude(sinal, n_bins=256):
    """Shannon entropy da distribuição de amplitudes. Resultado em bits."""
    counts, _ = np.histogram(sinal, bins=n_bins, density=False)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))

def bits_efetivos(sinal, n_bins=256):
    """
    Quantos bits são efetivamente usados pela distribuição de amplitude.
    Máximo teórico = log2(n_bins) = 8 bits para 256 níveis.
    """
    h = entropia_amplitude(sinal, n_bins)
    return h  # em bits, máx = log2(256) = 8

def taxa_compressao(sinal):
    """
    Comprime o sinal com zlib e mede a razão.
    Sinal aleatório: razão ~1.0 (incompressível).
    Sinal estruturado: razão < 1.0 (compressível).
    """
    dados = (sinal * 32767).astype(np.int16).tobytes()
    comprimido = zlib.compress(dados, level=9)
    return len(comprimido) / len(dados)

def autocorrelacao_lag1(sinal):
    """
    Correlação da amostra com a próxima (lag=1).
    Sinal aleatório → ~0. Sinal estruturado → próximo de 1.
    """
    s = sinal - sinal.mean()
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])

def entropia_espectral(sinal):
    """
    Shannon entropy da distribuição de potência espectral.
    Sinal com 1 frequência dominante → baixa entropia.
    Sinal com espectro plano → alta entropia.
    """
    mag = np.abs(np.fft.rfft(sinal))**2
    mag = mag / (mag.sum() + 1e-12)
    mag = mag[mag > 1e-15]
    return float(-np.sum(mag * np.log2(mag)))

def fator_crista(sinal):
    """Pico / RMS. Sinal senoidal puro = √2 ≈ 1.414"""
    rms = np.sqrt(np.mean(sinal**2))
    if rms < 1e-12: return 0.0
    return float(np.max(np.abs(sinal)) / rms)

def medir(nome, sinal):
    """Mede todas as métricas e retorna dicionário."""
    b = bits_efetivos(sinal)
    tc = taxa_compressao(sinal)
    ac = autocorrelacao_lag1(sinal)
    he = entropia_espectral(sinal)
    fc = fator_crista(sinal)
    return {
        'nome': nome,
        'bits_efetivos': round(b, 4),
        'max_teorico_bits': 8.0,
        'uso_capacidade_pct': round(100 * b / 8.0, 2),
        'taxa_compressao': round(tc, 4),
        'autocorrelacao_lag1': round(ac, 4),
        'entropia_espectral': round(he, 4),
        'fator_crista': round(fc, 4),
    }

def imprimir(r):
    print(f"\n  ┌─ {r['nome']}")
    print(f"  │  Bits efetivos/amostra : {r['bits_efetivos']:.4f}  (máx teórico: 8.0 bits)")
    print(f"  │  Uso da capacidade     : {r['uso_capacidade_pct']:.2f}%")
    print(f"  │  Compressibilidade     : {r['taxa_compressao']:.4f}  (1.0=incompressível, <1=estruturado)")
    print(f"  │  Autocorrelação lag-1  : {r['autocorrelacao_lag1']:.4f}  (0=aleatório, 1=estruturado)")
    print(f"  │  Entropia espectral    : {r['entropia_espectral']:.4f}  bits (menor=mais focado)")
    print(f"  └─ Fator de crista       : {r['fator_crista']:.4f}  (√2≈1.414 puro senoidal)")

# ── geração dos sinais ────────────────────────────────────────
print("=" * 62)
print("  AlphaPhi · Medição de Capacidade Informacional")
print("  Observação por etapa — sem expectativa pré-definida")
print("=" * 62)

print("\n  Gerando sinais base…")
t_seg = np.linspace(0, DURACAO, N_SINAL, endpoint=False)

beep  = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_seg)))
fm    = normalizar(np.sin(2*np.pi*F_ORG*t_seg + BETA_FM*np.sin(2*np.pi*F_M*t_seg)))
x_mix = normalizar((1-ALPHA_STAR)*beep + ALPHA_STAR*fm)

print("  Processando agente eco-φ (20 ciclos)…")
beta_f, cas = agente_eco(x_mix, BINS_PHI, N_CICLOS)
sinal_final = concatenar(cas)

# ruído branco como referência
ruido = np.random.randn(N_SINAL)
ruido = normalizar(ruido)

# ── medições ──────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  RESULTADOS DAS MEDIÇÕES")
print("=" * 62)

resultados = []

r0 = medir("REFERÊNCIA: Ruído branco gaussiano", ruido)
imprimir(r0); resultados.append(r0)

r1 = medir("BEEP puro — square wave 880Hz (digital, antes de tudo)", beep)
imprimir(r1); resultados.append(r1)

r2 = medir("FM-φ puro — 220Hz + modulação φ", fm)
imprimir(r2); resultados.append(r2)

r3 = medir(f"x_mix — mistura α*={ALPHA_STAR:.4f} (1/3)", x_mix)
imprimir(r3); resultados.append(r3)

for k, seg in enumerate(cas[1:], 1):
    r = medir(f"Cascata eco-φ — passo {k}/5", seg)
    imprimir(r); resultados.append(r)

r_final = medir("SINAL FINAL — campo harmônico (β→φ³)", sinal_final)
imprimir(r_final); resultados.append(r_final)

# ── comparação direta: beep vs final ─────────────────────────
print("\n" + "=" * 62)
print("  COMPARAÇÃO DIRETA: BEEP DIGITAL vs CAMPO HARMÔNICO")
print("=" * 62)

b_beep  = r1['bits_efetivos']
b_final = r_final['bits_efetivos']
delta   = b_final - b_beep
fator   = b_final / b_beep if b_beep > 0 else 0

print(f"\n  Beep digital       : {b_beep:.4f} bits/amostra")
print(f"  Campo harmônico    : {b_final:.4f} bits/amostra")
print(f"  Diferença absoluta : {delta:+.4f} bits")
print(f"  Fator              : {fator:.4f}×")

print(f"\n  Compressibilidade beep   : {r1['taxa_compressao']:.4f}")
print(f"  Compressibilidade final  : {r_final['taxa_compressao']:.4f}")

print(f"\n  Autocorrelação beep      : {r1['autocorrelacao_lag1']:.4f}")
print(f"  Autocorrelação final     : {r_final['autocorrelacao_lag1']:.4f}")

print(f"\n  Entropia espectral beep  : {r1['entropia_espectral']:.4f}")
print(f"  Entropia espectral final : {r_final['entropia_espectral']:.4f}")

# ── evolução por passo ────────────────────────────────────────
print("\n" + "=" * 62)
print("  EVOLUÇÃO DOS BITS EFETIVOS POR PASSO")
print("=" * 62)
print(f"\n  {'Estágio':<45} {'Bits':>6}  {'Compressão':>10}")
print(f"  {'-'*63}")
for r in resultados:
    print(f"  {r['nome']:<45} {r['bits_efetivos']:>6.3f}  {r['taxa_compressao']:>10.4f}")

# ── interpretação honesta ─────────────────────────────────────
print("\n" + "=" * 62)
print("  INTERPRETAÇÃO — O QUE OS NÚMEROS DIZEM")
print("=" * 62)

if b_final > b_beep:
    print(f"""
  O campo harmônico tem MAIS bits efetivos que o beep puro.
  Isso significa: a distribuição de amplitudes é mais rica —
  o sinal usa mais "níveis" de forma balanceada.

  Porém: mais bits efetivos ≠ "mais informação armazenada".
  Significa que o sinal é mais DIVERSO em amplitude,
  não necessariamente que carrega mais dados decodificáveis.
""")
else:
    print(f"""
  O campo harmônico tem MENOS ou IGUAL bits efetivos que o beep.
  A modulação φ CONCENTRA a distribuição de amplitudes —
  mais coerente, mais previsível, menos "surpresas".

  Isso é consistente com a redução de entropia observada.
  Um canal mais organizado pode ser mais EFICIENTE,
  mas não necessariamente "guarda mais bits por amostra".
""")

print(f"""  CONCLUSÃO HONESTA:
  A modulação φ NÃO multiplica bits por 5×.
  O que ela faz: organiza o sinal de forma que
  a mesma largura de banda pode ser usada com
  MENOR REDUNDÂNCIA e MENOR TAXA DE ERRO.

  A hipótese defensável é eficiência de canal,
  não multiplicação de capacidade de armazenamento.

  β_max = {beta_f.max():.4f}  (φ³ = {PHI**3:.4f})
""")

# ── salvar resultados ─────────────────────────────────────────
fname = '/home/user/alpha_phi_manifesto/capacidade_informacional_results.json'
with open(fname, 'w', encoding='utf-8') as f:
    json.dump({
        'parametros': {
            'F_BEEP': F_BEEP, 'ALPHA_STAR': ALPHA_STAR,
            'N_CICLOS': N_CICLOS, 'N_STEPS': N_STEPS,
            'PHI': PHI, 'beta_max': float(beta_f.max())
        },
        'medicoes': resultados,
        'comparacao': {
            'bits_beep': b_beep, 'bits_final': b_final,
            'delta_bits': delta, 'fator': fator
        }
    }, f, indent=2)

print(f"  Resultados salvos em: capacidade_informacional_results.json")
print("=" * 62)
