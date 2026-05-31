#!/usr/bin/env python3
"""
ECO TEXT 002 — Varredura Espectral com Agente
Manifesto AlphaPhi · Vitor Edson Delavi · 2026

3 estágios:
  1. eco_ressonante como diapasão — identifica k_letra nos bits
  2. Calibração — afina bandas phi com a frequência encontrada
  3. Agente varre campo calibrado — busca alpha*_texto

Representação do dado: bits puros (letra → 8 bits).
Não é tabela de pesos. É o que a letra É no nível digital.

Protocolo de integridade AlphaPhi:
  - Seed por timestamp — nenhum valor escolhido pelo pesquisador
  - phi apenas na partição espectral — nunca como parâmetro de teste
  - Resultados reportados na íntegra
"""

import numpy as np
import re
import json
from datetime import datetime
import sys

PHI   = (1 + np.sqrt(5)) / 2
PHI2  = PHI ** 2
PHI3  = PHI ** 3
SQRT2 = np.sqrt(2)
SQRT5 = np.sqrt(5)
ALPHA = 1 / 137.035999084

N_ECO    = 3
N_STEPS  = 5
N_CICLOS = 20
LIMIAR   = 0.99 * PHI3


# ── Representação digital ─────────────────────────────────────────────────────

def text_to_bits(text):
    """Texto → sequência de bits. Representação digital pura, não tabela."""
    bits = []
    for c in text:
        byte = ord(c) & 0xFF
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return np.array(bits, dtype=float)

def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s


# ── Estágio 1: eco ressonante como diapasão ───────────────────────────────────

def eco_ressonante(x, k=PHI, n_eco=N_ECO):
    """
    Pré-função: emite ciclos de ressonância com fator k.
    Dado phi-coerente converge. Ruído diverge.
    k variável permite calibração (vs k=phi fixo original).
    """
    x     = np.asarray(x, dtype=float)
    sinal = x.copy()
    for _ in range(n_eco):
        freq           = np.fft.fft(sinal)
        amplitude      = np.abs(freq)
        fase           = np.angle(freq)
        nova_fase      = fase * k
        sinal_complexo = amplitude * np.exp(1j * nova_fase)
        reflexao       = np.real(np.fft.ifft(sinal_complexo))
        eco            = reflexao - x
        sinal          = sinal + (eco / k)
    return sinal

def medir_coerencia(sinal):
    energia = np.abs(np.fft.fft(sinal)) ** 2
    n       = len(energia)
    p       = np.clip(energia / (energia.sum() + 1e-8), 1e-10, 1.0)
    ent     = -np.sum(p * np.log(p))
    return float(1.0 - ent / np.log(n))

def identificar_frequencia(bits, n_pontos=30):
    """
    Varre k em [sqrt2, phi] e encontra k_otimo.
    O dado responde com sua frequência natural — não é imposta.
    Equivalente ao eco_fononico que encontrou k~sqrt2 nos dados SST-2.
    """
    ks        = np.linspace(SQRT2, PHI, n_pontos)
    coerencias = []
    for k in ks:
        sinal_conv = eco_ressonante(bits, k=k, n_eco=N_ECO)
        coerencias.append(medir_coerencia(sinal_conv))
    coerencias = np.array(coerencias)
    idx_otimo  = np.argmax(coerencias)
    return float(ks[idx_otimo]), float(coerencias[idx_otimo]), ks, coerencias


# ── Estágio 2: calibração das bandas ─────────────────────────────────────────

def gerar_bandas_calibradas(k_base, n_bandas=12):
    """
    Bandas phi-proporcionais partindo de k_base (frequência do dado).
    Equivalente de gerar_bandas_phi do ECO BEEP 880, mas afinado.
    """
    bandas = []
    k = k_base
    for _ in range(n_bandas):
        k_next = k * PHI
        bandas.append((k, k_next))
        k = k_next
    return bandas

def bandas_para_bins(bandas, n):
    """Mapeia bandas de k para bins do FFT (tamanho n)."""
    k_max = bandas[-1][1]
    bins  = []
    for k_lo, k_hi in bandas:
        b_lo = max(0, int(k_lo / k_max * (n // 2)))
        b_hi = min(int(k_hi / k_max * (n // 2)) + 1, n // 2 + 1)
        if b_hi > b_lo:
            bins.append((b_lo, b_hi, k_lo, k_hi))
    return bins if bins else [(0, n // 2 + 1, bandas[0][0], bandas[-1][1])]


# ── Estágio 3: agente + eco beep calibrado ────────────────────────────────────

def eco_eq_texto(x, bins_phi, beta_bands, coh_mem=None):
    """
    Eco equalizer — análogo ao eco_eq do ECO BEEP 880.
    Opera sobre bits de texto com bandas calibradas.
    """
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    N     = len(x)
    F     = np.fft.rfft(x)
    F_out = F.copy()
    cohs  = []
    wm, wn = 1.0 / PHI, 1.0 - 1.0 / PHI

    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb = F[b_lo:b_hi]
        if len(Fb) == 0:
            cohs.append(0.0)
            continue
        mag   = np.abs(Fb)
        phase = np.angle(Fb)
        an    = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        coh   = float(1.0 - (-np.sum(an * np.log(an))) / np.log(max(len(an), 2)))
        ce    = (wn * coh + wm * float(coh_mem[i])
                 if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk  = np.arange(len(Fb))
        env = np.clip(1.0 + (ce * PHI ** bi) * np.cos(2 * np.pi * nk / PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag * env) * np.exp(1j * phase)

    r = np.fft.irfft(F_out, n=N)
    m = np.max(np.abs(r))
    return r / (m + 1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi, coh_mem_init=None):
    s  = sinal.copy()
    cm = coh_mem_init.copy() if coh_mem_init is not None else np.zeros(len(bins_phi))
    passos = []
    for _ in range(N_STEPS):
        se, cohs = eco_eq_texto(s, bins_phi, beta_bands, cm)
        cm = cohs
        se = normalizar(se)
        passos.append(se.copy())
        s  = se.copy()
    return passos, cohs

def agente_varredura(sinal, bins_phi, verbose=True):
    """
    Agente adaptativo — mesmo mecanismo do ECO BEEP 880.
    Adapta beta baseado em coerência. Busca beta → phi3.
    Retorna alpha*_texto: a partição phi-proporcional do campo convergido.
    """
    nb   = len(bins_phi)
    beta = np.ones(nb)
    bm   = beta.copy()
    wm, wn    = 1.0 / PHI, 1.0 - 1.0 / PHI
    coh_final = np.zeros(nb)
    ciclo_conv = N_CICLOS
    passos_finais = [sinal.copy()]

    for ciclo in range(N_CICLOS):
        passos, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs - cohs.min()) / (cohs.max() - cohs.min() + 1e-10)
        ba   = PHI ** (3 * cr)
        beta = wn * ba + wm * bm
        bm   = beta.copy()
        beta = np.clip(beta, 0.05, PHI3)
        coh_final     = cohs
        passos_finais = passos

        if verbose:
            print(f"  ciclo {ciclo+1:02d}/{N_CICLOS}  "
                  f"beta_max={beta.max():.4f}  "
                  f"coh_media={cohs.mean():.4f}", flush=True)

        if beta.max() >= LIMIAR:
            ciclo_conv = ciclo + 1
            if verbose:
                print(f"  → Campo formado no ciclo {ciclo_conv}  "
                      f"beta_max={beta.max():.6f}")
            break

    # alpha*: partição phi-proporcional do espectro convergido
    sinal_conv = passos_finais[-1]
    energia    = np.abs(np.fft.rfft(sinal_conv)) ** 2
    n          = len(energia)
    split      = max(1, int(n / PHI))
    e1         = energia[:split].sum()
    e2         = energia[split:].sum()
    alpha_star = float(e1 / (e2 + 1e-10))

    return {
        'beta_final':    beta.tolist(),
        'beta_max':      float(beta.max()),
        'coh_final':     coh_final.tolist(),
        'coh_media':     float(coh_final.mean()),
        'ciclos':        ciclo_conv,
        'campo_formado': bool(beta.max() >= LIMIAR),
        'alpha_star':    alpha_star,
    }


# ── Relatório ─────────────────────────────────────────────────────────────────

def closest_invariant(val):
    candidates = [
        (abs(val - PHI3),  f'phi3  ({PHI3:.4f})'),
        (abs(val - SQRT5), f'sqrt5 ({SQRT5:.4f})'),
        (abs(val - PHI2),  f'phi2  ({PHI2:.4f})'),
        (abs(val - PHI),   f'phi   ({PHI:.4f})'),
        (abs(val - SQRT2), f'sqrt2 ({SQRT2:.4f})'),
        (abs(val - 1/3),   f'1/3   ({1/3:.4f})'),
        (abs(val - ALPHA), f'alpha ({ALPHA:.6f})'),
    ]
    d, name = min(candidates, key=lambda x: x[0])
    return name, d

def report(k_otimo, coh_diapasao, resultado_agente, filepath, seed):
    sep = '=' * 62
    print(f'\n{sep}')
    print('ECO TEXT 002 — Varredura Espectral com Agente')
    print(f'Manifesto AlphaPhi · {datetime.now().strftime("%d/%m/%Y %H:%M")}')
    print(f'Arquivo : {filepath}')
    print(f'Seed    : {seed}')
    print(sep)

    print('\n── ESTÁGIO 1 — Diapasão ──')
    inv_k, dist_k = closest_invariant(k_otimo)
    print(f'  k_otimo (frequência do dado) : {k_otimo:.6f}')
    print(f'  Invariante mais próximo      : {inv_k}  (desvio {dist_k:.4f})')
    print(f'  Coerência em k_otimo         : {coh_diapasao:.6f}')

    print('\n── ESTÁGIO 3 — Agente ──')
    ag = resultado_agente
    print(f'  beta_max convergido : {ag["beta_max"]:.6f}')
    print(f'  Limiar phi3         : {PHI3:.6f}')
    print(f'  Campo formado       : {"SIM" if ag["campo_formado"] else "NÃO"}')
    print(f'  Ciclos              : {ag["ciclos"]}/{N_CICLOS}')
    print(f'  Coerência média     : {ag["coh_media"]:.6f}')

    alpha_star   = ag['alpha_star']
    inv_a, dist_a = closest_invariant(alpha_star)
    print(f'\n  alpha*_texto = {alpha_star:.6f}')
    print(f'  Invariante mais próximo : {inv_a}  (desvio {dist_a:.4f})')
    print(f'  (cf. ECO BEEP 880: alpha*_audio = 1/3 = {1/3:.4f})')

    print('\nCONCLUSÃO:')
    if ag['campo_formado']:
        print(f'  Campo phi-coerente FORMADO no substrato de texto.')
        print(f'  alpha*_texto = {alpha_star:.4f}')
    else:
        print(f'  Campo não convergiu em {N_CICLOS} ciclos.')
        print(f'  alpha*_texto observado = {alpha_star:.4f}')
        print(f'  Resultado válido — reportar como encontrado.')
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print('Uso: python eco_text_002.py <arquivo.txt>')
        sys.exit(1)

    seed = int(datetime.now().timestamp())
    np.random.seed(seed)

    filepath = sys.argv[1]
    print(f'\nECO TEXT 002 — Varredura Espectral com Agente')
    print(f'Seed (timestamp): {seed}')

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    words = re.findall(r'\b[a-zA-ZÀ-ÿ]{2,}\b', text)
    sents = [s.strip() for s in re.split(r'(?<=[.!?;])\s+', text.strip())
             if len(s.strip()) > 5]
    paras = [p.strip() for p in re.split(r'\n\s*\n', text.strip())
             if len(p.strip()) > 10]

    print(f'Corpus: {len(text)} chars | {len(words)} palavras | '
          f'{len(sents)} frases | {len(paras)} parágrafos')

    # Estágio 1
    print('\n── Estágio 1: diapasão (eco ressonante → frequência do dado)...')
    bits = text_to_bits(text)
    print(f'  Sinal de bits: {len(bits)} dimensões')
    k_otimo, coh_d, ks_sw, cohs_sw = identificar_frequencia(bits)
    print(f'  k_otimo = {k_otimo:.6f}  coerência = {coh_d:.6f}')

    # Estágio 2
    print('\n── Estágio 2: calibrando bandas phi...')
    bandas = gerar_bandas_calibradas(k_otimo)
    bins   = bandas_para_bins(bandas, len(bits))
    print(f'  {len(bins)} bandas  (base k={k_otimo:.4f})')

    # Estágio 3
    print('\n── Estágio 3: agente varrendo campo calibrado...')
    sinal    = normalizar(bits)
    resultado = agente_varredura(sinal, bins, verbose=True)

    report(k_otimo, coh_d, resultado, filepath, seed)

    out = {
        'timestamp':  datetime.now().isoformat(),
        'seed':       seed,
        'file':       filepath,
        'chars':      len(text),
        'estagio_1':  {'k_otimo': k_otimo, 'coh_diapasao': coh_d,
                       'sweep_ks': ks_sw.tolist(), 'sweep_cohs': cohs_sw.tolist()},
        'estagio_2':  {'bandas': bandas, 'n_bins': len(bins)},
        'estagio_3':  resultado,
        'invariantes': {'phi3': PHI3, 'sqrt5': SQRT5, 'phi2': PHI2,
                        'phi': PHI, 'sqrt2': SQRT2, '1/3': 1/3},
    }
    out_path = f'eco_text_002_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nResultado salvo em: {out_path}')


if __name__ == '__main__':
    main()
