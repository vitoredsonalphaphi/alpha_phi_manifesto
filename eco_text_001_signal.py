#!/usr/bin/env python3
"""
ECO TEXT 001 — Medicao Espectral Multi-Escala em Texto (versao sinal)
Manifesto AlphaPhi · Vitor Edson Delavi · 2026

Sem dependencias de rede. Trata o texto como sinal — equivalente direto
ao ECO BEEP 880: a sequencia de comprimentos em cada escala e o sinal.
FFT mede estrutura espectral. beta mede particao phi-proporcional.

Escalas:
  caractere -> palavra -> frase -> paragrafo

Uso: python eco_text_001_signal.py <arquivo.txt>
"""

import numpy as np
import re
import sys
import json
from datetime import datetime

PHI   = (1 + np.sqrt(5)) / 2
PHI2  = PHI ** 2
PHI3  = PHI ** 3
SQRT5 = np.sqrt(5)
ALPHA = 1 / 137.035999084


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def split_words(text):
    return re.findall(r'\b[a-zA-ZÀ-ÿ]{2,}\b', text)

def split_sentences(text):
    parts = re.split(r'(?<=[.!?;])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 5]

def split_paragraphs(text):
    parts = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def signal_chars(text):
    """Sinal: valor ASCII de cada caractere — o texto inteiro como onda."""
    return np.array([ord(c) for c in text if c.isprintable()], dtype=float)

def signal_word_lengths(words):
    """Sinal: comprimento de cada palavra — ritmo lexical."""
    return np.array([len(w) for w in words], dtype=float)

def signal_sentence_lengths(sentences):
    """Sinal: numero de palavras por frase — ritmo sintático."""
    return np.array([len(re.findall(r'\b\w+\b', s)) for s in sentences], dtype=float)

def signal_para_lengths(paras):
    """Sinal: numero de frases por paragrafo — ritmo estrutural."""
    def count_sents(p):
        return len(re.split(r'(?<=[.!?;])\s+', p.strip()))
    return np.array([count_sents(p) for p in paras], dtype=float)


def beta_phi(signal):
    """
    Razao espectral beta em particao phi-proporcional.
    Mesmo principio do ECO BEEP 880 — particao do espectro em 1/phi e phi/phi.
    """
    if len(signal) < 4:
        return 0.0
    energy = np.abs(np.fft.fft(signal - signal.mean()))
    n      = len(energy)
    split  = max(1, int(n / PHI))
    e1     = energy[:split].sum()
    e2     = energy[split:].sum()
    return float(e1 / (e2 + 1e-10))

def spectral_entropy(signal):
    """Entropia espectral normalizada — mede concentracao de energia."""
    if len(signal) < 4:
        return 0.0, 0.0
    energy = np.abs(np.fft.fft(signal - signal.mean()))
    n      = len(energy)
    enorm  = np.clip(energy / (energy.sum() + 1e-8), 1e-10, 1.0)
    ent    = -np.sum(enorm * np.log(enorm))
    coherence = 1.0 - ent / np.log(n)
    # variancia relativa como desvio
    std_rel = float(np.std(energy) / (np.mean(energy) + 1e-8))
    return float(coherence), std_rel

def measure_scale(name, signal):
    b       = beta_phi(signal)
    c, s    = spectral_entropy(signal)
    return {
        'scale':          name,
        'n':              len(signal),
        'mean':           float(np.mean(signal)),
        'std':            float(np.std(signal)),
        'beta':           b,
        'coherence':      c,
        'coherence_std':  s,
    }


def closest_invariant(val):
    candidates = [
        (abs(val - PHI3), f'phi3 ({PHI3:.4f})'),
        (abs(val - SQRT5), f'sqrt5 ({SQRT5:.4f})'),
        (abs(val - PHI2),  f'phi2 ({PHI2:.4f})'),
        (abs(val - PHI),   f'phi ({PHI:.4f})'),
    ]
    d, name = min(candidates, key=lambda x: x[0])
    return name, d


def report(scales, ratios, filepath, seed):
    sep = '=' * 62
    print(f'\n{sep}')
    print('ECO TEXT 001 — Relatorio Espectral Multi-Escala (sinal direto)')
    print(f'Manifesto AlphaPhi · {datetime.now().strftime("%d/%m/%Y %H:%M")}')
    print(f'Arquivo : {filepath}')
    print(f'Seed    : {seed}')
    print(sep)

    print('\nINVARIANTES DE REFERENCIA (ECO BEEP 880):')
    print(f'  phi3  = {PHI3:.6f}  <- atrator audio (880Hz)')
    print(f'  sqrt5 = {SQRT5:.6f}  <- invariante serial')
    print(f'  phi2  = {PHI2:.6f}')
    print(f'  phi   = {PHI:.6f}')

    print(f'\n{"Escala":<14}{"N":<6}{"media":<10}{"beta":<14}{"coerencia":<14}{"desvio"}')
    print('-' * 62)
    for s in scales:
        print(f'{s["scale"]:<14}{s["n"]:<6}{s["mean"]:<10.2f}'
              f'{s["beta"]:<14.6f}{s["coherence"]:<14.6f}+/-{s["coherence_std"]:.4f}')

    print('\nRAZOES ENTRE ESCALAS (atrator buscado):')
    print('-' * 62)
    ratio_vals = []
    for r in ratios:
        inv, dist = closest_invariant(r['ratio'])
        tag = f'  <- {inv}  (desvio {dist:.4f})'
        print(f'  {r["from"]:<12} -> {r["to"]:<12}  {r["ratio"]:.6f}{tag}')
        ratio_vals.append(r['ratio'])

    if ratio_vals:
        mean_r = float(np.mean(ratio_vals))
        std_r  = float(np.std(ratio_vals))
        inv, dist = closest_invariant(mean_r)
        print(f'\n  Media das razoes : {mean_r:.6f} +/- {std_r:.6f}')
        print(f'  Atrator mais proximo : {inv}  (desvio {dist:.4f})')

        print('\nCONCLUSAO:')
        if dist < 0.3:
            print(f'  Estrutura phi-proporcional DETECTADA.')
            print(f'  alpha* do substrato de linguagem (sinal) ~ {mean_r:.4f}')
        elif dist < 0.6:
            print(f'  Estrutura phi-proporcional PARCIAL — borda do atrator.')
            print(f'  Razao observada ({mean_r:.4f}) — desvio {dist:.4f} do invariante {inv}.')
        else:
            print(f'  Estrutura phi-proporcional nao detectada nesta escala.')
            print(f'  Razao observada ({mean_r:.4f}) — resultado valido, reportar como encontrado.')
    print(sep)


def main():
    if len(sys.argv) < 2:
        print('Uso: python eco_text_001_signal.py <arquivo.txt>')
        sys.exit(1)

    seed = int(datetime.now().timestamp())
    np.random.seed(seed)

    filepath = sys.argv[1]
    print(f'\nECO TEXT 001 — Medicao Espectral Multi-Escala (sinal direto)')
    print(f'Seed (timestamp): {seed}')

    text  = load_text(filepath)
    words = split_words(text)
    sents = split_sentences(text)
    paras = split_paragraphs(text)

    print(f'Corpus: {len(text)} chars | {len(words)} palavras | '
          f'{len(sents)} frases | {len(paras)} paragrafos')

    sig_char  = signal_chars(text)
    sig_word  = signal_word_lengths(words)
    sig_sent  = signal_sentence_lengths(sents)
    sig_para  = signal_para_lengths(paras)

    print('\nMedindo escalas...')
    s_char = measure_scale('caractere', sig_char)
    s_word = measure_scale('palavra',   sig_word)
    s_sent = measure_scale('frase',     sig_sent)
    s_para = measure_scale('paragrafo', sig_para)

    scales = [s_char, s_word, s_sent, s_para]
    sm     = {s['scale']: s for s in scales}

    ratios = []
    for a, b in [('caractere','palavra'), ('palavra','frase'), ('frase','paragrafo')]:
        b1 = sm[a]['beta']
        b2 = sm[b]['beta']
        if b1 > 1e-9:
            ratios.append({'from': a, 'to': b, 'ratio': b2 / b1})

    report(scales, ratios, filepath, seed)

    out = {
        'timestamp':  datetime.now().isoformat(),
        'seed':       seed,
        'file':       filepath,
        'chars':      len(text),
        'words':      len(words),
        'sentences':  len(sents),
        'paragraphs': len(paras),
        'invariants': {'phi3': PHI3, 'sqrt5': SQRT5, 'phi2': PHI2, 'phi': PHI},
        'scales':     scales,
        'ratios':     ratios,
    }
    out_path = f'eco_text_001_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nResultado salvo em: {out_path}')


if __name__ == '__main__':
    main()
