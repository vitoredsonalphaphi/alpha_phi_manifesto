#!/usr/bin/env python3
"""
ECO TEXT 001 — Medicao Espectral Multi-Escala em Texto
Manifesto AlphaPhi · Vitor Edson Delavi · 2026

Pre-funcao para o substrato de linguagem:
Mede estrutura espectral em 4 escalas (palavra -> frase -> paragrafo -> texto)
e busca o atrator phi — equivalente do alpha*=1/3 encontrado no ECO BEEP 880.

Protocolo de integridade:
- Seed gerada por timestamp — nenhum valor escolhido pelo pesquisador
- phi aparece apenas na particao espectral — nunca como parametro de teste
- Resultados reportados na integra

Uso: python eco_text_001.py <arquivo_de_texto.txt>
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import re
import sys
import json
from datetime import datetime

# Constantes AlphaPhi — irrevogaveis
PHI   = (1 + np.sqrt(5)) / 2    # 1.6180339887... — organizador de coerencia
PHI2  = PHI ** 2                 # 2.6180339887...
PHI3  = PHI ** 3                 # 4.2360679774... — atrator ECO BEEP 880
SQRT5 = np.sqrt(5)               # 2.2360679774... — invariante serial
ALPHA = 1 / 137.035999084        # 0.0072973...    — granularidade minima


# ── Segmentacao do texto ──────────────────────────────────────────────────────

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def split_words(text):
    return re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text)

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 15]

def split_paragraphs(text):
    parts = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 50]


# ── Medicoes espectrais ───────────────────────────────────────────────────────

def embed(model, units, sample=200):
    if len(units) > sample:
        idx = np.random.choice(len(units), sample, replace=False)
        units = [units[i] for i in sorted(idx)]
    return model.encode(units, batch_size=64, show_progress_bar=False)

def beta_phi(embeddings):
    """
    Razao espectral beta em particao phi-proporcional.
    Mesmo principio do ECO BEEP 880 aplicado a embeddings.
    """
    mean_emb = np.mean(embeddings, axis=0)
    energy   = np.abs(np.fft.fft(mean_emb))
    n        = len(energy)
    split    = int(n / PHI)
    e1       = energy[:split].sum()
    e2       = energy[split:].sum()
    return e1 / (e2 + 1e-10)

def coherence(embeddings):
    """Coerencia espectral media — mesma funcao do phi_spectral_modulator."""
    vals = []
    for emb in embeddings:
        energy = np.abs(np.fft.fft(emb))
        n      = len(energy)
        enorm  = np.clip(energy / (energy.sum() + 1e-8), 1e-10, 1.0)
        ent    = -np.sum(enorm * np.log(enorm))
        vals.append(1.0 - ent / np.log(n))
    return float(np.mean(vals)), float(np.std(vals))

def measure(model, units, name):
    print(f"  [{name}] {len(units)} unidades...", flush=True)
    embs = embed(model, units)
    b    = beta_phi(embs)
    c, s = coherence(embs)
    return {'scale': name, 'n': len(units), 'beta': b,
            'coherence': c, 'coherence_std': s}


# ── Relatorio ─────────────────────────────────────────────────────────────────

def closest_invariant(val):
    candidates = [(abs(val - PHI3), f'phi3 ({PHI3:.4f})'),
                  (abs(val - SQRT5), f'sqrt5 ({SQRT5:.4f})'),
                  (abs(val - PHI2),  f'phi2 ({PHI2:.4f})'),
                  (abs(val - PHI),   f'phi ({PHI:.4f})')]
    d, name = min(candidates, key=lambda x: x[0])
    return name, d

def report(scales, ratios, filepath):
    sep = '=' * 62
    print(f'\n{sep}')
    print('ECO TEXT 001 — Relatorio de Medicao Espectral Multi-Escala')
    print(f'Manifesto AlphaPhi · {datetime.now().strftime("%d/%m/%Y %H:%M")}')
    print(f'Arquivo: {filepath}')
    print(sep)

    print('\nINVARIANTES DE REFERENCIA (ECO BEEP 880):')
    print(f'  phi3  = {PHI3:.6f}  <- atrator audio (880Hz)')
    print(f'  sqrt5 = {SQRT5:.6f}  <- invariante serial')
    print(f'  phi2  = {PHI2:.6f}')
    print(f'  phi   = {PHI:.6f}')

    print(f'\n{"Escala":<14}{"N":<8}{"beta":<14}{"Coerencia":<14}{"Desvio"}')
    print('-' * 60)
    for s in scales:
        print(f'{s["scale"]:<14}{s["n"]:<8}{s["beta"]:<14.6f}'
              f'{s["coherence"]:<14.6f}+/-{s["coherence_std"]:.4f}')

    print('\nRAZOES ENTRE ESCALAS (atrator buscado):')
    print('-' * 60)
    ratio_vals = []
    for r in ratios:
        inv, dist = closest_invariant(r['ratio'])
        tag = f'  <- proximo de {inv} (desvio {dist:.4f})'
        print(f'  {r["from"]:<12} -> {r["to"]:<14} {r["ratio"]:.6f}{tag}')
        ratio_vals.append(r['ratio'])

    if ratio_vals:
        mean_r = float(np.mean(ratio_vals))
        std_r  = float(np.std(ratio_vals))
        inv, dist = closest_invariant(mean_r)
        print(f'\n  Media das razoes: {mean_r:.6f} +/- {std_r:.6f}')
        print(f'  Atrator mais proximo: {inv} (desvio {dist:.4f})')

        print('\nCONCLUSAO:')
        if dist < 0.3:
            print(f'  Estrutura phi-proporcional DETECTADA entre escalas.')
            print(f'  alpha* do substrato de linguagem ~ {mean_r:.4f}')
        else:
            print(f'  Estrutura phi-proporcional NAO detectada claramente.')
            print(f'  Razao observada ({mean_r:.4f}) nao converge para invariantes conhecidos.')
            print(f'  Resultado valido — reportar como encontrado.')
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print('Uso: python eco_text_001.py <arquivo.txt>')
        sys.exit(1)

    # Seed por timestamp — protocolo de integridade AlphaPhi
    seed = int(datetime.now().timestamp())
    np.random.seed(seed)

    filepath = sys.argv[1]
    print(f'\nECO TEXT 001 — Medicao Espectral Multi-Escala')
    print(f'Seed (timestamp): {seed}')
    print(f'Carregando modelo MiniLM...')

    model = SentenceTransformer('all-MiniLM-L6-v2')
    text  = load_text(filepath)

    words  = split_words(text)
    sents  = split_sentences(text)
    paras  = split_paragraphs(text)

    print(f'Corpus: {len(text)} caracteres | '
          f'{len(words)} palavras | {len(sents)} frases | '
          f'{len(paras)} paragrafos')
    print('\nMedindo escalas...')

    s_word = measure(model, words, 'palavra')
    s_sent = measure(model, sents, 'frase')
    s_para = measure(model, paras, 'paragrafo')
    s_text = measure(model, [text[:8000]], 'texto')

    scales = [s_word, s_sent, s_para, s_text]
    sm     = {s['scale']: s for s in scales}

    ratios = []
    for a, b in [('palavra','frase'), ('frase','paragrafo'), ('paragrafo','texto')]:
        b1 = sm[a]['beta']
        b2 = sm[b]['beta']
        if b1 > 0:
            ratios.append({'from': a, 'to': b, 'ratio': b2 / b1})

    report(scales, ratios, filepath)

    # Salvar resultado
    out = {
        'timestamp':  datetime.now().isoformat(),
        'seed':       seed,
        'file':       filepath,
        'chars':      len(text),
        'invariants': {'phi3': PHI3, 'sqrt5': SQRT5, 'phi2': PHI2, 'phi': PHI},
        'scales':     scales,
        'ratios':     ratios
    }
    out_path = f'eco_text_001_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nResultado salvo em: {out_path}')


if __name__ == '__main__':
    main()
