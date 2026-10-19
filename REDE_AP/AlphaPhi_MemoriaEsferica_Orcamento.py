# Resultados: REDE_AP/AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md
# EXPERIMENTO: MEMÓRIA ESFÉRICA — ORÇAMENTO DE LIGAÇÕES (critérios escritos na conversa em 19/10/2026, antes da execução)
# Mesmo desenho da etapa 1 (256 neurônios ±1, cargas 5..36, pistas 25%/40%, 10 sementes, pareamento exato, medida principal),
# com ligações BINÁRIAS e o MESMO número de ligações em todas as condições (o número que a regra φ produz).
# B-φ: saltos de Fibonacci (1,2,3,5,...,233) na ordem da esfera de Fibonacci.   B-φemb: mesmos saltos numa ordem embaralhada.
# B-G: os pares mais próximos na esfera.   B-A: pares ao acaso.   B-PM: B-G com 10% das ligações religadas ao acaso.
# L0: Hopfield completa, P=13, pista 25%: sucesso >= 95%.   L1: B-G > B-A.   L2: B-φ > B-A.   L3: B-φ > B-PM.   L4: B-φ > B-φemb.
# L5: B-φ >= 95% da medida principal de B-A usando <= 60% do comprimento total de fio.
# Regra: diferença pareada média > 2 EP e vantagem em >= 8 de 10 sementes.
import numpy as np, importlib.util, json
spec = importlib.util.spec_from_file_location('base', __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'AlphaPhi_MemoriaEsferica_Etapa1.py'))
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
N, CARGAS, SEM = base.N, base.CARGAS, list(range(10))
PF = base.fib_esfera(N); D = np.arccos(np.clip(PF @ PF.T, -1, 1))          # distância geodésica (rad)
FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

def mascara(pares):
    M = np.zeros((N, N)); i, j = np.array(pares).T; M[i, j] = 1; M[j, i] = 1; np.fill_diagonal(M, 0); return M
def pares_de(M):
    iu = np.triu_indices(N, 1); s = M[iu] > 0; return list(zip(iu[0][s], iu[1][s]))
def b_phi(ordem=None):
    o = np.arange(N) if ordem is None else ordem
    return mascara([(o[i], o[i + f]) for f in FIB for i in range(N - f)])
M_PHI = b_phi(); E = int(M_PHI[np.triu_indices(N, 1)].sum())
iu = np.triu_indices(N, 1)
def b_g():
    idx = np.argsort(D[iu], kind='stable')[:E]; return mascara(list(zip(iu[0][idx], iu[1][idx])))
def b_a(rng):
    idx = rng.choice(len(iu[0]), E, replace=False); return mascara(list(zip(iu[0][idx], iu[1][idx])))
def b_pm(rng, frac=.10):
    M = b_g().copy(); pr = pares_de(M); k = int(round(frac * E))
    for t in rng.choice(len(pr), k, replace=False): i, j = pr[t]; M[i, j] = M[j, i] = 0
    vazios = np.flatnonzero(M[iu] == 0); novos = rng.choice(vazios, k, replace=False)
    M[iu[0][novos], iu[1][novos]] = 1; M[iu[1][novos], iu[0][novos]] = 1; return M
def fio(M): return float((D * M)[iu].sum())

UM = np.ones((N, N)); np.fill_diagonal(UM, 0)
cond = {
    'H0':     lambda s: UM,
    'B-phi':  lambda s: M_PHI,
    'B-phiemb': lambda s: b_phi(np.random.default_rng(50_000 + s).permutation(N)),
    'B-G':    lambda s: b_g(),
    'B-A':    lambda s: b_a(np.random.default_rng(60_000 + s)),
    'B-PM':   lambda s: b_pm(np.random.default_rng(70_000 + s)),
}
if __name__ == '__main__':
    print(f'orçamento: {E} ligações ({E / len(iu[0]) * 100:.1f}% dos pares; grau médio {2 * E / N:.1f})')
    for c in cond:
        if c != 'H0':
            M = cond[c](0); assert int(M[iu].sum()) == E, c
            print(f'   {c:<9} ligações {int(M[iu].sum())}  fio total {fio(M):8.1f} rad  (semente 0)')
    R = {c: {fl: [base.avalia(f, s, fl) for s in SEM] for fl in (.25, .40)} for c, f in cond.items()}
    col = lambda c, fl=.25: [r['principal'] for r in R[c][fl]]
    print('\nMEDIDA PRINCIPAL, média ± dp em 10 sementes')
    print(f"{'cond':<10}{'pista 25%':>16}{'pista 40%':>16}   sobreposição por carga (25%)")
    for c in cond:
        a, b = np.array(col(c)), np.array(col(c, .40)); mc = np.mean([r['m_carga'] for r in R[c][.25]], 0)
        print(f"{c:<10}{a.mean():>10.4f}±{a.std():.3f}{b.mean():>10.4f}±{b.std():.3f}   " + ' '.join(f'{x:.3f}' for x in mc))
    print('\n=== CRITÉRIOS ===')
    l0 = np.mean([r['sucesso'][13] for r in R['H0'][.25]]); print(f'L0 H0 P=13 pista 25%: {l0:.3f} -> {"cumprido" if l0 >= .95 else "NÃO cumprido"}')
    for k, a, b in (('L1', 'B-G', 'B-A'), ('L2', 'B-phi', 'B-A'), ('L3', 'B-phi', 'B-PM'), ('L4', 'B-phi', 'B-phiemb')):
        d, se, nf, ok = base.pareado(col(a), col(b)); print(f'{k} {a} − {b}: {d:+.4f} (EP {se:.4f}, {nf}/10) -> {"cumprido" if ok else "NÃO cumprido"}')
    fios = {c: np.mean([fio(cond[c](s)) for s in SEM]) for c in cond if c != 'H0'}
    razao_m = np.mean(col('B-phi')) / np.mean(col('B-A')); razao_f = fios['B-phi'] / fios['B-A']
    print(f'L5 B-φ/B-A: medida {razao_m:.3f} (precisa >= 0,95), fio {razao_f:.3f} (precisa <= 0,60) -> {"cumprido" if razao_m >= .95 and razao_f <= .60 else "NÃO cumprido"}')
    print('   fio médio (rad): ' + ' | '.join(f'{c} {v:.0f}' for c, v in fios.items()))
    json.dump({'E': E, 'fios': fios, 'R': {c: {str(fl): v for fl, v in R[c].items()} for c in R}}, open('exp_memoria_orcamento.json', 'w'))
