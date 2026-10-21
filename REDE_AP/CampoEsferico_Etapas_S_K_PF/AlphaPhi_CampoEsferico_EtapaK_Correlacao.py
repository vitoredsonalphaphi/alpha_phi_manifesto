# EXPERIMENTO: ETAPA K — a AlphaFi resiste a memórias parecidas? (critérios aprovados pelo pesquisador em 20/10/2026, antes da execução)
# 12 memórias por semente em 610 pontos (Grade Esférica v1), correlação média pedida 0,0 / 0,2 / 0,4 / 0,6 / 0,8.
# Família COM FORMA (principal): memória = sinal(Y·(cos t · zc/|zc| + sin t · zμ/|zμ|)); zc = entrada do EcoBIP traduzida como na Etapa H;
#   zμ sorteado com a mesma energia por grau de zc; t calibrado só sobre os padrões (semente de calibração 999, bissecção), sem rede.
# Família SEM FORMA (controle): ancestral = sinal(Y·zc); cada memória troca ao acaso a fração q = (1 − √c)/2 dos neurônios.
# Redes: C (Hebb), C-PI (pseudo-inversa), A (AlphaFi: vizinhança σ = 0,091, fundo do campo), A-PI (descritiva). Pistas 25% (principal) e 40%.
# Sementes 110–119. K00: C identifica >= 0,8 no nível 0,0 (com forma, 25%). K01: correlação medida a até 0,05 do nível.
# K1: A > C (com forma, 0,6). K2: A > C-PI (com forma, 0,6). K3: (A − C-PI) em 0,6 > em 0,0. K4: (A − C-PI) com forma > sem forma, em 0,6.
# (pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
import os
H = os.path.dirname(os.path.abspath(__file__)); RP_DIR = os.path.join(H, '..', 'CampoEsferico_Etapas_P_A_B_H_RP')  # dados: rodar antes AlphaPhi_EcoBIP_B0_gera_dobras.py e copiar os .npy para esta pasta
spec = importlib.util.spec_from_file_location('RP', os.path.join(RP_DIR, 'AlphaPhi_CampoEsferico_EtapaRP_Reprojecao.py')); RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
G, Hh = RP.G, RP.Hh
N = 610; P = 12; NIV = [0.0, 0.2, 0.4, 0.6, 0.8]; SEM = list(range(110, 120)); FAM = ['com forma', 'sem forma']
P610 = G.pontos_encaixados(N); Y, GR = G.harmonicos(P610, 14)
dob = np.load(H + '/ecobip_dobras.npy'); ent = np.load(H + '/ecobip_entrada.npy')
Lg = np.array([np.log(Hh.energias(s) + 1e-6) for s in [ent] + list(dob)]); ZC = (Lg - Lg.mean(0))[0]
EL = np.array([np.mean(ZC[GR == L] ** 2) for L in range(15)])[GR]   # energia por grau, por coeficiente
D = G.distancia_geodesica(P610) / np.pi; S610 = 2 * RP.espac(N)
CV = np.exp(-(D / S610) ** 2); np.fill_diagonal(CV, 0)
se = .15 * np.sqrt(233 / N); FU = np.exp(-(D / se) ** 2) - .5 * np.exp(-(D / (3 * se)) ** 2); np.fill_diagonal(FU, 0)
def cor_media(X): C = X @ X.T / N; return C[np.triu_indices(len(X), 1)].mean()
def com_forma(t, rng):
    zs = rng.normal(size=(P, len(ZC))) * np.sqrt(EL); u = ZC / np.linalg.norm(ZC)
    V = np.cos(t) * u + np.sin(t) * zs / np.linalg.norm(zs, axis=1, keepdims=True); return np.where(V @ Y.T >= 0, 1., -1.)
def sem_forma(c, rng):
    a = np.where(Y @ ZC >= 0, 1., -1.); q = (1 - np.sqrt(c)) / 2; X = np.tile(a, (P, 1))
    X[rng.random((P, N)) < q] *= -1; return X
def calibrar(c):
    if c == 0: return np.pi / 2
    lo, hi = 0., np.pi / 2
    for _ in range(30):
        t = (lo + hi) / 2; m = np.mean([cor_media(com_forma(t, np.random.default_rng([999, j]))) for j in range(10)])
        lo, hi = (t, hi) if m > c else (lo, t)
    return (lo + hi) / 2
def redes(X):
    Cm = X @ X.T / N; Hb = X.T @ X / N; Pi = X.T @ np.linalg.solve(Cm, X) / N
    J = {'C': Hb, 'C-PI': Pi, 'A': CV * Hb + .02 * FU, 'A-PI': CV * Pi + .02 * FU}
    for j in J.values(): np.fill_diagonal(j, 0)
    return J
def tarefa(arg):
    fam, c, t = arg; out = {r: {'id25': [], 'id40': [], 'm25': [], 'm40': []} for r in ['C', 'C-PI', 'A', 'A-PI']}; cors = []
    ni = int(round(c * 10)); fi = FAM.index(fam)
    for s in SEM:
        rng = np.random.default_rng([1_600_000 + s, ni, fi]); X = com_forma(t, rng) if fam == 'com forma' else sem_forma(c, rng)
        cors.append(float(cor_media(X))); J = redes(X)
        for flip in (.25, .40):
            r2 = np.random.default_rng([1_700_000 + s, ni, fi, int(flip * 100)]); S0 = X.copy(); k = int(round(flip * N))
            for p in range(P): S0[p, r2.choice(N, k, replace=False)] *= -1
            for nome, j in J.items():
                S = Hh.dinamica(j, X, S0, np.random.default_rng([1_800_000 + s, ni, fi, int(flip * 100)]))
                tag = '25' if flip == .25 else '40'
                out[nome]['id' + tag].append(float(np.mean(np.argmax(S @ X.T, 1) == np.arange(P)))); out[nome]['m' + tag].append(float((S * X).mean(1).mean()))
    print(f'  {fam} {c} pronto', file=sys.stderr, flush=True)
    return (fam, c), {'cor': cors, 'r': out}

if __name__ == '__main__':
    T = {c: calibrar(c) for c in NIV}; print('calibração t (rad):', {c: round(t, 4) for c, t in T.items()})
    with Pool(5) as pool: R = dict(pool.map(tarefa, [(f, c, T[c]) for f in FAM for c in NIV]))
    for f in FAM:
        print(f'\n##### família {f} — identificação (acaso 0,083) · pista 25% | 40%')
        print(f"{'nível':<7}{'medida':>8}" + ''.join(f'{r:>15}' for r in ['C', 'C-PI', 'A', 'A-PI']))
        for c in NIV:
            d = R[(f, c)]; print(f"{c:<7}{np.mean(d['cor']):>+8.3f}" + ''.join(f"{np.mean(d['r'][r]['id25']):>8.3f} |{np.mean(d['r'][r]['id40']):>5.3f}" for r in ['C', 'C-PI', 'A', 'A-PI']))
        print('sobreposição final, pista 25%:  ' + ' · '.join(f"{c}: " + '/'.join(f"{np.mean(R[(f, c)]['r'][r]['m25']):.2f}" for r in ['C', 'C-PI', 'A']) for c in NIV) + '  (C/C-PI/A)')
    k00 = np.mean(R[('com forma', 0.0)]['r']['C']['id25']); print(f"\nK00 C identifica no nível 0,0 (com forma, 25%): {k00:.3f} -> {'cumprido' if k00 >= .8 else 'NÃO cumprido'}")
    desv = max(abs(np.mean(R[(f, c)]['cor']) - c) for f in FAM for c in NIV); print(f"K01 maior desvio da correlação pedida: {desv:.3f} -> {'cumprido' if desv <= .05 else 'NÃO cumprido'}")
    col = lambda f, c, r: np.array(R[(f, c)]['r'][r]['id25'])
    def teste(k, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{k}: {d.mean():+.3f} (EP {se:.3f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    print('=== CRITÉRIOS (identificação, pista 25%) ===')
    teste('K1 A − C (com forma, 0,6)', col('com forma', .6, 'A') - col('com forma', .6, 'C'))
    teste('K2 A − C-PI (com forma, 0,6)', col('com forma', .6, 'A') - col('com forma', .6, 'C-PI'))
    dv = lambda f, c: col(f, c, 'A') - col(f, c, 'C-PI')
    teste('K3 (A − C-PI) em 0,6 − em 0,0 (com forma)', dv('com forma', .6) - dv('com forma', 0.0))
    teste('K4 (A − C-PI) com forma − sem forma (0,6)', dv('com forma', .6) - dv('sem forma', .6))
    json.dump({f'{f}|{c}': v for (f, c), v in R.items()}, open(H + '/EtapaK_resultados.json', 'w'))
