# EXPERIMENTO: ETAPA VL4 — a decisão no além (critérios aprovados pelo pesquisador em 22/10/2026, antes da execução)
# Base da VL3 (Grade v2, anéis 75 × 8; regra delta local; neurônio N3; previsor ligado; mesmos trechos). Ruído branco +10 dB (principal) e 0 dB.
# Sementes 210–219. O além entrega: M (mistura macia, β = 8); Bc (mistura com β calibrado só pelas memórias: o menor β que dá peso >= 0,99
# à própria memória intacta, para todas as memórias); DC (a memória de maior ressonância com o estado final, exata); DS (decide só se o peso
# da escolhida, com β calibrado, for >= 0,5; se não, ganho 1 = som inalterado); DP (controle: decisão direto na pista); teto (oráculo).
# VL4-1: DC > 0 dB. VL4-2: DC > M. VL4-3: Bc > M. VL4-4: DC >= DP. VL4-5: DS >= 0 dB a +10 e a 0 dB. (pareado > 2 EP e >= 8/10; para VL4-4,
# "DC >= DP" = a média de DC − DP não é negativa além de 2 EP.) Descritivo: identificação na pista e no fim; trechos novos; fração do teto.
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V3 = load('V3', SP + 'exp_etapaVL3.py'); V2, LA, NA, E = V3.V2, V3.LA, V3.NA, V3.E
N = 610; SEM = list(range(210, 220)); SNRS = [10, 0]
def resson(S, X): return np.sign(S) @ X.T / N
def pesos(S, X, beta): m = resson(S, X); w = np.exp(beta * (m - m.max(1, keepdims=True))); return w / w.sum(1, keepdims=True)
def beta_calibrado(X):
    lo, hi = 0.1, 1e4
    for _ in range(100):
        b = np.sqrt(lo * hi); ok = np.diag(pesos(X, X, b)).min() >= 0.99
        hi, lo = (b, lo) if ok else (hi, b)
    return hi
VERS = ['M', 'Bc', 'DC', 'DS', 'DP', 'teto']
def rodar(arg):
    db, quais = arg; T = V3.Tarefa(); P = len(T.esc); X = T.X; J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X); bc = beta_calibrado(X)
    res = {v: [] for v in VERS}; res.update({'id_pista': [], 'id_fim': [], 'decide_DS': []})
    for s in SEM:
        idx = T.esc if quais == 'guardados' else [min(i + 1, len(T.segs) - 1) for i in T.esc]
        limpos = [T.segs[i] for i in idx]; ruid = [V2.ruido(c, db, np.random.default_rng([s, db + 50, k])) for k, c in enumerate(limpos)]
        S0 = np.array([T.pista(r) for r in ruid]); Sf = NA.recuperar(J, X, S0, np.random.default_rng([3_000_000 + s, db + 50]), g, True, True)
        kf = resson(Sf, X).argmax(1); kp = resson(S0, X).argmax(1); wb = pesos(Sf, X, bc)
        res['id_pista'].append(float(np.mean(kp == np.arange(P)))); res['id_fim'].append(float(np.mean(kf == np.arange(P))))
        dec = wb.max(1) >= 0.5; res['decide_DS'].append(float(dec.mean()))
        saida = {'M': pesos(Sf, X, 8) @ X, 'Bc': wb @ X, 'DC': X[kf], 'DP': X[kp]}
        ganho = lambda y, k: V2.snr(y, limpos[k]) - V2.snr(ruid[k], limpos[k])
        for v, S in saida.items(): res[v].append(float(np.mean([ganho(T.decod(S[k], ruid[k]), k) for k in range(P)])))
        res['DS'].append(float(np.mean([ganho(T.decod(X[kf[k]], ruid[k]), k) if dec[k] else 0.0 for k in range(P)])))
        res['teto'].append(float(np.mean([ganho(T.oraculo(limpos[k], ruid[k]), k) for k in range(P)])))
    print(f'  {db:+d} dB {quais} pronto', file=sys.stderr, flush=True)
    return arg, {'R': res, 'beta_calibrado': bc, 'P': P}
if __name__ == '__main__':
    with Pool(4) as pool: out = dict(pool.map(rodar, [(10, 'guardados'), (0, 'guardados'), (10, 'novos'), (0, 'novos')]))
    print(f"β calibrado: {out[(10, 'guardados')]['beta_calibrado']:.1f} · memórias: {out[(10, 'guardados')]['P']}")
    for (db, q), o in out.items():
        r = o['R']; print(f"\n##### {db:+d} dB · trechos {q} · identificação: pista {np.mean(r['id_pista']):.3f}, fim {np.mean(r['id_fim']):.3f} · DS decide em {np.mean(r['decide_DS']):.3f}")
        for v in VERS: print(f"   {v:<5}{np.mean(r[v]):+7.2f} dB (EP {np.std(r[v], ddof=1) / np.sqrt(10):.2f}) · fração do teto {np.mean(r[v]) / np.mean(r['teto']):+.2f}")
    col = lambda db, v: np.array(out[(db, 'guardados')]['R'][v])
    def teste(nome, d, nao_neg=False):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum()
        ok = (d.mean() > -2 * se) if nao_neg else (d.mean() > 2 * se and nf >= 8 and d.mean() > 0)
        print(f"{nome}: {d.mean():+.2f} dB (EP {se:.2f}, {nf}/10 a favor) -> {'cumprido' if ok else 'NÃO cumprido'}")
    print('\n=== CRITÉRIOS (trechos guardados) ===')
    teste('VL4-1 DC (+10 dB)', col(10, 'DC')); teste('VL4-2 DC − M (+10 dB)', col(10, 'DC') - col(10, 'M'))
    teste('VL4-3 Bc − M (+10 dB)', col(10, 'Bc') - col(10, 'M')); teste('VL4-4 DC − DP (+10 dB), não negativo', col(10, 'DC') - col(10, 'DP'), True)
    for db in SNRS:
        d = col(db, 'DS'); se = d.std(ddof=1) / np.sqrt(10)
        print(f"VL4-5 DS ({db:+d} dB) >= 0: {d.mean():+.2f} dB (EP {se:.2f}, mínimo {d.min():+.2f}) -> {'cumprido' if d.mean() > -2 * se else 'NÃO cumprido'}")
    json.dump({f'{db}|{q}': o for (db, q), o in out.items()}, open(SP + 'ecobip/exp_etapaVL4.json', 'w'))
