# EXPERIMENTO: ETAPA AU — a autonomia: ordenar sem dissolver (critérios aprovados pelo pesquisador em 30/10/2026, antes da execução)
# Base da Etapa LA (Grade v2; 12 memórias contínuas; regra delta local; neurônio N3; previsor β = 8). Sementes 280–289.
# Alvo T: a memória com uma variação legítima numa calota sorteada (≈10% dos neurônios; 5% e 20% descritivos): dentro da calota, T recebe
# uma forma nova e suave, nunca guardada (um campo do mesmo gerador das memórias, com outra semente). Pista: T + ruído 1,0 × o desvio
# (2,0 descritivo). Medidas: coerência (corr. com T fora da calota), preservação (corr. com T dentro), total (corr. com T na esfera).
# Versões: pista sozinha; consulta direta (a memória de maior ressonância com a pista, inteira); filtro local (cada neurônio = média dele
# com os vizinhos da Grade v2, pesos da vizinhança da memória, uma passada); Rede AP (recuperação da Etapa LA); Rede AP com β = 8φ (desc.).
# AU0: consulta preservação < 0,5 e pista coerência < 0,8. AU1: Rede > consulta (total). AU2: Rede > pista (total). AU3: Rede > filtro
# (total). AU4: Rede coerência > pista E Rede preservação > consulta. (pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
M = load('M', SP + 'exp_etapaSA_HZ.py'); NA, LA, FM = M.NA, M.LA, M.FM; N, P, PHI = M.N, M.P, M.PHI
U = NA.P2 / np.linalg.norm(NA.P2, axis=1, keepdims=True); CV = LA.CV
def alvo(X, s, frac):
    rng = np.random.default_rng([9_100_000 + s, int(frac * 100)]); Y = NA.memorias(10_000 + s)
    T = X.copy(); caps = []
    for k in range(P):
        c = rng.normal(size=3); c /= np.linalg.norm(c); dentro = U @ c > 1 - 2 * frac
        T[k, dentro] = Y[k, dentro]; caps.append(dentro)
    return T, np.array(caps)
def corr(a, b): return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0
def medir(S, T, caps):
    c = np.mean([corr(S[k, ~caps[k]], T[k, ~caps[k]]) for k in range(P)]); p = np.mean([corr(S[k, caps[k]], T[k, caps[k]]) for k in range(P)])
    t = np.mean([corr(S[k], T[k]) for k in range(P)]); return {'coer': float(c), 'pres': float(p), 'total': float(t)}
def filtro(S): return (S + S @ CV.T) / (1 + CV.sum(1))[None, :]
VERS = ['pista', 'consulta', 'filtro', 'rede', 'rede_8phi']
def tarefa(arg):
    ruido, frac = arg; out = {v: {'coer': [], 'pres': [], 'total': []} for v in VERS}; info = {'frac_real': [], 'corr_var_mem': [], 'ident_consulta': []}
    for s in range(280, 290):
        X = NA.memorias(s); T, caps = alvo(X, s, frac); S0 = NA.pista(T, ruido, s); J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X)
        info['frac_real'].append(float(caps.mean())); info['corr_var_mem'].append(float(np.mean([corr(T[k, caps[k]], X[k, caps[k]]) for k in range(P)])))
        kc = (np.sign(S0) @ X.T / N).argmax(1); info['ident_consulta'].append(float(np.mean(kc == np.arange(P))))
        rng = lambda: np.random.default_rng([9_200_000 + s, int(ruido * 10), int(frac * 100)])
        saidas = {'pista': S0, 'consulta': X[kc], 'filtro': filtro(S0),
                  'rede': M.recuperar(J, X, S0, rng(), g), 'rede_8phi': M.recuperar(J, X, S0, rng(), g, beta=lambda v: 8 * PHI)}
        for v, S in saidas.items():
            m = medir(S, T, caps)
            for k in m: out[v][k].append(m[k])
    print(f'  ruído {ruido}, região {frac} pronto', file=sys.stderr, flush=True); return arg, {'R': out, 'info': info}
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'conferir':
    X = NA.memorias(280); T, caps = alvo(X, 280, .10)
    print('fração real da calota:', round(float(caps.mean()), 3), '· corr. variação × memória dentro da calota:', round(float(np.mean([corr(T[k, caps[k]], X[k, caps[k]]) for k in range(P)])), 3))
    print('fora da calota T = X:', bool(np.all(T[~caps] == X[~caps])))
    S0 = NA.pista(T, 1.0, 280); print('pista: coerência/preservação/total', medir(S0, T, caps)); print('consulta:', medir(X[(np.sign(S0) @ X.T / N).argmax(1)], T, caps))
if __name__ == '__main__' and len(sys.argv) == 1:
    with Pool(4) as pool: res = dict(pool.map(tarefa, [(1.0, .10), (2.0, .10), (1.0, .05), (1.0, .20)]))
    for arg, o in res.items():
        r, I = o['R'], o['info']
        print(f"\n##### ruído {arg[0]} · região {arg[1]:.0%} (real {np.mean(I['frac_real']):.3f}) · variação×memória na região {np.mean(I['corr_var_mem']):+.3f} · consulta acerta a memória {np.mean(I['ident_consulta']):.3f}")
        print(f"{'versão':<11}{'coerência':>11}{'preservação':>13}{'total':>9}")
        for v in VERS: print(f"{v:<11}" + ''.join(f"{np.mean(r[v][k]):>{w}.4f}" for k, w in (('coer', 11), ('pres', 13), ('total', 9))))
    o = res[(1.0, .10)]['R']; c = lambda v, k: np.array(o[v][k])
    print('\n=== CRITÉRIOS (ruído 1,0, região 10%) ===')
    a, b = c('consulta', 'pres').mean(), c('pista', 'coer').mean()
    print(f"AU0 consulta preservação {a:.4f} (< 0,5) · pista coerência {b:.4f} (< 0,8) -> {'cumprido' if a < .5 and b < .8 else 'NÃO cumprido'}")
    M.teste('AU1 total Rede − consulta', c('rede', 'total') - c('consulta', 'total')); M.teste('AU2 total Rede − pista', c('rede', 'total') - c('pista', 'total'))
    M.teste('AU3 total Rede − filtro', c('rede', 'total') - c('filtro', 'total'))
    M.teste('AU4a coerência Rede − pista', c('rede', 'coer') - c('pista', 'coer')); M.teste('AU4b preservação Rede − consulta', c('rede', 'pres') - c('consulta', 'pres'))
    json.dump({f'{k[0]}|{k[1]}': v for k, v in res.items()}, open(SP + 'ecobip/exp_etapaAU.json', 'w'))
