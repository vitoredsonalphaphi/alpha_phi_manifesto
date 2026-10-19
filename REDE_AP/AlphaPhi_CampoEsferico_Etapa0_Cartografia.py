# Resultados: REDE_AP/AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md
# EXPERIMENTO: ETAPA 0 — CARTOGRAFIA DO CAMPO (critérios aprovados pelo pesquisador em 19/10/2026, antes da execução)
# Campos (256 neurônios): F1 esfera de Fibonacci; F2 posições aleatórias na esfera; F3 grade lat-long 16×16; F4 ligações de F1 embaralhadas.
# Ligação do campo vazio: e^-(d/0,15)^2 − 0,5·e^-(d/0,45)^2, d = distância geodésica normalizada (0..1).
# Ambientes: com forma = campo suave na esfera (soma de 400 bolhas gaussianas de sinal aleatório, comprimento 0,5, avaliada nos pontos de cada campo) → sinal ±1;
#            sem forma = ±1 ao acaso. 200 padrões por semente.
# R = fração da energia de cada padrão que cabe nas 16 formas principais (autovetores de maior autovalor) do campo. Sem forma: esperado 16/256.
# Censo do campo vazio (descritivo): 200 estados aleatórios, dinâmica por sinal até 50 voltas; atratores distintos, bacias, voltas.
# Memória: padrões do ambiente com forma, cargas 4/8/12/16, ligações e^-(d/0,35)^2 (embaralhadas em F4), pista 25%, medida principal.
# Z0: R(sem forma) entre 4,25% e 8,25% em todos os campos.  Z1: R F1 > F4.  Z2: R F1 > F2.  Z3: R F1 > F3 (ambiente com forma).
# Z4: memória no campo de maior R médio > no de menor R médio.  Regra: diferença pareada > 2 EP e vantagem em >= 8 de 10 sementes.
import numpy as np, importlib.util, json
spec = importlib.util.spec_from_file_location('base', __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'AlphaPhi_MemoriaEsferica_Etapa1.py'))
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
N, K, SEM, NPAD = 256, 16, list(range(10)), 200
CARGAS0 = [4, 8, 12, 16]

def latlong():
    th = (np.arange(16) + .5) / 16 * np.pi; ph = np.arange(16) / 16 * 2 * np.pi
    T, P = np.meshgrid(th, ph, indexing='ij'); T, P = T.ravel(), P.ravel()
    return np.c_[np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)]
def dnorm(P): return np.arccos(np.clip(P @ P.T, -1, 1)) / np.pi
def dog(d): C = np.exp(-(d / .15) ** 2) - .5 * np.exp(-(d / .45) ** 2); np.fill_diagonal(C, 0); return C
def viz(d): C = np.exp(-(d / .35) ** 2); np.fill_diagonal(C, 0); return C
def perm_pares(rng):
    iu = np.triu_indices(N, 1); return iu, rng.permutation(len(iu[0]))
def aplica_perm(C, iu, p):
    D = np.zeros_like(C); D[iu] = C[iu][p]; return D + D.T
def modos(C):
    w, V = np.linalg.eigh(C); return V[:, np.argsort(w)[::-1][:K]]
def R(U, X): return float(np.mean(((X @ U) ** 2).sum(1) / (X ** 2).sum(1)))
def suave(P, rng, n):
    c = rng.normal(size=(n, 400, 3)); c /= np.linalg.norm(c, axis=2, keepdims=True); s = rng.choice([-1, 1], size=(n, 400))
    d2 = ((P[None, None, :, :] - c[:, :, None, :]) ** 2).sum(-1)            # (n,400,N) distância de corda ao quadrado
    f = (s[:, :, None] * np.exp(-d2 / .25)).sum(1); return np.where(f >= 0, 1., -1.)
def censo(C, rng):
    S = rng.choice([-1., 1.], size=(NPAD, N)); vol = np.full(NPAD, base.MAXV)
    for v in range(base.MAXV):
        mud = np.zeros(NPAD, int)
        for i in rng.permutation(N):
            h = S @ C[:, i]; novo = np.where(h > 0, 1, np.where(h < 0, -1, S[:, i])); mud += novo != S[:, i]; S[:, i] = novo
        vol[(mud == 0) & (vol == base.MAXV)] = v + 1
        if (mud == 0).all(): break
    canon = np.where(S[:, :1] < 0, -S, S); _, cont = np.unique(canon, axis=0, return_counts=True)
    return len(cont), float(cont.max() / NPAD), float(vol.mean())
def memoria(Cm, P, seed):
    ms = []
    for Pc in CARGAS0:
        rng = np.random.default_rng([90_000 + seed, Pc]); X = suave(P, np.random.default_rng([91_000 + seed, Pc]), Pc)
        J = Cm * (X.T @ X) / N; np.fill_diagonal(J, 0); m, _, _ = base.recupera(J, X, .25, rng); ms.append(m.mean())
    return float(np.mean(ms)), [float(x) for x in ms]

if __name__ == '__main__':
    PF, PL = base.fib_esfera(N), latlong()
    res = {f: {'Rs': [], 'Rb': [], 'censo': [], 'mem': [], 'memc': []} for f in ('F1', 'F2', 'F3', 'F4')}
    H0 = []
    for s in SEM:
        P2 = base.aleat_esfera(N, np.random.default_rng(30_000 + s))
        iu, p = perm_pares(np.random.default_rng(20_000 + s))
        campos = {'F1': (PF, dog(dnorm(PF)), viz(dnorm(PF))), 'F2': (P2, dog(dnorm(P2)), viz(dnorm(P2))),
                  'F3': (PL, dog(dnorm(PL)), viz(dnorm(PL))),
                  'F4': (PF, aplica_perm(dog(dnorm(PF)), iu, p), aplica_perm(viz(dnorm(PF)), iu, p))}
        Xb = np.random.default_rng(40_000 + s).choice([-1., 1.], size=(NPAD, N))
        for f, (P, Cd, Cm) in campos.items():
            U = modos(Cd); Xs = suave(P, np.random.default_rng(50_000 + s), NPAD)
            res[f]['Rs'].append(R(U, Xs)); res[f]['Rb'].append(R(U, Xb))
            res[f]['censo'].append(censo(Cd, np.random.default_rng(60_000 + s)))
            m, mc = memoria(Cm, P, s); res[f]['mem'].append(m); res[f]['memc'].append(mc)
        UM = np.ones((N, N)); np.fill_diagonal(UM, 0); H0.append(memoria(UM, PF, s)[0])
        print(f'  semente {s} pronta', flush=True)
    print(f'\nRESSONÂNCIA R (16 formas de 256; esperado sem forma = {K/N:.4f}) e CENSO DO CAMPO VAZIO (média de 10 sementes)')
    print(f"{'campo':<6}{'R com forma':>14}{'R sem forma':>14}{'atratores':>11}{'maior bacia':>13}{'voltas':>8}{'memória':>10}   por carga " + ' '.join(f'P={c}' for c in CARGAS0))
    for f, r in res.items():
        cz = np.array(r['censo']); mc = np.mean(r['memc'], 0)
        print(f"{f:<6}{np.mean(r['Rs']):>9.4f}±{np.std(r['Rs']):.3f}{np.mean(r['Rb']):>9.4f}±{np.std(r['Rb']):.3f}{cz[:,0].mean():>11.1f}{cz[:,1].mean():>13.3f}{cz[:,2].mean():>8.1f}{np.mean(r['mem']):>10.4f}   " + ' '.join(f'{x:.3f}' for x in mc))
    print(f"H0 (todos ligados, pontos F1): memória {np.mean(H0):.4f}")
    print('\n=== CRITÉRIOS ===')
    z0 = {f: np.mean(r['Rb']) for f, r in res.items()}; ok0 = all(.0425 <= v <= .0825 for v in z0.values())
    print('Z0 R sem forma: ' + ', '.join(f'{f} {v:.4f}' for f, v in z0.items()) + f' -> {"cumprido" if ok0 else "NÃO cumprido"}')
    for k, b in (('Z1', 'F4'), ('Z2', 'F2'), ('Z3', 'F3')):
        d, se, nf, ok = base.pareado(res['F1']['Rs'], res[b]['Rs']); print(f'{k} R F1 − {b}: {d:+.4f} (EP {se:.4f}, {nf}/10) -> {"cumprido" if ok else "NÃO cumprido"}')
    medR = {f: np.mean(r['Rs']) for f, r in res.items()}; hi, lo = max(medR, key=medR.get), min(medR, key=medR.get)
    d, se, nf, ok = base.pareado(res[hi]['mem'], res[lo]['mem'])
    print(f'Z4 memória no campo de maior R ({hi}) − menor R ({lo}): {d:+.4f} (EP {se:.4f}, {nf}/10) -> {"cumprido" if ok else "NÃO cumprido"}')
    json.dump({f: {k: v for k, v in r.items()} for f, r in res.items()} | {'H0': H0}, open('exp_etapa0.json', 'w'))
