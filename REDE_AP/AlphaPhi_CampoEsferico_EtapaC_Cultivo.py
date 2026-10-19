# Resultados: REDE_AP/AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md
# EXPERIMENTO: ETAPA C — CULTIVO DO CAMPO (critérios escritos na conversa em 19/10/2026, antes da execução; escolha delegada pelo pesquisador)
# Pontos encaixados: ângulo áureo + altura frac(i·√2). Etapas N = 34, 55, 89, 144, 233.
# Ligação por etapa: e^-(d/σe)^2 − 0,5·e^-(d/3σe)^2, σe = 0,15·√(233/N), d = ângulo/π. Previsão por Funk–Hecke; faixa = graus com força >= 90% da máxima.
# Cultivado: estado aleatório em 34; a cada etapa, antigos mantêm o estado, novos nascem com o sinal da influência dos antigos; dinâmica até assentar.
# Construído: estado aleatório com todos os pontos em cada tamanho. 50 repetições.
# C0: condição do medidor <= 10 em todas as etapas.  C1 (cultivado) e C2 (construído): grau dominante na faixa em >= 80% das repetições, em todas as etapas.
# C3: energia final cultivado < construído (pareado, > 2 EP e >= 40/50).  C4: fração média de antigos que mantêm o estado >= 0,8 em todas as 4 passagens.
import numpy as np, importlib.util, json
spec = importlib.util.spec_from_file_location('t', __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'AlphaPhi_CampoEsferico_EtapaT_Previsor.py'))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
PHI = (1 + 5 ** .5) / 2; GA = 2 * np.pi * (1 - 1 / PHI)
ETAPAS = [(34, 3), (55, 4), (89, 5), (144, 7), (233, 9)]; REP = 50; MAXV = 50
def pontos(n):
    i = np.arange(1, n + 1); z = 1 - 2 * ((i * 2 ** .5) % 1); r = np.sqrt(1 - z * z); a = i * GA
    return np.c_[r * np.cos(a), r * np.sin(a), z]
def ligacao(P, se):
    d = np.arccos(np.clip(P @ P.T, -1, 1)) / np.pi; C = np.exp(-(d / se) ** 2) - .5 * np.exp(-(d / (3 * se)) ** 2); np.fill_diagonal(C, 0); return C
def dinamica(C, s, rng):
    s = s.copy()
    for v in range(MAXV):
        mud = 0
        for i in rng.permutation(len(s)):
            h = C[i] @ s; novo = 1. if h > 0 else (-1. if h < 0 else s[i]); mud += novo != s[i]; s[i] = novo
        if mud == 0: break
    return s
def energia(C, s): return float(-(s @ C @ s) / (2 * len(s)))
def grau(s, Y, g): return int(t.grau_dominante(s[None], Y, g)[0])

if __name__ == '__main__':
    PT = pontos(233); prev = {}
    for N, Lmax in ETAPAS:
        se = .15 * np.sqrt(233 / N); lam = t.funk_hecke(t.nucleo(se, 3 * se), 20)
        Y, g = t.harmonicos(PT[:N], Lmax)
        prev[N] = dict(se=se, Ls=int(lam.argmax()), faixa=[L for L in range(21) if lam[L] >= .9 * lam.max()], Y=Y, g=g, cond=float(np.linalg.cond(Y)), C=ligacao(PT[:N], se))
    print('C0 condição do medidor: ' + ', '.join(f'N={N} {prev[N]["cond"]:.1f}' for N, _ in ETAPAS) + f' -> {"cumprido" if all(prev[N]["cond"] <= 10 for N, _ in ETAPAS) else "NÃO cumprido"}')
    print('previsão: ' + ', '.join(f'N={N} faixa {prev[N]["faixa"]}' for N, _ in ETAPAS))
    R = {'cult': {N: [] for N, _ in ETAPAS}, 'const': {N: [] for N, _ in ETAPAS}, 'preserva': {N: [] for N, _ in ETAPAS[1:]}}
    for r in range(REP):
        rng = np.random.default_rng([900_000, r]); s = None
        for k, (N, _) in enumerate(ETAPAS):
            C = prev[N]['C']
            if s is None: s0 = rng.choice([-1., 1.], N)
            else:
                n0 = len(s); h = C[n0:, :n0] @ s; novos = np.where(h > 0, 1., np.where(h < 0, -1., rng.choice([-1., 1.], N - n0)))
                s0 = np.r_[s, novos]
            s1 = dinamica(C, s0, rng)
            if s is not None: R['preserva'][N].append(float((s1[:len(s)] == s).mean()))
            s = s1; R['cult'][N].append((grau(s, prev[N]['Y'], prev[N]['g']), energia(C, s)))
            sb = dinamica(C, np.random.default_rng([910_000, r, N]).choice([-1., 1.], N), np.random.default_rng([920_000, r, N]))
            R['const'][N].append((grau(sb, prev[N]['Y'], prev[N]['g']), energia(C, sb)))
    print(f"\n{'N':<5}{'faixa':>8}   {'cultivado: na faixa · graus (0..5)':<44}{'construído: na faixa · graus (0..5)':<44}{'preserva':>9}")
    ok1 = ok2 = True
    for N, _ in ETAPAS:
        f = prev[N]['faixa']; gc = np.array([x[0] for x in R['cult'][N]]); gb = np.array([x[0] for x in R['const'][N]])
        a, b = np.isin(gc, f).mean(), np.isin(gb, f).mean(); ok1 &= a >= .8; ok2 &= b >= .8
        pr = f"{np.mean(R['preserva'][N]):.3f}" if N in R['preserva'] else '—'
        print(f"{N:<5}{str(f):>8}   {a:.2f} · {np.bincount(gc, minlength=6)[:6].tolist()!s:<35}{b:.2f} · {np.bincount(gb, minlength=6)[:6].tolist()!s:<35}{pr:>9}")
    print(f"\nC1 cultivado na faixa em todas as etapas -> {'cumprido' if ok1 else 'NÃO cumprido'}")
    print(f"C2 construído na faixa em todas as etapas -> {'cumprido' if ok2 else 'NÃO cumprido'}")
    ec = np.array([x[1] for x in R['cult'][233]]); eb = np.array([x[1] for x in R['const'][233]]); d = eb - ec
    se_ = d.std(ddof=1) / np.sqrt(len(d)); ok3 = d.mean() > 2 * se_ and (d > 0).sum() >= 40
    print(f"C3 energia final: cultivado {ec.mean():.4f} | construído {eb.mean():.4f} | diferença {d.mean():+.4f} (EP {se_:.4f}, cultivado mais baixo em {(d>0).sum()}/50) -> {'cumprido' if ok3 else 'NÃO cumprido'}")
    pm = {N: np.mean(v) for N, v in R['preserva'].items()}
    print('C4 preservação: ' + ', '.join(f'{N}: {v:.3f}' for N, v in pm.items()) + f" -> {'cumprido' if all(v >= .8 for v in pm.values()) else 'NÃO cumprido'}")
    json.dump({k: {str(N): v for N, v in d_.items()} for k, d_ in R.items()}, open('exp_etapaC.json', 'w'))
