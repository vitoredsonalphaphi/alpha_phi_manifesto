# Resultados: REDE_AP/AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md, seção 7 · importa a versão canônica v1 (commit f796a3e)
# EXPERIMENTO: ETAPA F — ferramentas instaladas durante o crescimento do campo (critérios escritos na conversa em 19/10/2026, antes da execução)
# Pontos encaixados (Etapa C), N final 233, fundo β = 0,02 (ligação do campo da etapa final). Ferramentas da Etapa E (ℓ = 0,3/0,5/0,7/0,9; 4 padrões).
# Cada ferramenta entra numa etapa N_k e liga só os neurônios 0..N_k−1 (território). Cronogramas: A = 0,3@55, 0,5@89, 0,7@144, 0,9@233;
# B = invertido; C = todas em 233. Previsor condicional com máscara de território e correlação própria de cada ferramenta. Sementes 30–39.
# F1: Spearman(previsto, estável) >= 0,8 E erro médio <= 0,05 (24 combinações).  F2: Spearman(previsto, recuperação) >= 0,7.
# F3: estabilidade final das ferramentas 0,3 e 0,9 difere entre A e B (|dif| > 2 EP e mesmo sinal em >= 8/10), para as duas.
import numpy as np, importlib.util, os, json
from scipy.stats import norm, spearmanr
R = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('E', os.path.join(R, 'AlphaPhi_CampoEsferico_EtapaE_Ferramentas.py'))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E); base = E.base
N, CV, FUNDO, BETA, ELLS, NPAT = E.N, E.CV, E.FUNDO, .02, E.ELLS, E.NPAT
SEM = list(range(30, 40))
CRON = {'A': [(0, 55), (1, 89), (2, 144), (3, 233)], 'B': [(3, 55), (2, 89), (1, 144), (0, 233)], 'C': [(0, 233), (1, 233), (2, 233), (3, 233)]}

def montaJ(inst, X):
    J = BETA * FUNDO.copy()
    for k, Nk in inst:
        M = np.zeros((N, N)); M[:Nk, :Nk] = 1; J += M * CV * (X[k].T @ X[k]) / N
    np.fill_diagonal(J, 0); return J
def preve(k, Nk, v, inst, X, Gs):
    xi = np.zeros(N); xi[:Nk] = X[k][v][:Nk]; S = slice(0, Nk)
    mu = np.zeros(N); var = np.zeros(N)
    for l, Nl in inst:
        Ml = np.zeros((N, N)); Ml[:Nl, :Nl] = 1; W = Ml * CV * xi[None, :]
        lin = (W * Gs[l]).sum(1); quad = ((W @ Gs[l]) * W).sum(1); n_out = NPAT - (1 if l == k else 0)
        mu += n_out * xi * lin; var += n_out * np.maximum(quad - lin ** 2, 0)
        if l == k: mu += (Ml * CV * (xi[None, :] != 0)).sum(1) * (xi != 0)
    mu = mu / N + BETA * xi * (FUNDO @ xi); var = var / N ** 2
    return float(norm.cdf(mu[S] / np.sqrt(np.maximum(var[S], 1e-12))).mean())
def recupera_sub(J, Xs, rng):
    n = J.shape[0]; base.N = n; m, _, _ = base.recupera(J, Xs, .25, rng); base.N = 233; return m

if __name__ == '__main__':
    linhas = []
    for s in SEM:
        Gs = []
        for k, ell in enumerate(ELLS):
            A = E.suave(np.random.default_rng([510_000 + s, k]), 200, ell); Gs.append((A.T @ A) / 200)
        X = [E.suave(np.random.default_rng([610_000 + s, k]), NPAT, ell) for k, ell in enumerate(ELLS)]
        for nome, cron in CRON.items():
            pontos_aval = range(1, 5) if nome != 'C' else [4]
            for etapa in pontos_aval:
                inst = cron[:etapa]; J = montaJ(inst, X)
                for k, Nk in inst:
                    Xs = X[k][:, :Nk]; Jk = J[:Nk, :Nk]
                    est = float(((Xs @ Jk) * Xs > 0).mean())
                    pv = float(np.mean([preve(k, Nk, v, inst, X, Gs) for v in range(NPAT)]))
                    rec = float(recupera_sub(Jk, Xs, np.random.default_rng([710_000 + s, k, etapa, ord(nome)])).mean())
                    linhas.append((nome, etapa, k, Nk, s, pv, est, rec))
        print(f'  semente {s} pronta', flush=True)
    chave = [(l[0], l[1], l[2], l[3]) for l in linhas]; grupos = sorted(set(chave)); arr = np.array([l[5:] for l in linhas])
    M = np.array([arr[[q == g for q in chave]].mean(0) for g in grupos])
    print(f"\n{'cron.':>5}{'etapa':>6}{'ℓ':>5}{'território':>11}{'previsto':>10}{'estável':>9}{'recuperação':>13}")
    for g, r in zip(grupos, M): print(f'{g[0]:>5}{g[1]:>6}{ELLS[g[2]]:>5}{g[3]:>11}{r[0]:>10.3f}{r[1]:>9.3f}{r[2]:>13.3f}')
    rh, mh, rr = spearmanr(M[:, 0], M[:, 1])[0], float(np.abs(M[:, 0] - M[:, 1]).mean()), spearmanr(M[:, 0], M[:, 2])[0]
    print('\n=== CRITÉRIOS ===')
    print(f"F1 previsão × estável: Spearman {rh:.3f}, erro {mh:.4f} -> {'cumprido' if rh >= .8 and mh <= .05 else 'NÃO cumprido'}")
    print(f"F2 previsão × recuperação: Spearman {rr:.3f} -> {'cumprido' if rr >= .7 else 'NÃO cumprido'}")
    ok3 = True
    for k in (0, 3):
        a = np.array([l[6] for l in linhas if l[0] == 'A' and l[1] == 4 and l[2] == k]); b = np.array([l[6] for l in linhas if l[0] == 'B' and l[1] == 4 and l[2] == k])
        d = a - b; se = d.std(ddof=1) / np.sqrt(len(d)); n_mesmo = max((d > 0).sum(), (d < 0).sum()); ok = abs(d.mean()) > 2 * se and n_mesmo >= 8; ok3 &= ok
        print(f"F3 ferramenta ℓ={ELLS[k]}: estável final A {a.mean():.3f} | B {b.mean():.3f} | dif {d.mean():+.4f} (EP {se:.4f}, mesmo sinal {n_mesmo}/10) -> {'difere' if ok else 'não difere'}")
    print(f"F3 -> {'cumprido' if ok3 else 'NÃO cumprido'}")
    json.dump({'grupos': grupos, 'M': M.tolist()}, open('exp_etapaF.json', 'w'))
