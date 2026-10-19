# Resultados: REDE_AP/AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md, seção 6
# EXPERIMENTO: ETAPA E — previsor no campo cultivado, ferramentas instaladas por etapas (critérios escritos na conversa em 19/10/2026, antes da execução)
# Campo: 233 pontos encaixados (ângulo áureo + altura frac(i·√2)); fundo = ligação do campo da etapa final (σe=0,15, σi=0,45).
# Ferramentas: 4, cada uma com 4 padrões com forma, comprimentos de correlação 0,3 / 0,5 / 0,7 / 0,9, instaladas uma por vez.
# J = C_viz(σ=0,35) ∘ Hebb(ferramentas instaladas) + β·fundo, β ∈ {0; 0,02; 0,05}.
# Previsor condicional heterogêneo: cada ferramenta com a sua correlação (200 padrões separados); fundo entra exatamente na média.
# Comparação: correlação única (mistura das 4 ferramentas). Sementes 20–29. 30 combinações (etapa de instalação × ferramenta presente × β).
# E1: Spearman(previsto, estável) >= 0,8 E erro médio <= 0,05.  E2: Spearman(previsto, recuperação) >= 0,7.  E3: erro heterogêneo < erro único.
import numpy as np, importlib.util, os, json
from scipy.stats import norm, spearmanr
R = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('c', os.path.join(R, 'AlphaPhi_CampoEsferico_EtapaC_Cultivo.py'))
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c); t = c.t; base = t.base
base.N = 233  # a recuperação do módulo base usa N global
N = 233; ELLS = [.3, .5, .7, .9]; NPAT = 4; BETAS = [0., .02, .05]; SEM = list(range(20, 30))
P = c.pontos(N); D = np.arccos(np.clip(P @ P.T, -1, 1)) / np.pi
CV = np.exp(-(D / .35) ** 2); np.fill_diagonal(CV, 0); FUNDO = c.ligacao(P, .15)

def suave(rng, n, ell):
    cc = rng.normal(size=(n, 400, 3)); cc /= np.linalg.norm(cc, axis=2, keepdims=True); s = rng.choice([-1, 1], size=(n, 400))
    d2 = ((P[None, None] - cc[:, :, None]) ** 2).sum(-1); f = (s[:, :, None] * np.exp(-d2 / ell ** 2)).sum(1); return np.where(f >= 0, 1., -1.)

def preve(xi, outros_G, beta):
    """outros_G: lista com a correlação de cada OUTRA memória presente (uma entrada por memória)."""
    W = CV * xi[None, :]; S = CV.sum(1); mu = S + beta * N * xi * (FUNDO @ xi); var = np.zeros(N)
    for G in outros_G:
        lin = (W * G).sum(1); mu = mu + xi * lin; var = var + np.maximum(((W @ G) * W).sum(1) - lin ** 2, 0)
    return float(norm.cdf((mu / N) / np.sqrt(np.maximum(var, 1e-12) / N ** 2)).mean())

if __name__ == '__main__':
    linhas = []
    for s in SEM:
        Gs = []
        for k, ell in enumerate(ELLS):
            A = suave(np.random.default_rng([500_000 + s, k]), 200, ell); Gs.append((A.T @ A) / 200)
        Gmix = np.mean(Gs, 0)
        X = [suave(np.random.default_rng([600_000 + s, k]), NPAT, ell) for k, ell in enumerate(ELLS)]
        for beta in BETAS:
            for etapa in range(1, 5):
                todos = np.vstack(X[:etapa]); dono = np.repeat(np.arange(etapa), NPAT)
                J = CV * (todos.T @ todos) / N + beta * FUNDO; np.fill_diagonal(J, 0)
                est_todos = ((todos @ J) * todos > 0).mean(1)
                m, _, _ = base.recupera(J, todos, .25, np.random.default_rng([700_000 + s, etapa, int(beta * 100)]))
                for k in range(etapa):
                    idx = np.flatnonzero(dono == k); ph, pu = [], []
                    for v in idx:
                        outros = [Gs[dono[u]] for u in range(len(todos)) if u != v]
                        ph.append(preve(todos[v], outros, beta)); pu.append(preve(todos[v], [Gmix] * len(outros), beta))
                    linhas.append((beta, etapa, k, s, float(np.mean(ph)), float(np.mean(pu)), float(est_todos[idx].mean()), float(m[idx].mean())))
        print(f'  semente {s} pronta', flush=True)
    chave = [(l[0], l[1], l[2]) for l in linhas]; grupos = sorted(set(chave)); arr = np.array([l[4:] for l in linhas])
    M = np.array([arr[[q == g for q in chave]].mean(0) for g in grupos])
    print(f"\n{'β':>5}{'etapa':>6}{'ferr.':>6}{'ℓ':>5}{'prev. própria':>15}{'prev. única':>13}{'estável':>9}{'recuperação':>13}")
    for g, r in zip(grupos, M): print(f'{g[0]:>5.2f}{g[1]:>6}{g[2]+1:>6}{ELLS[g[2]]:>5}{r[0]:>15.3f}{r[1]:>13.3f}{r[2]:>9.3f}{r[3]:>13.3f}')
    rh, mh, rr = spearmanr(M[:, 0], M[:, 2])[0], np.abs(M[:, 0] - M[:, 2]).mean(), spearmanr(M[:, 0], M[:, 3])[0]
    mu_ = np.abs(M[:, 1] - M[:, 2]).mean()
    print('\n=== CRITÉRIOS ===')
    print(f"E1 previsão própria × estável: Spearman {rh:.3f}, erro {mh:.4f} -> {'cumprido' if rh >= .8 and mh <= .05 else 'NÃO cumprido'}")
    print(f"E2 previsão própria × recuperação: Spearman {rr:.3f} -> {'cumprido' if rr >= .7 else 'NÃO cumprido'}")
    print(f"E3 erro própria {mh:.4f} < erro única {mu_:.4f} -> {'cumprido' if mh < mu_ else 'NÃO cumprido'}")
    print(f"   (informativo) correlação única × estável: Spearman {spearmanr(M[:, 1], M[:, 2])[0]:.3f}")
    json.dump({'grupos': grupos, 'M': M.tolist()}, open('exp_etapaE.json', 'w'))
