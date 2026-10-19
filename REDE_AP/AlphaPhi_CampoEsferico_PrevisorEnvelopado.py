# Resultados: REDE_AP/AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md, seção 5
# EXPERIMENTO: PREVISOR ENVELOPADO (critérios aprovados pelo pesquisador em 19/10/2026, antes da execução)
# Previsão CONDICIONAL: estabilidade de uma memória conhecida ν, sabendo que as outras P−1 vêm do mesmo ambiente.
#   a_i = ξ_i h_i ;  média = [Σ_j C_ij + (P−1)·ξ_i Σ_j C_ij ξ_j Γ_ij]/N ;  variância = (P−1)/N² · [w_iᵀ Γ w_i − (Σ_j C_ij ξ_j Γ_ij)²], w_i = C_i ∘ ξ
#   fração estável prevista = média_i Φ(média/desvio), média sobre as memórias guardadas.
# Variantes: P-env (Γ = envelope: espectro de harmônicos esféricos até o grau 8, ajustado à correlação de 200 padrões separados,
#   dependente só da distância; diagonal = 1, o conteúdo fino local) · P-amostra (Γ bruta da amostra) · P-antigo (fórmula da Etapa T).
# Desenho da Etapa T (F1, F2, F3 × σ ∈ {0,20; 0,35; 0,60}), sementes novas 10–19.
# V1: P-env × estabilidade medida, ambiente com forma: Spearman >= 0,8 E erro médio <= 0,05.  V2: P-env × recuperação: Spearman >= 0,7.
# V3: ambiente sem forma, P-env: Spearman >= 0,8 E erro médio <= 0,05.  V4: erro médio de P-env < erro médio de P-amostra (com forma).
import numpy as np, importlib.util, os, json
from scipy.special import eval_legendre
from scipy.stats import norm, spearmanr
_d = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('t', os.path.join(_d, 'AlphaPhi_CampoEsferico_EtapaT_Previsor.py'))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t); e0, base = t.e0, t.base
N, LENV, SEM = 256, 8, list(range(10, 20))

def envelope(P, G):
    cosd = np.clip(P @ P.T, -1, 1); iu = np.triu_indices(N, 1)
    A = np.stack([eval_legendre(L, cosd[iu]) for L in range(LENV + 1)], 1); pL, *_ = np.linalg.lstsq(A, G[iu], rcond=None)
    Gm = sum(pL[L] * eval_legendre(L, cosd) for L in range(LENV + 1)); np.fill_diagonal(Gm, 1.0); return Gm
def condicional(C, G, X):
    P = len(X); S = C.sum(1); out = []
    for xi in X:
        W = C * xi[None, :]; lin = (W * G).sum(1); quad = ((W @ G) * W).sum(1)
        mu = (S + (P - 1) * xi * lin) / N; var = (P - 1) / N ** 2 * np.maximum(quad - lin ** 2, 1e-12)
        out.append(norm.cdf(mu / np.sqrt(var)).mean())
    return float(np.mean(out))

if __name__ == '__main__':
    PF, PL = base.fib_esfera(N), e0.latlong(); linhas = {'forma': [], 'sem': []}
    for s in SEM:
        P2 = base.aleat_esfera(N, np.random.default_rng(30_000 + s))
        for fn, P in (('F1', PF), ('F2', P2), ('F3', PL)):
            D = e0.dnorm(P)
            for sg in (.20, .35, .60):
                C = np.exp(-(D / sg) ** 2); np.fill_diagonal(C, 0)
                for amb, cargas in (('forma', [4, 8, 12, 16]), ('sem', [5, 13, 20, 28, 36])):
                    ger = (lambda n, r: e0.suave(P, r, n)) if amb == 'forma' else (lambda n, r: r.choice([-1., 1.], size=(n, N)))
                    A = ger(200, np.random.default_rng([300_000 + s, int(sg * 100)])); G = (A.T @ A) / len(A); Ge = envelope(P, G)
                    for Pc in cargas:
                        X = ger(Pc, np.random.default_rng([400_000 + s, int(sg * 100), Pc]))
                        est, rec = t.medido(C, X, s, Pc)
                        linhas[amb].append((fn, sg, Pc, s, condicional(C, Ge, X), condicional(C, G, X), t.previsto(C, G, Pc), est, rec))
        print(f'  semente {s} pronta', flush=True)
    res = {}
    for amb, L in linhas.items():
        chave = [(l[0], l[1], l[2]) for l in L]; grupos = sorted(set(chave)); arr = np.array([l[4:] for l in L])
        M = np.array([arr[[k == g for k in chave]].mean(0) for g in grupos])      # colunas: env, amostra, antigo, estável, recuperação
        r = {}
        for j, nome in enumerate(('P-env', 'P-amostra', 'P-antigo')):
            r[nome] = dict(rho_est=float(spearmanr(M[:, j], M[:, 3])[0]), mae=float(np.abs(M[:, j] - M[:, 3]).mean()), rho_rec=float(spearmanr(M[:, j], M[:, 4])[0]))
        res[amb] = r
        print(f'\nAmbiente {"COM forma" if amb == "forma" else "SEM forma"} ({len(grupos)} combinações, média de 10 sementes)')
        print(f"   {'variante':<11}{'Spearman × estável':>20}{'erro médio':>12}{'Spearman × recuperação':>25}")
        for nome, v in r.items(): print(f"   {nome:<11}{v['rho_est']:>20.3f}{v['mae']:>12.4f}{v['rho_rec']:>25.3f}")
        if amb == 'forma':
            print(f"   {'campo':<5}{'σ':>6}{'P':>4}{'P-env':>8}{'P-amostra':>11}{'P-antigo':>10}{'estável':>9}{'recup.':>8}")
            for g, m in zip(grupos, M): print(f'   {g[0]:<5}{g[1]:>6.2f}{g[2]:>4}' + ''.join(f'{x:>{w}.3f}' for x, w in zip(m, (8, 11, 10, 9, 8))))
    f, sm = res['forma']['P-env'], res['sem']['P-env']
    print('\n=== CRITÉRIOS ===')
    print(f"V1 P-env × estável (com forma): Spearman {f['rho_est']:.3f}, erro {f['mae']:.4f} -> {'cumprido' if f['rho_est'] >= .8 and f['mae'] <= .05 else 'NÃO cumprido'}")
    print(f"V2 P-env × recuperação (com forma): Spearman {f['rho_rec']:.3f} -> {'cumprido' if f['rho_rec'] >= .7 else 'NÃO cumprido'}")
    print(f"V3 P-env sem forma: Spearman {sm['rho_est']:.3f}, erro {sm['mae']:.4f} -> {'cumprido' if sm['rho_est'] >= .8 and sm['mae'] <= .05 else 'NÃO cumprido'}")
    print(f"V4 erro P-env {f['mae']:.4f} < erro P-amostra {res['forma']['P-amostra']['mae']:.4f} -> {'cumprido' if f['mae'] < res['forma']['P-amostra']['mae'] else 'NÃO cumprido'}")
    json.dump({'res': res, 'linhas': linhas}, open('previsor_envelopado.json', 'w'))
