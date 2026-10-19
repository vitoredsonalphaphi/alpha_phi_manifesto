# Resultados: REDE_AP/AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md
# EXPERIMENTO: ETAPA T — a teoria antes da simulação (critérios aprovados pelo pesquisador em 19/10/2026, antes da execução)
# T1: núcleo do campo vazio e^-(d/0,15)^2 − 0,5·e^-(d/0,45)^2 (d = ângulo/π). Previsão por Funk–Hecke: λ_L = 2π∫K(θ)P_L(cosθ)senθ dθ, L* = argmax (L=0..20).
#     Medida: 200 estados aleatórios × 5 sementes; grau dominante de cada atrator (ajuste por mínimos quadrados em harmônicos reais até L=14).
#     Critério T1: em F1, |L_dom − L*| <= 1 em >= 80% dos atratores.  F2, F3 informativos.  Descritivo: L* para σi/σe = φ, 2, 3.
# T2: campos F1/F2/F3 × σ ∈ {0,20; 0,35; 0,60} (C = e^-(d/σ)^2). Ambientes: com forma (cargas 4,8,12,16) e sem forma (5,13,20,28,36).
#     Γ estimado de 200 padrões separados. Previsão (aprox. gaussiana): fração estável na memória = média_i Φ(μ_i/σ_i),
#     μ_i = [Σ_j C_ij + (P−1) Σ_j C_ij Γ_ij²]/N ;  σ_i² = (P−1)/N² · Σ_jk C_ij C_ik (Γ_jk² − Γ_ij² Γ_ik²).
#     Medidas: fração estável real na memória guardada; recuperação real (pista 25%). 10 sementes.
#     T2a (por ambiente): Spearman(previsto, estável medido) >= 0,8 E erro médio absoluto <= 0,05.  T2b: Spearman(previsto, recuperação) >= 0,7.
import numpy as np, importlib.util, json
from scipy.special import eval_legendre, sph_harm_y
from scipy.integrate import quad
from scipy.stats import norm, spearmanr
spec = importlib.util.spec_from_file_location('e0', __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'AlphaPhi_CampoEsferico_Etapa0_Cartografia.py'))
e0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e0); base = e0.base
N, PHI = 256, (1 + 5 ** .5) / 2

def nucleo(se, si, k=.5): return lambda t: np.exp(-(t / np.pi / se) ** 2) - k * np.exp(-(t / np.pi / si) ** 2)
def funk_hecke(K, Lmax=20):
    return np.array([2 * np.pi * quad(lambda t: K(t) * eval_legendre(L, np.cos(t)) * np.sin(t), 0, np.pi, limit=200)[0] for L in range(Lmax + 1)])
def harmonicos(P, Lmax=14):
    th = np.arccos(np.clip(P[:, 2], -1, 1)); ph = np.arctan2(P[:, 1], P[:, 0]); cols, grau = [], []
    for L in range(Lmax + 1):
        for m in range(-L, L + 1):
            Y = sph_harm_y(L, abs(m), th, ph)
            cols.append(np.sqrt(2) * (Y.imag if m < 0 else Y.real) if m != 0 else Y.real); grau.append(L)
    return np.array(cols).T, np.array(grau)
def grau_dominante(S, Y, g):
    c, *_ = np.linalg.lstsq(Y, S.T, rcond=None); E = np.array([(c[g == L] ** 2).sum(0) for L in range(g.max() + 1)])
    return E.argmax(0)
def campo_vazio(C, rng):
    S = rng.choice([-1., 1.], size=(200, N))
    for v in range(base.MAXV):
        mud = 0
        for i in rng.permutation(N):
            h = S @ C[:, i]; novo = np.where(h > 0, 1, np.where(h < 0, -1, S[:, i])); mud += int((novo != S[:, i]).sum()); S[:, i] = novo
        if mud == 0: break
    return S
def previsto(C, G, P):
    G2 = G ** 2; np.fill_diagonal(G2, 1.0)
    mu = (C.sum(1) + (P - 1) * (C * G2).sum(1)) / N
    a = C * G2                                                   # C_ij Γ_ij²
    var = (P - 1) / N ** 2 * (np.einsum('ij,jk,ik->i', C, G2, C) - a.sum(1) ** 2)
    return float(norm.cdf(mu / np.sqrt(np.maximum(var, 1e-12))).mean())
def medido(C, X, seed, P):
    J = C * (X.T @ X) / N; np.fill_diagonal(J, 0)
    est = float(((X @ J) * X > 0).mean())
    m, _, _ = base.recupera(J, X, .25, np.random.default_rng([700_000 + seed, P, len(X)])); return est, float(m.mean())

if __name__ == '__main__':
    PF, PL = base.fib_esfera(N), e0.latlong()
    # ---------------- T1
    lam = funk_hecke(nucleo(.15, .45)); Ls = int(lam.argmax())
    print('T1 previsão (Funk–Hecke) λ_L, L=0..8: ' + ' '.join(f'{x:.3f}' for x in lam[:9]) + f'  ->  L* = {Ls}')
    for nome, r in (('φ', PHI), ('2', 2.0), ('3', 3.0)):
        l2 = funk_hecke(nucleo(.15, .15 * r)); print(f'   descritivo: σi/σe = {nome:<2} -> L* = {int(l2.argmax())}')
    t1 = {}
    for f in ('F1', 'F2', 'F3'):
        doms = []
        for s in range(5):
            P = PF if f == 'F1' else (base.aleat_esfera(N, np.random.default_rng(30_000 + s)) if f == 'F2' else PL)
            Y, g = harmonicos(P); S = campo_vazio(e0.dog(e0.dnorm(P)), np.random.default_rng(80_000 + s))
            doms.append(grau_dominante(S, Y, g))
        d = np.concatenate(doms); t1[f] = (float((np.abs(d - Ls) <= 1).mean()), np.bincount(d, minlength=15)[:10].tolist())
        print(f'   {f}: fração com |L_dom − L*| <= 1 = {t1[f][0]:.3f}   contagem de L_dom (0..9): {t1[f][1]}')
    print(f"T1 -> {'cumprido' if t1['F1'][0] >= .8 else 'NÃO cumprido'}\n")
    # ---------------- T2
    linhas = {'forma': [], 'sem': []}
    for s in range(10):
        P2 = base.aleat_esfera(N, np.random.default_rng(30_000 + s))
        for fn, P in (('F1', PF), ('F2', P2), ('F3', PL)):
            D = e0.dnorm(P)
            for sg in (.20, .35, .60):
                C = np.exp(-(D / sg) ** 2); np.fill_diagonal(C, 0)
                for amb, cargas in (('forma', [4, 8, 12, 16]), ('sem', [5, 13, 20, 28, 36])):
                    ger = (lambda n, r: e0.suave(P, r, n)) if amb == 'forma' else (lambda n, r: r.choice([-1., 1.], size=(n, N)))
                    A = ger(200, np.random.default_rng([300_000 + s, int(sg * 100)])); G = (A.T @ A) / len(A)
                    for Pc in cargas:
                        X = ger(Pc, np.random.default_rng([400_000 + s, int(sg * 100), Pc]))
                        pv = previsto(C, G, Pc); est, rec = medido(C, X, s, Pc)
                        linhas[amb].append((fn, sg, Pc, s, pv, est, rec))
        print(f'  T2 semente {s} pronta', flush=True)
    for amb, L in linhas.items():
        arr = np.array([[l[4], l[5], l[6]] for l in L]); chave = [(l[0], l[1], l[2]) for l in L]
        grupos = sorted(set(chave)); M = np.array([arr[[k == g for k in chave]].mean(0) for g in grupos])
        rho_a = spearmanr(M[:, 0], M[:, 1])[0]; mae = float(np.abs(M[:, 0] - M[:, 1]).mean()); rho_b = spearmanr(M[:, 0], M[:, 2])[0]
        print(f'\nT2 ambiente {"COM forma" if amb == "forma" else "SEM forma"} ({len(grupos)} combinações, média de 10 sementes)')
        print(f"   {'campo':<5}{'σ':>6}{'P':>4}{'previsto':>10}{'estável':>9}{'recuperação':>13}")
        for g, r in zip(grupos, M): print(f'   {g[0]:<5}{g[1]:>6.2f}{g[2]:>4}{r[0]:>10.3f}{r[1]:>9.3f}{r[2]:>13.3f}')
        print(f"   T2a Spearman {rho_a:.3f} (>= 0,8), erro médio {mae:.3f} (<= 0,05) -> {'cumprido' if rho_a >= .8 and mae <= .05 else 'NÃO cumprido'}")
        print(f"   T2b Spearman previsto × recuperação {rho_b:.3f} (>= 0,7) -> {'cumprido' if rho_b >= .7 else 'NÃO cumprido'}")
    json.dump({'lam': lam.tolist(), 'Ls': Ls, 't1': t1, 'T2': linhas}, open('exp_etapaT.json', 'w'))
