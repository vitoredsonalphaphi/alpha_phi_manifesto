# EXPERIMENTO: ETAPA IN — o processamento pelo interior da esfera (critérios aprovados pelo pesquisador em 23/10/2026, antes da execução)
# Base da Etapa LA (Grade v2; 12 memórias contínuas; regra delta local; neurônio N3; previsor). Sementes 230–239. Ruído 2,0 × o desvio
# (principal; se IN0 falhar, 3,0), 1,0 e 3,0 (descritivos).
# Descida: o estado da superfície é ajustado aos harmônicos até o grau 14 (pesos pela área) e levado à casca de 233 (r = 1/φ) pela extensão
# de Laplace (r^L). Na casca: memórias, ligações (regra delta, vizinhança da própria casca), ganho e previsor próprios; âncora α com os pesos
# da casca. Subida: o estado da casca é ajustado aos graus <= Lc (inverte r^L) e volta à superfície como "formas largas".
# Lc fixado ANTES de rodar pela regra de amostragem (L+1)² <= n/2: casca 233 → Lc = 9; casca 89 → Lc = 5.
# Versões: SUP; IN (formas largas do interior + detalhes da pista acima de Lc); IN-sp (sem pegada: só o interior); CO (controle: só as
# formas largas da pista, sem rede no interior); IN2 (descritivo: cascata 233 → 89 → 233 → superfície). Depois, recuperação na superfície.
# IN0: SUP altura < 0,95 no ruído principal. IN1: IN > SUP. IN2: IN > CO. IN3: IN > IN-sp. (pareado > 2 EP e >= 8/10). Descritivo: voltas.
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
LA = load('LA', SP + 'exp_etapaLA.py'); NA, FM, GV2 = LA.NA, LA.FM, LA.NA.GV2
SEM = list(range(230, 240)); RUIDOS = [2.0, 1.0, 3.0]; LMAX = 14; LAMB = 1e-3
class Casca:
    def __init__(self, n, r, lc):
        self.n, self.r, self.lc = n, r, lc; self.P = GV2.pontos_encaixados(n, raio=r); self.U = self.P / r
        w = GV2.pesos(self.P); self.W = w / w.sum(); CV, FU, WF = FM.campo(self.U); self.FU, self.WF = FU, WF
        self.VIZ = [np.where(CV[i] > 0.01)[0] for i in range(n)]
        Y, g = GV2.harmonicos(self.U, lc); self.Bsub = Y * (r ** g)[None, :]; self.gsub = g     # para a subida: valores na casca = Σ c r^L Y
    def delta(self, X):
        J = np.zeros((self.n, self.n))
        for i in range(self.n):
            A = X[:, self.VIZ[i]]; J[self.VIZ[i], i] = np.linalg.solve(A.T @ A + LAMB * np.eye(len(self.VIZ[i])), A.T @ X[:, i])
        J = J + self.WF * self.FU; np.fill_diagonal(J, 0); return J
SUPW = NA.W; Y2, G2 = NA.Y2, NA.GR                     # superfície: harmônicos até 14 e pesos pela área (os da Etapa NA)
def coef_sup(S):
    A = Y2 * np.sqrt(SUPW)[:, None]; return np.linalg.lstsq(A, (S * np.sqrt(SUPW)[None, :]).T, rcond=None)[0].T
def descer(S, casca):                                  # superfície (ou casca externa já em coeficientes) → valores na casca
    return coef_sup(S) @ (GV2.harmonicos(casca.U, LMAX)[0] * (casca.r ** GV2.harmonicos(casca.U, LMAX)[1])[None, :]).T
def subir_coef(Si, casca):                              # valores na casca → coeficientes de superfície até Lc
    A = casca.Bsub * np.sqrt(casca.W)[:, None]; return np.linalg.lstsq(A, (Si * np.sqrt(casca.W)[None, :]).T, rcond=None)[0].T
def largas_sup(c, lc):                                  # coeficientes (até Lc, ou todos) → formas largas (graus <= Lc) na superfície
    k = int((G2 <= lc).sum()); return c[:, :k] @ Y2[:, :k].T
def recuperar(J, X, S0, rng, g, W):
    """A recuperação da Etapa NA (N3, previsor ligado), com os pesos da casca na âncora; devolve o estado e as voltas."""
    N = X.shape[1]; S = S0.copy(); hesc = np.abs(S0 @ J).mean(1, keepdims=True)
    wm = lambda A: A @ W; ws = lambda A: np.sqrt(((A - wm(A)[:, None]) ** 2) @ W); mu_t = wm(X).mean(); sd_t = ws(X).mean()
    for v in range(FM.MAXV):
        extra = FM.FORCA_PREVISOR * FM.previsor(S, X); mud = 0.
        for i in rng.permutation(N):
            novo = np.tanh(g * (S @ J[:, i] / hesc[:, 0] + extra[:, i])); mud = max(mud, float(np.abs(novo - S[:, i]).max())); S[:, i] = novo
        m = wm(S); S = S - 0.5 * (m - mu_t)[:, None]; m = wm(S); sd = ws(S)
        S = np.clip(m[:, None] + (S - m[:, None]) * (1 + 0.5 * (sd_t / np.maximum(sd, 1e-9) - 1))[:, None], -1, 1)
        if mud < FM.TOL: break
    return S, v + 1
C233 = Casca(233, 1 / GV2.PHI, 9); C89 = Casca(89, GV2.PHI ** -2, 5)
VERS = ['SUP', 'IN', 'IN-sp', 'CO', 'IN2']
def tarefa(ruido):
    out = {v: {'alt': [], 'ident': [], 'sat': [], 'voltas': []} for v in VERS}; info = {'pista': [], 'alt_interior': [], 'mem_interior_vs_sup': []}
    for s in SEM:
        X = NA.memorias(s); S0 = NA.pista(X, ruido, s); JS = LA.ligacao_delta(X); gS = NA.ganho_ajustado(JS, X)
        rd = lambda k: np.random.default_rng([5_000_000 + s, int(ruido * 10), k])
        info['pista'].append(NA.medir(S0, X)['alt'])
        Xi = descer(X, C233); Ji = C233.delta(Xi); gi = NA.ganho_ajustado(Ji, Xi); Si, vi = recuperar(Ji, Xi, descer(S0, C233), rd(1), gi, C233.W)
        info['alt_interior'].append(float(np.mean([np.corrcoef(Si[k], Xi[k])[0, 1] for k in range(len(X))])))
        info['mem_interior_vs_sup'].append(float(np.mean([np.corrcoef(largas_sup(subir_coef(Xi, C233), 9)[k], X[k])[0, 1] for k in range(len(X))])))
        cS0 = coef_sup(S0); fino = S0 - largas_sup(cS0, 9); larg_i = largas_sup(subir_coef(Si, C233), 9)
        X89 = descer(X, C89); J89 = C89.delta(X89); g89 = NA.ganho_ajustado(J89, X89); S89, v89 = recuperar(J89, X89, descer(S0, C89), rd(2), g89, C89.W)
        c89 = subir_coef(S89, C89)
        # cascata: as formas largas (<= 5) do 89 substituem as da pista na casca 233; recuperação na 233; subida à superfície
        cS0_233 = coef_sup(S0); Ymid = GV2.harmonicos(C233.U, LMAX); B233 = Ymid[0] * (C233.r ** Ymid[1])[None, :]
        c_mix = cS0_233.copy(); c_mix[:, G2 <= 5] = c89; S2i, v2i = recuperar(Ji, Xi, c_mix @ B233.T, rd(3), gi, C233.W)
        inicio = {'SUP': S0, 'IN': larg_i + fino, 'IN-sp': larg_i, 'CO': largas_sup(cS0, 9),
                  'IN2': largas_sup(subir_coef(S2i, C233), 9) + fino}
        extra_v = {'SUP': 0, 'IN': vi, 'IN-sp': vi, 'CO': 0, 'IN2': v89 + v2i}
        for v in VERS:
            Sf, nv = recuperar(JS, X, np.clip(inicio[v], -1, 1), rd(10), gS, SUPW); m = NA.medir(Sf, X)
            for k in ('alt', 'ident', 'sat'): out[v][k].append(m[k])
            out[v]['voltas'].append(nv + extra_v[v])
    print(f'  ruído {ruido} pronto', file=sys.stderr, flush=True); return ruido, {'R': out, 'info': info}
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'conferir':
    X = NA.memorias(1); S0 = NA.pista(X, 1.0, 1); J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X)
    a = NA.recuperar(J, X, S0, np.random.default_rng(7), g, True, True); b, _ = recuperar(J, X, S0, np.random.default_rng(7), g, SUPW)
    print('recuperar (com pesos) reproduz NA.recuperar na superfície:', np.allclose(a, b))
    print('vizinhos por neurônio: casca 233', np.mean([len(v) for v in C233.VIZ]).round(1), '· casca 89', np.mean([len(v) for v in C89.VIZ]).round(1))
    Ya, ga = GV2.harmonicos(C233.U, 9); Yb, gb = GV2.harmonicos(C233.U, LMAX); print('ordem dos harmônicos consistente (grau <= 9 é o começo do grau <= 14):', np.allclose(Ya, Yb[:, :Ya.shape[1]]) and np.array_equal(gb[:len(ga)], ga) and np.array_equal(G2[:len(ga)], ga))
    c = coef_sup(X); print('subida exata das memórias (descer e subir até Lc = 9), correlação com as formas largas na superfície:',
                         round(float(np.mean([np.corrcoef(largas_sup(subir_coef(descer(X, C233), C233), 9)[k], largas_sup(c, 9)[k])[0, 1] for k in range(12)])), 4))
if __name__ == '__main__' and len(sys.argv) == 1:
    with Pool(3) as pool: res = dict(pool.map(tarefa, RUIDOS))
    for ruido in RUIDOS:
        R_, I = res[ruido]['R'], res[ruido]['info']
        print(f"\n##### ruído {ruido} × o desvio · altura da pista {np.mean(I['pista']):.3f} · na casca 233: altura recuperada {np.mean(I['alt_interior']):.3f}"
              f" · memória que sobe (formas largas) vs memória na superfície {np.mean(I['mem_interior_vs_sup']):.3f}")
        print(f"{'versão':<8}{'altura':>9}{'identif.':>10}{'saturação':>11}{'voltas':>9}")
        for v in VERS: print(f"{v:<8}" + ''.join(f"{np.mean(R_[v][k]):>{w}.3f}" for k, w in (('alt', 9), ('ident', 10), ('sat', 11))) + f"{np.mean(R_[v]['voltas']):>9.1f}")
    def teste(nome, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{nome}: {d.mean():+.3f} (EP {se:.3f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    a0 = np.mean(res[2.0]['R']['SUP']['alt']); pr = 2.0 if a0 < .95 else 3.0
    print(f"\n=== CRITÉRIOS === IN0 SUP altura no ruído 2,0: {a0:.3f} (< 0,95) -> {'cumprido; principal 2,0' if pr == 2.0 else 'NÃO cumprido; principal passa a 3,0'}")
    col = lambda v: np.array(res[pr]['R'][v]['alt'])
    teste(f'IN1 IN − SUP (ruído {pr})', col('IN') - col('SUP')); teste(f'IN2 IN − CO (ruído {pr})', col('IN') - col('CO')); teste(f'IN3 IN − IN-sp (ruído {pr})', col('IN') - col('IN-sp'))
    json.dump({str(k): v for k, v in res.items()}, open(SP + 'ecobip/exp_etapaIN.json', 'w'))
