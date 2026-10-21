# EXPERIMENTO: ETAPA PF — o batedor, a pré-função (critérios aprovados pelo pesquisador em 21/10/2026, antes da execução)
# A AlphaFi de 610 pontos (Grade Esférica v1; vizinhança σ = 0,091; fundo do campo) comparada só consigo mesma.
# Memórias: 12 com forma, como na Etapa K, níveis 0,0 (principal) e 0,4 (descritivo); pistas 40% (principal) e 25%; sementes 120–129.
# Versões: V0 binária · V1 contínua (s = tanh(3·h/h_esc)) · V2 contínua com resfriamento (ganho 0,5 → 3 em 30 voltas; descritiva)
#   V3 = V1 + batedor (versão tênue, ganho 0,8, 5 passos a partir da pista; direção e firmeza) · V4 = V1 + previsor (mistura das memórias
#   que ressoam: softmax(8·m), a cada volta) · V5 = V1 + as duas vias · V6 = V5 com o batedor lançado da pista de outra memória.
# Forças: batedor 0,5 e previsor 0,5 da escala do campo (h_esc = média de |h| na pista).
# Medidas: conclusão (sobreposição final do sinal com a memória certa; principal), forma por casca (graus 0–4 e 5–14), firmeza × acerto
# (área sob a curva), coerência das formas (descritiva), identificação (antiga).
# PF0: na V0, no fim da 1ª volta, |h| distingue certos de errados com área > 0,6. PF1: V1 > V0. PF2: V3 > V1. PF3: V5 > V3 e V5 > V4.
# PF4: V5 > V6. PF5: na V5, firmeza × acerto > 0,7. (conclusão; nível 0,0, pista 40%; pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
import os
H = os.path.dirname(os.path.abspath(__file__))  # importa a Etapa K desta pasta; dados: rodar antes AlphaPhi_EcoBIP_B0_gera_dobras.py e copiar os .npy para esta pasta
spec = importlib.util.spec_from_file_location('K', os.path.join(H, 'AlphaPhi_CampoEsferico_EtapaK_Correlacao.py')); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
G, Hh = K.G, K.Hh; N, P, Y, GR = K.N, K.P, K.Y, K.GR
SEM = list(range(120, 130)); GANHO = 3.0; GB = 0.8; PASSOS_B = 5; LB = LP = 0.5; BETA = 8.0; MAXV = 50; TOL = 1e-3
VERS = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
YP = np.linalg.pinv(Y)   # coeficientes por mínimos quadrados
def coefs(S): return S @ YP.T
def forma(S, X):
    a, b = coefs(S), coefs(X); out = []
    for L in range(15):
        m = GR == L; num = (a[:, m] * b[:, m]).sum(1); den = np.sqrt((a[:, m] ** 2).sum(1) * (b[:, m] ** 2).sum(1)) + 1e-12; out.append(num / den)
    return np.array(out).T   # P × 15
def coer(S):
    a = coefs(S); e = np.array([(a[:, GR == L] ** 2).sum(1) for L in range(15)]).T; p = e / e.sum(1, keepdims=True)
    return 1 - (-(p * np.log(p + 1e-12)).sum(1)) / np.log(15)
def auc(score, certo):
    score, certo = score.ravel(), certo.ravel(); n1 = certo.sum(); n0 = len(certo) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = np.argsort(np.argsort(score)) + 1; return (r[certo].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def batedor(J, S0, hesc):
    s = S0.copy(); U = []
    for _ in range(PASSOS_B):
        s = np.tanh(GB * (s @ J) / hesc); U.append(s / (np.abs(s).max(1, keepdims=True) + 1e-12))
    u = np.mean(U, 0); return np.sign(u), np.abs(u)
def continua(J, X, S0, rng, hesc, versao, B=None, grava=False):
    S = S0.copy(); hist = []
    for v in range(MAXV):
        g = (0.5 + 2.5 * min(1., v / 30)) if versao == 'V2' else GANHO
        extra = np.zeros_like(S)
        if versao in ('V3', 'V5', 'V6'): extra += LB * B
        if versao in ('V4', 'V5', 'V6'):
            m = np.sign(S) @ X.T / N; w = np.exp(BETA * (m - m.max(1, keepdims=True))); w /= w.sum(1, keepdims=True); extra += LP * (w @ X)
        mud = 0.
        for i in rng.permutation(N):
            novo = np.tanh(g * (S @ J[:, i] / hesc[:, 0] + extra[:, i])); mud = max(mud, float(np.abs(novo - S[:, i]).max())); S[:, i] = novo
        if grava: hist.append(S.copy())
        if mud < TOL and (versao != 'V2' or v >= 30): break
    return S, hist
def binaria(J, X, S0, rng, grava=False):
    S = S0.copy(); hist = []; h1 = None
    for v in range(MAXV):
        mud = 0
        for i in rng.permutation(N):
            h = S @ J[:, i]; novo = np.where(h > 0, 1., np.where(h < 0, -1., S[:, i])); mud += int((novo != S[:, i]).sum()); S[:, i] = novo
        if v == 0: h1 = S @ J; S1 = S.copy()
        if grava: hist.append(S.copy())
        if mud == 0 and v > 0: break
    return S, hist, h1, S1
def rodar(J, X, S0, s, ni, fl, grava=False):
    hesc = np.abs(S0 @ J).mean(1, keepdims=True); res = {}; hist = {}
    rng = lambda: np.random.default_rng([2_000_000 + s, ni, fl])
    S, hs, h1, S1 = binaria(J, X, S0, rng(), grava); res['V0'] = S; hist['V0'] = hs
    pf0 = auc(np.abs(h1), S1 == X)
    d, f = batedor(J, S0, hesc); Bm = d * f
    outra = np.roll(np.arange(P), -1); d6, f6 = batedor(J, S0[outra], hesc[outra]); B6 = d6 * f6
    for v in VERS[1:]:
        Bv = B6 if v == 'V6' else Bm
        res[v], hist[v] = continua(J, X, S0, rng(), hesc, v, Bv, grava)
    return res, hist, pf0, Bm
def medir(res, X):
    out = {}
    for v, S in res.items():
        sg = np.sign(S); sg[sg == 0] = 1; fm = forma(S, X)
        out[v] = {'conc': float((sg * X).mean()), 'ident': float(np.mean(np.argmax(sg @ X.T, 1) == np.arange(P))),
                  'gros': float(fm[:, :5].mean()), 'fino': float(fm[:, 5:].mean()), 'coer': float(coer(S).mean()),
                  'firm': float(np.nanmean([auc(np.abs(S[p]), sg[p] == X[p]) for p in range(P)])) if v != 'V0' else np.nan}
    return out
def tarefa(arg):
    c, flip = arg; T = K.calibrar(c); ni = int(round(c * 10)); fl = int(flip * 100); R = {v: {k: [] for k in ('conc', 'ident', 'gros', 'fino', 'coer', 'firm')} for v in VERS}; PF0 = []
    for s in SEM:
        X = K.com_forma(T, np.random.default_rng([1_900_000 + s, ni])); J = K.redes(X)['A']
        r2 = np.random.default_rng([1_950_000 + s, ni, fl]); S0 = X.copy(); k = int(round(flip * N))
        for p in range(P): S0[p, r2.choice(N, k, replace=False)] *= -1
        res, _, pf0, _ = rodar(J, X, S0, s, ni, fl); PF0.append(pf0); m = medir(res, X)
        for v in VERS:
            for kk in m[v]: R[v][kk].append(m[v][kk])
    print(f'  nível {c} pista {flip} pronto', file=sys.stderr, flush=True); return (c, flip), {'R': R, 'PF0': PF0}

if __name__ == '__main__':
    with Pool(4) as pool: out = dict(pool.map(tarefa, [(c, f) for c in (0.0, 0.4) for f in (.40, .25)]))
    for (c, f), d in out.items():
        R = d['R']; print(f'\n##### nível {c} · pista {int(f*100)}%   (PF0, área |h| na V0: {np.mean(d["PF0"]):.3f})')
        print(f"{'versão':<7}{'conclusão':>10}{'grosseiro':>10}{'fino':>8}{'firmeza×acerto':>16}{'coerência':>11}{'identificação':>15}")
        for v in VERS: print(f"{v:<7}" + ''.join(f"{np.nanmean(R[v][k]):>{w}.3f}" for k, w in (('conc', 10), ('gros', 10), ('fino', 8), ('firm', 16), ('coer', 11), ('ident', 15))))
    R = out[(0.0, .40)]['R']; col = lambda v: np.array(R[v]['conc'])
    def teste(nome, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{nome}: {d.mean():+.3f} (EP {se:.3f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}"); return ok
    print('\n=== CRITÉRIOS (conclusão; nível 0,0; pista 40%) ===')
    pf0 = np.mean(out[(0.0, .40)]['PF0']); print(f"PF0 área |h| distingue certos de errados (V0, 1ª volta): {pf0:.3f} -> {'cumprido' if pf0 > .6 else 'NÃO cumprido'}")
    teste('PF1 V1 − V0', col('V1') - col('V0')); teste('PF2 V3 − V1', col('V3') - col('V1'))
    a = teste('PF3a V5 − V3', col('V5') - col('V3')); b = teste('PF3b V5 − V4', col('V5') - col('V4')); print(f"PF3 -> {'cumprido' if a and b else 'NÃO cumprido'}")
    teste('PF4 V5 − V6', col('V5') - col('V6'))
    pf5 = np.nanmean(R['V5']['firm']); print(f"PF5 firmeza × acerto na V5: {pf5:.3f} -> {'cumprido' if pf5 > .7 else 'NÃO cumprido'}")
    json.dump({f'{c}|{f}': v for (c, f), v in out.items()}, open(H + '/EtapaPF_resultados.json', 'w'))
