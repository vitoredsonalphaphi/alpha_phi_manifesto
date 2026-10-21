# EXPERIMENTO: ETAPA EC — a estrutura de êxito em camadas (critérios aprovados pelo pesquisador em 21/10/2026, antes da execução)
# Mesmo desenho da Etapa EX (Ferramentas de Memória v1; 12 memórias com forma; níveis 0,0 e 0,4; pistas 40% e 25%); sementes 140–149.
# Versões: SP · T (memória inteira, previsor do canônico) · EX-G (graus 0–4 sempre) · EC-3 (0–4 nas voltas 1–10, 0–9 nas 11–20, 0–14 da
#   21 em diante) · EC-2 (0–4, depois 0–9: 100 de 225 números) · EC-inverso (10–14, depois 5–14, depois 0–14). Força 0,5, β = 8.
# Nota declarada: a camada 0–14 é a projeção da memória nos 225 harmônicos (a forma suave dela), não o padrão ±1 que o previsor T usa.
# EC0: T > SP. EC1: EC-3 > T. EC2: EC-3 > EC-inverso. EC3: (EC-2 − SP)/(T − SP) >= 0,8 (médias). EC4: EC-2 > EX-G na forma fina.
# (conclusão, salvo EC4; nível 0,0; pista 40%; pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
import os
H = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location('EX', os.path.join(H, 'AlphaPhi_CampoEsferico_EtapaEX_Estrutura.py')); EX = importlib.util.module_from_spec(_s); _s.loader.exec_module(EX)
FM, K, G, N, P, Y, GR, YP = EX.FM, EX.K, EX.G, EX.N, EX.P, EX.Y, EX.GR, EX.YP
SEM = list(range(140, 150)); VERS = ['SP', 'T', 'EX-G', 'EC-3', 'EC-2', 'EC-inverso']
def faixa(a, b): return (GR >= a) & (GR <= b)
CAL = {'EX-G': lambda v: faixa(0, 4),
       'EC-3': lambda v: faixa(0, 4) if v < 10 else (faixa(0, 9) if v < 20 else faixa(0, 14)),
       'EC-2': lambda v: faixa(0, 4) if v < 10 else faixa(0, 9),
       'EC-inverso': lambda v: faixa(10, 14) if v < 10 else (faixa(5, 14) if v < 20 else faixa(0, 14))}
def guia_camadas(X, cal):
    cache = {}
    def g(s, v):
        M = cal(v); k = M.tobytes()
        if k not in cache: C = X @ YP[M].T; cache[k] = (M, C, EX.rms(C @ Y[:, M].T))
        M, C, E = cache[k]; return EX.cos(s @ YP[M].T, C), E
    return g
def recuperar_camadas(J, S0, rng, g, guardar=False):
    S = S0.copy(); hesc = np.abs(S0 @ J).mean(1, keepdims=True); hist = []
    for v in range(FM.MAXV):
        m, E = g(np.sign(S), v); w = np.exp(FM.BETA_PREVISOR * (m - m.max(1, keepdims=True))); w /= w.sum(1, keepdims=True)
        extra = FM.FORCA_PREVISOR * (w @ E); mud = 0.
        for i in rng.permutation(N):
            novo = np.tanh(FM.GANHO * (S @ J[:, i] / hesc[:, 0] + extra[:, i])); mud = max(mud, float(np.abs(novo - S[:, i]).max())); S[:, i] = novo
        if guardar: hist.append(S.copy())
        if mud < FM.TOL and v >= 20: break      # não para antes de todas as camadas terem entrado
    return (S, hist) if guardar else S
def rodar(X, S0, s, ni, fl, guardar=False):
    J = FM.ligacoes(EX.PTS, X); rd = lambda: np.random.default_rng([2_300_000 + s, ni, fl]); out = {}; hist = {}
    for v, kw in (('SP', dict(usar_previsor=False)), ('T', dict(usar_previsor=True))):
        r = FM.recuperar(J, X, S0, rd(), guardar=guardar, **kw); out[v], hist[v] = (r if guardar else (r, None))
    for v in ('EX-G', 'EC-3', 'EC-2', 'EC-inverso'):
        r = recuperar_camadas(J, S0, rd(), guia_camadas(X, CAL[v]), guardar); out[v], hist[v] = (r if guardar else (r, None))
    return out, hist, J
def gerar(s, c, flip):
    T = K.calibrar(c); ni = int(round(c * 10)); fl = int(flip * 100)
    X = K.com_forma(T, np.random.default_rng([2_310_000 + s, ni])); r2 = np.random.default_rng([2_320_000 + s, ni, fl]); S0 = X.copy(); k = int(round(flip * N))
    for p in range(P): S0[p, r2.choice(N, k, replace=False)] *= -1
    return X, S0, ni, fl
def tarefa(arg):
    c, flip = arg; res = {v: {k: [] for k in ('conc', 'gros', 'fino', 'ident')} for v in VERS}
    for s in SEM:
        X, S0, ni, fl = gerar(s, c, flip); out, _, _ = rodar(X, S0, s, ni, fl)
        for v in VERS:
            for kk, vv in EX.medir(out[v], X).items(): res[v][kk].append(vv)
    print(f'  nível {c} pista {flip} pronto', file=sys.stderr, flush=True); return (c, flip), res

if __name__ == '__main__':
    with Pool(4) as pool: out = dict(pool.map(tarefa, [(c, f) for c in (0.0, 0.4) for f in (.40, .25)]))
    for (c, f), R_ in out.items():
        print(f'\n##### nível {c} · pista {int(f * 100)}%')
        print(f"{'versão':<12}{'conclusão':>10}{'grosseiro':>11}{'fino':>8}{'identificação':>15}")
        for v in VERS: print(f"{v:<12}" + ''.join(f"{np.mean(R_[v][k]):>{w}.3f}" for k, w in (('conc', 10), ('gros', 11), ('fino', 8), ('ident', 15))))
    R_ = out[(0.0, .40)]; col = lambda v, k='conc': np.array(R_[v][k])
    def teste(nome, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{nome}: {d.mean():+.3f} (EP {se:.3f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    print('\n=== CRITÉRIOS (nível 0,0; pista 40%) ===')
    teste('EC0 T − SP', col('T') - col('SP')); teste('EC1 EC-3 − T', col('EC-3') - col('T')); teste('EC2 EC-3 − EC-inverso', col('EC-3') - col('EC-inverso'))
    fr = (col('EC-2').mean() - col('SP').mean()) / (col('T').mean() - col('SP').mean()); print(f"EC3 fração do ganho com duas camadas (44%): {fr:.3f} (>= 0,8) -> {'cumprido' if fr >= .8 else 'NÃO cumprido'}")
    teste('EC4 EC-2 − EX-G (forma fina)', col('EC-2', 'fino') - col('EX-G', 'fino'))
    json.dump({f'{c}|{f}': v for (c, f), v in out.items()}, open(H + '/EtapaEC_resultados.json', 'w'))
