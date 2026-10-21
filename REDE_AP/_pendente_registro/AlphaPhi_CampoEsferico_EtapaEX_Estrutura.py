# EXPERIMENTO: ETAPA EX — a estrutura de êxito (critérios aprovados pelo pesquisador em 21/10/2026, antes da execução)
# Base: Ferramentas de Memória v1 (canônico provisório: campo afinado, dinâmica contínua, previsor), importadas sem alteração.
# Memórias: 12 com forma (como nas Etapas K e PF), níveis 0,0 (principal) e 0,4; pistas 40% (principal) e 25%; sementes 130–139.
# Versões: SP sem previsor · T previsor com a memória inteira · EX-G estrutura de êxito (graus 0–4: 25 de 225 coeficientes)
#   · EX-H holográfica (25 números por memória, projeção gaussiana fixa, semente 777) · EX-acaso (25 coeficientes de graus 0–4 com a
#   mesma energia, forma sorteada) · V0 binária (descritiva). Nas EX, a ressonância é o cosseno entre a parcela do estado atual e a
#   parcela guardada; pesos softmax(β = 8); o campo puxa para a forma reconstruída da parcela (rms 1), força 0,5. Nunca vê o resto.
# Medidas: conclusão (principal), forma grosseira (0–4) e fina (5–14), identificação.
# EX0: T > SP (sanidade). EX1: EX-G > SP. EX2: (EX-G − SP)/(T − SP) >= 0,5 (médias). EX3: EX-G > SP na forma fina. EX4: EX-H > SP.
# EX5: EX-G > EX-acaso. (conclusão, salvo EX3; nível 0,0; pista 40%; pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
import os
H = os.path.dirname(os.path.abspath(__file__))  # dados: rodar antes ../CampoEsferico_Etapas_P_A_B_H_RP/AlphaPhi_EcoBIP_B0_gera_dobras.py e copiar os .npy para ../CampoEsferico_Etapas_S_K_PF/
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
FM = load('FM', os.path.join(H, '..', 'AlphaPhi_FerramentasMemoria_v1.py'))
K = load('K', os.path.join(H, '..', 'CampoEsferico_Etapas_S_K_PF', 'AlphaPhi_CampoEsferico_EtapaK2_Afinada.py'))   # só para gerar as memórias com forma (mesma função das Etapas K e PF)
G = FM.G; N, P = 610, 12; PTS = G.pontos_encaixados(N); Y, GR = G.harmonicos(PTS, 14); YP = np.linalg.pinv(Y)
BAIXO = GR <= 4; YL, PL = Y[:, BAIXO], YP[BAIXO]; NC = int(BAIXO.sum())
R = np.random.default_rng(777).normal(size=(NC, N)) / np.sqrt(N)
SEM = list(range(130, 140)); VERS = ['V0', 'SP', 'T', 'EX-G', 'EX-H', 'EX-acaso']
def rms(v): return v / (np.sqrt((v ** 2).mean(-1, keepdims=True)) + 1e-12)
def cos(A, B): return (A @ B.T) / (np.linalg.norm(A, axis=1, keepdims=True) * np.linalg.norm(B, axis=1)[None, :] + 1e-12)
def guia(tipo, X, rng_acaso):
    if tipo == 'EX-G':
        C = X @ PL.T; E = rms(C @ YL.T); return lambda s: (cos(s @ PL.T, C), E)
    if tipo == 'EX-H':
        Z = X @ R.T; E = rms(Z @ R); return lambda s: (cos(s @ R.T, Z), E)
    if tipo == 'EX-acaso':
        C0 = X @ PL.T; C = rng_acaso.normal(size=C0.shape); C *= np.linalg.norm(C0, axis=1, keepdims=True) / np.linalg.norm(C, axis=1, keepdims=True)
        E = rms(C @ YL.T); return lambda s: (cos(s @ PL.T, C), E)
def recuperar_guiado(J, S0, rng, g, guardar=False):
    # mesma dinâmica de FM.recuperar, com o previsor trocado pela guia da parcela
    S = S0.copy(); hesc = np.abs(S0 @ J).mean(1, keepdims=True); hist = []
    for v in range(FM.MAXV):
        m, E = g(np.sign(S)); w = np.exp(FM.BETA_PREVISOR * (m - m.max(1, keepdims=True))); w /= w.sum(1, keepdims=True)
        extra = FM.FORCA_PREVISOR * (w @ E); mud = 0.
        for i in rng.permutation(N):
            novo = np.tanh(FM.GANHO * (S @ J[:, i] / hesc[:, 0] + extra[:, i])); mud = max(mud, float(np.abs(novo - S[:, i]).max())); S[:, i] = novo
        if guardar: hist.append(S.copy())
        if mud < FM.TOL: break
    return (S, hist) if guardar else S
def forma(S, X):
    a, b = S @ YP.T, X @ YP.T; out = []
    for L in range(15):
        m = GR == L; out.append((a[:, m] * b[:, m]).sum(1) / (np.sqrt((a[:, m] ** 2).sum(1) * (b[:, m] ** 2).sum(1)) + 1e-12))
    return np.array(out).T
def medir(S, X):
    sg = np.sign(S); sg[sg == 0] = 1; f = forma(S, X)
    return {'conc': float((sg * X).mean()), 'gros': float(f[:, :5].mean()), 'fino': float(f[:, 5:].mean()), 'ident': float(np.mean(np.argmax(sg @ X.T, 1) == np.arange(P)))}
def rodar(X, S0, s, ni, fl, guardar=False):
    J = FM.ligacoes(PTS, X); rd = lambda: np.random.default_rng([2_200_000 + s, ni, fl]); out = {}; hist = {}
    out['V0'] = FM.recuperar_binaria(J, S0, rd()); hist['V0'] = None
    for v, kw in (('SP', dict(usar_previsor=False)), ('T', dict(usar_previsor=True))):
        r = FM.recuperar(J, X, S0, rd(), guardar=guardar, **kw); out[v], hist[v] = (r if guardar else (r, None))
    for v in ('EX-G', 'EX-H', 'EX-acaso'):
        r = recuperar_guiado(J, S0, rd(), guia(v, X, np.random.default_rng([2_250_000 + s, ni])), guardar); out[v], hist[v] = (r if guardar else (r, None))
    return out, hist
def tarefa(arg):
    c, flip = arg; T = K.calibrar(c); ni = int(round(c * 10)); fl = int(flip * 100); res = {v: {k: [] for k in ('conc', 'gros', 'fino', 'ident')} for v in VERS}
    for s in SEM:
        X = K.com_forma(T, np.random.default_rng([2_100_000 + s, ni])); r2 = np.random.default_rng([2_150_000 + s, ni, fl]); S0 = X.copy(); k = int(round(flip * N))
        for p in range(P): S0[p, r2.choice(N, k, replace=False)] *= -1
        out, _ = rodar(X, S0, s, ni, fl)
        for v in VERS:
            for kk, vv in medir(out[v], X).items(): res[v][kk].append(vv)
    print(f'  nível {c} pista {flip} pronto', file=sys.stderr, flush=True); return (c, flip), res

if __name__ == '__main__':
    print(f'estrutura de êxito: {NC} de {Y.shape[1]} coeficientes ({NC / Y.shape[1]:.1%})')
    with Pool(4) as pool: out = dict(pool.map(tarefa, [(c, f) for c in (0.0, 0.4) for f in (.40, .25)]))
    for (c, f), R_ in out.items():
        print(f'\n##### nível {c} · pista {int(f * 100)}%')
        print(f"{'versão':<10}{'conclusão':>10}{'grosseiro':>11}{'fino':>8}{'identificação':>15}")
        for v in VERS: print(f"{v:<10}" + ''.join(f"{np.mean(R_[v][k]):>{w}.3f}" for k, w in (('conc', 10), ('gros', 11), ('fino', 8), ('ident', 15))))
    R_ = out[(0.0, .40)]; col = lambda v, k='conc': np.array(R_[v][k])
    def teste(nome, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{nome}: {d.mean():+.3f} (EP {se:.3f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    print('\n=== CRITÉRIOS (nível 0,0; pista 40%) ===')
    teste('EX0 T − SP', col('T') - col('SP')); teste('EX1 EX-G − SP', col('EX-G') - col('SP'))
    fr = (col('EX-G').mean() - col('SP').mean()) / (col('T').mean() - col('SP').mean()); print(f"EX2 fração do ganho recuperada pela estrutura de êxito: {fr:.3f} (>= 0,5) -> {'cumprido' if fr >= .5 else 'NÃO cumprido'}")
    teste('EX3 EX-G − SP (forma fina, graus 5–14)', col('EX-G', 'fino') - col('SP', 'fino'))
    teste('EX4 EX-H − SP', col('EX-H') - col('SP')); teste('EX5 EX-G − EX-acaso', col('EX-G') - col('EX-acaso'))
    print(f"(descritivo) EX-G − EX-H: {(col('EX-G') - col('EX-H')).mean():+.3f}")
    json.dump({f'{c}|{f}': v for (c, f), v in out.items()}, open(H + '/EtapaEX_resultados.json', 'w'))
