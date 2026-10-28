# EXPERIMENTO: ETAPA VL5 — o equilíbrio dos atratores e a confiança que mede a dúvida (critérios aprovados pelo pesquisador em 22/10/2026)
# Base da VL4; sementes 220–229; +10 dB (principal) e 0 dB; trechos guardados e novos.
# Parte A (descritiva): assinatura do ruído (pista só de ruído → memória de maior ressonância), tamanho de cada memória, matriz de confusão.
# Parte B: ressonância equilibrada z_k = (m_k − base_k) / desvio_k, com base e desvio medidos só pelas memórias com 1000 estados ±1 aleatórios;
#   no previsor, z é reescalado pela média dos desvios (mantém a escala de β = 8; não muda a ordem); aplicada no previsor e na decisão.
# Versões: DC (VL4), EQ, DP (pista, ressonância comum), EQ-P (pista, equilibrada), DS2 (EQ, decide só se pista e fim concordam; se não,
#   ganho 1), teto. VL5-1: identificação no fim EQ > DC. VL5-2: ganho EQ > DC. VL5-3: EQ >= EQ-P (não negativo além de 2 EP).
#   VL5-4a: trechos novos, DS2 > DC. VL5-4b: guardados, DS2 > 0 dB. (+10 dB; pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V4 = load('V4', SP + 'exp_etapaVL4.py'); V3, V2, LA, NA = V4.V3, V4.V2, V4.LA, V4.NA; FM = NA.FM
N = 610; SEM = list(range(220, 230)); resson = V4.resson
def base(X):
    R = np.sign(np.random.default_rng(4_000_000).normal(size=(1000, N))); m = R @ X.T / N; return m.mean(0), m.std(0)
def zres(S, X, b): return (resson(S, X) - b[0]) / b[1]
def previsor_eq(S, X, b):
    z = zres(S, X, b) * b[1].mean(); w = np.exp(FM.BETA_PREVISOR * (z - z.max(1, keepdims=True))); w /= w.sum(1, keepdims=True); return w @ X
def recuperar_eq(J, X, S0, rng, g, b):
    """Igual a NA.recuperar (âncora ligada, previsor ligado), com o previsor pela ressonância equilibrada."""
    S = S0.copy(); hesc = np.abs(S0 @ J).mean(1, keepdims=True); mu_t = NA.wmean(X).mean(); sd_t = NA.wstd(X).mean()
    for v in range(FM.MAXV):
        extra = FM.FORCA_PREVISOR * previsor_eq(S, X, b); mud = 0.
        for i in rng.permutation(N):
            novo = np.tanh(g * (S @ J[:, i] / hesc[:, 0] + extra[:, i])); mud = max(mud, float(np.abs(novo - S[:, i]).max())); S[:, i] = novo
        m = NA.wmean(S); S = S - 0.5 * (m - mu_t)[:, None]; m = NA.wmean(S); sd = NA.wstd(S)
        S = np.clip(m[:, None] + (S - m[:, None]) * (1 + 0.5 * (sd_t / np.maximum(sd, 1e-9) - 1))[:, None], -1, 1)
        if mud < FM.TOL: break
    return S
VERS = ['DC', 'EQ', 'DP', 'EQ-P', 'DS2', 'teto']
def rodar(arg):
    db, quais = arg; T = V3.Tarefa(); P = len(T.esc); X = T.X; J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X); b = base(X)
    res = {v: [] for v in VERS}; res.update({'id_' + v: [] for v in ('DC', 'EQ', 'DP', 'EQ-P')}); res['decide_DS2'] = []; conf = {'DC': [], 'EQ': [], 'DP': [], 'EQ-P': []}
    for s in SEM:
        idx = T.esc if quais == 'guardados' else [min(i + 1, len(T.segs) - 1) for i in T.esc]
        limpos = [T.segs[i] for i in idx]; ruid = [V2.ruido(c, db, np.random.default_rng([s, db + 50, k])) for k, c in enumerate(limpos)]
        S0 = np.array([T.pista(r) for r in ruid])
        Sd = NA.recuperar(J, X, S0, np.random.default_rng([4_100_000 + s, db + 50]), g, True, True)
        Se = recuperar_eq(J, X, S0, np.random.default_rng([4_100_000 + s, db + 50]), g, b)
        k = {'DC': resson(Sd, X).argmax(1), 'EQ': zres(Se, X, b).argmax(1), 'DP': resson(S0, X).argmax(1), 'EQ-P': zres(S0, X, b).argmax(1)}
        ganho = lambda y, j: V2.snr(y, limpos[j]) - V2.snr(ruid[j], limpos[j])
        for v in k:
            res['id_' + v].append(float(np.mean(k[v] == np.arange(P)))); conf[v].append(k[v].tolist())
            res[v].append(float(np.mean([ganho(T.decod(X[k[v][j]], ruid[j]), j) for j in range(P)])))
        dec = k['EQ-P'] == k['EQ']; res['decide_DS2'].append(float(dec.mean()))
        res['DS2'].append(float(np.mean([ganho(T.decod(X[k['EQ'][j]], ruid[j]), j) if dec[j] else 0.0 for j in range(P)])))
        res['teto'].append(float(np.mean([ganho(T.oraculo(limpos[j], ruid[j]), j) for j in range(P)])))
    print(f'  {db:+d} dB {quais} pronto', file=sys.stderr, flush=True)
    return arg, {'R': res, 'conf': conf}
def parteA():
    T = V3.Tarefa(); X = T.X; P = len(T.esc); b = base(X); cel = V3.E.ravel()
    print('=== PARTE A (descritiva) ===')
    print('tamanho de cada memória, |valor| médio nas células:', np.round(np.abs(X[:, cel]).mean(1), 3).tolist())
    print('fração de células com |valor| > 0,5:', np.round((np.abs(X[:, cel]) > .5).mean(1), 3).tolist())
    print('base (ressonância com estados aleatórios): média', np.round(b[0], 4).tolist(), '· desvio', np.round(b[1], 4).tolist())
    for db in (10, 0):
        cont = np.zeros(P, int); contz = np.zeros(P, int)
        for s in range(50):
            for k, i in enumerate(T.esc):
                c = T.segs[i]; so_ruido = V2.ruido(c, db, np.random.default_rng([9_000 + s, db + 50, k])) - c
                p = T.pista(so_ruido)[None, :]; cont[resson(p, X).argmax()] += 1; contz[zres(p, X, b).argmax()] += 1
        print(f'pista só de ruído (nível de {db:+d} dB de cada trecho, 50 × 12): memória escolhida — comum {cont.tolist()} · equilibrada {contz.tolist()}')
if __name__ == '__main__':
    parteA()
    with Pool(4) as pool: out = dict(pool.map(rodar, [(10, 'guardados'), (0, 'guardados'), (10, 'novos'), (0, 'novos')]))
    for (db, q), o in out.items():
        r = o['R']; print(f"\n##### {db:+d} dB · trechos {q} · identificação: " + ', '.join(f"{v} {np.mean(r['id_' + v]):.3f}" for v in ('DP', 'EQ-P', 'DC', 'EQ')) + f" · DS2 decide em {np.mean(r['decide_DS2']):.3f}")
        for v in VERS: print(f"   {v:<5}{np.mean(r[v]):+7.2f} dB (EP {np.std(r[v], ddof=1) / np.sqrt(10):.2f}) · fração do teto {np.mean(r[v]) / np.mean(r['teto']):+.2f}")
        if q == 'guardados': print('   confusão (semente 220): ' + ' · '.join(f"{v} {o['conf'][v][0]}" for v in o['conf']))
    col = lambda db, q, v: np.array(out[(db, q)]['R'][v])
    def teste(nome, d, nao_neg=False):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum()
        ok = (d.mean() > -2 * se) if nao_neg else (d.mean() > 2 * se and nf >= 8 and d.mean() > 0)
        print(f"{nome}: {d.mean():+.3f} (EP {se:.3f}, {nf}/10 a favor) -> {'cumprido' if ok else 'NÃO cumprido'}")
    print('\n=== CRITÉRIOS (+10 dB) ===')
    teste('VL5-1 identificação EQ − DC', col(10, 'guardados', 'id_EQ') - col(10, 'guardados', 'id_DC'))
    teste('VL5-2 ganho EQ − DC (dB)', col(10, 'guardados', 'EQ') - col(10, 'guardados', 'DC'))
    teste('VL5-3 ganho EQ − EQ-P (dB), não negativo', col(10, 'guardados', 'EQ') - col(10, 'guardados', 'EQ-P'), True)
    teste('VL5-4a novos: ganho DS2 − DC (dB)', col(10, 'novos', 'DS2') - col(10, 'novos', 'DC'))
    teste('VL5-4b guardados: ganho DS2 (dB)', col(10, 'guardados', 'DS2'))
    json.dump({f'{db}|{q}': o for (db, q), o in out.items()}, open(SP + 'ecobip/exp_etapaVL5.json', 'w'))
