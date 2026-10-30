# EXPERIMENTO: ETAPA DN — a dinâmica que ordena sem dissolver (critérios aprovados pelo pesquisador em 30/10/2026, antes da execução)
# Tarefa da Etapa AU (calota 10%, ruído 1,0; 2,0 e regiões 5%/20% descritivos; região 0% para DN5). Sementes 290–299.
# Discordância local: d = pista − memória reconhecida (a de maior ressonância com a pista); D_i = média de d em i e na vizinhança da
# Grade v2 (pesos da vizinhança da memória); z_i = |D_i| / mediana(|D|); força do previsor w_i = clip(1 − (z_i − 2)/2, 0, 1), fixa por
# recuperação. Pegada da entrada: onde o previsor se retira, o neurônio recebe λ·(1 − w_i)·pista_i, com λ = 0,5 (a mesma força do
# previsor que ele substitui; fixado antes de rodar). Versões: SUP (Etapa AU); SEL (previsor × w); SEL+P (previsor × w + pegada);
# SEL-AC (como SEL+P, com w embaralhado entre os neurônios); V1 (SUP parada na volta 1); pista; consulta; filtro.
# DN1: preservação SEL+P > SUP. DN2: coerência SEL+P > pista. DN3: SEL+P coerência > pista E preservação > consulta E total > consulta.
# DN4: preservação SEL+P > SEL-AC. DN5 (região 0%): total SEL+P >= SUP (não negativo além de 2 EP). (pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
A = load('A', SP + 'exp_etapaAU.py'); M, NA, LA, FM = A.M, A.NA, A.LA, A.FM; N, P = A.N, A.P; LAMB = 0.5
def forcas(S0, X):
    k = (np.sign(S0) @ X.T / N).argmax(1); d = S0 - X[k]; D = A.filtro(d); z = np.abs(D) / np.median(np.abs(D), axis=1, keepdims=True)
    return np.clip(1 - (z - 2) / 2, 0, 1)
def recuperar(J, X, S0, rng, g, w=None, pegada=False, nvoltas=FM.MAXV):
    S = S0.copy(); hesc = np.abs(S0 @ J).mean(1, keepdims=True); mu_t = NA.wmean(X).mean(); sd_t = NA.wstd(X).mean()
    for v in range(nvoltas):
        pv = M.previsor(S, X, FM.BETA_PREVISOR); extra = FM.FORCA_PREVISOR * pv * (1 if w is None else w)
        if pegada: extra = extra + LAMB * (1 - w) * S0
        mud = 0.
        for i in rng.permutation(N):
            novo = np.tanh(g * (S @ J[:, i] / hesc[:, 0] + extra[:, i])); mud = max(mud, float(np.abs(novo - S[:, i]).max())); S[:, i] = novo
        m = NA.wmean(S); S = S - 0.5 * (m - mu_t)[:, None]; m = NA.wmean(S); sd = NA.wstd(S)
        S = np.clip(m[:, None] + (S - m[:, None]) * (1 + 0.5 * (sd_t / np.maximum(sd, 1e-9) - 1))[:, None], -1, 1)
        if mud < FM.TOL: break
    return S
VERS = ['pista', 'consulta', 'filtro', 'SUP', 'SEL', 'SEL+P', 'SEL-AC', 'V1']
def tarefa(arg):
    ruido, frac = arg; out = {v: {'coer': [], 'pres': [], 'total': []} for v in VERS}; info = {'frac_protegida': [], 'protegida_na_calota': []}
    for s in range(290, 300):
        X = NA.memorias(s); T, caps = A.alvo(X, s, frac) if frac > 0 else (X.copy(), np.zeros(X.shape, bool)); S0 = NA.pista(T, ruido, s)
        J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X); w = forcas(S0, X); kc = (np.sign(S0) @ X.T / N).argmax(1)
        wac = np.array([wi[np.random.default_rng([9_300_000 + s, j]).permutation(N)] for j, wi in enumerate(w)])
        info['frac_protegida'].append(float((w < .5).mean()))
        if frac > 0: info['protegida_na_calota'].append(float(((w < .5) & caps).sum() / max((w < .5).sum(), 1)))
        rng = lambda: np.random.default_rng([9_400_000 + s, int(ruido * 10), int(frac * 100)])
        saidas = {'pista': S0, 'consulta': X[kc], 'filtro': A.filtro(S0), 'SUP': recuperar(J, X, S0, rng(), g),
                  'SEL': recuperar(J, X, S0, rng(), g, w), 'SEL+P': recuperar(J, X, S0, rng(), g, w, True),
                  'SEL-AC': recuperar(J, X, S0, rng(), g, wac, True), 'V1': recuperar(J, X, S0, rng(), g, nvoltas=1)}
        for v, S in saidas.items():
            m = A.medir(S, T, caps)
            for kk in m: out[v][kk].append(m[kk])
    print(f'  ruído {ruido}, região {frac} pronto', file=sys.stderr, flush=True); return arg, {'R': out, 'info': info}
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'conferir':
    X = NA.memorias(1); S0 = NA.pista(X, 1.0, 1); J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X)
    a = M.recuperar(J, X, S0, np.random.default_rng(5), g); b = recuperar(J, X, S0, np.random.default_rng(5), g); print('recuperar sem seleção reproduz a recuperação da Etapa AU:', np.allclose(a, b))
    X = NA.memorias(290); T, caps = A.alvo(X, 290, .10); S0 = NA.pista(T, 1.0, 290); w = forcas(S0, X)
    print(f'fração com previsor < 0,5: {(w < .5).mean():.3f} · dessa fração, quanto está na calota: {((w < .5) & caps).sum() / (w < .5).sum():.3f} · fração da calota coberta: {((w < .5) & caps).sum() / caps.sum():.3f}')
if __name__ == '__main__' and len(sys.argv) == 1:
    with Pool(4) as pool: res = dict(pool.map(tarefa, [(1.0, .10), (1.0, 0.0), (2.0, .10), (1.0, .05), (1.0, .20)]))
    for arg, o in res.items():
        r, I = o['R'], o['info']; ex = f" · do que foi protegido, na calota: {np.mean(I['protegida_na_calota']):.3f}" if I['protegida_na_calota'] else ''
        print(f"\n##### ruído {arg[0]} · região {arg[1]:.0%} · fração protegida (previsor < 0,5) {np.mean(I['frac_protegida']):.3f}{ex}")
        print(f"{'versão':<10}{'coerência':>11}{'preservação':>13}{'total':>9}")
        for v in VERS: print(f"{v:<10}" + ''.join(f"{np.mean(r[v][k]):>{w}.4f}" for k, w in (('coer', 11), ('pres', 13), ('total', 9))))
    o = res[(1.0, .10)]['R']; c = lambda v, k: np.array(o[v][k]); o0 = res[(1.0, 0.0)]['R']
    print('\n=== CRITÉRIOS (ruído 1,0, região 10%) ===')
    M.teste('DN1 preservação SEL+P − SUP', c('SEL+P', 'pres') - c('SUP', 'pres')); M.teste('DN2 coerência SEL+P − pista', c('SEL+P', 'coer') - c('pista', 'coer'))
    M.teste('DN3a coerência SEL+P − pista', c('SEL+P', 'coer') - c('pista', 'coer')); M.teste('DN3b preservação SEL+P − consulta', c('SEL+P', 'pres') - c('consulta', 'pres'))
    M.teste('DN3c total SEL+P − consulta', c('SEL+P', 'total') - c('consulta', 'total')); M.teste('DN4 preservação SEL+P − SEL-AC', c('SEL+P', 'pres') - c('SEL-AC', 'pres'))
    M.teste('DN5 (região 0%) total SEL+P − SUP, não negativo', np.array(o0['SEL+P']['total']) - np.array(o0['SUP']['total']), True)
    json.dump({f'{k[0]}|{k[1]}': v for k, v in res.items()}, open(SP + 'ecobip/exp_etapaDN.json', 'w'))
