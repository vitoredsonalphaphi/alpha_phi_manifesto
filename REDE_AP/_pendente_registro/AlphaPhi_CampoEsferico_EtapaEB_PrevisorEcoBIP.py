# EXPERIMENTO: ETAPA EB — o previsor com a estrutura do EcoBIP (critérios aprovados pelo pesquisador em 24/10/2026, antes da execução)
# Base da Etapa LA; sementes 260–269; ruído 4,0 (principal) e 3,0 (descritivo); 50 voltas fixas.
# EB: força de atração por neurônio β_i, começo decisivo β_i = 8; concordância c_i = 1 − |S_i − pv_i|/2, normalizada (min–máx) no campo;
#     alvo β_alvo,i = 8·φ^(3·c_rel,i − 3/2); transição β_i ← (1 − 1/φ)·β_alvo,i + (1/φ)·β_i (a mistura do EcoBIP).
# EB-R (descritivo): EB com a ordem de atualização pela espiral áurea da grade (índices), deslocada a cada volta pelo número de Fibonacci.
# CT: como EB, mas os alvos embaralhados entre os neurônios a cada volta (mesma média, espalhamento e memória 1/φ; sem relação com c_i).
# EB1: EB > SUP (identificação). EB2: EB >= SUP (altura; não negativo além de 2 EP). EB3: EB > CT (identificação). (pareado > 2 EP, >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
M = load('M', SP + 'exp_etapaSA_HZ.py'); NA, LA, FM = M.NA, M.LA, M.FM; N, P, PHI = M.N, M.P, M.PHI; B8 = FM.BETA_PREVISOR
FIB = [1, 1]
while len(FIB) < 60: FIB.append(FIB[-1] + FIB[-2])
def previsor_regiao(S, X, beta):
    m = np.sign(S) @ X.T / N; Z = beta[:, :, None] * m[:, None, :]; Z -= Z.max(2, keepdims=True); w = np.exp(Z); w /= w.sum(2, keepdims=True)
    return np.einsum('pik,ki->pi', w, X)
def recuperar(J, X, S0, rng, g, modo, nvoltas=50, adaptar=True):
    S = S0.copy(); hesc = np.abs(S0 @ J).mean(1, keepdims=True); mu_t = NA.wmean(X).mean(); sd_t = NA.wstd(X).mean()
    beta = np.full(S.shape, B8); hist = []
    for v in range(nvoltas):
        pv = previsor_regiao(S, X, beta) if modo != 'SUP' else M.previsor(S, X, B8)
        if modo != 'SUP' and adaptar:
            c = 1 - np.abs(S - pv) / 2; cr = (c - c.min(1, keepdims=True)) / (c.max(1, keepdims=True) - c.min(1, keepdims=True) + 1e-12)
            alvo = B8 * PHI ** (3 * cr - 1.5)
            if modo == 'CT': alvo = np.array([a[rng.permutation(N)] for a in alvo])
            beta_novo = (1 - 1 / PHI) * alvo + (1 / PHI) * beta
        extra = FM.FORCA_PREVISOR * pv
        ordem = np.roll(np.arange(N), FIB[v] % N) if modo == 'EB-R' else rng.permutation(N)
        for i in ordem:
            S[:, i] = np.tanh(g * (S @ J[:, i] / hesc[:, 0] + extra[:, i]))
        m = NA.wmean(S); S = S - 0.5 * (m - mu_t)[:, None]; m = NA.wmean(S); sd = NA.wstd(S)
        S = np.clip(m[:, None] + (S - m[:, None]) * (1 + 0.5 * (sd_t / np.maximum(sd, 1e-9) - 1))[:, None], -1, 1)
        if modo != 'SUP' and adaptar: beta = beta_novo
        hist.append((float(beta.mean()), float(beta.std())))
    return S, hist
VERS = ['SUP', 'EB', 'EB-R', 'CT']
def tarefa(ruido):
    out = {v: {'alt': [], 'ident': [], 'beta_fim': []} for v in VERS}
    for s in range(260, 270):
        X = NA.memorias(s); S0 = NA.pista(X, ruido, s); J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X)
        for v in VERS:
            Sf, h = recuperar(J, X, S0, np.random.default_rng([7_000_000 + s, int(ruido * 10)]), g, v)
            m = NA.medir(Sf, X); out[v]['alt'].append(m['alt']); out[v]['ident'].append(m['ident']); out[v]['beta_fim'].append(h[-1])
    print(f'  ruído {ruido} pronto', file=sys.stderr, flush=True); return ruido, out
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'conferir':
    X = NA.memorias(1); S0 = NA.pista(X, 1.0, 1)
    a = M.previsor(S0, X, B8); b = previsor_regiao(S0, X, np.full(S0.shape, B8)); print('previsor por região com β = 8 em todos reproduz o previsor atual:', np.allclose(a, b))
    J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X)
    s1, _ = recuperar(J, X, S0, np.random.default_rng(3), g, 'SUP', nvoltas=5); s2, _ = recuperar(J, X, S0, np.random.default_rng(3), g, 'EB', nvoltas=5, adaptar=False)
    print('EB sem adaptação reproduz SUP (5 voltas):', np.allclose(s1, s2))
    s3 = M.recuperar(J, X, S0, np.random.default_rng(3), g, nvoltas=5, parada=False); print('SUP reproduz a recuperação da Etapa SA:', np.allclose(s1, s3))
    print('faixa do alvo β:', round(B8 * PHI ** -1.5, 2), 'a', round(B8 * PHI ** 1.5, 2))
if __name__ == '__main__' and len(sys.argv) == 1:
    with Pool(2) as pool: res = dict(pool.map(tarefa, [4.0, 3.0]))
    for r in (4.0, 3.0):
        o = res[r]; print(f"\n##### ruído {r}\n{'versão':<6}{'altura':>9}{'identif.':>10}{'β médio no fim':>16}{'β desvio':>10}")
        for v in VERS: print(f"{v:<6}{np.mean(o[v]['alt']):>9.4f}{np.mean(o[v]['ident']):>10.4f}{np.mean([b[0] for b in o[v]['beta_fim']]):>16.2f}{np.mean([b[1] for b in o[v]['beta_fim']]):>10.2f}")
    o = res[4.0]; c = lambda v, k: np.array(o[v][k])
    print('\n=== CRITÉRIOS (ruído 4,0) ===')
    M.teste('EB1 identificação EB − SUP', c('EB', 'ident') - c('SUP', 'ident')); M.teste('EB2 altura EB − SUP, não negativo', c('EB', 'alt') - c('SUP', 'alt'), True)
    M.teste('EB3 identificação EB − CT', c('EB', 'ident') - c('CT', 'ident'))
    json.dump({str(k): v for k, v in res.items()}, open(SP + 'ecobip/exp_etapaEB.json', 'w'))
