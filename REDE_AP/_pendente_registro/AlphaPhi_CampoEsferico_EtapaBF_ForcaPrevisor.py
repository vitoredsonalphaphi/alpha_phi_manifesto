# EXPERIMENTO: ETAPA BF — a força do previsor (critérios aprovados pelo pesquisador em 24/10/2026, antes da execução)
# Base da Etapa LA; sementes 270–279; ruídos 2,0, 3,0 e 4,0 (principal); 50 voltas fixas. β = 8/φ, 8 (atual), 8·φ (principal), 8·φ², 8·φ³, ∞.
# BF1: ruído 4,0, altura β=8φ > β=8. BF2: ruídos 2,0 e 3,0, altura β=8φ >= β=8 (não negativo além de 2 EP). BF3: ruído 4,0, identificação
# β=8φ >= β=8 (não negativo). (pareado > 2 EP e >= 8/10.) Descritivo: a curva, captura precoce, β = ∞ (decisão pura; β = 1e6 numérico).
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
M = load('M', SP + 'exp_etapaSA_HZ.py'); NA, LA, PHI, P = M.NA, M.LA, M.PHI, M.P
BETAS = {'8/φ': 8 / PHI, '8': 8.0, '8φ': 8 * PHI, '8φ²': 8 * PHI ** 2, '8φ³': 8 * PHI ** 3, '∞': 1e6}
def tarefa(ruido):
    out = {b: {'alt': [], 'ident': [], 'captura': []} for b in BETAS}
    for s in range(270, 280):
        X = NA.memorias(s); S0 = NA.pista(X, ruido, s); J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X); certo = np.arange(P)
        for nome, b in BETAS.items():
            Sf, S5 = M.recuperar(J, X, S0, np.random.default_rng([8_000_000 + s, int(ruido * 10)]), g, beta=lambda v, b=b: b, nvoltas=50, parada=False, volta_obs=5)
            m = NA.medir(Sf, X); e5, ef = M.escolha(S5, X), M.escolha(Sf, X)
            out[nome]['alt'].append(m['alt']); out[nome]['ident'].append(m['ident']); out[nome]['captura'].append(float(np.mean((e5 != certo) & (ef == e5))))
    print(f'  ruído {ruido} pronto', file=sys.stderr, flush=True); return ruido, out
if __name__ == '__main__':
    with Pool(3) as pool: res = dict(pool.map(tarefa, [4.0, 3.0, 2.0]))
    for r in (2.0, 3.0, 4.0):
        o = res[r]; print(f"\n##### ruído {r}\n{'β':<6}{'altura':>9}{'identif.':>10}{'captura precoce':>17}")
        for b in BETAS: print(f"{b:<6}" + ''.join(f"{np.mean(o[b][k]):>{w}.4f}" for k, w in (('alt', 9), ('ident', 10), ('captura', 17))))
    c = lambda r, b, k: np.array(res[r][b][k])
    print('\n=== CRITÉRIOS ===')
    M.teste('BF1 ruído 4,0 — altura 8φ − 8', c(4.0, '8φ', 'alt') - c(4.0, '8', 'alt'))
    M.teste('BF2 ruído 2,0 — altura 8φ − 8, não negativo', c(2.0, '8φ', 'alt') - c(2.0, '8', 'alt'), True)
    M.teste('BF2 ruído 3,0 — altura 8φ − 8, não negativo', c(3.0, '8φ', 'alt') - c(3.0, '8', 'alt'), True)
    M.teste('BF3 ruído 4,0 — identificação 8φ − 8, não negativo', c(4.0, '8φ', 'ident') - c(4.0, '8', 'ident'), True)
    json.dump({str(k): v for k, v in res.items()}, open(SP + 'ecobip/exp_etapaBF.json', 'w'), ensure_ascii=False)
