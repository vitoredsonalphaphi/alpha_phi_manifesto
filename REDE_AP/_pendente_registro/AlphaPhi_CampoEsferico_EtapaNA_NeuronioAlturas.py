# EXPERIMENTO: ETAPA NA — o neurônio que guarda alturas (critérios aprovados pelo pesquisador em 22/10/2026, antes da execução)
# Grade Esférica v2 + Ferramentas de Memória v1 (canônico). 12 memórias CONTÍNUAS com forma (mesma regra das Etapas K e PF, nível 0,0):
# x = tanh(f / desvio de f). Estrago contínuo: x + ruído gaussiano com desvio = 1,0 × o desvio da memória (principal) ou 0,5 (descritivo),
# limitado a [−1, 1]. Sementes 180–189. Previsor ligado em todas as versões (força 0,5, β = 8); sem previsor, só descritivo.
# N0: neurônio de hoje (ganho 3). N1: ganho ajustado antes de rodar, só pelas memórias guardadas (o ganho g, numa grade de 0,10 a 3,00
#     de 0,05 em 0,05, que minimiza o erro quadrático entre cada memória e o neurônio aplicado a ela: x ≈ tanh(g·(h/h_esc + 0,5·x))).
# N2: âncora α: a cada volta, o neurônio α mede a média e o desvio do campo (pesos pela área da Grade v2) e corrige metade do desvio em
#     relação aos valores médios das memórias (média e desvio). N3: N1 + N2.
# Medidas: altura recuperada (correlação do estado final com a memória; principal), saturação (|valor| > 0,95), erro de altura, identificação.
# NA0: N0 satura (> 50%); se não, o teste para. NA1: N1 > N0. NA2: N2 > N0. NA3: N3 com altura recuperada >= 0,8 e saturação a até 10 pontos
# percentuais da das memórias. NA4: N3 > N1. (ruído principal; pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
import os
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, '..') + '/'; SP = os.path.join(R, 'CampoEsferico_Etapas_S_K_PF') + '/'  # dados: rodar antes o B0 e copiar os .npy
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
FM = load('FM', R + 'AlphaPhi_FerramentasMemoria_v1.py'); GV2 = load('GV2', R + 'AlphaPhi_GradeEsferica_v2.py'); K = load('K', SP + 'AlphaPhi_CampoEsferico_EtapaK2_Afinada.py')
N, P = 610, 12; P2 = GV2.pontos_encaixados(N); Y2, GR = GV2.harmonicos(P2, 14); W = GV2.pesos(P2); W = W / W.sum()
SEM = list(range(180, 190)); RUIDOS = [1.0, 0.5]; GRADE_G = np.round(np.arange(0.10, 3.0001, 0.05), 2)
def memorias(s):
    rng = np.random.default_rng([2_600_000 + s]); zs = rng.normal(size=(P, len(K.ZC))) * np.sqrt(K.EL)
    F = zs @ Y2.T; return np.tanh(F / F.std(1, keepdims=True))
def pista(X, ruido, s):
    rng = np.random.default_rng([2_650_000 + s, int(ruido * 10)]); return np.clip(X + ruido * X.std(1, keepdims=True) * rng.normal(size=X.shape), -1, 1)
def wmean(S): return S @ W
def wstd(S): m = wmean(S); return np.sqrt(((S - m[:, None]) ** 2) @ W)
def ganho_ajustado(J, X):
    hesc = np.abs(X @ J).mean(1, keepdims=True); h = (X @ J) / hesc + FM.FORCA_PREVISOR * X
    err = [np.mean((X - np.tanh(g * h)) ** 2) for g in GRADE_G]; return float(GRADE_G[int(np.argmin(err))])
def recuperar(J, X, S0, rng, g, ancora, previsor=True):
    S = S0.copy(); hesc = np.abs(S0 @ J).mean(1, keepdims=True); mu_t = wmean(X).mean(); sd_t = wstd(X).mean()
    for v in range(FM.MAXV):
        extra = FM.FORCA_PREVISOR * FM.previsor(S, X) if previsor else np.zeros_like(S); mud = 0.
        for i in rng.permutation(N):
            novo = np.tanh(g * (S @ J[:, i] / hesc[:, 0] + extra[:, i])); mud = max(mud, float(np.abs(novo - S[:, i]).max())); S[:, i] = novo
        if ancora:
            m = wmean(S); S = S - 0.5 * (m - mu_t)[:, None]; m = wmean(S); sd = wstd(S)
            S = np.clip(m[:, None] + (S - m[:, None]) * (1 + 0.5 * (sd_t / np.maximum(sd, 1e-9) - 1))[:, None], -1, 1)
        if mud < FM.TOL: break
    return S
def medir(S, X):
    c = np.array([np.corrcoef(S[k], X[k])[0, 1] for k in range(P)]); idt = np.argmax(np.corrcoef(S, X)[:P, P:], 1) == np.arange(P)
    return {'alt': float(c.mean()), 'sat': float((np.abs(S) > .95).mean()), 'err': float(np.sqrt(((S - X) ** 2).mean())), 'ident': float(idt.mean())}
VERS = ['N0', 'N1', 'N2', 'N3', 'N0-sem previsor', 'N3-sem previsor']
def tarefa(ruido):
    out = {v: {k: [] for k in ('alt', 'sat', 'err', 'ident')} for v in VERS}; info = {'g': [], 'sat_mem': [], 'alt_pista': []}
    for s in SEM:
        X = memorias(s); J = FM.ligacoes(P2, X); g1 = ganho_ajustado(J, X); S0 = pista(X, ruido, s)
        info['g'].append(g1); info['sat_mem'].append(float((np.abs(X) > .95).mean())); info['alt_pista'].append(medir(S0, X)['alt'])
        rd = lambda: np.random.default_rng([2_700_000 + s, int(ruido * 10)])
        for v, g, an, pv in (('N0', FM.GANHO, False, True), ('N1', g1, False, True), ('N2', FM.GANHO, True, True), ('N3', g1, True, True),
                             ('N0-sem previsor', FM.GANHO, False, False), ('N3-sem previsor', g1, True, False)):
            m = medir(recuperar(J, X, S0, rd(), g, an, pv), X)
            for k in m: out[v][k].append(m[k])
    print(f'  ruído {ruido} pronto', file=sys.stderr, flush=True); return ruido, {'R': out, 'info': info}
if __name__ == '__main__':
    with Pool(2) as pool: res = dict(pool.map(tarefa, RUIDOS))
    for ruido in RUIDOS:
        R_, I = res[ruido]['R'], res[ruido]['info']
        print(f"\n##### ruído {ruido} × o desvio da memória · ganho ajustado g = {np.mean(I['g']):.2f} (de {min(I['g']):.2f} a {max(I['g']):.2f}) · "
              f"saturação das memórias {np.mean(I['sat_mem']):.3f} · altura na pista {np.mean(I['alt_pista']):.3f}")
        print(f"{'versão':<17}{'altura':>9}{'saturação':>11}{'erro':>8}{'identificação':>15}")
        for v in VERS: print(f"{v:<17}" + ''.join(f"{np.mean(R_[v][k]):>{w}.3f}" for k, w in (('alt', 9), ('sat', 11), ('err', 8), ('ident', 15))))
    R_ = res[1.0]['R']; I = res[1.0]['info']; col = lambda v, k='alt': np.array(R_[v][k])
    def teste(nome, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{nome}: {d.mean():+.3f} (EP {se:.3f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    print('\n=== CRITÉRIOS (ruído principal) ===')
    s0 = col('N0', 'sat').mean(); print(f"NA0 a N0 satura: {s0:.3f} (> 0,50) -> {'cumprido' if s0 > .5 else 'NÃO cumprido — o teste para aqui'}")
    teste('NA1 N1 − N0', col('N1') - col('N0')); teste('NA2 N2 − N0', col('N2') - col('N0'))
    a3, s3, sm = col('N3').mean(), col('N3', 'sat').mean(), np.mean(I['sat_mem'])
    print(f"NA3 N3: altura {a3:.3f} (>= 0,8) · saturação {s3:.3f} contra {sm:.3f} das memórias (até 0,10) -> {'cumprido' if a3 >= .8 and abs(s3 - sm) <= .10 else 'NÃO cumprido'}")
    teste('NA4 N3 − N1', col('N3') - col('N1'))
    json.dump({str(k): v for k, v in res.items()}, open(os.path.join(H, 'EtapaNA_resultados.json'), 'w'))
