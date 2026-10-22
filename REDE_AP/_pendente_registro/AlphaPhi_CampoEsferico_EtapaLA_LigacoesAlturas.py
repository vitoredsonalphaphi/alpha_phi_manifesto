# EXPERIMENTO: ETAPA LA — as ligações para alturas e a leitura no além (critérios aprovados pelo pesquisador em 22/10/2026, antes da execução)
# Mesma base da Etapa NA (Grade v2; Ferramentas de Memória v1; 12 memórias contínuas; ruído 1,0 [principal] e 0,5 × o desvio); neurônio N3
# (ganho ajustado + âncora α); previsor ligado. Sementes 190–199.
# Regra delta local (Widrow–Hoff, citada de memória): cada neurônio i aprende os pesos com os vizinhos j em que a vizinhança da Grade v2 tem
# peso > 0,01, por mínimos quadrados com regularização λ = 0,001, para que X[:, vizinhos] · w ≈ X[:, i]; soma-se o fundo afinado do canônico.
# Leitura pelo campo: a mistura das memórias pesada pela ressonância (softmax(8·m), m = sinal(estado)·memórias/N) do estado final.
# Versões: H-estado, H-campo, D-estado, D-campo, pista-campo (controle: leitura pelo campo direto da pista), PI-local (descritivo).
# LA0: com a regra delta, uma memória intacta passada pelas ligações sai com correlação >= 0,98. LA1: D-estado > a própria pista.
# LA2: D-estado > H-estado. LA3: H-campo > H-estado. LA4: D-campo > pista-campo. (ruído principal; pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
import os
SP = os.path.dirname(os.path.abspath(__file__)) + '/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
NA = load('NA', SP + 'AlphaPhi_CampoEsferico_EtapaNA_NeuronioAlturas.py'); FM, P2, N, P = NA.FM, NA.P2, NA.N, NA.P
SEM = list(range(190, 200)); LAMB = 1e-3
CV, FU, WF = FM.campo(P2); VIZ = [np.where(CV[i] > 0.01)[0] for i in range(N)]
def ligacao_delta(X):
    J = np.zeros((N, N))
    for i in range(N):
        A = X[:, VIZ[i]]; w = np.linalg.solve(A.T @ A + LAMB * np.eye(len(VIZ[i])), A.T @ X[:, i]); J[VIZ[i], i] = w
    J = J + WF * FU; np.fill_diagonal(J, 0); return J
def ligacao_pi(X):
    Cm = X @ X.T / N; J = CV * (X.T @ np.linalg.solve(Cm, X)) / N + WF * FU; np.fill_diagonal(J, 0); return J
def campo(S, X):
    m = np.sign(S) @ X.T / N; w = np.exp(8 * (m - m.max(1, keepdims=True))); w /= w.sum(1, keepdims=True); return w @ X
def passagem(J, X): Z = X @ J; return float(np.mean([np.corrcoef(Z[k], X[k])[0, 1] for k in range(P)]))
VERS = ['H-estado', 'H-campo', 'D-estado', 'D-campo', 'pista-campo', 'PI-local']
def tarefa(ruido):
    out = {v: {k: [] for k in ('alt', 'sat', 'err', 'ident')} for v in VERS}; info = {'pista': [], 'pass_H': [], 'pass_D': [], 'pass_PI': [], 'g': []}
    for s in SEM:
        X = NA.memorias(s); S0 = NA.pista(X, ruido, s); JH = FM.ligacoes(P2, X); JD = ligacao_delta(X); JP = ligacao_pi(X)
        info['pista'].append(NA.medir(S0, X)['alt']); info['pass_H'].append(passagem(JH, X)); info['pass_D'].append(passagem(JD, X)); info['pass_PI'].append(passagem(JP, X))
        rd = lambda: np.random.default_rng([2_800_000 + s, int(ruido * 10)]); fim = {}
        for nome, J in (('H', JH), ('D', JD), ('PI', JP)):
            g = NA.ganho_ajustado(J, X); info['g'].append((nome, g)); fim[nome] = NA.recuperar(J, X, S0, rd(), g, True, True)
        for v, S in (('H-estado', fim['H']), ('H-campo', campo(fim['H'], X)), ('D-estado', fim['D']), ('D-campo', campo(fim['D'], X)),
                     ('pista-campo', campo(S0, X)), ('PI-local', fim['PI'])):
            m = NA.medir(S, X)
            for k in m: out[v][k].append(m[k])
    print(f'  ruído {ruido} pronto', file=sys.stderr, flush=True); return ruido, {'R': out, 'info': info}
if __name__ == '__main__':
    print(f'vizinhos por neurônio na regra delta: média {np.mean([len(v) for v in VIZ]):.1f} (de {min(len(v) for v in VIZ)} a {max(len(v) for v in VIZ)})')
    with Pool(2) as pool: res = dict(pool.map(tarefa, NA.RUIDOS))
    for ruido in NA.RUIDOS:
        R_, I = res[ruido]['R'], res[ruido]['info']
        print(f"\n##### ruído {ruido} × o desvio · altura da pista {np.mean(I['pista']):.3f} · memória intacta pelas ligações: Hebb {np.mean(I['pass_H']):.3f} · delta {np.mean(I['pass_D']):.4f} · PI-local {np.mean(I['pass_PI']):.3f}")
        print(f"{'versão':<13}{'altura':>9}{'saturação':>11}{'erro':>8}{'identificação':>15}")
        for v in VERS: print(f"{v:<13}" + ''.join(f"{np.mean(R_[v][k]):>{w}.3f}" for k, w in (('alt', 9), ('sat', 11), ('err', 8), ('ident', 15))))
    R_, I = res[1.0]['R'], res[1.0]['info']; col = lambda v: np.array(R_[v]['alt'])
    def teste(nome, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{nome}: {d.mean():+.3f} (EP {se:.3f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    print('\n=== CRITÉRIOS (ruído principal) ===')
    p0 = np.mean(I['pass_D']); print(f"LA0 memória intacta pelas ligações delta: {p0:.4f} (>= 0,98) -> {'cumprido' if p0 >= .98 else 'NÃO cumprido'}")
    teste('LA1 D-estado − pista', col('D-estado') - np.array(I['pista'])); teste('LA2 D-estado − H-estado', col('D-estado') - col('H-estado'))
    teste('LA3 H-campo − H-estado', col('H-campo') - col('H-estado')); teste('LA4 D-campo − pista-campo', col('D-campo') - col('pista-campo'))
    json.dump({str(k): v for k, v in res.items()}, open(SP + 'EtapaLA_resultados.json', 'w'))
