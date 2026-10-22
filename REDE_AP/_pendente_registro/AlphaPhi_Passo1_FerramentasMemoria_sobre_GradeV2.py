# PASSO 1 — AS FERRAMENTAS DE MEMÓRIA v1 SOBRE A GRADE ESFÉRICA v2 (critérios aprovados pelo pesquisador em 22/10/2026, antes da execução)
# Ferramentas de Memória v1 importadas sem alteração; os pontos passam a ser os da Grade v2 (altura por √3). Memórias, pistas e sementes
# da Etapa PF afinada (120–129; mesmas sequências aleatórias), com a forma de cada memória calculada nos pontos novos.
# F1 campo afinado: estáveis na própria memória >= 88%, fundo entre 0,12 e 0,26 da memória. F2 equivalência (pista 40%): V0, V3, V4, V5
# a até ±0,05 da v1 (0,221; 0,261; 0,420; 0,382). F3 V4 > V0. F4 V5 > V6. F5 identificação da binária com pista 25% >= 0,95.
# (nível 0,0; pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
import os
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, '..') + '/'; SP = os.path.join(R, 'CampoEsferico_Etapas_S_K_PF') + '/'  # dados: rodar antes o B0 e copiar os .npy
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
PF = load('PF', SP + 'AlphaPhi_CampoEsferico_EtapaPF2_Afinada.py'); K = PF.K; FM = load('FM', R + 'AlphaPhi_FerramentasMemoria_v1.py'); GV2 = load('GV2', R + 'AlphaPhi_GradeEsferica_v2.py')
P2 = GV2.pontos_encaixados(610); Y2, GR2 = GV2.harmonicos(P2, 14); assert (GR2 == K.GR).all()
K.Y = Y2; PF.Y = Y2; PF.YP = np.linalg.pinv(Y2)               # a forma das memórias e as medidas passam a ser calculadas na v2
def tarefa(flip):
    T = K.calibrar(0.0); ni = 0; fl = int(flip * 100); R_ = {v: {k: [] for k in ('conc', 'ident', 'gros', 'fino')} for v in PF.VERS}; est = []; raz = []
    for s in PF.SEM:
        X = K.com_forma(T, np.random.default_rng([1_900_000 + s, ni])); J = FM.ligacoes(P2, X)
        CV, FU, wf = FM.campo(P2); Jm = CV * (X.T @ X) / 610; np.fill_diagonal(Jm, 0)
        est.append(float(((X @ J) * X > 0).mean())); raz.append(float(np.abs(X @ (wf * FU)).mean() / np.abs(X @ Jm).mean()))
        r2 = np.random.default_rng([1_950_000 + s, ni, fl]); S0 = X.copy(); k = int(round(flip * 610))
        for p in range(12): S0[p, r2.choice(610, k, replace=False)] *= -1
        res, _, _, _ = PF.rodar(J, X, S0, s, ni, fl); m = PF.medir(res, X)
        for v in PF.VERS:
            for kk in R_[v]: R_[v][kk].append(m[v][kk])
    print(f'  pista {flip} pronta', file=sys.stderr, flush=True); return flip, {'R': R_, 'est': est, 'raz': raz}
if __name__ == '__main__':
    with Pool(2) as pool: out = dict(pool.map(tarefa, [.40, .25]))
    v1 = json.load(open(SP + 'EtapaPF2_resultados.json'))['0.0|0.4']['R']
    R4 = out[.40]['R']; col = lambda v: np.array(R4[v]['conc'])
    print(f"F1 campo afinado: estáveis na própria memória {np.mean(out[.40]['est']):.3f} (>= 0,88) · fundo/memória {np.mean(out[.40]['raz']):.3f} (0,12–0,26) -> "
          f"{'cumprido' if np.mean(out[.40]['est']) >= .88 and .12 <= np.mean(out[.40]['raz']) <= .26 else 'NÃO cumprido'}")
    print(f"\n{'versão':<7}{'v2: conclusão 40%':>19}{'v1':>8}{'diferença':>11}{'v2: identificação 25%':>23}")
    ok2 = True
    for v in PF.VERS:
        a, b = col(v).mean(), np.mean(v1[v]['conc']); d = a - b
        if v in ('V0', 'V3', 'V4', 'V5'): ok2 &= abs(d) <= .05
        print(f"{v:<7}{a:>19.3f}{b:>8.3f}{d:>+11.3f}{np.mean(out[.25]['R'][v]['ident']):>23.3f}")
    print(f"\nF2 equivalência (V0, V3, V4, V5 a até ±0,05 da v1) -> {'cumprido' if ok2 else 'NÃO cumprido'}")
    def teste(nome, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{nome}: {d.mean():+.3f} (EP {se:.3f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    teste('F3 V4 − V0 (o previsor ajuda)', col('V4') - col('V0')); teste('F4 V5 − V6 (o batedor certo importa)', col('V5') - col('V6'))
    idt = np.mean(out[.25]['R']['V0']['ident']); print(f"F5 identificação da binária com pista 25%: {idt:.3f} (>= 0,95) -> {'cumprido' if idt >= .95 else 'NÃO cumprido'}")
    json.dump({str(k): v for k, v in out.items()}, open(os.path.join(H, 'Passo1_FerramentasV2_resultados.json'), 'w'))
