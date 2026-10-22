# VERIFICAÇÃO DA ETAPA T E DO CULTIVO NOS PONTOS COM ALTURA POR √3 (critérios aprovados pelo pesquisador em 22/10/2026, antes da execução)
# Código canônico das Etapas T e C importado SEM alteração; só os pontos trocados (altura frac(i·√3) no lugar de frac(i·√2)).
# Nota declarada antes de rodar: a Etapa T original usou 256 pontos (não 233); V1–V3 usam 256 pontos encaixados nas duas alturas;
# V4 usa as etapas da Etapa C (34 → 233). Os pontos de √2 rodam lado a lado como referência, mesmas sementes.
# V1 medidor: padrões de grau conhecido 2, 3, 4 → acerta 100%, condição <= 10.  V2 campo vazio: |L_dom − L*| <= 1 em >= 90% dos atratores.
# V3 memória sem forma (45 combinações, como na T2a): Spearman >= 0,8 e erro médio <= 0,05.  V4 cultivo: na faixa prevista em >= 90% das
# 50 repetições, em todas as etapas.  V5: em V2 e V4, √3 não fica mais de 5 pontos percentuais abaixo de √2.
import numpy as np, importlib.util, os, sys
from scipy.stats import spearmanr
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
T = load('T', os.path.join(R, 'AlphaPhi_CampoEsferico_EtapaT_Previsor.py')); C = load('C', os.path.join(R, 'AlphaPhi_CampoEsferico_EtapaC_Cultivo.py'))
e0, base = T.e0, T.base; GA = C.GA
def pts(n, raiz):
    i = np.arange(1, n + 1); z = 1 - 2 * ((i * raiz) % 1); r = np.sqrt(1 - z * z); a = i * GA; return np.c_[r * np.cos(a), r * np.sin(a), z]
assert np.allclose(pts(233, 2 ** .5), C.pontos(233))
ALT = {'√2': 2 ** .5, '√3': 3 ** .5}; out = {}
# V1
for nome, raiz in ALT.items():
    P = pts(T.N, raiz); Y, g = T.harmonicos(P); cond = float(np.linalg.cond(Y)); acerto = []
    for L in (2, 3, 4):
        rng = np.random.default_rng([500, L]); m = g == L
        for k in range(100):
            c = np.zeros(len(g)); c[m] = rng.normal(size=m.sum()); s = np.sign(Y @ c); acerto.append(int(T.grau_dominante(s[None], Y, g)[0]) == L)
    out[('V1', nome)] = (float(np.mean(acerto)), cond)
    print(f"V1 medidor ({nome}, {T.N} pontos, até o grau 14): acerto {np.mean(acerto):.3f} · condição {cond:.2f}", flush=True)
# V2
lam = T.funk_hecke(T.nucleo(.15, .45)); Ls = int(lam.argmax()); print(f'V2 previsão: L* = {Ls}')
for nome, raiz in ALT.items():
    P = pts(T.N, raiz); Y, g = T.harmonicos(P); Cm = e0.dog(e0.dnorm(P)); doms = []
    for s in range(5): doms.append(T.grau_dominante(T.campo_vazio(Cm, np.random.default_rng(80_000 + s)), Y, g))
    d = np.concatenate(doms); out[('V2', nome)] = float((np.abs(d - Ls) <= 1).mean())
    print(f"V2 campo vazio ({nome}): fração com |L_dom − L*| <= 1 = {out[('V2', nome)]:.3f} · contagem (0..9): {np.bincount(d, minlength=15)[:10].tolist()}", flush=True)
# V3 (ambiente sem forma, F1 = pontos de √3; F2 e F3 como na Etapa T)
PF = pts(T.N, 3 ** .5); PL = e0.latlong(); linhas = []
for s in range(10):
    P2 = base.aleat_esfera(T.N, np.random.default_rng(30_000 + s))
    for fn, P in (('F1', PF), ('F2', P2), ('F3', PL)):
        D = e0.dnorm(P)
        for sg in (.20, .35, .60):
            Cm = np.exp(-(D / sg) ** 2); np.fill_diagonal(Cm, 0)
            ger = lambda n, r: r.choice([-1., 1.], size=(n, T.N))
            A = ger(200, np.random.default_rng([300_000 + s, int(sg * 100)])); Gm = (A.T @ A) / len(A)
            for Pc in [5, 13, 20, 28, 36]:
                X = ger(Pc, np.random.default_rng([400_000 + s, int(sg * 100), Pc])); pv = T.previsto(Cm, Gm, Pc); est, rec = T.medido(Cm, X, s, Pc)
                linhas.append(((fn, sg, Pc), pv, est))
    print(f'  V3 semente {s} pronta', file=sys.stderr, flush=True)
grupos = sorted(set(l[0] for l in linhas)); M = np.array([[np.mean([l[1] for l in linhas if l[0] == gk]), np.mean([l[2] for l in linhas if l[0] == gk])] for gk in grupos])
rho = spearmanr(M[:, 0], M[:, 1])[0]; mae = float(np.abs(M[:, 0] - M[:, 1]).mean()); out['V3'] = (rho, mae)
print(f"V3 memória sem forma ({len(grupos)} combinações): Spearman {rho:.3f} · erro médio {mae:.4f}", flush=True)
# V4 (o cultivo da Etapa C, com o código dela)
for nome, raiz in ALT.items():
    PT = pts(233, raiz); prev = {}
    for N, Lmax in C.ETAPAS:
        se = .15 * np.sqrt(233 / N); lam = T.funk_hecke(T.nucleo(se, 3 * se), 20); Y, g = T.harmonicos(PT[:N], Lmax)
        prev[N] = dict(faixa=[L for L in range(21) if lam[L] >= .9 * lam.max()], Y=Y, g=g, C=C.ligacao(PT[:N], se), cond=float(np.linalg.cond(Y)))
    cult = {N: [] for N, _ in C.ETAPAS}
    for r in range(C.REP):
        rng = np.random.default_rng([900_000, r]); s = None
        for N, _ in C.ETAPAS:
            Cm = prev[N]['C']
            if s is None: s0 = rng.choice([-1., 1.], N)
            else:
                n0 = len(s); h = Cm[n0:, :n0] @ s; s0 = np.r_[s, np.where(h > 0, 1., np.where(h < 0, -1., rng.choice([-1., 1.], N - n0)))]
            s = C.dinamica(Cm, s0, rng); cult[N].append(C.grau(s, prev[N]['Y'], prev[N]['g']))
    fr = {N: float(np.isin(cult[N], prev[N]['faixa']).mean()) for N, _ in C.ETAPAS}; out[('V4', nome)] = fr
    print(f"V4 cultivo ({nome}): " + ' · '.join(f"{N}: faixa {prev[N]['faixa']} → {fr[N]:.2f} (condição {prev[N]['cond']:.1f})" for N, _ in C.ETAPAS), flush=True)
print('\n=== CRITÉRIOS (pontos com altura por √3) ===')
v1 = out[('V1', '√3')]; print(f"V1 medidor: acerto {v1[0]:.3f}, condição {v1[1]:.2f} -> {'cumprido' if v1[0] == 1 and v1[1] <= 10 else 'NÃO cumprido'}")
v2 = out[('V2', '√3')]; print(f"V2 campo vazio: {v2:.3f} (>= 0,90) -> {'cumprido' if v2 >= .9 else 'NÃO cumprido'}")
print(f"V3 memória sem forma: Spearman {rho:.3f}, erro {mae:.4f} -> {'cumprido' if rho >= .8 and mae <= .05 else 'NÃO cumprido'}")
v4 = out[('V4', '√3')]; print(f"V4 cultivo: pior etapa {min(v4.values()):.2f} (>= 0,90) -> {'cumprido' if min(v4.values()) >= .9 else 'NÃO cumprido'}")
d2 = out[('V2', '√3')] - out[('V2', '√2')]; d4 = min(out[('V4', '√3')][N] - out[('V4', '√2')][N] for N, _ in C.ETAPAS)
print(f"V5 sem perda em relação a √2: V2 {d2:+.3f}; V4 pior etapa {d4:+.3f} (>= −0,05) -> {'cumprido' if d2 >= -.05 and d4 >= -.05 else 'NÃO cumprido'}")
