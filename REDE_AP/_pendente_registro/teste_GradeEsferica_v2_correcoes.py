# CORREÇÕES DA GRADE ESFÉRICA v2 (critérios aprovados pelo pesquisador em 22/10/2026, antes da execução)
# CORREÇÃO 1 — o peso de cada ponto pela área da sua célula (células de Voronoi esféricas, áreas exatas), só nas médias e energias.
#   G5': média pesada dos 610 valores a até 2% do valor no centro, no pior de 50 campos ao acaso.
#   G8': energia pesada a até 2% da verdadeira, em média, na superfície (grau 16) e nas cascas de 233, 89 e 34 pontos (até o grau
#        que cada uma resolve). Controle: os mesmos pesos embaralhados entre os pontos devem dar erro MAIOR que os pesos certos.
# CORREÇÃO 2 — a altura dos pontos (longitude fixa pelo ângulo áureo; encaixe mantido). Candidatas: √2 (a atual), √3, van der Corput
#   (base 2), número plástico (frac(i/ρ), ρ ≈ 1,3247). Para cada uma: G1' maior vão de cada começo (34 … 610) ≤ 1,5 × a referência
#   em TODOS os começos; G2 encaixe exato; G6 aninhamento médio ≥ 0,75; condição do medidor por harmônicos ≤ 10 em 34, 55, 89, 144,
#   233 pontos. Escolha: entre as que passam em tudo, a de menor pior vão; se nenhuma passar, fica a √2.
#   (Nota da proposta: a altura pelo número de ouro foi descartada antes de rodar: com a longitude também por φ, os pontos se alinham
#   numa espiral.) Uma vencedora NÃO substitui a v1: vira candidata da v2 e exige repetir a verificação da Etapa T nos pontos novos.
import numpy as np, importlib.util
from scipy.spatial import SphericalVoronoi
import os
SP = os.path.dirname(os.path.abspath(__file__)) + '/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
G = load('G2', SP + 'AlphaPhi_GradeEsferica_v2.py'); G1 = G.G1; rng = np.random.default_rng(2027)
def areas(U): return SphericalVoronoi(U, radius=1.0).calculate_areas()
print('=== CORREÇÃO 1: pesos pela área da célula ===')
P = G1.pontos_encaixados(610); W = areas(P); print(f'áreas: soma {W.sum():.6f} (4π = {4 * np.pi:.6f}); menor/maior célula {W.min() / W.mean():.2f} / {W.max() / W.mean():.2f} da média')
Wemb = rng.permutation(W)
Y16, g16 = G1.harmonicos(P, 16); d_s, d_w, d_e = [], [], []
for t in range(50):
    c = rng.normal(size=len(g16)) / (1 + g16); f = Y16 @ c; cen = G.centro_alfa(c, g16)
    d_s.append(abs(cen - f.mean()) / f.std()); d_w.append(abs(cen - (W @ f) / W.sum()) / f.std()); d_e.append(abs(cen - (Wemb @ f) / Wemb.sum()) / f.std())
ok5 = max(d_w) <= 0.02
print(f"G5' o centro α como média pesada: simples {np.mean(d_s):.3%} (pior {max(d_s):.3%}) · pesada {np.mean(d_w):.3%} (pior {max(d_w):.3%}) · pesos embaralhados {np.mean(d_e):.3%} (pior {max(d_e):.3%}) -> {'cumprido' if ok5 else 'NÃO cumprido'}")
res8 = {}; ctrl_ok = True
for r, Pk in G.pontos_cascas()[:4]:
    U = Pk / r; Lm = 16 if len(Pk) == 610 else G1.lmax_mensuravel(len(Pk)); Yk, gk = G1.harmonicos(U, Lm); Wk = areas(U); We = rng.permutation(Wk)
    es, ew, ee = [], [], []
    for t in range(50):
        c = rng.normal(size=len(gk)); fk = G.extensao(c, gk, Pk); ver = ((c ** 2) * r ** (2 * gk)).sum()
        es.append(abs((4 * np.pi / len(Pk)) * (fk ** 2).sum() / ver - 1)); ew.append(abs((Wk @ fk ** 2) / ver - 1)); ee.append(abs((We @ fk ** 2) / ver - 1))
    res8[len(Pk)] = (Lm, np.mean(es), np.mean(ew), max(ew), np.mean(ee)); ctrl_ok &= np.mean(ee) > np.mean(ew)
ok8 = all(v[2] <= 0.02 for v in res8.values())
for n, (Lm, s_, w_, wp, e_) in res8.items(): print(f"   casca de {n} pontos (até o grau {Lm}): simples {s_:.2%} · pesada {w_:.2%} (pior {wp:.2%}) · embaralhados {e_:.2%}")
print(f"G8' conservação com pesos pela área -> {'cumprido' if ok8 else 'NÃO cumprido'} · controle (embaralhados piores que os certos em todas as cascas): {'sim' if ctrl_ok else 'não'}")
print('\n=== CORREÇÃO 2: a altura dos pontos ===')
ANG = G1.ANGULO_AUREO; RHO = 1.3247179572447460
def vdc(i):
    x, d = 0.0, 0.5
    while i: x += d * (i & 1); i >>= 1; d /= 2
    return x
ALT = {'√2 (atual)': lambda i: (i * np.sqrt(2)) % 1, '√3': lambda i: (i * np.sqrt(3)) % 1, 'van der Corput': vdc, 'plástico': lambda i: (i / RHO) % 1}
def pontos(n, h):
    i = np.arange(1, n + 1); z = 1 - 2 * np.array([h(k) for k in i]); rr = np.sqrt(np.clip(1 - z ** 2, 0, 1)); a = i * ANG; return np.c_[rr * np.cos(a), rr * np.sin(a), z]
print('reprodução da v1 pela fórmula genérica (√2):', bool(np.allclose(pontos(610, ALT['√2 (atual)']), P)))
def fib_n(n):
    i = np.arange(n) + 0.5; z = 1 - 2 * i / n; a = np.pi * (1 + 5 ** 0.5) * i; rr = np.sqrt(1 - z ** 2); return np.c_[rr * np.cos(a), rr * np.sin(a), z]
Q = fib_n(20000); ref = {n: None for n in range(34, 611)}
def vao(A): return float(np.arccos(np.clip((Q @ A.T).max(1), -1, 1)).max())
for n in ref: ref[n] = vao(fib_n(n))
resultado = {}
for nome, h in ALT.items():
    Pn = pontos(610, h); rat = np.array([vao(Pn[:n]) / ref[n] for n in range(34, 611)])
    enc = all(np.allclose(pontos(n, h), Pn[:n]) for n in (34, 89, 233, 377))
    fr = []
    for a, b in zip(G.ESTAGIOS[:-1], G.ESTAGIOS[1:]):
        A, B = Pn[:a], Pn[:b]; cg = np.argmax(Q @ A.T, 1); cf = np.argmax(Q @ B.T, 1); dono = np.argmax(B @ A.T, 1)
        fr.append(np.mean([np.mean(cg[cf == j] == dono[j]) for j in range(b) if (cf == j).any()]))
    cond = {n: float(np.linalg.cond(G1.harmonicos(Pn[:n], G1.lmax_mensuravel(n))[0])) for n in (34, 55, 89, 144, 233)}
    passa = rat.max() <= 1.5 and enc and np.mean(fr) >= .75 and max(cond.values()) <= 10
    resultado[nome] = (rat.max(), np.median(rat), (rat > 1.5).sum(), enc, np.mean(fr), max(cond.values()), passa)
    print(f"{nome:<15} G1': pior {rat.max():.2f} · mediana {np.median(rat):.2f} · começos acima de 1,5: {(rat > 1.5).sum()} · encaixe {enc} · aninhamento {np.mean(fr):.2f} · condição máx {max(cond.values()):.2f} -> {'PASSA em tudo' if passa else 'não passa'}")
cand = [k for k, v in resultado.items() if v[-1]]
print('\nescolha:', (min(cand, key=lambda k: resultado[k][0]) if cand else 'nenhuma candidata passa: fica a √2 (atual)'))
