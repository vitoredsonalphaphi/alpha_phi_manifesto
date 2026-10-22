# TESTES DA GRADE ESFÉRICA v2 (G1–G8), sem rede e sem dado — fundamento aprovado pelo pesquisador em 22/10/2026.
# Limites numéricos declarados AQUI, antes de rodar (a Parte III do registro não tinha números):
# G1  o maior vão de cada começo n (34 … 610) ≤ 1,5 × o maior vão de uma esfera de Fibonacci feita para exatamente n pontos
# G2  os n primeiros pontos idênticos para todo n ≤ 610 (exato)
# G3  densidade por área de cada casca a até 10% da densidade da superfície
# G4  campo de grau L0 (0 … 9) estendido: em cada casca que mede L0, o grau dominante é L0 e a amplitude é r^L0 com erro ≤ 5%
# G5  valor no centro × média dos 610 valores da superfície: diferença ≤ 2% do desvio do campo
# G6  células aninhadas: em média, ≥ 75% de cada célula fina cai na célula grosseira do seu centro (estágios encaixados consecutivos)
# G7  harmônicos até o grau 16 em 610 pontos e o endereço faixa × instante: ida e volta com erro ≤ 1e-8
# G8  energia: (4π/N)·Σ f² × Σ c² com erro ≤ 2% em média (campos ao acaso); na superfície até o grau 16; em cada casca interna,
#     até o grau que ela resolve ((L+1)² ≤ n/2), comparando com Σ c² r^(2L)
# Correção de 22/10 (antes de interpretar): na primeira execução, a linha das cascas internas do G8 repetia a superfície, porque
# nenhuma casca interna resolve o grau 16; agora cada casca é medida até o seu grau, como diz o critério. G1–G7 sem mudança.
import numpy as np, importlib.util
import os
SP = os.path.dirname(os.path.abspath(__file__)) + '/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
G = load('G2', SP + 'AlphaPhi_GradeEsferica_v2.py'); G1 = G.G1; rng = np.random.default_rng(2026)
def fib_n(n):
    i = np.arange(n) + 0.5; z = 1 - 2 * i / n; a = np.pi * (1 + 5 ** 0.5) * i; rr = np.sqrt(1 - z ** 2); return np.c_[rr * np.cos(a), rr * np.sin(a), z]
Q = fib_n(20000)        # pontos de prova, densos
def vao(A): return float(np.arccos(np.clip((Q @ A.T).max(1), -1, 1)).max())
ok = {}
# G1
P = G1.pontos_encaixados(610); piores = []
for n in range(34, 611):
    piores.append(vao(P[:n]) / vao(fib_n(n)))
piores = np.array(piores); ok['G1'] = piores.max() <= 1.5
print(f"G1 endereço progressivo: maior vão / vão de referência — mediana {np.median(piores):.2f}, pior {piores.max():.2f} (n = {34 + int(piores.argmax())}); acima de 1,5 em {(piores > 1.5).sum()} de {len(piores)} começos -> {'cumprido' if ok['G1'] else 'NÃO cumprido'}")
alea = np.mean([vao(np.array([x / np.linalg.norm(x) for x in rng.normal(size=(n, 3))])) / vao(fib_n(n)) for n in (34, 89, 233, 610)])
print(f"    (comparação: pontos ao acaso dão {alea:.2f} vezes o vão de referência)")
# G2
ok['G2'] = all(np.allclose(G1.pontos_encaixados(n), P[:n]) for n in range(1, 611))
print(f"G2 crescer não move os pontos antigos: {'cumprido' if ok['G2'] else 'NÃO cumprido'}")
# G3
dens = [n / (4 * np.pi * r ** 2) for r, n in G.CASCAS]; d0 = dens[0]; desv = max(abs(d / d0 - 1) for d in dens)
ok['G3'] = desv <= 0.10
print("G3 cascas áureas: " + ' · '.join(f"r={r:.3f}: {n} pts" for r, n in G.CASCAS) + f" · maior desvio de densidade {desv:.1%} -> {'cumprido' if ok['G3'] else 'NÃO cumprido'}")
# G4
erros = []; graus_ok = True; cas = G.pontos_cascas()
for L0 in range(10):
    Yd, gd = G1.harmonicos(P, 16); c = np.zeros(len(gd)); c[gd == L0] = rng.normal(size=(gd == L0).sum())
    for r, Pk in cas:
        n = len(Pk); Lm = G1.lmax_mensuravel(n)
        if Lm < L0 or n < 13: continue
        f = G.extensao(c, gd, Pk); Yk, gk = G1.harmonicos(Pk / r, Lm); ck, en = G1.decompor(f, Yk, gk)
        graus_ok &= int(np.argmax(en)) == L0
        amp = np.linalg.norm(ck[gk == L0]) / np.linalg.norm(c[gd == L0]); erros.append(abs(amp / r ** L0 - 1))
ok['G4'] = graus_ok and max(erros) <= 0.05
print(f"G4 base para o interior: grau dominante certo em todas as cascas que o medem: {graus_ok}; maior erro da amplitude r^L {max(erros):.2%} -> {'cumprido' if ok['G4'] else 'NÃO cumprido'}")
# G5
Yd, gd = G1.harmonicos(P, 16); dif = []
for t in range(50):
    c = rng.normal(size=len(gd)) / (1 + gd); f = Yd @ c; dif.append(abs(G.centro_alfa(c, gd) - f.mean()) / f.std())
ok['G5'] = max(dif) <= 0.02
print(f"G5 o neurônio α central é a média da superfície: diferença média {np.mean(dif):.3%}, pior {max(dif):.3%} do desvio -> {'cumprido' if ok['G5'] else 'NÃO cumprido'}")
# G6
fr = []
for a, b in zip(G.ESTAGIOS[:-1], G.ESTAGIOS[1:]):
    cg, cf = G.celulas(a, Q), G.celulas(b, Q); A = G1.pontos_encaixados(b); dono = G.celulas(a, A)   # célula grosseira do centro de cada célula fina
    v = [np.mean(cg[cf == j] == dono[j]) for j in range(b) if (cf == j).any()]; fr.append(np.mean(v))
ok['G6'] = np.mean(fr) >= 0.75
print("G6 células por escala, aninhamento: " + ' · '.join(f"{a}→{b}: {x:.2f}" for a, b, x in zip(G.ESTAGIOS[:-1], G.ESTAGIOS[1:], fr)) + f" · média {np.mean(fr):.2f} -> {'cumprido' if ok['G6'] else 'NÃO cumprido'}")
# G7
Y16, g16 = G1.harmonicos(P, 16); c = rng.normal(size=len(g16)); c2, _ = G1.decompor(Y16 @ c, Y16, g16); e1 = float(np.abs(c2 - c).max())
V2 = load('V2', SP + 'AlphaPhi_GradeEsferica_v2_proposta.py'); M = rng.normal(size=(V2.NFX, V2.NT)); e2 = float(np.abs(V2.da_esfera(V2.na_esfera(M)) - M).max())
ok['G7'] = e1 <= 1e-8 and e2 <= 1e-8
print(f"G7 escrita e leitura: harmônicos até o grau 16, erro {e1:.1e}; endereço faixa × instante, erro {e2:.1e} -> {'cumprido' if ok['G7'] else 'NÃO cumprido'}")
# G8
er = []; erc = {}
for t in range(50):
    c = rng.normal(size=len(g16)); f = Y16 @ c; er.append(abs((4 * np.pi / 610) * (f ** 2).sum() / (c ** 2).sum() - 1))
for r, Pk in cas[1:4]:
    Lm = G1.lmax_mensuravel(len(Pk)); Yk, gk = G1.harmonicos(Pk / r, Lm); e = []
    for t in range(50):
        c = rng.normal(size=len(gk)); fk = G.extensao(c, gk, Pk); e.append(abs((4 * np.pi / len(Pk)) * (fk ** 2).sum() / ((c ** 2) * r ** (2 * gk)).sum() - 1))
    erc[(round(r, 3), len(Pk), Lm)] = (np.mean(e), max(e))
ok['G8'] = np.mean(er) <= 0.02 and all(v[0] <= 0.02 for v in erc.values())
print(f"G8 conservação da energia: superfície (até o grau 16) {np.mean(er):.2%} (pior {max(er):.2%}); " + ' · '.join(f"casca r={k[0]} ({k[1]} pts, até o grau {k[2]}): {v[0]:.2%} (pior {v[1]:.2%})" for k, v in erc.items()) + f" -> {'cumprido' if ok['G8'] else 'NÃO cumprido'}")
c = rng.normal(size=len(g16)); Coh, Entr = G.sepstro(c, g16); print(f"    Sépstro de um campo ao acaso: Coh {Coh:.3f} + Entr {Entr:.3f} = {Coh + Entr:.3f} (por construção)")
print('\nRESUMO: ' + ' · '.join(f"{k} {'✓' if v else '✗'}" for k, v in ok.items()))
