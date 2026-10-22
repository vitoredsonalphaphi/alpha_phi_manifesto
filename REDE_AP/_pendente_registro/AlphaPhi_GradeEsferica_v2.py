# AlphaPhi — Grade Esférica de Referência v2 (candidata ao canônico; testes G1–G8 em 22/10: G2, G3, G4, G6, G7 cumpridos; G1, G5, G8 não) · 22/10/2026 · Vitor Edson Delavi · Claude
# Construída SOBRE a Grade Esférica v1 (importada sem alteração). Fundamento: REGISTRO_ELEMENTOS_REDE_AP.md, Partes I e III;
# Entradas 346 (a estrutura que antecede as ferramentas), 352–355 (direções, o além, o ponto e o campo, a respiração).
#   B6/G1–G2  endereço progressivo: o índice do ângulo áureo é o endereço; qualquer começo cobre a esfera; crescer não move nada
#   B1/B2/G3  cascas radiais em proporção áurea: r_k = φ^(−k), k = 0 … 6, com F(15−2k) pontos (610, 233, 89, 34, 13, 5, 2):
#             a densidade por área é a mesma em todas as cascas, e cada casca usa os PRIMEIROS endereços da superfície (o tronco)
#   B3/G4     base para o interior: a extensão harmônica (cada grau L decai como r^L rumo ao centro)
#   G5        o neurônio α central: o valor do campo no centro = a média da superfície (teorema do valor médio)
#   B4/G6     células por escala: cada estágio de Fibonacci define células (a região mais próxima de cada um dos n primeiros pontos)
#   G7        escrita e leitura: harmônicos e o endereço faixa × instante (Etapa VL2) vão e voltam sem perda
#   B5/G8     conservação: a energia do campo se conserva entre pontos, harmônicos e cascas; Coh + Entr = 1 por casca (por construção)
import numpy as np, importlib.util, os
_d = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
_s = importlib.util.spec_from_file_location('G1', os.path.join(_d, 'AlphaPhi_GradeEsferica_v1.py')); G1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(G1)
PHI = G1.PHI; ALPHA = G1.ALPHA; SEAL = G1.SEAL
N_SUP = 610
FIBO = [2, 5, 13, 34, 89, 233, 610]                       # F3, F5, F7, F9, F11, F13, F15
CASCAS = [(PHI ** (-k), FIBO[::-1][k]) for k in range(7)]  # (raio, pontos): (1, 610), (1/φ, 233), (1/φ², 89), …
ESTAGIOS = [34, 55, 89, 144, 233, 377, 610]

def endereco(i):
    """O endereço i (1-based na v1) → ponto na superfície. O endereço é o próprio índice do ângulo áureo."""
    return G1.pontos_encaixados(i)[-1]
def pontos_cascas():
    """Lista de (raio, pontos 3D) de cada casca; cada casca usa os primeiros endereços da superfície, no seu raio."""
    return [(r, G1.pontos_encaixados(n, raio=r)) for r, n in CASCAS]
def extensao(c, grau, P):
    """Campo harmônico com coeficientes c (na superfície) avaliado em pontos P quaisquer do interior: Σ c_Lm r^L Y_Lm."""
    r = np.linalg.norm(P, axis=1); u = P / np.maximum(r, 1e-12)[:, None]; Y, g = G1.harmonicos(u, int(grau.max()))
    return Y @ (c * 1.0) * 0 + (Y * (r[:, None] ** g[None, :])) @ c
def centro_alfa(c, grau):
    """Valor no centro (r = 0): só o grau 0 sobrevive."""
    return float(c[grau == 0].sum() * (1 / np.sqrt(4 * np.pi)))
def celulas(n, Q):
    """Célula de cada ponto Q no estágio n: o índice do mais próximo entre os n primeiros endereços."""
    A = G1.pontos_encaixados(n); return np.argmax(Q @ A.T, axis=1)
def sepstro(c, grau):
    """Coh + Entr = 1 de um campo: Entr = entropia normalizada da energia por grau; Coh = 1 − Entr."""
    e = np.array([(c[grau == L] ** 2).sum() for L in range(int(grau.max()) + 1)]); p = e / e.sum(); p = p[p > 0]
    entr = float(-(p * np.log(p)).sum() / np.log(int(grau.max()) + 1)); return 1 - entr, entr
