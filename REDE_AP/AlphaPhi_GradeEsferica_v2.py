# AlphaPhi — Grade Esférica de Referência v2 (canônica) · 22/10/2026 · Vitor Edson Delavi · Claude
# A base que antecede as ferramentas (Entrada 346; REGISTRO_ELEMENTOS_REDE_AP.md, Partes I, III e V). Construída SOBRE a v1, que
# permanece congelada e intacta. Testes: G1–G8 e as duas correções (22/10); verificação das Etapas T e C nos pontos novos (V1–V5, 22/10).
#   B1/B2  pontos encaixados: longitude pelo ângulo áureo (como na v1) e ALTURA frac(i·√3) (v1: frac(i·√2)); qualquer começo cobre a
#          esfera (G1': pior vão 1,43 × a referência; condição do medidor 2,6); crescer não move os pontos antigos (G2)
#   B1     cascas em proporção áurea: r = φ^(−k), k = 0 … 6, com F(15−2k) pontos (610, 233, 89, 34, 13, 5, 2): densidade igual (G3);
#          cada casca usa os primeiros endereços da superfície, no seu raio
#   B3     extensão harmônica para o interior: cada grau L decai como r^L (G4)
#   B4     células por escala: cada estágio de Fibonacci define células (Voronoi); aninhamento médio 0,82 (G6)
#   B5     pesos pela área da célula (Voronoi esférico exato) nas médias e nas energias: o centro α como média pesada; a energia se
#          conserva em cada casca (testes G5' e G8' feitos nos pontos de √2: pior 1,0%; 0,6% a 1,0%; nos pontos de √3, ver a autoverificação)
#   B6     o endereço é o índice do ângulo áureo; escrita e leitura de uma matriz (ex.: faixa × instante) por anéis de latitude (G7)
# Regra: todo teste novo importa este módulo. A v1 continua sendo a base dos resultados canônicos de 19 a 21/10.
import numpy as np, importlib.util, os
from scipy.spatial import SphericalVoronoi
_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location('G1', os.path.join(_d, 'AlphaPhi_GradeEsferica_v1.py')); G1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(G1)
PHI, ALPHA, SEAL, ANGULO_AUREO = G1.PHI, G1.ALPHA, G1.SEAL, G1.ANGULO_AUREO
R_CENTRO, R_INTERIOR, R_SUPERFICIE = G1.R_CENTRO, G1.R_INTERIOR, G1.R_SUPERFICIE
ALTURA = 3 ** 0.5
N_SUP = 610
CASCAS = [(PHI ** (-k), n) for k, n in enumerate([610, 233, 89, 34, 13, 5, 2])]
ESTAGIOS = [34, 55, 89, 144, 233, 377, 610]
harmonicos, decompor, grau_dominante, lmax_mensuravel, para_esfericas, para_cartesianas = (
    G1.harmonicos, G1.decompor, G1.grau_dominante, G1.lmax_mensuravel, G1.para_esfericas, G1.para_cartesianas)

def pontos_encaixados(n, raio=1.0):
    """n pontos: longitude i·ângulo áureo, altura 1 − 2·frac(i·√3), i = 1 … n. Os n primeiros não dependem do total."""
    i = np.arange(1, n + 1); z = 1 - 2 * ((i * ALTURA) % 1); r = np.sqrt(np.clip(1 - z * z, 0, 1)); a = i * ANGULO_AUREO
    return raio * np.c_[r * np.cos(a), r * np.sin(a), z]
def distancia_geodesica(P):
    U = P / np.linalg.norm(P, axis=1, keepdims=True); return np.arccos(np.clip(U @ U.T, -1, 1))
def pesos(P):
    """Área da célula de cada ponto (Voronoi esférico, na esfera unitária); soma 4π."""
    U = P / np.linalg.norm(P, axis=1, keepdims=True); return SphericalVoronoi(U, radius=1.0).calculate_areas()
def media(f, P):
    w = pesos(P); return float(w @ f / w.sum())
def energia(f, P):
    """Energia do campo f sobre a casca dos pontos P (integral na esfera unitária das direções)."""
    return float(pesos(P) @ (np.asarray(f) ** 2))
def pontos_cascas():
    return [(r, pontos_encaixados(n, raio=r)) for r, n in CASCAS]
def extensao(c, grau, P):
    """Campo harmônico de coeficientes c (na superfície) nos pontos P do interior: Σ c_Lm r^L Y_Lm."""
    r = np.linalg.norm(P, axis=1); U = P / np.maximum(r, 1e-12)[:, None]; Y, g = harmonicos(U, int(grau.max()))
    return (Y * (r[:, None] ** g[None, :])) @ c
def centro_alfa(f, P):
    """O neurônio α central: a média do campo na superfície, pesada pela área (teorema do valor médio)."""
    return media(f, P)
def celulas(n, Q):
    A = pontos_encaixados(n); return np.argmax(Q @ A.T, axis=1)
def sepstro(c, grau):
    """Coh + Entr = 1 (por construção): Entr = entropia normalizada da energia por grau."""
    e = np.array([(c[grau == L] ** 2).sum() for L in range(int(grau.max()) + 1)]); p = e / e.sum(); p = p[p > 0]
    entr = float(-(p * np.log(p)).sum() / np.log(int(grau.max()) + 1)); return 1 - entr, entr
def enderecos_aneis(linhas, colunas, n=N_SUP):
    """Endereço de uma matriz linhas × colunas na esfera: linha = anel de latitude (do norte ao sul), coluna = posição em volta."""
    P = pontos_encaixados(n); ordem = np.argsort(-P[:, 2], kind='stable'); E = np.full((linhas, colunas), -1)
    for k in range(linhas):
        anel = ordem[k * colunas:(k + 1) * colunas]; E[k] = anel[np.argsort(np.arctan2(P[anel, 1], P[anel, 0]))]
    return E
def escrever(M, E, n=N_SUP):
    v = np.zeros(n); v[E.ravel()] = np.asarray(M).ravel(); return v
def ler(v, E):
    return np.asarray(v)[E.ravel()].reshape(E.shape)

# ── autoverificação: python3 AlphaPhi_GradeEsferica_v2.py ──
if __name__ == '__main__':
    ok = True; P = pontos_encaixados(610); rng = np.random.default_rng(0)
    t = all(np.allclose(pontos_encaixados(n), P[:n]) for n in ESTAGIOS); ok &= t; print('encaixe (crescer não move os pontos):', t)
    cond = {n: float(np.linalg.cond(harmonicos(P[:n], lmax_mensuravel(n))[0])) for n in (34, 55, 89, 144, 233)}; t = max(cond.values()) <= 10; ok &= t
    print('condição do medidor:', ', '.join(f'{n}: {c:.2f}' for n, c in cond.items()))
    dens = [n / (4 * np.pi * r ** 2) for r, n in CASCAS]; t = max(abs(d / dens[0] - 1) for d in dens) <= .10; ok &= t; print('cascas áureas, densidade igual:', t)
    w = pesos(P); t = abs(w.sum() - 4 * np.pi) < 1e-9; ok &= t; print('pesos somam 4π:', t)
    Y, g = harmonicos(P, 16); dif = []
    for k in range(20):
        c = rng.normal(size=len(g)) / (1 + g); f = Y @ c; dif.append(abs(centro_alfa(f, P) - c[g == 0][0] / np.sqrt(4 * np.pi)) / f.std())
    t = max(dif) <= .02; ok &= t; print(f'centro α = média pesada (pior {max(dif):.2%}):', t)
    c = rng.normal(size=len(g)); t = abs(energia(Y @ c, P) / (c ** 2).sum() - 1) <= .05; ok &= t; print('energia conservada na superfície:', t)
    E = enderecos_aneis(75, 8); M = rng.normal(size=(75, 8)); t = np.array_equal(ler(escrever(M, E), E), M) and len(np.unique(E)) == 600; ok &= t; print('escrita e leitura:', t)
    print('AUTOVERIFICAÇÃO:', 'OK' if ok else 'FALHOU')
