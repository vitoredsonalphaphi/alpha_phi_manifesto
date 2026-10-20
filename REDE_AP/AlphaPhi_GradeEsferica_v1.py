# Grade Esférica de Referência AlphaPhi — v1 (20/10/2026)
# O equivalente da grade euclidiana para o campo esférico da Rede AP, fixado uma vez. Todo teste novo IMPORTA este módulo.
#
#   Posição ........ (r, θ, φ): raio, colatitude (0 no polo norte, π no sul), longitude (−π a π)
#   Pontos ......... encaixados pelo ângulo áureo (os n primeiros de qualquer etapa contêm os da anterior) — Etapa C
#   Frequência ..... harmônicos esféricos reais Y_Lm: grau L (a "frequência" da esfera) e ordem m (a direção) — Etapa T
#   Raio ........... cascas do modelo canônico: centro α (r = 0) → interior (0 < r < 1) → superfície do campo harmônico (r = 1)
#   Leitura ........ decomposição de um campo em graus, e reconstrução de cada grau numa grade (θ, φ) para as cascas do scanner
#
# Constantes do núcleo duro (CLAUDE.md): φ, α = 1/137,035999, SEAL = 1/φ.
import numpy as np
from scipy.special import sph_harm_y

PHI = (1 + 5 ** .5) / 2
ALPHA = 1 / 137.035999
SEAL = 1 / PHI
ANGULO_AUREO = 2 * np.pi * (1 - 1 / PHI)

# ── raios canônicos (modelo espacial: do centro para fora; nunca inverter) ──
R_CENTRO, R_INTERIOR, R_SUPERFICIE = 0.0, 1 / PHI, 1.0

def pontos_encaixados(n, raio=1.0):
    """n pontos na esfera de raio dado: ângulo áureo em longitude, altura frac(i·√2). Os n primeiros não dependem do total."""
    i = np.arange(1, n + 1); z = 1 - 2 * ((i * 2 ** .5) % 1); rr = np.sqrt(1 - z * z); a = i * ANGULO_AUREO
    return raio * np.c_[rr * np.cos(a), rr * np.sin(a), z]

def para_esfericas(P):
    """(x, y, z) → (r, θ, φ)."""
    r = np.linalg.norm(P, axis=1); th = np.arccos(np.clip(P[:, 2] / np.maximum(r, 1e-15), -1, 1)); ph = np.arctan2(P[:, 1], P[:, 0])
    return r, th, ph

def para_cartesianas(r, th, ph):
    return np.c_[r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)]

def distancia_geodesica(P):
    """Distância ao longo da esfera (ângulo entre os pontos, em radianos), entre todos os pares."""
    U = P / np.linalg.norm(P, axis=1, keepdims=True); return np.arccos(np.clip(U @ U.T, -1, 1))

def harmonicos(P, Lmax=14):
    """Base real Y_Lm nos pontos P (mesma construção da Etapa T). Devolve (Y [n × (Lmax+1)²], grau [(Lmax+1)²])."""
    _, th, ph = para_esfericas(P); cols, grau = [], []
    for L in range(Lmax + 1):
        for m in range(-L, L + 1):
            Yc = sph_harm_y(L, abs(m), th, ph)
            cols.append(np.sqrt(2) * (Yc.imag if m < 0 else Yc.real) if m != 0 else Yc.real); grau.append(L)
    return np.array(cols).T, np.array(grau)

def condicao(P, Lmax):
    """Número de condição do medidor por harmônicos (≤ 10 = medidor confiável; critério C0 da Etapa C)."""
    return float(np.linalg.cond(harmonicos(P, Lmax)[0]))

def lmax_mensuravel(n):
    """Maior grau medível com folga em n pontos: (L+1)² ≤ n/2 (regra usada nas Etapas C e T)."""
    return int(np.floor(np.sqrt(n / 2))) - 1

def decompor(campo, Y, grau):
    """Coeficientes do campo na base (mínimos quadrados) e energia por grau L."""
    c = np.linalg.lstsq(Y, campo, rcond=None)[0]
    energia = np.array([(c[grau == L] ** 2).sum(0) for L in range(grau.max() + 1)])
    return c, energia

def grau_dominante(campo, Y, grau):
    return int(np.argmax(decompor(campo, Y, grau)[1]))

def grade_angular(n_theta=13, n_phi=25):
    """Grade (θ, φ) regular para desenhar cascas."""
    th = np.linspace(0, np.pi, n_theta); ph = np.linspace(-np.pi, np.pi, n_phi); TH, PH = np.meshgrid(th, ph, indexing='ij')
    return th, ph, para_cartesianas(np.ones(TH.size), TH.ravel(), PH.ravel())

def cascas(c, grau, Ygrade, forma):
    """Reconstrói, para cada grau L, a parte do campo daquele grau na grade angular: (L+1 cascas, *forma)."""
    _, gg = None, grau
    return np.array([(Ygrade[:, grau == L] @ c[grau == L]).reshape(forma) for L in range(grau.max() + 1)])

def raio_da_casca(L, Lmax, r_min=0.18):
    """Raio de leitura da casca do grau L: grau 0 perto do centro, grau Lmax na superfície (do centro para fora)."""
    return r_min + (1 - r_min) * L / max(Lmax, 1)

# ── autoverificação: roda com  python3 AlphaPhi_GradeEsferica_v1.py ──
if __name__ == '__main__':
    import importlib.util, os
    d = os.path.dirname(os.path.abspath(__file__)); ok = True
    P = pontos_encaixados(233); r, th, ph = para_esfericas(P)
    print(f'raios: centro {R_CENTRO} · interior 1/φ = {R_INTERIOR:.6f} · superfície {R_SUPERFICIE}')
    t1 = np.allclose(pontos_encaixados(89), pontos_encaixados(233)[:89]); ok &= t1; print(f'encaixe 89 ⊂ 233: {t1}')
    t2 = np.allclose(para_cartesianas(r, th, ph), P); ok &= t2; print(f'ida e volta (x,y,z) ↔ (r,θ,φ): {t2}')
    for n in (34, 55, 89, 144, 233):
        c_ = condicao(pontos_encaixados(n), lmax_mensuravel(n)); ok &= c_ <= 10; print(f'n={n:<4} Lmax medível {lmax_mensuravel(n):<2} condição {c_:.2f}')
    sp = importlib.util.spec_from_file_location('c', os.path.join(d, 'AlphaPhi_CampoEsferico_EtapaC_Cultivo.py'))
    c = importlib.util.module_from_spec(sp); sp.loader.exec_module(c)
    t3 = np.allclose(c.pontos(233), P); ok &= t3; print(f'mesmos pontos da Etapa C (canônica v1): {t3}')
    Yv1, gv1 = c.t.harmonicos(P, 9); Y, g = harmonicos(P, 9); t4 = np.allclose(Yv1, Y) and (gv1 == g).all(); ok &= t4
    print(f'mesma base de harmônicos da Etapa T (canônica v1): {t4}')
    v = Y[:, g == 3] @ np.random.default_rng(0).normal(size=7); t5 = grau_dominante(v, Y, g) == 3; ok &= t5; print(f'recupera um campo de grau 3: {t5}')
    print('AUTOVERIFICAÇÃO:', 'OK' if ok else 'FALHOU')
