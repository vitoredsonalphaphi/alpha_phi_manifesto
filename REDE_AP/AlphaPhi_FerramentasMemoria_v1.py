# AlphaPhi — Ferramentas de Memória v1 (Rede AP no campo esférico) · 21/10/2026 · Vitor Edson Delavi · Claude
# Versão provisória do canônico: vale até que estas mesmas ferramentas sejam refinadas. Base: AlphaPhi_GradeEsferica_v1.py.
# Reúne, sem alteração de comportamento, as ferramentas que responderam na Etapa PF afinada (conversa de 21/10/2026):
#   1. a AFINAÇÃO do fundo do campo: peso escalado pela massa da vizinhança (fundo/memória ≈ 0,16; estáveis na memória 91,5%)
#   2. o PREVISOR: a cada volta, a mistura das memórias instaladas que mais ressoam com o estado (softmax β = 8), força 0,5
#   3. o BATEDOR: pré-função tênue (ganho 0,8, 5 passos) a partir da pista; direção × firmeza; força 0,5
# Não incluídos (sem efeito positivo na Etapa PF): o neurônio contínuo sozinho (PF1) e o resfriamento (V2).
# Agnóstico por construção: nada aqui depende do sinal de origem; as memórias entram como padrões ±1 nos pontos da grade.
import numpy as np, importlib.util, os
_d = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location('grade', os.path.join(_d, 'AlphaPhi_GradeEsferica_v1.py')); G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)

PESO_FUNDO_BASE = 0.02; SIGMA_BASE = 0.35          # desenho original do fundo (Etapas C, E, H)
GANHO = 3.0; GANHO_BATEDOR = 0.8; PASSOS_BATEDOR = 5; FORCA_BATEDOR = 0.5; FORCA_PREVISOR = 0.5; BETA_PREVISOR = 8.0
MAXV = 50; TOL = 1e-3

def espacamento(N):
    """Espaçamento médio entre pontos (geodésica normalizada por π)."""
    return np.sqrt(4 * np.pi / N) / np.pi

def campo(P):
    """Vizinhança da memória (σ = 2 × espaçamento), fundo do campo (DoG, σe = 0,15·√(233/N)) e o peso AFINADO do fundo."""
    N = len(P); D = G.distancia_geodesica(P) / np.pi
    CV = np.exp(-(D / (2 * espacamento(N))) ** 2); np.fill_diagonal(CV, 0)
    C35 = np.exp(-(D / SIGMA_BASE) ** 2); np.fill_diagonal(C35, 0)
    se = .15 * np.sqrt(233 / N); FU = np.exp(-(D / se) ** 2) - .5 * np.exp(-(D / (3 * se)) ** 2); np.fill_diagonal(FU, 0)
    return CV, FU, PESO_FUNDO_BASE * CV.sum(1).mean() / C35.sum(1).mean()

def ligacoes(P, X):
    """Ligações da AlphaFi esférica afinada: vizinhança ∘ Hebb + fundo com peso escalado."""
    CV, FU, wf = campo(P); N = len(P); J = CV * (X.T @ X) / N + wf * FU; np.fill_diagonal(J, 0); return J

def batedor(J, S0, hesc):
    """Pré-função: versão tênue da rede a partir da pista, sem decidir; devolve direção × firmeza."""
    s = S0.copy(); U = []
    for _ in range(PASSOS_BATEDOR):
        s = np.tanh(GANHO_BATEDOR * (s @ J) / hesc); U.append(s / (np.abs(s).max(1, keepdims=True) + 1e-12))
    u = np.mean(U, 0); return np.sign(u) * np.abs(u)

def previsor(S, X):
    """Mistura das memórias instaladas pesada pela ressonância com o estado atual."""
    N = X.shape[1]; m = np.sign(S) @ X.T / N; w = np.exp(BETA_PREVISOR * (m - m.max(1, keepdims=True))); w /= w.sum(1, keepdims=True); return w @ X

def recuperar(J, X, S0, rng, usar_previsor=True, usar_batedor=False, B=None, guardar=False):
    """Dinâmica de recuperação (neurônio contínuo, assíncrona) com previsor e/ou batedor. Devolve o estado final (e o histórico)."""
    N = S0.shape[1]; S = S0.copy(); hesc = np.abs(S0 @ J).mean(1, keepdims=True); hist = []
    if usar_batedor and B is None: B = batedor(J, S0, hesc)
    for v in range(MAXV):
        extra = np.zeros_like(S)
        if usar_batedor: extra += FORCA_BATEDOR * B
        if usar_previsor: extra += FORCA_PREVISOR * previsor(S, X)
        mud = 0.
        for i in rng.permutation(N):
            novo = np.tanh(GANHO * (S @ J[:, i] / hesc[:, 0] + extra[:, i])); mud = max(mud, float(np.abs(novo - S[:, i]).max())); S[:, i] = novo
        if guardar: hist.append(S.copy())
        if mud < TOL: break
    return (S, hist) if guardar else S

def recuperar_binaria(J, S0, rng):
    """Dinâmica binária (±1), assíncrona, como nas etapas anteriores."""
    N = S0.shape[1]; S = S0.copy()
    for v in range(MAXV):
        mud = 0
        for i in rng.permutation(N):
            h = S @ J[:, i]; novo = np.where(h > 0, 1., np.where(h < 0, -1., S[:, i])); mud += int((novo != S[:, i]).sum()); S[:, i] = novo
        if mud == 0 and v > 0: break
    return S

def conclusao(S, X):
    """Régua própria: sobreposição final do sinal com a memória certa (0 = acaso, 1 = memória completa)."""
    sg = np.sign(S); sg[sg == 0] = 1; return float((sg * X).mean())

# ── autoverificação: roda com  python3 AlphaPhi_FerramentasMemoria_v1.py ──
if __name__ == '__main__':
    P = G.pontos_encaixados(610); CV, FU, wf = campo(P); print(f'peso do fundo afinado: {wf:.5f} (base 0,02)')
    rng = np.random.default_rng(0); Y, gr = G.harmonicos(P, 14); X = np.where(rng.normal(size=(12, Y.shape[1])) @ Y.T >= 0, 1., -1.)
    J = ligacoes(P, X); Jm = CV * (X.T @ X) / 610; np.fill_diagonal(Jm, 0)
    r = np.abs(X @ (wf * FU)).mean() / np.abs(X @ Jm).mean(); est = ((X @ J) * X > 0).mean()
    print(f'razão fundo/memória {r:.2f} · neurônios estáveis na própria memória {est:.3f}')
    S0 = X.copy(); S0[:, rng.choice(610, 244, replace=False)] *= -1
    for nome, kw in (('binária', None), ('previsor', dict(usar_previsor=True)), ('batedor', dict(usar_previsor=False, usar_batedor=True)), ('as duas vias', dict(usar_previsor=True, usar_batedor=True))):
        S = recuperar_binaria(J, S0, np.random.default_rng(1)) if kw is None else recuperar(J, X, S0, np.random.default_rng(1), **kw)
        print(f'  {nome:<13} conclusão {conclusao(S, X):.3f}')
    print('AUTOVERIFICAÇÃO:', 'OK' if (r < 0.3 and est > 0.85) else 'FALHOU')
