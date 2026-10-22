# AlphaPhi — Grade Esférica v2 (PROPOSTA, NÃO CANÔNICA; usada na Etapa VL2; o fundamento ainda está em construição, Entradas 346–355) · 22/10/2026 · Vitor Edson Delavi · Claude
# A estrutura que antecede as ferramentas (Entrada 346): dá ENDEREÇO a cada dado na esfera.
# Base: a Grade Esférica v1 (pontos de Fibonacci encaixados, harmônicos), sem alteração.
#   · células: os 610 pontos; 75 anéis de latitude (do polo norte ao sul) × 8 pontos por anel = 600 células; 10 pontos neutros
#   · a FREQUÊNCIA ganha latitude: anel k = faixa logarítmica k de 75, de 40 Hz a 20 kHz (grave no norte, agudo no sul)
#   · o TEMPO ganha longitude: dentro do anel, os pontos ordenados em volta da esfera = os 8 instantes de 0,125 s do trecho de 1 s
#   · cada célula guarda a ALTURA da faixa naquele instante (valor contínuo), e não só o sinal
# Precisão: com os neurônios na superfície, a frequência vai na latitude, não no raio; com o eixo radial, poderá ir para o raio.
import numpy as np, importlib.util, os
from scipy.signal import stft, istft
_g = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'AlphaPhi_GradeEsferica_v1.py')
_s = importlib.util.spec_from_file_location('G1', _g); G1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(G1)
N = 610; NFX = 75; NT = 8; FS = 44100; SEG = FS          # trecho de 1 s
NPERSEG = 2048; HOP = 512
P = G1.pontos_encaixados(N)
def _enderecos():
    ordem = np.argsort(-P[:, 2], kind='stable'); end = np.full((NFX, NT), -1)
    for k in range(NFX):
        anel = ordem[k * NT:(k + 1) * NT]; lon = np.arctan2(P[anel, 1], P[anel, 0]); end[k] = anel[np.argsort(lon)]
    return end
END = _enderecos()                                         # END[faixa, instante] = índice do ponto
NEUTROS = np.setdiff1d(np.arange(N), END.ravel())
FAIXAS = np.geomspace(40, 20000, NFX + 1)
_f = np.fft.rfftfreq(NPERSEG, 1 / FS)
BANDAS = [np.where((_f >= a) & (_f < b))[0] for a, b in zip(FAIXAS[:-1], FAIXAS[1:])]
VAZIA = [k for k in range(NFX) if len(BANDAS[k]) == 0]
for k in VAZIA: BANDAS[k] = np.array([int(np.argmin(abs(_f - np.sqrt(FAIXAS[k] * FAIXAS[k + 1]))))])   # faixa vazia recebe o ponto mais próximo (compartilhado)
def espectro(seg):
    """STFT do trecho (Hann, 2048, passo 512)."""
    return stft(seg, FS, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary='even', padded=True)
def instante_do_quadro(t):
    return np.clip((t / (SEG / FS) * NT).astype(int), 0, NT - 1)
def energias(seg):
    """Energia de cada faixa em cada instante: matriz 75 × 8."""
    _, t, Z = espectro(seg); E = np.zeros((NFX, NT)); inst = instante_do_quadro(t); A = np.abs(Z) ** 2
    for k in range(NFX):
        for j in range(NT): E[k, j] = A[BANDAS[k]][:, inst == j].sum()
    return E
def na_esfera(M):
    """Matriz 75 × 8 → vetor de 610 neurônios (neutros = 0)."""
    v = np.zeros(N); v[END.ravel()] = M.ravel(); return v
def da_esfera(v):
    return v[END.ravel()].reshape(NFX, NT)
def aplicar_ganho(seg, G):
    """Ganho por faixa e instante (75 × 8) aplicado ao próprio som (a fase e o detalhe fino são os do som)."""
    _, t, Z = espectro(seg); inst = instante_do_quadro(t); g = np.zeros(Z.shape); n = np.zeros(Z.shape[0])
    for k in range(NFX): g[BANDAS[k]] += G[k][inst][None, :]; n[BANDAS[k]] += 1
    fora = n == 0; g[~fora] /= n[~fora][:, None]; g[fora] = 1.0     # fora de 40 Hz–20 kHz: sem alteração
    _, y = istft(Z * g, FS, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=True); return y[:len(seg)]
if __name__ == '__main__':
    print('células:', END.size, '· neutros:', len(NEUTROS), '· faixas vazias preenchidas pelo vizinho:', len(VAZIA))
    lat = np.degrees(np.arccos(P[END[:, 0], 2])); print(f'latitude do anel 0 (grave): {90 - lat[0]:.1f}° · anel 74 (agudo): {90 - lat[-1]:.1f}°')
    x = np.random.default_rng(0).normal(size=SEG); y = aplicar_ganho(x, np.ones((NFX, NT))); print('ganho 1 devolve o mesmo som:', bool(np.allclose(x, y, atol=1e-6)))
    print('endereços únicos:', len(np.unique(END)) == END.size)
