# PENDENTE DE REGISTRO — ver README desta pasta
# EXPERIMENTO: ETAPA P — o Phantom (serial sem selagem, código original) como fundo do campo esférico cultivado
# (critérios escritos na conversa em 19/10/2026, antes da execução)
# Tradução: energia do sinal em cada banda φ do próprio script (grave → agudo) = peso do grau esférico L = 0, 1, 2, ...
# Núcleo K(θ) = Σ_L e_L · P_L(cos θ); fundo B_ij = K(θ_ij), diagonal zero; todos os fundos com a mesma energia (norma de Frobenius)
# de 0,02 × fundo da Etapa E. Controles: nenhum, Phantom embaralhado (ordem das bandas), ruído branco, seno 880 Hz, fundo da Etapa E.
# Ferramentas da Etapa E (4 × 4 padrões, ℓ = 0,3/0,5/0,7/0,9), todas instaladas; sementes 40–49.
# P1: Phantom > nenhum; P2: Phantom > embaralhado; P3: Phantom > ruído E > seno (estabilidade média das 4 ferramentas; pareado > 2 EP e >= 8/10).
# P4: Phantom × fundo da Etapa E (descritivo). P5: previsor × estabilidade em todas as condições: Spearman >= 0,8 E erro <= 0,05.
import numpy as np, importlib.util, os, json
from scipy.special import eval_legendre
from scipy.stats import norm, spearmanr
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, '..')
spec = importlib.util.spec_from_file_location('E', os.path.join(R, 'AlphaPhi_CampoEsferico_EtapaE_Ferramentas.py'))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E); base = E.base
N, CV, FUNDO, ELLS, NPAT, P = E.N, E.CV, E.FUNDO, E.ELLS, E.NPAT, E.P
SEM = list(range(40, 50)); FS = 44100

src = open(os.path.join(R, '..', 'AlphaPhi_SerialPhantom_SemSelagem_COLAB.py')).read().split('\n')
g = {}; exec('\n'.join(src[:src.index(next(l for l in src if l.startswith('BANDAS')))+1]), g)   # só constantes e bandas φ do script original
BANDAS = g['BANDAS']
def energias(sig):
    F = np.abs(np.fft.rfft(sig)) ** 2; f = np.fft.rfftfreq(len(sig), 1 / FS)
    return np.array([F[(f >= a) & (f < b)].sum() for a, b in BANDAS])
cosd = np.clip(P @ P.T, -1, 1)
def fundo_de(e):
    e = e / e.sum(); K = sum(e[L] * eval_legendre(L, cosd) for L in range(len(e))); np.fill_diagonal(K, 0); return K
alvo = np.linalg.norm(.02 * FUNDO)
def normaliza(B): return B * alvo / np.linalg.norm(B)


def _serial_phantom(pasta):
    """Gera (uma vez) o serial sem selagem com o código ORIGINAL, sem alterá-lo, e guarda em .npy."""
    import numpy as _np
    arq = os.path.join(pasta, 'serial_phantom_sem_selagem.npy')
    if not os.path.exists(arq):
        src = open(os.path.join(pasta, '..', '..', 'AlphaPhi_SerialPhantom_SemSelagem_COLAB.py')).read().split('\n')
        corte = next(i for i, l in enumerate(src) if l.startswith('def scanner_agnostico'))
        g = {'__name__': 'phantom'}; exec('\n'.join(src[:corte]), g); _np.save(arq, g['SERIAL'])
    return arq

if __name__ == '__main__':
    serial = np.load(_serial_phantom(H))
    t = np.arange(len(serial)) / FS
    e_ph = energias(serial); e_ru = energias(np.random.default_rng(1).normal(size=len(serial))); e_se = energias(np.sin(2 * np.pi * 880 * t))
    perm = np.random.default_rng(2).permutation(len(e_ph))
    print(f'bandas φ do script: {len(BANDAS)} · energia relativa do Phantom por banda (grave → agudo):')
    print('   ' + ' '.join(f'{x:.3f}' for x in e_ph / e_ph.sum()))
    FUNDOS = {'nenhum': np.zeros((N, N)), 'Phantom': normaliza(fundo_de(e_ph)), 'Phantom embaralhado': normaliza(fundo_de(e_ph[perm])),
              'ruído branco': normaliza(fundo_de(e_ru)), 'seno 880 Hz': normaliza(fundo_de(e_se)), 'campo da Etapa E': .02 * FUNDO}
    res = {k: [] for k in FUNDOS}; pts = []
    for s in SEM:
        Gs = []
        for k, ell in enumerate(ELLS):
            A = E.suave(np.random.default_rng([520_000 + s, k]), 200, ell); Gs.append((A.T @ A) / 200)
        X = [E.suave(np.random.default_rng([620_000 + s, k]), NPAT, ell) for k, ell in enumerate(ELLS)]
        todos = np.vstack(X); dono = np.repeat(np.arange(4), NPAT)
        for nome, B in FUNDOS.items():
            J = CV * (todos.T @ todos) / N + B; np.fill_diagonal(J, 0)
            est = ((todos @ J) * todos > 0).mean(1)
            m, _, _ = base.recupera(J, todos, .25, np.random.default_rng([720_000 + s, len(nome)]))
            pv = []
            for v in range(len(todos)):
                xi = todos[v]; W = CV * xi[None, :]; mu = CV.sum(1) + N * xi * (B @ xi); var = np.zeros(N)
                for u in range(len(todos)):
                    if u == v: continue
                    G_ = Gs[dono[u]]; lin = (W * G_).sum(1); mu = mu + xi * lin; var += np.maximum(((W @ G_) * W).sum(1) - lin ** 2, 0)
                pv.append(norm.cdf((mu / N) / np.sqrt(np.maximum(var, 1e-12) / N ** 2)).mean())
            pv = np.array(pv)
            res[nome].append(dict(est=float(est.mean()), rec=float(m.mean()), por_ferr=[float(est[dono == k].mean()) for k in range(4)]))
            for k in range(4): pts.append((nome, k, s, float(pv[dono == k].mean()), float(est[dono == k].mean())))
        print(f'  semente {s} pronta', flush=True)
    print(f"\n{'fundo':<22}{'estabilidade':>13}{'recuperação':>13}   por ferramenta (ℓ = 0,3 / 0,5 / 0,7 / 0,9)")
    for nome, L in res.items():
        pf = np.mean([r['por_ferr'] for r in L], 0)
        print(f"{nome:<22}{np.mean([r['est'] for r in L]):>13.4f}{np.mean([r['rec'] for r in L]):>13.4f}   " + ' / '.join(f'{x:.3f}' for x in pf))
    col = lambda n: [r['est'] for r in res[n]]
    print('\n=== CRITÉRIOS ===')
    for k, b in (('P1', 'nenhum'), ('P2', 'Phantom embaralhado'), ('P3', 'ruído branco'), ('P3', 'seno 880 Hz'), ('P4 (descritivo)', 'campo da Etapa E')):
        d, se, nf, ok = base.pareado(col('Phantom'), col(b))
        print(f"{k:<16} Phantom − {b:<20}: {d:+.4f} (EP {se:.4f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    chave = sorted(set((p[0], p[1]) for p in pts)); M = np.array([[np.mean([p[3] for p in pts if (p[0], p[1]) == c]), np.mean([p[4] for p in pts if (p[0], p[1]) == c])] for c in chave])
    rho, mae = spearmanr(M[:, 0], M[:, 1])[0], float(np.abs(M[:, 0] - M[:, 1]).mean())
    print(f"P5 previsor × estabilidade ({len(chave)} combinações): Spearman {rho:.3f}, erro {mae:.4f} -> {'cumprido' if rho >= .8 and mae <= .05 else 'NÃO cumprido'}")
    json.dump({'res': res, 'pts': pts, 'e_ph': e_ph.tolist()}, open(os.path.join(H, 'exp_etapaP.json'), 'w'))
