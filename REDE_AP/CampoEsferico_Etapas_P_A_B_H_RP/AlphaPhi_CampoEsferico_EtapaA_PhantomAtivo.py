# PENDENTE DE REGISTRO — ver README desta pasta
# EXPERIMENTO: ETAPA A — Phantom desligado, inscrito e ATIVO (sinal de áudio tocando durante a dinâmica)
# (critérios escritos na conversa em 19/10/2026, antes da execução)
# Rede: 233 pontos cultivados, fundo do campo (0,02 × ligação da Etapa E), 4 ferramentas da Etapa E instaladas. Sementes 50–59.
# Ativo: serial sem selagem (código original). Cada banda φ do script dirige no tempo uma forma esférica do grau L = índice da banda
# (combinação aleatória fixa de harmônicos reais de grau L). h_ext(k) = γ · Σ_b s_b(t_k) · u_b, com t_k percorrendo o serial ao longo
# das atualizações. γ: RMS de h_ext = (1/φ³) × escala do sinal da memória (média de Σ_j C_ij / N). Controles ativos: ruído branco e seno 880 Hz.
# Inscrito: núcleo espectral do Phantom (Etapa P) como ligação fixa, mesma energia de 0,02 × fundo.
# Medidas: preservação (parte da memória, roda com h_ext); recuperação com pistas de 25% e 40%.
# A1: recuperação 40% ativo > desligado. A2: ativo > ruído ativo. A3: ativo > seno ativo. A4: preservação ativo >= desligado − 0,01.
# Regra: diferença pareada > 2 EP e >= 8/10 (A4: média da diferença >= −0,01).
import numpy as np, importlib.util, os, json
from scipy.special import eval_legendre
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, '..')
spec = importlib.util.spec_from_file_location('E', os.path.join(R, 'AlphaPhi_CampoEsferico_EtapaE_Ferramentas.py'))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E); t_ = E.t
N, CV, FUNDO, ELLS, NPAT, P = E.N, E.CV, E.FUNDO, E.ELLS, E.NPAT, E.P
PHI = (1 + 5 ** .5) / 2; FS = 44100; SEM = list(range(50, 60)); MAXV = 50; K = MAXV * N
src = open(os.path.join(R, '..', 'AlphaPhi_SerialPhantom_SemSelagem_COLAB.py')).read().split('\n')
g = {}; exec('\n'.join(src[:src.index(next(l for l in src if l.startswith('BANDAS'))) + 1]), g); BANDAS = g['BANDAS']
cosd = np.clip(P @ P.T, -1, 1); Y, grau = t_.harmonicos(P, 14)

def bandas_no_tempo(sig):
    F = np.fft.rfft(sig); f = np.fft.rfftfreq(len(sig), 1 / FS); idx = np.linspace(0, len(sig) - 1, K).astype(int); out = []
    for a, b in BANDAS:
        Fb = np.where((f >= a) & (f < b), F, 0); out.append(np.fft.irfft(Fb, len(sig))[idx])
    return np.array(out)                                   # (15 bandas, K instantes)
def nucleo_inscrito(sig):
    Fp = np.abs(np.fft.rfft(sig)) ** 2; f = np.fft.rfftfreq(len(sig), 1 / FS)
    e = np.array([Fp[(f >= a) & (f < b)].sum() for a, b in BANDAS]); e /= e.sum()
    Kn = sum(e[L] * eval_legendre(L, cosd) for L in range(len(e))); np.fill_diagonal(Kn, 0); return Kn * np.linalg.norm(.02 * FUNDO) / np.linalg.norm(Kn)
def campo_externo(sb, rng):
    u = []
    for L in range(len(sb)):
        c = rng.normal(size=(grau == L).sum()); v = Y[:, grau == L] @ c; u.append(v / np.linalg.norm(v) * np.sqrt(N))
    Hx = np.array(u).T @ sb                                 # (N, K)
    escala = np.mean(CV.sum(1)) / N; return Hx * (escala / PHI ** 3) / np.sqrt(np.mean(Hx ** 2))
def dinamica(J, S0, rng, Hx=None, registra=None):
    S = S0.copy(); k = 0; reg_h, reg_m = [], []
    for v in range(MAXV):
        for i in rng.permutation(N):
            h = S @ J[:, i] + (Hx[i, k] if Hx is not None else 0.)
            S[:, i] = np.where(h > 0, 1., np.where(h < 0, -1., S[:, i]))
            if registra is not None: reg_h.append(float(h[registra[0]] if np.ndim(h) else h)); reg_m.append(float((S[registra[0]] * registra[1]).mean()))
            k += 1
    return S, reg_h, reg_m


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
    serial = np.load(_serial_phantom(H)); tt = np.arange(len(serial)) / FS
    SB = {'Phantom': bandas_no_tempo(serial), 'ruído': bandas_no_tempo(np.random.default_rng(3).normal(size=len(serial))),
          'seno 880': bandas_no_tempo(np.sin(2 * np.pi * 880 * tt))}
    KIN = nucleo_inscrito(serial)
    conds = ['desligado', 'inscrito', 'Phantom ativo', 'ruído ativo', 'seno 880 ativo']
    res = {c: {'pres': [], 'r25': [], 'r40': []} for c in conds}; topo = {}
    for s in SEM:
        X = [E.suave(np.random.default_rng([630_000 + s, k]), NPAT, ell) for k, ell in enumerate(ELLS)]; todos = np.vstack(X)
        J0 = CV * (todos.T @ todos) / N + .02 * FUNDO; np.fill_diagonal(J0, 0)
        for c in conds:
            J = J0 + (KIN if c == 'inscrito' else 0); Hx = None
            if c.endswith('ativo'): Hx = campo_externo(SB[c.replace(' ativo', '')], np.random.default_rng([640_000 + s]))
            for med, flip in (('pres', 0.), ('r25', .25), ('r40', .40)):
                rng = np.random.default_rng([740_000 + s, int(flip * 100)]); S0 = todos.copy()
                kf = int(round(flip * N))
                for p in range(len(S0)): S0[p, rng.choice(N, kf, replace=False)] *= -1 if kf else 1
                reg = (0, todos[0]) if (s == SEM[0] and med == 'r40') else None
                S, rh, rm = dinamica(J, S0, np.random.default_rng([750_000 + s, int(flip * 100)]), Hx, reg)
                res[c][med].append(float((S * todos).mean(1).mean()))
                if reg is not None: topo[c] = {'h': rh, 'm': rm}
        print(f'  semente {s} pronta', flush=True)
    print(f"\n{'estado do Phantom':<18}{'preservação':>13}{'recup. 25%':>12}{'recup. 40%':>12}")
    for c in conds: print(f"{c:<18}" + ''.join(f"{np.mean(res[c][m]):>12.4f} " for m in ('pres', 'r25', 'r40')))
    E_ = E.base
    print('\n=== CRITÉRIOS ===')
    for k_, a, b in (('A1', 'Phantom ativo', 'desligado'), ('A2', 'Phantom ativo', 'ruído ativo'), ('A3', 'Phantom ativo', 'seno 880 ativo')):
        d, se, nf, ok = E_.pareado(res[a]['r40'], res[b]['r40']); print(f"{k_} recuperação 40%: {a} − {b}: {d:+.4f} (EP {se:.4f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    d = np.array(res['Phantom ativo']['pres']) - np.array(res['desligado']['pres'])
    print(f"A4 preservação: ativo − desligado = {d.mean():+.4f} (>= −0,01) -> {'cumprido' if d.mean() >= -.01 else 'NÃO cumprido'}")
    d, se, nf, ok = E_.pareado(res['inscrito']['r40'], res['desligado']['r40']); print(f"(descritivo) inscrito − desligado, recuperação 40%: {d:+.4f} (EP {se:.4f}, {nf}/10)")
    json.dump({'res': res, 'topo': topo}, open(os.path.join(H, 'exp_etapaA.json'), 'w'))
