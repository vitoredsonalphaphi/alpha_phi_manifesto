# PENDENTE DE REGISTRO — ver README desta pasta
# EXPERIMENTO: ETAPA B — o EcoBIP 880 original como tarefa comum à rede convencional (Hopfield) e à rede AlphaFi esférica
# (critérios aprovados pelo pesquisador em 20/10/2026, antes da execução; controle corrigido e declarado antes de rodar)
# Memórias: as 5 dobras da cascata do EcoBIP 880 (BEEP880_17S.py original, campo harmônico no ciclo 10, β máx 4,2098).
# Tradução para a esfera: energia em cada banda φ do script (15) → peso da forma de grau L = índice da banda; padrão = sinal(Σ √e_b · u_b),
# u_b = combinação aleatória fixa (por semente) de harmônicos reais de grau b. Mesma tradução para todos os sinais.
# Convencional: Hopfield, todos ligados. AlphaFi: 233 pontos cultivados, vizinhança σ=0,35 ∘ Hebb + 0,02 × fundo do campo.
# Controle aprovado (A): onda quadrada 880 Hz sem cascata, 5 trechos seguidos. Controle estabelecido (B, informativo): a mesma cascata,
# com o β final do EcoBIP, aplicada só à quadrada.
# B1: recuperação (pista 25%) AlphaFi difere da convencional (|dif| > 2 EP e mesmo sinal em >= 8/10).
# B2: (AlphaFi − convencional) com o EcoBIP > o mesmo com o controle A (> 2 EP e >= 8/10). Sementes 60–69.
import numpy as np, importlib.util, os, json, pickle
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, '..')  # rodar antes AlphaPhi_EcoBIP_B0_gera_dobras.py
spec = importlib.util.spec_from_file_location('E', os.path.join(R, 'AlphaPhi_CampoEsferico_EtapaE_Ferramentas.py'))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E); base, t_ = E.base, E.t
N, CV, FUNDO, P = E.N, E.CV, E.FUNDO, E.P
PHI = (1 + 5 ** .5) / 2; FS = 44100; SEM = list(range(60, 70)); MAXV = 50
Y, grau = t_.harmonicos(P, 14)
BANDAS = pickle.load(open(os.path.join(H, 'ecobip_meta.pkl'), 'rb'))['BANDAS']

def energias(sig):
    F = np.abs(np.fft.rfft(sig)) ** 2; f = np.fft.rfftfreq(len(sig), 1 / FS)
    e = np.array([F[(f >= a) & (f < b)].sum() for a, b in BANDAS]); return e / e.sum()
def bases(rng):
    U = []
    for L in range(len(BANDAS)):
        v = Y[:, grau == L] @ rng.normal(size=(grau == L).sum()); U.append(v / np.linalg.norm(v))
    return np.array(U)
def padroes(sinais, U):
    # CORREÇÃO aprovada (20/10): traduzir a diferença — log da energia por banda menos o perfil médio das 5 memórias do mesmo sinal
    Lg = np.array([np.log(energias(s) + 1e-6) for s in sinais]); Z = Lg - Lg.mean(0)
    return np.array([np.where(z @ U >= 0, 1., -1.) for z in Z])
def redes(X):
    Hb = (X.T @ X) / N
    Jc = Hb.copy(); np.fill_diagonal(Jc, 0)
    Ja = CV * Hb + .02 * FUNDO; np.fill_diagonal(Ja, 0); return {'convencional': Jc, 'AlphaFi esférica': Ja}
def mede(J, X, s):
    out = {}
    for nome, flip in (('pres', 0.), ('r25', .25), ('r40', .40)):
        rng = np.random.default_rng([800_000 + s, int(flip * 100)]); S0 = X.copy(); k = int(round(flip * N))
        if k:
            for p in range(len(S0)): S0[p, rng.choice(N, k, replace=False)] *= -1
        base.N = N; m, _, Sf = base.recupera(J, X, flip, rng) if k else (None, None, None)
        if not k:
            S = X.copy(); r2 = np.random.default_rng([810_000 + s])
            for v in range(MAXV):
                mud = 0
                for i in r2.permutation(N):
                    h = S @ J[:, i]; novo = np.where(h > 0, 1., np.where(h < 0, -1., S[:, i])); mud += int((novo != S[:, i]).sum()); S[:, i] = novo
                if mud == 0: break
            Sf = S; m = (S * X).mean(1)
        out[nome] = float(m.mean())
        if nome == 'r25': out['ident'] = float(np.mean(np.argmax(Sf @ X.T, 1) == np.arange(len(X))))
    E_ = (Sf @ Y) ** 2; eL = np.array([E_[:, grau == L].sum(1) for L in range(15)]).T; pL = eL / eL.sum(1, keepdims=True) + 1e-12
    out['coh'] = float((1 - (-(pL * np.log(pL)).sum(1)) / np.log(15)).mean()); return out

if __name__ == '__main__':
    dobras = np.load(os.path.join(H, 'ecobip_dobras.npy')); n = dobras.shape[1]; tt = np.arange(n) / FS
    quad = np.sign(np.sin(2 * np.pi * 880 * tt)); ctrlA = [quad[i * n // 5:(i + 1) * n // 5] for i in range(5)]
    ctrlB = np.load(os.path.join(H, 'ctrl_quadrada_cascata.npy'))
    SINAIS = {'EcoBIP 880': list(dobras), 'controle (quadrada com cascata)': list(ctrlB)}   # controle A retirado (degenerado)
    print('B00 correlação média entre as memórias (média das 10 sementes; precisa < 0,3):')
    for nome, sig in SINAIS.items():
        cs = [((lambda X: (X @ X.T / N)[np.triu_indices(5, 1)].mean())(padroes(sig, bases(np.random.default_rng([900_000 + s]))))) for s in SEM]
        print(f'   {nome:<34} {np.mean(cs):+.3f} -> {"cumprido" if np.mean(cs) < .3 else "NÃO cumprido — sem tarefa de memória"}')
    res = {k: {'convencional': [], 'AlphaFi esférica': []} for k in SINAIS}
    for s in SEM:
        U = bases(np.random.default_rng([900_000 + s]))
        for nome, sig in SINAIS.items():
            X = padroes(sig, U)
            for rn, J in redes(X).items(): res[nome][rn].append(mede(J, X, s))
        print(f'  semente {s} pronta', flush=True)
    print(f"\n{'sinal':<36}{'rede':<18}{'preserv.':>9}{'recup.25%':>10}{'recup.40%':>10}{'identif.':>9}{'Coh':>7}")
    for nome in SINAIS:
        for rn in ('convencional', 'AlphaFi esférica'):
            L = res[nome][rn]; f = lambda k: np.mean([x[k] for x in L])
            print(f"{nome:<36}{rn:<18}{f('pres'):>9.3f}{f('r25'):>10.3f}{f('r40'):>10.3f}{f('ident'):>9.2f}{f('coh'):>7.3f}")
    col = lambda nome, rn, k='r25': np.array([x[k] for x in res[nome][rn]])
    d_eco = col('EcoBIP 880', 'AlphaFi esférica') - col('EcoBIP 880', 'convencional'); se = d_eco.std(ddof=1) / np.sqrt(10); ns = max((d_eco > 0).sum(), (d_eco < 0).sum())
    print('\n=== CRITÉRIOS ===\nB0 código original: campo harmônico no ciclo 10, β máx 4,2098 -> cumprido (impresso pelo script original)')
    print(f"B1 AlphaFi − convencional (EcoBIP, pista 25%): {d_eco.mean():+.4f} (EP {se:.4f}, mesmo sinal {ns}/10) -> {'difere' if abs(d_eco.mean()) > 2 * se and ns >= 8 else 'não difere'}")
    for c in ('controle (quadrada com cascata)',):
        d_c = col(c, 'AlphaFi esférica') - col(c, 'convencional'); dd = d_eco - d_c; se2 = dd.std(ddof=1) / np.sqrt(10)
        ok = dd.mean() > 2 * se2 and (dd > 0).sum() >= 8
        rot = 'B2'
        print(f"{rot} vantagem com EcoBIP − vantagem com {c}: {dd.mean():+.4f} (EP {se2:.4f}, {(dd>0).sum()}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    json.dump(res, open(os.path.join(H, 'exp_etapaB2.json'), 'w'))
