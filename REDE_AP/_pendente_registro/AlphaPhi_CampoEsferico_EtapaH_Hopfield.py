# PENDENTE DE REGISTRO — ver README desta pasta
# EXPERIMENTO: ETAPA H — Hopfield moderna (o eco do dado) sobre a Grade Esférica de Referência v1
# (critérios aprovados pelo pesquisador em 20/10/2026, antes da execução). Versão aprovada: 233 pontos; versão declarada antes de rodar,
# pela regra da própria grade (225 formas pedem >= 450 pontos): 610 pontos, com a escala do fundo ajustada ao espaçamento (Etapa C).
# Memórias: EcoBIP 880 original — entrada + 5 dobras (6). Tradução: 225 faixas log de 20 Hz a 20 kHz → 225 harmônicos (ordem por grau);
# padrão = sinal(Y · z), z = log da energia − perfil médio das 6. Sem sorteio na tradução. Controle: quadrada + a mesma cascata (6).
# Eco do dado: a cada volta, pesos w = softmax(β · m_μ) com m_μ = sobreposição do estado ATUAL com cada memória; β = 8;
# campo somado = λ · Σ w_μ ξ_μ, λ = 0,5 × escala do sinal de memória da rede. Condições: C, A, C+eco, A+eco, A+eco errado
# (pesos numa memória sorteada ≠ alvo), eco puro (sem ligações). Pistas 25% e 40%. Sementes 80–89.
# H00: correlação média entre memórias < 0,3. H1: A+eco > A. H2: A+eco > A+eco errado. H3: A+eco difere de C+eco. H4: A+eco > eco puro.
# (identificação com pista de 40%; pareado > 2 EP e >= 8/10; H3 bilateral.)
import numpy as np, importlib.util, os, json, sys
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, '..')  # rodar antes AlphaPhi_EcoBIP_B0_gera_dobras.py
spec = importlib.util.spec_from_file_location('G', os.path.join(R, 'AlphaPhi_GradeEsferica_v1.py')); G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
FS = 44100; BETA = 8.0; LAMB = 0.5; MAXV = 50; SEM = list(range(80, 90)); NB = 225
FAIXAS = np.geomspace(20, 20000, NB + 1)

def energias(sig):
    F = np.abs(np.fft.rfft(sig)) ** 2; f = np.fft.rfftfreq(len(sig), 1 / FS)
    return np.array([F[(f >= a) & (f < b)].sum() for a, b in zip(FAIXAS[:-1], FAIXAS[1:])]) / F.sum()
def montar(N):
    P = G.pontos_encaixados(N); Y, grau = G.harmonicos(P, 14); D = G.distancia_geodesica(P) / np.pi
    CV = np.exp(-(D / .35) ** 2); np.fill_diagonal(CV, 0)
    se = .15 * np.sqrt(233 / N); FU = np.exp(-(D / se) ** 2) - .5 * np.exp(-(D / (3 * se)) ** 2); np.fill_diagonal(FU, 0)
    return P, Y, CV, FU
def padroes(sinais, Y):
    Lg = np.array([np.log(energias(s) + 1e-6) for s in sinais]); Z = Lg - Lg.mean(0)
    return np.array([np.where(Y @ z >= 0, 1., -1.) for z in Z])
def dinamica(J, X, S0, rng, eco=None, alvo_errado=None, lam=0.):
    N = S0.shape[1]; S = S0.copy()
    for v in range(MAXV):
        if eco:
            m = S @ X.T / N; w = np.exp(BETA * (m - m.max(1, keepdims=True))); w /= w.sum(1, keepdims=True)   # softmax(β·m), m ∈ [−1, 1]
            if alvo_errado is not None: w = np.eye(len(X))[alvo_errado]
            Gc = lam * (w @ X)
        mud = 0
        for i in rng.permutation(N):
            h = (S @ J[:, i] if J is not None else 0.) + (Gc[:, i] if eco else 0.)
            novo = np.where(h > 0, 1., np.where(h < 0, -1., S[:, i])); mud += int((novo != S[:, i]).sum()); S[:, i] = novo
        if mud == 0 and v > 0: break
    return S
def rodar(N, sinais_dict):
    P, Y, CV, FU = montar(N); res = {}
    for nome, sinais in sinais_dict.items():
        X = padroes(sinais, Y); cor = (X @ X.T / N)[np.triu_indices(len(X), 1)].mean()
        Hb = X.T @ X / N; Jc = Hb.copy(); np.fill_diagonal(Jc, 0); Ja = CV * Hb + .02 * FU; np.fill_diagonal(Ja, 0)
        esc_c = (N - 1) / N; esc_a = CV.sum(1).mean() / N
        conds = {'C': (Jc, False, 0, None), 'A': (Ja, False, 0, None), 'C+eco': (Jc, True, LAMB * esc_c, None),
                 'A+eco': (Ja, True, LAMB * esc_a, None), 'A+eco errado': (Ja, True, LAMB * esc_a, 'errado'), 'eco puro': (None, True, 1., None)}
        r = {c: {'id25': [], 'id40': [], 'm40': []} for c in conds}
        for s in SEM:
            for flip in (.25, .40):
                rng = np.random.default_rng([1_000_000 + s, int(flip * 100), N]); S0 = X.copy(); k = int(round(flip * N))
                for p in range(len(S0)): S0[p, rng.choice(N, k, replace=False)] *= -1
                errado = np.array([(p + 1 + rng.integers(len(X) - 1)) % len(X) for p in range(len(X))])
                for c, (J, eco, lam, modo) in conds.items():
                    S = dinamica(J, X, S0, np.random.default_rng([1_100_000 + s, int(flip * 100), N]), eco, errado if modo else None, lam)
                    ide = float(np.mean(np.argmax(S @ X.T, 1) == np.arange(len(X))))
                    r[c]['id25' if flip == .25 else 'id40'].append(ide)
                    if flip == .40: r[c]['m40'].append(float((S * X).mean(1).mean()))
        res[nome] = {'cor': float(cor), 'r': r}
    return res

if __name__ == '__main__':
    dob = np.load(os.path.join(H, 'ecobip_dobras.npy')); ent = np.load(os.path.join(H, 'ecobip_entrada.npy'))  # ecobip_entrada.npy: o sinal antes da cascata (gerado junto)
    ctrl = np.load(os.path.join(H, 'ctrl_quadrada_cascata.npy')); n = ctrl.shape[1]; tt = np.linspace(0, 7.0, n, endpoint=False)
    quad = np.sign(np.sin(2 * np.pi * 880 * tt))
    SIN = {'EcoBIP 880': [ent] + list(dob), 'controle (quadrada com cascata)': [quad] + list(ctrl)}
    tudo = {}
    for N in (233, 610):
        print(f'\n######## {N} pontos (grau medível até {G.lmax_mensuravel(N)}; {"versão aprovada" if N == 233 else "versão pela regra da grade"}) ########', flush=True)
        res = rodar(N, SIN); tudo[N] = res
        for nome, d in res.items():
            print(f"{nome}: H00 correlação entre memórias {d['cor']:+.3f} -> {'cumprido' if d['cor'] < .3 else 'NÃO cumprido'}")
            print(f"   {'condição':<14}{'ident. 25%':>11}{'ident. 40%':>11}{'sobrep. 40%':>12}")
            for c, v in d['r'].items(): print(f"   {c:<14}{np.mean(v['id25']):>11.3f}{np.mean(v['id40']):>11.3f}{np.mean(v['m40']):>12.3f}")
        r = res['EcoBIP 880']['r']; col = lambda c: np.array(r[c]['id40'])
        def par(a, b, bil=False):
            d = col(a) - col(b); se = d.std(ddof=1) / np.sqrt(len(d)) if d.std() > 0 else 0.
            nf = max((d > 0).sum(), (d < 0).sum()) if bil else (d > 0).sum()
            ok = (abs(d.mean()) if bil else d.mean()) > 2 * se and nf >= 8 and (abs(d.mean()) > 0 if bil else d.mean() > 0)
            return d.mean(), se, nf, ok
        print('   === CRITÉRIOS (EcoBIP, identificação com pista 40%) ===')
        for k, a, b, bil in (('H1', 'A+eco', 'A', False), ('H2', 'A+eco', 'A+eco errado', False), ('H3', 'A+eco', 'C+eco', True), ('H4', 'A+eco', 'eco puro', False)):
            dm, se, nf, ok = par(a, b, bil)
            print(f"   {k} {a} − {b}: {dm:+.3f} (EP {se:.3f}, {nf}/10) -> {('difere' if ok else 'não difere') if bil else ('cumprido' if ok else 'NÃO cumprido')}")
        rc = res['controle (quadrada com cascata)']['r']; dd = np.array(r['A+eco']['id40']) - np.array(r['C+eco']['id40']); dc = np.array(rc['A+eco']['id40']) - np.array(rc['C+eco']['id40'])
        print(f"   (descritivo) A+eco − C+eco: EcoBIP {dd.mean():+.3f} | controle {dc.mean():+.3f}")
    json.dump({str(k): v for k, v in tudo.items()}, open(os.path.join(H, 'exp_etapaH.json'), 'w'))
