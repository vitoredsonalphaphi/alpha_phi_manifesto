# EXPERIMENTO: ETAPA S — memórias feitas de quadros do STFT (critérios aprovados pelo pesquisador em 20/10/2026, antes da execução)
# Sinais: EcoBIP 880 original (entrada + 5 dobras, 7 s cada); controle: quadrada + a mesma cascata. STFT: janela Hann de 1 s, passo 0,25 s.
# Tradução: 225 faixas log de 40 Hz a 20 kHz → 225 harmônicos (grau até 14); faixa sem ponto do espectro recebe o ponto mais próximo
# do centro geométrico; log da energia (+1e-6) menos o perfil médio dos candidatos; padrão = sinal(Y · z).
# Escolha: 12 memórias, uma a uma, o candidato com a menor correlação máxima (em valor absoluto) com as já escolhidas; a primeira é o
# quadro inicial da entrada. Esfera principal: 610 pontos (Grade Esférica v1); vizinhança = 2 × espaçamento (σ 0,091 em 610; 0,148 em 233).
# Condições: C610, A610, A610-H (σ 0,35), A610+eco e C610+eco (λ = 0,5 / 2 / 8; principal λ = 2), eco puro, A233, A610-reproj, A610-zero.
# Pistas 25% e 40%. Sementes 100–109. Critérios (identificação, pista 40%; pareado > 2 EP e >= 8/10):
# S00: correlação média < 0,3 e máxima < 0,5 (610 pontos). S01: C610 identifica >= 0,8 com pista 25%. (Se falharem, para sem conclusão.)
# S1: A610 difere de C610 (bilateral). S2: A610 > A610-H. S3: A610+eco(2) > A610. S4: A610+eco(2) > eco puro.
# S5: A610-reproj > A233. S6: A610-reproj >= A610-zero − 0,05 (média). Descritivo: S1 no controle; λ 0,5 e 8.
import numpy as np, importlib.util, os, json, sys
H = os.path.dirname(os.path.abspath(__file__)); RP_DIR = os.path.join(H, '..', 'CampoEsferico_Etapas_P_A_B_H_RP')  # dados: rodar antes AlphaPhi_EcoBIP_B0_gera_dobras.py e copiar os .npy para esta pasta
spec = importlib.util.spec_from_file_location('RP', os.path.join(RP_DIR, 'AlphaPhi_CampoEsferico_EtapaRP_Reprojecao.py')); RP = importlib.util.module_from_spec(spec); spec.loader.exec_module(RP)
G, Hh = RP.G, RP.Hh
FS = 44100; WIN = FS; HOP = FS // 4; NB = 225; P_MEM = 12; SEM = list(range(100, 110))
FAIXAS = np.geomspace(40, 20000, NB + 1); CENTROS = np.sqrt(FAIXAS[:-1] * FAIXAS[1:])
f = np.fft.rfftfreq(WIN, 1 / FS); JAN = np.hanning(WIN)
IDX = [np.where((f >= a) & (f < b))[0] for a, b in zip(FAIXAS[:-1], FAIXAS[1:])]
IDX = [ix if len(ix) else np.array([int(np.argmin(abs(f - c)))]) for ix, c in zip(IDX, CENTROS)]
VAZIAS = sum(1 for a, b in zip(FAIXAS[:-1], FAIXAS[1:]) if not ((f >= a) & (f < b)).any())
def quadros(sinais):
    E, org = [], []
    for k, s in enumerate(sinais):
        for t0 in range(0, len(s) - WIN + 1, HOP):
            F = np.abs(np.fft.rfft(s[t0:t0 + WIN] * JAN)) ** 2; e = np.array([F[ix].sum() for ix in IDX]); E.append(e / F.sum()); org.append((k, t0 / FS))
    Lg = np.log(np.array(E) + 1e-6); return Lg - Lg.mean(0), org
def escolher(Xc):
    N = Xc.shape[1]; C = np.abs(Xc @ Xc.T / N); esc = [0]
    while len(esc) < P_MEM:
        m = C[:, esc].max(1); m[esc] = np.inf; esc.append(int(np.argmin(m)))
    return esc
def memorias(sinais, Y):
    Z, org = quadros(sinais); Xc = np.where(Z @ Y.T >= 0, 1., -1.); esc = escolher(Xc); return Xc[esc], [org[i] for i in esc], len(org)
def eco_esc(J, N, conv):
    return (N - 1) / N if conv else None
def correr(J, X, cond, eco=False, lam=0.):
    N = X.shape[1]; r = {'id25': [], 'id40': [], 'm40': []}
    for s in SEM:
        for flip in (.25, .40):
            rng = np.random.default_rng([1_400_000 + s, int(flip * 100)]); S0 = X.copy(); k = int(round(flip * N))
            for p in range(len(S0)): S0[p, rng.choice(N, k, replace=False)] *= -1
            S = Hh.dinamica(J, X, S0, np.random.default_rng([1_500_000 + s, int(flip * 100), N]), eco, None, lam)
            ide = float(np.mean(np.argmax(S @ X.T, 1) == np.arange(len(X)))); r['id25' if flip == .25 else 'id40'].append(ide)
            if flip == .40: r['m40'].append(float((S * X).mean(1).mean()))
    print(f'  {cond} pronto', file=sys.stderr, flush=True); return r

if __name__ == '__main__':
    dob = np.load(H + '/ecobip_dobras.npy'); ent = np.load(H + '/ecobip_entrada.npy'); ctrl = np.load(H + '/ctrl_quadrada_cascata.npy')
    tt = np.linspace(0, 7.0, ctrl.shape[1], endpoint=False); quad = np.sign(np.sin(2 * np.pi * 880 * tt))
    P610 = G.pontos_encaixados(610); P233 = P610[:233]; Y610, gr = G.harmonicos(P610, 14)
    print(f'faixas sem ponto do espectro (recebem o vizinho): {VAZIAS} de {NB}')
    res = {}; s233, s610 = 2 * RP.espac(233), 2 * RP.espac(610)
    for nome, sinais in (('EcoBIP 880', [ent] + list(dob)), ('controle', [quad] + list(ctrl))):
        X, org, ncand = memorias(sinais, Y610); N = 610
        Cm = X @ X.T / N; tri = Cm[np.triu_indices(P_MEM, 1)]
        print(f'\n##### {nome}: {ncand} quadros candidatos; escolhidos (sinal, s): ' + ' '.join(f'({k},{t:.2f})' for k, t in org))
        print(f'S00 correlação entre memórias: média {tri.mean():+.3f} · máxima {tri.max():+.3f} · máx |.| {abs(tri).max():.3f} -> '
              f'{"cumprido" if tri.mean() < .3 and tri.max() < .5 else "NÃO cumprido"}')
        D = G.distancia_geodesica(P610) / np.pi
        J = {'C610': RP.conv(X, N), 'A610': RP.ligacoes(P610, s610, X, N), 'A610-H': RP.ligacoes(P610, .35, X, N)}
        r = {}
        conds = ['C610', 'A610'] if nome == 'controle' else ['C610', 'A610', 'A610-H']
        for c in conds: r[c] = correr(J[c], X, c)
        if nome == 'EcoBIP 880':
            CVs = np.exp(-(D / s610) ** 2); np.fill_diagonal(CVs, 0); esc_a = CVs.sum(1).mean() / N; esc_c = (N - 1) / N
            for lam in (.5, 2., 8.):
                r[f'A610+eco λ{lam:g}'] = correr(J['A610'], X, f'A610+eco λ{lam:g}', True, lam * esc_a)
                r[f'C610+eco λ{lam:g}'] = correr(J['C610'], X, f'C610+eco λ{lam:g}', True, lam * esc_c)
            r['eco puro'] = correr(None, X, 'eco puro', True, 1.)
            X233 = X[:, :233]; JA = RP.ligacoes(P233, s233, X233, 233); Jr = J['A610'].copy(); Jr[:233, :233] = JA * 233 / 610
            r['A233'] = correr(JA, X233, 'A233'); r['A610-reproj'] = correr(Jr, X, 'A610-reproj'); r['A610-zero'] = r['A610']
        print(f"   {'condição':<17}{'ident. 25%':>11}{'ident. 40%':>11}{'sobrep. 40%':>12}")
        for c, v in r.items(): print(f"   {c:<17}{np.mean(v['id25']):>11.3f}{np.mean(v['id40']):>11.3f}{np.mean(v['m40']):>12.3f}")
        s01 = np.mean(r['C610']['id25']); print(f"S01 C610 identifica com pista 25%: {s01:.3f} -> {'cumprido' if s01 >= .8 else 'NÃO cumprido'}")
        col = lambda c: np.array(r[c]['id40'])
        def par(a, b, bil=False):
            d = col(a) - col(b); se = d.std(ddof=1) / np.sqrt(len(d)); nf = max((d > 0).sum(), (d < 0).sum()) if bil else (d > 0).sum()
            return d.mean(), se, nf, ((abs(d.mean()) if bil else d.mean()) > 2 * se and nf >= 8 and d.mean() != 0)
        if nome == 'EcoBIP 880':
            print('=== CRITÉRIOS (identificação, pista 40%) ===')
            for k, a, b, bil in (('S1', 'A610', 'C610', True), ('S2', 'A610', 'A610-H', False), ('S3', 'A610+eco λ2', 'A610', False),
                                 ('S4', 'A610+eco λ2', 'eco puro', False), ('S5', 'A610-reproj', 'A233', False)):
                dm, se, nf, ok = par(a, b, bil); print(f"{k} {a} − {b}: {dm:+.3f} (EP {se:.3f}, {nf}/10) -> {('difere' if ok else 'não difere') if bil else ('cumprido' if ok else 'NÃO cumprido')}")
            d6 = (col('A610-reproj') - col('A610-zero')).mean(); print(f"S6 A610-reproj − A610-zero: {d6:+.3f} (>= −0,05) -> {'cumprido' if d6 >= -.05 else 'NÃO cumprido'}")
        else:
            dm, se, nf, ok = par('A610', 'C610', True); print(f"(descritivo) S1 no controle: A610 − C610 {dm:+.3f} (EP {se:.3f}, {nf}/10)")
        res[nome] = {'org': org, 'cor_media': float(tri.mean()), 'cor_max': float(tri.max()), 'r': r}
    json.dump(res, open(H + '/EtapaS_resultados.json', 'w'))
