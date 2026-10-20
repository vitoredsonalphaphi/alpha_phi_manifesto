# EXPERIMENTO: ETAPA RP — a reprojeção (critérios aprovados pelo pesquisador em 20/10/2026, antes da execução)
# Grade Esférica v1. Memórias da Etapa H (EcoBIP: entrada + 5 dobras; 225 faixas → 225 harmônicos).
# Vizinhança da memória = 2 × espaçamento médio: σ = 0,148 (233) e 0,091 (610); fundo do campo σe = 0,15·√(233/N), σi = 3σe, peso 0,02.
# Gatilho (R0): graus pedidos (14) > medível em 233 (9) e <= medível em 610 (16).
# Reprojeção: 233 → 610 pontos encaixados; o bloco dos 233 antigos mantém as ligações da rede de 233; o resto na nova escala.
# Condições: A233-H (σ 0,35), A233, A610-reproj, A610-zero, C233, C610. Sem eco. Pistas 25% e 40%. Sementes 90–99.
# R1: A233 > A233-H. R2: A610-reproj > A233. R3: A610-reproj >= A610-zero − 0,05 (média pareada). R4: A610-reproj difere de C610.
# (identificação com pista 40%; pareado > 2 EP e >= 8/10; R4 bilateral.)
# Nota declarada: o "previsor antes e depois" foi substituído pela estabilidade MEDIDA na memória (o previsor condicional pede a
# estatística de um ambiente, e aqui há só 6 memórias fixas).
import numpy as np, importlib.util, os, json
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, '..')  # rodar antes AlphaPhi_EcoBIP_B0_gera_dobras.py
spec = importlib.util.spec_from_file_location('G', os.path.join(R, 'AlphaPhi_GradeEsferica_v1.py')); G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
sp2 = importlib.util.spec_from_file_location('Hh', os.path.join(H, 'AlphaPhi_CampoEsferico_EtapaH_Hopfield.py'))
Hh = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(Hh)
SEM = list(range(90, 100)); FS = 44100
def espac(N): return np.sqrt(4 * np.pi / N) / np.pi
def ligacoes(P, sig_mem, X, N):
    D = G.distancia_geodesica(P) / np.pi; CV = np.exp(-(D / sig_mem) ** 2); np.fill_diagonal(CV, 0)
    se = .15 * np.sqrt(233 / N); FU = np.exp(-(D / se) ** 2) - .5 * np.exp(-(D / (3 * se)) ** 2); np.fill_diagonal(FU, 0)
    J = CV * (X.T @ X) / N + .02 * FU; np.fill_diagonal(J, 0); return J
def conv(X, N):
    J = X.T @ X / N; np.fill_diagonal(J, 0); return J

if __name__ == '__main__':
    dob = np.load(os.path.join(H, 'ecobip_dobras.npy')); ent = np.load(os.path.join(H, 'ecobip_entrada.npy')); sinais = [ent] + list(dob)
    P610 = G.pontos_encaixados(610); P233 = P610[:233]
    Y610, gr = G.harmonicos(P610, 14); Y233 = Y610[:233]
    print(f'R0 gatilho: graus pedidos 14 | medível em 233: {G.lmax_mensuravel(233)} | em 610: {G.lmax_mensuravel(610)} -> '
          f'{"cumprido" if 14 > G.lmax_mensuravel(233) and 14 <= G.lmax_mensuravel(610) else "NÃO cumprido"}')
    X233 = Hh.padroes(sinais, Y233); X610 = Hh.padroes(sinais, Y610)
    print('memórias idênticas nos 233 pontos antigos depois da reprojeção:', bool((X610[:, :233] == X233).all()))
    s233, s610 = 2 * espac(233), 2 * espac(610); print(f'vizinhança: σ(233) = {s233:.4f} · σ(610) = {s610:.4f}')
    J = {'A233-H': ligacoes(P233, .35, X233, 233), 'A233': ligacoes(P233, s233, X233, 233), 'A610-zero': ligacoes(P610, s610, X610, 610),
         'C233': conv(X233, 233), 'C610': conv(X610, 610)}
    Jr = J['A610-zero'].copy(); Jr[:233, :233] = J['A233'] * 233 / 610; J['A610-reproj'] = Jr      # bloco antigo preservado (mesma escala 1/N da rede maior)
    ordem = ['A233-H', 'A233', 'A610-reproj', 'A610-zero', 'C233', 'C610']; res = {c: {'id25': [], 'id40': [], 'm40': [], 'est': []} for c in ordem}
    for c in ordem:
        X = X233 if c.endswith('233') or c == 'A233-H' else X610; N = X.shape[1]
        res[c]['est'] = float(((X @ J[c]) * X > 0).mean())
        for s in SEM:
            for flip in (.25, .40):
                rng = np.random.default_rng([1_200_000 + s, int(flip * 100)]); S0 = X.copy(); k = int(round(flip * N))
                for p in range(len(S0)): S0[p, rng.choice(N, k, replace=False)] *= -1
                S = Hh.dinamica(J[c], X, S0, np.random.default_rng([1_300_000 + s, int(flip * 100), N]))
                ide = float(np.mean(np.argmax(S @ X.T, 1) == np.arange(len(X)))); res[c]['id25' if flip == .25 else 'id40'].append(ide)
                if flip == .40: res[c]['m40'].append(float((S * X).mean(1).mean()))
        print(f'  {c} pronto', flush=True)
    print(f"\n{'condição':<13}{'estável':>9}{'ident. 25%':>12}{'ident. 40%':>12}{'sobrep. 40%':>13}")
    for c in ordem: print(f"{c:<13}{res[c]['est']:>9.3f}{np.mean(res[c]['id25']):>12.3f}{np.mean(res[c]['id40']):>12.3f}{np.mean(res[c]['m40']):>13.3f}")
    col = lambda c: np.array(res[c]['id40'])
    def par(a, b, bil=False):
        d = col(a) - col(b); se = d.std(ddof=1) / np.sqrt(len(d)); nf = max((d > 0).sum(), (d < 0).sum()) if bil else (d > 0).sum()
        return d.mean(), se, nf, ((abs(d.mean()) if bil else d.mean()) > 2 * se and nf >= 8 and d.mean() != 0)
    print('\n=== CRITÉRIOS (identificação, pista 40%) ===')
    for k, a, b, bil in (('R1', 'A233', 'A233-H', False), ('R2', 'A610-reproj', 'A233', False), ('R4', 'A610-reproj', 'C610', True)):
        dm, se, nf, ok = par(a, b, bil); print(f"{k} {a} − {b}: {dm:+.3f} (EP {se:.3f}, {nf}/10) -> {('difere' if ok else 'não difere') if bil else ('cumprido' if ok else 'NÃO cumprido')}")
    d3 = (col('A610-reproj') - col('A610-zero')).mean(); print(f"R3 A610-reproj − A610-zero: {d3:+.3f} (>= −0,05) -> {'cumprido' if d3 >= -.05 else 'NÃO cumprido'}")
    json.dump(res, open(os.path.join(H, 'EtapaRP_resultados.json'), 'w'))
