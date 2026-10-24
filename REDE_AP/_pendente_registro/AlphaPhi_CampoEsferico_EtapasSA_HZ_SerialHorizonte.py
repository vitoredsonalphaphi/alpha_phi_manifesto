# EXPERIMENTO: ETAPAS SA (o serial de atratores) e HZ (o horizonte) — critérios aprovados pelo pesquisador em 24/10/2026, antes da execução
# Base da Etapa LA (Grade v2; 12 memórias contínuas; regra delta local; neurônio N3; previsor; âncora α).
# SA (sementes 250–259; ruído 3,0 principal, 2,0 e 4,0 descritivos; todas as versões com exatamente 50 voltas, sem parada antecipada):
#   SUP (β = 8 fixo); SN (5 × 10 voltas, β = 8/φ⁴ → 8/φ³ → 8/φ² → 8/φ → 8); SE (4 estágios de 13,13,12,12 voltas: previsor só nas formas
#   até o grau 2, 5, 9 e, no último, o previsor inteiro); SNE (os dois; 5 estágios de nitidez × graus 2,5,9,14,14); INV (β de 8 → 8/φ⁴).
#   SA0: SUP identificação <= 0,95 (senão ruído 4,0; senão para). SA1: SN > SUP (identificação). SA2: SN >= SUP (altura; não negativo além
#   de 2 EP). SA3: SN > INV (identificação). Descritivo: SE, SNE, captura precoce (memória escolhida na volta 5 errada e mantida até o fim).
# HZ (sementes 240–249; ruído 3,0 principal, 4,0 de reserva; recuperação da Etapa LA, com parada antecipada):
#   horizonte = o estado visto de longe: graus <= 5, atenuados por r^L com r = φ⁻², normalizado para ter ε × a força média do sinal local;
#   ε = 1/φ⁴. SUP; HZ-S (sempre); HZ-D (× d_i = 1 − |S_i|); CT (sinal fraco ao acaso, mesma estrutura de graus <= 5 e mesma força, fixo por
#   recuperação, × d_i); varredura descritiva de HZ-D com ε = 1/φ⁶, 1/φ⁴, 1/φ².
#   HZ0: SUP altura < 0,95 e pelo menos um erro de identificação (senão ruído 4,0). HZ1: casos claros (SUP acerta e altura >= 0,95):
#   correlação média entre os finais de HZ-D e SUP >= 0,99. HZ2: HZ-D > SUP (altura). HZ3: HZ-D > CT. HZ4: HZ-D >= HZ-S (não negativo além
#   de 2 EP). (pareado > 2 EP e >= 8/10.)
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
SP = '/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
LA = load('LA', SP + 'exp_etapaLA.py'); NA, FM = LA.NA, LA.FM
N, P = NA.N, NA.P; PHI = 1.6180339887; Y2, G2 = NA.Y2, NA.GR; W = NA.W
_A = Y2 * np.sqrt(W)[:, None]; PINV = np.linalg.pinv(_A) * np.sqrt(W)[None, :]     # coeficientes = S @ PINV.T (mínimos quadrados pesados)
def coef(S): return S @ PINV.T
def largas(S, L):
    k = int((G2 <= L).sum()); return coef(S)[:, :k] @ Y2[:, :k].T
K5 = int((G2 <= 5).sum()); ATEN = (PHI ** -2) ** G2[:K5]
def horizonte(S):
    return (coef(S)[:, :K5] * ATEN[None, :]) @ Y2[:, :K5].T
def previsor(S, X, beta):
    m = np.sign(S) @ X.T / N; w = np.exp(beta * (m - m.max(1, keepdims=True))); w /= w.sum(1, keepdims=True); return w @ X
def recuperar(J, X, S0, rng, g, beta=lambda v: FM.BETA_PREVISOR, lmax=lambda v: None, hz=None, nvoltas=FM.MAXV, parada=True, volta_obs=None):
    """A recuperação da Etapa NA (N3, âncora α, previsor), com β e alcance do previsor por volta, e o horizonte opcional:
    hz = (ε, tipo, Hfixo): tipo 'S' (sempre), 'D' (× dúvida), 'CT' (Hfixo ao acaso, × dúvida)."""
    S = S0.copy(); hesc = np.abs(S0 @ J).mean(1, keepdims=True); mu_t = NA.wmean(X).mean(); sd_t = NA.wstd(X).mean(); obs = None
    for v in range(nvoltas):
        pv = previsor(S, X, beta(v)); L = lmax(v)
        if L is not None: pv = largas(pv, L)
        extra = FM.FORCA_PREVISOR * pv
        if hz is not None:
            eps, tipo, Hf = hz; H = Hf if tipo == 'CT' else horizonte(S)
            loc = np.abs(S @ J / hesc).mean(1, keepdims=True); H = eps * loc * H / np.maximum(np.abs(H).mean(1, keepdims=True), 1e-12)
            extra = extra + (H if tipo == 'S' else H * (1 - np.abs(S)))
        mud = 0.
        for i in rng.permutation(N):
            novo = np.tanh(g * (S @ J[:, i] / hesc[:, 0] + extra[:, i])); mud = max(mud, float(np.abs(novo - S[:, i]).max())); S[:, i] = novo
        m = NA.wmean(S); S = S - 0.5 * (m - mu_t)[:, None]; m = NA.wmean(S); sd = NA.wstd(S)
        S = np.clip(m[:, None] + (S - m[:, None]) * (1 + 0.5 * (sd_t / np.maximum(sd, 1e-9) - 1))[:, None], -1, 1)
        if volta_obs is not None and v + 1 == volta_obs: obs = S.copy()
        if parada and mud < FM.TOL: break
    return (S, obs) if volta_obs is not None else S
def escolha(S, X): return np.argmax(np.corrcoef(S, X)[:P, P:], 1)
# ---------------- SA ----------------
B8 = FM.BETA_PREVISOR
SA_V = {'SUP': (lambda v: B8, lambda v: None),
        'SN': (lambda v: B8 / PHI ** (4 - v // 10), lambda v: None),
        'SE': (lambda v: B8, lambda v: [2, 5, 9, None][min(v // 13 if v < 26 else 2 + (v - 26) // 12, 3)]),
        'SNE': (lambda v: B8 / PHI ** (4 - v // 10), lambda v: [2, 5, 9, None, None][v // 10]),
        'INV': (lambda v: B8 / PHI ** (v // 10), lambda v: None)}
def tarefa_SA(ruido):
    out = {v: {'alt': [], 'ident': [], 'captura': []} for v in SA_V}
    for s in range(250, 260):
        X = NA.memorias(s); S0 = NA.pista(X, ruido, s); J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X); certo = np.arange(P)
        for v, (bf, lf) in SA_V.items():
            Sf, S5 = recuperar(J, X, S0, np.random.default_rng([6_000_000 + s, int(ruido * 10)]), g, bf, lf, nvoltas=50, parada=False, volta_obs=5)
            m = NA.medir(Sf, X); e5, ef = escolha(S5, X), escolha(Sf, X)
            out[v]['alt'].append(m['alt']); out[v]['ident'].append(m['ident']); out[v]['captura'].append(float(np.mean((e5 != certo) & (ef == e5))))
    print(f'  SA ruído {ruido} pronto', file=sys.stderr, flush=True); return ('SA', ruido), out
# ---------------- HZ ----------------
def tarefa_HZ(ruido):
    VER = ['SUP', 'HZ-S', 'HZ-D', 'CT', 'HZ-D e=1/phi6', 'HZ-D e=1/phi2']
    out = {v: {'alt': [], 'ident': [], 'final': []} for v in VER}; dmed = []
    for s in range(240, 250):
        X = NA.memorias(s); S0 = NA.pista(X, ruido, s); J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X)
        rng = lambda: np.random.default_rng([6_100_000 + s, int(ruido * 10)])
        cr = np.random.default_rng([6_200_000 + s, int(ruido * 10)]).normal(size=(P, K5)) * np.sqrt(np.mean(coef(X)[:, :K5] ** 2, 0))[None, :]
        Hct = (cr * ATEN[None, :]) @ Y2[:, :K5].T
        conf = {'SUP': None, 'HZ-S': (PHI ** -4, 'S', None), 'HZ-D': (PHI ** -4, 'D', None), 'CT': (PHI ** -4, 'CT', Hct),
                'HZ-D e=1/phi6': (PHI ** -6, 'D', None), 'HZ-D e=1/phi2': (PHI ** -2, 'D', None)}
        for v, hz in conf.items():
            Sf = recuperar(J, X, S0, rng(), g, hz=hz); m = NA.medir(Sf, X)
            out[v]['alt'].append(m['alt']); out[v]['ident'].append(m['ident']); out[v]['final'].append(Sf.tolist())
        dmed.append(float(np.mean(1 - np.abs(np.array(out['HZ-D']['final'][-1])))))
    print(f'  HZ ruído {ruido} pronto', file=sys.stderr, flush=True); return ('HZ', ruido), {'R': out, 'd_final': dmed}
def rodar(arg): return tarefa_SA(arg[1]) if arg[0] == 'SA' else tarefa_HZ(arg[1])
def teste(nome, d, nao_neg=False):
    d = np.asarray(d); se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum()
    ok = (d.mean() > -2 * se) if nao_neg else (d.mean() > 2 * se and nf >= 8 and d.mean() > 0)
    print(f"{nome}: {d.mean():+.4f} (EP {se:.4f}, {nf}/10 a favor) -> {'cumprido' if ok else 'NÃO cumprido'}")
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'conferir':
    X = NA.memorias(1); S0 = NA.pista(X, 1.0, 1); J = LA.ligacao_delta(X); g = NA.ganho_ajustado(J, X)
    a = NA.recuperar(J, X, S0, np.random.default_rng(7), g, True, True); b = recuperar(J, X, S0, np.random.default_rng(7), g)
    print('recuperar (β fixo, sem horizonte, com parada) reproduz NA.recuperar:', np.allclose(a, b))
    print('betas do SN por estágio:', [round(SA_V['SN'][0](v), 3) for v in (0, 10, 20, 30, 40)], '· INV:', [round(SA_V['INV'][0](v), 3) for v in (0, 10, 20, 30, 40)])
    print('graus do SE por volta (0,13,26,38,49):', [SA_V['SE'][1](v) for v in (0, 12, 13, 25, 26, 37, 38, 49)], '· SNE:', [SA_V['SNE'][1](v) for v in (0, 10, 20, 30, 40)])
    print('horizonte: fração da energia do estado nos graus <= 5:', round(float((largas(S0, 5) ** 2).sum() / (S0 ** 2).sum()), 3))
if __name__ == '__main__' and len(sys.argv) == 1:
    with Pool(4) as pool: res = dict(pool.map(rodar, [('SA', 3.0), ('HZ', 3.0), ('SA', 2.0), ('SA', 4.0), ('HZ', 4.0)]))
    print('=================== ETAPA SA — o serial de atratores ===================')
    for r in (2.0, 3.0, 4.0):
        o = res[('SA', r)]; print(f"\n##### ruído {r}" + f"\n{'versão':<6}{'altura':>9}{'identif.':>10}{'captura precoce':>17}")
        for v in SA_V: print(f"{v:<6}" + ''.join(f"{np.mean(o[v][k]):>{w}.4f}" for k, w in (('alt', 9), ('ident', 10), ('captura', 17))))
    i3 = np.mean(res[('SA', 3.0)]['SUP']['ident']); i4 = np.mean(res[('SA', 4.0)]['SUP']['ident'])
    pr = 3.0 if i3 <= .95 else (4.0 if i4 <= .95 else None)
    print(f"\n=== CRITÉRIOS SA === SA0 SUP identificação: ruído 3,0 {i3:.4f} · ruído 4,0 {i4:.4f} -> " + (f'principal {pr}' if pr else 'NÃO cumprido nos dois: não há sumidouros nesta tarefa; a etapa para'))
    if pr:
        o = res[('SA', pr)]; c = lambda v, k: np.array(o[v][k])
        teste('SA1 identificação SN − SUP', c('SN', 'ident') - c('SUP', 'ident')); teste('SA2 altura SN − SUP, não negativo', c('SN', 'alt') - c('SUP', 'alt'), True)
        teste('SA3 identificação SN − INV', c('SN', 'ident') - c('INV', 'ident'))
    print('\n=================== ETAPA HZ — o horizonte ===================')
    for r in (3.0, 4.0):
        o = res[('HZ', r)]['R']; print(f"\n##### ruído {r} · dúvida média no fim (HZ-D) {np.mean(res[('HZ', r)]['d_final']):.3f}\n{'versão':<16}{'altura':>9}{'identif.':>10}")
        for v in o: print(f"{v:<16}{np.mean(o[v]['alt']):>9.4f}{np.mean(o[v]['ident']):>10.4f}")
    o3 = res[('HZ', 3.0)]['R']; a3 = np.mean(o3['SUP']['alt']); e3 = np.mean(o3['SUP']['ident']) < 1
    pr = 3.0 if (a3 < .95 and e3) else 4.0
    print(f"\n=== CRITÉRIOS HZ === HZ0 SUP no ruído 3,0: altura {a3:.4f}, algum erro {e3} -> principal {pr}")
    o = res[('HZ', pr)]['R']; c = lambda v: np.array(o[v]['alt']); corr = []
    for s in range(10):
        Fs, Fd = np.array(o['SUP']['final'][s]), np.array(o['HZ-D']['final'][s]); X = NA.memorias(240 + s); ok = escolha(Fs, X) == np.arange(P)
        alt = np.array([np.corrcoef(Fs[k], X[k])[0, 1] for k in range(P)])
        corr += [np.corrcoef(Fs[k], Fd[k])[0, 1] for k in range(P) if ok[k] and alt[k] >= .95]
    print(f"HZ1 não interferência nos casos claros ({len(corr)} casos): correlação média {np.mean(corr):.4f} (mínima {np.min(corr):.4f}) -> {'cumprido' if np.mean(corr) >= .99 else 'NÃO cumprido'}")
    teste('HZ2 altura HZ-D − SUP', c('HZ-D') - c('SUP')); teste('HZ3 altura HZ-D − CT', c('HZ-D') - c('CT')); teste('HZ4 altura HZ-D − HZ-S, não negativo', c('HZ-D') - c('HZ-S'), True)
    for r in res:
        if r[0] == 'HZ':
            for v in res[r]['R']: res[r]['R'][v].pop('final')
    json.dump({f'{k[0]}|{k[1]}': v for k, v in res.items()}, open(SP + 'ecobip/exp_etapaSA_HZ.json', 'w'))
