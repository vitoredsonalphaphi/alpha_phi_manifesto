# EXPERIMENTO: ETAPA VL3 — o violão sobre a base nova (critérios aprovados pelo pesquisador em 22/10/2026, antes da execução)
# Grade Esférica v2 canônica (endereço por anéis: 75 faixas × 8 instantes); memória com alturas (como na VL2, modo local); ligações pela regra
# delta local + fundo afinado (Etapa LA); neurônio N3 (ganho ajustado + âncora α; Etapas NA/LA); previsor ligado. O além: a altura estimada
# vira ganho aplicado ao próprio som ruidoso. 12 trechos da gravação de 13/05/2026; ruído branco a +10 dB (principal) e 0 dB; sementes 200–209.
# VL3-00: >= 6 memórias, correlação máxima < 0,5; pista com correlação 0,2–0,8. VL3-1: D-campo, ganho de nitidez > 0 dB. VL3-2: D-estado > 0 dB.
# VL3-3: D-campo >= 70% do teto. (+10 dB, trechos guardados; pareado > 2 EP e >= 8/10.) Descritivo: trechos novos (0,25 s depois), 0 dB.
import numpy as np, importlib.util, json, sys
from multiprocessing import Pool
import os
SP = os.path.dirname(os.path.abspath(__file__)) + '/'; R = os.path.join(SP, '..') + '/'
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V2 = load('V2', SP + 'AlphaPhi_CampoEsferico_EtapaVL2_ViolaoAudivel.py'); LA = load('LA', SP + 'AlphaPhi_CampoEsferico_EtapaLA_LigacoesAlturas.py'); GV2 = load('GV2', R + 'AlphaPhi_GradeEsferica_v2.py')
NA, G2 = LA.NA, V2.G2; E = GV2.enderecos_aneis(G2.NFX, G2.NT); SEM = list(range(200, 210)); SNRS = [10, 0]
assert np.allclose(NA.P2, GV2.pontos_encaixados(610))
class Tarefa:
    def __init__(self):
        x = V2.ler(); t0s = list(range(0, len(x) - V2.SEG - V2.HOPT + 1, V2.HOPT)); segs = [x[t:t + V2.SEG] for t in t0s]
        pot = np.array([np.mean(s ** 2) for s in segs]); vivo = pot > np.quantile(pot, .10)
        self.t0s = [t for t, v in zip(t0s, vivo) if v]; self.segs = [s for s, v in zip(segs, vivo) if v]
        En = np.array([G2.energias(s) for s in self.segs]); self.eps = 1e-10 * En.mean(); Lg = np.log(En + self.eps)
        self.perfil = Lg.mean(0); self.sd = (Lg - self.perfil).std()
        XL = np.array([GV2.escrever(np.tanh((l - self.perfil) / self.sd), E) for l in Lg]); C = np.abs(V2.cosm(XL)); esc = [0]
        while len(esc) < V2.PMAX:
            m = C[:, esc].max(1); m[esc] = np.inf; i = int(np.argmin(m))
            if m[i] >= V2.LIM: break
            esc.append(i)
        self.esc = esc; self.X = XL[esc]
    def pista(self, seg): return GV2.escrever(np.tanh((np.log(G2.energias(seg) + self.eps) - self.perfil) / self.sd), E)
    def decod(self, S, ruidoso):
        z = self.sd * np.arctanh(np.clip(GV2.ler(S, E), -.999, .999)); Eh = np.exp(z + self.perfil); En = G2.energias(ruidoso) + self.eps
        return G2.aplicar_ganho(ruidoso, np.clip(np.sqrt(Eh / En), .05, 1.))
    def oraculo(self, limpo, ruidoso):
        return G2.aplicar_ganho(ruidoso, np.clip(np.sqrt((G2.energias(limpo) + self.eps) / (G2.energias(ruidoso) + self.eps)), .05, 1.))
def rodar(arg, guardar=False, so=None):
    db, quais = arg; T = Tarefa(); P = len(T.esc); J = LA.ligacao_delta(T.X); g = NA.ganho_ajustado(J, T.X); cel = E.ravel()
    res = {k: [] for k in ('D-estado', 'D-campo', 'teto', 'pista_corr', 'alt_estado', 'alt_campo')}; filme = None
    for s in (SEM if so is None else [so]):
        idx = T.esc if quais == 'guardados' else [min(i + 1, len(T.segs) - 1) for i in T.esc]
        limpos = [T.segs[i] for i in idx]; ruid = [V2.ruido(c, db, np.random.default_rng([s, db + 50, k])) for k, c in enumerate(limpos)]
        S0 = np.array([T.pista(r) for r in ruid]); alvo = np.array([T.pista(c) for c in limpos])
        res['pista_corr'].append(float(np.mean([np.corrcoef(S0[k, cel], alvo[k, cel])[0, 1] for k in range(P)])))
        Sf = NA.recuperar(J, T.X, S0, np.random.default_rng([2_900_000 + s, db + 50]), g, True, True); Sc = LA.campo(Sf, T.X)
        for nome, S in (('D-estado', Sf), ('D-campo', Sc)):
            res[nome].append(float(np.mean([V2.snr(T.decod(S[k], ruid[k]), limpos[k]) - V2.snr(ruid[k], limpos[k]) for k in range(P)])))
        res['alt_estado'].append(float(np.mean([np.corrcoef(Sf[k, cel], alvo[k, cel])[0, 1] for k in range(P)])))
        res['alt_campo'].append(float(np.mean([np.corrcoef(Sc[k, cel], alvo[k, cel])[0, 1] for k in range(P)])))
        res['teto'].append(float(np.mean([V2.snr(T.oraculo(limpos[k], ruid[k]), limpos[k]) - V2.snr(ruid[k], limpos[k]) for k in range(P)])))
        if guardar: filme = dict(T=T, limpos=limpos, ruid=ruid, Sf=Sf, Sc=Sc, idx=idx)
    print(f'  {db:+d} dB {quais} pronto', file=sys.stderr, flush=True)
    return (arg, res, filme) if guardar else (arg, res)
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'sanidade':
    T = Tarefa(); P = len(T.esc); C = V2.cosm(T.X)[np.triu_indices(P, 1)]; cel = E.ravel()
    print(f"{P} memórias; trechos (s): " + ' '.join(f'{T.t0s[i] / V2.FS:.2f}' for i in T.esc) + f" · correlação máxima {np.abs(C).max():.3f}")
    for db in SNRS:
        cc = [np.corrcoef(T.pista(V2.ruido(T.segs[i], db, np.random.default_rng([s, db + 50, k])))[cel], T.X[k, cel])[0, 1] for k, i in enumerate(T.esc) for s in SEM]
        print(f"  pista a {db:+d} dB: correlação com a forma limpa {np.mean(cc):.3f}")
if __name__ == '__main__' and len(sys.argv) == 1:
    with Pool(4) as pool: out = dict(pool.map(rodar, [(10, 'guardados'), (0, 'guardados'), (10, 'novos'), (0, 'novos')]))
    for (db, q), r in out.items():
        print(f"\n##### {db:+d} dB · trechos {q} · pista {np.mean(r['pista_corr']):.3f} → altura: estado {np.mean(r['alt_estado']):.3f}, campo {np.mean(r['alt_campo']):.3f}")
        for k in ('D-estado', 'D-campo', 'teto'): print(f"   {k:<9}{np.mean(r[k]):+7.2f} dB (EP {np.std(r[k], ddof=1) / np.sqrt(10):.2f})")
    r = out[(10, 'guardados')]; col = lambda k: np.array(r[k])
    def teste(nome, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{nome}: {d.mean():+.2f} dB (EP {se:.2f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    print('\n=== CRITÉRIOS (+10 dB, trechos guardados) ===')
    teste('VL3-1 D-campo: ganho de nitidez', col('D-campo')); teste('VL3-2 D-estado: ganho de nitidez', col('D-estado'))
    fr = col('D-campo').mean() / col('teto').mean(); print(f"VL3-3 D-campo / teto: {fr:.2f} (>= 0,70) -> {'cumprido' if fr >= .7 else 'NÃO cumprido'}")
    json.dump({f'{db}|{q}': v for (db, q), v in out.items()}, open(SP + 'EtapaVL3_resultados.json', 'w'))
