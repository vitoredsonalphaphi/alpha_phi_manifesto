# Versão auditada (pareamento exato por carga e pista). Resultados: REDE_AP/AlphaPhi_MemoriaEsferica_Etapa1_RESULTADOS_19out2026.md
# EXPERIMENTO: MEMÓRIA ESFÉRICA — primeira etapa (critérios aprovados pelo pesquisador em 19/10/2026, antes da execução)
# 256 neurônios ±1 na esfera de Fibonacci; padrões aleatórios; Hebb × mapa CPPN C(i,j) da distância geodésica.
# Cargas P = 5, 13, 20, 28, 36. Pistas com 25% (principal) e 40% (robustez) dos neurônios invertidos.
# Atualização assíncrona (ordem aleatória por volta, a mesma para os padrões recuperados em paralelo), até 50 voltas.
# Sucesso: m >= 0,95. Medida principal: sobreposição média nas cinco cargas.
# Condições: H0 Hopfield; G1 esfera + CPPN escolhido pelo eco; C1 C embaralhado; C2 C constante = média de C;
#            C3 posições aleatórias, mesma fórmula; C4 fórmula ao acaso entre as cinco.
# Eco: sondagem com 5 sementes próprias (100–104), escolhe a fórmula de maior medida principal; avaliação nas sementes 0–9.
# Âncora α (estágio 2, só na fórmula do eco): h_i + a·s_i, com a ∈ {0, 1/137, 1/3}, pista de 40%.
# Critério de comparação: diferença pareada média > 2 EP e vantagem em >= 8 de 10 sementes.
# K0: H0, P=13 (5%), pista 25%: sucesso >= 95%.  K1: G1>C1.  K2: G1>C2.  K3: G1>C3.  K4: G1>C4.  K5: a=1/137>a=0 e a=1/3>a=0 (pista 40%).
import numpy as np, sys, json
PHI = (1 + 5 ** .5) / 2
N = 256; CARGAS = [5, 13, 20, 28, 36]; SEM_TESTE = list(range(10)); SEM_ECO = list(range(100, 105))
MAXV = 50

def fib_esfera(n):
    i = np.arange(n); z = 1 - 2 * (i + .5) / n; r = np.sqrt(1 - z * z); t = i * 2 * np.pi * (1 - 1 / PHI)
    return np.c_[r * np.cos(t), r * np.sin(t), z]
def aleat_esfera(n, rng):
    v = rng.normal(size=(n, 3)); return v / np.linalg.norm(v, axis=1, keepdims=True)
def dist_norm(P):
    return np.arccos(np.clip(P @ P.T, -1, 1)) / np.pi           # 0..1
FORMULAS = {
    'viz_larga':   lambda d: np.exp(-(d / .60) ** 2),
    'viz_media':   lambda d: np.exp(-(d / .35) ** 2),
    'viz_estreita':lambda d: np.exp(-(d / .20) ** 2),
    'viz_aneis_phi': lambda d: np.exp(-(d / .35) ** 2) * np.cos(2 * np.pi * np.log(d + .03) / np.log(PHI)),
    'viz_repeticao': lambda d: np.exp(-(d / .35) ** 2) * np.cos(3 * np.pi * d),
}
def mapa(P, nome):
    C = FORMULAS[nome](dist_norm(P)); np.fill_diagonal(C, 0); return C
def embaralha(C, rng):
    iu = np.triu_indices(len(C), 1); v = C[iu].copy(); rng.shuffle(v)
    D = np.zeros_like(C); D[iu] = v; return D + D.T
def constante(C):
    m = C[np.triu_indices(len(C), 1)].mean(); D = np.full_like(C, m); np.fill_diagonal(D, 0); return D

def recupera(J, X, flip, rng, a=0.0, traj=None):
    P = len(X); S = X.copy()
    k = int(round(flip * N))
    for p in range(P): S[p, rng.choice(N, k, replace=False)] *= -1
    voltas = MAXV
    for v in range(MAXV):
        mud = 0
        for i in rng.permutation(N):
            h = S @ J[:, i] + a * S[:, i]
            novo = np.where(h > 0, 1, np.where(h < 0, -1, S[:, i]))
            mud += int((novo != S[:, i]).sum()); S[:, i] = novo
            if traj is not None: traj.append(S[0, traj_idx].copy())
        if mud == 0: voltas = v + 1; break
    m = (S * X).mean(1)
    return m, voltas, S

def sepstro(J, S):
    H = np.abs(S @ J) + 1e-12; p = H / H.sum(1, keepdims=True)
    coh = 1 - (-(p * np.log(p)).sum(1)) / np.log(N); return float(coh.mean())

def avalia(Cfun, seed, flip, a=0.0):
    """Cfun(seed) -> C. Retorna medida principal, sucesso por carga, voltas médias, Coh médio."""
    C = Cfun(seed)
    ms, suc, vol, coh = [], {}, [], []
    for P in CARGAS:
        rng = np.random.default_rng([10_000 + seed, P, int(flip * 100)])   # AUDITORIA: gerador próprio por carga e pista (pareamento exato)
        X = rng.choice([-1, 1], size=(P, N)).astype(float)
        J = C * (X.T @ X) / N; np.fill_diagonal(J, 0)
        m, v, S = recupera(J, X, flip, rng, a)
        ms.append(m.mean()); suc[P] = float((m >= .95).mean()); vol.append(v); coh.append(sepstro(J, S))
    return dict(principal=float(np.mean(ms)), m_carga=[float(x) for x in ms], sucesso=suc, voltas=float(np.mean(vol)), coh=float(np.mean(coh)))

def pareado(a, b):
    d = np.array(a) - np.array(b); se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(se), int((d > 0).sum()), bool(d.mean() > 2 * se and (d > 0).sum() >= 8)

traj_idx = np.array([0, 85, 170])
if __name__ == '__main__':
    PF = fib_esfera(N); UM = np.ones((N, N)); np.fill_diagonal(UM, 0)
    # ---- eco: sondagem
    eco = {}
    for nome in FORMULAS:
        Cn = mapa(PF, nome); eco[nome] = float(np.mean([avalia(lambda s: Cn, s, .25)['principal'] for s in SEM_ECO]))
    escolhida = max(eco, key=eco.get)
    print('SONDAGEM DO ECO (sementes 100–104, pista 25%, medida principal):')
    for k, v in sorted(eco.items(), key=lambda kv: -kv[1]): print(f'   {k:<15} {v:.4f}')
    print(f'   escolhida: {escolhida}\n')
    CG = mapa(PF, escolhida)
    nomes = list(FORMULAS)
    cond = {
        'H0': lambda s: UM,
        'G1': lambda s: CG,
        'C1': lambda s: embaralha(CG, np.random.default_rng(20_000 + s)),
        'C2': lambda s: constante(CG),
        'C3': lambda s: mapa(aleat_esfera(N, np.random.default_rng(30_000 + s)), escolhida),
        'C4': lambda s: mapa(PF, nomes[np.random.default_rng(40_000 + s).integers(len(nomes))]),
    }
    R = {}
    for c, f in cond.items():
        R[c] = {fl: [avalia(f, s, fl) for s in SEM_TESTE] for fl in (.25, .40)}
        print(f'  {c} pronto', flush=True)
    A = {a: [avalia(lambda s: CG, s, .40, a) for s in SEM_TESTE] for a in (0.0, 1 / 137, 1 / 3)}
    col = lambda c, fl, k='principal': [r[k] for r in R[c][fl]]
    print('\nMEDIDA PRINCIPAL (sobreposição média nas 5 cargas), média ± dp em 10 sementes')
    print(f"{'cond':<5}{'pista 25%':>16}{'pista 40%':>16}{'voltas':>9}{'Coh':>8}   sobreposição por carga (25%): " + ' '.join(f'P={p}' for p in CARGAS))
    for c in cond:
        a, b = np.array(col(c, .25)), np.array(col(c, .40))
        mc = np.mean([r['m_carga'] for r in R[c][.25]], 0)
        print(f"{c:<5}{a.mean():>10.4f}±{a.std():.3f}{b.mean():>10.4f}±{b.std():.3f}{np.mean(col(c,.25,'voltas')):>9.1f}{np.mean(col(c,.25,'coh')):>8.4f}   " + ' '.join(f'{x:.3f}' for x in mc))
    print('\nSUCESSO (m >= 0,95) por carga, pista 25%:')
    for c in cond:
        print(f'  {c:<4}' + '  '.join(f'P={p}: {np.mean([r["sucesso"][p] for r in R[c][.25]]):.2f}' for p in CARGAS))
    print('\n=== CRITÉRIOS ===')
    k0 = np.mean([r['sucesso'][13] for r in R['H0'][.25]]); print(f'K0 H0, P=13, pista 25%: sucesso {k0:.3f} -> {"cumprido" if k0 >= .95 else "NÃO cumprido"}')
    for k, c in (('K1', 'C1'), ('K2', 'C2'), ('K3', 'C3'), ('K4', 'C4')):
        d, se, nf, ok = pareado(col('G1', .25), col(c, .25))
        print(f'{k} G1 − {c} (pista 25%): {d:+.4f} (EP {se:.4f}, {nf}/10) -> {"cumprido" if ok else "NÃO cumprido"}')
    pa = lambda a: [r['principal'] for r in A[a]]
    for a, nome in ((1 / 137, '1/137'), (1 / 3, '1/3')):
        d, se, nf, ok = pareado(pa(a), pa(0.0))
        print(f'K5 a={nome} − a=0 (pista 40%): {d:+.4f} (EP {se:.4f}, {nf}/10) -> {"cumprido" if ok else "NÃO cumprido"}')
    print(f"   médias pista 40%: a=0 {np.mean(pa(0.0)):.4f} | a=1/137 {np.mean(pa(1/137)):.4f} | a=1/3 {np.mean(pa(1/3)):.4f}")
    d, se, nf, ok = pareado(col('G1', .25), col('H0', .25))
    print(f'(sem critério) G1 − H0 (pista 25%): {d:+.4f} (EP {se:.4f}, {nf}/10)')
    json.dump({'eco': eco, 'escolhida': escolhida, 'R': {c: {str(fl): v for fl, v in R[c].items()} for c in R},
               'A': {str(a): v for a, v in A.items()}}, open('memoria_esferica_etapa1.json', 'w'))

