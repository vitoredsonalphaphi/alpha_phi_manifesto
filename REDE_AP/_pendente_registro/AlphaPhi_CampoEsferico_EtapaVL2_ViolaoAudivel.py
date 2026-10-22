# EXPERIMENTO: ETAPA VL2 — a Grade Esférica v2 e o violão audível (critérios aprovados pelo pesquisador em 22/10/2026, antes da execução)
# Gravação: violao_20260513.wav. Trechos de 1 s (passo 0,25 s), fora os 10% de menor potência.
# MODO LOCAL (Grade v2): 75 faixas × 8 instantes em 600 células; z = log(energia) − perfil médio da célula; x = tanh(z / desvio) (contínuo).
# MODO HOLOGRÁFICO (v1, contínuo): 225 faixas do segundo inteiro → 225 harmônicos; u = Y·(z/desvio); x = tanh(u / desvio de u).
# Memórias: o mais distante dos já escolhidos (cosseno, |.|), parando em 0,5; até 12 (no modo local; o holográfico usa os mesmos trechos).
# Ruído branco no áudio, +10 dB (principal) e 0 dB; sementes 170–179. Rede: AlphaFi afinada, Ferramentas de Memória v1 sem alteração:
# sem guia (SP) e com o previsor (T). Saída: a altura estimada de cada faixa vira um GANHO (entre 0,05 e 1) aplicado ao próprio som ruidoso.
# Medida principal: ganho de nitidez = SNR(saída, violão limpo) − SNR(ruidoso, violão limpo), em dB.
# V2-00: >= 6 memórias, correlação máxima < 0,5. V2-01: correlação da pista com a forma limpa (modo local) entre 0,2 e 0,8.
# V2-1: local com previsor, ganho de nitidez > 0. V2-2: local − holográfico (com previsor) > 0. V2-3: local com previsor − local sem guia > 0.
# Descritivo: 0 dB; trechos não guardados (0,25 s depois); o teto (ganho com a energia limpa verdadeira, o "oráculo").
import numpy as np, wave, importlib.util, json, sys
from multiprocessing import Pool
import os
H = os.path.dirname(os.path.abspath(__file__)); OUT = H
WAV = os.path.join(H, '..', '..', 'violao_20260513.wav')
def load(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
G2 = load('G2', os.path.join(H, 'AlphaPhi_GradeEsferica_v2_proposta.py')); FM = load('FM', os.path.join(H, '..', 'AlphaPhi_FerramentasMemoria_v1.py'))
G1 = G2.G1; N = G2.N; PTS = G2.P; Y, GR = G1.harmonicos(PTS, 14); YP = np.linalg.pinv(Y)
FS = G2.FS; SEG = G2.SEG; HOPT = FS // 4; PMAX = 12; LIM = 0.5; SEM = list(range(170, 180)); SNRS = [10, 0]
F225 = np.geomspace(40, 20000, 226); _f = np.fft.rfftfreq(G2.NPERSEG, 1 / FS)
B225 = [np.where((_f >= a) & (_f < b))[0] for a, b in zip(F225[:-1], F225[1:])]
B225 = [b if len(b) else np.array([int(np.argmin(abs(_f - np.sqrt(F225[k] * F225[k + 1]))))]) for k, b in enumerate(B225)]
def ler():
    with wave.open(WAV) as w: return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
def e225(seg):
    _, t, Z = G2.espectro(seg); A = (np.abs(Z) ** 2).sum(1); return np.array([A[b].sum() for b in B225])
def ganho225(seg, g):
    _, t, Z = G2.espectro(seg); gg = np.zeros(Z.shape[0]); n = np.zeros(Z.shape[0])
    for k, b in enumerate(B225): gg[b] += g[k]; n[b] += 1
    fora = n == 0; gg[~fora] /= n[~fora]; gg[fora] = 1.
    from scipy.signal import istft
    _, y = istft(Z * gg[:, None], FS, nperseg=G2.NPERSEG, noverlap=G2.NPERSEG - G2.HOP, boundary=True); return y[:len(seg)]
def snr(y, c): return 10 * np.log10((c ** 2).sum() / (((y - c) ** 2).sum() + 1e-20))
def cosm(A): n = np.linalg.norm(A, axis=1, keepdims=True); return (A @ A.T) / (n * n.T + 1e-12)
class Tarefa:
    def __init__(self):
        x = ler(); t0s = list(range(0, len(x) - SEG - HOPT + 1, HOPT)); segs = [x[t:t + SEG] for t in t0s]
        pot = np.array([np.mean(s ** 2) for s in segs]); vivo = pot > np.quantile(pot, .10)
        self.x = x; self.t0s = [t for t, v in zip(t0s, vivo) if v]; self.segs = [s for s, v in zip(segs, vivo) if v]
        E = np.array([G2.energias(s) for s in self.segs]); self.eps = 1e-10 * E.mean()
        Lg = np.log(E + self.eps); self.perfil = Lg.mean(0); self.sd = (Lg - self.perfil).std()
        XL = np.array([G2.na_esfera(np.tanh((l - self.perfil) / self.sd)) for l in Lg])
        C = np.abs(cosm(XL)); esc = [0]
        while len(esc) < PMAX:
            m = C[:, esc].max(1); m[esc] = np.inf; i = int(np.argmin(m))
            if m[i] >= LIM: break
            esc.append(i)
        self.esc = esc; self.XL = XL[esc]
        E2 = np.array([e225(self.segs[i]) for i in range(len(self.segs))]); self.eps2 = 1e-10 * E2.mean(); L2 = np.log(E2 + self.eps2)
        self.perfil2 = L2.mean(0); self.sd2 = (L2 - self.perfil2).std(); U = ((L2 - self.perfil2) / self.sd2) @ Y.T; self.sdu = U.std()
        self.XH = np.tanh(U[esc] / self.sdu)
    def pista_local(self, seg): return G2.na_esfera(np.tanh((np.log(G2.energias(seg) + self.eps) - self.perfil) / self.sd))
    def pista_holo(self, seg): return np.tanh((((np.log(e225(seg) + self.eps2) - self.perfil2) / self.sd2) @ Y.T) / self.sdu)
    def decod_local(self, S, ruidoso):
        z = self.sd * np.arctanh(np.clip(G2.da_esfera(S), -.999, .999)); Eh = np.exp(z + self.perfil); En = G2.energias(ruidoso) + self.eps
        return G2.aplicar_ganho(ruidoso, np.clip(np.sqrt(Eh / En), .05, 1.))
    def decod_holo(self, S, ruidoso):
        u = self.sdu * np.arctanh(np.clip(S, -.999, .999)); z = self.sd2 * (YP @ u); Eh = np.exp(z + self.perfil2); En = e225(ruidoso) + self.eps2
        return ganho225(ruidoso, np.clip(np.sqrt(Eh / En), .05, 1.))
    def oraculo(self, limpo, ruidoso):
        return G2.aplicar_ganho(ruidoso, np.clip(np.sqrt((G2.energias(limpo) + self.eps) / (G2.energias(ruidoso) + self.eps)), .05, 1.))
def ruido(seg, db, rng):
    r = rng.normal(size=len(seg)); r *= np.sqrt(np.mean(seg ** 2) / 10 ** (db / 10) / np.mean(r ** 2)); return seg + r
def rodar(arg, guardar=False, so=None):
    db, quais = arg; T = Tarefa(); P = len(T.esc); JL = FM.ligacoes(PTS, T.XL); JH = FM.ligacoes(PTS, T.XH)
    res = {k: [] for k in ('L-SP', 'L-T', 'H-SP', 'H-T', 'oraculo', 'pista_corr')}; filme = None
    for s in (SEM if so is None else [so]):
        if quais == 'guardados': idx = T.esc
        else: idx = [min(i + 1, len(T.segs) - 1) for i in T.esc]          # o trecho 0,25 s depois
        limpos = [T.segs[i] for i in idx]; ruid = [ruido(c, db, np.random.default_rng([s, db + 50, k])) for k, c in enumerate(limpos)]
        SL = np.array([T.pista_local(r) for r in ruid]); SH = np.array([T.pista_holo(r) for r in ruid])
        alvo = np.array([T.pista_local(c) for c in limpos]); cel = G2.END.ravel()
        res['pista_corr'].append(float(np.mean([np.corrcoef(SL[k, cel], alvo[k, cel])[0, 1] for k in range(P)])))
        rd = lambda: np.random.default_rng([2_500_000 + s, db + 50]); saida = {}; hist = {}
        for nome, J, X, S0, kw in (('L-SP', JL, T.XL, SL, dict(usar_previsor=False)), ('L-T', JL, T.XL, SL, dict(usar_previsor=True)),
                                   ('H-SP', JH, T.XH, SH, dict(usar_previsor=False)), ('H-T', JH, T.XH, SH, dict(usar_previsor=True))):
            Sf, hist[nome] = FM.recuperar(J, X, S0, rd(), guardar=True, **kw); saida[nome] = Sf
            dec = T.decod_local if nome[0] == 'L' else T.decod_holo
            res[nome].append(float(np.mean([snr(dec(Sf[k], ruid[k]), limpos[k]) - snr(ruid[k], limpos[k]) for k in range(P)])))
        res['oraculo'].append(float(np.mean([snr(T.oraculo(limpos[k], ruid[k]), limpos[k]) - snr(ruid[k], limpos[k]) for k in range(P)])))
        if guardar: filme = dict(T=T, limpos=limpos, ruid=ruid, SL=SL, SH=SH, saida=saida, hist=hist, idx=idx)
    print(f'  {db:+d} dB {quais} pronto', file=sys.stderr, flush=True)
    return (arg, res, filme) if guardar else (arg, res)

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'sanidade':
    T = Tarefa(); P = len(T.esc); C = cosm(T.XL)[np.triu_indices(P, 1)]
    print(f'{len(T.segs)} trechos candidatos; {P} memórias; trechos (s): ' + ' '.join(f'{T.t0s[i] / FS:.2f}' for i in T.esc))
    print(f"V2-00 memórias {P} · correlação máxima {np.abs(C).max():.3f} · média {C.mean():+.3f} -> {'cumprido' if P >= 6 and np.abs(C).max() < .5 else 'NÃO cumprido'}")
    cel = G2.END.ravel()
    for db in SNRS:
        cc = [np.corrcoef(T.pista_local(ruido(T.segs[i], db, np.random.default_rng([s, db + 50, k])))[cel], T.XL[k, cel])[0, 1] for k, i in enumerate(T.esc) for s in SEM]
        print(f"  pista a {db:+d} dB: correlação com a forma limpa {np.mean(cc):.3f} (mín {np.min(cc):.3f}, máx {np.max(cc):.3f})")
if __name__ == '__main__' and len(sys.argv) == 1:
    with Pool(4) as pool: out = dict(pool.map(rodar, [(10, 'guardados'), (0, 'guardados'), (10, 'novos'), (0, 'novos')]))
    for (db, q), R in out.items():
        print(f'\n##### {db:+d} dB · trechos {q} · ganho de nitidez em dB (média de 10 ruídos; correlação da pista {np.mean(R["pista_corr"]):.3f})')
        for k in ('L-SP', 'L-T', 'H-SP', 'H-T', 'oraculo'): print(f'   {k:<8}{np.mean(R[k]):+8.2f} dB   (EP {np.std(R[k], ddof=1) / np.sqrt(10):.2f})')
    R = out[(10, 'guardados')]; col = lambda k: np.array(R[k])
    def teste(nome, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{nome}: {d.mean():+.2f} dB (EP {se:.2f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    print('\n=== CRITÉRIOS (+10 dB, trechos guardados) ===')
    teste('V2-1 local com previsor: ganho de nitidez', col('L-T')); teste('V2-2 local − holográfico (com previsor)', col('L-T') - col('H-T'))
    teste('V2-3 local: com previsor − sem guia', col('L-T') - col('L-SP'))
    json.dump({f'{db}|{q}': v for (db, q), v in out.items()}, open(OUT + '/EtapaVL2_resultados.json', 'w'))
