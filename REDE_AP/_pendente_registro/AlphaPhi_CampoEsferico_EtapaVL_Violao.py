# EXPERIMENTO: ETAPA VL — o violão na rede AP (critérios aprovados pelo pesquisador em 21/10/2026, antes da execução)
# Gravação: violao_20260513.wav (do pesquisador, 13/05/2026; 94 s, mono, 44,1 kHz). Quadros do STFT: janela Hann de 1 s, passo 0,25 s;
# fora os 10% de menor energia. Tradução: 225 faixas log de 40 Hz a 20 kHz (faixa vazia recebe o ponto mais próximo) → 225 harmônicos;
# log(energia relativa + 1e-6) menos o perfil médio dos candidatos; padrão = sinal(Y·z). Memórias: o mais distante dos já escolhidos,
# parando quando o próximo teria correlação máxima |.| >= 0,5; até 12; começa no primeiro quadro não silencioso.
# Estrago: ruído branco somado ao ÁUDIO do trecho (SNR 10, 0 [principal], −5 dB, pela potência do trecho); 10 ruídos (sementes 160–169).
# Versões (AlphaFi afinada, Ferramentas de Memória v1): V0 binária · SP sem guia · T previsor · EX-G estrutura de êxito (graus 0–4).
# Decisão aprovada depois da sanidade (só a medida da pista, sem resultado da rede): nível principal = +10 dB (0 dB deu sobreposição
# 0,124, abaixo da faixa 0,2–0,8); 0 e −5 dB ficam descritivos.
# VL00: >= 6 memórias, correlação média < 0,3 e máxima < 0,5. VL01: sobreposição da pista (0 dB) com a forma limpa entre 0,2 e 0,8.
# VL1: SP limpa (conclusão final − sobreposição da pista > 0). VL2: T > SP. VL3: EX-G > SP. VL4: identificação com T >= 0,8.
# (pista a +10 dB; pareado > 2 EP e >= 8/10.)
import numpy as np, wave, importlib.util, json, sys
from multiprocessing import Pool
import os
H = os.path.dirname(os.path.abspath(__file__)); OUT = H
WAV = os.path.join(H, '..', '..', 'violao_20260513.wav')
_s = importlib.util.spec_from_file_location('EX', os.path.join(H, 'AlphaPhi_CampoEsferico_EtapaEX_Estrutura.py')); EX = importlib.util.module_from_spec(_s); _s.loader.exec_module(EX)
FM, G, N, Y, GR, YP = EX.FM, EX.G, EX.N, EX.Y, EX.GR, EX.YP
FS = 44100; WIN = FS; HOP = FS // 4; NB = 225; PMAX = 12; LIM = 0.5; SEM = list(range(160, 170)); SNRS = [10, 0, -5]
FAIXAS = np.geomspace(40, 20000, NB + 1); CENTROS = np.sqrt(FAIXAS[:-1] * FAIXAS[1:]); f = np.fft.rfftfreq(WIN, 1 / FS); JAN = np.hanning(WIN)
IDX = [np.where((f >= a) & (f < b))[0] for a, b in zip(FAIXAS[:-1], FAIXAS[1:])]
IDX = [ix if len(ix) else np.array([int(np.argmin(abs(f - c)))]) for ix, c in zip(IDX, CENTROS)]
def ler():
    with wave.open(WAV) as w: x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
    return x
def energia_faixas(seg):
    F = np.abs(np.fft.rfft(seg * JAN)) ** 2; e = np.array([F[ix].sum() for ix in IDX]); return e / F.sum()
def preparar():
    x = ler(); t0s = list(range(0, len(x) - WIN + 1, HOP)); segs = [x[t:t + WIN] for t in t0s]
    pot = np.array([np.mean(s ** 2) for s in segs]); vivo = pot > np.quantile(pot, 0.10)
    segs = [s for s, v in zip(segs, vivo) if v]; t0s = [t for t, v in zip(t0s, vivo) if v]
    Lg = np.log(np.array([energia_faixas(s) for s in segs]) + 1e-6); perfil = Lg.mean(0)
    Xc = np.where((Lg - perfil) @ Y.T >= 0, 1., -1.); C = np.abs(Xc @ Xc.T / N); esc = [0]
    while len(esc) < PMAX:
        m = C[:, esc].max(1); m[esc] = np.inf; i = int(np.argmin(m))
        if m[i] >= LIM: break
        esc.append(i)
    return x, segs, t0s, perfil, Xc[esc], esc, len(segs)
def pista(seg, snr, perfil, rng):
    ruido = rng.normal(size=len(seg)); ruido *= np.sqrt(np.mean(seg ** 2) / 10 ** (snr / 10) / np.mean(ruido ** 2))
    z = np.log(energia_faixas(seg + ruido) + 1e-6) - perfil; return np.where(Y @ z >= 0, 1., -1.)

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'sanidade':
    x, segs, t0s, perfil, X, esc, nc = preparar(); P = len(X); Cm = X @ X.T / N; tri = Cm[np.triu_indices(P, 1)] if P > 1 else np.array([0.])
    print(f'{nc} quadros não silenciosos; {P} memórias; trechos (s): ' + ' '.join(f'{t0s[i] / FS:.2f}' for i in esc))
    print(f"VL00 memórias {P} · correlação média {tri.mean():+.3f} · máxima {tri.max():+.3f} -> {'cumprido' if P >= 6 and tri.mean() < .3 and tri.max() < .5 else 'NÃO cumprido'}")
    for snr in SNRS:
        ov = [float((pista(segs[i], snr, perfil, np.random.default_rng([s, snr + 50, k])) * X[k]).mean()) for k, i in enumerate(esc) for s in SEM]
        print(f"  pista a {snr:+d} dB: sobreposição com a forma limpa {np.mean(ov):.3f} (mín {np.min(ov):.3f}, máx {np.max(ov):.3f})")

VERS = ['V0', 'SP', 'T', 'EX-G']
def rodar_snr(snr, guardar=False, so_semente=None):
    x, segs, t0s, perfil, X, esc, nc = preparar(); J = FM.ligacoes(EX.PTS, X); P = len(X)
    res = {v: {k: [] for k in ('conc', 'pista', 'limpa', 'ident', 'gros', 'fino')} for v in VERS}; filme = None
    for s in (SEM if so_semente is None else [so_semente]):
        S0 = np.array([pista(segs[i], snr, perfil, np.random.default_rng([s, snr + 50, k])) for k, i in enumerate(esc)])
        ov0 = float((S0 * X).mean()); rd = lambda: np.random.default_rng([2_400_000 + s, snr + 50]); out = {}; hist = {}
        out['V0'] = FM.recuperar_binaria(J, S0, rd()); hist['V0'] = [out['V0']]
        for v, kw in (('SP', dict(usar_previsor=False)), ('T', dict(usar_previsor=True))):
            out[v], hist[v] = FM.recuperar(J, X, S0, rd(), guardar=True, **kw)
        out['EX-G'], hist['EX-G'] = EX.recuperar_guiado(J, S0, rd(), EX.guia('EX-G', X, None), True)
        for v in VERS:
            m = EX.medir(out[v], X); res[v]['conc'].append(m['conc']); res[v]['pista'].append(ov0); res[v]['limpa'].append(m['conc'] - ov0)
            for k in ('ident', 'gros', 'fino'): res[v][k].append(m[k])
        if guardar: filme = dict(X=X, S0=S0, out=out, hist=hist, esc=esc, t0s=t0s, segs=segs, perfil=perfil)
    print(f'  {snr:+d} dB pronto', file=sys.stderr, flush=True)
    return (snr, res, filme) if guardar else (snr, res)

if __name__ == '__main__' and len(sys.argv) == 1:
    with Pool(3) as pool: out = dict(pool.map(rodar_snr, SNRS))
    for snr in SNRS:
        R = out[snr]; print(f'\n##### pista a {snr:+d} dB')
        print(f"{'versão':<7}{'pista':>8}{'conclusão':>11}{'limpou':>9}{'grosseiro':>11}{'fino':>8}{'identificação':>15}")
        for v in VERS: print(f"{v:<7}" + ''.join(f"{np.mean(R[v][k]):>{w}.3f}" for k, w in (('pista', 8), ('conc', 11), ('limpa', 9), ('gros', 11), ('fino', 8), ('ident', 15))))
    R = out[10]; col = lambda v, k='conc': np.array(R[v][k])
    def teste(nome, d):
        se = d.std(ddof=1) / np.sqrt(len(d)); nf = (d > 0).sum(); ok = d.mean() > 2 * se and nf >= 8 and d.mean() > 0
        print(f"{nome}: {d.mean():+.3f} (EP {se:.3f}, {nf}/10) -> {'cumprido' if ok else 'NÃO cumprido'}")
    print('\n=== CRITÉRIOS (pista a +10 dB) ===')
    teste('VL1 SP limpa (conclusão − pista)', col('SP', 'limpa')); teste('VL2 T − SP', col('T') - col('SP')); teste('VL3 EX-G − SP', col('EX-G') - col('SP'))
    idt = col('T', 'ident').mean(); print(f"VL4 identificação com o previsor: {idt:.3f} (>= 0,8) -> {'cumprido' if idt >= .8 else 'NÃO cumprido'}")
    json.dump({str(k): v for k, v in out.items()}, open(OUT + '/EtapaVL_resultados.json', 'w'))
