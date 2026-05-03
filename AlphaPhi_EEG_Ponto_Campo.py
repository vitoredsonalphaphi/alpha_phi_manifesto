# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Cole em UMA célula do Google Colab e rode.
#
# Experimento: eco_ponto vs eco_campo vs eco_unissono
#
#   eco_ponto   — modula cada época individualmente (abordagem anterior)
#   eco_campo   — modula a estatística coletiva (espectro médio da população)
#                 fitado no treino, aplicado com a mesma correção no teste
#   eco_unissono — campo → ponto em sequência

import os, subprocess
subprocess.run(['pip', 'install', '-q', 'mne'], check=True)
for run in ['R04','R06','R08','R10']:
    os.system(f'wget -q https://physionet.org/files/eegmmidb/1.0.0/S001/S001{run}.edf -O S001{run}.edf')
print("Download concluído.")

import numpy as np, mne
from scipy import stats
mne.set_log_level('WARNING')

PHI   = (1 + np.sqrt(5)) / 2
K_MIN = np.sqrt(2)
N_ECO = 3; N_SEEDS = 20; N_EP = 60; HIDDEN = 89; LR = 0.01

# ── carregamento ───────────────────────────────────────────────────────────────
def carregar_eeg(arquivos, canal='C3', epoch_len=1.0):
    T1, T2 = [], []
    for arq in arquivos:
        raw = mne.io.read_raw_edf(arq, preload=True, verbose=False)
        raw.filter(8, 30, fir_design='firwin', verbose=False)
        fs  = raw.info['sfreq']
        events, _ = mne.events_from_annotations(raw, verbose=False)
        ch_idx = raw.ch_names.index(canal) if canal in raw.ch_names else 0
        data   = raw.get_data()[ch_idx]
        n_pts  = int(epoch_len * fs)
        for ev in events:
            onset, codigo = ev[0], ev[2]
            if onset + n_pts > len(data): continue
            seg = data[onset:onset+n_pts]
            seg = seg / (np.std(seg) + 1e-8)
            idx = np.linspace(0, len(seg)-1, 256).astype(int)
            seg = seg[idx]
            if   codigo == 1: T1.append(seg)
            elif codigo == 2: T2.append(seg)
    print(f"  {canal}: T1={len(T1)}, T2={len(T2)}")
    return np.array(T1), np.array(T2)

# ── eco no PONTO (individual) ──────────────────────────────────────────────────
def medir_campo_pop(X):
    """Coerência da população (usa espectro médio)."""
    fb = np.fft.fft(X, axis=-1)
    am = np.abs(fb).mean(axis=0)
    an = np.clip(am / (am.sum() + 1e-8), 1e-10, 1.0)
    e  = -np.sum(an * np.log(an))
    c  = float(1.0 - e / np.log(X.shape[-1]))
    return K_MIN + (PHI - K_MIN) * c, c

def eco_ponto(X, coupling=PHI):
    """Eco aplicado a cada época individualmente."""
    k, _ = medir_campo_pop(X)
    s = X.copy()
    for _ in range(N_ECO):
        fr  = np.fft.fft(s, axis=-1)
        ref = np.real(np.fft.ifft(np.abs(fr) * np.exp(1j * np.angle(fr) * k), axis=-1))
        s   = s + (ref - X) * coupling
    return s, k

# ── eco no CAMPO (coletivo) ────────────────────────────────────────────────────
def eco_campo_fit(Xtr):
    """
    Fit: calcula a correção φ sobre o espectro médio do treino.
    Retorna a correção (vetor 1D) e diagnósticos.
    """
    F_mean = np.fft.fft(Xtr, axis=-1).mean(axis=0)   # espectro médio do campo
    mag    = np.abs(F_mean)
    phase  = np.angle(F_mean)

    # coerência do campo
    a   = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
    e   = -np.sum(a * np.log(a))
    coh = float(1.0 - e / np.log(len(F_mean)))
    k   = K_MIN + (PHI - K_MIN) * coh

    # φ-envelope sobre o espectro médio
    n_idx   = np.arange(len(F_mean))
    phi_env = 1.0 + coh * np.cos(2.0 * np.pi * n_idx / PHI)
    phi_env = np.clip(phi_env, 0.05, None)

    F_eco   = (mag * phi_env) * np.exp(1j * phase * k)

    # correção = diferença temporal entre campo-eco e campo-original
    campo_eco  = np.real(np.fft.ifft(F_eco))
    campo_orig = np.real(np.fft.ifft(F_mean))
    correcao   = campo_eco - campo_orig          # shape (n_samples,)

    return correcao, k, coh

def eco_campo_apply(X, correcao):
    """Apply: usa a correção fitada no treino."""
    return X + correcao[np.newaxis, :]

# ── rede ───────────────────────────────────────────────────────────────────────
def golden(x): return PHI * np.tanh(x / PHI)
def sig(x):    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def cg(g, mn=1.0):
    n = np.linalg.norm(g); return g * mn / n if n > mn else g

def treinar(Xtr, ytr, Xte, yte, seed):
    d   = Xtr.shape[1]
    rng = np.random.default_rng(seed + 999)
    W1  = rng.normal(0, np.sqrt(2/d), (d, HIDDEN)); b1 = np.zeros(HIDDEN)
    W2  = rng.normal(0, np.sqrt(2/HIDDEN), (HIDDEN, 1)); b2 = np.zeros(1)
    for _ in range(N_EP):
        idx = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr)-31, 32):
            Xb, yb = Xtr[idx[i:i+32]], ytr[idx[i:i+32]]
            z1 = Xb @ W1 + b1; a1 = golden(z1)
            yh = sig(a1 @ W2 + b2).squeeze(); dL = (yh - yb) / len(yb)
            dW2 = a1.T @ dL.reshape(-1, 1); db2 = dL.sum(keepdims=True)
            dz1 = (dL.reshape(-1, 1) * W2.T) * (1 - np.tanh(z1/PHI)**2)
            W1 -= LR * cg(Xb.T @ dz1); b1 -= LR * np.clip(dz1.sum(0), -1, 1)
            W2 -= LR * cg(dW2);         b2 -= LR * np.clip(db2, -1, 1)
    return float(np.mean((sig(golden(Xte @ W1 + b1) @ W2 + b2).squeeze() >= 0.5) == yte))

# ── experimento ────────────────────────────────────────────────────────────────
def rodar(arquivos, canal='C3'):
    print(f"\n── Canal {canal} " + "─"*40)
    T1, T2 = carregar_eeg(arquivos, canal)
    n = min(len(T1), len(T2))
    if n < 12: print("  Épocas insuficientes."); return None
    T1, T2 = T1[:n], T2[:n]
    X = np.vstack([T1, T2]); y = np.array([1]*n + [0]*n, dtype=float)

    import time; TS = int(time.time())
    SEEDS = [TS + i*137 for i in range(N_SEEDS)]
    n_tr  = int(0.7 * len(X))

    rG, rP, rC, rU = [], [], [], []
    k_ps, k_cs, coh_cs = [], [], []

    print(f"  {'seed':>6} {'G':>7} {'Ponto':>7} {'Campo':>7} {'Uníssono':>9} {'k_p':>8} {'k_c':>8} {'coh_c':>7}")

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        Xs, ys   = X[idx], y[idx]
        Xtr, ytr = Xs[:n_tr], ys[:n_tr]
        Xte, yte = Xs[n_tr:], ys[n_tr:]

        # baseline
        aG = treinar(Xtr, ytr, Xte, yte, seed)

        # eco no ponto
        Xtr_p, k_p = eco_ponto(Xtr)
        Xte_p, _   = eco_ponto(Xte)
        aP = treinar(Xtr_p, ytr, Xte_p, yte, seed)

        # eco no campo (fit treino, apply treino+teste)
        corr, k_c, coh_c = eco_campo_fit(Xtr)
        Xtr_c = eco_campo_apply(Xtr, corr)
        Xte_c = eco_campo_apply(Xte, corr)
        aC = treinar(Xtr_c, ytr, Xte_c, yte, seed)

        # eco uníssono: campo → ponto
        Xtr_u, _ = eco_ponto(Xtr_c)
        Xte_u, _ = eco_ponto(Xte_c)
        aU = treinar(Xtr_u, ytr, Xte_u, yte, seed)

        rG.append(aG); rP.append(aP); rC.append(aC); rU.append(aU)
        k_ps.append(k_p); k_cs.append(k_c); coh_cs.append(coh_c)

        print(f"  {seed%10000:>6} {aG:>7.3f} {aP:>7.3f} {aC:>7.3f} {aU:>9.3f} {k_p:>8.5f} {k_c:>8.5f} {coh_c:>7.4f}")

    G, P, C, U = map(np.array, [rG, rP, rC, rU])
    _, p_p = stats.wilcoxon(P, G)
    _, p_c = stats.wilcoxon(C, G)
    _, p_u = stats.wilcoxon(U, G)
    _, p_uc = stats.wilcoxon(U, C)
    _, p_up = stats.wilcoxon(U, P)

    print(f"\n  G={G.mean():.4f} | Ponto={P.mean():.4f} | Campo={C.mean():.4f} | Uníssono={U.mean():.4f}")
    print(f"  Δ(Ponto−G)    ={P.mean()-G.mean():+.4f}  p={p_p:.5f}")
    print(f"  Δ(Campo−G)    ={C.mean()-G.mean():+.4f}  p={p_c:.5f}")
    print(f"  Δ(Uníssono−G) ={U.mean()-G.mean():+.4f}  p={p_u:.5f}")
    print(f"  Δ(U−Campo)    ={U.mean()-C.mean():+.4f}  p={p_uc:.5f}")
    print(f"  Δ(U−Ponto)    ={U.mean()-P.mean():+.4f}  p={p_up:.5f}")
    print(f"  k_ponto={np.mean(k_ps):.5f}  k_campo={np.mean(k_cs):.5f}  coh_campo={np.mean(coh_cs):.4f}")

    melhor = max([('G',G.mean()),('Ponto',P.mean()),('Campo',C.mean()),('Uníssono',U.mean())], key=lambda x:x[1])
    conf = '✅ CONFIRMA' if melhor[0] != 'G' and melhor[1] > G.mean() + 0.02 else '⚠️ NÃO CONFIRMA'
    print(f"  Melhor: {melhor[0]}={melhor[1]:.4f}  →  {conf}")

    return {"canal":canal,"n":n,
            "G":float(G.mean()),"P":float(P.mean()),"C":float(C.mean()),"U":float(U.mean()),
            "p_u":float(p_u),"k_p":float(np.mean(k_ps)),"k_c":float(np.mean(k_cs))}

arqs = ["S001R04.edf","S001R06.edf","S001R08.edf","S001R10.edf"]
res  = {c: rodar(arqs, c) for c in ["C3","C4","Cz"]}

print("\n\n══ RESUMO ══════════════════════════════════════════════")
print(f"{'Canal':<6} {'G':>8} {'Ponto':>8} {'Campo':>8} {'Uníssono':>10} {'k_p':>8} {'k_c':>8}")
for c, r in res.items():
    if r: print(f"{c:<6} {r['G']:>8.4f} {r['P']:>8.4f} {r['C']:>8.4f} {r['U']:>10.4f} {r['k_p']:>8.5f} {r['k_c']:>8.5f}")
