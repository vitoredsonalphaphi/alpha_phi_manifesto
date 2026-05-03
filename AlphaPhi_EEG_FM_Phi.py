# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Cole em UMA célula do Google Colab e rode.

import os, subprocess
subprocess.run(['pip', 'install', '-q', 'mne'], check=True)
# 4 runs de imagery: R04+R08 (fist imagery) e R06+R10 (fists/feet imagery)
for run in ['R04','R06','R08','R10']:
    os.system(f'wget -q https://physionet.org/files/eegmmidb/1.0.0/S001/S001{run}.edf -O S001{run}.edf')
print("Download concluído.")

import numpy as np, mne
from scipy import stats
mne.set_log_level('WARNING')

PHI   = (1 + np.sqrt(5)) / 2
K_MIN = np.sqrt(2)
N_ECO = 3; N_SEEDS = 20; N_EP = 60; HIDDEN = 89; LR = 0.01

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

def medir_campo(X):
    fb  = np.fft.fft(X, axis=-1)
    am  = np.abs(fb).mean(axis=0)
    an  = np.clip(am/(am.sum()+1e-8), 1e-10, 1.0)
    ent = -np.sum(an * np.log(an))
    coh = float(1.0 - ent / np.log(X.shape[-1]))
    return K_MIN + (PHI - K_MIN)*coh, coh

def eco(X, coupling):
    k, _ = medir_campo(X); s = X.copy()
    for _ in range(N_ECO):
        fr  = np.fft.fft(s, axis=-1)
        ref = np.real(np.fft.ifft(np.abs(fr)*np.exp(1j*np.angle(fr)*k), axis=-1))
        s   = s + (ref - X) * coupling
    return s, k

def modular_fm_phi(X, beta=PHI, alpha_mix=0.3):
    """
    Pré-modulação FM-φ:
      - detecta frequência dominante f_c de cada época (em bins)
      - cria componente FM: sin(2π·f_c·t + β·sin(2π·f_m·t))  com f_m = f_c/φ
      - mistura com original: (1-α)·original + α·FM
    """
    out = np.zeros_like(X)
    N   = X.shape[-1]
    t   = np.linspace(0, 1, N, endpoint=False)
    for i, x in enumerate(X):
        F       = np.fft.rfft(x)
        dom_bin = int(np.argmax(np.abs(F[1:])) + 1)
        f_c     = float(dom_bin)
        f_m     = f_c / PHI
        fm_comp = np.sin(2*np.pi*f_c*t + beta * np.sin(2*np.pi*f_m*t))
        fm_comp = fm_comp / (np.max(np.abs(fm_comp)) + 1e-10)
        x_norm  = x / (np.std(x) + 1e-8)
        mixed   = (1 - alpha_mix)*x_norm + alpha_mix*fm_comp
        out[i]  = mixed / (np.max(np.abs(mixed)) + 1e-10)
    return out

def golden(x): return PHI * np.tanh(x/PHI)
def sig(x):    return 1/(1+np.exp(-np.clip(x,-500,500)))
def cg(g, mn=1.0):
    n=np.linalg.norm(g); return g*mn/n if n>mn else g

def treinar(Xtr, ytr, Xte, yte, seed):
    d = Xtr.shape[1]
    rng = np.random.default_rng(seed+999)
    W1=rng.normal(0,np.sqrt(2/d),(d,HIDDEN)); b1=np.zeros(HIDDEN)
    W2=rng.normal(0,np.sqrt(2/HIDDEN),(HIDDEN,1)); b2=np.zeros(1)
    for _ in range(N_EP):
        idx=rng.permutation(len(Xtr))
        for i in range(0,len(Xtr)-31,32):
            Xb,yb=Xtr[idx[i:i+32]],ytr[idx[i:i+32]]
            z1=Xb@W1+b1; a1=golden(z1)
            yh=sig(a1@W2+b2).squeeze(); dL=(yh-yb)/len(yb)
            dW2=a1.T@dL.reshape(-1,1); db2=dL.sum(keepdims=True)
            dz1=(dL.reshape(-1,1)*W2.T)*(1-np.tanh(z1/PHI)**2)
            W1-=LR*cg(Xb.T@dz1); b1-=LR*np.clip(dz1.sum(0),-1,1)
            W2-=LR*cg(dW2); b2-=LR*np.clip(db2,-1,1)
    return float(np.mean((sig(golden(Xte@W1+b1)@W2+b2).squeeze()>=0.5)==yte))

def rodar(arquivos, canal='C3'):
    print(f"\n── Canal {canal} " + "─"*40)
    T1, T2 = carregar_eeg(arquivos, canal)
    n = min(len(T1), len(T2))
    if n < 12: print("  Épocas insuficientes."); return None
    T1, T2 = T1[:n], T2[:n]
    X = np.vstack([T1,T2]); y = np.array([1]*n+[0]*n, dtype=float)

    import time; TS = int(time.time())
    SEEDS = [TS + i*137 for i in range(N_SEEDS)]
    n_tr  = int(0.7*len(X))

    rG, rV2, rFM, ks, ks_fm = [], [], [], [], []
    print(f"  {'seed':>6} {'G':>7} {'V2':>7} {'FM+V2':>8} {'k_raw':>8} {'k_fm':>8}")

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        Xs,ys = X[idx],y[idx]
        Xtr,ytr = Xs[:n_tr],ys[:n_tr]
        Xte,yte = Xs[n_tr:],ys[n_tr:]

        aG = treinar(Xtr,ytr,Xte,yte,seed)

        k_raw = medir_campo(Xtr)[0]
        Xtr2,_ = eco(Xtr, PHI); Xte2,_ = eco(Xte, PHI)
        aV2 = treinar(Xtr2,ytr,Xte2,yte,seed)

        Xtr_fm = modular_fm_phi(Xtr)
        Xte_fm = modular_fm_phi(Xte)
        k_fm   = medir_campo(Xtr_fm)[0]
        Xtr3,_ = eco(Xtr_fm, PHI); Xte3,_ = eco(Xte_fm, PHI)
        aFM = treinar(Xtr3,ytr,Xte3,yte,seed)

        rG.append(aG); rV2.append(aV2); rFM.append(aFM)
        ks.append(k_raw); ks_fm.append(k_fm)
        print(f"  {seed%10000:>6} {aG:>7.3f} {aV2:>7.3f} {aFM:>8.3f} {k_raw:>8.5f} {k_fm:>8.5f}")

    G,V2,FM = np.array(rG),np.array(rV2),np.array(rFM)
    _,p_v2   = stats.wilcoxon(V2,G)
    _,p_fm   = stats.wilcoxon(FM,G)
    _,p_fmv2 = stats.wilcoxon(FM,V2)

    print(f"\n  G={G.mean():.4f} | V2={V2.mean():.4f} | FM+V2={FM.mean():.4f}")
    print(f"  Δ(V2−G)  ={V2.mean()-G.mean():+.4f}  p={p_v2:.5f}")
    print(f"  Δ(FM−G)  ={FM.mean()-G.mean():+.4f}  p={p_fm:.5f}")
    print(f"  Δ(FM−V2) ={FM.mean()-V2.mean():+.4f}  p={p_fmv2:.5f}")
    print(f"  k_raw={np.mean(ks):.5f}  k_fm={np.mean(ks_fm):.5f}")
    conf = '✅ CONFIRMA' if FM.mean()>G.mean() and p_fm<0.05 else '⚠️ NÃO CONFIRMA'
    print(f"  FM+V2 vs G: {conf}")
    return {"canal":canal,"n":n,"G":float(G.mean()),
            "V2":float(V2.mean()),"FM":float(FM.mean()),
            "p_fm":float(p_fm),"k_raw":float(np.mean(ks)),"k_fm":float(np.mean(ks_fm))}

arqs = ["S001R04.edf","S001R06.edf","S001R08.edf","S001R10.edf"]
res  = {c: rodar(arqs,c) for c in ["C3","C4","Cz"]}

print("\n\n══ RESUMO ══════════════════════════════════════")
print(f"{'Canal':<6} {'G':>8} {'V2':>8} {'FM+V2':>8} {'k_raw':>8} {'k_fm':>8}")
for c,r in res.items():
    if r: print(f"{c:<6} {r['G']:>8.4f} {r['V2']:>8.4f} {r['FM']:>8.4f} {r['k_raw']:>8.5f} {r['k_fm']:>8.5f}")
