# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Cole tudo isso em UMA célula do Google Colab e rode.

import os, subprocess
subprocess.run(['pip', 'install', '-q', 'mne'], check=True)
os.system('wget -q https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf -O S001R04.edf')
os.system('wget -q https://physionet.org/files/eegmmidb/1.0.0/S001/S001R06.edf -O S001R06.edf')
print("Download concluído.")

import numpy as np, json, mne
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
    X = np.asarray(X, dtype=float)
    k, _ = medir_campo(X); s = X.copy()
    for _ in range(N_ECO):
        fr = np.fft.fft(s, axis=-1)
        ref = np.real(np.fft.ifft(np.abs(fr)*np.exp(1j*np.angle(fr)*k), axis=-1))
        s = s + (ref - X) * coupling
    return s, k

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
    if n < 20:
        print("  Épocas insuficientes."); return None
    T1, T2 = T1[:n], T2[:n]
    X = np.vstack([T1,T2]); y = np.array([1]*n+[0]*n, dtype=float)
    import time; TS = int(time.time())
    SEEDS = [TS + i*137 for i in range(N_SEEDS)]
    n_tr = int(0.7*len(X))
    rG, rV1, rV2, ks = [], [], [], []
    print(f"  {'seed':>6} {'G':>7} {'V1':>7} {'V2':>7} {'k':>8}")
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        Xs,ys = X[idx],y[idx]
        Xtr,ytr,Xte,yte = Xs[:n_tr],ys[:n_tr],Xs[n_tr:],ys[n_tr:]
        aG  = treinar(Xtr,ytr,Xte,yte,seed)
        Xv1,k = eco(Xtr,1.0/medir_campo(Xtr)[0])[0], medir_campo(Xtr)[0]
        aV1 = treinar(eco(Xtr,1/k)[0],ytr,eco(Xte,1/k)[0],yte,seed)
        aV2 = treinar(eco(Xtr,PHI)[0],ytr,eco(Xte,PHI)[0],yte,seed)
        rG.append(aG); rV1.append(aV1); rV2.append(aV2); ks.append(k)
        print(f"  {seed%10000:>6} {aG:>7.3f} {aV1:>7.3f} {aV2:>7.3f} {k:>8.5f}")
    G,V1,V2 = np.array(rG),np.array(rV1),np.array(rV2)
    _,pv2 = stats.wilcoxon(V2,G)
    _,pv2v1 = stats.wilcoxon(V2,V1)
    km = float(np.mean(ks))
    print(f"\n  G={G.mean():.4f} | V1={V1.mean():.4f} | V2={V2.mean():.4f}")
    print(f"  Δ(V2-G)={V2.mean()-G.mean():+.4f}  p={pv2:.6f}")
    print(f"  Δ(V2-V1)={V2.mean()-V1.mean():+.4f}  p={pv2v1:.6f}")
    print(f"  k_campo={km:.5f}  √2={np.sqrt(2):.5f}")
    print(f"  {'✅' if V2.mean()>G.mean() and pv2<0.05 else '⚠️'} Eco V2 vs baseline")
    return {"canal":canal,"n":n,"G":float(G.mean()),"V1":float(V1.mean()),
            "V2":float(V2.mean()),"p":float(pv2),"k":km}

arqs = ["S001R04.edf","S001R06.edf"]
res  = {c: rodar(arqs, c) for c in ["C3","C4","Cz"]}

print("\n\n══ RESUMO ══════════════════════════════")
print(f"{'Canal':<6} {'G':>8} {'V2(φ)':>8} {'Δ':>8} {'k':>10}")
for c,r in res.items():
    if r: print(f"  {c:<4} {r['G']:>8.4f} {r['V2']:>8.4f} {r['V2']-r['G']:>+8.4f} {r['k']:>10.5f}")

with open("eeg_real_results.json","w") as f: json.dump(res,f,indent=2)
print("\nSalvo: eeg_real_results.json")
