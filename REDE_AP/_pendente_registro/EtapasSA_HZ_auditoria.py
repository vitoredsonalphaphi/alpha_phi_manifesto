import numpy as np, importlib.util
SP='/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
s=importlib.util.spec_from_file_location('M',SP+'exp_etapaSA_HZ.py'); M=importlib.util.module_from_spec(s); s.loader.exec_module(M)
NA=M.NA; P=M.P; s0=250; X=NA.memorias(s0); S0=NA.pista(X,2.0,s0); J=M.LA.ligacao_delta(X); g=NA.ganho_ajustado(J,X)
alt=lambda S: float(np.mean([np.corrcoef(S[k],X[k])[0,1] for k in range(P)]))
print('ruído 2,0, semente 250 — altura média ao longo das voltas (a pista começa em', round(alt(S0),3),')')
for v,(bf,lf) in M.SA_V.items():
    linha=[]
    for nv in (1,3,5,10,20,30,50):
        S=M.recuperar(J,X,S0,np.random.default_rng([6_000_000+s0,20]),g,bf,lf,nvoltas=nv,parada=False); linha.append(round(alt(S),3))
    print(f'  {v:<4}', linha)
mix=M.previsor(S0,X,M.B8/M.PHI**4); m=np.sign(S0)@X.T/610
print('previsor com β = 8/φ⁴ na pista: peso da memória certa (média)', round(float(np.mean(np.exp(1.167*(m-m.max(1,keepdims=True)))[np.arange(P),np.arange(P)]/np.exp(1.167*(m-m.max(1,keepdims=True))).sum(1))),3), '· (com β = 8:', round(float(np.mean(np.exp(8*(m-m.max(1,keepdims=True)))[np.arange(P),np.arange(P)]/np.exp(8*(m-m.max(1,keepdims=True))).sum(1))),3),')')
print('separação das ressonâncias na pista: certa', round(float(np.mean(np.diag(m))),3),'· maior errada', round(float(np.mean(np.sort(m-np.diag(np.diag(m))-np.eye(P)*9,1)[:,-1])),3))
