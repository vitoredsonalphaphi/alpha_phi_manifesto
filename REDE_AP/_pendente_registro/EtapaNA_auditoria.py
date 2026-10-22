import numpy as np, importlib.util
s=importlib.util.spec_from_file_location('NA',__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'AlphaPhi_CampoEsferico_EtapaNA_NeuronioAlturas.py')); NA=importlib.util.module_from_spec(s); s.loader.exec_module(NA)
FM=NA.FM; P=NA.P
fid1,fid3,ponto,leit=[],[],[],[]
for sd in NA.SEM:
    X=NA.memorias(sd); J=NA.FM.ligacoes(NA.P2,X); g1=NA.ganho_ajustado(J,X)
    hesc=np.abs(X@J).mean(1,keepdims=True); h=(X@J)/hesc+FM.FORCA_PREVISOR*X
    for g,L in ((g1,fid1),(3.0,fid3)):
        Z=np.tanh(g*h); L.append(np.mean([np.corrcoef(Z[k],X[k])[0,1] for k in range(P)]))
    Hn=(X@J)/hesc; ponto.append(np.mean([np.corrcoef(Hn[k],X[k])[0,1] for k in range(P)]))
    S0=NA.pista(X,1.0,sd); S=NA.recuperar(J,X,S0,np.random.default_rng([2_700_000+sd,10]),g1,True,True)
    w=np.exp(8*((np.sign(S)@X.T/610)-(np.sign(S)@X.T/610).max(1,keepdims=True))); w/=w.sum(1,keepdims=True); Lx=w@X
    leit.append(np.mean([np.corrcoef(Lx[k],X[k])[0,1] for k in range(P)]))
print(f"a memória intacta passa pelo campo de ligações: correlação do campo com a memória {np.mean(ponto):.3f}")
print(f"a memória intacta passa pelo neurônio: com o ganho ajustado {np.mean(fid1):.3f} · com o ganho 3 {np.mean(fid3):.3f}")
print(f"(descritivo, não previsto) ler o resultado pelo previsor no fim (a mistura das memórias pesada pela ressonância, N3): {np.mean(leit):.3f}")
