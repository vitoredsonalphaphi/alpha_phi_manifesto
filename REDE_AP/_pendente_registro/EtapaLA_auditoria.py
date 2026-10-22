import numpy as np, importlib.util
s=importlib.util.spec_from_file_location('LA',__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'AlphaPhi_CampoEsferico_EtapaLA_LigacoesAlturas.py')); LA=importlib.util.module_from_spec(s); s.loader.exec_module(LA)
NA=LA.NA; r=[]; p=[]; ident=[]
for sd in LA.SEM:
    X=NA.memorias(sd); S0=NA.pista(X,1.0,sd); JD=LA.ligacao_delta(X); g=NA.ganho_ajustado(JD,X)
    S=NA.recuperar(JD,X,S0,np.random.default_rng([2_800_000+sd,10]),g,True,False); m=NA.medir(S,X); r.append(m['alt']); ident.append(m['ident']); p.append(NA.medir(S0,X)['alt'])
print(f"(descritivo) regra delta SEM previsor: altura {np.mean(r):.3f} (pista {np.mean(p):.3f}) · identificação {np.mean(ident):.3f}")
