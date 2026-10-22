import numpy as np, importlib.util
SP='/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
s=importlib.util.spec_from_file_location('V4',SP+'exp_etapaVL4.py'); V=importlib.util.module_from_spec(s); s.loader.exec_module(V)
T=V.V3.Tarefa(); X=T.X; P=len(T.esc); J=V.LA.ligacao_delta(X); g=V.NA.ganho_ajustado(J,X); bc=V.beta_calibrado(X)
C=V.resson(X,X); print('ressonância entre memórias (máx fora da diagonal por memória):',np.round(np.sort(C-np.eye(P)*9,1)[:,-1],3).tolist())
for s in (210,215):
    ruid=[V.V2.ruido(T.segs[i],10,np.random.default_rng([s,60,k])) for k,i in enumerate(T.esc)]
    S0=np.array([T.pista(r) for r in ruid]); Sf=V.NA.recuperar(J,X,S0,np.random.default_rng([3_000_000+s,60]),g,True,True)
    print('semente',s,'escolha na pista',V.resson(S0,X).argmax(1).tolist(),'· no fim',V.resson(Sf,X).argmax(1).tolist(),'· peso máx (β cal.) no fim',np.round(V.pesos(Sf,X,bc).max(1),3).min())
