import numpy as np, importlib.util
SP='/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
s=importlib.util.spec_from_file_location('A',SP+'exp_etapaAU.py'); A=importlib.util.module_from_spec(s); s.loader.exec_module(A)
M,NA,LA=A.M,A.NA,A.LA; s0=280; X=NA.memorias(s0); T,caps=A.alvo(X,s0,.10); S0=NA.pista(T,1.0,s0); J=LA.ligacao_delta(X); g=NA.ganho_ajustado(J,X)
print('ruído 1,0, região 10%, semente 280 — preservação e coerência ao longo das voltas (com previsor)')
for nv in (1,2,3,5,10,50):
    S=M.recuperar(J,X,S0,np.random.default_rng([9_200_000+s0,10,10]),g,nvoltas=nv,parada=False); m=A.medir(S,T,caps); print(f'  volta {nv:>2}: preservação {m["pres"]:+.3f} · coerência {m["coer"]:.3f}')
S=NA.recuperar(J,X,S0,np.random.default_rng(1),g,True,False); m=A.medir(S,T,caps); print(f'sem previsor (só as ligações e a âncora): preservação {m["pres"]:+.3f} · coerência {m["coer"]:.3f} · total {m["total"]:.3f}')
Z=S0@J; m=A.medir(Z,T,caps); print(f'uma passagem pelas ligações delta (sem neurônio): preservação {m["pres"]:+.3f} · coerência {m["coer"]:.3f}')
