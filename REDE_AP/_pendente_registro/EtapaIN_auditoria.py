import numpy as np, importlib.util
SP='/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
s=importlib.util.spec_from_file_location('IN',SP+'exp_etapaIN.py'); M=importlib.util.module_from_spec(s); s.loader.exec_module(M)
NA=M.NA; X=NA.memorias(230); S0=NA.pista(X,2.0,230); C=M.C233
c=lambda A,B: float(np.mean([np.corrcoef(A[k],B[k])[0,1] for k in range(len(A))]))
Xi=M.descer(X,C); Si0=M.descer(S0,C); Ji=C.delta(Xi); gi=NA.ganho_ajustado(Ji,Xi)
print('energia da memória por faixa de grau (fração): <=9', round(float((M.coef_sup(X)[:, M.G2<=9]**2).sum()/(M.coef_sup(X)**2).sum()),3), '· resíduo acima de 14 (fora do ajuste):', round(float(1-((M.coef_sup(X)@M.Y2.T)**2).sum()/(X**2).sum()),3))
print('na casca 233: desvio das memórias', np.round(Xi.std(1).mean(),3), '· correlação pista×memória', round(c(Si0,Xi),3), '· ganho ajustado', gi)
print('memória intacta pelas ligações da casca:', round(c(Xi@Ji,Xi),4))
Sf,v=M.recuperar(Ji,Xi,Si0,np.random.default_rng(1),gi,C.W); print('recuperada na casca:', round(c(Sf,Xi),3), '· voltas', v, '· desvio do estado final', np.round(Sf.std(1).mean(),3))
R=np.sign(Xi)@Xi.T/233; d=np.diag(R); print('ressonância cruzada máxima relativa à própria, na casca:', round(float(np.max(np.abs(R-np.diag(d)))/d.mean()),3))
