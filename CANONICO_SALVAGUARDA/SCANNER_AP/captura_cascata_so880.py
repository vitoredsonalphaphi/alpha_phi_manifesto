# OBSERVAÇÃO: as seis fases da cascata do EcoBIP 880 (o beep e os cinco passos do eco), cada uma levada à esfera pelos anéis da Grade v2
import numpy as np, importlib.util, json
from scipy.signal import stft
from scipy.spatial import ConvexHull
def banda(P,f,lo,hi):
    b=np.where((f>=lo)&(f<hi))[0]
    if len(b): return P[b].mean(0)
    c=np.sqrt(lo*hi); return np.array([np.interp(c,f,P[:,j]) for j in range(P.shape[1])])  # faixa mais estreita que a resolução: interpola
SP='/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
def load(n,p): s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
D=load('D',SP+'exp_etapaDN.py'); NA=D.NA; G=NA.GV2; N=D.N
import so880 as O
FS=O.FS; cas=O.CAS
LIN,COL=61,10; edges=np.geomspace(40,16000,LIN+1)
B=[]
for s in cas:
    f,t,Z=stft(s,FS,nperseg=2048,noverlap=1536); P=np.abs(Z)**2
    M=np.log10(np.array([banda(P,f,edges[i],edges[i+1]) for i in range(LIN)])+1e-10)[:,20:20+COL]; B.append(M)
B=np.array(B); mu,sd=B.mean(),B.std()
Eend=G.enderecos_aneis(LIN,COL)
fases=[]
for M in B:
    v=np.zeros(N); v[Eend.ravel()]=np.tanh((M-mu)/sd).ravel(); fases.append(v)
fases=np.array(fases)
U=NA.P2/np.linalg.norm(NA.P2,axis=1,keepdims=True); L=G.lmax_mensuravel(N); Y,gr=G.harmonicos(U,L)
en=[np.round(G.decompor(f,Y,gr)[1],5).tolist() for f in fases]
for k in range(len(fases)): print('fase',k,'· correlação com a fase anterior:',round(float(np.corrcoef(fases[k],fases[k-1])[0,1]),3) if k else '-', '· energia média nos anéis:',round(float(fases[k].mean()),3))
json.dump({'pts':np.round(U,4).tolist(),'faces':ConvexHull(U).simplices.tolist(),'E':Eend.tolist(),'LIN':LIN,'COL':COL,
 'fases':np.round(fases,3).tolist(),'en':en,'edges':[round(float(e)) for e in edges]},open('dados_cascata_so880.json','w'))
