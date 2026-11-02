# Scanner AP — OBSERVAÇÃO (sem critérios, nada vai ao repositório): o EcoBIP 880 processado na Rede AP (Grade v2, regra delta, N3, previsor)
import numpy as np, importlib.util, json
from scipy.signal import stft
from scipy.spatial import ConvexHull
def banda(P,f,lo,hi):
    b=np.where((f>=lo)&(f<hi))[0]
    if len(b): return P[b].mean(0)
    c=np.sqrt(lo*hi); return np.array([np.interp(c,f,P[:,j]) for j in range(P.shape[1])])  # faixa mais estreita que a resolução: interpola
SP='/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
def load(n,p): s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
D=load('D',SP+'exp_etapaDN.py'); A,M,NA,LA,FM=D.A,D.M,D.NA,D.LA,D.FM; N=D.N; G=NA.GV2
import so880 as O
FS=O.FS
sinais={nm:sig for nm,sig in zip(O.NOMES,O.CAS)}
LIN,COL=61,10; edges=np.geomspace(40,16000,LIN+1)
def quadros(sig):
    f,t,Z=stft(sig,FS,nperseg=2048,noverlap=1536); P=np.abs(Z)**2
    B=np.array([banda(P,f,edges[i],edges[i+1]) for i in range(LIN)]); B=np.log10(B+1e-10)
    out=[]
    for c0 in (20,60):
        Q=B[:,c0:c0+COL]; z=(Q-Q.mean())/(Q.std()+1e-9); out.append(np.tanh(z))
    return out
Eend=G.enderecos_aneis(LIN,COL)
def esfera(Q): v=np.zeros(N); v[Eend.ravel()]=Q.ravel(); return v
nomes=[];X=[]
for k,s in sinais.items():
    for j,Q in enumerate(quadros(s)): nomes.append(f'{k} ({"a" if j==0 else "b"})'); X.append(esfera(Q))
X=np.array(X); K=nomes.index('5.º ponto de dobra (a)')
print('memórias:',len(X),'· sem lugar na esfera:',N-LIN*COL)
J=LA.ligacao_delta(X); g=NA.ganho_ajustado(J,X)
T=X[K:K+1]; S0=NA.pista(np.repeat(T,1,0),1.0,7)  # pista: o EcoBIP 880 com ruído
# modo recuperação
def recup(S0):
    S=S0.copy(); hesc=np.abs(S0@J).mean(1,keepdims=True); mu_t=NA.wmean(X).mean(); sd_t=NA.wstd(X).mean(); rng=np.random.default_rng(7); fr=[]
    for v in range(FM.MAXV+1):
        pv=M.previsor(S,X,FM.BETA_PREVISOR); prev=FM.FORCA_PREVISOR*pv; local=(S@J)/hesc
        m=np.sign(S)@X.T/N
        fr.append({'S':S[0].copy(),'local':local[0].copy(),'prev':prev[0].copy(),'res':m[0].copy()})
        if v==FM.MAXV: break
        mud=0.
        for i in rng.permutation(N):
            novo=np.tanh(g*(S@J[:,i]/hesc[:,0]+prev[:,i])); mud=max(mud,float(np.abs(novo-S[:,i]).max())); S[:,i]=novo
        mm=NA.wmean(S); S=S-0.5*(mm-mu_t)[:,None]; mm=NA.wmean(S); sd=NA.wstd(S)
        S=np.clip(mm[:,None]+(S-mm[:,None])*(1+0.5*(sd_t/np.maximum(sd,1e-9)-1))[:,None],-1,1)
        if mud<FM.TOL:
            pv=M.previsor(S,X,FM.BETA_PREVISOR); fr.append({'S':S[0].copy(),'local':((S@J)/hesc)[0].copy(),'prev':(FM.FORCA_PREVISOR*pv)[0].copy(),'res':(np.sign(S)@X.T/N)[0].copy()}); break
    return fr
# modo livre: só a vizinhança da Grade v2 (sem memória), o mesmo neurônio e a mesma normalização
def livre(S0,gl=1.6,nv=30):
    S=S0.copy(); mu_t=NA.wmean(X).mean(); sd_t=NA.wstd(X).mean(); fr=[]
    for v in range(nv+1):
        viz=A.filtro(S); fr.append({'S':S[0].copy(),'viz':viz[0].copy()})
        if v==nv: break
        S=np.tanh(gl*viz/ (np.abs(viz).mean(1,keepdims=True)+1e-9))
        mm=NA.wmean(S); S=S-0.5*(mm-mu_t)[:,None]; mm=NA.wmean(S); sd=NA.wstd(S)
        S=np.clip(mm[:,None]+(S-mm[:,None])*(1+0.5*(sd_t/np.maximum(sd,1e-9)-1))[:,None],-1,1)
    return fr
FR=recup(S0); FL=livre(S0)
ref=M.recuperar(J,X,S0,np.random.default_rng(7),g)
print('voltas recuperação:',len(FR)-1,'· voltas livre:',len(FL)-1,'· captura reproduz M.recuperar:',np.allclose(ref[0],FR[-1]['S'],atol=1e-6))
print('ressonância final com cada memória (recuperação):',{nomes[i]:round(float(FR[-1]['res'][i]),3) for i in np.argsort(-FR[-1]['res'])[:3]})
U=NA.P2/np.linalg.norm(NA.P2,axis=1,keepdims=True); L=G.lmax_mensuravel(N); Y,gr=G.harmonicos(U,L)
cascas=[(r,G.pontos_encaixados(n,raio=r)) for r,n in G.CASCAS[1:]]
def pack(fr,keys):
    out=[]
    for f in fr:
        d={}
        for k in keys:
            c,en=G.decompor(f[k],Y,gr); d[k]=np.round(f[k],3).tolist(); d[k+'_en']=np.round(en,5).tolist()
            if k=='S': d['S_int']=[np.round(G.extensao(c,gr,Q),3).tolist() for r,Q in cascas]
        if 'res' in f: d['res']=np.round(f['res'],3).tolist()
        out.append(d)
    return out
dados={'pts':np.round(U,4).tolist(),'faces':ConvexHull(U).simplices.tolist(),'E':Eend.tolist(),'LIN':LIN,'COL':COL,'nomes':nomes,'K':K,
 'alvo':np.round(T[0],3).tolist(),'pista':np.round(S0[0],3).tolist(),'cascas':[{'r':float(r),'pts':np.round(Q,4).tolist()} for r,Q in cascas],
 'rec':pack(FR,['S','local','prev']),'liv':pack(FL,['S','viz']),'edges':[round(float(e)) for e in edges]}
json.dump(dados,open('dados_ecobip_so880.json','w')); print('ok')
