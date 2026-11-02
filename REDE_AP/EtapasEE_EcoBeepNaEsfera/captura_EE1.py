# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0
# Captura para o scanner do processamento real do ECO BEEP 880 dentro da esfera (Etapa EE1, teste paralelo)
import numpy as np, base64, json
import exp_etapaEE1 as E
O,PHI,aneis,P=E.O,E.PHI,E.aneis,E.P
CICLOS=[1,2,5,10,20]
def eco_env(x,beta,cm):
    F_fora,campo=E.para_esfera(x); novo=campo.copy(); cohs=[]; envs=np.ones(len(campo)); wm=1/PHI; wn=1-wm
    for i,g in enumerate(aneis):
        z=campo[g]; mag=np.abs(z); ph=np.angle(z); an=np.clip(mag/(mag.sum()+1e-8),1e-10,1.0)
        coh=float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2))); ce=wn*coh+wm*float(cm[i]); cohs.append(coh)
        env=np.clip(1.0+(ce*PHI**float(beta[i]))*np.cos(2.0*np.pi*np.arange(len(g))/PHI),0.05,None); novo[g]=(mag*env)*np.exp(1j*ph); envs[g]=env
    r=E.da_esfera(F_fora,novo,len(x)); return r/(np.max(np.abs(r))+1e-10),np.array(cohs),envs
energia={};rot={};betas={}
beta=np.ones(len(aneis)); bm=beta.copy(); wm=1/PHI; wn=1-wm
for c in range(1,21):
    s=E.x_mix.copy(); cm=np.zeros(len(aneis)); fases=[s]; envs=[]
    for _ in range(5):
        se,cohs,env=eco_env(s,beta,cm); cm=cohs; se=O['normalizar'](se); fases.append(se); envs.append(env); s=se.copy()
    if c in CICLOS:
        energia[c]=[np.log10(np.abs(E.para_esfera(f)[1])+1e-9) for f in fases]; rot[c]=[np.log(e) for e in envs]; betas[c]=beta.copy()
    cr=(cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10); beta=wn*PHI**(3*cr)+wm*bm; bm=beta.copy(); beta=np.clip(beta,0.05,PHI**3)
b_ref,c_ref=O['agente_eco'](E.x_mix,E.BINS,n_ciclos=20)
print('o 5.º ponto do ciclo 20 confere com o original:',np.allclose(fases[-1],c_ref[-1]))
def q(v,lo,hi): return np.clip(np.round((v-lo)/(hi-lo)*255),0,255).astype(np.uint8)
allE=np.concatenate([np.concatenate(energia[c]) for c in CICLOS]); lo,hi=np.percentile(allE,[1,99.5])
eb=b''.join(q(v,lo,hi).tobytes() for c in CICLOS for v in energia[c])
rb=b''.join(q(v,-3.0,1.7).tobytes() for c in CICLOS for v in rot[c])
faixa=np.zeros(len(P),np.uint8)
for i,g in enumerate(aneis): faixa[g]=i
d={'n':len(P),'ciclos':CICLOS,'pos':base64.b64encode(P.astype(np.float32).tobytes()).decode(),'faixa':base64.b64encode(faixa.tobytes()).decode(),
 'E':base64.b64encode(eb).decode(),'R':base64.b64encode(rb).decode(),'elo':float(lo),'ehi':float(hi),
 'bandas':[[round(b[2]),round(b[3])] for b in E.BINS],'beta':{c:[round(float(x),3) for x in betas[c]] for c in CICLOS}}
json.dump(d,open('dados_EE1.json','w')); print('tamanho (MB):',round(len(json.dumps(d))/1e6,2))
