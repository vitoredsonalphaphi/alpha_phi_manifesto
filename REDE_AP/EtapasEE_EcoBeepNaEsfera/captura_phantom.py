# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0
# OBSERVAÇÃO (sem critérios; nada vai ao repositório): o Phantom (Serial φ sem selagem, código original AlphaPhi_SerialPhantom_SemSelagem_COLAB.py,
# executado sem alterar nenhuma linha de cálculo) na esfera de 33 059 pontos. Trecho central de 1,5 s do serial (o tamanho de um cone).
# Forma A (a tríade): o trecho passa pelo ECO BEEP 880 na esfera (agente de 20 ciclos, 5 pontos de dobra). Forma B (esfera + scanner): o trecho na esfera, sem o ECO.
import numpy as np, json, base64, hashlib
import exp_etapaEE1 as E
R='/home/user/alpha_phi_manifesto/AlphaPhi_SerialPhantom_SemSelagem_COLAB.py'
src=open(R).read(); print('SHA-256 do Phantom:',hashlib.sha256(src.encode()).hexdigest())
src=src.split('# Verificar entropia')[0].replace('import plotly.graph_objects as go','')
G={}; exec(src,G); S=G['SERIAL']; N=E.N; c=len(S)//2; seg=E.O['normalizar'](S[c-N//2:c-N//2+N])
print('serial:',len(S),'amostras · trecho central:',len(seg))
O,PHI,aneis=E.O,E.PHI,E.aneis
def eco_env(x,beta,cm):
    F_fora,campo=E.para_esfera(x); novo=campo.copy(); cohs=[]; envs=np.ones(len(campo)); wm=1/PHI; wn=1-wm
    for i,g in enumerate(aneis):
        z=campo[g]; mag=np.abs(z); ph=np.angle(z); an=np.clip(mag/(mag.sum()+1e-8),1e-10,1.0)
        coh=float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2))); ce=wn*coh+wm*float(cm[i]); cohs.append(coh)
        env=np.clip(1.0+(ce*PHI**float(beta[i]))*np.cos(2.0*np.pi*np.arange(len(g))/PHI),0.05,None); novo[g]=(mag*env)*np.exp(1j*ph); envs[g]=env
    r=E.da_esfera(F_fora,novo,len(x)); return r/(np.max(np.abs(r))+1e-10),np.array(cohs),envs
CICLOS=[1,2,5,10,20]; energia={}; rot={}
beta=np.ones(len(aneis)); bm=beta.copy(); wm=1/PHI; wn=1-wm
for cc in range(1,21):
    s=seg.copy(); cm=np.zeros(len(aneis)); fases=[s]; envs=[]
    for _ in range(5):
        se,cohs,env=eco_env(s,beta,cm); cm=cohs; se=O['normalizar'](se); fases.append(se); envs.append(env); s=se.copy()
    if cc in CICLOS: energia[cc]=[np.log10(np.abs(E.para_esfera(f)[1])+1e-9) for f in fases]; rot[cc]=[np.log(e) for e in envs]
    cr=(cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10); beta=wn*PHI**(3*cr)+wm*bm; bm=beta.copy(); beta=np.clip(beta,0.05,PHI**3)
i=int(np.argmax(beta)); print('Forma A: atrator do ECO sobre o Phantom:',round((E.BINS[i][2]+E.BINS[i][3])/2),'Hz · β máx',round(float(beta.max()),3))
def entr(sig):
    F=np.abs(np.fft.rfft(sig)); F=np.clip(F/(F.sum()+1e-10),1e-10,1.0); return float(-np.sum(F*np.log(F)))
print('Forma A: Δentropia',round(entr(fases[0])-entr(fases[-1]),4))
def q(v,lo,hi): return np.clip(np.round((v-lo)/(hi-lo)*255),0,255).astype(np.uint8)
d=json.load(open('dados_EE1b.json'))
lo,hi=d['elo'],d['ehi']   # a mesma escala de energia do ECO BEEP 880, para comparar
d['E_A']=base64.b64encode(b''.join(q(v,lo,hi).tobytes() for cc in CICLOS for v in energia[cc])).decode()
d['R_A']=base64.b64encode(b''.join(q(v,-3.0,1.7).tobytes() for cc in CICLOS for v in rot[cc])).decode()
d['E_B']=base64.b64encode(q(energia[1][0],lo,hi).tobytes()).decode()
json.dump(d,open('dados_EE_phantom.json','w')); print('ok')
