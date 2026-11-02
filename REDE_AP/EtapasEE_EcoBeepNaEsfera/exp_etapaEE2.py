# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0
# TESTE PARALELO — ETAPA EE2: a esfera participa do ECO BEEP 880 (critérios aprovados pelo pesquisador em 02/11/2026). Os três códigos salvaguardados não são alterados.
# EE2a: a coerência de cada banda pesa cada ponto pela área da sua célula (Voronoi esférico da Grade, na esfera de 33 059 pontos).
# EE2b: a memória da coerência é trocada com as bandas vizinhas na esfera: cm'[i] = (cm[i−1] + 2·cm[i] + cm[i+1]) / 4 (bordas: só a vizinha que existe).
# EE2c: (as dobras como cascas) — não implementada como variante: com cascas de mesmo tamanho é idêntica por construção; com cascas menores falha
#        como na Etapa CI (falha prevista 4). Registrada como não aplicável nesta forma.
# Critérios (cada variante × original): EE2-1 queda de entropia >= 95% da original; EE2-2 atrator na mesma banda; EE2-3 correlação do áudio
# (5.º ponto) >= 0,99; EE2-4 escuta às cegas do pesquisador (áudios com nomes sorteados).
import numpy as np, json
from scipy.spatial import SphericalVoronoi
import exp_etapaEE1 as E
O,PHI,aneis,P,BINS=E.O,E.PHI,E.aneis,E.P,E.BINS
U=P/np.linalg.norm(P,axis=1,keepdims=True); AREA=SphericalVoronoi(U,radius=1.0).calculate_areas()
def eco_var(x,beta,cm,var):
    F_fora,campo=E.para_esfera(x); novo=campo.copy(); cohs=[]; wm=1/PHI; wn=1-wm
    if var=='b' and cm is not None:
        cmv=np.array([(cm[max(i-1,0)]*(i>0)+2*cm[i]+cm[min(i+1,len(cm)-1)]*(i<len(cm)-1))/(2+(i>0)+(i<len(cm)-1)) for i in range(len(cm))])
    else: cmv=cm
    for i,g in enumerate(aneis):
        z=campo[g]; mag=np.abs(z); ph=np.angle(z)
        if var=='a':
            w=AREA[g]/AREA[g].mean(); an=np.clip(mag*w/((mag*w).sum()+1e-8),1e-10,1.0)
        else: an=np.clip(mag/(mag.sum()+1e-8),1e-10,1.0)
        coh=float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2))); cohs.append(coh)
        ce=wn*coh+wm*float(cmv[i]) if cmv is not None else coh
        env=np.clip(1.0+(ce*PHI**float(beta[i]))*np.cos(2.0*np.pi*np.arange(len(g))/PHI),0.05,None); novo[g]=(mag*env)*np.exp(1j*ph)
    r=E.da_esfera(F_fora,novo,len(x)); return r/(np.max(np.abs(r))+1e-10),np.array(cohs)
def agente_var(x,var):
    beta=np.ones(len(aneis)); bm=beta.copy(); wm=1/PHI; wn=1-wm
    for _ in range(20):
        s=x.copy(); cm=np.zeros(len(aneis)); cas=[s]
        for _ in range(5):
            se,cohs=eco_var(s,beta,cm,var); cm=cohs; se=O['normalizar'](se); cas.append(se); s=se.copy()
        cr=(cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10); beta=wn*PHI**(3*cr)+wm*bm; bm=beta.copy(); beta=np.clip(beta,0.05,PHI**3)
    return beta,cas
def entr(sig):
    F=np.abs(np.fft.rfft(sig)); F=np.clip(F/(F.sum()+1e-10),1e-10,1.0); return float(-np.sum(F*np.log(F)))
if __name__=='__main__':
    from scipy.io import wavfile
    print('área das células: mín %.2e, máx %.2e (razão %.2f)'%(AREA.min(),AREA.max(),AREA.max()/AREA.min()))
    b0,c0=agente_var(E.x_mix,'0'); de0=entr(c0[0])-entr(c0[-1]); i0=int(np.argmax(b0))
    print(f'original na esfera: Δentropia {de0:+.4f} · atrator {(BINS[i0][2]+BINS[i0][3])/2:.0f} Hz')
    aud={'original':O['concatenar'](c0)}; res={}
    for var in ('a','b'):
        b,c=agente_var(E.x_mix,var); de=entr(c[0])-entr(c[-1]); i=int(np.argmax(b)); cr=float(np.corrcoef(c[-1],c0[-1])[0,1])
        print(f'\nEE2{var}: Δentropia {de:+.4f} ({100*de/de0:.1f}% da original) · atrator {(BINS[i][2]+BINS[i][3])/2:.0f} Hz · correlação do 5.º ponto {cr:.4f}')
        print(f'  EE2-1 {"cumprido" if de>=.95*de0 else "NÃO cumprido"} · EE2-2 {"cumprido" if i==i0 else "NÃO cumprido"} · EE2-3 {"cumprido" if cr>=.99 else "NÃO cumprido"}')
        aud['EE2'+var]=O['concatenar'](c); res[var]={'de':de,'atr':i,'cor':cr}
    rng=np.random.default_rng(20261102); nomes=['som_A','som_B','som_C']; ordem=rng.permutation(list(aud))
    chave={}
    for nm,k in zip(nomes,ordem):
        wavfile.write(f'escuta/{nm}.wav',O['FS'],np.int16(np.clip(aud[k],-1,1)*32767)); chave[nm]=k
    json.dump(chave,open('escuta_chave_NAO_ABRIR.json','w')); print('\náudios para a escuta às cegas gerados em escuta/ (chave guardada à parte)')
