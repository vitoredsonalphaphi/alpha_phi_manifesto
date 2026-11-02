# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0
# Scanner Topográfico AP — processamento real do ECO BEEP 880 na esfera (canônico em 02/11/2026) · Vitor Edson Delavi · Claude

# TESTE PARALELO — ETAPA EE1: o ECO BEEP 880 (α* = 1/3) processado DENTRO da esfera (critérios aprovados pelo pesquisador em 02/11/2026)
# Os três códigos salvaguardados não são alterados: o ECO BEEP 880 é lido de CANONICO_SALVAGUARDA (só leitura) e a esfera usa a Grade v2 canônica.
# Montagem: um ponto da esfera para cada frequência de cada banda áurea; as bandas viram faixas de latitude (graves ao norte); dentro de cada
# banda, as frequências seguem em ordem ao redor do anel (longitude), e a rotação cos(2πn/φ) percorre o anel. O campo da esfera guarda amplitude
# e fase. As frequências fora das bandas (abaixo de 20 Hz) ficam fora da esfera, intactas, como no original.
# EE1a: áudio da esfera idêntico ao original (diferença máxima < 1e-6). EE1b: mesma queda de entropia (+5,4258) e mesmo atrator (42 Hz).
import numpy as np, importlib.util, os
R='/home/user/alpha_phi_manifesto/'
src=open(R+'CANONICO_SALVAGUARDA/ECO_BEEP_880/ECO_BEEP_880_alpha_1_3.py').read()
O={}; exec(src.split('# ── ECO BEEP 880, α* = 1/3')[0].replace('from IPython.display import Audio, display',''),O)   # só as funções, sem executar
s=importlib.util.spec_from_file_location('G',R+'REDE_AP/AlphaPhi_GradeEsferica_v2.py'); G=importlib.util.module_from_spec(s); s.loader.exec_module(G)
PHI,FS,DUR=O['PHI'],O['FS'],O['DURACAO']; N=int(FS*DUR)
BINS=O['bandas_para_bins'](O['gerar_bandas_phi'](),N)
x_mix=O['normalizar']((1.0-O['ALPHA'])*O['gerar_beep'](O['F_BEEP'])+O['ALPHA']*O['gerar_fm_phi']())
# ── a esfera: endereços (um ponto por frequência de cada banda) ───────────────
tam=[b-a for a,b,_,_ in BINS]; NP=sum(tam); P=G.pontos_encaixados(NP)
ordem_lat=np.argsort(-P[:,2],kind='stable'); aneis=[]; k=0
for t in tam:
    grupo=ordem_lat[k:k+t]; lon=np.arctan2(P[grupo,1],P[grupo,0]); aneis.append(grupo[np.argsort(lon,kind='stable')]); k+=t
bin_do_ponto=np.zeros(NP,int)
for (a,b,_,_),g in zip(BINS,aneis): bin_do_ponto[g]=np.arange(a,b)
# ── o processamento, no campo da esfera ──────────────────────────────────────
def para_esfera(x):
    F=np.fft.rfft(x); return F, F[bin_do_ponto]                      # (o que fica fora da esfera, o campo da esfera)
def da_esfera(F_fora,campo,N):
    F_out=F_fora.copy()
    for g in aneis: F_out[bin_do_ponto[g]]=campo[g]                  # mesma ordem de escrita das bandas do original
    return np.fft.irfft(F_out,n=N)
def eco_esfera(x,beta_bands,coh_mem=None):
    F_fora,campo=para_esfera(x); novo=campo.copy(); cohs=[]; wm=1.0/PHI; wn=1.0-wm
    for i,g in enumerate(aneis):
        bi=float(beta_bands[i]) if i<len(beta_bands) else 1.0
        z=campo[g]; mag=np.abs(z); ph=np.angle(z)
        an=np.clip(mag/(mag.sum()+1e-8),1e-10,1.0); coh=float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        ce=wn*coh+wm*float(coh_mem[i]) if (coh_mem is not None and i<len(coh_mem)) else coh; cohs.append(coh)
        n_idx=np.arange(len(g))                                       # a posição ao redor do anel
        env=np.clip(1.0+(ce*PHI**bi)*np.cos(2.0*np.pi*n_idx/PHI),0.05,None); novo[g]=(mag*env)*np.exp(1j*ph)
    r=da_esfera(F_fora,novo,len(x)); return r/(np.max(np.abs(r))+1e-10),np.array(cohs)
def cascata_esfera(s0,beta):
    cas=[s0]; s=s0.copy(); cm=np.zeros(len(aneis))
    for _ in range(O['N_STEPS']):
        se,cohs=eco_esfera(s,beta,cm); cm=cohs; se=O['normalizar'](se); cas.append(se); s=se.copy()
    return cas,cohs
def agente_esfera(s0,n_ciclos=20):
    beta=np.ones(len(aneis)); bm=beta.copy(); wm=1/PHI; wn=1-wm
    for _ in range(n_ciclos):
        cas,cohs=cascata_esfera(s0,beta); cr=(cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        beta=wn*PHI**(3*cr)+wm*bm; bm=beta.copy(); beta=np.clip(beta,0.05,PHI**3)
    return beta,cas
if __name__=='__main__':
    print(f'pontos na esfera: {NP} (um por frequência das {len(BINS)} bandas áureas) · frequências fora da esfera: {N//2+1-len(set(bin_do_ponto))}')
    b0,c0=O['agente_eco'](x_mix,BINS,n_ciclos=20); b1,c1=agente_esfera(x_mix)
    d=max(float(np.max(np.abs(a-b))) for a,b in zip(c0,c1)); print(f'EE1a diferença máxima entre o áudio original e o da esfera (6 fases): {d:.2e} -> {"cumprido" if d<1e-6 else "NÃO cumprido"}')
    ent=O.get('entropia_espectral')
    def entr(sig):
        F=np.abs(np.fft.rfft(sig)); F=np.clip(F/(F.sum()+1e-10),1e-10,1.0); return float(-np.sum(F*np.log(F)))
    de0=entr(c0[0])-entr(c0[-1]); de1=entr(c1[0])-entr(c1[-1])
    i0,i1=int(np.argmax(b0)),int(np.argmax(b1)); at0=(BINS[i0][2]+BINS[i0][3])/2; at1=(BINS[i1][2]+BINS[i1][3])/2
    print(f'EE1b queda de entropia: original {de0:+.4f} · esfera {de1:+.4f} · atrator: original {at0:.0f} Hz · esfera {at1:.0f} Hz -> {"cumprido" if abs(de0-de1)<1e-6 and i0==i1 else "NÃO cumprido"}')
