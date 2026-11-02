# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0
# OBSERVAÇÃO (teste paralelo; sem critérios): o Serial φ (10 cones angulares do ECO-BIP 880, código original AlphaPhi_SerialPhantom_COLAB.py,
# Entrada 282) na esfera de 33 059 pontos. Funções do código original executadas sem alterar nenhuma linha de cálculo.
# Proposta 1 (serial sozinho): cada cone — a entrada (2/3 beep + 1/3 FM-φ + dither), os 5 pontos de dobra e o cone selado — posto na esfera.
# Proposta 2 (serial + ECO na esfera): cada cone selado passa pelo ECO BEEP 880 (α* = 1/3) processado na esfera (Etapa EE1); dobras 0–5, ciclo 20.
import numpy as np, json, base64, hashlib
import exp_etapaEE1 as E
R='/home/user/alpha_phi_manifesto/AlphaPhi_SerialPhantom_COLAB.py'; src=open(R).read(); print('SHA-256 do Serial φ:',hashlib.sha256(src.encode()).hexdigest())
corte=src.index('# ── SERIAL PHANTOM'); G={}
exec(src[:corte].replace('import matplotlib.pyplot as plt','').replace('import matplotlib.gridspec as gridspec',''),G)
assert G['N_SINAL']==E.N
def lg(x): return np.log10(np.abs(E.para_esfera(x)[1])+1e-9)
O,PHI,aneis=E.O,E.PHI,E.aneis
S1=[];S2=[];atr=[];bmx=[]
for i in range(10):
    t=np.linspace(0,G['DURACAO'],G['N_SINAL'],endpoint=False)
    beep=G['normalizar'](np.sign(np.sin(2*np.pi*G['F_BEEP']*t))); fm=G['normalizar'](np.sin(2*np.pi*G['F_ORG']*t+G['BETA_FM']*np.sin(2*np.pi*G['F_M']*t)))
    x=G['normalizar']((1-G['ALPHA_STAR'])*beep+G['ALPHA_STAR']*fm); rng=np.random.default_rng(seed=i); x=G['normalizar'](x+rng.standard_normal(G['N_SINAL'])*G['DITHER_AMP'])
    b,cas=G['agente_eco'](x,G['BINS_PHI'],G['N_CICLOS']); sel=G['selar_hermetico'](cas[-1])
    ref,_=G['gerar_cone_phantom'](i); assert np.allclose(sel*G['PHANTOM_AMP'],ref)
    S1.append([lg(f) for f in list(cas)+[sel]])
    bE,cE=E.agente_esfera(G['normalizar'](sel)); S2.append([lg(f) for f in cE])
    k=int(np.argmax(bE)); atr.append(round((E.BINS[k][2]+E.BINS[k][3])/2)); bmx.append(round(float(bE.max()),3))
    print(f'cone {i+1}: β máx no cone {b.max():.3f} · ECO na esfera sobre o cone selado: atrator {atr[-1]} Hz, β máx {bmx[-1]}',flush=True)
d=json.load(open('dados_EE1b.json')); lo,hi=d['elo'],d['ehi']
def q(v): return np.clip(np.round((v-lo)/(hi-lo)*255),0,255).astype(np.uint8)
out={k:d[k] for k in ('n','pos','faixa','idx','tam','bandas')}
out['S1']=base64.b64encode(b''.join(q(v).tobytes() for c in S1 for v in c)).decode()
out['S2']=base64.b64encode(b''.join(q(v).tobytes() for c in S2 for v in c)).decode()
out['atr']=atr; out['bmx']=bmx
json.dump(out,open('dados_serial.json','w')); print('tamanho MB',round(len(json.dumps(out))/1e6,1))
