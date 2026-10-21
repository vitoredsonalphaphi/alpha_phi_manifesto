# Conferência: o módulo canônico reproduz exatamente o código que gerou a Etapa PF afinada (mesmas entradas, mesma semente)
import numpy as np, importlib.util
def load(n,p): s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
PF=load('PF','AlphaPhi_CampoEsferico_EtapaPF2_Afinada.py'); FM=load('FM','../AlphaPhi_FerramentasMemoria_v1.py'); K=PF.K
s=120; X=K.com_forma(K.calibrar(0.0),np.random.default_rng([1_900_000+s,0])); J1=K.redes(X)['A']; J2=FM.ligacoes(FM.G.pontos_encaixados(610),X)
print('ligações idênticas:',np.allclose(J1,J2))
r2=np.random.default_rng([1_950_000+s,0,40]); S0=X.copy()
for p in range(12): S0[p,r2.choice(610,244,replace=False)]*=-1
res,_,_,Bm=PF.rodar(J1,X,S0,s,0,40)
rng=lambda: np.random.default_rng([2_000_000+s,0,40])
hesc=np.abs(S0@J1).mean(1,keepdims=True)
print('batedor idêntico:',np.allclose(FM.batedor(J2,S0,hesc),Bm))
for v,kw in (('V3',dict(usar_previsor=False,usar_batedor=True)),('V4',dict(usar_previsor=True)),('V5',dict(usar_previsor=True,usar_batedor=True))):
    S=FM.recuperar(J2,X,S0,rng(),**kw); print(v,'idêntico:',np.allclose(S,res[v]),'conclusão',round(FM.conclusao(S,X),3))
print('V0 idêntico:',np.array_equal(FM.recuperar_binaria(J2,S0,rng()),res['V0']))
