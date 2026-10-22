import numpy as np, importlib.util, sys
sys.argv=['x','lib']
s=importlib.util.spec_from_file_location('VL3',__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'AlphaPhi_CampoEsferico_EtapaVL3_ViolaoBaseNova.py')); V=importlib.util.module_from_spec(s); s.loader.exec_module(V)
(_,res,F)=V.rodar((10,'guardados'),guardar=True,so=200)
T=F['T']; P=len(T.esc); cel=V.E.ravel(); X=T.X
m=np.sign(F['Sf'])@X.T/610; idt=np.argmax(m,1); print('identificação pelo previsor no fim (trecho certo):', np.mean(idt==np.arange(P)), idt.tolist())
mp=np.sign(np.array([T.pista(r) for r in F['ruid']]))@X.T/610; print('identificação já na pista:', np.mean(np.argmax(mp,1)==np.arange(P)))
gm=[V.V2.snr(T.decod(X[k],F['ruid'][k]),F['limpos'][k])-V.V2.snr(F['ruid'][k],F['limpos'][k]) for k in range(P)]
gr=[V.V2.snr(T.decod(X[idt[k]],F['ruid'][k]),F['limpos'][k])-V.V2.snr(F['ruid'][k],F['limpos'][k]) for k in range(P)]
print(f'ganho se a leitura devolvesse a memória certa exata: {np.mean(gm):+.2f} dB · a memória que a rede reconheceu, exata: {np.mean(gr):+.2f} dB')
print('ganho por trecho (D-campo):', [round(V.V2.snr(T.decod(F["Sc"][k],F["ruid"][k]),F["limpos"][k])-V.V2.snr(F["ruid"][k],F["limpos"][k]),1) for k in range(P)])
