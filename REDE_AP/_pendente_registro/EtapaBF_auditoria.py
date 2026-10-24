import numpy as np, importlib.util, json
SP='/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
s=importlib.util.spec_from_file_location('B',SP+'exp_etapaBF.py'); B=importlib.util.module_from_spec(s); s.loader.exec_module(B)
NA,P=B.NA,B.P; d=json.load(open(SP+'ecobip/exp_etapaBF.json'))
print('Contribuição do campo = rede inteira − consulta direta (a memória mais parecida com a pista, sem dinâmica nenhuma)')
for r in ('2.0','3.0','4.0'):
    al,idt=[],[]
    for s0 in range(270,280):
        X=NA.memorias(s0); S0=NA.pista(X,float(r),s0); k=(np.sign(S0)@X.T/610).argmax(1); m=NA.medir(X[k],X); al.append(m['alt']); idt.append(m['ident'])
    print(f'ruído {r}: consulta direta altura {np.mean(al):.4f}, identificação {np.mean(idt):.4f} · rede com β=8: {np.mean(d[r]["8"]["alt"]):.4f}/{np.mean(d[r]["8"]["ident"]):.4f} · β=8φ: {np.mean(d[r]["8φ"]["alt"]):.4f}/{np.mean(d[r]["8φ"]["ident"]):.4f} · β=∞: {np.mean(d[r]["∞"]["alt"]):.4f}/{np.mean(d[r]["∞"]["ident"]):.4f}')
    print(f'   contribuição do campo na altura: β=8 {np.mean(d[r]["8"]["alt"])-np.mean(al):+.4f} · β=8φ {np.mean(d[r]["8φ"]["alt"])-np.mean(al):+.4f} · β=∞ {np.mean(d[r]["∞"]["alt"])-np.mean(al):+.4f}')
