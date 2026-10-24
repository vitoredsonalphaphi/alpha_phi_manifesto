import numpy as np, importlib.util, json
SP='/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/'
s=importlib.util.spec_from_file_location('E',SP+'exp_etapaEB.py'); E=importlib.util.module_from_spec(s); s.loader.exec_module(E)
M,NA,LA=E.M,E.NA,E.LA; d=json.load(open(SP+'ecobip/exp_etapaEB.json'))['4.0']
res={}
for b in (12.6,):
    al,idt=[],[]
    for s0 in range(260,270):
        X=NA.memorias(s0); S0=NA.pista(X,4.0,s0); J=LA.ligacao_delta(X); g=NA.ganho_ajustado(J,X)
        Sf=M.recuperar(J,X,S0,np.random.default_rng([7_000_000+s0,40]),g,beta=lambda v,b=b: b,nvoltas=50,parada=False)
        m=NA.medir(Sf,X); al.append(m['alt']); idt.append(m['ident'])
    res[b]=(al,idt); print(f'auditoria: previsor atual com β fixo = {b} (a média final da EB): altura {np.mean(al):.4f} · identificação {np.mean(idt):.4f}')
al,idt=res[12.6]
M.teste('EB − β fixo 12,6 (altura)', np.array(d['EB']['alt'])-np.array(al)); M.teste('EB − β fixo 12,6 (identificação)', np.array(d['EB']['ident'])-np.array(idt))
print('sementes com identificação igual entre EB e SUP:', int(np.sum(np.array(d['EB']['ident'])==np.array(d['SUP']['ident']))),'de 10')
