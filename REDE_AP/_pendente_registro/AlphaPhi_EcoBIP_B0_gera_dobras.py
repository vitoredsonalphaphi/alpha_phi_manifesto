# PENDENTE DE REGISTRO — Etapa B, passo B0: roda o EcoBIP 880 ORIGINAL (BEEP880_17S.py) sem alterar nenhuma linha de cálculo,
# até o fim da convergência (campo harmônico no ciclo 10, β máx 4,2098), e guarda as 5 dobras da cascata e o controle
# (a mesma cascata, com o β final, aplicada só à onda quadrada). O IPython é substituído só na importação de áudio.
import sys, types, os, pickle, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); RAIZ = os.path.join(H, '..', '..')
ip = types.ModuleType('IPython'); dp = types.ModuleType('IPython.display'); dp.Audio = lambda *a, **k: None; dp.display = lambda *a, **k: None
ip.display = dp; sys.modules.setdefault('IPython', ip); sys.modules.setdefault('IPython.display', dp)
src = open(os.path.join(RAIZ, 'BEEP880_17S.py')).read().split('\n')
corte = next(i for i, l in enumerate(src) if l.startswith('# ── [7–17s]'))
g = {'__name__': 'ecobip'}; exec('\n'.join(src[:corte]), g)
passos, _ = g['cascata_eq'](g['sinal_bip'], g['beta'], g['BINS'])
n = len(g['sinal_bip']); t = np.linspace(0, g['DUR_BIP'], n, endpoint=False)
quad = g['normalizar'](np.sign(np.sin(2 * np.pi * g['F_BEEP'] * t))); ctrl, _ = g['cascata_eq'](quad, g['beta'], g['BINS'])
np.save(os.path.join(H, 'ecobip_dobras.npy'), np.array(passos)); np.save(os.path.join(H, 'ecobip_entrada.npy'), g['sinal_bip']); np.save(os.path.join(H, 'ctrl_quadrada_cascata.npy'), np.array(ctrl))
pickle.dump({'BANDAS': g['BANDAS']}, open(os.path.join(H, 'ecobip_meta.pkl'), 'wb')); print('dobras e controle guardados')
