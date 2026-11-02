# A única fonte: a célula extraída do ECO BEEP 880, α* = 1/3 (idêntica, amostra por amostra, ao áudio do código do Manifesto 01)
import os
_src=open('/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/m01/ECO_BEEP_880_so.py').read()
_src=_src.replace('from IPython.display import Audio, display','display=lambda *a:None\nAudio=lambda *a:None').replace('"eco_beep_880.wav"','"/tmp/claude-0/-home-user-alpha-phi-manifesto/511dd08d-d61c-528e-93c9-dc1bc2243dfb/scratchpad/scannerAP/eco_beep_880_fonte.wav"')
E={'os':os}; exec(_src,E)
FS=E['FS']; ENTRADA=E['x_mix']; CAS=E['cas']
NOMES=['Entrada','1.º ponto de dobra','2.º ponto de dobra','3.º ponto de dobra','4.º ponto de dobra','5.º ponto de dobra']
