# Etapas EE — o ECO BEEP 880 na esfera (testes paralelos, 02/11/2026)

**Critérios aprovados pelo pesquisador antes de cada execução.** Os três códigos salvaguardados (`CANONICO_SALVAGUARDA/`) não foram alterados: são lidos, nunca editados.

| Etapa | O que | Resultado |
|---|---|---|
| **EE1** | o ECO BEEP 880 (α\* = 1/3) processado dentro da esfera: 33 059 pontos, um por frequência das 15 bandas áureas | áudio **idêntico** ao original (diferença 0, nas 6 fases); Δentropia +5,4258; atrator 42 Hz: **cumprido** |
| **EE2a** | a coerência de cada banda pesada pela área das células (Voronoi esférico) | Δentropia +5,3376 (98,4%); atrator 42 Hz; correlação do 5.º ponto 0,9995: EE2-1, 2, 3 **cumpridos** |
| **EE2b** | a memória da coerência trocada com as bandas vizinhas | Δentropia +5,3399 (98,4%); atrator 42 Hz; correlação 1,0000: EE2-1, 2, 3 **cumpridos** |
| **EE2c** | as dobras como cascas | não implementada (idêntica por construção com cascas iguais; falharia como a Etapa CI com cascas menores) |
| **EE2-4** | escuta às cegas do pesquisador (som_A, som_B, som_C) | **pendente** |

**Leitura:** a esfera pode participar do ECO BEEP 880 sem estragá-lo; nestas formas, não acrescenta.

## Observação: o Phantom na esfera (sem critérios)

Código original `AlphaPhi_SerialPhantom_SemSelagem_COLAB.py` (SHA-256 41d43999…cd9, conferido), trecho central de 1,5 s do Serial φ.
- **Forma B (Phantom sozinho):** a esfera fica quase uniformemente iluminada: o *dither* (amplitude 1/φ⁵) dá energia a todas as frequências.
- **Forma A (Phantom + ECO):** o ECO BEEP 880 leva o Phantom ao mesmo atrator (42 Hz, β = φ³); Δentropia +3,82 (contra +5,43 do ECO BEEP 880). Hipótese a testar: o atrator é assinatura do instrumento, e não só do sinal.

## Arquivos

`exp_etapaEE1.py`, `exp_etapaEE2.py`, `captura_EE1.py`, `captura_phantom.py` (os dois de captura gravam os dados do scanner). O scanner com estas vistas: `CANONICO_SALVAGUARDA/SCANNER_AP/esfera_ecobeep880_processamento_real_v2.html` (a v1 permanece intacta).

---
*Florianópolis · 2 de novembro de 2026 · Vitor Edson Delavi · Claude*
