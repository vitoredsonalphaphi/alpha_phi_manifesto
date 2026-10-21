# Campo Esférico — Etapas S, K e PF, e a afinação do fundo · Resultados (20 e 21/10/2026)

**Status:** registrado a pedido do pesquisador em 21/10/2026 · os pontos de auditoria em aberto estão listados em cada seção
**Antecedente:** `AlphaPhi_CampoEsferico_Etapas_P_A_B_H_RP_RESULTADOS_20out2026.md`
**Código e saídas:** `REDE_AP/CampoEsferico_Etapas_S_K_PF/` (antes `_pendente_registro/`); a conferência do canônico provisório foi refeita a partir dessa pasta em 21/10 e deu saída idêntica
**Canônico:** `REDE_AP/AlphaPhi_FerramentasMemoria_v1.py` (SHA-256 `d8dc24674b55d127b4e9ad1b989194c4cbfafefc3e6d5c00f3fbbf4d7bbba295`), acréscimos de 21/10 em `CANONICO_CampoEsferico_v1.md` (regras 4–6)
**Entradas do Manifesto 03:** 341 (a régua), 342 (a régua própria, sem competição), 343 (o neurônio ponto e campo), 344 (o batedor)
**Regra comum:** critérios escritos na conversa antes de cada execução; comparação pareada = diferença média > 2 erros-padrão e vantagem em ≥ 8 de 10 sementes; todos os números abaixo foram impressos por código executado nas sessões de 20 e 21/10/2026.

## Síntese

| Teste | Pergunta | Resultado |
|---|---|---|
| Etapa S — memórias com quadros do STFT | memórias mais ricas tiradas do EcoBIP ao longo do tempo | **sanidade falhou, sem conclusão:** os quadros de um mesmo sinal são quase iguais |
| Etapa S corrigida | parar a escolha quando a próxima memória teria correlação ≥ 0,5 | **não rodada:** só 5 memórias (mínimo combinado: 6) |
| Etapa K — memórias com semelhança controlada | a AlphaFi resiste melhor a memórias parecidas? | **não** contra a pseudo-inversa; só contra a Hebb clássica |
| Etapa PF — o batedor | a pré-função e o previsor ajudam a AlphaFi a completar a memória? (sem competição) | o previsor ajuda; o batedor quase nada; **auditoria: fundo do campo desregulado** |
| Afinação (K e PF repetidas) | com o fundo na proporção original | a memória se firma (91,5% estáveis, antes 71%); a conclusão quase dobra; o batedor passa a acrescentar |

*Em palavras:* a principal descoberta destas etapas foi um **erro de calibração do assistente**: desde 20/10, o fundo do campo falava duas vezes mais alto que a memória. Afinado, a rede estabiliza a memória e as ferramentas respondem. Com memórias parecidas entre si, a rede afinada fica pior que a desregulada.

## 1. Etapa S — memórias com quadros do STFT (20/10)

`AlphaPhi_CampoEsferico_EtapaS_STFT.py` · 150 quadros candidatos (EcoBIP: entrada + 5 dobras; janela de 1 s, passo de 0,25 s), 225 faixas de 40 Hz a 20 kHz → 225 harmônicos; 12 memórias escolhidas "o mais distante dos já escolhidos"; 610 pontos; sementes 100–109.

- **S00 não cumprido:** correlação média +0,072, máxima +0,728 (9 pares acima de 0,5). Controle: máxima +0,921.
- **S01 não cumprido:** a convencional identifica 0,217 com pista de 25% (o mínimo era 0,8).
- Pela regra aprovada antes de rodar, o teste para aí. **Auditoria (descritiva):** os quadros de um mesmo sinal têm correlação média de +0,49 a +0,98; a escolha pegou 5 quadros da 2ª dobra e nenhum da 4ª; a convencional leva as pistas de 9 das 12 memórias a uma única memória.
- **Correção (S corrigida):** parar quando a próxima memória teria correlação ≥ 0,5. Deu 5 memórias (EcoBIP) e 2 (controle), abaixo do mínimo de 6; o pesquisador foi avisado antes de rodar, e a Etapa K tomou o lugar.

## 2. Etapa K — memórias com semelhança controlada (21/10)

`AlphaPhi_CampoEsferico_EtapaK_Correlacao.py` · 12 memórias por semente, correlação pedida 0,0 a 0,8; família com forma esférica (principal) e sem forma; redes C (Hebb), C-PI (pseudo-inversa), A (AlphaFi), A-PI (descritiva); sementes 110–119; medida principal: identificação com pista de 25%.

**Sanidade:** K00 cumprido (C = 1,000 no nível 0,0); K01 cumprido (desvio máximo 0,006).

| Correlação (com forma) | C | C-PI | A | A-PI |
|---|---|---|---|---|
| 0,0 | 1,000 | 1,000 | 0,925 | 0,950 |
| 0,2 | 0,083 | 1,000 | 0,817 | 0,892 |
| 0,4 | 0,100 | 1,000 | 0,583 | 0,750 |
| 0,6 | 0,100 | 1,000 | 0,233 | 0,625 |
| 0,8 | 0,092 | 1,000 | 0,083 | 0,492 |

**Critérios (nível 0,6):** K1 A > C não (+0,133, 7/10); K2 A > C-PI não (−0,767, 0/10); K3 não (−0,692); K4 não (+0,117, 6/10).

**Leitura:** a hipótese nascida da Etapa S não se sustenta contra a pseudo-inversa. A Hebb clássica desaba já em 0,2; a AlphaFi decai aos poucos. **Revisão do enquadramento (Entrada 342):** a partir desta etapa, o pesquisador estabeleceu que a AlphaFi não deve ser medida em competição com a convencional; os números acima ficam como registro, e a convencional, como referência.

## 3. Etapa PF — o batedor, a pré-função (21/10)

`AlphaPhi_CampoEsferico_EtapaPF_Batedor.py` · a AlphaFi de 610 pontos comparada só consigo mesma, em sete versões (V0 binária; V1 ponto e campo; V2 com resfriamento; V3 com o batedor; V4 com o previsor; V5 as duas vias; V6 batedor enganado); 12 memórias com forma; sementes 120–129. Medida principal: a **conclusão** (sobreposição final do sinal com a memória certa; 0 = acaso, 1 = completa), nível 0,0, pista de 40%.

| Versão | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|---|
| Conclusão | 0,122 | 0,109 | 0,100 | 0,120 | 0,221 | 0,231 | 0,218 |

PF0 não (área 0,546); PF1 não (−0,013); PF2 cumprido (+0,011, 9/10); PF3 cumprido (+0,111 e +0,010); PF4 não (+0,013, 7/10); PF5 não (0,584).

**Auditoria — erro de calibração do assistente:** quando a vizinhança da memória foi estreitada pela resolução (20/10: σ 0,35 → 0,091), o peso do fundo (0,02) não foi ajustado. Na esfera de 610 pontos, o campo do fundo ficou **2,12 vezes** maior que o da memória (no desenho original: 0,20 a 0,26); com esse fundo, só 71% dos neurônios ficam estáveis na própria memória (sem ele, 92%). Afeta as Etapas RP, S, K e PF (`EtapaPF_auditoria_fundo_21out2026.txt`).

## 4. A afinação — Etapas K e PF repetidas (21/10)

`AlphaPhi_CampoEsferico_EtapaK2_Afinada.py`, `AlphaPhi_CampoEsferico_EtapaPF2_Afinada.py` · correção aprovada pelo pesquisador: peso do fundo = 0,02 × (massa da vizinhança com σ 0,091) / (massa com σ 0,35) = 0,0015. Razão fundo/memória 0,16; neurônios estáveis na própria memória 91,5%. Mesmas sementes e critérios.

**PF afinada (conclusão, nível 0,0, pista 40%):**

| Versão | V0 | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|---|---|
| Antes | 0,122 | 0,109 | 0,100 | 0,120 | 0,221 | 0,231 | 0,218 |
| Afinada | **0,221** | 0,217 | 0,203 | 0,261 | **0,420** | 0,382 | 0,314 |

PF2 **cumprido** (+0,044, 10/10); PF4 **cumprido** (+0,068, 9/10: o batedor certo importa); PF0 (0,585), PF1 (−0,004), PF3 (V5 − V4 = −0,038) e PF5 (0,667) **não**. Com pista de 25%: o previsor leva a conclusão a 0,794 (90% dos neurônios certos), com firmeza × acerto 0,812; identificação 1,000 em V0, V3, V4 e V5.

**K afinada (identificação, com forma, pista 25%):** A = 1,000 / 0,933 / 0,417 / 0,108 / 0,083; A-PI = 1,000 / 1,000 / 0,992 / 0,875 / 0,692 (níveis 0,0 a 0,8). Com memórias distintas, a AlphaFi passa a identificar todas (antes 0,925; sobreposição 0,47, antes 0,26); com memórias parecidas, cai mais depressa que antes (0,4: 0,417, antes 0,583). K1–K4 não cumpridos.

**Leitura:** a afinação firma a memória e torna o batedor útil; o previsor é a via mais forte, e somar o batedor a ele não acrescenta; o neurônio contínuo sozinho segue sem efeito. O fundo forte, desregulado, protegia contra a semelhança entre memórias.

**Na página** (`pagina_pre_funcao_na_esfera.html`, botão "afinada / antes"): a topografia da forma por grau passa de um relevo baixo e recortado a um planalto alto e largo, a memória firme em todas as escalas.

## Auditoria em aberto

1. Com pista de 40%, a melhor conclusão afinada é 0,42 (71% dos neurônios certos): a memória ainda fica incompleta.
2. As versões contínuas chegam ao limite de 50 voltas sem estabilizar (tolerância 0,001).
3. A proporção fundo/memória que serve às memórias distintas não serve às parecidas; falta um fundo que se ajuste sozinho.
4. Todas as memórias derivam do EcoBIP 880 (agenda V: outros substratos).
5. A pseudo-inversa usa informação global (a inversa da correlação das memórias), que a AlphaFi local não usa.

## Próximo passo

Memórias parecidas com a rede afinada (um fundo que se ajuste à semelhança); a estabilização do neurônio contínuo; outros substratos; a régua refinada da Entrada 341 (memória por regiões, por forma, no campo e no caminho).

---
*Florianópolis · 21 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
