# Pendente de registro — Etapas S, K, PF e a afinação (20 e 21/10/2026)

**Status:** executada, **não registrada**. Os critérios de sanidade **S00 e S01 falharam**; pela regra aprovada antes de rodar, o teste para aí, sem conclusão. Esta pasta existe para que código e números não se percam.

| Arquivo | Conteúdo |
|---|---|
| `AlphaPhi_CampoEsferico_EtapaS_STFT.py` | o teste (importa a Etapa RP e a Grade Esférica v1) |
| `EtapaS_saida_20out2026.txt` | saída impressa |
| `EtapaS_auditoria_20out2026.txt` | auditoria: pares correlacionados, correlação dentro de cada sinal, para onde vai cada pista |
| `EtapaS_resultados.json` | números completos |

**Sanidade:** S00 não cumprido (média +0,072, mas máxima +0,728; 9 pares acima de 0,5); S01 não cumprido (a convencional identifica 0,217 com pista 25%). No controle, máxima +0,921 e convencional 0,208.

**Auditoria (descritiva, sem valor de conclusão):** os quadros de um mesmo sinal são quase iguais (correlação média de +0,49 a +0,98 dentro de cada sinal); a escolha pelo mais distante pegou 5 quadros da dobra 2 e nenhum da dobra 4. A convencional leva as pistas de 9 das 12 memórias à memória 11 (um atrator que engole as memórias correlacionadas); a AlphaFi 610 mantém a maioria separada (identificação 0,767 com pista 25%). Como a tarefa não passou na sanidade, essa diferença **não** é resultado.

**Correção proposta (a aprovar antes de rodar de novo):** escolher as memórias pela mesma regra, mas parar quando a próxima tiver correlação máxima ≥ 0,5 (o número de memórias sai da regra, não é escolhido); o resto igual.

## Etapa S corrigida — não rodada

A regra corrigida (parar quando a próxima memória teria correlação ≥ 0,5) deu 5 memórias no EcoBIP e 2 no controle, abaixo do mínimo de 6 combinado; o pesquisador foi avisado antes de rodar. Em vez dela, foi aprovada a Etapa K.

## Etapa K — a AlphaFi resiste a memórias parecidas? (21/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaK_Correlacao.py`, `EtapaK_saida_21out2026.txt`, `EtapaK_resultados.json`. 12 memórias por semente em 610 pontos, correlação pedida 0,0 a 0,8, duas famílias (com forma esférica; sem forma). Redes: C (Hebb), C-PI (pseudo-inversa), A (AlphaFi), A-PI (descritiva). Sementes 110–119; medida principal: identificação com pista 25%.

**Sanidade:** K00 cumprido (C = 1,000 no nível 0,0); K01 cumprido (desvio máximo 0,006).

**Identificação, com forma, pista 25% (acaso 0,083):**

| Correlação | C | C-PI | A | A-PI |
|---|---|---|---|---|
| 0,0 | 1,000 | 1,000 | 0,925 | 0,950 |
| 0,2 | 0,083 | 1,000 | 0,817 | 0,892 |
| 0,4 | 0,100 | 1,000 | 0,583 | 0,750 |
| 0,6 | 0,100 | 1,000 | 0,233 | 0,625 |
| 0,8 | 0,092 | 1,000 | 0,083 | 0,492 |

**Critérios (nível 0,6):** K1 A > C **não** (+0,133, 7/10); K2 A > C-PI **não** (−0,767, 0/10); K3 **não** (−0,692); K4 **não** (+0,117, 6/10).

**Leitura:** a hipótese nascida da Etapa S **não se sustenta** contra a convencional melhorada: a pseudo-inversa identifica 1,000 em todos os níveis. Descritivo: a Hebb clássica desaba já em 0,2 (vai à parte comum das memórias), enquanto a AlphaFi decai aos poucos (0,817 em 0,2; 0,583 em 0,4) — a observação da Etapa S era real, mas é vantagem só sobre a versão mais simples. A forma esférica ajuda a AlphaFi (nível 0,0: 0,925 com forma, 0,675 sem forma), e a pseudo-inversa dentro da AlphaFi (A-PI) melhora, sem alcançar a C-PI. A sobreposição final da AlphaFi no nível 0,0 é baixa (0,26): ela acerta qual é a memória, mas para longe dela.

**Auditoria em aberto:** (1) a pseudo-inversa usa a inversa da matriz de correlação das memórias (informação global), o que a AlphaFi local não usa: a comparação é dura por desenho, como combinado; (2) uma só tradução (a entrada do EcoBIP como parte comum); (3) a baixa sobreposição final da AlphaFi, já vista nas Etapas H, RP e S, não foi investigada.

## Etapa PF — o batedor, a pré-função (21/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaPF_Batedor.py`, `EtapaPF_saida_21out2026.txt`, `EtapaPF_resultados.json`, `EtapaPF_auditoria_fundo_21out2026.txt`, `pagina_pre_funcao_na_esfera.html`. A AlphaFi de 610 pontos comparada só consigo mesma, em sete versões (V0 binária a V6 batedor enganado), pela régua própria. Sementes 120–129.

**Conclusão da memória (nível 0,0, pista 40%):** V0 0,122 · V1 0,109 · V2 0,100 · V3 0,120 · V4 0,221 · V5 0,231 · V6 0,218. **Critérios:** PF0 não (área 0,546); PF1 não (−0,013); PF2 cumprido (+0,011, 9/10); PF3 cumprido (+0,111 e +0,010); PF4 não (+0,013, 7/10); PF5 não (0,584). Leitura: o previsor faz quase todo o ganho; o batedor acrescenta pouco, e o batedor enganado quase não atrapalha; o ponto e campo sozinho não ajuda.

**Auditoria — erro de calibração do assistente:** quando a vizinhança da memória foi estreitada pela resolução (regra de 20/10), o peso do fundo do campo (0,02) não foi ajustado. Na esfera de 610 pontos, o campo do fundo ficou 2,12 vezes maior que o da memória (no desenho original, com σ 0,35: 0,20 a 0,26); com esse fundo, só 71% dos neurônios ficam estáveis na própria memória (sem ele, 92%). Isso afeta as Etapas RP, S, K e PF e pode explicar por que a AlphaFi "para longe da memória". As versões contínuas também chegaram ao limite de 50 voltas sem convergir (tolerância 0,001). **Correção proposta:** peso do fundo escalado pela massa da vizinhança, para manter a razão do desenho original; repetir K e PF.

## A afinação — Etapas K e PF repetidas com o fundo escalado (21/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaK2_Afinada.py`, `AlphaPhi_CampoEsferico_EtapaPF2_Afinada.py`, saídas `EtapaK2_saida_21out2026.txt` e `EtapaPF2_saida_21out2026.txt`, os `.json`, e a página `pagina_pre_funcao_na_esfera.html` (agora com "afinada" e "antes"). Correção aprovada pelo pesquisador: peso do fundo = 0,02 × (vizinhança com σ 0,091) / (vizinhança com σ 0,35) = 0,0015. Razão fundo/memória 0,16 (antes 2,12); neurônios estáveis na própria memória 91,5% (antes 71%). Mesmas sementes e critérios.

**Etapa K afinada (identificação, com forma, pista 25%):** a AlphaFi passa a identificar 1,000 com memórias distintas (antes 0,925; sobreposição 0,47, antes 0,26), mas cai mais depressa com memórias parecidas (0,4: 0,417, antes 0,583) — o fundo forte protegia contra a semelhança. K1–K4 **não cumpridos**. Descritivo, não previsto: a AlphaFi com pseudo-inversa (A-PI) chega a 1,000 / 1,000 / 0,992 / 0,875 / 0,692 nos cinco níveis.

**Etapa PF afinada (conclusão, nível 0,0, pista 40%):** V0 0,221 · V1 0,217 · V2 0,203 · V3 0,261 · V4 0,420 · V5 0,382 · V6 0,314 (todas acima do antes). **PF2 cumprido** (+0,044, 10/10), **PF4 cumprido** (+0,068, 9/10: o batedor certo importa); PF0, PF1, PF3 (V5 < V4: −0,038), PF5 (0,667) **não**. Com pista de 25%, o previsor leva a conclusão a 0,794 e a firmeza × acerto a 0,812.

**Leitura:** a afinação firma a memória da AlphaFi e torna o batedor útil; o previsor sozinho continua sendo a via mais forte, e somar o batedor a ele não acrescenta. O ponto e campo sozinho segue sem efeito. A conclusão ainda fica longe de 1,0 com pista de 40% (0,42 no melhor caso). As versões contínuas continuam sem estabilizar em 50 voltas.

---
*Florianópolis · 21 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
