# Pendente de registro — Etapas S e K (20 e 21/10/2026)

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

---
*Florianópolis · 21 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
