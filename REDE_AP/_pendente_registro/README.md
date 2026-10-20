# Pendente de registro — Etapa S (memórias com quadros do STFT), 20/10/2026

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

---
*Florianópolis · 20 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
