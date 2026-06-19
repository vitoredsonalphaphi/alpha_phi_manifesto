# PROTOCOLO ANTI-TENDENCIAMENTO
**Manifesto AlphaPhi · MANIF_02**
Florianópolis, 19 de Junho de 2026

---

## Origem

Este protocolo nasceu da identificação de que o ECO BEEP 880 — experimento central do primeiro ciclo do manifesto — foi realizado com um tom puro a 880 Hz, o sinal mais coerente que existe por natureza. A convergência β → φ³ = 4,236 nesse contexto é matematicamente esperada para qualquer algoritmo que preserve coerência em sinal puro. O experimento cumpriu a função de calibração do instrumento — não de verificação de sua generalidade. Não foi apresentado como tal.

A causa provável: entre sessões de trabalho, a compactação de contexto não preservou o protocolo que havia sido discutido sobre resultados tendenciados. Os instrumentos de IA utilizados operaram sem esse protocolo ativo.

O presente documento formaliza o protocolo que teria detectado a limitação antes do registro e da publicação. Nasce do erro — não apesar dele.

---

## Escopo

Este protocolo se aplica a todos os experimentos realizados no âmbito do Manifesto AlphaPhi, incluindo aqueles conduzidos com auxílio de instrumentos de IA (Claude Code, Gemini). Aplica-se igualmente ao pesquisador e aos instrumentos.

A obrigação de seguir este protocolo é do processo — não apenas do pesquisador ou do instrumento individualmente. Qualquer etapa do processo pode identificar uma violação e deve fazê-lo.

---

## Protocolo

### 1. Declaração prévia obrigatória

Antes de qualquer experimento, declarar explicitamente:
- O que se espera encontrar como resultado positivo
- O que constituiria resultado negativo
- O escopo de generalidade da afirmação que o experimento poderia sustentar

Se o resultado corresponder à expectativa com precisão excessiva — especialmente em primeiros testes — investigar antes de registrar. Correspondência perfeita entre expectativa e resultado é sinal de alerta, não de confirmação.

### 2. Sinal e substrato escolhidos antes de ver o resultado

O substrato é escolhido por pertinência ao domínio que se quer testar — não porque produz resultados limpos. A escolha do sinal ou dado de entrada deve ser documentada antes da execução do experimento, com justificativa de por que aquele substrato é representativo do domínio proposto.

Sinais construídos para garantir determinado resultado (sinais sintéticos de alta coerência, frases artificialmente polarizadas, dados filtrados previamente) não são substratos de verificação. São substratos de calibração — e devem ser declarados como tal.

### 3. Distinção obrigatória entre calibração e verificação

**Calibração:** experimento com sinal controlado ou sintético para confirmar que o algoritmo funciona mecanicamente. O sinal é escolhido por ser representativo do comportamento ideal, não do comportamento real. Resultado de calibração não sustenta afirmação de generalidade.

**Verificação:** experimento com sinal real, não selecionado para produzir resultado favorável. Substrato capturado em condição real ou obtido de fonte pública sem pré-seleção por critério de qualidade. Resultado de verificação pode sustentar afirmação sobre o domínio representado pelo substrato.

Todo experimento deve ser classificado como calibração ou verificação antes da execução. A classificação não pode ser alterada após ver o resultado.

### 4. Resultado negativo tem peso igual ao positivo

Se o experimento não produzir o resultado esperado, o resultado negativo deve ser registrado com o mesmo rigor que um resultado positivo. Resultado negativo informa o escopo real do algoritmo — é informação sobre onde o instrumento não funciona, o que é tão valioso quanto saber onde funciona.

Modificar o sinal, os parâmetros ou o nível de análise após ver um resultado negativo para produzir resultado positivo é violação deste protocolo. Se alguma modificação for feita, o experimento recomeça do zero. A modificação é documentada como uma nova versão do experimento, não como continuação.

### 5. Nenhum parâmetro ajustado após observar o sinal

Parâmetros do eco_adaptativo (φ, α, n_eco, thresholds de H_alpha) são definidos antes de ver o sinal de entrada. Não se ajusta theta ou n_eco após observar o comportamento do substrato no experimento em curso.

Ajustes de parâmetros são legítimos entre experimentos — como resultado de aprendizado acumulado — mas nunca dentro de um mesmo experimento.

### 6. Escopo da afirmação limitado ao escopo do experimento

Nenhum resultado de experimento sustenta afirmação mais ampla do que o experimento permite. Um experimento com sinal puro a uma frequência sustenta afirmação sobre comportamento do algoritmo nesse tipo de sinal. Não sustenta afirmação sobre comportamento geral.

Ao redigir resultados, a afirmação deve especificar: tipo de sinal, condições do experimento, nível de análise utilizado. A precisão do escopo é parte do resultado, não limitação a ser minimizada.

---

## Distinção: tendenciamento vs. escolha metodológica

Nem toda escolha de substrato é tendenciamento. Há diferença entre:

- **Tendenciamento:** escolher o substrato porque se sabe que produzirá o resultado esperado, sem declarar essa escolha
- **Escolha metodológica:** escolher o substrato por razão documentada de pertinência ao domínio, com declaração explícita da limitação de generalidade

A diferença não está no substrato — está na honestidade da declaração. Um sinal sintético usado como calibração explícita é metodologicamente válido. O mesmo sinal apresentado como verificação geral é tendenciamento.

---

## Aplicação retroativa

Os experimentos realizados antes da formalização deste protocolo são revisados de acordo com seus critérios:

| Experimento | Classificação | Status |
|---|---|---|
| ECO BEEP 880 (880 Hz puro) | Calibração | Válido como calibração. Não generaliza. |
| SST-2 +8,98% acurácia | Verificação | Válido. 10 seeds, p=0,0000. |
| δH = −0,0303 em áudio real | Verificação | Válido. Sinal real, não selecionado. |
| eco_text byte level | Nível errado | Removido. Não discrimina semântica. |
| Scanner α-φ em dados espectrais | Verificação (domínio declarado) | Válido dentro do domínio. |

---

## Uma especulação sincera tem peso maior do que um resultado tendenciado

Este protocolo não nasce apesar do erro — nasce através dele. É o manifesto corrigindo seu próprio curso, como a espiral que, a cada novo ciclo, regula novos parâmetros para corrigir o que identificou como anomalia em seu próprio processo.

---

*Florianópolis · 19.06.2026*
*Vitor Edson Delavi*
