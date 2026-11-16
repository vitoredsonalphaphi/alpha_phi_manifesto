# Protocolo de aceitação e revisão de pontes (v0)

**Ponte · Laboratório de Tradutibilidade**
**Status: RASCUNHO v0 · proposta do assistente, a pedido do pesquisador (16/11/2026) · aguarda a revisão do pesquisador · não é canônico · revisável (Entrada 414).**
**Estágio:** refinamento do instrumento de tradução (ética → estética → geometria). **Não instrumentaliza a Rede AP e não usa valores numéricos** (Entrada 417).
**Origem:** Entradas 414 (revisibilidade e núcleo irrevogável), 415 (convergência dos instrumentos), 417 (o estágio) e 418; esboço da seção 4 da Entrada 417.

**Como ler:** a AGM e o FDR entram aqui como **lógicas**, e não como cálculos. O FDR, como cálculo, exigiria valores p, que não existem hoje. Verificação das fontes: AGM (Alchourrón, Gärdenfors e Makinson, 1985), a prioridade epistêmica (Gärdenfors e Makinson, 1988) e o sistema de Doyle (1979) foram **confirmados em busca**, sem leitura do texto completo; o FDR (Benjamini–Hochberg), a variante para dependência e o alinhamento de redes estão **de memória**.

---

## 1. O que é uma ponte

Uma **ponte** é uma ligação **proposta** entre dois itens registrados, com: um **motivo**, um **grau de proximidade** (1 analogia, 2 homologia de forma, 3 identidade de medida), uma **origem** (quem propôs) e as **fontes** do motivo. Hoje existem 66 pontes, todas propostas pelo assistente; o pesquisador aceitou 65 e recusou 1.

Uma ponte **não é** uma afirmação de que os dois itens são o mesmo. É uma hipótese de que há uma relação, no grau declarado.

## 2. Princípios (herdados das Entradas 414 e 417)

1. **O núcleo irrevogável fica fora do protocolo.** Nenhuma revisão de ponte o altera.
2. **Revisar é acrescentar, não apagar.** Cada mudança guarda data, gatilho, o que mudou e o estado anterior.
3. **As razões acompanham a afirmação.** Se uma razão cai, sabe-se o que depende dela.
4. **A aceitação do pesquisador é juízo humano, e não evidência independente** (como já diz o Plano de Tradutibilidade). O protocolo mantém as duas coisas separadas.
5. **Mudança mínima, com prioridade.** Cede primeiro o que está mais abaixo na ordem de prioridade.
6. **Flexibilidade não é imunidade.** Toda revisão registra a causa e a previsão anterior; se qualquer dado pudesse ser acomodado reorganizando as pontes, elas deixariam de poder ser testadas.

## 3. Os estados de uma ponte

| Estado | Significado | O que a sustenta |
|---|---|---|
| **P · proposta** | Existe, tem motivo, grau e fontes; ainda sem juízo | A proposta |
| **A · aceita (juízo do pesquisador)** | O pesquisador a aceitou | Juízo humano, **não independente** |
| **E · com evidência independente** | Passou por pelo menos uma verificação que não depende do pesquisador nem de quem a propôs | Ver a seção 6 |
| **V · em revisão** | Um gatilho foi acionado | Ver a seção 5 |
| **S · substituída** | Outra ponte ou estrutura a substituiu; aponta para a sucessora | A razão da substituição |
| **R · recusada** | O pesquisador a recusou; **fica guardada, com a razão** | A razão da recusa |

**Regras de transição:**
- P → A ou R, por decisão do pesquisador.
- A → E, só com uma verificação da seção 6.
- Qualquer estado → V, por um gatilho da seção 5. De V, volta ao estado anterior, ou vai a S ou R.
- **O grau de proximidade só sobe com evidência do tipo correspondente:** o grau 3 (identidade de medida) exige uma medida; o grau 2, uma homologia de forma verificável; o grau 1 basta com o motivo.
- Nada é apagado: R e S ficam consultáveis, e uma ponte recusada pode voltar a P com uma **razão nova**.

## 4. Aceitação em lote (a lógica do FDR, sem cálculo)

1. **Fixar o conjunto antes de julgar.** Cada **rodada** declara a lista das pontes que serão julgadas. Acrescentar pontes depois cria outra rodada; não se muda o conjunto de uma rodada em andamento. (Sem isso, nenhum controle de erro tem sentido.)
2. **Aceitar o conjunto admitindo erro.** Aceitar não é afirmar que cada ponte é verdadeira. Cada rodada declara, em palavras, a **tolerância** a pontes fracas entre as aceitas (baixa, média ou alta), antes de julgar.
3. **Ordenar pela força da evidência, e não pela ordem de leitura.** Aceita primeiro o que tem mais base (E; depois A com a base textual verificada; depois A só pelo motivo). O que fica abaixo da linha da tolerância é **aceito provisoriamente**, com essa marca.
4. **Mais propostas pedem mais cuidado.** Quanto mais pontes numa rodada, maior a chance de algumas serem aceitas por acaso. A taxa de aceitação muito alta de uma rodada, em especial quando todas foram propostas por uma só origem, é um **sinal para a verificação cega** (seção 6), e não uma prova de qualidade.
5. **Quando existirem valores p** (medidas das pontes), aplicar o procedimento de Benjamini–Hochberg à rodada; havendo forte dependência entre as pontes (elas compartilham itens), usar a variante conservadora ou a versão por permutação (de memória, a conferir).

## 5. Revisão (a lógica da AGM)

**Gatilhos de revisão:**
- **(a)** informação nova sobre um dos itens ligados (a ficha mudou);
- **(b)** contradição entre duas pontes aceitas, ou entre uma ponte e um canônico ou o núcleo;
- **(c)** uma previsão registrada que falhou;
- **(d)** uma estrutura proposta como melhor, segundo um critério **escrito antes** (explica ou prevê mais com custo igual); ela entra **em paralelo** e só então é promovida.

**Ordem de prioridade (a mais alta cede por último):**
1. o **núcleo irrevogável** (nunca é cedido por este protocolo);
2. os **canônicos congelados** (mudam só por versão nova, com aprovação do pesquisador);
3. o **cinturão protetor** (hipóteses);
4. as **fichas**;
5. as **pontes**, e entre elas: as com evidência independente (E) cedem depois das só aceitas (A); a que tem mais pontes dependentes cede depois.

**Mudança mínima.** Antes de recusar uma ponte, tentar nesta ordem: (i) baixar o grau; (ii) acrescentar uma qualificação ao motivo; (iii) dividir a ponte em duas, uma firme e uma fraca; só então (iv) recusar ou substituir.

**Razões (à maneira de Doyle).** Cada ponte guarda as razões: o motivo, as fontes, quem a aceitou, a evidência. Quando uma razão cai, o protocolo **lista as pontes que dependiam dela** e as marca V.

**Registro de versão.** Cada mudança anota: data, gatilho, o que mudou, o estado anterior e quem decidiu. (É o princípio da Entrada 414 aplicado às pontes.)

**Recuperação.** Uma ponte recusada ou substituída pode voltar, com uma razão nova, e a razão antiga fica registrada.

## 6. Evidência independente e verificação cega

Para uma ponte passar de **A** a **E**, ao menos uma destas, e sempre declarada:
- **(i) Leitura cega:** outra pessoa ou outra IA julga a plausibilidade da ligação **sem ver o grau proposto nem o motivo**, e o resultado é comparado com o estado da ponte.
- **(ii) Base textual:** a fonte do motivo é lida e conferida no original (e não de memória).
- **(iii) Medida:** quando houver uma medida da proximidade dos dois itens, com o seu controle.

**Amostra de verificação cega para uma rodada.** Sortear um pequeno número de pontes da rodada (a proposta é um punhado, a decidir pelo pesquisador), esconder o motivo e o grau, e pedir o juízo a um leitor independente. Se o leitor discorda de várias, a rodada volta a V; se concorda, as pontes da amostra passam a E, e as demais continuam A, **sem herdar** a evidência.

**Alinhamento de leituras.** Comparar o conjunto de pontes com outra leitura (por exemplo, a triangulação ou a semântica sobre os mesmos itens) e listar três coisas: as pontes **concordantes** (as duas leituras as veem), as **isoladas** (só uma leitura as vê) e as **discordantes** (as leituras se contradizem). As isoladas e as discordantes alimentam o gatilho (b).

## 7. Decisão sobre as cascas (a pedido do pesquisador, deixada ao assistente)

Para a versão da esfera das cinco cascas (cinco canais do aparelho de tradução, uma casca por canal), a **atribuição adotada na v0** é a ordem em que a Agenda da Ponte lista os canais, do centro para fora:

| Casca (do centro para a superfície) | Canal |
|---|---|
| 1 (a mais interna, raio 1/φ⁴) | triangulação |
| 2 (raio 1/φ³) | semântica |
| 3 (raio 1/φ²) | térmico |
| 4 (raio 1/φ) | cromático |
| 5 (a superfície, raio 1) | ritmo |

**Razões:** (a) é a ordem da própria Agenda; (b) a triangulação é o **único canal testado e canônico**, e fica no centro, o que é coerente com o modelo espacial (o centro é a âncora) e com a ordem de prioridade da seção 5; (c) o revezamento vai do centro para fora, como no modelo canônico.

**Cuidado:** nesta atribuição o **raio marca o estágio e, por coincidência de escolha, também a prioridade** (a triangulação é a mais prioritária e está no centro). Isto é uma **escolha**, e não uma necessidade: o raio e a prioridade são duas coisas, e o protocolo funciona sem a esfera. Decisão do assistente, **reversível** pelo pesquisador. A cópia 3D da página das cinco esferas, em artefato, usa esta atribuição nos rótulos.

## 8. Rodada 0: o que o protocolo pede do estado de hoje (qualitativo)

Estado: 66 pontes propostas pelo assistente; **65 aceitas (A)** e **1 recusada (R)** pelo pesquisador, em 16/11/2026; **nenhuma com evidência independente (E)**.

1. **Registrar como "A provisória"** as 65 aceitas, até haver uma verificação da seção 6. A aceitação é do pesquisador, e a origem é toda do assistente.
2. **A alta taxa de aceitação** (65 de 66) pode refletir boas propostas ou deferência à origem; o protocolo não decide qual. Pede a **amostra de verificação cega** da seção 6.
3. **A razão da ponte recusada não está registrada.** (O motivo proposto pelo assistente está na ficha; a razão da recusa, não.) Pelo princípio 3, cabe ao pesquisador registrar a razão para que a ponte fique guardada com ela.
4. **O conjunto está fixado** (66), e qualquer ponte nova seria outra rodada.
5. **A tolerância** da rodada 0 não foi declarada: a decidir pelo pesquisador (baixa, média ou alta).

## 9. Limites

- É uma **proposta de método**, sem teste em prática. A primeira aplicação seria a rodada 0.
- Hoje não há valores p nem medida das pontes: o FDR e o alinhamento funcionam como **lógica**, e não como cálculo.
- A confirmação por **busca** das fontes é parcial (sem leitura dos textos completos).
- O leitor independente da verificação cega **não existe ainda**: precisa ser definido (outra pessoa, ou outra IA, com o cuidado de não ser a mesma origem das propostas).

## 10. Decisões em aberto, do pesquisador

1. A tolerância da rodada 0 (baixa, média ou alta).
2. Quem seria o leitor independente da verificação cega.
3. A razão da recusa da ponte recusada.
4. Se a atribuição de canais às cascas da seção 7 fica como está.

---
*Florianópolis · 16 de novembro de 2026 · Sessão Good Morning*
*Vitor Edson Delavi · Claude*
