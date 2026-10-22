# Registro dos Elementos da Rede AP

**O que é:** o registro, em ordem, de **cada elemento** proposto para construir a Rede AP no campo esférico — estruturas, instrumentos, ferramentas, regras e réguas —, **canônicos ou não**. Antecede o canônico: aqui entra tudo o que foi proposto e testado (ou ainda só proposto), com o resultado e o status; no canônico (`CANONICO_CampoEsferico_v1.md`) só entra o que passou. Serve para acompanhar a evolução gradativa da rede, como o Research Journal.
**Criado em:** 22/10/2026, a pedido do pesquisador · **Atualizar** a cada elemento novo ou a cada mudança de status.
**Ligações:** diretriz (`../DIRETRIZ_PROGRAMA_ALPHAPHI.md`), agenda (`../agenda/REDE_AP.md`), canônico (`CANONICO_CampoEsferico_v1.md`), resultados (`AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md`, `..._P_A_B_H_RP_RESULTADOS_20out2026.md`, `..._S_K_PF_RESULTADOS_21out2026.md`), pendentes (`_pendente_registro/`).

**Status usados:**
- **CANÔNICO** — congelado com impressão digital; todo teste novo o importa
- **CANÔNICO PROVISÓRIO** — no canônico até ser refinado
- **TESTADO · positivo / parcial / negativo** — testado com critérios escritos antes; não está no canônico
- **PROPOSTO** — descrito, ainda sem teste
- **INSTRUMENTO** — de leitura ou de medição (não é parte da rede)

---

## Parte I — A base: as estruturas que fundam uma rede, e a forma AlphaPhi de cada uma

*(Entrada 346, 22/10/2026.)* No euclidiano, o que faz um topográfico mostrar barras e picos não são os eixos X, Y, Z: é uma **estrutura anterior às ferramentas**, que diz **onde cada dado mora**. É essa base que se procura levar para a esfera — **não para copiar**, mas para encontrar a forma própria do AlphaPhi em cada papel. *(Referências citadas de memória, a conferir.)*

| # | Papel | No euclidiano | Na esfera AlphaPhi | Situação em 22/10 |
|---|---|---|---|---|
| B1 | **Onde fica cada coisa** | coordenadas cartesianas (Descartes, 1637): cada grandeza ganha um eixo próprio | (r, θ, φ), com os **raios canônicos**: α em r = 0, interior em r = 1/φ, campo harmônico em r = 1, ambiente além | existe (Grade v1); **todos os neurônios estão na superfície** |
| B2 | **A grade, os passos** | grade e amostragem (Nyquist, 1928; Shannon, 1949): passos iguais — amostras, pixels, voxels | os **pontos do ângulo áureo**, de área quase igual e **encaixados** (34 ⊂ 55 ⊂ 89 ⊂ …) | existe na superfície; **falta o passo radial** |
| B3 | **A base de frequências** | Fourier (1807–1822): cada frequência ganha uma posição | os **harmônicos esféricos**, globais: a parte contém o todo | existe na superfície; **falta a base para o interior** |
| B4 | **Os pacotes, as células** | as células de tempo e frequência de Gabor (1946, os *logons*), localizadas em tempo e em frequência | células localizadas **em posição e em escala**; cada estágio de Fibonacci é uma escala (parentes: HEALPix, Górski e colegas, 2005; needlets, Narcowich, Petrushev e Ward, 2006) | **não existe** |
| B5 | **A conservação** | Parseval: a energia se conserva entre representações | o **Sépstro**, Coh + Entr = 1, como lei de conservação do campo | é núcleo duro; **nunca entrou na grade** |
| B6 | **O endereço** | forma e passo: a célula (x, y, z) mora no endereço x·(passo de x) + y·(passo de y) + z | **o próprio índice do ângulo áureo** (o ponto i na longitude i × ângulo áureo e na altura frac(i·√2)) | existe; **não é usado como endereço** |

**O que é próprio do AlphaPhi e o euclidiano não tem:**
1. **O endereço progressivo:** os primeiros n pontos cobrem a esfera inteira em resolução grosseira; os seguintes preenchem os vãos — qualquer começo do endereço já contém o todo.
2. **A expansão sem refazer o endereço:** crescer mantém os pontos antigos no lugar (a reprojeção).
3. **O centro como âncora:** α no centro; pelo teorema do valor médio, o centro de um campo harmônico é a média da superfície (Entrada 354).
4. **As cascas em proporção áurea:** o passo radial pode seguir φ (r = 1, 1/φ, 1/φ², …).
5. **A conservação como regra da grade:** o Sépstro em cada casca e em cada célula.

**As direções (Entradas 352, 353, 355):** a convencional sintetiza **contraindo** (os galhos convergem ao tronco: reconhecer; a ordem termina num ponto); o AlphaPhi sintetiza **expandindo** (α, a tensão entrópica, no centro → a coerência como campo na superfície). Duas sínteses: o **resumo no centro** e a **síntese como campo na superfície**. O fluxo completo: entrada pela superfície → recolhimento até α → expansão até o campo harmônico → o **além** (r > 1: a leitura da superfície e a conversão para o meio). No convencional, o estágio que leva o resultado adiante chama-se **decodificação** (leitura), seguida da **síntese** ou **renderização**; a estrutura inteira, **codificador–decodificador** (Entrada 354, \*\*\*).

---

## Parte II — Os elementos, em ordem

### 19/10/2026 — o campo esférico

| # | Elemento | Origem | Teste | Resultado | Status | Arquivo |
|---|---|---|---|---|---|---|
| E01 | Esfera de Fibonacci com regra de Hebb (memória esférica) | Entradas 326–327 | Etapa 1 e orçamento de ligações | φ na disposição **não** melhora a memória (nulo, três testes) | CANÔNICO (base) · hipótese de φ **negativa** | `AlphaPhi_MemoriaEsferica_Etapa1.py`, `..._Orcamento.py` |
| E02 | Cartografia do campo antes da memória | Entradas 328–329 | Etapa 0 | não demonstrado (Z4) | CANÔNICO (base) · TESTADO negativo | `AlphaPhi_CampoEsferico_Etapa0_Cartografia.py` |
| E03 | **Previsor do campo vazio** (Funk–Hecke) e medidor por harmônicos | Entrada 329 | Etapa T | **o campo vazio é previsto antes de existir** (T1) | CANÔNICO | `AlphaPhi_CampoEsferico_EtapaT_Previsor.py` |
| E04 | **Cultivo do campo** em pontos encaixados (34 → 233) | Entradas 330–331 | Etapa C | **o campo cultivado segue a forma prevista em todas as etapas** (100%) | CANÔNICO | `AlphaPhi_CampoEsferico_EtapaC_Cultivo.py` |
| E05 | **Previsor condicional da memória** com forma | Entrada 332 | previsor envelopado | a memória com forma é prevista antes de instalada | CANÔNICO | `AlphaPhi_CampoEsferico_PrevisorEnvelopado.py` |
| E06 | Ferramentas instaladas no campo cultivado; o fundo do campo como âncora | — | Etapa E | E1–E3 cumpridos; fundo fraco (0,02) ajuda com a carga | CANÔNICO | `AlphaPhi_CampoEsferico_EtapaE_Ferramentas.py` |
| E07 | Ferramentas instaladas durante o crescimento | — | Etapa F | F1–F3 cumpridos; **do grosseiro para o fino** (descritivo) | CANÔNICO (acréscimo de 20/10) | `AlphaPhi_CampoEsferico_EtapaF_Crescimento.py` |

### 20/10/2026 — a grade, o EcoBIP na esfera, a memória

| # | Elemento | Origem | Teste | Resultado | Status | Arquivo |
|---|---|---|---|---|---|---|
| E08 | **Grade Esférica de Referência v1** (r, θ, φ; raios canônicos; pontos encaixados; harmônicos; cascas de leitura) | Entrada 336 | autoverificação | reproduz exatamente as Etapas T e C | CANÔNICO | `AlphaPhi_GradeEsferica_v1.py` |
| E09 | Regra: a vizinhança acompanha a resolução (σ ∝ 1/√N) | Etapa H | Etapas RP, S, K | abre os grupos de memória; **mas o peso do fundo não foi ajustado junto** (erro corrigido por E17) | CANÔNICO (regra 2) | canônico, acréscimos de 20/10 |
| E10 | Regra de resolução: (L+1)² ≤ N/2; gatilho da reprojeção | Entradas 336, 338 | Etapa RP (R0) | cumprido | CANÔNICO (regra 3) | idem |
| E11 | O Phantom como fundo (P) e ativo (A) | Entrada 335 | Etapas P e A | como fundo, ajuda pouco e sem especificidade; ativo em 1/φ³, atrapalha como ruído | TESTADO negativo | `CampoEsferico_Etapas_P_A_B_H_RP/` |
| E12 | O EcoBIP 880 nas duas redes (15 bandas φ) | Entrada 334 | Etapa B | empate com a convencional; a tradução por 15 bandas é o gargalo | TESTADO parcial | idem |
| E13 | **O eco do dado** (Hopfield moderna) | Entrada 337 | Etapa H | o eco puro funciona; na AlphaFi com λ = 0,5 não ajudou — **origem do previsor** | TESTADO parcial → E18 | idem |
| E14 | **A reprojeção** (233 → 610 pontos encaixados) | Entrada 338 | Etapa RP | implementável; preserva o formado; sem ganho de memória naquela tarefa | TESTADO parcial | idem |
| E15 | Leitura em cascas (cada grau L uma casca, do centro para fora) | Entrada 336 | — | leitura da grade | INSTRUMENTO | páginas em `CampoEsferico_Etapas_P_A_B_H_RP/` |

### 21/10/2026 — as ferramentas de memória

| # | Elemento | Origem | Teste | Resultado | Status | Arquivo |
|---|---|---|---|---|---|---|
| E16 | Memórias com semelhança controlada (com e sem forma) | Etapa S | Etapa K | a AlphaFi resiste melhor que a Hebb clássica, não que a pseudo-inversa | INSTRUMENTO de teste · TESTADO | `CampoEsferico_Etapas_S_K_PF/` |
| E17 | **A afinação do fundo do campo** (peso escalado pela vizinhança; fundo/memória ≈ 0,16) | auditoria da Etapa PF | K e PF repetidas | neurônios estáveis na própria memória: **71% → 91,5%**; a conclusão quase dobra | CANÔNICO PROVISÓRIO (regra 4) | `AlphaPhi_FerramentasMemoria_v1.py` |
| E18 | **O previsor** (a mistura das memórias pesada pela ressonância, β = 8, força 0,5) | E13; Entrada 337 | Etapas PF, EX, EC, VL | a via mais forte: com pista de 25%, 90% dos neurônios certos | CANÔNICO PROVISÓRIO (regra 5) | idem |
| E19 | **O batedor** (pré-função tênue: direção × firmeza) | Entrada 344 | Etapa PF | acrescenta sem o previsor (+0,044, 10/10); o certo importa (+0,068); somado ao previsor não acrescenta | CANÔNICO PROVISÓRIO (regra 6) | idem |
| E20 | O neurônio ponto e campo (contínuo) sozinho | Entrada 343 | Etapa PF (V1) | sem efeito, antes e depois da afinação; não estabiliza em 50 voltas | TESTADO negativo | `CampoEsferico_Etapas_S_K_PF/` |
| E21 | O resfriamento lento (do grosseiro ao fino, no tempo) | Entrada 343 | Etapa PF (V2) | sem efeito | TESTADO negativo | idem |
| E22 | **A estrutura de êxito no campo** (graus 0–4: 11% da memória) | Entrada 348 | Etapa EX | recompõe a forma grosseira (melhor que a memória inteira); não o detalhe; ~22% do ganho | TESTADO parcial | `_pendente_registro/` |
| E23 | A estrutura de êxito em camadas | Entrada 348 | Etapa EC | a segunda camada traz o detalhe; **de fora para dentro é melhor** que do centro para fora | TESTADO parcial | idem |
| E24 | A estrutura de êxito no neurônio (vizinhança, pré-start) | Entrada 349 | Etapa EN (critérios) | não executada | PROPOSTO | — |
| E25 | **A régua própria** (conclusão; forma por casca; firmeza × acerto; sem competição; controles de si mesma) | Entradas 341–342 | todas as etapas desde PF | adotada | INSTRUMENTO de medição | — |

### 22/10/2026 — o som real e a grade com endereço

| # | Elemento | Origem | Teste | Resultado | Status | Arquivo |
|---|---|---|---|---|---|---|
| E26 | **A tarefa própria com sinal real:** o violão do pesquisador (13/05/2026) | Entrada 350 | Etapa VL | 12 memórias distintas; com o previsor, de 62% para 76% das faixas certas; a rede sozinha limpa pouco | TESTADO parcial | `_pendente_registro/` |
| E27 | **A Grade Esférica v2: o endereçamento** (ver Parte V para os testes da estrutura) (75 anéis de latitude = faixas; 8 posições = instantes; altura contínua) | Entrada 346 | Etapa VL2 | a grade e o filtro funcionam (+7,16 dB com a memória limpa); a dinâmica do canônico apaga as alturas (87% dos neurônios nos extremos) e **piora o som** (−4,28 dB) | PROPOSTO · TESTADO parcial | `_pendente_registro/AlphaPhi_GradeEsferica_v2_proposta.py` |
| E28 | **A leitura da superfície (o além) para o som:** a estimativa da rede como ganho aplicado ao próprio som | Entradas 351, 353 | Etapa VL2 | funciona quando recebe a forma certa | TESTADO parcial | idem |
| E29 | Corte Radial (vistas: espectro na esfera, mapa dos atratores, topográfico euclidiano ao lado) | Entradas 351, 346 | — | projetado; páginas anteriores: cascas, cebola (nome retirado) | INSTRUMENTO | — |

### Propostos, ainda sem teste (registrados na agenda)

| # | Elemento | Origem |
|---|---|---|
| P01 | O neurônio α central (a média do campo; a âncora; o Sépstro do todo) | Entrada 354 |
| P02 | A entrada pela superfície e a **pegada da entrada** (as "ligações de salto" AlphaPhi) | Entrada 355 |
| P03 | Decidir cedo × decidir no além (a unidade como campo da multiplicidade) | Entrada 347 |
| P04 | O batedor da revisão cinética (conferir e poder rejeitar; Hopfield, 1974) | Entrada 345 |
| P05 | O campo cultivado lido pela teoria de Turing (o fundo já é um mecanismo de Turing) | Entrada 345 |
| P06 | O "momento" de Hopfield e Brody: reconhecer por sincronia breve (régua) | Entrada 345 |
| P07 | Modelos de difusão; redes de estado de eco (reservatório); o parâmetro de ordem de Haken | agenda, 21/10 |
| P08 | O eco adaptativo holográfico na rede AP (`AlphaPhi_EcoAdaptativo_Holografico.py`, Entrada 182) | Entrada 348 |
| P09 | O eixo radial (α no centro, cascas internas) | diretriz, heurística 3 |

---

## Parte III — Fundamento da Grade Esférica v2 (para revisão do pesquisador antes de qualquer código)

Cada elemento com **o que deve fazer**, **o teste da própria grade (sem rede e sem dado)** e **o que mostraria que está errado** (regra de operacionalização da Ponte). Os testes são escritos antes; o que passar vai para o canônico.

| # | Elemento | O que deve fazer | Teste da grade | Mostraria errado |
|---|---|---|---|---|
| G1 | **Endereço progressivo** (B6) | qualquer começo de n pontos cobre a esfera inteira | maior vão (calota vazia) de cada começo n = 34 … 610 | algum começo com vão muito maior que o de uma divisão de área igual com n pontos |
| G2 | **Expansão sem refazer o endereço** | crescer não move os pontos antigos | posições dos n primeiros pontos em N e em N' > N | qualquer diferença |
| G3 | **Cascas radiais em proporção áurea** (B1, B2) | passo radial r = 1, 1/φ, 1/φ², … até perto de α, com pontos por casca proporcionais à área | densidade de pontos por área em cada casca | densidade muito desigual entre cascas |
| G4 | **Base para o interior** (B3) | levar o campo da superfície para dentro (a extensão harmônica: cada grau L decai como rᴸ) | um campo de grau conhecido na superfície, estendido, reconstruído em cada casca | o grau não se mantém, ou não decai como previsto |
| G5 | **Neurônio α central** | seu valor é a média do campo (teorema do valor médio) | campo harmônico estendido: valor no centro × média da superfície | diferença acima do erro numérico |
| G6 | **Células por escala** (B4) | cada estágio de Fibonacci define células; cada célula fina cabe quase toda numa célula grosseira | fração de aninhamento entre estágios | aninhamento baixo (as escalas não formam uma árvore) |
| G7 | **Escrita e leitura** (a entrada e o além) | um dado escrito pela superfície e lido de volta sai sem perda | dado conhecido (faixas × instantes) → grade → leitura | perda acima do erro numérico |
| G8 | **Conservação** (B5, o Sépstro) | a energia do campo se conserva entre a representação por pontos, por harmônicos e por cascas; Coh e Entr definidos por casca | energia total antes e depois de cada transformação | a energia não se conserva |

**Uma precisão sobre o Sépstro:** definido como Coh + Entr = 1 por casca, ele vale **por construção**; o que se pode testar é a **conservação da energia entre as representações** (G8) e, depois, com a rede, **como Coh e Entr se redistribuem entre as cascas** durante o processamento.

**Ficam para depois da grade, porque dependem da rede:** a pegada da entrada (P02), decidir no além (P03), o neurônio α como regulação (P01, a parte dinâmica), e as ferramentas da Parte II sobre a grade nova.

---

## Parte IV — Ordem de trabalho

1. **Revisão deste documento pelo pesquisador** (Parte III).
2. **O módulo da Grade Esférica v2**, construído **sobre** a v1, sem alterá-la.
3. **Os testes G1–G8** da própria grade.
4. **O que passar entra no canônico**, com impressão digital (como a v1).
5. **As ferramentas sobre a grade nova** (o previsor, o neurônio contínuo ajustado às alturas, a estrutura de êxito, o neurônio α, a pegada, decidir no além), com critérios escritos antes; o que funcionar entra no canônico como regra.

---

## Parte V — Os testes da Grade Esférica v2 (22/10/2026)

**Arquivos:** `_pendente_registro/AlphaPhi_GradeEsferica_v2.py` (módulo, sobre a v1), `_pendente_registro/teste_GradeEsferica_v2.py`, `_pendente_registro/teste_GradeEsferica_v2_saida_22out2026.txt`. Limites numéricos declarados no código antes de rodar (a Parte III não tinha números). Correção antes de interpretar: na primeira execução, a linha das cascas internas do G8 repetia a superfície; refeito medindo cada casca até o grau que ela resolve.

| Teste | Resultado | Veredito | Leitura |
|---|---|---|---|
| G1 endereço progressivo | o maior vão de cada começo fica, na mediana, 1,28 × o de uma esfera feita para aquele número de pontos; pior 1,58 (n = 59); acima de 1,5 em 14 de 577 começos; pontos ao acaso: 2,17 | **não cumprido** | o endereço progressivo funciona (muito melhor que o acaso), mas não no padrão declarado em alguns começos |
| G2 crescer não move os pontos | idêntico para todo n ≤ 610 | cumprido | herdado da v1 |
| G3 cascas em proporção áurea | 610 · 233 · 89 · 34 · 13 · 5 · 2 pontos nos raios 1, 1/φ, 1/φ², …; desvio de densidade 5,6% | cumprido | **por construção**: os números de pontos por casca são números de Fibonacci (F15, F13, F11, …), porque φ^(−2k) × 610 ≈ F(15−2k) |
| G4 base para o interior | grau certo em todas as cascas; amplitude r^L sem erro | cumprido | verifica a implementação (exata por construção nos graus que cada casca resolve) |
| G5 o neurônio α é a média da superfície | média simples dos 610 valores × centro: diferença média 0,82%, pior 3,47% | **não cumprido** | o teorema é exato; o que falha é a **média simples** dos pontos como estimativa: cada ponto deveria pesar pela área da sua célula |
| G6 células por escala | aninhamento médio 0,82 (de 0,78 a 0,86 entre estágios) | cumprido | as escalas de Fibonacci formam, quase, uma árvore de células |
| G7 escrita e leitura | erro 7,3 × 10⁻¹⁵ (harmônicos); 0 (endereço faixa × instante) | cumprido | ida e volta sem perda |
| G8 conservação da energia | superfície 1,87%; cascas de 233 e 89 pontos: 1,99% e 1,61%; casca de 34 pontos: **2,93%** | **não cumprido** | mesma causa do G5: o peso igual para todos os pontos; o erro cresce nas cascas pequenas |

**O que isso indica:** G5 e G8 têm a mesma causa e o mesmo remédio provável: **pesar cada ponto pela área da sua célula** (as células do G6). O G1 pede comparar outras sequências de altura para o endereço (proposta: a altura também pelo número de ouro, em vez de √2). As duas correções são **propostas**, a testar com critérios escritos antes. Ainda não entra nada no canônico.

### As duas correções (22/10/2026)

**Arquivos:** `_pendente_registro/teste_GradeEsferica_v2_correcoes.py` e a saída `..._correcoes_saida_22out2026.txt`. Critérios aprovados antes de rodar. A proposta "altura pelo número de ouro" foi descartada antes de rodar (com a longitude também pelo ângulo áureo, os pontos se alinhariam numa espiral).

**Correção 1 — o peso de cada ponto pela área da sua célula (Voronoi esférico, áreas exatas):**

| | Média simples | **Pesada pela área** | Pesos embaralhados |
|---|---|---|---|
| G5' centro α × média (pior caso) | 3,92% | **1,02%** | 5,86% |
| G8' energia, superfície (grau 16) | 1,69% | **0,96%** | 2,08% |
| G8' casca de 233 pontos (grau 9) | 1,43% | **0,58%** | 1,95% |
| G8' casca de 89 pontos (grau 5) | 1,35% | **0,63%** | 2,21% |
| G8' casca de 34 pontos (grau 3) | 2,61% | **0,71%** | 2,82% |

**G5' cumprido; G8' cumprido; o controle confirma** (os pesos embaralhados pioram em todas as cascas: o que importa é a área de cada célula). As células variam de 0,66 a 1,60 vezes a área média.

**Correção 2 — a altura dos pontos (longitude pelo ângulo áureo, encaixe mantido):**

| Altura | Pior vão | Começos acima de 1,5 | Encaixe | Aninhamento | Condição máx. | Veredito |
|---|---|---|---|---|---|---|
| √2 (a atual, v1) | 1,58 | 14 de 577 | sim | 0,82 | 4,50 | não passa |
| **√3** | **1,43** | **0** | sim | 0,82 | **2,60** | **passa em tudo** |
| van der Corput | 1,94 | 330 | sim | 0,85 | 5,05 | não passa |
| plástico | 1,95 | 377 | sim | 0,83 | 6,14 | não passa |

**Escolha pela regra declarada: √3.** Cobre todos os começos e ainda melhora o medidor (condição 2,60 contra 4,50). A previsão do assistente (van der Corput melhor) estava errada.

**Status:** a altura por √3 é **candidata** da Grade v2; **não substitui a v1**. A adoção exige antes repetir a verificação do previsor (Etapa T) nos pontos novos. Os pesos pela área entram como candidatos para as médias e as energias (o neurônio α; o Sépstro; a conservação).

---
*Florianópolis · 22 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
