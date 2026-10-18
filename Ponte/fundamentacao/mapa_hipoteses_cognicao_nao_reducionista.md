# Mapa de hipóteses: cognição, memória e campo além do cérebro-como-circuito

**Status:** rascunho (atualizado em 18/10/2026: H10 dividida em H10a e H10b, com a magnetorrecepção da Caltech) · **fontes conferidas em resumos de busca; os artigos originais NÃO foram lidos** (como na Ficha 1 do método) · classificação feita pelo assistente, a revisar pelo pesquisador
**Data:** 18 de outubro de 2026
**Origem:** pedido do pesquisador (18/10/2026), depois da discussão sobre memória, campo e reducionismo (Entrada 323 e mensagens seguintes). O pesquisador apontou que o assistente reflete a literatura dominante e que isso pode ser um viés; este mapa responde conferindo fontes, uma a uma.
**Ligações:** `../README.md` (regra de operacionalização) · `campo_organizador_entre_ambitos.md` · `modelo_de_referencia_rede_neural.md` · Entrada 323.

---

## Regra que orienta este mapa (corrigida em 18/10/2026)

**Toda noção filosófica entra nos testes, desde que traduzida em algo mensurável.** A tradução é o trabalho do projeto. O que **conta como evidência** é o que passa no teste, com controles e critério de falha escritos antes. O que ainda não tem tradução fica como **"tradução pendente"**, e não como "fora do escopo".

Estados usados (a auditoria simétrica do projeto, com dois intermediários):
**demonstrado** · **parcial** (evidência real, escopo limitado) · **contestado** (há evidência e contraevidência) · **não demonstrado** · **refutado**.
"Não demonstrado" **não** é "falso".

---

## 0. Base de comparação (o que a corrente dominante sustenta)

### M0. A memória tem substrato em grupos de neurônios (engramas)
- **Afirma:** a memória é armazenada em conjuntos específicos de neurônios e sinapses ("células de engrama"), que podem ser marcados e reativados.
- **Evidência:** revisão de Josselyn e Tonegawa (*Science*, 2020) sobre engramas; experimentos de reativação por luz (optogenética) restauram a lembrança mesmo depois de bloquear a consolidação sináptica. Fonte conferida em resumo; a revisão e os artigos primários não foram lidos.
- **Status:** **demonstrado** em roedores, para memórias específicas.
- **Lacuna:** não responde sobre insight, intuição nem sobre memória fora do sistema nervoso.
- **Fonte:** [Josselyn e Tonegawa, versão aceita (MIT DSpace)](https://dspace.mit.edu/handle/1721.1/126261)

---

## 1. Hipóteses que ampliam onde a cognição ou a memória estão

### H1. Mente estendida (Clark e Chalmers, 1998)
- **Afirma:** processos cognitivos podem incluir objetos do ambiente quando cumprem o mesmo papel funcional que um processo interno ("princípio de paridade"). Exemplo: o caderno de Otto como memória.
- **Evidência:** é uma **tese filosófica**, discutida por comentadores; não é um experimento. As objeções apontam diferenças funcionais entre crença comum e crença estendida.
- **Status:** **contestado** (no plano conceitual).
- **Operacionalizar (proposta):** definir o critério de paridade em termos medidos (acesso, confiabilidade, endosso) e testar em tarefas com e sem o dispositivo externo. Tradução **pendente** para o campo.
- **Fonte:** [Resenha na NDPR](https://ndpr.nd.edu/reviews/the-extended-mind/)

### H2. Cognição 4E (corporificada, situada, estendida, enativa)
- **Afirma:** a cognição depende do corpo e do ambiente, e não só do cérebro. Origem em Varela e colegas (1991) e em Clark e Chalmers (1998); o rótulo "4E" vem de encontros de 2006-2007.
- **Evidência:** família heterogênea de abordagens, descrita como fragmentada.
- **Status:** **contestado**, com resultados empíricos parciais em subáreas (não avaliados aqui).
- **Operacionalizar (proposta):** escolher uma das quatro e definir uma medida (por exemplo, desempenho com e sem acesso ao corpo ou ao ambiente).
- **Fonte:** [Resumo geral (Wikipedia, 4E cognition)](https://en.wikipedia.org/wiki/4E_cognition) (fonte secundária; a entrada da Stanford Encyclopedia não foi recuperada)

### H3. Memória e padrão guardados fora dos neurônios, dentro do corpo (Levin)
- **Afirma:** tecidos sem neurônios guardam "memórias de padrão" bioelétricas; e, em planárias, uma memória aprendida teria persistido depois da regeneração de uma nova cabeça.
- **Evidência:** Shomrat e Levin (*Journal of Experimental Biology*, 2013): planárias treinadas e decapitadas, depois de regenerarem a cabeça, mostraram retenção da familiaridade com o ambiente texturizado. Os autores sugerem rastros fora do cérebro, **mecanismo não identificado**. Revisão de bioeletricidade do desenvolvimento descreve padrões bioelétricos guardados em tecido por mais de uma semana. Baluška e Levin (2016) argumentam que a sinalização usada na memória e decisão é anterior ao sistema nervoso.
- **Status:** **parcial** (resultado publicado, mecanismo em aberto). **Replicação independente: não verificada.**
- **Operacionalizar (proposta):** já é experimental em biologia; para o projeto, a ponte é o **campo bioelétrico como variável medida** (ligação com a Ponte e com `campo_organizador_entre_ambitos.md`).
- **Fontes:** [Tufts, 2013](https://now.tufts.edu/2013/07/18/flatworms-lose-their-heads-not-their-memories) · [Baluška e Levin, 2016](https://www.frontiersin.org/articles/10.3389/fpsyg.2016.00902/text) · [Bioeletricidade do desenvolvimento (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7935058)

---

## 2. Campos eletromagnéticos e o cérebro

### H4. Campos elétricos de neurônios influenciam neurônios vizinhos (acoplamento efáptico)
- **Afirma:** o campo gerado por uma célula polariza a membrana de outras, influenciando o disparo e a sincronia.
- **Evidência:** efeitos de campo são bem estabelecidos em condições patológicas (epilepsia); em cérebros saudáveis o papel funcional é "sugestivo, menos claro" (revisão de Weiss e Faber, 2010, conforme o resumo). Os campos são pequenos, e isolar o efeito in vivo é difícil. Estudos recentes de modelagem relatam mudança de latência de disparos e aumento de sincronia; um comunicado de imprensa do MIT (2026) liga a variabilidade neural a efeitos efápticos (**conferir o artigo antes de usar**).
- **Status:** **parcial**.
- **Operacionalizar (proposta):** em redes artificiais, acrescentar acoplamento de campo entre unidades vizinhas e medir contra controles (é a ponte direta com a hipótese do campo na rede AP).
- **Fontes:** [Revisão de 2010 (PubMed Central)](https://pmc.ncbi.nlm.nih.gov/articles/pmid/20508749) · [Comunicado do MIT, 2026](https://news.mit.edu/index%2Ephp/2026/electric-fields-help-guide-neural-activity-0715) (imprensa, peso baixo)

### H5. Campos eletromagnéticos aplicados de fora alteram a atividade neural
- **Afirma:** um campo externo intenso e focal induz atividade nos neurônios.
- **Evidência:** a estimulação magnética transcraniana (TMS) induz um campo elétrico no cérebro que pode evocar potenciais de ação; consenso de especialistas sobre o que é estimulado (axônios, em especial terminais em cristas de giros).
- **Status:** **demonstrado** (para campos intensos, focais e aplicados de propósito).
- **Lacuna:** não se aplica a campos ambientais fracos, nem diz nada sobre **conteúdo** de informação.
- **Fonte:** [Consenso sobre o que a TMS estimula (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9753778/)

### H6. Campos ambientais fracos afetam a cognição
- **Afirma:** exposição a campos de radiofrequência ou de frequência extremamente baixa altera memória e atenção.
- **Evidência:** revisão sistemática e meta-análise de 2024 (*Environment International*): evidência de certeza muito baixa a baixa, **sem associação clara** e sem relação dose-resposta. Meta-análise anterior (ELF, 9 estudos): pouca evidência de efeitos; achados inconsistentes e difíceis de reproduzir.
- **Status:** **não demonstrado** (evidência fraca e inconsistente), **não refutado**.
- **Fonte:** [Environment International, jun/2024 (DOAJ)](https://doaj.org/article/03c3564e147445978361cba423c52f0a)

### H7. O campo eletromagnético do cérebro é o substrato da consciência (CEMI, McFadden)
- **Afirma:** a consciência é a componente do campo eletromagnético cerebral que integra informação e a "devolve" aos neurônios motores.
- **Evidência:** o campo existe e é medido (EEG e MEG); a pergunta é se ele tem **papel causal**. O próprio autor reconhece que, quando as teorias foram propostas, não havia evidência direta de que a sincronia fosse funcional e não epifenômeno. Esta busca **não encontrou crítica revisada por pares**; só discussão em fóruns (peso baixo).
- **Status:** **não demonstrado**.
- **Operacionalizar (proposta):** testar se perturbar o campo (sem perturbar os disparos) altera o desempenho de tarefas; em redes artificiais, versão análoga com acoplamento de campo.
- **Fonte:** [McFadden, comparação de teorias (*Entropy*, 2023)](https://doaj.org/article/8783cb57198a4d5fb6d38747a9c6b3df)

### H8. Processos quânticos em microtúbulos (Orch OR, Penrose e Hameroff)
- **Afirma:** a consciência surge de colapsos quânticos orquestrados em microtúbulos dos neurônios.
- **Evidência:** a estimativa de Tegmark (2000) dá coerência de cerca de 10⁻¹³ s, curta demais; Hagan, Hameroff e Tuszyński respondem que o cálculo se baseou num modelo híbrido e que, corrigido, o tempo chega a 10⁻⁵ a 10⁻⁴ s. Uma revisão de 2019 considera que o argumento da decoerência talvez pese menos que **outras evidências contrárias** (por exemplo, a organização da rede de tubulina em tecido de camundongo). **Não encontrei material de 2024-2025.**
- **Status:** **contestado**.
- **Fontes:** [Tegmark, 2000](https://arxiv.org/pdf/quant-ph/0005025) · [Revisão de 2019](https://arxiv.org/pdf/1910.08423)

---

## 3. Informação por canal extracorpóreo

### H9. Acesso a informação futura ou remota (precognição, "psi")
- **Afirma:** a mente acessa informação sem os canais sensoriais conhecidos.
- **Evidência:** Bem (2011) relatou efeitos; três tentativas **pré-registradas** de replicação (Ritchie, Wiseman e French, *PLoS ONE*, 2012; n total = 150) não encontraram efeito (p combinado = 0,83). Bem respondeu pedindo mais pesquisa. **Não encontrei meta-análise nesta busca**, então não afirmo o estado do conjunto da literatura.
- **Status:** **não demonstrado**, com replicações negativas pré-registradas. **Não refutado em geral.**
- **Fontes:** [Replicação no Goldsmiths](https://research.gold.ac.uk/id/eprint/7055) · [Artigo no PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3303812)

### H10a. O corpo humano tem receptores de campo ainda não identificados (magnetorrecepção)
- **Afirma:** o cérebro humano responde a campos magnéticos do tamanho do campo da Terra, o que sugere um transdutor biológico ainda não conhecido (hipótese dos autores: cristais de magnetita biogênica).
- **Evidência (Wang, Kirschvink, Shimojo e colegas, *eNeuro*, 2019, Caltech e Universidade de Tóquio):** 34 participantes; EEG durante rotações de um campo do tamanho do da Terra. A amplitude do alfa (8 a 13 Hz) **cai** depois de rotações horizontais em sentido **anti-horário**, **somente quando a componente vertical estática aponta para baixo** (a orientação do Hemisfério Norte); **não** houve resposta com a mesma rotação e a vertical para cima, nem com rotação horária, nem com campo fixo. O efeito depende da componente **estática** do campo, o que, segundo os autores, afasta a indução elétrica. Eles testaram o artefato de indução (flutuação de 10 a 40 μV durante a rotação de 100 ms) com experimentos com fantoma e dizem que ele não explica o efeito. O efeito apareceu **só em parte dos participantes**; a "replicação" foi **dentro do próprio estudo**, em quem respondeu forte. O artigo original **não foi lido por inteiro** (a leitura direta falhou na sessão); os detalhes vêm de resumos e da imprensa científica.
- **Replicação independente:** **não encontrada** em duas buscas (2021 a 2025). **Crítica publicada:** **não encontrada** (só imprensa, que trata o resultado de forma mais forte do que o próprio artigo). Existe uma página do laboratório (Human Magnetic Reception Laboratory, Caltech) e o repositório do Caltech lista material suplementar; **dados abertos não verificados**.
- **Hipótese de apoio:** magnetita no tecido cerebral humano. Kirschvink e colegas (*PNAS*, 1992) a relataram em tecido de cadáveres. O debate posterior é se ela é biogênica ou poluição de combustão (Maher e colegas, 2016, citado de memória, não conferido); um estudo mais recente (*Scientific Reports*) relata distribuição semelhante em vários cérebros, o que Kirschvink leu como indício contra contaminação simples (detalhes e ano não verificados).
- **Status:** **parcial, não replicado.** Um indício real e fraco.
- **O que não mostra:** o resultado é de **detecção de orientação de um campo físico conhecido** (como uma bússola). **Não é um canal de conteúdo** (insight, memória). O alfa cair é uma resposta **inespecífica** a muitos estímulos.
- **Operacionalizar (proposta):** (a) pedir ou acompanhar uma **replicação independente** com os mesmos controles (direção oposta, vertical para cima, campo fixo, fantoma); (b) usar o **desenho de controle por assimetria** como modelo nos experimentos do projeto; (c) para a rede AP, o análogo é um teste de **sensibilidade direcional** a uma entrada que gira, com o controle de sentido oposto de mesma energia.
- **Fontes:** [Wang e colegas, *eNeuro* 6(2), 2019](https://www.eneuro.org/content/6/2/ENEURO.0483-18.2019) · [Laboratório de Recepção Magnética Humana (Caltech)](https://maglab.caltech.edu/human-magnetic-reception-laboratory/) · [Registro no repositório do Caltech](https://authors.library.caltech.edu/records/whbb7-57e98) · [Kirschvink e colegas, 1992 (PNAS, PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC49775)

### H10b. Existe uma fonte de campo com conteúdo de informação (insight, memória) acessada pelo indivíduo (a hipótese do pesquisador)
- **Afirma:** do mesmo modo que o Wi-Fi é um campo acessível pelo celular, o indivíduo acessa um campo do qual vêm informações (insight, intuição), mediante predisposição.
- **Evidência:** **nenhum estudo direto encontrado.** A analogia técnica é consistente em si (o Wi-Fi é um campo externo ao aparelho, com fonte e receptor conhecidos). Falta o equivalente de **emissor e de conteúdo** para o caso humano. H5 mostra sensibilidade do cérebro a campos fortes; H10a, um indício de sensibilidade a campos fracos de orientação; H6 e H9 não sustentam efeitos de campos fracos sobre a cognição nem acesso extracorpóreo.
- **Status:** **não demonstrado; tradução pendente.**
- **Operacionalizar (proposta, não executada):** (a) estudos com **blindagem** de campos e tarefas de evocação de memória, intuição e resolução de problemas, com controle e avaliação cega; (b) em redes artificiais, **canal de campo** entre unidades, com versão com canal bloqueado como controle; (c) critério de falha escrito antes. Qualquer efeito precisaria ser replicado por terceiros.
- **Formulação precisa do que não está demonstrado:** não há demonstração de acesso à informação **por um canal além dos sentidos conhecidos e sem emissor identificado**. O ser humano já acessa informação extracorpórea pela visão, pela audição e pelo tato.

---

## 4. O que o mapa mostra

- As hipóteses que **ampliam o suporte da cognição** dentro do corpo e do ambiente material (H1 a H5) têm literatura publicada, com graus diferentes de evidência. **Não são todas marginais.**
- As que **atribuem à cognição um canal de informação fora do corpo** (H9, H10b) não têm, nesta busca, demonstração; H9 tem replicações negativas pré-registradas.
- **O ponto de apoio mais firme** para o projeto, por ser demonstrado e mensurável, é H5 (a sensibilidade do cérebro a campos) e a parte parcial de H4, H3 e H10a (magnetorrecepção, não replicada).
- Esta busca **não esgota** a literatura: foi um conjunto de buscas simples, com fontes em resumo.

## 5. Itens abertos

- [ ] Ler os artigos originais de H3, H4, H8 e H9 (não lidos).
- [ ] Buscar meta-análises de H9 e críticas revisadas por pares de H7.
- [ ] Buscar estudos de blindagem eletromagnética e cognição (H10b).
- [ ] Ler o artigo da Caltech por inteiro (grupos de resposta, intensidade do campo, câmara de blindagem, tamanho do efeito) e verificar se há dados abertos (H10a).
- [ ] Buscar replicações da magnetorrecepção humana por EEG em bases de preprints e no PubMed; nesta sessão não apareceu nenhuma.
- [ ] O pesquisador revisar a classificação e a coluna "operacionalizar".
- [ ] Decidir se H4 e H7 geram um experimento concreto na rede AP (acoplamento de campo entre unidades, com controle).
