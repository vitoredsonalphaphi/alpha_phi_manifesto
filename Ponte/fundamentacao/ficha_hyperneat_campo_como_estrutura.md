# Ficha — HyperNEAT, ES-HyperNEAT e o campo como estrutura prévia da rede
## Insumo para a Ponte `[f ○ c]` (fundamentação) e para a agenda Rede AP

**Status:** **RASCUNHO** · levantamento de 19/10/2026 · rótulos: *confirmado agora (fonte secundária)* / *de memória* / *hipótese*
**Origem:** pedido do pesquisador (sessão de 19/10/2026): verificar quais pesquisas instrumentalizam um campo como prerrogativa para a estruturação da rede, e **como** o fazem, para servir de base de método (não de cópia) à formação do campo da Rede AP.
**Relação:** `campo_organizador_entre_ambitos.md` · `modelo_de_referencia_rede_neural.md` · Entrada 325 do Manifesto 03

**Limite desta ficha:** os artigos originais **não foram lidos**. O acesso a arXiv, OSF, ScienceDirect e bioRxiv falhou nesta sessão. Tudo abaixo vem de resumos de busca, de uma página de pacote de software e de resumos institucionais. Antes de citar o método em texto formal, é preciso ler os originais.

---

## 1. Esquema de leitura (o mesmo para toda pesquisa)

Cada pesquisa desta ficha é lida pelos mesmos campos. É também o esquema proposto para alimentar o eco ressonante do metaprocessador (ver seção 5).

| Campo | Pergunta |
|---|---|
| **Espaço** | Onde as unidades ficam? Quem escolhe as posições, e quando? |
| **Campo** | O que é o "campo"? Uma função, um gradiente, uma penalidade? |
| **O que o campo gera** | Pesos, presença de conexões, posição de neurônios, módulos? |
| **Fixo ou aprendido** | O campo é dado de antemão ou é evoluído/treinado? |
| **Resultado reportado** | O que os autores dizem que ganharam? |
| **Limites** | O que é caro, frágil ou não mostrado? |
| **Verificação** | De onde vem a informação desta ficha |

---

## 2. HyperNEAT (Stanley, D'Ambrosio e Gauci, 2009)

| Campo | Leitura |
|---|---|
| **Espaço** | Os neurônios ficam em posições de um espaço geométrico, o **substrato**. As posições e o número de neurônios são **decididos por uma pessoa, antes da execução**, e não mudam durante a evolução. |
| **Campo** | Uma pequena rede chamada **CPPN** (*Compositional Pattern-Producing Network*). Ela funciona como uma função sobre as coordenadas: recebe a posição de dois pontos e devolve um valor. |
| **O que o campo gera** | O **peso** da conexão entre os dois pontos e, em algumas variantes, se a conexão **existe**. Cada conexão possível do substrato é consultada. |
| **Fixo ou aprendido** | O CPPN é **evoluído** (pelo algoritmo NEAT). A rede grande não é evoluída diretamente: ela é lida do CPPN. |
| **Resultado reportado** | A rede herda regularidades geométricas do problema (simetria, repetição). O mesmo CPPN pode gerar redes de tamanhos diferentes. |
| **Limites** | O posicionamento manual dos neurônios. |
| **Verificação** | *confirmado agora (fonte secundária)*: resumos de busca, a página do pacote goESHyperNEAT e citações em apresentações. Que o CPPN usa funções como seno e gaussiana para produzir simetria e repetição é *de memória*. |

*Em palavras:* primeiro se decide o espaço; depois uma função desenhada sobre esse espaço diz como as partes se ligam. A geometria vem antes da conexão.

---

## 3. ES-HyperNEAT (Risi e Stanley, 2010 e 2012)

| Campo | Leitura |
|---|---|
| **Espaço** | O mesmo substrato, mas as posições dos **neurônios ocultos** deixam de ser escolhidas à mão. |
| **Campo** | O mesmo CPPN. |
| **O que o campo gera** | Além dos pesos, **onde nascem os neurônios ocultos e com que densidade**. O método lê a posição dos nós a partir do próprio padrão de pesos: coloca mais nós onde a saída do CPPN **varia mais**, usando uma decomposição do tipo árvore quaternária (*quadtree*). |
| **Fixo ou aprendido** | Evoluído. O substrato pode mudar a geometria durante o treino. |
| **Resultado reportado** | Testado em tarefas múltiplas, navegação em labirinto e uma **retina modular**. Os autores relatam que as redes geradas assumem propriedades como **topografia neural e regularidade geométrica**. |
| **Limites** | Caro: a busca percorre todo o hipercubo de coordenadas. Uma variante posterior (*Iterated ES-HyperNEAT*) busca só cortes bidimensionais. |
| **Verificação** | *confirmado agora (fonte secundária)*: resumos do artigo de 2012 (*Artificial Life*), da versão de 2010 (GECCO) e de uma tese. |

*Em palavras:* aqui o campo não só liga as partes, ele decide **onde as partes aparecem**. É o caso da literatura mais próximo de "o campo precede a forma" em rede artificial.

---

## 4. Outras pesquisas lidas pelo mesmo esquema

| Pesquisa | Espaço | Campo | Gera | Fixo ou aprendido | Verificação |
|---|---|---|---|---|---|
| **Redes espacialmente embutidas** (seRNN, Achterberg, Akarca e colegas, *Nature Machine Intelligence*, 2023) | unidades em coordenadas 3D | penalidade que encarece conexões longas, mais um termo de comunicabilidade | a estrutura da conectividade durante o treino | o espaço é fixo; os pesos são treinados | *confirmado agora (fonte secundária)*; as propriedades emergentes exatas **não** foram confirmadas nos trechos |
| **Informação posicional** (Wolpert, biologia do desenvolvimento) | o tecido | gradiente de morfógeno com limiares | regiões e identidades celulares | dado pelo organismo | *confirmado agora (fonte secundária)*. As fontes descrevem as células **lendo** a posição, e não um pré-padrão fixo de antemão. Há um modelo alternativo ("vigilância de vizinhança") |
| **Pré-padrão bioelétrico** (Manicka e Levin, *Cell Reports Physical Science*, 2025, DOI 10.1016/j.xcrp.2025.102865) | modelo computacional de tecido | campo bioelétrico | pré-padrão morfogenético | — | **não verificado**: só título, revista e DOI. O resumo não foi obtido |
| **Programas de desenvolvimento neural** e **HyperNCA** (Najarro, Sudhakaran, Risi e colegas, 2022 e 2023) | sem espaço prévio | regras locais de crescimento | a rede inteira, por crescimento | aprendido | *confirmado agora (fonte secundária)*; não há campo prévio |

---

## 5. O que isto oferece à Rede AP (*hipótese*)

Esta seção é proposta, não resultado.

- **Método aproveitável, citado e implementado de forma própria:** (1) definir o espaço antes; (2) definir uma função de campo sobre o espaço; (3) consultar o campo para obter pesos, posições ou densidade. Usar o método publicado como base, com citação e implementação própria, é prática normal; copiar código exigiria verificar a licença.
- **A diferença que a Rede AP testaria:** no HyperNEAT o campo é **evoluído**; na proposta Alpha-Phi o campo seria **dado** por φ e ancorado por α. Isso é testável contra controles com a mesma liberdade (campo aleatório, grade regular, outras razões, campo evoluído).
- **O gancho por ferramenta:** o mais próximo na literatura é o ES-HyperNEAT, em que o campo determina onde nascem os neurônios. Não foi encontrado um campo que dê **a cada módulo um direcionamento próprio**. Os detalhes do gancho estão em aberto, como o pesquisador indicou.
- **O eco ressonante no metaprocessador:** o esquema da seção 1 é a forma proposta de "treinar" o eco com poucas pesquisas (parametrizá-lo por fichas, e não por treino estatístico). Para não repetir o viés de construção (Entrada 315), o conjunto precisa incluir também casos em que uma estrutura prévia **não ajudou**. Nenhum desses casos foi levantado ainda.

---

## 6. Pendências

1. Ler os originais: Stanley, D'Ambrosio e Gauci (2009, *Artificial Life*); Risi e Stanley (2012, *Artificial Life*); Achterberg e colegas (2023); Manicka e Levin (2025).
2. Obter o resumo e o método de Manicka e Levin (2025).
3. Levantar casos negativos: estruturas prévias que não melhoraram o aprendizado.
4. Escrever os critérios do experimento do campo com ganchos antes de qualquer código.

---

## Fontes consultadas (resumos de busca)

- [HyperNEAT: Generative NeuroEvolution for Deep Learning (arXiv 1312.5355)](https://arxiv.org/pdf/1312.5355)
- [goESHyperNEAT (página do pacote)](https://pkg.go.dev/github.com/vdyagilev/goESHyperNEAT)
- [Evolving the placement and density of neurons in the HyperNEAT substrate (UCF)](https://stars.library.ucf.edu/scopus2010/1039)
- [Enhancing ES-HyperNEAT to evolve more complex regular neural networks (UCF)](https://stars.library.ucf.edu/scopus2010/2704)
- [An enhanced hypercube-based encoding for evolving the placement, density and connectivity of neurons (ITU)](https://pure.itu.dk/da/publications/an-enhanced-hypercube-based-encoding-for-evolving-the-placement-d/)
- [Spatially-embedded recurrent neural networks (bioRxiv)](https://www.biorxiv.org/content/10.1101/2022.11.17.516914.full.pdf)
- [Positional Information, a concept underpinning developmental biology (Aberdeen)](https://abdn.elsevierpure.com/en/publications/positional-information-a-concept-underpinning-our-understanding-o/)
- [Publicações de bioeletricidade, Laboratório Levin](https://as.tufts.edu/biology/levin-lab/publications/bioelectricity)

---
*Florianópolis · 19 de outubro de 2026 · Ponte · Vitor Edson Delavi · Claude*
