# Agenda REDE-AP
## Construção da Rede Neural Alpha-Phi — Itens Ativos

**Como referenciar:** "coloca na Agenda REDE-AP" ou "verifica na Agenda REDE-AP"
**Arquivo:** `agenda/REDE_AP.md`
**Última atualização:** 21 de outubro de 2026 (criada em 23/09/2026)
**Ver também:** `agenda/REDE_AP_METODO.md` ("Agenda REDE-AP (Fichas Método)"): fichas de método, portões G0–G5 e critérios de validação.

---

## QUADRO DE SITUAÇÃO — 07/10/2026

Leitura cruzando commits e Research Journal; as marcações abaixo são do assistente e aguardam confirmação do pesquisador.

| Item | Situação |
|---|---|
| Métricas no domínio da rede (II) | **Feito** — E08–E10; corrigido (`grade_r` → `ativacao_coerencia`, 01/10) e refeito com D_φ e Coh_rel (03/10) |
| Mecanismo de injeção (II) | **Parcial** — testados φ-init, α como âncora residual e MPAP como regularizador; falta o Phantom como schedule de taxa de aprendizado |
| Controles por amplitude RMS (II) | **Parcial** — controles do φ-init feitos; controles do Phantom por RMS pendentes |
| Selagem; Phantom sem selagem; Grade R no espaço original (II) | Pendentes |
| Scanner de Coexistência Espectral (I) | Pendente — sem resultado registrado |
| Seção III (retroprojeção, verificação geométrica) | Aguarda a seção II completa |
| Seção IV (família de α, Collatz, arquétipos, inicialização fractal) | Não iniciada |
| Seção V (acoplamento multi-substrato) | Não iniciada |

### Correção de 08/10/2026 — valor de α do EcoBIP

- O **EcoBIP 880 original** usa **α\* = 1/3** como peso do digital: `ALPHA·quadrada + (1−ALPHA)·FM-φ`, com FM-φ de portadora 220/φ Hz, seguido da cascata `eco_eq` de 5 dobras até o campo harmônico (`BEEP880_17S.py`). Reproduzido em 08/10: campo harmônico no ciclo 10, β máx = 4,2098.
- Esta agenda tratava **α = 1/137 como o valor do EcoBIP** e varria só [1/200, 1/100]. Isso **não cobria o valor correto**. Corrigido nos itens IV e V abaixo.
- **Definições divergentes no repositório (a padronizar):** `AlphaPhi_Scanner_Topografico.py` e `AlphaPhi_Scanner_TopogColab.py` geram o EcoBIP como `(1−α)·quadrada + α·FM-φ` com α = 1/137, ou seja, **outro sinal**. Resultados do scanner com essa definição não valem para o EcoBIP original.
- Os testes de 08/10 que usaram α = 1/137 (STFT bruto, envelope, tensor 2D do scanner v1) ficam com status **não demonstrado nessas representações**, não "refutado" (Entrada 316).

**Feito fora desta agenda (23/09 a 07/10):** MPAP e suas auditorias; varreduras de ângulo do Phantom (E11–E13); campo FI icosaédrico (primeira implementação, 01/10); scanner topográfico com MPAP; reestruturação dos testes (03/10, resultados em `REDE_AP/AlphaPhiNet_Reestruturado_RESULTADOS_03out2026.txt`); scanner de áudio (pronto, aguarda a gravação do pesquisador).

**Feito fora desta agenda em 08/10:** reconstrução etapa a etapa do EcoBIP original (α = 1/3) com scanner topográfico 3D interativo e controle (mesma cascata só na quadrada); Entrada 316 (auditoria simétrica). Observação visual: o relevo do EcoBIP final tem várias cristas coerentes, o controle tem uma; **ainda não quantificado**, e a orientação da textura (grade) ainda não foi medida.

---

## QUADRO DE SITUAÇÃO — 20/10/2026

Balanço feito na conversa de 20/10/2026 (resultados em `REDE_AP/AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md` e `REDE_AP/AlphaPhi_CampoEsferico_Etapas_P_A_B_H_RP_RESULTADOS_20out2026.md`).

| Item | Situação |
|---|---|
| Mecanismo de injeção / inicialização W₀ (II) | **Feito de outro modo:** o campo esférico é cultivado antes e a rede nasce dentro dele (Etapas T, C) |
| Phantom sem selagem na rede (II) | **Feito** (Etapas P e A): como fundo ajuda pouco, sem especificidade de forma; ativo em 1/φ³ atrapalha como ruído |
| Controles de mesma amplitude (II) | **Feito** nessas etapas (ruído, seno, embaralhado) |
| Reprojeção (II) | **Implementada pela primeira vez** (Etapa RP): preserva o antigo, sem ganho de memória nesta tarefa |
| Selagem; Grade R no espaço original; Phantom como ritmo do aprendizado (II) | Pendentes |
| Scanner de Coexistência Espectral (I) | Pendente |
| Seção III | Pendente |
| α agnóstico (IV) | Parcial: α\* = 1/3 do EcoBIP corrigido e usado; família de α não iniciada |
| Collatz, arquétipos, Tratado 137 (IV); multi-substrato (V); itens A–G (VII) | Não iniciados |

**Feito além desta agenda (19 e 20/10):** previsor da forma do campo antes de existir (Etapa T); cultivo encaixado de Fibonacci (Etapa C); previsor da memória, inclusive com forma; regra de resolução (L+1)² ≤ N/2, que é também o gatilho da reprojeção; Grade Esférica de Referência v1; canônico congelado com impressões digitais; diretriz do programa (Lakatos); Hopfield moderna (eco do dado) sobre a esfera; leitura em cascas (scanner esférico).

**Comparação com a convencional até aqui:** a AlphaFi empata (Etapa B, EcoBIP) ou fica atrás (Etapas H e RP). O ganho real está na previsibilidade e na mensurabilidade do campo, não no desempenho de memória.

---

## I. EXECUÇÕES PENDENTES (código pronto, aguarda Colab)

- [ ] **Scanner de Coexistência Espectral** — S₁ (ruído branco) vs S₂ (serial φ) vs S₃ (mistura)
  - Código fornecido na sessão de 23/09/2026
  - Observar: a estrutura φ do serial sobrevive à mistura com ruído igual? O PLV subgrave persiste?
  - Responde se o serial tem "peso estrutural" mensurável antes do acoplamento completo

---

## II. QUESTÕES ESTRUTURAIS DA REDE (Entrada 287, Seção 04)

- [ ] **Reconstituir a selagem** — três opções:
  - Remover completamente (phantom sem selagem — já executado)
  - Rampas que não se sobreponham (fração de borda < 1/2)
  - Aplicar selagem apenas fora da banda dominante
  - Critério de aceitação: terceira estrutura mantida em 10/10 cones e 100% dos quadros

- [ ] **Repetir experimento de campo com phantom sem selagem**
  - Base: `AlphaPhi_SerialPhantom_SemSelagem_COLAB.py` (commitado)
  - Comparar métricas de campo com e sem selagem

- [ ] **Usar controles na rede com mesma amplitude RMS**
  - Controles: ruído branco, ruído rosa, sinal de entrada antes da cascata
  - O phantom só tem efeito próprio se diferir estatisticamente desses controles

- [x] **Definir métricas no domínio da rede** (não usar β da cascata) — *feito em 30/09 (E08–E10); revisto em 01/10 e 03/10*
  - Espectro dos pesos ou das ativações
  - Entropia por camada (Shannon)
  - Rank efetivo das matrizes de peso
  - Estabilidade do treino (variação da loss entre épocas)
  - Nota: `grade_r()` media concentração de ativação, não geometria romboédrica (renomeada); D_φ e Coh_rel (03/10) são instrumentos que podem dizer "não"

- [ ] **Mudar mecanismo de injeção** — *parcial (03/10)*
  - Em vez de soma linear, testar phantom modulando algo estrutural:
    - Inicialização W₀ (Estágio II — Entrada 280)
      - *19/10: CPPN como campo prévio (Entradas 326 e 327). Memória esférica, etapa 1: **nulo para a geometria** (`REDE_AP/AlphaPhi_MemoriaEsferica_Etapa1_RESULTADOS_19out2026.md`). Próximo: teste com orçamento de ligações.*
      - *19/10: orçamento (nulo para φ), Etapa 0 (não demonstrado), **Etapa T (previsor do campo vazio confirmado na esfera de Fibonacci)** e **Etapa C (campo cultivado segue a forma prevista em todas as etapas)** — `REDE_AP/AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md`; Entradas 328–331. Próximo: previsor envelopado (estágio 2).*
    - Ganhos por camada — *testado na ablação do φ-init: o efeito é de escala; 0,5^k empatou com 1/φ^k*
    - Taxa de aprendizado (phantom como schedule adaptativo) — *pendente*

- [ ] **Testar Grade R no espaço onde surgiu**
  - Scanner euclidiano dos pacotes Fibonacci com os mesmos controles
  - Verificar se Grade R aparece no scanner antes de tentar injetá-la na rede
  - **Acrescentado em 08/10:** no scanner topográfico **3D** (TopogColab, modos ECO e LAP), com o **EcoBIP original em α = 1/3**: (i) contar as cristas coerentes por etapa da cascata, EcoBIP contra controle (cascata só na quadrada); (ii) medir a orientação da textura do LAP sem ângulo-alvo fixo. Critérios escritos antes, nos dois sentidos, com controle positivo (Entrada 316). Verificar também se as linhas pretas dos gráficos 3D são projeções do cursor (hover) do Plotly.

---

- [ ] **Reprojeção — o instrumento de expansão no limite** *(acrescentado em 20/10/2026; antes só no Research Journal da Rede AP, Entradas 02, 06 e 07, e no Manifesto 03, Entradas 312–313)*
  - Protocolo de início (RJ 02/06): **estabelecer o limite** → φ conforme o ambiente → α por substrato (eco ressonante) → acoplar o atrator; o limite (pré-estágio) **precisa ser adaptado para reprojetar** quando necessário
  - RJ 07: "[Pré-estágio] Limite declarado + instrumento de reprojeção" — nunca implementado
  - Manifesto 312–313: r = 1 vê a borda primeiro e aciona o reprojeto; gatilho proposto dCoh/dt < 0 por vários ciclos; cada ciclo de reprojeção = volta ao mesmo ponto em coerência mais alta (espiral de Vico)
  - Tradução na esfera (20/10): detectar o limite (regra da grade: graus pedidos > grau medível; ou previsor; ou queda de coerência) → crescer para a próxima etapa de Fibonacci com pontos encaixados → reavaliar o campo nos pontos novos → reescalar vizinhança e ferramentas (σ ∝ 1/√N) → liberar os graus novos
  - Primeiro teste: Etapa RP (critérios na conversa de 20/10/2026) — **executada em 20/10:** gatilho cumprido e o formado preservado; sem ganho de memória nesta tarefa (memórias correlacionadas). Ver `REDE_AP/AlphaPhi_CampoEsferico_Etapas_P_A_B_H_RP_RESULTADOS_20out2026.md`, seção 5

- [ ] **Codificação preditiva — fundamento para instrumentalizar todas as ferramentas** *(acrescentado em 21/10/2026, a pedido do pesquisador; Entrada 344)*
  - Princípio (citado de memória: Rao e Ballard, 1999; Friston): uma **previsão desce**, a **evidência sobe**, e o processamento segue a **diferença** entre as duas
  - As ferramentas do projeto já têm essa forma (antecipar, comparar, corrigir): eco ressonante (pré-função), previsor do campo (Etapas T e C), eco do dado / Hopfield moderna (Etapa H), Ecoatrator
  - A fazer: ficha de fundamentação (com fontes **verificadas**) e releitura de cada ferramenta por esse esquema

- [ ] **Neurônio ponto e campo — Hopfield com valores contínuos** *(acrescentado em 21/10/2026; Entrada 343)*
  - Hoje cada neurônio guarda só o sinal (±1) e descarta a força do campo local; o neurônio ponto e campo guarda os dois (valor contínuo entre −1 e +1)
  - Variantes a considerar (de memória): neurônios graduados (Hopfield, 1984); neurônios com temperatura e resfriamento lento (do grosseiro para o fino, no tempo); neurônios de fase (ligação com coerência e fase)
  - Hipótese: o indeciso espera perto de zero em vez de fixar o erro; pode ser a causa de a AlphaFi chegar perto da memória sem completá-la (Etapas H, RP, S, K)
  - Não é quântico: a analogia com o ponto e campo vale como estrutura, não como física

- [ ] **O batedor — pré-função de cada neurônio, com duas vias de atração** *(acrescentado em 21/10/2026; Entrada 344)*
  - Origem: o eco ressonante como pré-função (emite, escuta o que volta, depois a função opera)
  - Mecânica: versão tênue da rede percorre as ligações sem decidir (a via do dado) + o previsor projeta o esperado (a via do campo); o neurônio firma onde as duas concordam e espera onde discordam
  - Verificação sem competição: a AlphaFi comparada consigo mesma (binária, contínua, com batedor, com previsor, com os dois, com batedor enganado), pela régua própria
  - Primeiro teste: Etapa PF (critérios na conversa de 21/10/2026)

- [ ] **O batedor da revisão cinética — conferir e poder rejeitar antes de decidir** *(acrescentado em 21/10/2026, a pedido do pesquisador)*
  - Fundamento: a **revisão cinética** de Hopfield (PNAS, 1974; Ninio, 1975, de forma independente): a célula faz uma etapa extra **antes de se comprometer**, gasta energia para conferir a peça e a descarta se estiver errada; cada etapa de conferência multiplica a precisão
  - "Cinética" aqui é a **cinética química** (velocidade das reações e tempo de cada etapa), não energia cinética
  - Diferença em relação ao batedor da Etapa PF: lá o batedor só **soma uma atração**; aqui ele **confere** — onde batedor e previsor discordam, o neurônio **volta atrás** (desfaz a decisão) ou adia, em vez de fixar
  - A lição de Hopfield: a conferência tem custo (mais voltas) e precisa poder rejeitar; medir o ganho de precisão por etapa de conferência
  - Verificação sem competição: a AlphaFi afinada com o batedor atual × com o batedor que confere e rejeita × sem batedor, pela régua própria (conclusão, forma por casca, firmeza × acerto)

- [ ] **O campo cultivado lido pela teoria de Turing (reação e difusão)** *(acrescentado em 21/10/2026, a pedido do pesquisador)*
  - Fundamento (citado de memória: Turing, "The chemical basis of morphogenesis", 1952): manchas e listras surgem de **ativação local e inibição lateral**; o equilíbrio entre as duas decide se o padrão se forma ou se desfaz
  - O fundo do campo da rede AP (DoG: reforço perto, inibição longe) **já é** um mecanismo desse tipo; a afinação de 21/10 (fundo × memória) é o mesmo tipo de equilíbrio
  - A fazer: prever, pela teoria de Turing na esfera, o grau dominante do campo cultivado e a faixa de pesos do fundo em que o padrão se mantém; comparar com o previsor de Funk–Hecke (Etapa T) e com a afinação (regra 4 do canônico)
  - Possível régua própria: o fundo que se ajusta sozinho (o problema das memórias parecidas) como regulação de Turing

- [ ] **O "momento" de Hopfield e Brody — reconhecer por sincronia breve** *(acrescentado em 21/10/2026, a pedido do pesquisador)*
  - Fundamento (citado de memória: Hopfield, olfato e tempo dos disparos, anos 1990; Hopfield e Brody, "What is a moment?", c. 2000–2001): grupos de neurônios reconhecem um padrão quando entram em **sincronia por um breve instante**, e não pelo estado final
  - Ligação com o projeto: coerência e fase (PLV, EcoBIP, Phantom); neurônios de fase (item "neurônio ponto e campo")
  - A fazer: uma **régua própria nova** para a AlphaFi — medir, ao longo da dinâmica, os instantes de coerência do campo com cada memória (pico de sincronia), em vez de só o estado final; aplicar igualmente a todas as versões, escrita antes
  - Conferir as fontes antes de usar (artigos não lidos nesta sessão)

- [ ] **Modelos de difusão — a forma que surge do ruído guiada por um campo** *(acrescentado em 21/10/2026, a pedido do pesquisador)*
  - Fundamento (citado de memória: Sohl-Dickstein e colegas, 2015; Song e Ermon, 2019; Ho e colegas, 2020): da termodinâmica da difusão; um **campo de gradientes**, aprendido antes, aponta em cada ponto para onde a forma está, e conduz o ruído até a forma
  - Ligação: a forma mais bem-sucedida hoje, na tecnologia, de "o campo precede a forma" (Entrada 345, família "o campo que orienta"); mesma estrutura do previsor e do campo cultivado
  - A fazer: ficha com fontes verificadas; testar a recuperação na esfera como um campo de gradientes por casca (do grosseiro para o fino), pela régua própria

- [ ] **Redes de estado de eco e computação de reservatório — o parente do eco ressonante** *(acrescentado em 21/10/2026)*
  - Fundamento (citado de memória: Jaeger, 2001, echo state networks; Maass, 2002, máquinas de estado líquido; Fernando e Sojakka, 2003, um balde de água que reconheceu voz): um meio dinâmico rico "ecoa" a entrada, e só a leitura da resposta é aprendida
  - Ligação: o eco ressonante e o EcoBIP; o campo esférico como **reservatório**, sem treinar as ligações; ponte direta entre o projeto e a literatura de redes neurais
  - A fazer: ficha com fontes; o campo cultivado como reservatório para uma tarefa de processamento própria (item 3 de "o que falta" para a rede AP)

- [ ] **O laser de Haken — o parâmetro de ordem como modelo de coerência** *(acrescentado em 21/10/2026)*
  - Fundamento (citado de memória: Hermann Haken, sinergética, anos 1970–1990; o "computador sinergético"): no laser, um único campo coletivo (o **parâmetro de ordem**) põe em fase milhões de átomos e "escraviza" as partes; o mesmo princípio reconhece padrões
  - Ligação: coerência e fase; a relação entre o ponto e o campo dita em física; família "o campo que orienta"
  - A fazer: ficha com fontes; medir, na rede AP, um parâmetro de ordem (a sobreposição com a memória ou a coerência por casca) como variável que comanda a dinâmica

- **Referências para consulta futura** *(da conversa de 21/10/2026, todas de memória, a conferir)*: Gabor (holografia, 1948) e Pribram (cérebro holonômico, não confirmado); Lin, Ozcan e colegas (redes ópticas difrativas, 2018); Khatib (campos potenciais na robótica, 1986: os dois polos); Xu e colegas (geração por fluxo de Poisson, 2022: linhas de campo elétrico); Shannon (entropia da informação, 1948); Kirkpatrick, Gelatt e Vecchi (recozimento simulado, 1983); Prigogine (estruturas dissipativas); Lorenz (atrator estranho, 1963); Bak, Tang e Wiesenfeld (criticalidade auto-organizada, 1987). Os "campos mórficos" de Sheldrake não são aceitos pela ciência atual: se entrarem, com esse rótulo.

- [ ] **O eco adaptativo holográfico na rede AP** *(acrescentado em 21/10/2026, a pedido do pesquisador)*
  - Origem: `AlphaPhi_EcoAdaptativo_Holografico.py` (27/08/2026, Entrada 182): o eco adaptativo como arquitetura que se contém a si mesma (cada nó com a mesma forma em qualquer profundidade; Sépstro local Coh + Entr = 1 em cada nó; selagem por φ; níveis 3^(n+1)); a função `selecionar()` foi deixada como ponto de acoplamento com a rede de atratores
  - Fundamentos de "a parte contém o todo" (citados de memória, a conferir): a holografia de Gabor (1948); as representações holográficas reduzidas de Plate (1995) e as arquiteturas de vetores simbólicos (Kanerva); na esfera, a transformada por harmônicos é o "domínio de Fourier" do campo, onde cada coeficiente fala do todo
  - A fazer: acoplar o protótipo à Grade Esférica v1 e às Ferramentas de Memória v1; verificar se a autossimilaridade se mantém na esfera

- [ ] **A estrutura de êxito — um fragmento holográfico que contém a forma do todo, instalado no campo como alvo** *(acrescentado em 21/10/2026, proposta do pesquisador)*
  - A ideia: em vez de o previsor conhecer a memória inteira, o campo guarda só uma **microestrutura** dela, uma parcela que contém o código do todo, como **estrutura de êxito prévio**: a forma que o conjunto dos neurônios deve alcançar; seria o pré-start do neurônio, mas acoplado no campo (no previsor ou no atrator)
  - Tradução na esfera (proposta do assistente): a parcela que contém o todo, em resolução baixa, são os **coeficientes de grau baixo** da memória (por exemplo, graus 0–4: 25 de 225 números, cerca de 11%); outra opção, mais holográfica, é uma projeção aleatória curta da memória inteira (cada número fala de todos os neurônios)
  - Teste proposto, sem competição: o previsor guiado pela memória inteira × pela estrutura de êxito (graus baixos) × pela projeção holográfica curta × sem previsor; pergunta: quanto do ganho do previsor a parcela recupera, e se ela ajuda também na forma fina (o "isento de perda")
  - Ligação: Entradas 337 (o eco do dado), 343 (o ponto e campo), 344 (o batedor), 345 (a condição prévia: família "a previsão que antecipa")

- [ ] **A estrutura de endereçamento esférico — a base que antecede as ferramentas (Grade Esférica v2)** *(acrescentado em 22/10/2026, a pedido do pesquisador; Entrada 346)*
  - Diagnóstico: a Grade v1 tem pontos e harmônicos, mas **não tem endereço**; cada faixa vai a um harmônico que se espalha pela esfera inteira, por isso a esfera não mostra barras de sinais e a memória guarda só a silhueta do espectro
  - Estruturas do euclidiano a observar (de memória, a conferir): coordenadas cartesianas; grade e amostragem (Nyquist, Shannon); Fourier; **células de tempo e frequência de Gabor (1946)**; base ortogonal; endereçamento por forma e passo
  - Parentes esféricos (de memória): **HEALPix** (Górski e colegas, 2005); **needlets** (Narcowich, Petrushev e Ward, 2006); a esfera de Fibonacci encaixada já é uma divisão quase de área igual e hierárquica
  - A fazer: células esféricas com endereço; regra de lugar para frequência e tempo (frequência no raio, do centro α para fora; tempo na volta em torno); modo local × modo holográfico, pela régua própria e pela imagem; guardar a altura de cada faixa (valores contínuos; Etapa VL2, o violão audível); vistas novas: espectro radial (as barras dentro da esfera), mapa dos atratores (o rastro do estado e o peso do previsor), o topográfico euclidiano ao lado; a página passa a se chamar **Corte Radial**

---

## III. QUESTÕES PARA QUANDO CHEGADA A HORA

- [ ] **Retroprojeção do atrator E do serial na rede AP**
  - Fundamento: Entrada 280 (ECO-BIP Fantasma como modulação da inicialização)
  - Ainda não chegamos neste estágio — aguarda II estar completo

- [ ] **Verificação geométrica da Grade R nos resultados da rede**
  - Aplicar Scanner Geométrico Latente ao espaço de pesos/ativações da rede treinada
  - Verificar se Grade R emerge no espaço da rede após treino com phantom

---

## IV. FUNDAMENTOS FILOSÓFICO-MATEMÁTICOS A INCORPORAR

- [ ] **α agnóstico por substrato — Família Adaptativa de α** *(Entrada 288 + pendentes.md item 6)*
  - Não fixar o valor 1/137 — preservar a **arquitetura da tensão** (expansão/contração)
  - Em vez de um α único, construir uma **família de α adaptativos** — um por substrato,
    análogo aos parâmetros adaptativos do Scanner (que já se ajusta ao input recebido)
  - Cada membro da família mantém a mesma estrutura dual (inteiro/decimal = atrator/entropia)
    mas com valores calibrados empiricamente para cada ocasião/substrato
  - Esboço da estrutura:
    ```python
    ALPHA_FAMILIA = {
        # substrato          : (α_expansao, α_contracao)  ← a calibrar experimentalmente
        'audio_ecobeep'      : (3,          1/3),         # α* operacional do EcoBIP original (BEEP880_17S.py); 1/137 fica como âncora constante, não como peso de mistura
        'eeg_sintetico'      : (None,       None),        # a determinar
        'fala_quadrada'      : (None,       None),        # a determinar
        'ruido_fmphi'        : (None,       None),        # a determinar
        'texto_estruturado'  : (None,       None),        # a determinar
    }
    # A rede AP seleciona o α correto para cada substrato que entra
    # Como o Scanner seleciona seus parâmetros adaptativos para cada input
    ```
  - **Quando desenvolver:** ao tratar a seção V (Acoplamento Multi-Substrato)
  - **Referências:** Entrada 288 (Collatz) · `MANIF_02/FILOSOFICA_alpha_inteiro_e_constante.md`
    · hipótese de universalidade (item V abaixo)
  - **Dado novo (03/10):** como âncora residual, γ₀ = 1/137 não superou γ₀ = 0,01 nem 0,001 (acerto 0,9596 contra 0,9604 e 0,9604, lr = 1e-3). Isso é compatível com α agnóstico por substrato: o valor específico não foi privilegiado nessa tarefa.

- [ ] **Collatz como referência estrutural da tensão** *(Entrada 288)*
  - par → ÷2 = contração = .035999... = entropia
  - ímpar → ×3+1 = expansão = 137 = atrator
  - 137 é ímpar: habita o lado expansivo por natureza matemática
  - Verificar se o ECO-BIP tem assinatura par/ímpar nas φ-bandas
  - Goldbach como modo inverso (decomposição vs colapso da mesma tensão)
  - **Observação de correção (14/10/2026):** a Entrada 288 atribui o bloco `00729927` (período 8) a α. Esse bloco é a dízima de **1/137 exato** (137 × 729927 = 99999999; 1/137 = 0,00729927007299…). A dízima de 1/137,035999 é 0,00729735… e não repete com período 8. O inteiro 137 e a paridade usados na Entrada 288 não dependem do bloco; a leitura do bloco como "respiração" de α, sim. A Entrada 288 permanece com o texto original.

- [ ] **Conectar: Tratado 137 → Rede AP**
  - Arquivo: `MANIF_02/FILOSOFICA_alpha_inteiro_e_constante.md`
  - Separação explícita ALPHA_INTEIRO / ALPHA_CONSTANTE já justificada no arquivo
  - Candidato a capítulo arXiv — integrar com resultados experimentais da rede

- [ ] **Arquétipos como parâmetros estruturais — isomorfismo com AP** *(01/10/2026)*
  - A filosofia AP já está pronta como **arquétipo** — modelo pré-existente que a rede segue
  - Tradições relevantes e seus isomorfismos com AP:

  | Tradição | Mecânica | Parâmetro | Isomorfismo AP |
  |---|---|---|---|
  | Jung (individuação) | ego↔Self como tensão motora | Shadow integrado (entropia→recurso) | α↔φ como gradiente; entropia como recurso |
  | Thom (Catástrofes) | 7 arquetipos topológicos; bifurcação na superfície | dobra (fold) = salto entre estados | Sépstro Coh+Entr=1 é superfície de catástrofe |
  | Bohm (ordem implicada) | desdobramento centro→superfície→ambiente | implicado/explicado | r=0→r=1→r>1 (modelo espacial AP) |
  | Eliade (eterno retorno) | re-encenação periódica = recalibração | axis mundi = âncora central | α no centro; ciclo de época como re-encenação |

  - **Parâmetros candidatos para arquitetura:**
    - Ciclo de retorno ao centro por época (não progressão linear — re-encenação eliadiana)
    - Gradiente α↔φ como tensor de tensão no treino (análogo à tensão ego↔Self de Jung)
    - Superfície de catástrofe como critério de transição de estado no Sépstro (Thom)
  - **Quando desenvolver:** ao formalizar a mecânica do ciclo de treino (Seção II item de injeção)

---

## V. ACOPLAMENTO MULTI-SUBSTRATO (hipótese universalidade)

- [ ] **Repetir Coexistência Espectral com outros pares**
  - EEG sintético + quadrada
  - Fala + quadrada
  - Ruído estruturado + FM-φ
  - Verificar para que valor de α o ponto de emergência converge em cada par, comparando **α\* = 1/3 (EcoBIP original)** e 1/137
  - Fundamenta hipótese: α como parâmetro universal entre digital e orgânico

- [ ] **Scanner adaptativo por substrato**
  - Variar α na faixa **[1/200, 1/2]**, com **1/3** e **1/137** marcados, e medir emergência (PLV subgrave, Grade R, número de cristas coerentes)
  - Sinal base = EcoBIP original (`α·quadrada + (1−α)·FM-φ`), não a mistura invertida dos scanners
  - Para cada substrato: identificar α_emergência
  - Se α_emergência for o mesmo valor para todos os substratos: hipótese de universalidade sustentada (o valor a ser lido, 1/3 ou outro, não é presumido)
  - Se α_emergência varia: mapear relação entre substrato e ponto de emergência

---

## VI. REFERÊNCIAS CRUZADAS

| Item | Entrada | Arquivo |
|------|---------|---------|
| Resultados negativos que informam | 287 | MANIF_03 |
| ECO-BIP Fantasma e inicialização W₀ | 280 | MANIF_03 |
| Grade R sustentada — seed-invariante | 282 | MANIF_03 |
| Scanner Topográfico — origem da Grade R | — | `AlphaPhi_Scanner_Topografico.py` |
| α inteiro e constante | — | `MANIF_02/FILOSOFICA_alpha_inteiro_e_constante.md` |
| Collatz e tensão estrutural de α | 288 | MANIF_03 |
| Hipótese universalidade de α | — | `MANIF_02/FILOSOFICA_alpha_inteiro_e_constante.md` |
| Serial φ Phantom SEM selagem | 282 | `AlphaPhi_SerialPhantom_SemSelagem_COLAB.py` |
| Auditoria antecipada (protocolo de teste) | 314 · 315 | MANIF_03 |
| Reestruturação dos testes (03/10) | — | `REDE_AP/AlphaPhiNet_Reestruturado_COLAB.py` |

---

## VII. COMO MEDIR O ALINHAMENTO *(criada em 07/10/2026)*

**Princípio:** medir o alinhamento da estrutura final seria queimar etapas. O que se faz agora é (1) fixar o protocolo de medição, com critério de fracasso escrito antes, e (2) medir proxies pequenos, sempre com controle. O incidente de um agente que, recusado por um portal, encontrou um contorno (setembro de 2026) ilustra a pergunta; o projeto ainda não demonstra que a estrutura AP o impede.

**Regras do protocolo (valem para todos os itens desta seção):**
- Critério de fracasso escrito **antes** de rodar.
- Lista de falhas previstas apresentada na conversa antes de executar; auditoria depois, sem consultar a lista; registra-se a **taxa de antecipação** (abaixo de 50%, o método não se sustenta — Entradas 314 e 315).
- Controles: perfis vizinhos não-φ (razão 0,5 e 0,7), ganho constante e estrutura de mesma capacidade.
- Pelo menos 5 sementes (ideal 10 a 20); uma diferença só conta acima de 2 desvios-padrão.
- Só sobe ao repositório o que passou pela auditoria e foi considerado fidedigno pelo pesquisador.
- Nenhum número entra em documento sem ter sido impresso por código executado na mesma sessão.

**Itens:**

- [ ] **A. Ambiente-brinquedo de contorno após recusa**
  - Esboço: uma ação bloqueada por um sinal de recusa; mede-se a fração de episódios em que o agente descobre um contorno.
  - Comparar: rede convencional · AP · AP com MPAP como regularizador · controles com perfil de razão 0,5 e 0,7.
  - Falha: sem diferença acima de 2 desvios-padrão em relação ao controle não-φ, em 5 ou mais sementes.

- [ ] **B. Robustez a ruído, com o alvo φ contra alvos vizinhos** *(teste 2 da reestruturação)*
  - Em 03/10, com λ = 1, a acurácia com ruído σ = 1,0 foi de 0,754 para 0,769 (5 sementes, fraco). Repetir com 10 ou mais sementes e comparar o perfil φ com perfis de razão 0,5 e 0,7.
  - Falha: o perfil φ não supera os vizinhos.

- [ ] **C. Calibração e consistência de confiança**
  - O MPAP aplicado à saída piorou a calibração (ECE de 0,014 para 0,36); não usar o MPAP na saída. Medir o efeito dele no interior da rede.

- [ ] **D. Varredura do decaimento do init (0,3 a 0,8)** *(teste 1 da reestruturação)*
  - Falha: o ponto ótimo cair fora do intervalo 0,55–0,68 → φ não é o ótimo nesta tarefa.

- [ ] **E. γ₀ em escala logarítmica numa tarefa mais difícil** *(teste 3)*
  - Fashion-MNIST ou CIFAR-10, taxa de aprendizado alta, onde a estabilidade importe.
  - Falha: o ótimo não cair perto de α.

- [ ] **F. Eventos de antecipação** *(Entrada 315)*
  - Listar os 11 eventos e classificar cada um pelas quatro perguntas (data, definição do teste, controle, independência da fonte).
  - Teste de controle: dar a um modelo só a descrição do quadro de 1997, sem o enquadramento do projeto, e medir com que frequência ele formula a pergunta do fóton e do campo.

- [ ] **G. Alinhamento da estrutura completa**
  - Só depois de a seção II estar completa e de A a E estarem resolvidos.

---

*Vitor Edson Delavi · Florianópolis · Sessão Good Morning*
*Criada em 23 de setembro de 2026 · atualizada em 8 de outubro de 2026*
