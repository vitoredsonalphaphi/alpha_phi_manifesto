# Agenda REDE-AP
## Construção da Rede Neural Alpha-Phi — Itens Ativos

**Como referenciar:** "coloca na Agenda REDE-AP" ou "verifica na Agenda REDE-AP"
**Arquivo:** `agenda/REDE_AP.md`
**Última atualização:** 8 de outubro de 2026 (criada em 23/09/2026)

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
    - Ganhos por camada — *testado na ablação do φ-init: o efeito é de escala; 0,5^k empatou com 1/φ^k*
    - Taxa de aprendizado (phantom como schedule adaptativo) — *pendente*

- [ ] **Testar Grade R no espaço onde surgiu**
  - Scanner euclidiano dos pacotes Fibonacci com os mesmos controles
  - Verificar se Grade R aparece no scanner antes de tentar injetá-la na rede
  - **Acrescentado em 08/10:** no scanner topográfico **3D** (TopogColab, modos ECO e LAP), com o **EcoBIP original em α = 1/3**: (i) contar as cristas coerentes por etapa da cascata, EcoBIP contra controle (cascata só na quadrada); (ii) medir a orientação da textura do LAP sem ângulo-alvo fixo. Critérios escritos antes, nos dois sentidos, com controle positivo (Entrada 316). Verificar também se as linhas pretas dos gráficos 3D são projeções do cursor (hover) do Plotly.

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
