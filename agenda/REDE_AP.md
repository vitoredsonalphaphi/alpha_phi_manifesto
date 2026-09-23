# Agenda REDE-AP
## Construção da Rede Neural Alpha-Phi — Itens Ativos

**Como referenciar:** "coloca na Agenda REDE-AP" ou "verifica na Agenda REDE-AP"
**Arquivo:** `agenda/REDE_AP.md`
**Última atualização:** 23 de setembro de 2026

---

## I. EXECUÇÕES PENDENTES (código pronto, aguarda Colab)

- [ ] **Scanner de Coexistência Espectral** — S₁ (ruído branco) vs S₂ (serial φ) vs S₃ (mistura)
  - Código fornecido na sessão de 23/09/2026
  - Observar: a estrutura φ do serial sobrevive à mistura com ruído igual? O PLV subgrave persiste?
  - Responde se o serial tem "peso estrutural" mensurável antes do acoplamento completo

- [ ] **Scanner Geométrico Latente — Grade R**
  - Código fornecido na sessão de 23/09/2026
  - 4 painéis: canvas Z(t,log_f) · autocorrelação 2D · mapa de gradiente · histograma polar
  - Observar: picos em Δt≈inter-cone e Δlog_f≈log₂φ na autocorrelação → Grade R confirmada
  - Histograma polar: picos a ~60° → simetria romboédrica

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

- [ ] **Definir métricas no domínio da rede** (não usar β da cascata)
  - Espectro dos pesos ou das ativações
  - Entropia por camada (Shannon)
  - Rank efetivo das matrizes de peso
  - Estabilidade do treino (variação da loss entre épocas)

- [ ] **Mudar mecanismo de injeção**
  - Em vez de soma linear, testar phantom modulando algo estrutural:
    - Inicialização W₀ (Estágio II — Entrada 280)
    - Ganhos por camada
    - Taxa de aprendizado (phantom como schedule adaptativo)

- [ ] **Testar Grade R no espaço onde surgiu**
  - Scanner euclidiano dos pacotes Fibonacci com os mesmos controles
  - Verificar se Grade R aparece no scanner antes de tentar injetá-la na rede

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

- [ ] **α agnóstico por substrato** *(Entrada 288 + agenda/pendentes.md item 6)*
  - Não fixar o valor 1/137 — preservar a **arquitetura da tensão**
  - O operador expansivo e o contrátil, em proporção de emergência para cada substrato
  - Forma prática no código:
    ```python
    ALPHA_EXPANSAO   = 137        # inteiro ímpar — normalização, campo, atrator
    ALPHA_CONTRACAO  = 1/137.036  # decimal — granularidade mínima, introspecção
    # Para outros substratos: calibrar estes valores empiricamente
    ```

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

---

## V. ACOPLAMENTO MULTI-SUBSTRATO (hipótese universalidade)

- [ ] **Repetir Coexistência Espectral com outros pares**
  - EEG sintético + quadrada
  - Fala + quadrada
  - Ruído estruturado + FM-φ
  - Verificar se ponto de emergência converge para α=1/137 em todos os pares
  - Fundamenta hipótese: α como parâmetro universal entre digital e orgânico

- [ ] **Scanner adaptativo por substrato**
  - Variar α na faixa [1/200, 1/100] e medir emergência (PLV subgrave, Grade R)
  - Para cada substrato: identificar α_emergência
  - Se α_emergência ≈ 1/137 para todos: hipótese confirmada
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

---

*Vitor Edson Delavi · Florianópolis · Sessão Good Morning*
*Criada em 23 de setembro de 2026*
