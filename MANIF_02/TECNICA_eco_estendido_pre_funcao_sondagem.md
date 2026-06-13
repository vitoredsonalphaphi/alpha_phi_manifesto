# Eco-Ressonante Estendido — Pré-Função de Sondagem de Fase

**Manifesto AlphaPhi · Vitor Edson Delavi**
**Construído em: junho/2026 · SST-2 como substrato-alvo**

---

## Origem da Extensão

O eco-ressonante já possui uma pré-função: antes de agir, ele observa o
substrato, mede H_alpha da entrada bruta, e adapta a intensidade do eco
de acordo com o que encontra. Essa capacidade de leitura antes da ação
é o que distingue o eco-ressonante de uma semente passiva.

O limite da versão anterior: a pré-função lia apenas a **Fase 0** —
o substrato na entrada, antes de qualquer transformação pela rede.
Não sabia o que acontecia com o dado ao longo das camadas internas.

O problema que isso criava:

> O substrato de caracteres (janelas de frequência de letras) mostrou
> H_alpha = 0.478 — coerência adequada para α residir. Mas a acurácia
> ficou em 50.9% (classificação aleatória). O eco formou campo (10/10),
> mas o sinal semântico não estava na entrada. Estava em outra profundidade
> — ou não estava em lugar nenhum.

A pré-função anterior não distinguia entre essas duas situações.
A extensão resolve isso.

---

## A Mecânica

A pré-função estendida executa **N_SONDA ciclos de sondagem** antes de
fixar o modo de ação. Em cada ciclo, propaga um batch pela rede e mede,
em cada fase intermediária, dois valores:

**coh_alpha_f** — coerência de α na fase f:

```
h_norm = |ativações| / sum(|ativações|)   # distribuição de energia
H_f    = -sum(h_norm × log(h_norm))       # entropia das ativações
H_alpha_f = H_f / max(log(n_units), log(137))
coh_f  = 1 - H_alpha_f
```

**disc_f** — discriminabilidade entre classes na fase f:

```
disc_f = mean((mu_positivo - mu_negativo)²)
```

**score_f** — critério de residência de α:

```
score_f = coh_f × disc_f
```

A fase ótima é aquela onde α pode **simultaneamente** se estabilizar
(alta coerência) e **distinguir** (alta discriminabilidade). O score
captura as duas condições de uma vez.

---

## Justificativa do Critério

O critério `coh × disc` não é uma escolha arbitrária.
É o corolário direto do que foi estabelecido sobre α:

- **coh_alpha** mede onde α encontra estrutura suficiente para residir —
  onde a distribuição de energia das ativações não está saturada nem
  colapsada. α precisa de espaço de expressão.

- **disc** mede onde o substrato oferece informação semântica real —
  onde as representações internas da rede efetivamente separam as classes.
  Se disc ≈ 0 em todas as fases, o substrato não contém o sinal buscado.

Separados, cada um é insuficiente:
- Alta coh com disc ≈ 0 → α reside, mas não há o que fazer
- Alta disc com coh ≈ 0 → sinal existe, mas α não consegue medir

Juntos, eles identificam onde α age com eficiência.

---

## Diagnóstico de Substrato

Se `max(score_f) < limiar` em todas as fases, a pré-função emite
diagnóstico:

> *α não encontra residência — sinal semântico ausente em todas as fases.
> Substrato incompatível com a tarefa.*

Isso não é falha experimental. É informação.

O substrato de caracteres (SST-2 foton) é esperado que dispare este
diagnóstico: a distribuição de letras a-z não carrega sinal de sentimento.
"Fantástico" e "terrível" têm distribuições de letras semelhantes. Nenhuma
fase da rede pode separar o que o substrato não distingue.

O diagnóstico correto é mais valioso que um resultado falso positivo.

---

## As Cinco Fases da Arquitetura

```
Fase 0  (substrato)   : texto → 5 janelas × 27 chars = 135 dims
                        (entrada bruta — rotação do fóton)

Fase 1  (compressão)  : 135 → 55
                        Fibonacci: primeira bifurcação
                        (o dado começa a se reorganizar)

Fase 2  (expansão)    : 55 → 89
                        Fibonacci: segunda bifurcação
                        (recomposição em espaço maior)

Fase 3  (nativo α)    : 89 → 137
                        1/α = 137 — espaço nativo de α
                        (régua de entropia em dimensão natural)

Fase 4  (classificação): 137 → 1
                        decisão final
```

A pré-função sonda as Fases 1, 2, 3. Fase 0 é o substrato (já medido
pela semente original). Fase 4 é a saída — só contém a decisão, não
a representação.

A dimensão 137 da Fase 3 não é arbitrária: é o denominador de α = 1/137.
O eco-ressonante original assumia que α residia naturalmente na Fase 3.
A pré-função estendida verifica se essa suposição é correta — e
eventualmente descobre que pode não ser para todos os substratos.

---

## Por que SST-2 como Substrato-Alvo

SST-2 (Stanford Sentiment Treebank 2) é a escolha por três razões:

1. **Sinal semântico binário e bem definido** — positivo/negativo, sem
   ambiguidade de rótulo. Se α acopla com o sinal, o resultado é claro.

2. **Substrato testado em dois modos opostos:**
   - Caracteres (SST-2 foton) → substrato sem sinal semântico → diagnóstico esperado
   - Embeddings semânticos → substrato com sinal → fase ótima a identificar

   A comparação entre os dois modos valida a pré-função: ela deve
   diagnosticar o primeiro como inadequado e identificar fase no segundo.

3. **Benchmark estabelecido** — permite comparação com resultados
   externos. BERT-base atinge ~93% em SST-2. Nossa rede FC simples não
   compete com isso. O que testamos é se α, operando via pré-função,
   melhora a eficiência de convergência dentro de nossa arquitetura.

---

## Conexão com o Eco-Ressonante Original

O eco-ressonante original:
```
observa substrato (Fase 0) → mede H_alpha → calcula beta → age
```

O eco estendido:
```
sonda cada fase (N_SONDA ciclos) → mede score_f por fase
→ identifica fase_otima → reside ali → age a partir dali
```

A estrutura de "observar antes de agir" é a mesma.
O que muda: a profundidade da observação.

O eco original era um instrumento com uma frequência de medição.
O eco estendido é um instrumento com N_FASES frequências simultâneas —
e escolhe a mais ressonante.

---

## Arquivo de Implementação

`AlphaPhi_ECO_PreFase_COLAB.py`

Configs:
- **AA** — baseline (sem semente)
- **BB** — semente fixa na Fase 3 (método anterior)
- **CC** — pré-função adaptativa (fase_otima por sondagem)

Métrica principal: acurácia SST-2 + perfil de scores por fase + diagnóstico.

---

*Florianópolis · junho de 2026*
*Conecta: FILOSOFICA_alpha_atrator_entropia_de_si_mesmo.md*
*Conecta: AlphaPhi_ECO_Foton_COLAB.py · AlphaPhi_SST2_Semente137_COLAB.py*
