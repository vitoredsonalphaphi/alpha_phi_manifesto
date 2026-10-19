# Campo Esférico — Orçamento, Etapa 0, Etapa T e Etapa C · Resultados (19/10/2026)

**Status:** auditado · registrado com aval do pesquisador
**Antecedente:** `AlphaPhi_MemoriaEsferica_Etapa1_RESULTADOS_19out2026.md` (memória esférica, etapa 1: nulo para a geometria)
**Entradas do Manifesto 03:** 328 a 332 (os enunciados que antecederam cada teste)
**Regra comum:** critérios escritos na conversa antes de cada execução; comparação pareada = diferença média > 2 erros-padrão e vantagem em ≥ 80% das sementes ou repetições; todos os números abaixo foram impressos por código executado na sessão de 19/10/2026.

## Síntese

| Teste | Pergunta | Resultado |
|---|---|---|
| Orçamento de ligações | com o mesmo número de ligações, a geometria escolhe ligações melhores? | **nulo para φ**; observação de economia de fio |
| Etapa 0 — cartografia | medir o campo antes prevê onde a memória funciona? | **não demonstrado** (Z4); observação: a geometria ajuda quando o ambiente tem forma, mas não é φ |
| **Etapa T — teoria antes da simulação** | a forma do campo e a memória podem ser previstas antes? | **campo vazio: previsto (T1)**; memória sem forma: prevista (T2); memória com forma: **não** prevista |
| **Etapa C — cultivo** | o campo cultivado em etapas segue a forma prevista? | **sim, em todas as etapas (C1, C2)**; o cultivo deixa história; não aprofunda o atrator (C3) nem preserva na mudança de forma (C4) |

---

## 1. Orçamento de ligações (`AlphaPhi_MemoriaEsferica_Orcamento.py`)

Mesmo desenho da etapa 1, ligações binárias, **2.464 ligações** (7,5% dos pares, grau médio 19,2) em todas as condições.

| Condição | Medida principal (pista 25%) | Fio total (rad) |
|---|---|---|
| H0 todos ligados | 0,9217 | — |
| B-A ao acaso | 0,5124 | 3.895 |
| B-φ saltos de Fibonacci | 0,4707 | 1.729 |
| B-φemb mesmos saltos, ordem embaralhada | 0,4673 | 3.876 |
| B-PM pequeno mundo (10% religado) | 0,4429 | 1.281 |
| B-G só vizinhos | 0,4146 | 969 |

| Critério | Resultado | Veredito |
|---|---|---|
| L0 sanidade | 1,000 | cumprido |
| L1 B-G > B-A | −0,0978 (0/10) | não cumprido |
| L2 B-φ > B-A | −0,0418 (0/10) | não cumprido |
| L3 B-φ > B-PM | +0,0278 (9/10) | cumprido, **confundido**: B-φ usa 35% mais fio |
| L4 B-φ > B-φemb | +0,0034 (5/10) | não cumprido |
| L5 ≥ 95% da memória de B-A com ≤ 60% do fio | 0,919 da memória com 0,444 do fio | não cumprido |

**Observação (sem critério prévio):** B-φ lembra o mesmo que o seu embaralhado com 45% do fio. **Limitação:** padrões aleatórios não têm forma espacial, então nenhuma geometria podia ajudar por construção; e as cargas a partir de 13 excedem a capacidade de redes com ~19 ligações por neurônio.

## 2. Etapa 0 — cartografia do campo (`AlphaPhi_CampoEsferico_Etapa0_Cartografia.py`)

| Campo | R com forma | R sem forma | Atratores do campo vazio (de 200) | Memória |
|---|---|---|---|---|
| F1 Fibonacci | 0,4027 | 0,0625 | 200,0 | 0,7300 |
| F2 aleatório | 0,4146 | 0,0620 | 190,6 | 0,7599 |
| F3 grade lat-long | 0,4222 | 0,0638 | 2,0 | 0,6699 |
| F4 F1 embaralhado | 0,0570 | 0,0635 | 200,0 | 0,6681 |
| H0 todos ligados | — | — | — | 0,7457 |

| Critério | Resultado | Veredito |
|---|---|---|
| Z0 R sem forma ∈ [4,25%; 8,25%] | 6,20% a 6,38% | cumprido |
| Z1 R F1 > F4 (controle positivo) | +0,3457 (10/10) | cumprido |
| Z2 R F1 > F2 | −0,0119 (0/10) | não cumprido |
| Z3 R F1 > F3 | −0,0196 (0/10) | não cumprido |
| Z4 memória no campo de maior R > menor R | F3 − F4: +0,0018 (6/10) | não cumprido |

**Auditoria:** Z4 mediu a ressonância com a ligação do campo vazio e construiu a memória com outra ligação; o campo medido não era o campo construído (falha de desenho do assistente). **Observações (sem critério prévio):** memória F1 − F4 = +0,0619 (9/10): com ambiente com forma, a geometria passa a ajudar (na etapa 1, com padrões aleatórios, a mesma comparação deu zero). Mas F2 − F1 = +0,0299 (8/10): não é φ.

## 3. Etapa T — a teoria antes da simulação (`AlphaPhi_CampoEsferico_EtapaT_Previsor.py`)

**Fontes conferidas (resumos):** Turing (1952), *Phil. Trans. R. Soc. B* 237; Amari (1977), *Biol. Cybern.* 27, doi 10.1007/BF00337259; Amit, Gutfreund e Sompolinsky (1985), *Phys. Rev. Lett.* 55, 1530 (capacidade ≳ 0,14); Derrida, Gardner e Zippelius (1987), *Europhys. Lett.* O teorema de Funk–Hecke (formas naturais de um núcleo isotrópico na esfera = harmônicos esféricos) é citado de memória.

**T1 — campo vazio.** Previsão (Funk–Hecke), força por grau L = 0..8: −1,605 −0,401 0,305 **0,336** 0,226 0,131 0,068 0,032 0,013 → **L\* = 3** (grau 2 quase empatado). Medida em 1.000 atratores:

| Campo | dentro de ±1 de L\* | contagem do grau dominante (0..9) |
|---|---|---|
| **F1 Fibonacci** | **1,000** (94,3% exatamente em L=3) | 0, 0, 57, 943, 0, … |
| F2 aleatório | 0,581 — **inválido** (ver auditoria) | 0, 0, 81, 398, 102, 41, 57, 114, 107, 48 |
| F3 grade | 1,000 — com reserva | 0, 0, 548, 452, 0, … |

**T1: cumprido.** Descritivo: razões inibição/reforço φ, 2 e 3 dão o mesmo L\* = 3.

**Auditoria do medidor** (padrões de grau conhecido 2, 3, 4): F1 acerta 100% (condição 3,5); F2 acerta 18% a 40% (condição 3,7 × 10²) — o resultado de F2 é defeito do medidor, não da teoria; F3 acerta 100%, mas a matriz é quase singular (condição 4,7 × 10¹⁶). **A esfera de Fibonacci é a única disposição em que o campo é previsível e mensurável.**

**T2 — memória.**

| Ambiente | T2a previsão × estabilidade medida | T2b previsão × recuperação |
|---|---|---|
| sem forma (45 combinações) | Spearman 0,981; erro médio 0,006 — **cumprido** | Spearman 0,963 — **cumprido** |
| com forma (36 combinações) | Spearman 0,236; erro médio 0,030 — não cumprido | Spearman 0,379 — não cumprido |

Com forma, a previsão gaussiana satura em ~0,99 e perde a ordem (falha prevista antes).

## 4. Etapa C — cultivo do campo (`AlphaPhi_CampoEsferico_EtapaC_Cultivo.py`)

Pontos **encaixados** (ângulo áureo + altura frac(i·√2)): os pontos de cada etapa contêm os da anterior. Etapas 34, 55, 89, 144, 233; escala da ligação acompanha o espaçamento. Previsão por etapa (faixa = graus com força ≥ 90% da máxima): **1, 1, 2, 2, {2, 3}** — do grosseiro ao fino. 50 repetições.

| Etapa | Faixa | Cultivado: grau dominante | Construído de uma vez | Preservação (cultivado) |
|---|---|---|---|---|
| 34 | 1 | 1 em 50/50 | 1 em 50/50 | — |
| 55 | 1 | 1 em 50/50 | 1 em 47/50 | 0,917 |
| 89 | 2 | 2 em 50/50 | 2 em 50/50 | 0,710 |
| 144 | 2 | 2 em 50/50 | 2 em 50/50 | 0,848 |
| 233 | 2, 3 | **2** em 49/50 | **3** em 46/50 | 0,899 |

| Critério | Resultado | Veredito |
|---|---|---|
| C0 condição do medidor ≤ 10 | 1,6 a 4,5 | cumprido |
| **C1 cultivado na faixa em todas as etapas** | 100% | **cumprido** |
| **C2 construído na faixa em todas as etapas** | 94% a 100% | **cumprido** |
| C3 energia final cultivado < construído | −2,2593 contra −2,2985; mais baixo em 20/50 | não cumprido |
| C4 preservação ≥ 0,8 em todas as passagens | 0,917 / **0,710** / 0,848 / 0,899 | não cumprido (falha na passagem de forma 1 → 2, prevista antes) |

**Leitura.** (1) O previsor funciona como **atrator do desenvolvimento**: o campo cresce do grosseiro para o fino seguindo, em cada etapa, a forma calculada antes de existir. (2) **O cultivo deixa história:** na etapa final, onde a previsão admitia os graus 2 e 3, o campo construído de uma vez foi ao grau 3 (o mais forte) e o cultivado permaneceu no grau 2 (a forma que já tinha), preservando 90% dos pontos, ao custo de um atrator um pouco menos fundo — observação descritiva, sem critério prévio. (3) Quando a forma prevista muda de natureza, o campo se reorganiza (71% preservado).

## Exploração sem registro como resultado

Comparação de disposições encaixadas (condição do medidor e uniformidade, N = 34 a 233): Fibonacci padrão (não encaixa) 1,1 / 0,91–0,93; ângulo áureo + altura √2 (encaixa) 1,6–4,5 / 0,57–0,93; ângulo áureo + van der Corput (encaixa) 1,6–5,1 / 0,27–0,75. Uma primeira tentativa com 1/φ e 1/φ² nas duas coordenadas foi **inválida**: os dois números são dependentes (1/φ² = 1 − 1/φ) e todos os pontos caíram numa espiral.

## Próximo passo

**Previsor envelopado** (estágio 2): separar cada padrão com forma em envelope (harmônicos de grau baixo, previstos modo a modo) e conteúdo fino (quase aleatório, onde T2 já funciona), para prever a memória com forma antes de as ferramentas entrarem no campo cultivado.

---
*Florianópolis · 19 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
