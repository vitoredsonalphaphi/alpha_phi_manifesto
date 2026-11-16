# Estrutura fina: o que a física mostra, e o que isso sugere ao instrumento de tradução

**Status: RASCUNHO · conceitual, sem teste · estágio: refinamento do instrumento de tradução (Entrada 417) · origem: Entradas 416 e 420 (16/11/2026) · revisável (Entrada 414).**

**Como ler:** os números das tabelas foram **impressos por código nesta sessão** (hidrogênio, constantes arredondadas, fórmula de Dirac até a ordem α⁴). Os fatos qualitativos foram **confirmados em busca** (resumos de fontes: notas de curso, arXiv, Wikipedia; textos completos não lidos). O que não tem essa marca está **de memória, a conferir**. As propostas da seção 5 são **leituras do assistente**, grau 1 (analogia), e não resultados.

---

## 1. O que é

Na teoria "grosseira" do átomo de hidrogênio (Bohr, Schrödinger, sem spin), a energia de um nível depende só do número quântico principal **n**; níveis com o mesmo n são iguais (degenerados). A **estrutura fina** é o desdobramento desses níveis em subníveis muito próximos, que depende também de **j** (o momento angular total). Vem de **três contribuições** (confirmado em busca): o acoplamento **spin-órbita**, a correção **relativística** da massa e o termo de **Darwin** (este afeta só os orbitais s). O resultado de Dirac depende de n e j, e a escala relativa à energia grosseira é da ordem de **(Zα)²**. A constante de estrutura fina α ≈ 1/137,036 fixa essa escala.

## 2. Como a física se organiza: uma hierarquia de correções pequenas

Confirmado em busca: as escalas formam uma **hierarquia**, cada uma uma correção pequena da anterior, controlada por uma potência de α (a hiperfina entra também com a razão entre as massas do elétron e do próton):

| Nível | O que é | Escala (busca) | Valor calculado (hidrogênio) |
|---|---|---|---|
| Grosseira | só n; sem spin nem relatividade | ≈ Ry ≈ 13,6 eV | Bohr: n=1 −13,6057 eV; n=2 −3,4014 eV |
| Fina | spin-órbita, relatividade, Darwin | ≈ α² Ry | n=1: deslocamento −1,811 × 10⁻⁴ eV; n=2: j=1/2 −5,660 × 10⁻⁵ eV, j=3/2 −1,132 × 10⁻⁵ eV |
| Lamb | efeito de vácuo (eletrodinâmica quântica) | ≈ α³ Ry (ordem de grandeza; o termo líder é α⁵mc² com logaritmos) | α⁵mc² = 1,057 × 10⁻⁵ eV (sem os logaritmos) |
| Hiperfina | momento magnético do núcleo | ≈ (mₑ/mₚ) α² Ry | (não calculado aqui) |

**Dados impressos por código:** α = 1/137,035999; α² = 5,325 × 10⁻⁵; α⁴ = 2,836 × 10⁻⁹. Para o n = 2, a **separação entre j = 3/2 e j = 1/2 é 4,528 × 10⁻⁵ eV (10,95 GHz)**, igual a α⁴mc²/32; a razão entre essa separação e a energia de Bohr do n = 2 é **α²/4 = 1,331 × 10⁻⁵**. (Os valores de Lamb e de hiperfina reais não foram calculados.)

**A forma, e não os números, é o que interessa:** cada nível **refina** o anterior **sem substituí-lo**; o anterior permanece como base.

## 3. A distribuição dos elétrons (de memória, a conferir)

A distribuição em camadas vem de **regras simples**: os números quânticos (n, l, m, spin) e a **exclusão de Pauli**. A camada n comporta 2n² elétrons (2, 8, 18, 32) e a subcamada l comporta 2(2l+1). Isso é a **estrutura grosseira**. A regra do octeto é uma heurística, com exceções. A estrutura fina é uma correção **sobre** essa distribuição, e não a sua origem (já registrado na Entrada 416).

## 4. Um cuidado sobre "as duas linhas"

O "dois" da estrutura fina vem do **spin 1/2**: para l > 0, j = l ± 1/2, dois valores. Para os orbitais s (l = 0) só existe j = 1/2, **sem desdobramento** por spin-órbita (confirmado em busca: o spin-órbita é nulo para l = 0). Portanto a **dualidade é específica da física do spin**, e não um princípio de "dois" que valha em todo âmbito. **O que se transfere é a forma da organização** (seção 5), e não a contagem.

## 5. O que a forma sugere ao instrumento (hipóteses do assistente, grau 1, sem teste)

- **H1. Hierarquia de correções.** A leitura **grossa** de um texto (a semântica: direção e distância do significado) seria a base, e cada outro canal (térmico, cromático, ritmo) uma **correção pequena e limitada**, que refina sem substituir.
- **H2. A correção fina como desfazer de degenerescência.** Uma leitura grossa deixa várias leituras **degeneradas** (frases com o mesmo peso semântico, por exemplo); uma camada fina as **separa**. É a "vírgula" que o pesquisador descreve (Entrada 420): um termo pequeno que desfaz o empate, sem desfazer a base.
- **H3. Valor e tolerância.** Cada camada leva, escritos antes, **um peso pequeno e uma margem** (resultado específico com adendo de flexibilidade).
- **H4. Crescer sem multiplicar regras.** Um **único princípio de organização** (a hierarquia de correções) para todo o fichário, mesmo que as fichas se multipliquem (requisito da Entrada 420).

**Falhas previstas e cuidados:**
- A analogia é **de forma**; o "dois" é do spin (seção 4).
- **Numerologia:** pesos em potências de φ, de α ou de qualquer número, sem argumento, são escolha, e não estrutura.
- A hierarquia só vale se cada camada for **de fato menor e independente** da anterior. **Critério de falha:** se a camada fina **muda a leitura grossa** em vez de refiná-la, ela não é correção, é substituição.
- **Medida que falta:** uma leitura grossa mensurável de um texto e as suas camadas. Hoje só a triangulação tem procedimento (Entrada 406).

## 6. Próximos passos propostos

1. **Conferir as fontes lendo** (Sommerfeld, 1916; Dirac, 1928; uma referência de física atômica), e não por resumos de busca.
2. **No papel**, escrever a leitura grossa e a primeira correção de **uma frase** do texto, antes de qualquer código.
3. Só então, critérios no formato da Entrada 409.

## 7. Limites

Rascunho conceitual. Os números calculados são do hidrogênio, com constantes arredondadas e a fórmula de Dirac até α⁴; não substituem uma referência. A relação com o instrumento é **analogia de forma**, sem teste.

---
*Florianópolis · 16 de novembro de 2026 · Sessão Good Morning*
*Vitor Edson Delavi · Claude*
