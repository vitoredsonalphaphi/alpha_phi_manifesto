# O campo que organiza, entre âmbitos — levantamento de analogias
## Insumo para a Ponte `[f ○ c]` (fundamentação)

**Status:** **RASCUNHO** · levantamento de 11/10/2026 · rótulos: *confirmado agora (fonte secundária)* / *de memória* / *hipótese*
**Origem:** pedido do pesquisador (Entrada 320): "quais outros exemplos na literatura, a exemplo de Levin, comentam um campo e uma estrutura organizada pelo campo?"
**Relação:** `../hipoteses/triangulacao_como_esqueleto.md` · `../organograma/`

---

A ideia central que você descreve é esta: **o mesmo papel** (um campo que organiza as partes) aparece em âmbitos diferentes. O que se compara não são os estados, e sim o **papel do campo**. Fiz o levantamento e separo o que **confirmei agora na web** (fontes secundárias, em geral resumos) do que cito **de memória**.

## Exemplos por âmbito

| Âmbito | Autor / obra | O que o "campo" organiza | Verificação |
|---|---|---|---|
| Biologia | Levin e Adams: pré-padrão de voltagem | o campo de voltagem marca onde surgirão olhos e boca e **instrui** a forma | confirmado (revisão de 2023) |
| Biologia | Waddington e Kauffman: paisagem epigenética e redes booleanas | tipos celulares como **atratores** de uma rede; cada vale reúne os estados que convergem para ele | confirmado (revisão; um preprint de 2025 mostra nuances) |
| Física | Haken, sinergética: "princípio da escravização" | poucos **parâmetros de ordem** governam os modos rápidos; o laser como exemplo | confirmado (fonte secundária) |
| Computação | Hopfield (1982), Nobel de Física de 2024 com Hinton | memória associativa: padrões como **atratores** de uma rede inspirada na física de spins | confirmado |
| Computação | Autômatos celulares neurais (Mordvintsev, Randazzo, Niklasson e **Levin**, 2020) | uma regra local compartilhada que **cresce e regenera** um padrão a partir de uma semente | confirmado (secundário; o artigo original não foi aberto) |
| Computação | Computação amorfa (Abelson e outros, MIT, 1999) | comportamento coerente a partir de partes não confiáveis, com metáforas da morfogênese | confirmado; **não** achei ligação documentada com Levin |
| Sociocultural | Lewin (campo, espaço de vida) e Bourdieu (campos sociais) | o ambiente psicológico ou social **estrutura** a conduta; Lewin tomou emprestado da física | confirmado |
| Sociocultural | Morin, pensamento complexo (1990); von Bertalanffy, teoria geral de sistemas | o todo como organizador (circularidade entre parte e todo) | **de memória** |
| Filosofias orientais | Rede de Indra (Huayan); o Tao; Advaita | o um no todo; ordem sem comando | **de memória** |

**Onde a informática citou Levin:** o caso confirmado é o dos autômatos celulares neurais, em que Levin é **coautor**. Outros trabalhos que o citem eu **não verifiquei**.

## O parâmetro comum que aparece

Em quase todos os exemplos há quatro elementos (leitura minha, a revisar):
1. **Um coletivo de unidades locais.**
2. **Uma variável de ordem ou campo**, de poucas dimensões, que emerge do coletivo e **volta a restringir** as unidades (causalidade circular, o centro da proposta de Haken).
3. **Estados estáveis (atratores)** que preservam a forma-alvo.
4. **Recuperação após perturbação** (regeneração em Levin e nos autômatos; canalização em Waddington).

Esses quatro são **candidatos operacionais** a "parâmetro de intercambiabilidade": a variável de ordem (por exemplo, a coerência relativa do coletivo), o número de atratores e a capacidade de recuperação.

## Cuidados

- **Analogia de nome não é analogia de mecanismo.** Ela só conta se o mecanismo for paralelo, e não porque a palavra "campo" aparece nos dois lados.
- **"Fractal"** (a mesma condição em várias escalas) é uma afirmação a **testar**: cada um desses trabalhos descreve a organização numa escala.
- **Seleção de exemplos:** eu procurei exemplos que confirmam a sua ideia. Há nuances (por exemplo, atratores de células maduras podem ser menos estáveis que a trajetória que leva a eles, num preprint de 2025), e a teoria de Levin não é consenso.
- As fontes que li são **resumos de busca**, e não os artigos originais.

## Método para a Ponte: a "ficha de analogia entre âmbitos"

Campos: âmbito · autor/obra · o que é o campo · o que ele organiza · mecanismo · tipo de evidência · status de verificação · correspondência na Ponte (polo, mediador, forma) · **critério de falha** (o que mostraria que a analogia é só verbal). No organograma, seria um nó `tipo: analogia`.

Fontes consultadas:
- [Revisão de 2023 sobre pré-padrões bioelétricos](https://pmc.ncbi.nlm.nih.gov/articles/PMC10277013) e [estudo de 2011 sobre voltagem e pH no ectoderma](https://pmc.ncbi.nlm.nih.gov/articles/PMC3243095)
- [Revisão da paisagem epigenética de atratores](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4407578/)
- [Sinergética (Haken)](https://en.wikipedia.org/wiki/Synergetics_(Haken))
- [Nobel de Física 2024 (Hopfield e Hinton)](https://paw.princeton.edu/article/john-hopfield-wins-nobel-physics-work-artificial-neural-networks)
- [Growing Neural Cellular Automata (Distill, 2020)](https://science.ecosyste.ms/projects/39168)
- [Computação amorfa (MIT)](https://dspace.mit.edu/handle/1721.1/5929)
- [Lewin e Bourdieu: a noção de campo](https://reis.cis.es/index.php/reis/article/view/304)


---

## Modelo de ficha (para o organograma)

```yaml
- id: analogia_<nome>
  tipo: analogia
  ambito: biologia | fisica | computacao | sociocultural | filosofia_oriental
  autor_obra: ""
  o_campo_e: ""            # o que é o "campo" neste âmbito
  organiza: ""             # o que ele organiza
  mecanismo: ""            # como (precisa ser paralelo ao da Ponte, não só o nome)
  evidencia: ""            # teorema | experimento | modelo | observação | metáfora
  verificacao: confirmado_agora | de_memoria
  correspondencia_na_ponte: ""   # polo, mediador ou forma de triangulação
  criterio_de_falha: ""    # o que mostraria que a analogia é só verbal
```

*Rascunho · Florianópolis · 11 de outubro de 2026*
