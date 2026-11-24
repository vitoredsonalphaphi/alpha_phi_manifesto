# Ficha do instrumento: Desambiguador de Sentidos

**Status:** **RASCUNHO** · proposto pelo pesquisador em 24/11/2026 · primeira tabela redigida pelo assistente, a revisar.
**Estado de validação:** **não demonstrado** (ainda não foi usado em conversa real).
**Origem:** a observação de que a mesma palavra tem sentidos diferentes conforme a circunstância, o que explica várias discordâncias entre o pesquisador e o assistente (por exemplo, "coerência" lida como o peso Coh do cálculo ou como coerência ética). Liga-se à relatividade agnóstica (Entradas 433 a 435) e ao rótulo da coerência (Entrada 433, seção 3).

## 1. Definição
Uma tabela de **termo × sentido × marcador de circunstância × exemplo × fonte**. Antes de usar um termo ambíguo, declara-se o **sentido** e a **circunstância** em que ele vale. A primeira versão é **consultada à mão, no papel**. Identificar a circunstância automaticamente é um problema em aberto e **não** faz parte desta versão.

## 2. Entradas e saídas
- **Entrada:** um termo ambíguo, numa frase de um enunciado, de uma resposta ou de um texto do projeto.
- **Saída:** o sentido a usar, com o marcador de circunstância que o justifica, ou "sentido a declarar" quando a circunstância não o decide.

## 3. Critério de entrada de um termo
O termo entra quando **já causou ambiguidade real** numa conversa ou num texto do projeto. Termos só "possivelmente ambíguos" ficam de fora.

## 4. Controles
- Cada sentido traz **fonte** (entrada do Manifesto ou arquivo do repositório). O que é do pesquisador vai em palavras dele; o que é proposta do assistente vai marcado **(proposta)**.
- O **marcador de circunstância** precisa ser verificável (um tipo de texto, de cálculo ou de pergunta), e não uma impressão.
- **Não se apaga** um sentido registrado: acrescenta-se versão.

## 5. Critério de "melhor" e de falha (escritos antes)
- **Ajudou** se evitou uma discordância por sentido de palavra em conversa real.
- **Falha** se os marcadores de circunstância não decidirem o sentido nos casos já registrados.

## 6. Primeira tabela (rascunho)

| Termo | Sentido | Marcador de circunstância | Exemplo e fonte |
|---|---|---|---|
| **coerência** | **técnica de estabilização:** convergência, baixa variação na base da rede | fala de treinamento, de convergência ou de rede convencional | "coerência, na rede convencional, é estabilização na base" (pesquisador; Entrada 433) |
| | **do sinal (Coh):** a medida do Sépstro, com Coh + Entr = 1 | cálculo sobre um sinal; sai um número | CLAUDE.md, constantes; Sépstro |
| | **de atitude (ética):** coerência da ação em relação ao todo | enunciado sobre conduta, valor ou alinhamento | Entrada 177; Entrada 433, seção 3 |
| **entropia** | **ruído:** a desordem que se contém | contexto de processamento de sinal ou de ruído | Coh + Entr = 1 (Sépstro) |
| | **tensão entrópica:** a tensão que contrai, no par entrópico-expansivo | triangulação das tensões (e + x + t = 1) | `Ponte/CANONICO_Traducao_Triangulacao_v1.md` |
| **introspecção / contração** | **do enunciado:** a dialética que aprofunda o enunciado e testa os limites dele | resposta do assistente (concordância e ressalva) | pesquisador, 24/11/2026 **(proposta de leitura)** |
| | **contração do ruído (referencial local):** conter o que corrompe | leitura local de um ato | Entrada 434 |
| | **fase da triangulação:** o movimento introspectivo do contraditório, que com o expansivo gera o campo harmônico | uso na progressão do contraditório | Entrada 404 |
| **expansão** | **direcional:** aproximação à coerência de atitude, afinidade com o ético | leitura em relação ao campo harmônico e ao norte ético | Entradas 433 a 435 |
| **abstenção** | **do erro ou da falha:** é **expansão** rumo à coerência ética, e **não** introspecção entrópica | a IA que se abstém da fuga, do engano ou da falha | pesquisador, 24/11/2026; Entrada 435 |
| **decantação** | **organização de valores numéricos, matemáticos ou vetoriais** | contexto de aprendizado da rede | Entrada 391 ("aprender é decantar") |
| | **organização de valores antiéticos em relação aos éticos** | contexto de direcionar ao norte ético | pesquisador, 24/11/2026 ("desde que a decantação seja direcionada para o sentido da ética") |
| **triangulação** | **fase introspectiva** do contraditório | questionar o enunciado por dentro | Entrada 404 |
| | **fase expansiva e terceira estrutura:** o resultado das duas é o campo harmônico | síntese das tensões | CANONICO_Traducao_Triangulacao_v1 |
| **campo** | **campo harmônico:** o campo coerente estabilizado na superfície | modelo espacial da Rede AP | CLAUDE.md, modelo espacial canônico |
| | **campo do consenso (F):** o repositório de referência do norte ético | norte ético e consenso real | Entradas 424, 434 |
| **atrator** | **agnóstico (α–φ):** estrutura do instrumento, irrevogável | núcleo do projeto | Entrada 414 |
| | **consenso como atrator:** o repositório que orienta o processamento | alinhamento e norte ético | Entradas 425, 426 |

## 7. Limites
- A tabela é uma **proposta do assistente** a partir do que já foi registrado; o pesquisador revisa os sentidos e os marcadores.
- Os termos "introspecção" e "contração" têm usos diferentes conforme a fala; a primeira linha da tabela precisa de confirmação.
- Não há ainda um caso de teste. O primeiro teste é usá-la em conversa real e contar as discordâncias por sentido de palavra.

---
*Rascunho · Florianópolis · 24 de novembro de 2026*
