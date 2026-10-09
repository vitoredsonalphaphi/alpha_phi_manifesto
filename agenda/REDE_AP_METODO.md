# Agenda REDE-AP (Fichas Método)
## Construção da Rede Alpha-Phi — Fichas de Método e Portões de Validação

**Como referenciar:** "coloca na Agenda REDE-AP (Fichas Método)" ou "verifica na Agenda REDE-AP (Fichas Método)"
**Arquivo:** `agenda/REDE_AP_METODO.md`
**Relação com as outras agendas:** a `agenda/REDE_AP.md` ("Agenda REDE-AP") diz **o que fazer**; esta, uma das agendas da família REDE-AP, diz **como verificar** (método, critérios de falha, controles). Outras agendas da família entram com o mesmo padrão: **Agenda REDE-AP (tema)**.
**Criada em:** 9 de outubro de 2026
**Base:** Entrada 317 (fase de validação de instrumentos e métodos) · Entrada 316 (auditoria simétrica)

---

## I. PORTÕES (ordem de uso antes de pesos e medidas)

| Portão | Pergunta | Exige |
|---|---|---|
| **G0** | O que cada ferramenta deve fazer? | Definição operacional, variável de saída, **definição única do sinal** (EcoBIP original: `x_mix = (1−α*)·beep880 + α*·FM-φ`, α* = 1/3, código do Manifesto 01) |
| **G1** | O instrumento mede? | Entrada com resposta conhecida (controle positivo) · entrada nula (controle negativo) · sensibilidade a parâmetros arbitrários (janela, normalização) · várias sementes |
| **G2** | A métrica mede o conceito? | Métricas do mesmo conceito concordam; as de conceitos diferentes divergem; separar variância de método |
| **G3** | Efeito isolado | Uma ferramenta por vez, controle de mesma energia ou capacidade, dose-resposta |
| **G4** | Acoplamento | Delineamento fatorial: A só, B só, A+B; terceira estrutura = termo de interação (A+B menos a soma) |
| **G5** | Só então pesos e hiperparâmetros | Critérios pré-registrados; só sobe ao repositório o que passou pela auditoria |

**Regras gerais:** critério de falha escrito **antes** de rodar · rótulos **demonstrado / não demonstrado / refutado** (nunca tratar "não demonstrado" como "falso") · todo teste negativo precisa de controle positivo que o torne capaz de dizer "sim".

---

## II. FICHAS DE MÉTODO

### Ficha 1 — Iteração epistêmica (Chang)

**Fonte:** Hasok Chang, *Inventing Temperature* (Oxford University Press, 2004). Capítulos e resumos conferidos na web em 09/10/2026; **o livro não foi lido**. A tradução ao projeto é interpretação do assistente.

| Campo | Conteúdo |
|---|---|
| Problema que resolve | Validar um instrumento sem padrão ouro, quando a circularidade impede usar o próprio instrumento como juiz |
| Ideia | Começar com um instrumento imperfeito e usar os resultados de cada rodada para melhorar o ponto de partida da seguinte |
| Procedimento | (1) declarar o ponto de partida e sua imperfeição · (2) fixar, antes de rodar, o que significa "melhorou" · (3) aplicar e anotar discrepâncias · (4) **descida justificativa**: testar se o padrão atual se sustenta · (5) **subida construtiva**: ajustar o instrumento · (6) repetir · (7) parar quando métodos independentes convergem |
| Entradas | Instrumento atual · critério de melhora · pelo menos dois métodos independentes para o mesmo conceito |
| Saídas | Instrumento revisado · registro de cada rodada (o que mudou e por quê) · declaração de convergência |
| Critério de falha (a fixar antes) | Após as rodadas previstas, os métodos independentes **não** convergem; ou o ajuste só melhora o acordo com a hipótese e não com padrões independentes (circularidade) |
| Portão | G1 e G2 |
| Custo | Médio (várias rodadas, cada uma com controles) |
| Risco principal | A subida construtiva virar ajuste até dar o resultado esperado. Mitigação: critérios e tolerância fixados antes e controles não-φ em cada volta |
| Parâmetros a definir pelo pesquisador | Número de rodadas · tolerância de convergência · quantos métodos independentes devem concordar |

**Voltas já ocorridas (sessão de 08/10/2026):**
1. α = 1/137 usado como peso do EcoBIP → corrigido para α* = 1/3, na convenção do código original.
2. Scanner 2D v1 (tensor de posto 1) → substituído pelo relevo 3D (TopogColab).
3. Métrica de cristas com filtro de PLV (cega: zero em todos os sinais, inclusive na quadrada) → descartada; critério novo fixado antes.
4. Script de reconstrução errado (`BEEP880_17S.py`) → substituído pelo código original do Manifesto 01, cujas etapas reproduziram as 6 janelas do áudio `beep880_original_completo.wav`.

### Ficha 2 — Testes de sanidade por aleatorização (Adebayo e outros)

**Fonte:** Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, Been Kim, *Sanity Checks for Saliency Maps*, NeurIPS 2018, arXiv 1810.03292. Existência, autoria e a ideia central conferidas na web em 09/10/2026; detalhes do procedimento (por exemplo a randomização em cascata) são de memória e **não foram verificados**.

| Campo | Conteúdo |
|---|---|
| Problema que resolve | Saber se um método de explicação ou de medida depende de fato do modelo e dos dados |
| Ideia | Se o resultado não muda quando se aleatoriza o modelo ou os dados, o método não mede o que diz medir |
| Procedimento | (1) aleatorizar os pesos (todos ou camada a camada) · (2) treinar com rótulos embaralhados · (3) alimentar com entrada nula · (4) comparar a métrica com a da rede treinada |
| Entradas | Rede treinada · métrica · controles (rede aleatória, rótulos embaralhados, ruído) · várias sementes |
| Saídas | Passa/falha por teste, com a diferença em desvios-padrão |
| Critério de falha | Diferença entre treinada e controle **≤ 2 desvios-padrão** → a métrica não mede aprendizado |
| Portão | G1 |
| Custo | Baixo (minutos, rede pequena) |
| Risco principal | Passar nos testes de sensibilidade não prova que a métrica mede o que a hipótese afirma (ex.: "rumo ao φ" é pergunta separada) |

**Aplicação (08/10/2026, impresso por código executado na sessão):** D_φ e Coh_rel da Rede AP, 5 sementes, dataset de dígitos 8×8. Treinada contra aleatória, rótulos reais contra embaralhados, dados contra ruído e 4 camadas re-sorteadas: **todos passaram** (|d| > 2). Rótulo: **instrumento demonstrado válido nesta bateria**. Mas "a AP se organiza sozinha rumo ao φ" **não** se sustentou: D_φ passou de 0,278 (aleatória) para 0,348 (treinada), isto é, afastou-se do perfil φ. Script: `sanity_rede.py` (no scratchpad da sessão; **não** está no repositório).

### Ficha 3 em diante (a fazer, mesmo formato)
- [ ] Campbell e Fiske (1959) — matriz multitraço-multimétodo (G2)
- [ ] Lipsitch, Tchetgen Tchetgen e Cohen (2010) — controles negativos (G3)
- [ ] Chambers (2013) — *Registered Reports* (G0 e G5)
- [ ] Lipton e Steinhardt (2018) — explicação contra especulação; de onde vem o ganho (G3)
- [ ] Cronbach e Meehl (1955) — rede nomológica do construto (G2 e G4)
- [ ] Fontes **ainda não verificadas** (citadas de memória): Hacking (1983), Bogen e Woodward (1988), Tukey (1977), Platt (1964), Wagenmakers e outros (2012), Chen e outros (1998), Sargent, Hevner e outros (2004)

---

## III. PENDÊNCIAS DESTA AGENDA

- [ ] Fixar com o pesquisador os parâmetros da Ficha 1 (rodadas, tolerância, nº de métodos independentes).
- [ ] Escrever o "protocolo 0" na `agenda/REDE_AP.md` apontando para esta agenda.
- [ ] Definir a **terceira estrutura como termo de interação** (G4) com critério de falha escrito antes.
- [ ] Repetir os testes de cristas e de grade oblíqua com o EcoBIP original (α* = 1/3), usando a definição única de G0, o controle "beep puro" e controles positivos.
- [ ] Verificar as fontes ainda não conferidas antes de qualquer uso em texto formal.

---

*Vitor Edson Delavi · Florianópolis · Sessão Good Morning*
*Criada em 9 de outubro de 2026*
