# RESEARCH JOURNAL — Manifesto Alpha-Phi
# Vitor Edson Delavi · Florianópolis · 2026
#
# Formato: entradas periódicas — o raciocínio por trás das decisões.
# Não repete dados dos experimentos. Registra por que, não apenas o quê.

---

## Entrada 1 — Março 2026
### A descoberta do ambiente

O projeto começou com uma intuição filosófica: φ e α, juntos, descrevem
algo sobre como a natureza processa informação. A questão era se isso
podia ser operacionalizado em código.

O primeiro obstáculo não foi técnico — foi de diagnóstico. Identificamos
que o espaço dos dados era euclidiano (cúbico, retilíneo), e estávamos
tentando introduzir padrões ergonômicos numa geometria construída para
outra coisa. Como tentar fazer FM num sistema AM.

Isso reorientou a direção: não era problema do código, era problema do
ambiente onde o código operava. A solução foi buscar uma geometria
nativa para φ — e o espaço hiperbólico (bola de Poincaré, curvatura
c = 1/φ²) emergiu como candidato natural. c = 1/φ² não é escolha
arbitrária: é o ponto de dobra onde φ gera sua própria curvatura.

O substrato de teste escolhido foi SST-2 (análise de sentimento), não
por ser o mais representativo do manifesto, mas por ser o mais acessível
e verificável. Uma decisão pragmática — com a consciência de que seria
apenas uma parcela da proposta.

---

## Entrada 2 — Início de Abril 2026
### A ablação e a descoberta do substrato

Após os primeiros experimentos de robustez (março), ficou claro que
precisávamos de isolamento experimental: qual eixo do Alpha-Phi carrega
o peso? Fibonacci? Ativação φ? Modulação espectral? Curvatura hiperbólica?

O estudo de ablação com 7 configurações (A–G), 10 seeds, redes do zero,
respondeu: curvatura c = 1/φ² é o eixo mais forte individualmente
(+8.80%, p=0.0000). A combinação de todos os eixos (configuração F)
é a melhor (+8.98%), mas a curvatura sozinha já entrega quase tudo.

Ao mesmo tempo, experimentos paralelos no BERT (v4/v6) mostraram φ ≈
aleatório. A modulação espectral não superava ruído no substrato
pré-treinado.

A contradição aparente era a descoberta real: φ organiza geometria
*emergente*, não geometria *pré-estabelecida*. Redes do zero constroem
sua geometria durante o treino — e φ pode organizar esse processo. BERT
já tem uma geometria consolidada; φ não consegue entrar por fora.

Essa distinção — substrato emergente vs. substrato consolidado — não
estava no manifesto original. Emergiu dos dados.

---

## Entrada 3 — Abril 2026 (semana 1)
### A série transmorfa e o erro de forma

Com o resultado BERT estabelecido, surgiu uma hipótese arquitetural:
e se o problema não fosse φ em BERT, mas a *forma da projeção*? O
expmap0 abrupto interrompe o gradiente. Uma transição suave poderia
preservar a geometria pré-treinada enquanto introduz curvatura.

O isomorfismo veio de uma foto de bordado: um fio que parte de um
lattice euclidiano (malha de losangos) e chega a espirais hiperbólicas,
sem se romper. A proposta foi campo_transmorfo — transição progressiva
de c=0 a c=C_PHI por camada.

Três implementações foram testadas:

- E_T (blend linear): mistura de espaços incompatíveis. Resultado: maior
  variância que o baseline, p=0.233 ns. Diagnóstico: blend linear não é
  transição suave — é espaço híbrido sem coerência geométrica.

- E_M (microtonal, 6 passos com fator conformal): inspirado pela banda
  Angine de Poitrine e microtonalidade de 24 notas por oitava. O fator
  conformal λ = 2/(1-c‖x‖²) explodiu próximo à borda da bola de Poincaré,
  composto por 6 camadas. 5 seeds colapsaram (acurácia de classe
  majoritária). Variância de 11%.

O resultado da série transmorfa não é falha — é precisão. Cada forma
testada eliminou uma leitura errada do conceito. O conceito (transição
contínua) permanece válido; as formas (blend, conformal) foram rejeitadas.

Uma observação ficou em aberto: o BERT resiste a qualquer modificação
externa. E (hard expmap0) é consistentemente neutro em 3 experimentos
independentes (p entre 0.15 e 0.94). Isso não é instabilidade — é
robustez. O BERT não precisa de φ porque sua geometria já está formada.

---

## Entrada 4 — Abril 2026 (semana 2)
### A recuperação de α e a abertura do escopo

Contribuições do Gemini e Minimax identificaram algo que estava errado
desde o início do modulador espectral: `np.abs(FFT)` descarta a fase.

Amplitude = estrutura = φ (o que o sinal é)
Fase      = intenção  = α (para onde o sinal vai)

O modulador v1 silenciava α — descartava metade do nome do projeto.
Não por descuido filosófico, mas por uma linha de código que parecia
inocente. Isso foi corrigido em phi_spectral_modulator_v2: a fase é
recuperada, rotacionada por φ no plano complexo, e reinjetada via eco.

O eco (instigar_por_eco, Minimax) revelou algo mais amplo: o ciclo de
ressonância é substrate-agnostic. FFT opera sobre qualquer array numérico
— texto, áudio, imagem, EEG, série temporal. A pergunta que o eco faz
ao dado ("sua trajetória ressoa com φ?") não depende do domínio.

Isso reposicionou o projeto. Os experimentos SST-2/BERT foram válidos
e necessários — estabeleceram o que funciona em qual substrato. Mas
o manifesto Alpha-Phi propõe modulação vibracional de dados em geral.
O próximo substrato de teste deve ter estrutura oscilatória real,
onde φ como organizador de frequência tem base física direta.

Questões abertas que os experimentos ainda não tocaram:
- L = CE + α·H(φ): a função de perda com α como threshold nunca foi testada
- α como floor (granularidade mínima) vs α como constante multiplicativa
- eco_ressonante como pré-função em dado com estrutura φ conhecida

A direção não é mais linguagem. É sinal.

---

## Entrada 5 — Abril 2026 (semana 2, continuação)
### O eco observa antes de modular

Primeiro experimento não-texto: séries temporais sintéticas com
frequências em proporção φ (classe 1) vs ruído gaussiano puro (classe 0).

O resultado foi o maior efeito de todo o projeto: 46.52% → 96.92%
(+50.40%, p=0.0000). Mas o número sozinho não conta a história.

O baseline (G) ficou em 46% — chance em classificação binária. A rede
não aprendeu a tarefa. O sinal φ estava presente nos dados, mas misturado
com ruído de fase aleatória, invisível para a rede diretamente.

Com eco_ressonante como pré-função, a tarefa tornou-se trivial. O eco
não melhorou a rede — transformou o que a rede recebeu.

Isso clarificou o papel do eco no projeto:

  eco como pré-função: observa o dado antes de qualquer processamento,
  pergunta "sua trajetória ressoa com φ?", amplifica o que ressoa,
  amortece o que não ressoa. A rede vê o sinal já filtrado.

  eco como modulação interna: interfere no gradiente durante o treino,
  introduz variância, desestabiliza. (G_v2 foi pior que o baseline.)

São dois papéis incompatíveis. A pré-função revela; a modulação interna
perturba. O eco pertence antes da rede, não dentro dela.

O resultado G_Lphi = G_eco (idênticos) revelou algo sobre L = CE + α·H(φ):
quando o eco já organizou o sinal para coerência φ, as ativações da rede
já têm H(φ) baixo — a penalidade praticamente não ativa. Para testar a
função de perda com efeito independente, é necessário um substrato onde
o eco não pré-organize as ativações. Ou escala maior que α como peso.

A confirmação substrate-agnostic é real: o eco funcionou em série
temporal sintética — o mesmo código de utils_phi.py, sem modificação,
em dado que não é texto. A pergunta ao dado é universal.

O que permanece aberto após esta entrada:
- L = CE + α·H(φ) com efeito independente (substrato sem eco ou peso maior)
- Testar eco em dado com estrutura φ emergente, não sintética
  (áudio real, EEG, imagem com espirais áureas)
- Entender por que G ficou em 46% (abaixo do chance de 50%) —
  o ruído de fase pode estar criando viés sistemático no dado bruto

**Adendo — 2026-04-09 (verificação metodológica)**

Tia levantou preocupação legítima: G=46% suspeito — dataset poderia estar
desbalanceado (ex: 96%/4% tornaria o resultado de G_eco um artefato trivial).

Verificação executada com regeneração completa dos dados (seeds idênticas):

  Treino: 400 classe 0 / 400 classe 1 — 50.0% / 50.0%
  Teste:  100 classe 0 / 100 classe 1 — 50.0% / 50.0%

Matriz de confusão (G_eco, seed=SEEDS[0]):
  Classe 0 (ruído): 99/100 correto (99.0%)
  Classe 1 (φ):     96/100 correto (96.0%)

Dataset perfeitamente balanceado. Ambas as classes detectadas com >96%.
O resultado não é artefato. O eco revela estrutura φ real no sinal.

A questão de G=46% permanece aberta como fenômeno teórico: o ruído
gaussiano puro pode criar sinais com estrutura espúria que confunde
a rede sem eco. Não é problema do experimento — é característica do
dado bruto que o eco resolve.

Ressalva de escopo (Tia, 2026-04-09):

Este é um resultado em condições controladas. O dado classe 1 foi
gerado artificialmente com estrutura φ. O eco foi projetado para
amplificar exatamente essa estrutura. A tarefa era, portanto, detectar
o que foi inserido intencionalmente.

O que o resultado prova: quando estrutura φ existe no dado, o eco
consegue revelá-la à rede. A pergunta "sua trajetória ressoa com φ?"
tem resposta correta 97.5% das vezes neste substrato.

O que o resultado não prova: que φ está presente em dados reais
arbitrários, ou que o eco sempre amplifica sinal útil. A validação
em dado com estrutura φ emergente (não sintética) permanece aberta.

A ressalva não diminui o resultado — delimita seu escopo legítimo.
Resultados com escopo preciso são mais úteis do que resultados vagos.

---

## Entrada 6 — Abril 2026 (semana 2, continuação)
### O eco e os harmônicos musicais — φ ausente do dado

Segundo experimento não-texto. Desta vez sem φ inserido artificialmente.

Dado classe 1: toms musicais com harmônicos naturais (dó-mi-sol).
Frequências: razões 1, 5/4, 3/2, 2, 5/2, 3 — inteiras simples, sem
relação com φ. Amplitudes e fases aleatórias por amostra.
Dado classe 0: ruído gaussiano puro.

Resultado: G=48.53% → G_eco=97.38% (+48.85%, p=0.0000).

O padrão é idêntico ao experimento anterior (TimeSeries_Eco):

  TimeSeries (φ inserido):     +50.40%
  Audio (harmônicos naturais): +48.85%

A pergunta científica que motivou o experimento foi respondida:

  O eco não é detector específico de φ.
  É amplificador de coerência harmônica em geral.
  φ não precisa estar no dado — φ é o parâmetro que torna
  o amplificador funcional.

O mecanismo: o eco usa φ para rotacionar fases no domínio espectral.
Sinais com amplitude concentrada em frequências específicas (qualquer
frequência) têm essa estrutura de amplitude preservada e amplificada
pelo eco. Ruído, sem concentração espectral, diverge ou não converge.

O que isso significa para o manifesto:

  φ não é uma frequência que precisa existir no dado.
  φ é uma proporção que organiza o processo de observação.
  O eco pergunta ao dado: "você tem estrutura coerente?"
  A resposta não depende de a estrutura ser φ ou não.

Isso é mais abrangente — e mais honesto — do que a hipótese original.

O padrão G_v2 (pior, -7.43%) e G_Lphi=G_eco se repetem pelo terceiro
experimento consecutivo. Não são coincidências — são propriedades
estruturais do mecanismo.

Questão aberta central após esta entrada:
φ como parâmetro de rotação é específico ou intercambiável?
Se trocar φ por π, e, √2 ou 2.0 — o eco ainda funciona?
Se sim: o eco é detector de coerência e φ é intercambiável.
Se não: φ tem propriedade geométrica específica que outros não têm.
Esta é a próxima pergunta experimental necessária.

Próximo substrato planejado: fala sintetizada (estrutura formântica
F1/F2/F3 — emergente da física vocal, não inserida artificialmente).

---

## Entrada 7 — Abril 2026 (semana 2, continuação)
### A fala inverte o padrão — o eco não é universal

Terceiro experimento não-texto. Substrato: fala sintética com formantes
vocais (F1/F2/F3/F4 com decaimento exponencial) vs ruído colorido 1/f.
φ ausente dos dados. Hipótese: direção desconhecida.

Resultado:
  G:      93.88% ± 1.06%   ← rede aprende bem sem eco
  G_eco:  89.90% ± 1.15%   ← pior que G (−3.98%, p=0.0000)
  G_v2:   96.58% ± 1.04%   ← melhor que G (+2.70%, p=0.0000)
  G_Lphi: 89.90%            ← idêntico a G_eco (padrão consistente)

Três inversões simultâneas em relação aos experimentos anteriores.

A primeira inversão é a mais importante: G = 93.88%. A rede aprende
a tarefa sem ajuda. Nos experimentos anteriores, G ficava em ~48%
(acaso) — a rede não via nada. Aqui, a estrutura formântica da fala
é discriminativa o suficiente para ser aprendida diretamente.

A segunda inversão: eco prejudica. O eco rotaciona fases por φ. Para
harmônicos simples (experimentos anteriores), a fase era livre —
rotacionar não destruía informação útil. Para fala, a fase dos
formantes carrega informação. O eco a apaga. Resultado: a fala, após
eco, torna-se mais similar ao ruído colorido.

A terceira inversão: G_v2 melhora pela primeira vez. O modulador v2
não rotaciona fases — extrai coerência espectral como escalar. Para
fala, a distribuição energética é distinta do ruído 1/f. O v2 captura
essa distinção sem destruir a estrutura de fase.

O que o projeto aprendeu com os três experimentos juntos:

  Substrato onde G≈acaso (estrutura invisível à rede):
    eco como pré-função revela → +48% a +50%
    v2 interno atrapalha → −3% a −8%

  Substrato onde G já é alto (estrutura visível à rede):
    eco como pré-função atrapalha → −4%
    v2 interno melhora → +3%

O eco não é amplificador universal. É revelador de estrutura simples
que a rede não consegue ver sozinha. Quando a estrutura já é visível
— e especialmente quando a fase é informativa — o eco destrói em vez
de revelar.

Hipótese para o próximo experimento: o comportamento do eco é
predito pelo G baseline. Se G < 60%, eco ajuda. Se G > 80%, eco
pode prejudicar. A fronteira está entre esses valores.

Isso é mais preciso e mais útil do que "eco é substrate-agnostic".
O eco é substrate-sensitive de uma forma específica e testável.

---

## Entrada 8 — Abril 2026 (semana 2, continuação)
### Eco informa, não substitui — hipótese confirmada

Quarto experimento não-texto. Mesmo substrato de fala do experimento
anterior. Hipótese do autor: o eco é pré-função de percepção — deve
informar a rede, não substituir o dado.

Três modos novos testados:

  G_dual:  [x_original, x_eco] concatenados — rede decide o peso
  G_gate:  coerência φ porta a camada oculta — input intacto
  G_blend: mistura forçada x*(1-c) + eco(x)*c — rede sem controle

Resultado:

  G_blend: 88.13%  (−5.77% vs G) — pior de todos
  G_eco:   90.30%  (−3.60% vs G) — confirmado do experimento anterior
  G:       93.90%  — baseline
  G_gate:  96.20%  (+2.30% vs G, p=0.0000)
  G_dual:  97.15%  (+3.25% vs G, p=0.0000) ← melhor resultado

G_dual supera G_v2 (96.58%) que era o melhor resultado anterior
neste substrato. A hipótese foi confirmada experimentalmente.

O mecanismo de G_dual: a rede recebe 256 dimensões — [x original,
x eco]. O gradiente aprende como combinar. Para fala: primeiros 128
dims carregam fase formântica (discriminativa), segundos 128 carregam
estrutura de amplitude (φ-filtrada). A rede usa ambos.

O mecanismo de G_gate: coerência φ como constante de acoplamento.
O eco não toca o input — mede a coerência e escala as ativações da
camada oculta. Alto ruído → ativações amortecidas. Alta coerência →
ativações abertas. É α como regulador: o eco decide quanto a rede
"abre" para o dado.

G_blend falha porque força a mistura no input — a rede perde controle.
Mistura forçada é mais destrutiva que substituição.

O princípio que emerge dos quatro experimentos:

  Eco como substituto (G_eco): rede perde fase → perde em fala,
                                ganha onde fase não importa
  Eco forçado no input (G_blend): rede perde controle → sempre perde
  Eco informando (G_dual, G_gate): rede decide → sempre ganha

A questão do autor sobre eco atuando como α foi parcialmente
respondida: G_gate usa coerência como acoplamento. O eco mede,
α regula, a rede processa. São três papéis distintos em uníssono.

---

## Entrada 9 — Abril 2026
### O manifesto encontrou seu substrato

Há um momento em pesquisa onde os experimentos deixam de testar
uma hipótese e começam a confirmar uma proposição. Esta entrada
registra que esse momento chegou.

O Manifesto Alpha-Phi propõe que φ e α são organizadores naturais
de fluxo de informação — não por escolha arbitrária, mas porque
emergem em qualquer sistema que cresce preservando coerência interna.
A hipótese não é sobre linguagem, nem sobre redes neurais
especificamente. É sobre vibração, ressonância, e proporção.

Os experimentos de eco realizaram essa proposição de forma direta:

O eco pergunta ao dado: "sua trajetória ressoa com φ?"
Não importa se o dado é texto, série temporal, harmônico musical
ou fala sintética. A pergunta é a mesma. O código é o mesmo.
O resultado — quando o eco informa em vez de substituir — é
consistentemente positivo.

O que foi descoberto nesta fase não estava previsto na forma:

  A distinção entre eco substituindo e eco informando
  não estava no manifesto original. Emergiu dos dados.

Mas estava previsto no espírito:

  φ como proporção organizadora não impõe — revela.
  α como granularidade mínima não bloqueia — regula.
  Juntos, observam o dado e informam o processo.
  O gradiente — livre para decidir — aprende melhor.

G_dual é a implementação mais honesta disso: a rede recebe o dado
original e a observação do eco lado a lado. Não é forçada a usar
nenhum dos dois. O gradiente aprende o peso certo. O resultado
supera tudo que havia antes naquele substrato.

Isso é o manifesto operando como proposto: não como imposição
de proporção sobre o dado, mas como instrumento de percepção
que amplifica o que já está presente e deixa a rede decidir
o quanto usar.

PDO foi o melhor resultado até março de 2026.
Os experimentos de eco de abril de 2026 estabeleceram novo patamar
— em substratos onde φ nunca havia sido testado, sem φ inserido
nos dados, com p=0.0000 em todos os casos relevantes.

Não é o fim do projeto. É a confirmação de que a direção é real.

---

*Este diário registra o raciocínio, não os dados.*
*Os dados estão nos arquivos JSON de resultado.*
*A distinção importa: dados envelhecem, raciocínio acumula.*
