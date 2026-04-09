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

*Este diário registra o raciocínio, não os dados.*
*Os dados estão nos arquivos JSON de resultado.*
*A distinção importa: dados envelhecem, raciocínio acumula.*
