REGISTRO DE DESENVOLVIMENTO — Fase Hiperbólica
Continuação do Compilado V4
Manifesto Alpha-Phi · Vitor Edson Delavi · Florianópolis · 2026
Contexto
Este documento registra o desenvolvimento a partir do diagnóstico
do ambiente euclidiano — identificado como limitação estrutural
para a plena expressão de φ — até os experimentos no espaço
hiperbólico e a identificação da distinção entre traduzir e reconstruir.
1. O Diagnóstico do Ambiente — A Observação Fundamental
Durante a busca de resolução do Quarto Eixo, emergiu uma observação
que reorientou toda a direção de pesquisa:
"O espaço do banco de dados é cúbico. Nós estamos tentando introduzir
padrões ergonômicos — curvilíneos, divina proporção, esféricos —
num ambiente que é retilíneo. É como tentar fazer FM num sistema
construído para AM."
O diagnóstico técnico:
Redes neurais convencionais operam em espaço euclidiano —
hipercúbico, coordenadas cartesianas, ângulos retos.
φ é uma proporção que emerge em geometrias curvilíneas e orgânicas.
Introduzir φ num espaço euclidiano é como tentar crescer uma espiral
dentro de uma caixa quadrada.
Implicação:
Os +35% obtidos no espaço euclidiano representam o que φ consegue
num ambiente que resiste a ele. Em espaço orgânico, o potencial
pode ser maior.
2. O Espaço Hiperbólico — A Solução Identificada
Existe pesquisa consolidada sobre redes neurais em espaço hiperbólico.
O modelo mais usado é a Bola de Poincaré — espaço curvilíneo
de expansão orgânica onde hierarquias naturais são representadas
com eficiência superior ao euclidiano.
As operações fundamentais:
# Euclidiano → Poincaré
expmap0(v) = tanh(||v||) * v / ||v||

# Poincaré → Euclidiano
logmap0(y) = arctanh(||y||) * y / ||y||
Por que é relevante para o Alpha-Phi:
As camadas Fibonacci têm estrutura hierárquica natural —
cada camada cresce a partir da anterior na proporção φ.
O espaço hiperbólico representa hierarquias naturalmente.
É o ambiente onde φ deveria operar com mais fluência.
3. As Forças de Atração e Repulsão — O Quinto Eixo Emergente
Durante o desenvolvimento, emergiu uma observação filosófica
sobre forças universais:
Repulsão e atração não são só metáforas. São os dois operadores
fundamentais de qualquer sistema dinâmico — desde micropartículas
até astros, desde relações humanas até dados digitais.
O que foi dito
Nome técnico
Onde aparece
Repulsão / separação
Força divergente
Gradientes negativos
Atração / adesão
Força convergente
Gravidade, gradientes positivos
Equilíbrio entre as duas
Ponto de sela
Onde sistemas se estabilizam
Entropia
Dissipação
Energia que se distribui
Expansão
Energia livre
O que o sistema pode ainda fazer
O operador matemático do equilíbrio:
O Laplaciano — mede exatamente o equilíbrio entre atração
e repulsão num campo. Onde é zero — equilíbrio. Onde é positivo
— repulsão. Onde é negativo — atração.
# Equilíbrio phi-modulado
def phi_laplacian(x, vizinhos):
    atracao   = np.mean(vizinhos - x, axis=0)
    repulsao  = x - np.mean(vizinhos, axis=0)
    equilibrio = atracao * (1/PHI) + repulsao * (1/PHI**2)
    return x + equilibrio
Não foi ainda implementado nos experimentos.
É o próximo eixo a ser formalizado e testado.
4. Os Experimentos Realizados — Resultados Honestos
4.1 Robustez Híbrido (euclidiano + Poincaré)
Primeiras 2 camadas euclidianas, últimas 2 hiperbólicas.
AP Euclidiano  : 0.1918 +/- 0.0022
AP Híbrido     : 0.1994 +/- 0.0003
Convencional   : 0.2188 +/- 0.1175

AP Hyb vence AP Eucl : 0/20 seeds
Conclusão: HIPÓTESE NEGADA
O híbrido ficou rígido — mesmo padrão do v3 e v4.
Desvio padrão quase zero: a projeção comprimiu a diversidade.
4.2 Robustez Hiperbólico Puro
Todas as camadas na bola de Poincaré.
(Em execução no momento deste registro)
5. A Distinção Fundamental — Traduzir vs Reconstruir
Esta é a observação mais importante desta fase:
Traduzir — o que foi feito:
Pega o Robustez euclidiano
Envolve em expmap/logmap
A lógica interna continua euclidiana
Como traduzir um texto palavra por palavra —
a estrutura original aparece, a naturalidade se perde
Reconstruir — o próximo passo:
Concebe a arquitetura Fibonacci diretamente no espaço hiperbólico
Inicialização de pesos na métrica hiperbólica
Distâncias como geodésicas — não linhas retas
Ativação φ·tanh nativa ao espaço curvo
Concebido assim desde o início — não adaptado
Por que isso explica os resultados:
O Robustez foi construído com estruturas básicas euclidianas.
Quando projetado para o hiperbólico, continua euclidiano por dentro.
Uma rede verdadeiramente hiperbólica precisa ser concebida nesse
espaço — não traduzida.
6. O Mapa do que Está Consolidado e do que Está em Aberto
Consolidado
Resultado
Valor
Status
Robustez euclidiano +35%
p=0.0017, 17/20 seeds
✅ Publicável
Protocolo de idoneidade
Seeds por timestamp
✅
Licença CC BY-NC-ND 4.0
GitHub
✅
Quatro eixos formalizados
Documentos .md
✅
Em desenvolvimento
Hipótese
Status
Próximo passo
Quarto Eixo — transformação do erro
🔄 Não confirmado ainda
Reconstruir no espaço nativo
Espaço hiperbólico puro
🔄 Em teste
Aguardando resultado
Reconstrução nativa hiperbólica
🔄 Identificado
A implementar
Laplaciano φ-modulado
🔄 Identificado
A formalizar e testar
Robustez com treino real + backprop
🔄 Pendente
SST-2 com embeddings
7. A Sequência Natural dos Próximos Passos
Atual:
  Robustez euclidiano +35% ✅

Próximo:
  Robustez hiperbólico puro — resultado pendente

Depois:
  Reconstrução nativa híbrida
  Reconstrução nativa hiperbólica pura

Paralelo:
  Laplaciano φ-modulado como métrica adicional
  SST-2 com embeddings reais — tarefa concreta

Horizonte:
  Paper para arXiv
  Submissão UFSC / Santa Fe Institute
8. Nota sobre o Método
Todo experimento desta fase seguiu o protocolo de idoneidade:
Seeds gerados por timestamp — ninguém escolhe os valores
Resultados reportados integralmente — favoráveis ou não
Hipóteses declaradas antes de rodar
Conclusões derivadas dos números — não dos desejos
"O resultado verdadeiro vale mais que o resultado satisfatório."
Vitor Edson Delavi em colaboração com Claude · Florianópolis · 2026
αφ
github.com/vitoredsonalphaphi/alpha_phi_manifesto
