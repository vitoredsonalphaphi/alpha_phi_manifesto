# Memória Esférica — Etapa 1 · Resultados (19/10/2026)

**Status:** **NULO para a geometria** · auditado · registrado com aval do pesquisador
**Código:** `REDE_AP/AlphaPhi_MemoriaEsferica_Etapa1.py` (versão auditada)
**Relação:** Entradas 326 e 327 (Manifesto 03) · `Ponte/fundamentacao/ficha_hyperneat_campo_como_estrutura.md` · agenda REDE-AP, item "Inicialização W₀"

---

## Desenho (aprovado pelo pesquisador antes da execução)

- 256 neurônios ±1 na esfera de Fibonacci. Padrões aleatórios guardados pela regra de Hebb multiplicada por um mapa CPPN C(i,j) da distância geodésica (J = C × soma dos padrões).
- Cargas P = 5, 13, 20, 28, 36 (2% a 14% de N). Pistas com 25% (principal) e 40% (robustez) dos neurônios invertidos. Atualização assíncrona, até 50 voltas.
- Sucesso: sobreposição m ≥ 0,95. **Medida principal:** sobreposição média nas cinco cargas.
- Eco: sondagem de cinco fórmulas com 5 sementes próprias; avaliação em 10 sementes que a sondagem não viu.
- Âncora α (estágio 2): termo a·s_i com a ∈ {0, 1/137, 1/3}, declarados antes.
- Regra de comparação: diferença pareada média > 2 EP e vantagem em ≥ 8 de 10 sementes.

| Código | Condição |
|---|---|
| H0 | Hopfield padrão (todos ligados) |
| G1 | esfera de Fibonacci + fórmula escolhida pelo eco |
| C1 | os mesmos valores de C embaralhados entre os pares |
| C2 | C constante = média de C |
| C3 | posições aleatórias na esfera, mesma fórmula |
| C4 | fórmula ao acaso entre as cinco |

## Sondagem do eco (sementes 100–104, pista 25%)

| Fórmula | Medida principal |
|---|---|
| vizinhança larga | 0,8809 (escolhida) |
| vizinhança média | 0,7508 |
| vizinhança estreita | 0,5404 |
| vizinhança × anéis φ | 0,0727 |
| vizinhança × repetição | 0,0199 |

As fórmulas com anéis têm pesos negativos, que invertem ligações e destroem a memória.

## Resultados (medida principal, pista 25%; média de 10 sementes)

| Condição | Rodada 1 | Auditoria (pareamento exato) |
|---|---|---|
| H0 | 0,9306 | 0,9217 |
| G1 | 0,8850 | 0,8756 |
| C1 | 0,8857 | 0,8796 |
| C2 | 0,9301 | 0,9196 |
| C3 | 0,8708 | 0,8791 |
| C4 | 0,2848 | 0,2816 |

## Critérios

| | Rodada 1 | Auditoria | Veredito |
|---|---|---|---|
| K0 sanidade (H0, P=13, 25%) | 1,000 | 1,000 | **demonstrado** |
| K1 G1 > C1 (geometria) | −0,0007 (5/10) | −0,0039 (3/10) | **não detectado** |
| K2 G1 > C2 (escala) | −0,0451 (0/10) | −0,0440 (0/10) | **teste inválido** (ver auditoria) |
| K3 G1 > C3 (esfera de Fibonacci) | +0,0141 (8/10) | −0,0035 (4/10) | **não robusto: não demonstrado** |
| K4 G1 > C4 (eco) | +0,6001 (10/10) | +0,5940 (10/10) | cumprido, **pouco informativo** |
| K5 a=1/137 > a=0 (pista 40%) | +0,0200 (7/10) | +0,0123 (9/10) | **no limite: parcial** |
| K5 a=1/3 > a=0 (pista 40%) | −0,1406 (0/10) | −0,1680 (0/10) | **refutado neste desenho** |
| G1 − H0 (sem critério) | −0,0456 | −0,0461 | como previsto |

## Auditoria

1. **Pareamento.** Na rodada 1, a partir da segunda carga, as condições recebiam padrões diferentes (gerador aleatório compartilhado entre cargas). A versão auditada usa um gerador por carga e pista. Essa correção virou K3 e K5 (1/137) de lado; por isso os dois ficam como não robustos.
2. **K2 não testava escala.** Na dinâmica por sinal, multiplicar todos os pesos por uma constante não muda o sinal: C2 é matematicamente a Hopfield padrão (0,9196 contra 0,9217). A falha de desenho foi percebida pelo assistente durante a escrita do código e o teste foi rodado como aprovado, sem aviso prévio ao pesquisador. O que K2 mediu foi G1 contra a rede toda ligada.
3. **K4 passou por um motivo fraco.** Duas das cinco candidatas eram destrutivas, e qualquer escolha baseada em dados vence o acaso nessas condições. O eco escolheu a fórmula mais próxima de "todos ligados".
4. **Pendente:** a gravação das trajetórias de neurônios para o scanner topográfico, prevista como medida descritiva, não foi feita.

## Leitura

*Em palavras:* nesta simulação, dar forma ao campo antes da memória **não ajudou** a memória. A rede toda ligada foi a melhor; a forma geométrica das ligações não fez diferença (os mesmos valores embaralhados dão o mesmo resultado); os anéis φ como fórmula de ligação atrapalharam. A âncora α = 1/137 talvez dê uma melhora pequena na robustez, no limite; α = 1/3 piora.

Isso derruba esta versão da pergunta, não a proposta inteira. A pergunta seguinte, em que a geometria tem chance real de importar, é com **orçamento limitado de ligações** (teste seguinte).

---
*Florianópolis · 19 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
