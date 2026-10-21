# Pendente de registro — Etapas EX e EC (a estrutura de êxito), 21/10/2026

**Status:** executada, **não registrada**. Esta pasta existe para que código e números não se percam.

| Arquivo | Conteúdo |
|---|---|
| `AlphaPhi_CampoEsferico_EtapaEX_Estrutura.py` | o teste (importa as Ferramentas de Memória v1 do canônico e a geração de memórias da Etapa K afinada) |
| `EtapaEX_saida_21out2026.txt` | saída impressa |
| `EtapaEX_resultados.json` | números completos |
| `pagina_estrutura_de_exito.html` | scanner topográfico (relevos estáticos) |

**A pergunta (proposta do pesquisador):** uma parcela da memória que contém a forma do todo, instalada no campo como alvo, orienta os neurônios a recompor a memória, inclusive o detalhe que ela não contém? Parcela: graus 0–4 (25 de 225 coeficientes, 11,1%). Sementes 130–139.

**Resultado (nível 0,0, pista 40%):**

| Versão | Conclusão | Forma grosseira | Forma fina | Identificação |
|---|---|---|---|---|
| SP sem previsor | 0,210 | 0,189 | 0,214 | 0,767 |
| T memória inteira | 0,427 | 0,404 | 0,450 | 0,917 |
| EX-G estrutura de êxito | 0,259 | **0,448** | 0,224 | 0,667 |
| EX-H holográfica | 0,236 | 0,228 | 0,241 | 0,775 |
| EX-acaso | 0,185 | 0,165 | 0,192 | 0,700 |

**Critérios:** EX0 cumprido (+0,216, 10/10); EX1 cumprido (+0,049, 10/10); EX2 **não** (a parcela traz 0,225 do ganho; pedia 0,5); EX3 **não** (forma fina +0,010, 6/10); EX4 cumprido (+0,026, 9/10); EX5 cumprido (+0,074, 10/10).

**Leitura:** a estrutura de êxito recompõe muito bem o que ela contém — a forma grosseira sobe de 0,189 para 0,448, acima até do previsor inteiro (0,404); com pista de 25%, de 0,437 para 0,764 — mas não leva os neurônios a completar o detalhe: a forma fina quase não muda (com pista de 25%, cai de 0,468 para 0,429). A parte contém o todo, mas só na resolução dela. A identificação com a estrutura de êxito fica abaixo da rede sem previsor (0,667 contra 0,767): guiar só pelo grosseiro às vezes leva à memória vizinha.

**Auditoria em aberto:** (1) uma só parcela testada (graus 0–4); (2) a força da guia foi a mesma do previsor (0,5), sem varredura; (3) as versões contínuas continuam chegando ao limite de 50 voltas.

## Etapa EC — a estrutura de êxito em camadas (21/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaEC_Camadas.py`, `EtapaEC_saida_21out2026.txt`, `EtapaEC_resultados.json`, `pagina_cebola_da_esfera.html` (a esfera cortada como cebola: camadas por fase, transparência, tensão dos neurônios, topográfico das camadas). Sementes 140–149.

**Resultado (nível 0,0, pista 40%; conclusão):** SP 0,237 · T 0,444 · EX-G 0,277 · EC-3 (do centro para fora) 0,339 · EC-2 0,316 · **EC-inverso (de fora para dentro) 0,414**.

**Critérios:** EC0 cumprido (+0,207); EC1 **não** (EC-3 − T = −0,105, 0/10); EC2 **não**, no sentido oposto (EC-3 − EC-inverso = −0,076, 0/10: o inverso é melhor); EC3 **não** (0,381 do ganho); EC4 cumprido (a segunda camada traz detalhe: +0,060 no fino, 10/10).

**Leitura:** a previsão "do centro para fora" falhou. O campo da rede já sustenta sozinho a forma grosseira (o fundo e a vizinhança favorecem o suave); o que falta é o detalhe fino, onde a pista ruidosa estraga mais. A guia ajuda mais quando dá à rede o que ela não tem: guia e campo se complementam.

**Notas declaradas:** a camada 0–14 é a projeção da memória nos 225 harmônicos, não o padrão ±1 do previsor T; as versões em camadas não podem parar antes da volta 21 (todas as camadas precisam ter entrado).
