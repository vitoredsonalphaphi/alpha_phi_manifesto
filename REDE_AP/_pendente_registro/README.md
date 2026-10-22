# Pendente de registro — Etapas EX, EC e VL, 21/10/2026

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

## Etapa VL — o violão na rede AP (21/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaVL_Violao.py`, `EtapaVL_sanidade_21out2026.txt`, `EtapaVL_saida_21out2026.txt`, `EtapaVL_resultados.json`, `pagina_violao_na_esfera.html` (cebola, sonificação, resultados). Gravação: `violao_20260513.wav` (do pesquisador). Primeira tarefa com sinal real: guardar a forma harmônica de 12 trechos e recuperar um deles com ruído no próprio áudio.

**Sanidade:** VL00 cumprido (12 memórias; correlação média −0,010, máxima +0,170). VL01 a 0 dB **não** (pista 0,124, abaixo de 0,2): com aval do pesquisador, antes de rodar a rede, o nível principal passou a +10 dB (pista 0,242).

**Resultado (+10 dB):** pista 0,242 → SP 0,258 (limpou +0,016) · **T 0,514 (limpou +0,272)** · EX-G 0,247 · V0 0,282. **VL1 cumprido** (a rede sozinha limpa, +0,016, 9/10); **VL2 cumprido** (+0,256, 10/10); VL3 **não** (−0,011); VL4 **não** (identificação com o previsor 0,642; acaso 0,083).

**Leitura:** com sinal real, o previsor tira ruído de verdade: de 62% para 76% das faixas certas. A rede sozinha limpa muito pouco. A estrutura de êxito firma a forma grosseira (0,183 → 0,406), mas não a conclusão. Reconhecer o trecho certo entre 12 fica em 64%.

**Notas:** a sonificação é a forma tornada audível (ruído moldado pelo perfil de faixas), não o violão reconstruído; o trecho do scanner é o primeiro não silencioso (0,00 s), não escolhido pelo ouvido.

## Etapa VL2 — a Grade Esférica v2 e o violão audível (22/10/2026)

**Arquivos:** `AlphaPhi_GradeEsferica_v2_proposta.py` (proposta, **não canônica**), `AlphaPhi_CampoEsferico_EtapaVL2_ViolaoAudivel.py`, `EtapaVL2_saida_22out2026.txt`, `EtapaVL2_auditoria_22out2026.txt`, `EtapaVL2_resultados.json`. A grade dá endereço a cada dado: 75 anéis de latitude = faixas de frequência; 8 posições em volta = instantes de 0,125 s; altura contínua. A saída é um ganho aplicado ao próprio som.

**Sanidade:** V2-00 cumprido (12 memórias; correlação máxima 0,186); V2-01 cumprido (0,483 a +10 dB).

**Resultado (ganho de nitidez, +10 dB, trechos guardados):** local sem guia −4,80 dB · local com previsor **−4,28 dB** · holográfico −3,58 / −3,86 dB · o teto (com a energia limpa verdadeira) +7,14 dB. A 0 dB: local +0,83 / +0,89 dB; holográfico +2,26 / +2,14 dB. **V2-1 não** (a rede piorou o som); **V2-2 não**; **V2-3 cumprido** (+0,52 dB, 9/10).

**Auditoria:** a dinâmica do canônico (feita para memórias de + e −) empurra 87% dos neurônios aos extremos, enquanto só 8% das alturas do violão estão nos extremos: a rede apaga as alturas e se afasta da forma limpa (correlação 0,23, contra 0,48 da pista). A grade e o filtro funcionam: se a rede devolvesse a memória limpa, o ganho seria de +7,16 dB.

**Decisão do pesquisador (22/10):** parar os testes e estruturar antes o fundamento da grade esférica (Entradas 346–355; agenda REDE-AP).

## Etapa NA — o neurônio que guarda alturas (22/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaNA_NeuronioAlturas.py`, `EtapaNA_auditoria.py`, saídas e resultados. Grade v2 + Ferramentas de Memória v1; 12 memórias contínuas com forma (saturação das memórias 6,8%); pista com ruído (principal: 1,0 × o desvio; altura da pista 0,716); previsor ligado. Sementes 180–189.

**Resultado (altura recuperada, ruído principal):** N0 0,633 (saturação 93,6%) · N1 ganho ajustado (g ≈ 0,53) 0,551 (13,3%) · N2 âncora α 0,630 (0,0%) · N3 0,553 (0,1%). **NA0 cumprido** (o defeito reproduzido); **NA1, NA2, NA3, NA4 não**. O ganho ajustado e a âncora **acabam com a saturação**, mas a altura recuperada **piora**; e **todas as versões terminam abaixo da própria pista** (0,716).

**Auditoria:** o problema não está no neurônio, está nas **ligações**. Uma memória intacta, passada uma única vez pelo campo de ligações (regra de Hebb com vizinhança), já sai com correlação 0,849: as ligações locais deformam as memórias contínuas, e as voltas acumulam a deformação. A memória contínua não é um ponto estável da rede. Descritivo, não previsto: ler o resultado no fim pelo previsor (a mistura das memórias pesada pela ressonância) dá 0,884, acima da pista. **Próximo proposto:** (1) uma regra de ligação em que as memórias contínuas sejam pontos estáveis; (2) a leitura no além pelo campo (Entrada 347), com critérios escritos antes.
