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

## Etapa LA — as ligações para alturas e a leitura no além (22/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaLA_LigacoesAlturas.py`, `EtapaLA_auditoria.py`, saídas e resultados. Mesma base da Etapa NA; sementes 190–199. Regra delta local: ~54 vizinhos por neurônio.

**Resultado (ruído principal; altura da pista 0,718):** H-estado 0,540 · H-campo 0,864 · **D-estado 0,978** · **D-campo 0,999** · pista-campo (controle) 0,988 · PI-local 0,613. **LA0 cumprido** (memória intacta pelas ligações delta: 1,0000; Hebb 0,848); **LA1 cumprido** (+0,260, 10/10: a rede limpa); **LA2 cumprido** (+0,438); **LA3 cumprido** (+0,324: ler pelo campo é melhor); **LA4 cumprido, por pouco** (+0,011, 10/10; a previsão do assistente era que falharia).

**Auditoria (descritiva):** sem o previsor, a rede com a regra delta **desaba** (altura 0,128; identificação 0,342). A limpeza vem do **encontro** das duas coisas: as ligações delta deixam de deformar a memória, e o previsor reconhece e guia. As ligações sozinhas tornam a memória um ponto **estável**, mas não um **atrator** (a rede não volta a ela por conta própria). E ler direto da pista pelo campo já dá 0,988: com memórias distintas, o reconhecimento faz quase tudo.

**Próximo proposto:** (1) testar em trechos que a rede **nunca viu** (onde só reconhecer não basta); (2) o violão sobre esta base; (3) entender por que as ligações delta não atraem sem o previsor.

## Etapa VL3 — o violão sobre a base nova (22/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaVL3_ViolaoBaseNova.py`, `EtapaVL3_auditoria.py`, saídas e resultados. Grade v2 (endereço por anéis), memória com alturas, regra delta, neurônio N3, previsor, leitura pelo estado e pelo campo; sementes 200–209.

**Sanidade:** 12 memórias (correlação máxima 0,186); pista a +10 dB com correlação 0,482. Cumprida.

**Resultado (ganho de nitidez):** +10 dB, guardados: estado −4,86 dB · campo −4,84 dB · teto +7,14 dB. 0 dB, guardados: estado +2,91 · campo **+3,32** · teto +8,79. Trechos novos: +10 dB, −8,0; 0 dB, +1,4 a +1,6. **VL3-1, VL3-2, VL3-3 não cumpridos.** A altura recuperada subiu (de 0,48 para 0,75), mas não o bastante para o filtro.

**Auditoria (semente 200):** o previsor reconhece o trecho certo em 75% (na pista, 83%); se a leitura devolvesse a memória certa exata, o ganho seria +7,12 dB; a memória que a rede reconheceu, exata, daria +3,28 dB. Mesmo nos trechos reconhecidos o ganho é negativo: a leitura pelo campo é uma **mistura macia** de várias memórias (β = 8 é fraco para memórias tão pouco separadas na pista do violão), e a mistura borra as alturas. Nos trechos novos, a altura fica em 0,11: a rede aplica a forma de um trecho guardado (sem generalização), como previsto.

**Próximo proposto:** a força do reconhecimento (β) calibrada à separação das memórias, declarada antes; e a questão de fundo da Entrada 347: no além, o meio pede **um** som — a leitura como resultante macia borra; a leitura que decide, só no além, pode ser necessária.

## Etapa VL4 — a decisão no além (22/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaVL4_DecisaoNoAlem.py` (importa a VL3 pelo caminho do scratchpad da sessão; a mesma VL3 está nesta pasta), `EtapaVL4_auditoria.py`, saídas e resultados. Base da VL3; sementes 210–219. O além entrega: M (mistura, β = 8), Bc (mistura com β calibrado só pelas memórias: 15,5), DC (a memória de maior ressonância, exata), DS (decide só com peso ≥ 0,5), DP (controle: decisão direto na pista), teto.

**Resultado (ganho de nitidez, trechos guardados):** +10 dB: M −4,84 · Bc +2,42 · **DC +3,28** · DS +3,28 · DP **+4,10** · teto +7,14 dB. 0 dB: M +3,35 · Bc +6,07 · **DC +6,22** · DS +6,22 · DP +5,87 · teto +8,79 dB. Identificação (+10 dB): pista 0,792, fim 0,750; (0 dB): pista 0,542, fim 0,625. Trechos novos: +10 dB, de −7,9 a −6,5 dB em todas as versões; 0 dB, +1,6 a +2,9.

**Critérios:** VL4-1 cumprido (+3,28 dB, 10/10); VL4-2 cumprido (DC − M +8,12 dB); VL4-3 cumprido (Bc − M +7,26 dB); **VL4-4 não cumprido** (DC − DP −0,82 dB, 0/10: com ruído leve a rede reconhece pior que a pista); VL4-5 cumprido nos números (+3,28 e +6,22 dB), **mas a auditoria mostra que a DS decidiu em 100% dos casos, inclusive nos trechos novos**: é idêntica à DC e não protegeu nada; o critério passou porque a DC é positiva, não pela proteção.

**Auditoria:** o erro de pesos baixos (EP ≈ 0,01) vem de a rede cair sempre no mesmo atrator para cada trecho (identificação no fim 0,75 em todas as 10 sementes). Memória 9 é um **atrator-sumidouro**: os trechos 0, 6 e 7 terminam nela; o trecho 0 já é confundido com a 9 na pista; o 7, reconhecido certo na pista na semente 210, é puxado para a 9 pela rede. As memórias são bem separadas entre si (ressonância cruzada máxima 0,03 a 0,10), então o problema não é a separação: é a pista ruidosa parecer mais com algumas memórias e o atrator puxar para elas. O peso de confiança com β calibrado fica ≥ 0,987 no fim: a rede termina sempre "certa de si", mesmo quando errada.

**Leitura:** a decisão no além resolveu a mistura que borrava (de −4,8 para +3,3 dB a +10 dB; de +3,4 para +6,2 dB a 0 dB, 71% do teto). Com ruído forte, o processamento da rede acrescenta ao reconhecimento (0,54 → 0,63; DC > DP); com ruído leve, tira (0,79 → 0,75; DC < DP), pelo sumidouro. Continua sendo memória: nos trechos novos, nenhuma versão ajuda a +10 dB.

**Próximo proposto:** entender e corrigir o sumidouro (por que a memória 9 atrai; equilíbrio dos atratores); uma confiança que meça a dúvida de verdade (a confiança do estado final não serve, porque é sempre alta; talvez a da pista, ou a mudança entre pista e fim).

## Etapa IN — o processamento pelo interior da esfera (23/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaIN_Interior.py` (importa a Etapa LA pelo caminho do scratchpad da sessão), `EtapaIN_auditoria.py`, saídas e resultados. Base da Etapa LA; sementes 230–239; descida à casca de 233 (r = 1/φ) pela extensão de Laplace, recuperação na casca, subida das formas largas (grau ≤ 9; na cascata, casca de 89 com grau ≤ 5, fixados antes de rodar), recuperação na superfície.

**Resultado (altura recuperada):** ruído 2,0: SUP **0,964** · IN 0,313 · IN-sp 0,164 · CO 0,645 · IN2 0,729. Ruído 3,0: SUP **0,927** · IN 0,228 · IN-sp 0,159 · CO 0,415 · IN2 0,453. Ruído 1,0: SUP 0,978 · IN 0,576 · CO 0,862 · IN2 0,886. **IN0 não cumprido** (SUP 0,964 no ruído 2,0; o principal passou a 3,0, como decidido antes); **IN1 e IN2 não cumpridos** (IN muito pior que SUP e que a simples suavização); IN3 cumprido (+0,069, 8/10), sem importância diante do resto. Voltas: todas as versões atingem o limite de 50 por recuperação; a medida de eficiência não informa.

**Auditoria:** três causas. (1) **A premissa falhou para estas memórias:** só 45% da energia delas está nos graus ≤ 9; a memória mora sobretudo nos detalhes finos (o tanh cria detalhe), então as formas largas que sobem carregam só 0,67 da memória, e até a simples suavização (CO) piora. (2) **Erro de desenho do assistente:** na casca, r^L encolhe as memórias (desvio 0,067, contra cerca de 0,6 na superfície); o ganho ajustado bateu no mínimo da grade (0,10) e o estado final ficou seis vezes maior que as memórias; a recuperação na casca não funcionou (0,20), embora as ligações guardem a memória (0,9997). Faltou normalizar a amplitude no interior. (3) **Um achado de Laplace, não um erro:** no interior, as memórias ficam parecidas entre si (ressonância cruzada 0,85 da própria): quanto mais perto do centro, mais o campo é a média, e as diferenças se apagam. A distinção entre memórias vive na superfície.

**Leitura:** o interior, como foi feito, atrapalha. A superfície sozinha já recupera bem (0,93 mesmo com ruído 3,0). O que fica de verdadeiro: o centro resume e unifica (a média); a diferença está na superfície — o processamento não deve buscar distinguir no interior.

## Etapas SA (o serial de atratores) e HZ (o horizonte) — 24/10/2026

**Arquivos:** `AlphaPhi_CampoEsferico_EtapasSA_HZ_SerialHorizonte.py` (importa a Etapa LA pelo caminho do scratchpad da sessão), `EtapasSA_HZ_auditoria.py`, saídas e resultados. Base da Etapa LA. SA: sementes 250–259, 50 voltas fixas. HZ: sementes 240–249.

**SA (identificação; ruído 4,0, o principal pelo SA0):** SUP 0,892 · SN 0,250 · SE 0,258 · SNE 0,242 · INV 0,758. No ruído 2,0: SUP 1,000 · SN 0,508 · INV 1,000. **SA0 cumprido só no ruído 4,0; SA1, SA2 e SA3 não cumpridos** (SN muito pior que SUP e que o controle ao contrário). Captura precoce perto de zero em todas: os sumidouros desta tarefa são raros.

**Auditoria SA:** com β baixo, o previsor dá à memória certa só cerca de 1/12 do peso (0,108 na pista; com β = 8, 0,384): ele vira a **média de todas as memórias**, e o estado se degrada volta a volta (altura 0,76 → 0,43 no ruído 2,0); quando o β sobe, a informação da pista já se perdeu. O controle ao contrário (decisivo primeiro) chega a 0,967 e depois se degrada ao amolecer. **Leitura:** nesta rede, o previsor só funciona decisivo; o começo macio não é "multiplicidade preservada", é a média que apaga (Entrada 356). A hipótese do serial do macio ao decisivo não se sustentou.

**HZ (altura; ruído 3,0):** SUP 0,937 · HZ-S 0,486 · HZ-D 0,775 · CT 0,832 · HZ-D com ε = 1/φ⁶ 0,915 · com ε = 1/φ² 0,230. **HZ1 não cumprido** (nos 100 casos claros, correlação média 0,817 entre os finais com e sem horizonte; mínima perto de 0: o horizonte, mesmo sutil, muda a trajetória de casos decididos); **HZ2 e HZ3 não cumpridos** (o horizonte do próprio campo é pior que um sinal ao acaso); HZ4 cumprido (só na dúvida é melhor que sempre). Dúvida média no fim 0,36: o d_i não é seletivo.

**Leitura HZ:** o horizonte vindo das formas largas do próprio estado reforça o erro (previsão de falha 1) e carrega pouco (18% da energia do estado nos graus ≤ 5); quanto maior ε, pior. Não se sustentou nesta forma.

**O que fica (as duas):** na Rede AP atual, a recuperação depende de um atrator **decisivo desde o início**; todo amolecimento ou sinal adicional de conjunto, por mais sutil, desvia a trajetória. Não há entrada no Manifesto (regra do pesquisador, 24/10: entrada só com resultado coerente).

## Etapa EB — o previsor com a estrutura do EcoBIP (24/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaEB_PrevisorEcoBIP.py` (importa as Etapas SA/HZ e LA pelo caminho do scratchpad da sessão), `EtapaEB_auditoria.py`, saídas e resultados. Sementes 260–269; 50 voltas fixas. EB: força por neurônio β_i (começo 8), alvo 8·φ^(3·c_rel − 3/2) pela concordância com o previsor, transição com memória 1/φ (a mistura do EcoBIP).

**Resultado (ruído 4,0):** SUP altura 0,801, identificação 0,850 · **EB 0,884 / 0,925** · EB-R 0,843 / 0,892 · CT 0,884 / 0,917. β médio no fim: EB 12,6 (desvio 2,7); CT 12,1. **EB1 não cumprido** (+0,075 na identificação, mas só 5/10 a favor: 5 sementes empatadas); **EB2 cumprido** (+0,083 na altura, 10/10); **EB3 não cumprido** (EB ≈ CT: a coerência não acrescenta).

**Auditoria:** o previsor atual com β fixo = 12,6 (a média final da EB) dá altura 0,905 e identificação 0,942 — **melhor que a EB** (EB − β fixo: −0,021 na altura, 0/10; −0,017 na identificação). **Todo o ganho veio de o previsor ficar mais decisivo (β médio maior), e não da estrutura do EcoBIP** (força por região guiada pela coerência, memória 1/φ). A rotação áurea na ordem de atualização (EB-R) foi pior que a ordem ao acaso no ruído 4,0.

**Leitura:** a estrutura do EcoBIP, nesta adaptação, não se sustentou no previsor. O que apareceu é coerente com as Etapas SA e HZ: **o atrator decisivo ajuda**; um β maior que 8 melhora a recuperação neste ruído. Esse último ponto é observação da auditoria, não critério escrito antes: precisa de teste próprio (varredura de β com critérios), e o β = 8 vem da Etapa PF (Ferramentas de Memória v1).
