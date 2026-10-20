# Campo Esférico — Etapas P, A, B, H e RP · Resultados (19 e 20/10/2026)

**Status:** registrado a pedido do pesquisador em 20/10/2026 · os pontos de auditoria ainda em aberto estão listados em cada seção e não foram apagados
**Antecedente:** `AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md` (Etapas T, C, E, F; canônico v1, commit `f796a3e`)
**Código e saídas:** `REDE_AP/CampoEsferico_Etapas_P_A_B_H_RP/` (antes `_pendente_registro/`); base: `REDE_AP/AlphaPhi_GradeEsferica_v1.py` (SHA-256 `0512264d7d3ede7ab3563fa17eb7b8ac551e6556063ab86afd6cbe2cf2dbf299`)
**Entradas do Manifesto 03:** 333 a 339 (os enunciados de 20/10/2026 que antecederam cada teste)
**Regra comum:** critérios escritos na conversa antes de cada execução; comparação pareada = diferença média > 2 erros-padrão e vantagem em ≥ 8 de 10 sementes; todos os números abaixo foram impressos por código executado nas sessões de 19 e 20/10/2026.

## Síntese

| Teste | Pergunta | Resultado |
|---|---|---|
| Etapa P — Phantom como fundo | a forma do Phantom inscrita no campo ajuda a memória? | ajuda pouco e **sem especificidade de forma** (embaralhado igual) |
| Etapa A — Phantom ativo (1/φ³) | o Phantom tocando preserva o sinal da rede? | **não**: atrapalha como ruído ou seno da mesma amplitude |
| Etapa B — EcoBIP 880 nas duas redes | a AlphaFi esférica difere da convencional no sinal de origem? | **empate** no EcoBIP; desvantagem da AlphaFi no controle |
| Etapa H — Hopfield moderna (eco do dado) | o eco do dado melhora a AlphaFi? | **não** com λ = 0,5; o eco puro funciona; a AlphaFi junta as memórias em 2 grupos (vizinhança larga) |
| Etapa RP — reprojeção 233 → 610 | crescer a esfera e reprojetar melhora a memória? | **implementável e preserva o antigo** (R0); **sem ganho** nesta tarefa |

*Em palavras:* nenhuma destas etapas mostrou vantagem da rede AlphaFi sobre a convencional. Elas mostraram **onde está o limite**: memórias correlacionadas e uma vizinhança que filtra o fino. E deixaram três ferramentas novas prontas: a Grade Esférica v1, a leitura em cascas e a reprojeção.

## 1. Etapa P — o Phantom como fundo inscrito (19/10)

`AlphaPhi_CampoEsferico_EtapaP_PhantomFundo.py` · serial gerado pelo código original `AlphaPhi_SerialPhantom_SemSelagem_COLAB.py` (SHA-256 `41d43999d2716a29a003d10fa3c45c0bf709d6a7230b8630dd35fb8cfc630bd9`), sem alteração.

| Critério | Resultado | Veredito |
|---|---|---|
| P0 recuperação do original | terceira estrutura em 100% dos quadros | cumprido (os números da Entrada 287 vieram de outra medida e não se reproduzem exatamente) |
| P1 Phantom > nenhum fundo | +0,0043 (9/10) | cumprido |
| P2 Phantom > embaralhado | +0,0006 (6/10) | **não** |
| P3 Phantom > ruído e > seno | +0,0084; +0,0030 | cumprido |
| P4 Phantom × fundo do próprio campo | −0,0180 | o fundo do campo é melhor |
| P5 previsor | Spearman 0,993, erro 0,0054 | cumprido |

## 2. Etapa A — o Phantom ativo (19/10)

`AlphaPhi_CampoEsferico_EtapaA_PhantomAtivo.py` · amplitude 1/φ³.

| Estado | Preservação |
|---|---|
| desligado | 0,709 |
| inscrito | 0,604 |
| Phantom ativo | 0,598 |
| ruído ativo | 0,595 |
| seno ativo | 0,611 |

A1, A2, A3 **não cumpridos**; A4 (não destrói) **não cumprido** (queda de 0,111).

**Auditoria em aberto (P e A):** (1) na Etapa A o "inscrito" foi somado ao fundo do campo (fundo dobrado), ao contrário da Etapa P, em que o substituía; (2) só a amplitude 1/φ³ foi testada; (3) a rede da Etapa A está perto do limite (16 memórias com forma); (4) a tradução banda φ → grau esférico é uma escolha do assistente.

## 3. Etapa B — o EcoBIP 880 nas duas redes (20/10)

`AlphaPhi_EcoBIP_B0_gera_dobras.py` (roda o `BEEP880_17S.py` original, α\* = 1/3) e `AlphaPhi_CampoEsferico_EtapaB_EcoBIP.py`.

- **B0 cumprido:** o código original reproduz o registro (campo harmônico no ciclo 10, β máx 4,2098).
- **Primeira tentativa degenerada** (guardada em `EtapaB_saida_tentativa1_degenerada_20out2026.txt`): a energia por banda deixou as memórias quase iguais (correlação 0,84; controle sem cascata 1,000); identificação = acaso. Correção aprovada antes de rodar de novo: log da energia menos o perfil médio; critério B00; controle = a mesma cascata na quadrada.

| Sinal | Rede | Preservação | Identificação (acaso 0,20) |
|---|---|---|---|
| EcoBIP 880 | convencional | 0,824 | 0,40 |
| EcoBIP 880 | AlphaFi esférica | 0,822 | 0,40 |
| quadrada com cascata | convencional | 0,891 | 0,58 |
| quadrada com cascata | AlphaFi esférica | 0,757 | 0,40 |

B00 cumprido (−0,174 e −0,177). **B1 não difere** (−0,0019, 7/10: empate). **B2 cumprido** (+0,110, 8/10), lido como: a desvantagem da AlphaFi desaparece com o EcoBIP; não é vantagem.

**Auditoria em aberto:** (1) a recuperação é igual com ou sem pista: a rede vai sempre aos mesmos estados; (2) cada memória nasce de 15 números; (3) o empate pode vir da suavidade dos padrões, não do EcoBIP como tal (não verificado).

**Leituras descritivas (sem critério):** o scanner em três camadas mostrou que a tradução por 15 bandas φ é o gargalo; as três pulsações simétricas da camada 2 vêm das bandas graves (20–85 Hz, oscilação de 40% a 57%); os picos da 5ª dobra estão em 137 Hz (F_M = 220/φ), 220 Hz, 52/180/304 Hz e 2.640/4.400 Hz, e o 880 Hz quase desaparece. Na resposta das redes, a AlphaFi põe 0,41 da energia nos graus 0–2 e 0,27 nos graus 8–14; a convencional, 0,49 e 0,24 (uma repetição).

## 4. Etapa H — a Hopfield moderna, o eco do dado (20/10)

`AlphaPhi_CampoEsferico_EtapaH_Hopfield.py` · 6 memórias (entrada + 5 dobras), 225 faixas → 225 harmônicos, sem sorteio · eco: pesos softmax(β·m), β = 8, λ = 0,5 da escala da rede · 233 pontos (aprovado) e 610 (pela regra da grade, declarado antes).

| Condição (identificação, pista 40%; acaso 0,17) | 233 | 610 |
|---|---|---|
| eco puro | 0,667 | 0,650 |
| convencional + eco | 0,467 | 0,500 |
| convencional | 0,400 | 0,400 |
| AlphaFi + eco | 0,333 | 0,333 |
| AlphaFi | 0,333 | 0,333 |

H00 cumprido (−0,13). H1, H2, H4 **não**. H3 **difere**: a AlphaFi fica atrás (−0,13 / −0,17).

**Auditoria:** a AlphaFi junta as 6 memórias em 2 grupos; com ou sem fundo, idêntico; a vizinhança da memória (σ = 0,35, fixa) filtra as formas finas; eco mais forte ajuda (λ = 2 → 0,500; λ = 8 → 0,633), mas então é o eco que faz o trabalho. Daí a regra **"a vizinhança acompanha a resolução"** (acréscimos ao canônico, 20/10).

## 5. Etapa RP — a reprojeção (20/10)

`AlphaPhi_CampoEsferico_EtapaRP_Reprojecao.py` · memórias da Etapa H · vizinhança = 2 × espaçamento médio (σ = 0,148 em 233; 0,091 em 610) · reprojeção: 233 → 610 pontos encaixados, o bloco dos 233 antigos conserva as ligações da rede de 233 · sem eco · sementes 90–99.

| Condição | Estável | Identificação 25% | Identificação 40% | Sobreposição 40% |
|---|---|---|---|---|
| A233-H (σ 0,35) | 0,943 | 0,333 | 0,333 | 0,732 |
| A233 (σ 0,148) | 0,918 | 0,417 | 0,417 | 0,653 |
| A610-reproj | 0,842 | 0,400 | 0,400 | 0,398 |
| A610-zero (σ 0,091) | 0,811 | 0,483 | 0,467 | 0,241 |
| C233 | 0,948 | 0,500 | 0,367 | 0,764 |
| C610 | 0,951 | 0,467 | 0,417 | 0,815 |

| Critério | Resultado | Veredito |
|---|---|---|
| R0 gatilho | graus pedidos 14 > medível em 233 (9) e ≤ medível em 610 (16); memórias idênticas nos 233 pontos antigos | **cumprido** |
| R1 A233 > A233-H | +0,083 (EP 0,045, 5/10) | não |
| R2 A610-reproj > A233 | −0,017 (EP 0,068, 4/10) | não |
| R3 A610-reproj ≥ A610-zero − 0,05 | −0,067 | não |
| R4 A610-reproj difere de C610 | −0,017 (EP 0,046, 3/10) | não difere |

**Auditoria:** (1) as 6 memórias não são independentes: correlação média −0,13, mas um par chega a +0,86, e todas as redes, inclusive a convencional, juntam as pistas em 2 ou 3 grupos; (2) a vizinhança escalada abre os grupos (A233: a pista 0 volta à memória 0 em 8/10, contra 0/10 com σ 0,35), sem chegar aos critérios; (3) em 610 pontos a sobreposição cai (estados mistos); (4) o bloco antigo foi multiplicado por 233/610 para ficar na escala 1/N da rede maior, escolha que está no código e não foi discutida antes; (5) 6 memórias tornam a medida grosseira (múltiplos de 1/6). **Nota declarada no código:** o "previsor antes e depois" foi substituído pela estabilidade medida, porque o previsor condicional pede a estatística de um ambiente e aqui há 6 memórias fixas.

## Próximo passo

Memórias menos correlacionadas (quadros do STFT do EcoBIP ao longo do tempo) e repetir H (com λ declarado em 0,5 / 2 / 8) e RP. Depois: Phantom ativo no lugar do fundo, com varredura de amplitude (1/φ³, 1/φ⁵, 1/φ⁷); eixo radial (Etapa R); verificação independente (célula do Colab; outra IA auditando o código).

---
*Florianópolis · 20 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
