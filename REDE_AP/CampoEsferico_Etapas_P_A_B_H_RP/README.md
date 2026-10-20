# Campo Esférico — Etapas P, A, B, H e RP (19 e 20/10/2026)

**Status:** **registrado** em 20/10/2026, a pedido do pesquisador, em `REDE_AP/AlphaPhi_CampoEsferico_Etapas_P_A_B_H_RP_RESULTADOS_20out2026.md` (Entradas 333–339 do Manifesto 03). Esta pasta, antes `_pendente_registro/`, guarda o código, as saídas e as páginas. Os pontos de auditoria em aberto continuam listados abaixo. Reprodução: rodar primeiro `AlphaPhi_EcoBIP_B0_gera_dobras.py` (gera os `.npy`); a Etapa RP foi refeita a partir desta pasta em 20/10 e deu saída idêntica.

## O que está aqui

| Arquivo | Conteúdo |
|---|---|
| `AlphaPhi_CampoEsferico_EtapaP_PhantomFundo.py` | Etapa P: o Phantom (serial sem selagem) como **fundo inscrito** do campo cultivado |
| `AlphaPhi_CampoEsferico_EtapaA_PhantomAtivo.py` | Etapa A: Phantom **desligado, inscrito e ativo** (o serial tocando durante a dinâmica) |
| `EtapaP_saida_19out2026.txt`, `EtapaA_saida_19out2026.txt` | saídas impressas pelas execuções |
| `EtapaP_resultados.json`, `EtapaA_resultados.json` | números completos (a Etapa A inclui o registro do campo local do neurônio usado no scanner) |
| `pagina_cultivo_do_campo.html`, `pagina_phantom_na_rede.html` | cópias das páginas publicadas (scanner da Etapa C e scanner da Etapa A) |

O serial do Phantom é gerado pelo **código original** `AlphaPhi_SerialPhantom_SemSelagem_COLAB.py` (SHA-256 `41d43999d2716a29a003d10fa3c45c0bf709d6a7230b8630dd35fb8cfc630bd9`), sem alteração, na primeira execução; o `.npy` (4 MB) não é guardado porque é reproduzível. Os scripts importam a versão canônica v1 (commit `f796a3e`).

## Resumo dos resultados (critérios escritos antes de cada execução)

**Etapa P — Phantom como fundo (energia por banda φ → peso do grau esférico):** P0 recuperação do original cumprido (terceira estrutura em 100% dos quadros; os números da Entrada 287 não são reproduzidos exatamente porque vieram de outra medida); P1 Phantom > nenhum **cumprido** (+0,0043, 9/10); P2 Phantom > embaralhado **não** (+0,0006, 6/10); P3 Phantom > ruído e > seno **cumprido** (+0,0084; +0,0030); P4 o fundo do próprio campo é melhor que o Phantom (−0,0180); P5 previsor **cumprido** (Spearman 0,993, erro 0,0054). Leitura: ajuda pouco e sem especificidade de forma; a inibição do fundo do campo pesa mais.

**Etapa A — Phantom ativo (amplitude 1/φ³):** preservação desligado 0,709 / inscrito 0,604 / ativo 0,598 / ruído ativo 0,595 / seno ativo 0,611. A1, A2, A3 **não cumpridos**; A4 (não destrói) **não cumprido** (queda de 0,111). Leitura: nesta amplitude, o Phantom ativo **não preserva** o sinal da rede; atrapalha como ruído ou seno de mesma amplitude.

## Auditoria pendente (a resolver antes do registro)

1. Na Etapa A, o "inscrito" foi **somado** ao fundo do campo (fundo dobrado), ao contrário da Etapa P, em que **substituía** o fundo: confusão de desenho.
2. Só uma amplitude foi testada (1/φ³); amplitudes menores não foram testadas.
3. A rede da Etapa A está perto do limite (16 memórias com forma; preservação de 0,709 mesmo desligada).
4. A tradução banda φ → grau esférico é uma escolha do assistente, não a única possível.

## Etapa B — o EcoBIP 880 nas duas redes (20/10/2026)

**Arquivos:** `AlphaPhi_EcoBIP_B0_gera_dobras.py` (roda o EcoBIP original e guarda as 5 dobras e o controle; rodar primeiro), `AlphaPhi_CampoEsferico_EtapaB_EcoBIP.py`, `EtapaB_saida_tentativa1_degenerada_20out2026.txt`, `EtapaB_saida_corrigida_20out2026.txt`, `EtapaB_resultados.json`, `pagina_ecobip_na_esfera.html`.

**B0:** o `BEEP880_17S.py` original reproduz o registro (campo harmônico no ciclo 10, β máx 4,2098). **Primeira tentativa: degenerada** — a tradução pela energia por banda deixou as memórias quase iguais (correlação 0,84; controle sem cascata 1,000) e as duas redes caíram no mesmo estado (identificação = acaso). **Correção aprovada:** log da energia menos o perfil médio das 5 dobras; critério novo B00 (correlação < 0,3); controle = a mesma cascata na quadrada.

**Resultado corrigido (10 sementes):** B00 cumprido (−0,174 e −0,177). EcoBIP: convencional 0,824 · AlphaFi 0,822 (B1 **não difere**: empate). Controle: convencional 0,869 · AlphaFi 0,757. B2 **cumprido** (+0,110, 8/10), lido como: a desvantagem da AlphaFi desaparece com o EcoBIP — não é vantagem.

**Auditoria pendente:** (1) a recuperação é igual com ou sem pista e igual à preservação: a rede vai sempre aos mesmos estados; só 2 das 5 memórias são estáveis (identificação 0,40); (2) cada memória nasce de 15 números (as bandas), pouco para 233 neurônios; (3) o empate pode vir da suavidade dos padrões (ligação de vizinhança), não do EcoBIP como tal — não verificado. **Próximo:** memórias mais ricas (por exemplo, quadros do STFT ao longo do tempo).

## Etapa H — Hopfield moderna (o eco do dado) sobre a Grade Esférica v1 (20/10/2026)

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaH_Hopfield.py`, `EtapaH_saida_20out2026.txt`. Memórias: entrada + 5 dobras do EcoBIP, 225 faixas → 225 harmônicos (sem sorteio). Versão aprovada (233 pontos) e versão pela regra da grade (610 pontos), declarada antes de rodar.

**Resultado (identificação, pista 40%, EcoBIP; acaso 0,17):** eco puro 0,667 / 0,650; convencional+eco 0,467 / 0,500; convencional 0,400 / 0,400; AlphaFi+eco 0,333 / 0,333; AlphaFi 0,333 / 0,333 (233 / 610 pontos). H00 cumprido; H1, H2, H4 **não**; H3 **difere** (AlphaFi atrás: −0,13 / −0,17).

**Auditoria (descritiva):** a AlphaFi junta as 6 memórias em 2 grupos (pistas 0–2 → memória 1; 3–5 → memória 4); com ou sem fundo, idêntico; o eco mais forte ajuda (λ = 2 → 0,500; λ = 8 → 0,633). Leitura: a vizinhança da memória (σ = 0,35, fixa) filtra as formas finas, onde as memórias ricas diferem; a regra de escala da Etapa C (σ ∝ 1/√N) não foi aplicada à ligação de memória. **Próximo proposto:** vizinhança escalada pela resolução e eco com força declarada em 0,5 / 2 / 8.

## Etapa RP — a reprojeção (233 → 610 pontos encaixados), 20/10/2026

**Arquivos:** `AlphaPhi_CampoEsferico_EtapaRP_Reprojecao.py`, `EtapaRP_saida_20out2026.txt`, `EtapaRP_auditoria_20out2026.txt`, `EtapaRP_resultados.json`. Memórias da Etapa H (6, EcoBIP). Vizinhança da memória = 2 × espaçamento médio (σ = 0,148 em 233; 0,091 em 610). Sem eco. Sementes 90–99.

**Resultado (identificação, pista 40%; acaso 0,17):** A233-H (σ 0,35) 0,333 · A233 0,417 · A610-reproj 0,400 · A610-zero 0,467 · C233 0,367 · C610 0,417. **R0 cumprido** (14 > 9 e ≤ 16; memórias idênticas nos 233 pontos antigos). **R1 não** (+0,083, 5/10), **R2 não** (−0,017), **R3 não** (−0,067, abaixo de −0,05), **R4 não difere** de C610 (−0,017).

**Auditoria (descritiva):** (1) as 6 memórias não são independentes: correlação média −0,13, mas um par chega a +0,86, e todas as redes, incluindo a convencional, juntam as pistas em 2 ou 3 grupos; a tarefa limita mais do que a vizinhança. (2) A vizinhança escalada **abre** os grupos (A233: a pista 0 volta à memória 0 em 8/10, contra 0/10 com σ 0,35), mas não o bastante para os critérios. (3) Em 610 pontos, a sobreposição cai (0,40 e 0,24 contra 0,65 em 233): com σ estreito e só 6 memórias, a dinâmica fica em estados mistos. (4) O bloco antigo foi multiplicado por 233/610 para ficar na escala 1/N da rede maior; essa escolha está no código, mas não foi discutida antes. (5) A medida é grosseira: 6 memórias, logo cada semente vale múltiplos de 1/6. **Leitura:** a reprojeção é **implementável e preserva o antigo** (R0), mas nesta tarefa **não trouxe ganho** de memória. Próximo sugerido: memórias menos correlacionadas (quadros do STFT) antes de repetir.

**Scanner em cascas da Etapa RP** (`pagina_cascas_da_reprojecao.html`, descritivo, sem critério): A233 × A610 reprojetada, semente 90, pista 40%; o filme da dinâmica termina no mesmo estado da execução registrada (conferido nas duas redes). Em 233 pontos só aparecem as cascas 0–9; em 610, 0–14.

**Próximos passos combinados:** varredura de amplitude do Phantom ativo (1/φ³, 1/φ⁵, 1/φ⁷) ou o Phantom ativo no lugar do fundo; depois, registro conjunto como nas etapas anteriores.

---
*Florianópolis · 19 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
