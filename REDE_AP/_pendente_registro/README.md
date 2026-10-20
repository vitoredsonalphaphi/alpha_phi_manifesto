# Pendente de registro — Etapas P e A (o Phantom no campo esférico), 19/10/2026

**Status:** executado, **ainda não registrado**. O pesquisador pediu para registrar junto com resultados futuros. Esta pasta existe só para que código e números **não se percam** quando a sessão terminar. Nada aqui foi incorporado ao arquivo de resultados nem ao Manifesto.

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

**Próximos passos combinados:** varredura de amplitude do Phantom ativo (1/φ³, 1/φ⁵, 1/φ⁷) ou o Phantom ativo no lugar do fundo; depois, registro conjunto como nas etapas anteriores.

---
*Florianópolis · 19 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
