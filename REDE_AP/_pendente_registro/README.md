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

**Próximos passos combinados:** varredura de amplitude do Phantom ativo (1/φ³, 1/φ⁵, 1/φ⁷) ou o Phantom ativo no lugar do fundo; depois, registro conjunto como nas etapas anteriores.

---
*Florianópolis · 19 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*
