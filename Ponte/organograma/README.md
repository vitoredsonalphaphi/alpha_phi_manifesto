# Organograma da Ponte `[f ○ c]`

**Status:** rascunho **v0** (10/10/2026), aguarda revisão do pesquisador.

## O que é
Um mapa em árvore do desenvolvimento da tradutibilidade (ética → estética <> geometria → matemática → IA). Cada nó tem **estágio, status, fonte e, quando existe, critério de falha**. A cor do nó é o status.

## Arquivos
| Arquivo | Função |
|---|---|
| `organograma.yaml` | **A fonte.** Edita-se aqui (nós, status, notas). |
| `gerar_organograma.py` | Gera as imagens a partir do `.yaml`. |
| `imagens/` | Imagens geradas, com a data no nome. **Versões antigas ficam guardadas.** |

## Status (cores)
`demonstrado` · `em uso` · `a reauditar` · `parcial` · `hipótese` · `não demonstrado` · `refutado` · `em aberto`
"Não demonstrado" **não** significa "falso". "Refutado" só se um teste mostrou o contrário.

## Formas de triangulação (campo `forma`)
1. **mediador** — o ponto entre os extremos (o `<>`)
2. **terceira via** — o terceiro fora do eixo dos polos
3. **dialética** — tese, antítese, refinamento
4. **ciclo** — sequência de domínios que volta

## Como atualizar
1. Editar `organograma.yaml` (mudar `status`, `nota`, acrescentar nós) e a `data` em `meta`.
2. Rodar `python3 gerar_organograma.py` (precisa de `matplotlib` e `pyyaml`).
3. As imagens novas entram em `imagens/` com a nova data; as antigas permanecem.

## Regras
- Só entra resultado numérico **impresso por código executado na sessão**.
- O assistente propõe; o **pesquisador decide** status e estrutura.
- Este organograma é leitura do **estado**, não prova de nada.
