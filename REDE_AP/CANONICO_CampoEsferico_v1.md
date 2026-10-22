# Campo Esférico AlphaPhi — Versão Canônica v1 (congelada em 19/10/2026)

**Commit de congelamento:** `f796a3e` · **Branch:** `claude/good-morning-N6f3S`
*(Uma etiqueta Git `campo-esferico-v1` foi criada, mas o ambiente desta sessão só permite enviar o branch de trabalho; a referência fixa é o commit acima, permanente no histórico. A etiqueta pode ser criada depois pelo próprio GitHub, apontando para esse commit.)*
**Resultados:** `AlphaPhi_CampoEsferico_Etapas_RESULTADOS_19out2026.md` (seções 1 a 6) e `AlphaPhi_MemoriaEsferica_Etapa1_RESULTADOS_19out2026.md`
**Entradas do Manifesto 03:** 326 a 332

## Por que congelar

Estes arquivos são a **própria estrutura** do campo esférico da Rede AP até aqui, não um exemplo dela. O caso do EcoBIP 880 mostrou o risco: uma cópia alterada (um peso trocado, α = 1/137 no lugar de α\* = 1/3) produziu resultados que não valiam para o sinal original. Esta versão fica fixa para que todo teste futuro parta exatamente dela.

## O que cada arquivo é

| Arquivo | Papel |
|---|---|
| `AlphaPhi_MemoriaEsferica_Etapa1.py` | base: esfera de Fibonacci, regra de Hebb, recuperação, comparação pareada |
| `AlphaPhi_MemoriaEsferica_Orcamento.py` | teste com orçamento de ligações |
| `AlphaPhi_CampoEsferico_Etapa0_Cartografia.py` | campos candidatos, ambiente com forma, ressonância |
| `AlphaPhi_CampoEsferico_EtapaT_Previsor.py` | **previsor do campo vazio** (Funk–Hecke) e medidor por harmônicos esféricos |
| `AlphaPhi_CampoEsferico_EtapaC_Cultivo.py` | **cultivo do campo** em pontos encaixados pelo ângulo áureo |
| `AlphaPhi_CampoEsferico_PrevisorEnvelopado.py` | **previsor condicional da memória** com forma |
| `AlphaPhi_CampoEsferico_EtapaE_Ferramentas.py` | previsor com ferramentas instaladas no campo cultivado |

## Impressões digitais (SHA-256)

Qualquer alteração em qualquer arquivo, mesmo de uma vírgula, muda a impressão digital.

    07b66cc5c5231023156d8616448d7b3844600d251ac1175e15fefebd2a438f0c  AlphaPhi_MemoriaEsferica_Etapa1.py
    2a22dea2715c491b06a10cb522c0ed0ee6c317f58a8ccb8fa04c0cb46cd7dfbd  AlphaPhi_MemoriaEsferica_Orcamento.py
    b622fe29f5d125b3ebe2cd1430252af26b4cbbe54ed745869c5e7845529c432f  AlphaPhi_CampoEsferico_Etapa0_Cartografia.py
    2df41886e7c6d3bdd5cf016e5815310d4b91f694995b5e3a1300121aafbaebe1  AlphaPhi_CampoEsferico_EtapaT_Previsor.py
    e377763d7efed34fa02820238acb481affe169e628aea6174361233b0054930c  AlphaPhi_CampoEsferico_EtapaC_Cultivo.py
    d3ea947a753964ffc72a8216d4fb65b38b77456c8fedc398be5719936dbcf44e  AlphaPhi_CampoEsferico_PrevisorEnvelopado.py
    8d81c215477aecdcd53912c23a6072f7dd48574178e49f89ff783b2e1b789e88  AlphaPhi_CampoEsferico_EtapaE_Ferramentas.py

**Como conferir** (na pasta `REDE_AP`): `sha256sum -c` com as linhas acima, ou comparar uma a uma com `sha256sum <arquivo>`.

## Regra

1. Testes novos **importam** estes arquivos; não os copiam nem os editam.
2. Uma mudança necessária gera uma **v2**, com arquivos novos, novo commit de congelamento e registro do que mudou e por quê. A v1 permanece intacta.
3. Antes de cada teste novo, conferir as impressões digitais.

---
*Florianópolis · 19 de outubro de 2026 · Rede AP · Vitor Edson Delavi · Claude*

## Acréscimos canônicos (20/10/2026) — sem alterar nenhum arquivo da v1

| Arquivo | Papel | SHA-256 |
|---|---|---|
| `AlphaPhi_GradeEsferica_v1.py` | **Grade Esférica de Referência v1**: posição (r, θ, φ), raios canônicos (0, 1/φ, 1), pontos encaixados, harmônicos, decomposição por grau, cascas de leitura. Autoverificação: mesmos pontos da Etapa C e mesma base da Etapa T | `0512264d7d3ede7ab3563fa17eb7b8ac551e6556063ab86afd6cbe2cf2dbf299` |
| `AlphaPhi_CampoEsferico_EtapaF_Crescimento.py` | Etapa F (ferramentas instaladas durante o crescimento), registrada em 19/10 | `9bfaa146936b1d3165ade58e1f2cd3cfed850a3b9586d1c1215c502896707bb1` |

**Regras que entram no canônico a partir de 20/10:**
1. **Todo teste novo importa a Grade Esférica v1.**
2. **A vizinhança acompanha a resolução** (σ ∝ 1/√N), no campo e na ligação de memória (lição da Etapa H).
3. **O grau medível segue a regra da grade:** (L+1)² ≤ N/2. Um sinal que pede graus acima disso pede uma esfera maior (gatilho da reprojeção).

**Ainda fora do canônico** (executados e registrados em 20/10/2026 em `AlphaPhi_CampoEsferico_Etapas_P_A_B_H_RP_RESULTADOS_20out2026.md`; código em `CampoEsferico_Etapas_P_A_B_H_RP/`): Etapas P, A, B, H e o scanner em cascas. Entram aqui quando forem registrados.

## Acréscimos canônicos (21/10/2026) — provisórios, sem alterar nenhum arquivo da v1

A pedido do pesquisador: as ferramentas que responderam de forma positiva na estabilização da memória entram no canônico **até que sejam refinadas**.

| Arquivo | Papel | SHA-256 |
|---|---|---|
| `AlphaPhi_FerramentasMemoria_v1.py` | **Ferramentas de Memória v1**: o campo afinado (peso do fundo escalado), o previsor, o batedor, a dinâmica de recuperação (contínua) e a binária, e a régua da conclusão. Importa a Grade Esférica v1. Conferido: reproduz **exatamente** o código da Etapa PF afinada (ligações, batedor e as dinâmicas V0, V3, V4, V5 idênticos; `CampoEsferico_Etapas_S_K_PF/conferencia_FerramentasMemoria_v1_saida.txt`) | `d8dc24674b55d127b4e9ad1b989194c4cbfafefc3e6d5c00f3fbbf4d7bbba295` |

**Regras que entram no canônico a partir de 21/10:**
4. **O fundo do campo acompanha a vizinhança:** peso = 0,02 × (massa da vizinhança da memória) / (massa com σ = 0,35), para manter a razão fundo/memória do desenho original (≈ 0,16–0,26). Corrige o erro de calibração das Etapas RP, S, K e PF. Ganho medido: neurônios estáveis na própria memória de 71% para 91,5%.
5. **O previsor é a via padrão de recuperação** (força 0,5, β = 8): a ferramenta mais forte da Etapa PF (conclusão 0,420 com pista de 40%; 0,794 com pista de 25%).
6. **O batedor está disponível** (força 0,5): acrescenta quando usado sem o previsor (PF2, +0,044, 10/10), e o batedor certo importa (PF4, +0,068, 9/10); somado ao previsor **não** acrescenta (PF3). Usar declarado em cada teste.

**Não entram** (sem efeito positivo): o neurônio contínuo sozinho, o "ponto e campo" sem outra via (PF1, duas vezes), e o resfriamento lento (V2). Continuam como hipótese na agenda.

**Agnóstico:** por construção, nenhuma ferramenta depende do sinal de origem (as memórias entram como padrões ±1 nos pontos da grade). Até aqui só foi **testado** com memórias derivadas do EcoBIP 880; a verificação em outros substratos é a agenda V.

**Resultados de origem:** Etapas S, K, PF e a afinação, registradas em 21/10/2026 em `AlphaPhi_CampoEsferico_Etapas_S_K_PF_RESULTADOS_21out2026.md`; código em `CampoEsferico_Etapas_S_K_PF/`.

## Acréscimos canônicos (22/10/2026) — a Grade Esférica v2, sem alterar nenhum arquivo da v1

| Arquivo | Papel | SHA-256 |
|---|---|---|
| `AlphaPhi_GradeEsferica_v2.py` | **Grade Esférica de Referência v2**, a base que antecede as ferramentas (Entrada 346): pontos encaixados com longitude pelo ângulo áureo e **altura por √3**; **cascas em proporção áurea** (r = φ^(−k), com 610, 233, 89, 34, 13, 5, 2 pontos); **extensão harmônica** para o interior; **células por escala** de Fibonacci; **pesos pela área da célula** nas médias e nas energias (o centro α como média pesada; a conservação da energia); **endereço** por anéis para escrita e leitura. Construída sobre a v1, que importa sem alteração | `dfa16b5b65ea5e6b18c829cfb35e25bfd2be1ab9c85124d3327dfc84c8244054` |

**Como entrou:** testes G1–G8 e as duas correções (critérios aprovados antes de rodar; `_pendente_registro/teste_GradeEsferica_v2*.py`) e a **verificação das Etapas T e C nos pontos novos**, com o código canônico importado sem alteração (`_pendente_registro/verificacao_EtapaT_altura_raiz3.py`): V1 medidor 100% (condição 7,8; com √2, 16,9 em 256 pontos até o grau 14); V2 campo vazio 1,000 dentro de ±1 de L\* = 3; V3 memória sem forma Spearman 0,978 e erro 0,006; V4 cultivo na faixa prevista em 100% das repetições em todas as etapas; V5 sem nenhuma perda em relação a √2. Decisão combinada antes de rodar: se V1–V5 passassem, a v2 entraria com a altura por √3.

**Regras que entram no canônico a partir de 22/10:**
7. **Todo teste novo importa a Grade Esférica v2.** A v1 continua congelada e é a base dos resultados canônicos de 19 a 21/10.
8. **Médias e energias pesam cada ponto pela área da sua célula.**
9. **As Ferramentas de Memória v1 foram verificadas nos pontos da v1;** usá-las sobre a v2 pede, antes, a conferência de que reproduzem o comportamento (a fazer).
