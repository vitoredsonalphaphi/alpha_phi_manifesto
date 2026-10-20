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

**Ainda fora do canônico** (executados, pendentes de registro em `_pendente_registro/`): Etapas P, A, B, H e o scanner em cascas. Entram aqui quando forem registrados.
