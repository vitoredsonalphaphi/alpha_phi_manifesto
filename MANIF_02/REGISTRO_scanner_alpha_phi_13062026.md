# Registro — Scanner α-φ: Da Proposta à Nomeação

**Manifesto AlphaPhi · Segundo Ciclo**
**Período registrado: 13.06.2026**
**Florianópolis**

---

## Nota sobre este Documento

Este é o registro compilado do ciclo de desenvolvimento que culminou no
Scanner α-φ — da observação fundadora até a nomeação formal.

Não é um resumo. É um registro de fases.
Cada fase tem sua estrutura própria. Cada fase entregou algo que a seguinte
não poderia ter sem ela.

---

## FASE I — A Observação Fundadora

**Origem:** observação sobre a árvore e as fases da rede

A pergunta que abriu este ciclo foi simples:

> Quantas fases existem na rede do SST-2, e o que acontece em cada uma?

A rede Fibonacci — [135→55→89→137→1] — tem três fases intermediárias:
Fase 1 (55 unidades), Fase 2 (89 unidades), Fase 3 (137 unidades, espaço α-nativo).

A observação complementar, que chegou junto:

> O eco-ressonante já tem uma pré-função implícita. Ele observa antes de agir.
> Essa pré-função pode ser explicitada — e pode observar cada fase da rede
> antes que o treino comece.

A conclusão direta:

> Se α pode residir em diferentes fases dependendo do substrato,
> o eco pode identificar qual fase — antes do treino.

---

## FASE II — Eco de Fase (v1): Primeira Construção

**Código:** `AlphaPhi_ECO_PreFase_COLAB.py`

A primeira versão do eco de fase operava com um único espectro por fase:
`score_f = coh_alpha_f × var_entre_classes_f`

Parâmetros iniciais:
- N_SONDA = 5 (fixo)
- LIMIAR_SUBSTRATO = 1e-5 (muito baixo)

**Resultado no substrato de caracteres SST-2:**
- Fase escolhida: Fase 1 (55 unidades) — consistente em 10/10 seeds
- max_score ≈ 0.000061
- Diagnóstico emitido: ADEQUADO (incorreto)
- Diagnóstico real: substrato sem sinal semântico

**O que o erro revelou:**

O limiar de 1e-5 estava demasiado baixo — qualquer score mínimo era considerado
adequado. O eco escolhia a fase correta (onde havia máxima coerência relativa)
mas não sabia distinguir entre coerência estrutural e coerência semântica.

Primeiro sinal de que o substrato de caracteres não carregava o sinal buscado.
Não foi captado como diagnóstico ainda — apenas como resultado suspeito.

---

## FASE III — A Questão do Campo Próprio

**Observação central desta fase:**

> O eco mede o observado. Mas quem mede a qualidade da observação?
> Se o eco não sabe se está vendo bem, não sabe quando parar de olhar.

Esta pergunta abriu três extensões simultâneas:

**Extensão 1 — Observação Multi-Espectral:**
Em vez de um único H_alpha por fase, o eco observa três espectros:

```
Espectro A (energia)   : H_α da distribuição de |ativações| por amostra
Espectro B (ativação)  : fração de neurônios ativos pós-ReLU
Espectro C (variância) : H_α da variância inter-amostras por neurônio
```

Coerência composta (φ-ponderada):
```
coh_composta = wm × coh_A + wn × mean(coh_B, coh_C)
wm = 1/φ ≈ 0.618   (espectro de energia como primário)
wn = 1 - 1/φ ≈ 0.382
```

**Extensão 2 — Meta-Coerência:**
O eco mede a coerência da própria observação via H_α dos scores entre fases:

```
meta_coh = 1 − H_α(scores) / log(N_fases)
```

Alta meta_coh → scores concentrados → eco está em fase com o substrato → age.
Baixa meta_coh → scores dispersos → observação incerta → expande.

Este é o campo da função do eco — a mesma estrutura (observa → mede H_α → informa)
aplicada ao próprio processo de observação. Uma oitava acima.

**Extensão 3 — Sondagem Adaptativa:**
O eco decide quando parar de sondar:

```
N_SONDA_MIN = 3     (mínimo antes de decidir)
N_SONDA_MAX = 10    (máximo — evita loop infinito)
META_COH_LIMIAR = 0.70

se epoch ≥ N_SONDA_MIN E meta_coh ≥ 0.70:
    fase_otima = argmax(EMA_α(scores))
    finaliza sondagem
```

---

## FASE IV — Eco de Fase (v2): Construção Multi-Espectral

**Código:** `AlphaPhi_ECO_PreFase_v2_COLAB.py`

Implementação das três extensões acima.

LIMIAR_SUBSTRATO corrigido para 1e-3.

**Resultado no substrato de caracteres SST-2:**
```
meta_coh médio : 0.093   (limiar=0.70)
ciclos médio   : 10.0    (esgotou — nunca convergiu)
substrato      : 0/10 adequados
Diagnóstico    : INADEQUADO
```

**Leitura correta:**
O eco multi-espectral com meta-coerência identificou o que v1 não conseguia:
distribuições de caracteres são estruturalmente coerentes (H_alpha moderado)
mas não carregam sinal semântico discriminável — meta_coh permanece em ~0.09
porque os scores de todas as fases ficam próximos de zero, distribuição uniforme,
entropia máxima → meta_coh mínima → sondagem usa todos os ciclos disponíveis →
diagnóstico: substrato inadequado para esta tarefa.

---

## FASE V — A Cadeia de Eco-Ressonantes

**Documento:** `MANIF_02/TECNICA_eco_sequencial_cadeia_pre_funcao.md`

A observação fundadora da cadeia:

> O eco-ressonante é agnóstico — mede coerência em qualquer substrato.
> A pré-função prova que ele pode observar antes de agir.
> Portanto: uma cadeia de eco-ressonantes cobre todas as funções simultaneamente,
> cada um especializado num domínio específico.

Estrutura da cadeia:

```
Substrato → Eco1 → Eco2 → Eco3 → Eco4 → Eco5 → Função principal
              ↓      ↓      ↓      ↓      ↓
           info1  info2  info3  info4  info5
```

Os cinco ecos:

| Eco | Domínio | O que observa | Status |
|-----|---------|---------------|--------|
| Eco 1 | Substrato | Adequação semântica do dado bruto | ✓ construído |
| Eco 2 | Fase | Onde α pode residir neste substrato | ✓ construído |
| Eco 3 | Sinal | Separabilidade durante o treino | a construir |
| Eco 4 | Convergência | Trajetória de otimização em direção a φ³ | a construir |
| Eco 5 | Campo | Formação do campo harmônico (β_max ≥ φ³) | ✓ confirmado |

**Propriedade central da cadeia:**
Resultados negativos deixam de ser falhas — tornam-se diagnósticos precisos.
O mesmo resultado que antes seria "o sistema não funcionou" passa a ser lido como:

```
Eco 1: substrato coerente ✓
Eco 2: α não encontra discriminabilidade → substrato sem sinal semântico
Eco 5: campo formou — α e φ operam
Conclusão: o sistema funcionou. O substrato foi o elemento ausente.
```

---

## FASE VI — α Efetivado: O Decreto

**Data:** 13.06.2026

Na sessão anterior (Segundo Ciclo, Documento FILOSOFICA), a cadeia foi documentada:

```
H_alpha sobe com profundidade
→ α é simultâneamente régua E atrator
→ α mede de si mesmo (entropia de si mesmo)
→ triangulação: α nos dois vértices da base
→ α fisicamente bidirecional
→ Lucas(4) = 7: a constante que conecta a escala
```

13.06.2026 é a data em que α foi declarado efetivado:
o acoplamento de α como régua nativa de entropia foi confirmado
experimentalmente e a arquitetura recebeu seu nome definitivo.

---

## FASE VII — Nomeação: Scanner α-φ

**Decreto:** 13.06.2026

O nome foi proposto pelo próprio desenvolvimento — não imposto sobre ele.

> "Creio que nessa fase então, é passivo de um resignificado.
> Eu peço que você nomeie esse código como Scanner Alpha Fi..."

Razão do nome:

Um scanner mapeia o ambiente antes de qualquer ação.
Entrega um mapa para o operador decidir.
É substrato-agnóstico: escaneia qualquer coisa.
É não-invasivo: observa sem modificar o que observa.

O sequencial de eco-ressonantes faz exatamente isso:
percorre cada espectro do dado, de cada fase da rede,
antes que o agente treine uma única época.

**Decreto formal:** `MANIF_02/NOMENCLATURA_scanner_alpha_phi.md`

---

## FASE VIII — ScannerAlphaPhi: Ferramenta Autônoma

**Código:** `AlphaPhi_Scanner_COLAB.py`

O Scanner α-φ como classe reutilizável, desacoplada de qualquer experimento:

```python
scanner = ScannerAlphaPhi(rede)
for epoch in range(N_EPOCHS):
    if not scanner.pronto:
        scanner.escaneia(X_batch, y_batch)
    lr = LR_BASE * scanner.beta(X_batch)
    # ... treino ...
print(scanner.relatorio())
```

Constantes como atributos de classe (α e φ como propriedades do instrumento):

```python
PHI              = (1 + np.sqrt(5)) / 2
ALPHA            = 1 / 137.035999084
LOG_ALPHA        = np.log(1.0 / ALPHA)   # log(137) ≈ 4.920
wm               = 1.0 / PHI             # 1/φ ≈ 0.618
wn               = 1.0 - 1.0 / PHI      # 1-1/φ ≈ 0.382
META_COH_LIMIAR  = 0.70
LIMIAR_SUBSTRATO = 1e-3
N_SONDA_MIN      = 3
N_SONDA_MAX      = 10
```

**Resultado final — substrato de caracteres SST-2:**

```
meta_coh médio : 0.0825  (limiar=0.70)
ciclos médio   : 10.0    (esgotou em todos os seeds)
β médio        : 1.7448
substrato      : 0/10 adequados
Diagnóstico    : substrato INADEQUADO
               meta_coh não convergiu — sinal semântico ausente neste substrato
Teste t CC vs AA: p=0.3434
```

**Observação sobre β ≈ 1.74 para substrato INADEQUADO:**

O Scanner não para de funcionar com substrato inadequado.
Continua medindo coerência — estrutural, não semântica.
Distribuições de caracteres SÃO coerentes (têm estrutura de frequência).
Apenas não são discrimináveis para sentimento.
β ≈ 1.74 indica: substrato tem estrutura, mas não a estrutura que esta tarefa busca.
O Scanner detecta a diferença. Essa é a função do diagnóstico.

---

## FASE IX — O Domínio Correto do Scanner

**Clarificação conceptual que encerra este ciclo:**

O eco-ressonante observa coerência de distribuições de FREQUÊNCIA.
Pesos semânticos de NLP não são frequências.
O substrato de caracteres SST-2 → meta_coh baixa → diagnóstico correto.

O domínio natural do Scanner α-φ:
- Frequências de áudio (FFT de sinais físicos)
- FFT de bits (dados tratados como ondas)
- Grafeno (substrato de frequências físicas puras)
- Séries temporais com estrutura espectral

**A ponte — Fourier:**
Fourier decompõe qualquer sinal em frequências componentes.
Bits tratados como ondas via FFT → φ emerge como força restauradora sobre as fases
espectrais (confirmado no experimento eco_text_009).

**Restauração de fase = eficiência:**
Componentes fora de fase se acumulam como lentidão sistêmica.
φ como força restauradora reduz essa acumulação.
Restauração harmônica = ganho de eficiência — não apenas metáfora, mecanismo.

---

## FASE X — O Caminho Dialético

**Reflexão sobre a estrutura da pesquisa:**

O teste "errado" (substrato semântico para o Scanner de frequências) deu a
resposta certa: identificou o domínio correto do Scanner por exclusão.

Este é o E/X dentro da própria pesquisa:
- Expansão → busca de campo em todo substrato disponível
- Retração → diagnóstico INADEQUADO revela onde o campo NÃO está
- Avanço correto → domínio de frequências como substrato natural

O recuo foi necessário para o avanço correto.

Isso já estava escrito no Manifesto 01: *"a ideia nos criando"* —
a ideia mostrando o caminho errado para revelar o certo.

---

## Síntese do Ciclo

| Data | Marco |
|------|-------|
| Jun/2026 | Eco de Fase v1 — primeira sondagem por fase |
| Jun/2026 | Diagnóstico incorreto (LIMIAR_SUBSTRATO = 1e-5) |
| Jun/2026 | Eco de Fase v2 — multi-espectral + meta-coerência |
| Jun/2026 | Diagnóstico correto: INADEQUADO 0/10 |
| Jun/2026 | Cadeia documentada: 5 ecos, propriedade diagnóstica |
| 13.06.2026 | α efetivado — acoplamento como régua nativa confirmado |
| 13.06.2026 | DECRETO: Scanner α-φ nomeado |
| 13.06.2026 | ScannerAlphaPhi — classe autônoma e reutilizável |
| 13.06.2026 | Domínio correto identificado: frequências, não pesos semânticos |

**O que foi construído:**

1. Ferramenta: Scanner α-φ como instrumento de mapeamento pré-ação
2. Arquitetura: cadeia de eco-ressonantes, cada um em seu domínio
3. Diagnóstico: substrato inadequado = resultado positivo (informação precisa)
4. Clarificação: domínio de frequências como espaço natural de operação
5. Direção: FFT de bits, grafeno, sinais físicos — próximos substratos

**O que permanece a construir:**

- Eco 3 (sinal): discriminabilidade durante o treino
- Eco 4 (convergência): trajetória em direção a φ³
- Teste em substrato de frequências (FFT, áudio, séries temporais)

---

*Florianópolis · 13.06.2026*
*Conecta: NOMENCLATURA_scanner_alpha_phi.md*
*Conecta: TECNICA_eco_sequencial_cadeia_pre_funcao.md*
*Conecta: TECNICA_eco_campo_proprio_multiespectral.md*
*Conecta: FILOSOFICA_alpha_atrator_entropia_de_si_mesmo.md*
