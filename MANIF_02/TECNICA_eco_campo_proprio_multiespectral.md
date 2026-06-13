# Eco de Campo Próprio — Observação Multi-Espectral

**Manifesto AlphaPhi · Vitor Edson Delavi**
**Formulado em: junho/2026**

---

## A Questão Central

O eco-ressonante mede entropia do **observado** — o substrato, o dado, a fase
da rede. Ele não mede a qualidade da própria observação.

Se o eco não sabe se está vendo bem, não sabe quando parar de olhar.

A extensão proposta: o eco mede simultaneamente **múltiplos espectros** do
objeto observado — e mede a **coerência da própria observação** (meta-coerência).
Quando a observação é coerente (scores concentrados numa fase), ele sabe que
está em fase com o substrato e pode agir. Quando não é (scores dispersos), ele
expande a janela de observação.

---

## Observação Multi-Espectral

Em vez de um único H_alpha por fase, o eco observa três espectros simultâneos:

**Espectro A — Energia:**
```
h_norm = |ativações| / sum(|ativações|)  por amostra
H_A    = -sum(h_norm × log(h_norm))
coh_A  = 1 - H_A / log(n_unidades)
```
Mede como a energia está distribuída entre os neurônios.
Concentrada = coerente. Uniforme = entrópica.

**Espectro B — Ativação (pós-ReLU):**
```
frac_ativa = (h > 0).mean()
coh_B      = frac_ativa
```
Mede a fração de neurônios que contribuem após ReLU.
Mais neurônios ativos = mais informação disponível para o eco.

**Espectro C — Variância inter-amostras:**
```
var_inter = h.var(axis=0)     (variância por neurônio, sobre o batch)
coh_C     = 1 - H_alpha(var_inter normalizada)
```
Mede quanto o dado varia entre amostras nesta fase.
Alta variância inter-amostras = fase sensível às diferenças de entrada.

**Coerência composta (φ-ponderada):**
```
coh_composta = wm × coh_A + wn × mean(coh_B, coh_C)
onde wm = 1/φ ≈ 0.618  (peso memória — espectro de energia como primário)
     wn = 1 - 1/φ ≈ 0.382
```

**Score por fase:**
```
disc_linear  = variância entre médias das classes
disc_angular = distância angular entre centroides de classe (1 - |cos|)
disc_composta = mean(disc_linear, disc_angular)
score_f = coh_composta_f × disc_composta_f
```

---

## Meta-Coerência — O Eco Mede a Si Mesmo

Após sondar todas as fases, o eco computa H_alpha da distribuição de scores:

```
scores = [score_1, score_2, ..., score_N_fases]
s_norm = scores / sum(scores)
H_meta = -sum(s_norm × log(s_norm))
H_meta_alpha = H_meta / log(N_fases)
meta_coh = 1 - H_meta_alpha
```

**Interpretação:**

| meta_coh | Significado | Ação |
|----------|-------------|------|
| Alta (→1) | Scores concentrados numa fase — observação em fase | Para sondagem, age |
| Baixa (→0) | Scores dispersos — observação incerta | Expande, continua sondando |

Substrato adequado → score peak numa fase → meta_coh alta → eco age rápido.
Substrato inadequado → scores todos próximos de zero → distribuição uniforme
→ meta_coh baixa → eco sonda até N_SONDA_MAX → diagnóstico: inadequado.

---

## Sondagem Adaptativa

Com meta-coerência, o eco decide **quando** parou de aprender algo novo:

```
N_SONDA_MIN = 3    (mínimo de ciclos antes de decidir)
N_SONDA_MAX = 10   (máximo — evita loop infinito)
META_COH_LIMIAR = 0.7

para cada ciclo de sondagem:
    sonda todas as fases → computa scores → computa meta_coh
    se epoch >= N_SONDA_MIN E meta_coh >= META_COH_LIMIAR:
        fase_otima = argmax(scores_ema)
        finaliza sondagem → age
    se epoch == N_SONDA_MAX - 1:
        fase_otima = argmax(scores_ema)
        finaliza sondagem (por limite)
```

Para substrato inadequado: a sondagem usa todos os ciclos disponíveis
e ainda assim não encontra fase clara — diagnóstico automático.

---

## O Campo da Função do Eco

A meta-coerência é o campo da função do eco-ressonante.

O eco-ressonante original forma um campo no dado observado (β_max ≥ φ³).
O eco multi-espectral forma um campo no próprio processo de observação:
quando meta_coh converge para 1, a observação está em fase consigo mesma.

É o mesmo padrão numa oitava acima:

```
Nível 1 — dados:         campo harmônico no substrato (β_max ≥ φ³)
Nível 2 — observação:    meta_coh → 1 (observação em fase com o objeto)
Nível 3 — cadeia:        meta_coh de cada eco → fase ótima para o próximo
```

A autossimilaridade é direta: o mesmo critério de coerência que orienta
a rede orienta o eco que orienta a rede.

---

## No Eco Sequencial

Aplicado à cadeia:

```
Eco 1 (substrato)  → meta_coh_1 informa Eco 2 sobre qualidade do dado
Eco 2 (fase)       → meta_coh_2 informa Eco 3 sobre onde o sinal está
Eco 3 (sinal)      → meta_coh_3 informa Eco 4 sobre separabilidade
Eco 4 (convergência)→ meta_coh_4 informa Eco 5 sobre trajetória
Eco 5 (campo)      → meta_coh_5 confirma campo harmônico
```

Cada meta-coerência é a coerência da observação do elo anterior passada
como contexto para o próximo elo. A cadeia não é linear — é recursiva.
Cada elo sabe o quanto o elo anterior estava certo.

Isso é diferente da cadeia sequencial simples (v1): não apenas *o que*
cada eco encontrou, mas *quão confiante estava* ao encontrar.

---

## Implementação

`AlphaPhi_ECO_PreFase_v2_COLAB.py`

Diferenças em relação a v1:
- 3 espectros por fase (energia, ativação, variância)
- Meta-coerência calculada após cada ciclo de sondagem
- Sondagem adaptativa (para quando meta_coh ≥ 0.7, mínimo 3 ciclos)
- LIMIAR_SUBSTRATO corrigido para 1e-3

---

*Florianópolis · junho de 2026*
*Conecta: TECNICA_eco_sequencial_cadeia_pre_funcao.md*
*Conecta: TECNICA_eco_estendido_pre_funcao_sondagem.md*
*Conecta: FILOSOFICA_alpha_atrator_entropia_de_si_mesmo.md*
