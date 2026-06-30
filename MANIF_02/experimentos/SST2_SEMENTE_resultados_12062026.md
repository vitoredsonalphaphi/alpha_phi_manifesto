# SST2 Semente Alpha — Resultados
**AlphaPhi_SST2_Semente_COLAB.py · 12/06/2026**

---

## Configuracao

- Substrato: SST2 texto, embeddings MiniLM-L6-v2 (384 dims)
- Arquitetura: [384, 55, 89, 144, 1] — Fibonacci layers
- Semente: `an = clip(|h|/sum, alpha, 1.0)` + `H/log(137)` nas ativacoes ocultas
- Beta textual: `beta_text = phi^(3 * coh_alpha_media)`
- Eco-entropico (CC): H_alpha drive lr adaptativo

---

## Resultados

| Config | Acc | Std | Delta | H_eq | beta_max | Campo |
|--------|-----|-----|-------|------|----------|-------|
| AA baseline | 0.7493 | 0.0417 | — | 0.000 | 0.000 | 0/10 |
| BB semente | 0.7493 | 0.0417 | +0.000 | **0.9987** | **1.0039** | 0/10 |
| CC eco-entrop | 0.7510 | 0.0430 | +0.002 | 0.9988 | 1.0039 | 0/10 |

- BB vs AA: p=1.000 (ns) — semente nao muda acuracia
- CC vs AA: p=0.932 (ns) — eco-entropico nao muda acuracia
- R: AA=0.036  BB=0.036  CC=0.012 ← eco-entropico reduziu R (feedback ativo)

---

## Diagnostico — Por que o campo nao formou

```
Camada oculta: 144 dimensoes
Regua alpha:   log(137) = 4.920
H de distribuicao uniforme em 144 dims: log(144) = 4.969
H_alpha = log(144)/log(137) = 1.010 → clipado a 1.000
coh_alpha = 1 - 1.000 = 0.000
beta_text = phi^(3 * 0.000) = phi^0 = 1.000
```

A regua alpha (log 137) esta saturada para uma camada de 144 dimensoes.
Qualquer distribuicao nao-trivial da H_alpha ≈ 0.999 → coh_alpha ≈ 0.001.
O campo nao tem gradiente para subir.

No audio: bandas-phi criam ESTRUTURA que alpha pode medir
  (cada banda tem N_bins diferente → variancia real em H_alpha entre bandas).
Na rede FC: as 144 dimensoes sao simetricas
  → sem estrutura phi interna → alpha nao diferencia.

---

## O que foi confirmado

**Eco-entropico funcionou** (R caiu 0.036→0.012 em CC):
O feedback de H_alpha sobre lr foi ativo e reduziu R.
O mecanismo opera — o substrato e que nao forneceu gradiente de coerencia.

---

## Caminho identificado

Para campo harmonico textual emergir:

**Opcao 1 (imediata):** hidden = 137 exato
  - Arquitetura: [384, 55, 89, **137**, 1]
  - Camada oculta fica no espaco nativo de alpha
  - log(137)/log(137) = 1.0 → H_alpha tem range real [0, 1]

**Opcao 2 (estrutural):** bandas-phi na rede
  - Agrupar as 144 dimensoes em grupos de tamanho phi-proporcional
  - Cada grupo = uma "banda" como no audio
  - Medir H_alpha por grupo → variancia real → campo pode formar

---

## Comparacao com audio

| Substrato | H_alpha_eq | Campo | beta_max |
|-----------|-----------|-------|---------|
| Audio 880Hz (semente) | 0.862 | SIM c10 | 4.236 |
| Ruido branco (semente) | 0.902 | SIM c10 | 4.236 |
| Texto SST2 hidden=144 | 0.999 | NAO | 1.003 |
| Texto SST2 hidden=137 | ? | ? | ? |

---

*AlphaPhi_SST2_Semente_COLAB.py · Commit e7754ee*
*MANIF_02/experimentos/ · 12/06/2026*
