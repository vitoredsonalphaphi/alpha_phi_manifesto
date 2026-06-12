# Eco Beep 880Hz — Semente Alpha — Resultados
**AlphaPhi_Eco_Beep880_Semente_COLAB.py · 12/06/2026**

---

## Parametros

- Substrato: 880Hz (beep + FM 220Hz), alpha_sinal=1/3
- Semente: `an = clip(mag/sum, alpha, 1.0)` + `H/log(137)` = coh_alpha
- Ablacao: AA (N=0) | BB (N=14) | CC (N=137 obs+rol) | DD (N=137 desde c1)

---

## Resultados

| Config | Campo | Ciclo | R_final | R/phi | coh_med |
|--------|-------|-------|---------|-------|---------|
| AA sem roldana | SIM | c10 | 2.392 = **327.9α** | 1.4786 | 0.1819 |
| BB N=14 | SIM | c10 | 2.393 = 327.9α | 1.4787 | 0.1818 |
| CC N=137 obs+rol | SIM | c10 | 2.394 = 328.1α | 1.4796 | 0.1812 |
| DD N=137 desde c1 | SIM | c10 | 2.395 = 328.2α | 1.4802 | 0.1812 |

Referencia historica (coh_raw): R_natural = phi = 1.618 = 221.7α (ratio=1.0)
Com semente alpha (coh_alpha): R_natural -> 328α (ratio=1.479)

---

## Analise

### 1. Campo: ciclo 10 — confirmado (4a vez independente)
AA/BB/CC/DD: todas no ciclo 10. Nenhuma variacao.

### 2. AA = BB = CC = DD (diferenca de 0.3alpha — ruido)
As roldanas phi-alternadas NAO diferenciam com coh_alpha.
No original (coh_raw): CC e DD diferenciavam visivelmente de AA.
Com semente: a diferenca AA→DD e apenas 0.3alpha.

### 3. Por que as roldanas nao diferenciam?
coh_alpha medio = 0.181 para todas as bandas — gradiente quase nulo.
No substrato 880Hz, o floor alpha (1/137) aplaina a distribuicao:
bandas sem sinal: mag≈0 → an uniforme → H=log(N_bins) → coh_alpha baixo
bandas com sinal: mag concentrado → H baixo → coh_alpha alto
Mas o mean e ~0.18 em todas — sem variancia suficiente para roldana atuar.

### 4. R crescendo sem convergir (254α → 280α → 304α → 328α)
R nao convergiu em 20 ciclos. O EMA (tau=1/alpha=137 ciclos) precisa de N_CICLOS≈137 para encontrar R_natural.
Estimativa: R_natural semente 880Hz ≈ 4.5-5.0 (616-685α) se rodado com N_CICLOS=100.

### 5. coh levemente menor em CC/DD (0.1812 vs 0.1819)
A roldana N=137 reduz levemente a coerencia media — move a entropia
levemente em direcao ao polo periférico. Efeito real, mas marginal.

---

## Diagnostico — O que a semente revelou

O eco-ressonante original funcionava porque coh_raw criava gradiente ENTRE bandas:
- Bandas com energia: coh_raw alta (0.6-0.9)
- Bandas sem energia: coh_raw baixa (0.1-0.3)
- Gradiente: permitia roldana agir diferentemente por banda

Com coh_alpha, o floor alpha comprime o gradiente:
- Bandas ricas: coh_alpha ~ 0.8 (H_alpha baixo)
- Bandas pobres: coh_alpha ~ 0.1 (H_alpha alto)
- Mean ~ 0.18 — gradiente ainda existe mas e assimetrico

O atrator phi^3 e mais forte que o gradiente comprimido.

---

## Proximo passo identificado

Para roldanas atuarem com coh_alpha: usar gradiente relativo por ciclo
em vez de absoluto. Ja presente no beta-update (`cr = coh normalizado`),
mas nao no ce_eff. Proposta:

```python
# em vez de: ce_eff = ce * ce_scale
# usar:      ce_eff = ce_relativo * ce_scale
# onde:      ce_relativo = (coh - min_coh) / (max_coh - min_coh + 1e-10)
```

Ou: rodar N_CICLOS=100 para encontrar R_natural real com semente.

---

*AlphaPhi_Eco_Beep880_Semente_COLAB.py · Commit 6756682*
*MANIF_02/experimentos/ · 12/06/2026*
