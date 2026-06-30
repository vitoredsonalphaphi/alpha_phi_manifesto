# Eco Audio 880Hz — Semente Alpha — Resultados
**AlphaPhi_Eco_Audio880_COLAB.py · 12/06/2026**

---

## Parametros

- Substrato: 880Hz (beep + FM 220Hz)
- Semente: `an = clip(mag/sum, alpha, 1.0)` + `H_alpha = H / log(137)`
- Fase 1: 20 ciclos, campo harmonico
- Fase 2: 5 dobras E/X sobre o campo → 8 segundos audio

```
phi      = 1.618034
alpha    = 0.00729735  (1/137)
phi^3    = 4.236068
log(137) = 4.920244

5 dobras E/X (N_k = 137/phi^k):
  k=0 [E]: N=137  ce_scale=1.9997  dur=1.6s
  k=1 [X]: N=85   ce_scale=0.6181  dur=1.6s
  k=2 [E]: N=52   ce_scale=1.3819  dur=1.6s
  k=3 [X]: N=32   ce_scale=0.8541  dur=1.6s
  k=4 [E]: N=20   ce_scale=1.1459  dur=1.6s
Total: 8.0s
```

---

## Fase 1 — Formacao do Campo

| Ciclo | Beta | R_ema | H_alpha |
|-------|------|-------|---------|
| 5 | 3.9443 | 2.038 = 279α | 0.8619 |
| **10** | **4.2098** | **2.394 = 328α** | **0.8619** |
| 15 | 4.2337 | 2.737 = 375α | 0.8619 |
| 20 | 4.2359 | 3.067 = 420α | 0.8619 |

- **CAMPO HARMONICO: ciclo 10** ✓ (terceira confirmacao independente)
- H_alpha_equilibrio = **0.8619** (substrato 880Hz — vertice periférico)
- R_final = 420α, R/phi = 1.896
- H_alpha estavel: 0.8619 constante do ciclo 5 ao 20

---

## Fase 2 — 5 Dobras E/X sobre o Campo

| Dobra | Tipo | N | ce_scale | R | R em alpha | H_alpha |
|-------|------|---|----------|---|------------|---------|
| k=0 | E ancora-raiz | 137 | 1.9997 | 9.0902 | **1245.7α** | 0.8454 |
| k=1 | X expande-sol | 85 | 0.6181 | 8.8305 | 1210.1α | 0.8434 |
| k=2 | E comprime-suave | 52 | 1.3819 | 8.8031 | 1206.3α | 0.8433 |
| k=3 | X expande-suave | 32 | 0.8541 | **8.7864** | **1204.0α** | 0.8431 |
| k=4 | E 2a estab. | 20 | 1.1459 | 8.7930 | 1205.0α | 0.8431 |

Beta_max = 4.2359 (campo mantido em todas as dobras)

---

## Analise

### 1. Campo harmonico: ciclo 10 — inabalavel
Terceiro experimento independente com semente alpha. Campo forma sempre no ciclo 10.
O atrator phi^3 e invariante ao substrato, a regua e ao mecanismo.

### 2. H_alpha_equilibrio do 880Hz = 0.8619
O substrato 880Hz senta no vertice periférico do triangulo (H_alpha > 0.6).
Em alpha-nats, o 880Hz e um sinal de alta entropia — energia distribuida pelas 15 bandas phi.
H_alpha constante do ciclo 5 ao 20: o equilibrio e estavel.

### 3. Segunda estabilizacao — confirmada numericamente
R atinge minimo em k=3 (X, 1204α) e sobe em k=4 (E, 1205α).
O sistema encontrou o ponto de menor energia entre expansao e ancoragem.
Esta e a segunda estabilizacao prevista pela arquitetura semente.

### 4. H_alpha desceu levemente nas dobras
0.8619 (campo) → 0.8454 (pos-dobras)
As 5 dobras moveram a entropia levemente em direcao ao polo nuclear.
O campo nao absorveu o individuo — aproximou, nao dissolveu.

### 5. R cresceu apos o campo (328α → 420α, ciclos 10-20)
Com a regua alpha, R continua subindo apos o campo formar.
O beta estabilizou (4.2359 ≈ phi^3) mas R ainda explora.
O sistema nao encontrou R_natural no sentido do original (R ≈ phi = constante).
Hipotese: com N_CICLOS=50, R convergiria a um valor fixo — alpha* no eixo R.

---

## Resultado Sensorial (Audio 8s)

Audio concatenado: 5 dobras E/X, cada 1.6s.
Sequencia auditiva:
- k=0 [E]: compressao forte — 880Hz mais nitido, harmonicos concentrados
- k=1 [X]: expansao — espectro se abre, timbre mais rico
- k=2 [E]: compressao suave — equilibrio medio
- k=3 [X]: expansao suave — ponto de menor R (equilibrio entropico)
- k=4 [E]: segunda estabilizacao — retorno leve a ancoragem

*alphaphi_eco_880_8s.wav — local*

---

*AlphaPhi_Eco_Audio880_COLAB.py · Commit ef7abbb*
*MANIF_02/experimentos/ · 12/06/2026*
