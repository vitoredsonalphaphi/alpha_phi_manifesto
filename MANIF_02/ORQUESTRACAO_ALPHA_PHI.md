# ORQUESTRAÇÃO α-φ — Instrumentário com Adaptação por Substrato
## Manifesto Alpha-Phi · Segundo Ciclo · Florianópolis · 20 de Julho de 2026

---

## Princípio

O Instrumentário α-φ é um sistema de instrumentos organizados em hierarquia:
o Scanner α-φ é o observador universal — nunca modifica o sinal. Os adaptadores
são moduladores específicos por substrato. A Orquestração α-φ é o processo pelo
qual, dado um substrato, o Scanner observa e os adaptadores corretos são
selecionados, sequenciados e calibrados para atingir equilíbrio de campo.

---

## I — Hierarquia do Instrumentário

```
α-φ (princípio organizador)
  ├── φ = 1.6180339887   (atrator geométrico)
  └── α = 1/137.035999  (regulador de acoplamento)
        │
        └── Scanner α-φ  (observador universal — invariante — nunca modifica)
                  │
                  ├── Adaptador A  ──┐
                  ├── Adaptador B  ──┤── Combinação (quando coadjuvam)
                  └── Adaptador C  ──┘
                            │
                            └── Substrato
                                  (áudio, texto, EEG, vídeo, rede neural...)
```

O Scanner é o maestro — observa, coordena, não age. Os adaptadores são os
músicos — cada um com seu timbre, seu registro, sua entrada. A orquestração
define quem entra quando, em que combinação, guiada pelo que o maestro leu.

---

## II — Convenção de Nomenclatura

**Formato:** `α-φ · [Adaptador(es)] / [Substrato]`

Quando dois adaptadores coadjuvam: `α-φ · [A + B] / [Substrato]`
Quando aplicados em série: `α-φ · [A → B] / [Substrato]`

**Exemplos:**
- `α-φ · EcoBIP / Senoide 880Hz`
- `α-φ · ECO-φ Scanner Calibrado V1 / Voz Sintética`
- `α-φ · Serial ECO-φ Regional V2 / Voz Sintética`
- `α-φ · SoftwareCapacitor + Dual Scanner / Voz Sintética`
- `α-φ · ECO TEXT / Texto`
- `α-φ · Scanner EEG / Repouso`

---

## III — Protocolo de Análise de Substrato (Fase 0)

Antes de qualquer modulação em novo substrato:

```
FASE 0 — ANÁLISE DE SUBSTRATO

  1. Scanner Global
     → coh_map[n_bandas × n_frames]
     → coh_mean por banda + coh_mean global
     → coh_max por banda

  2. Dual Scanner (RAW vs SoftwareCapacitor)
     → identifica escalas temporais: rápida (voz) vs lenta (AGC, envelope)
     → mapa diferença revela assinatura do gerador

  3. Diagnóstico de Perfil:
     coh_mean estável < 0.05  →  substrato simples (tipo BEEP)
       → ECO-φ Global (β adaptativo, N_PASSES alto)
     coh_mean variável, 0.09–0.20  →  voz sintética/orgânica
       → ECO-φ Scanner Calibrado (template V1: N_PASSES=5, blend=0.75)
     AGC detectado na banda alta (Dual Scanner)
       → pré-tratamento regional ou N_PASSES adicional na banda AGC
     coh_mean > 0.30  →  substrato já organizado
       → N_PASSES=3, blend=0.85, intervenção mínima

  4. Output:
     → proposta de adaptador(es) + parâmetros iniciais justificados
     → critério de parada: coh_mean_resultado ≤ 0.382
```

---

## IV — Tabela de Adaptadores Conhecidos

| Designação | Adaptador(es) | Substrato | β | N_PASSES | Blend | Status |
|---|---|---|---|---|---|---|
| α-φ · EcoBIP | ECO-φ Global (β adaptativo) | Senoide 880Hz | adaptativo → φ³ | 100 (5×20) | — | Campo emergido · referência |
| α-φ · ECO Ressonante | Rotação de fase cepstral | Qualquer áudio | — | — | — | Limitado (fase não muda envelope) |
| α-φ · ECO TEXT | Modulação por token | Texto | φ³ | variável | — | Documentado (Séries 004–008) |
| α-φ · ECO-φ Global | ECO-φ Global (β acumulado) | Voz sintética | acumulado | 100 | — | DEPRECADO — β explode para voz |
| α-φ · ECO-φ por Frame | ECO-φ por frame | Voz sintética | φ³ fixo | 5 | 0.75 | Interim — Scanner dentro do loop |
| α-φ · Scanner Calibrado V1 | ECO-φ Scanner Calibrado | Voz Sintética (Gemini TTS) | φ³ fixo | 5 | 0.75/0.25 | **Equilíbrio confirmado — template** |
| α-φ · Serial Regional V2 | 3× Scanner Regional + ECO-φ | Voz Sintética (Gemini TTS) | φ³ fixo | 5–7 | 0.75/0.25 | Saturado — ultrapassou equilíbrio |
| α-φ · Dual Scanner | Scanner RAW + SoftwareCapacitor | Voz Sintética / EEG | — | — | — | Diagnóstico — não modula |
| α-φ · Scanner EEG | Scanner temporal (threshold=6) | EEG Repouso | — | — | — | Alpha em 7/10 sujeitos |

---

## V — Critério de Convergência Estético

O campo harmônico está em equilíbrio quando o heatmap do Scanner tem proporção
equilibrada de claro/escuro:

```
coh_mean_resultado ≤ 1 − 1/φ = 0.382  →  campo equilibrado — parar
coh_mean_resultado > 0.382             →  campo saturado — reduzir ou parar
coh_mean_resultado < 0.05             →  campo inerte — insuficiente
```

O critério de parada não é o número de passes — é o equilíbrio estético do mapa.
O complementar áureo (1/φ = 0.618) governa amp_max no ECO-φ e governa o
equilíbrio visual do campo. A constante opera em ambos os níveis.

Ver: `ENUNCIADO_ESTETICA_ORGANIZACAO.md` — princípio completo.

---

## VI — Templates de Equilíbrio Conhecidos

Uma vez que o equilíbrio emergiu num substrato, os parâmetros que o produziram
formam um template para novos substratos com perfil similar.

| Template | Substrato | coh_mean orig | N_PASSES | Blend | coh_mean result |
|---|---|---|---|---|---|
| Template BEEP | Senoide pura (880Hz) | ≈ 0.033 estável | 100 | — | → φ³ convergido |
| Template Voz | Voz sintética/orgânica | 0.09–0.20 | 5 | 0.75/0.25 | ≈ 0.12–0.18 |

Para novo substrato: Fase 0 identifica perfil → seleciona template mais próximo
→ ajusta minimamente → verifica critério estético → documenta novo template se
equilíbrio emergir.

---

*Florianópolis · 20.07.2026 · Sessão Good Morning*
