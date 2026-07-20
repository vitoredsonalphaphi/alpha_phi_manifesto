# PROTOCOLO ECO-φ AUDIO — Versão 1
## Scanner → Mapa → Modulação Calibrada por Substrato

**Manifesto Alpha-Phi · Segundo Ciclo**
**Instrumento:** Scanner α-φ + ECO-φ Calibrado
**Substrato:** Voz Sintética (Gemini TTS) · extensível a voz orgânica
**Florianópolis · 20 de Julho de 2026 · Sessão Good Morning**

---

## Princípio

O campo harmônico emerge quando a modulação respeita o território do substrato — não quando é imposta sobre ele.

O Scanner lê primeiro. O ECO-φ segue o que o Scanner encontrou. Não o contrário.

---

## Sequência do Protocolo

```
FASE 1 — OBSERVAR
  Scanner α-φ sobre sinal ORIGINAL (sem tocar)
  → coh_map[n_bandas × n_frames] — mapa fixo do substrato

FASE 2 — MAPEAR
  Visualizar coh_map: identificar onde estão grupos de alta coerência
  (vogais = coh alta, consoantes = média, silêncios = baixa)
  Verificar coerência média por banda antes de prosseguir

FASE 3 — MODULAR
  ECO-φ com calibragem fixa do Scanner
  5 passagens — cada frame modulado conforme seu coh do mapa
  β = φ³ fixo (atrator já conhecido)
  amp = min(coh × φ^β, 1/φ) — amplitude travada no complementar áureo

FASE 4 — INTEGRAR
  Blend: 0,75 × eco + 0,25 × original
  Verificar resultado auditivo e visual
```

---

## Parâmetros (Voz Sintética FS=16000Hz)

| Parâmetro | Valor | Justificativa |
|---|---|---|
| FS | 16000 Hz | taxa de amostragem alvo |
| N_FFT | 1024 | resolução espectral |
| HOP | 512 (N_FFT/2) | overlap 50% |
| MIN_BINS | 6 | mínimo de bins por banda φ |
| Bandas φ | 9 (129–8000 Hz) | geométricas: cada banda = φ × anterior |
| β | φ³ = 4,2361 | atrator do ECO-φ — fixo, já convergido |
| amp_max | 1/φ = 0,618 | complementar áureo — env ∈ [0,382; 1,618] |
| N_PASSES | 5 | suficiente para voz (100 causa "voz de vidro") |
| Blend | 0,75 eco + 0,25 original | preserva clareza de fala |

---

## Cabeçalho Padrão para Código

Todo código de modulação deve identificar seu instrumento e substrato:

```python
# ═══════════════════════════════════════════════════════════════════════
# MANIFESTO ALPHA-PHI · Segundo Ciclo
# Instrumento : Scanner α-φ + ECO-φ Calibrado por Substrato
# Substrato   : Voz Sintética (Gemini TTS) · FS=16000Hz
# Protocolo   : Observar → Mapear → Modular
# Parâmetros  : β=φ³, N_PASSES=5, amp_max=1/φ, blend=0.75
# Sessão      : Good Morning · 20.07.2026
# ═══════════════════════════════════════════════════════════════════════
```

---

## Diagnóstico de Resultados

| Sintoma auditivo | Causa | Correção |
|---|---|---|
| "Voz de vidro" | β explodiu (β >> φ³) | Cap β em φ³; amp_max=1/φ |
| "Eco no cano" | Fase STFT inconsistente | Blend com original |
| Sem mudança perceptível | amp muito pequena ou N_PASSES=1 | Aumentar N_PASSES; verificar coh_map |
| Saturação/brilho persistente | AGC da Gemini sobrepõe ECO-φ | Modular diretamente Banda 8 (6081-8000Hz) antes do ECO-φ |
| "Redondo", "polifônico" | ECO-φ redistribuindo harmônicos corretamente | Prosseguir — campo formando |

---

## Critério de Convergência

**Campo harmônico em formação:** "redondo", "polifônico", "menos ácido", ausência de headache
**Campo harmônico emergido:** expansão sensorial análoga ao EcoBIP 880Hz — terceira estrutura sensorial inequívoca

O EcoBIP 880Hz atingiu emergência completa com 1 fonte (senoide pura).
A voz sintética tem ~25 fontes simultâneas variando por frame.
A emergência em voz é gradual — convergência assintótica, não pontual.

---

## Isomorfismo Visual/Auditivo

Quando o mapa do Scanner (coh_map visualizado como heatmap inferno) parece esteticamente reorganizado — a mesma reorganização está presente no áudio.

Não é coincidência: são dois domínios de representação da mesma estrutura de campo φ-ressonante. A reorganização do campo é única — suas representações em diferentes domínios percebem a mesma coisa.

Este isomorfismo é evidência de campo: quando o visual e o auditivo convergem para a mesma percepção de ordem, a estrutura que os gerou é real.

---

## Extensão a Outros Substratos

**Voz orgânica:** Mesmo protocolo. Coerência naturalmente mais alta e variada (sem AGC). Resultado esperado: emergência mais rápida e mais clara.

**Outros áudios:** Ajustar FS, N_FFT conforme taxa de amostragem. Bandas φ se recalculam automaticamente.

**Vídeo:** Cada frame de vídeo = sinal 2D. Bandas φ = bandas de frequência espacial (wavelets φ-escaladas). Scanner lê coerência espacial. ECO-φ redistribui energia espacial por banda. A "espessura de cada grupo de barras" torna-se "densidade de estrutura espacial por região".

---

*Florianópolis · 20.07.2026 · Sessão Good Morning*
