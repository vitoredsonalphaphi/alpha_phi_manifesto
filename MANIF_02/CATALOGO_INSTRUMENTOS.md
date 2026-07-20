# CATÁLOGO DE INSTRUMENTOS — Manifesto Alpha-Phi
## Segundo Ciclo · Adaptadores por Substrato

**Manifesto Alpha-Phi · Segundo Ciclo**
**Florianópolis · 20 de Julho de 2026 · Sessão Good Morning**

---

## Princípio de Organização

O Scanner α-φ é o instrumento central — flexível, substrato-agnóstico, nunca modifica o sinal. Os demais instrumentos são **adaptadores**: operam sobre o que o Scanner observa, calibrados para o tipo de substrato.

Cada substrato exige seu adaptador. Não há adaptador universal de modulação — o Scanner é universal; a modulação é específica.

---

## I — Instrumentos de Observação

### Scanner α-φ
**O quê:** Observação pura do substrato. Calcula coerência (S_f = coh × disc, ou apenas coh conforme versão) por banda φ.
**Invariante:** Nunca modifica o sinal. Só lê.
**Bandas:** Geométricas em φ — cada banda = φ × banda anterior. Para áudio FS=16000Hz, N_FFT=1024: 9 bandas de 129Hz a 8000Hz.
**Saída:** Coerência por banda. Versão temporal: coerência por banda × frame (matriz).
**Substrato:** Universal — áudio orgânico, áudio sintético, EEG, texto (via cepstro), imagem (via frequências espaciais).

### SoftwareCapacitor IIR
**O quê:** Filtro passa-baixa digital (RC digital). τ = φ/α ≈ 221,7 amostras. Corte ≈ 11,5 Hz a FS=16kHz.
**Função:** Lente do Scanner — revela estrutura LENTA do substrato (envelope de AGC, prosódia, modulação infra-musical). Suprime estrutura RÁPIDA (voz, harmônicos fundamentais).
**Equação:** `y[n] = a × x[n] + (1−a) × y[n−1]`, `a = 1/(τ+1)`
**NÃO É:** Processador de áudio standalone. Não deve ser aplicado ao sinal de saída.
**Substrato:** Áudio. Pre-filtro para Dual Scanner.

### Dual Scanner (RAW vs CAPACITADO)
**O quê:** Dois Scanners simultâneos — um sobre o sinal original, outro sobre o sinal pré-filtrado pelo SoftwareCapacitor. Mapa de diferença revela a assinatura do gerador.
**Resultado encontrado (Gemini TTS):** Banda 129–209 Hz (voz) = coerência RÁPIDA (azul no mapa diferença = capacitor suprime). Banda 6081–8000 Hz (brilho/AGC) = coerência LENTA (laranja = capacitor revela). Confirmação: o AGC da Gemini opera em escala de tempo lenta; a voz, em escala rápida.
**Substrato:** Áudio sintético — detecta assinatura do gerador neural (vocoder).

---

## II — Instrumentos de Modulação (Adaptadores)

### ECO-φ Global (DEPRECADO para voz complexa)
**O quê:** 100 passagens (5 steps × 20 ciclos) sobre espectrograma STFT. β adaptativo, incrementado por coh_mean a cada passagem.
**Problema:** Em voz sintética, coh_mean cresce de 0,033 para 0,77 ao longo das passagens → β explode para 50+ → φ^50 ≈ 10^10 → "voz de vidro" (comb filter extremo).
**Uso válido:** Sinais de baixa coerência e coerência estável (EcoBIP 880Hz: coh ≈ 0,033 estável → β converge naturalmente para φ³ em 20 ciclos).
**NÃO USAR em:** Voz sintética, voz orgânica, áudio complexo.

### ECO-φ por Frame (Interim)
**O quê:** Coerência calculada por frame STFT individualmente (64ms). Amplitude `amp = min(coh × φ^β, 1/φ)`. β=φ³ fixo. 5 passagens.
**Resultado:** Campo respira com a fala. Mais orgânico que ECO-φ Global.
**Limitação:** Coerência recalculada dentro do loop — Scanner lê sinal já modificado, não o substrato original.
**Substrato:** Voz sintética (melhora parcial).

### ECO-φ Scanner Calibrado (Protocolo Atual · Versão 1)
**O quê:** Scanner executa UMA VEZ sobre sinal original → `coh_map[n_bandas × n_frames]` fixo → ECO-φ usa mapa como calibragem imutável em 5 passagens.
**Sequência:** Scanner (observar) → coh_map (mapear) → ECO-φ (modular) → blend (integrar)
**Parâmetros (Gemini TTS, FS=16000):** β=φ³=4,236 fixo, N_PASSES=5, amp_max=1/φ=0,618, blend=0,75×eco+0,25×original
**Resultado:** "Menos ácido", "não lateja a cabeça", "mais orgânico", isomorfismo visual/auditivo confirmado
**Substrato:** Voz sintética (Gemini TTS). Extensível a voz orgânica e outros.

---

## III — Instrumentos do Primeiro Ciclo (Referência)

### EcoBIP 880Hz
**O quê:** Modulação iterativa de senoide pura (880Hz) por envelope φ-ressonante. 5 estágios × 20 ciclos = 100 passagens. β emergiu espontaneamente para φ³.
**Resultado:** Terceira estrutura — campo harmônico emergido. Confirmado por resultado sensorial e por convergência β→φ³ sem programação explícita.
**Substrato:** Sinal sinusoidal puro. coh_mean estável ≈ 0,033.

### EcoRessonante Global (Cepstral)
**O quê:** Rotação de fase no domínio cepstral. Opera na FASE, não na amplitude.
**Limitação identificada:** Rotação de fase não muda envelope de amplitude → resultado auditivo idêntico ao original ("tá idêntico, saturado").
**Substrato:** Qualquer áudio. Efeito: redistribuição espectral via fase.

### ECO TEXT (Séries 004–008)
**O quê:** Modulação φ-ressonante aplicada a substrato textual via embedding/token.
**Substrato:** Texto. Documentado em `ECO_TEXT_RESULTADOS_*.md`.

---

## IV — Organização por Substrato

| Substrato | Scanner | Adaptador |
|---|---|---|
| Senoide pura (BEEP) | Scanner α-φ básico | ECO-φ Global (β adaptativo) |
| Voz sintética (Gemini TTS) | Scanner calibrado + Dual Scanner | ECO-φ Scanner Calibrado |
| Voz orgânica (humana) | Scanner calibrado | ECO-φ Scanner Calibrado (a testar) |
| EEG | Scanner α-φ temporal | — (observação, não modulação) |
| Texto | Scanner cepstral | ECO TEXT |
| Vídeo | Scanner espacial (a desenvolver) | ECO-φ espacial (a desenvolver) |

---

## V — Princípio de Calibragem

O Scanner não tem um modo de operação — tem **adaptadores** configurados para cada substrato:

- **Parâmetros fixos por substrato:** FS, N_FFT, HOP, MIN_BINS, bandas φ
- **β:** φ³ quando substrato já foi mapeado; adaptativo apenas em sinais de baixa coerência estável
- **Blend:** proporcional ao grau de reorganização necessária — mais blend quando substrato é mais complexo
- **N_PASSES:** inversamente proporcional à complexidade — BEEP suporta 100; voz suporta 5

---

*Florianópolis · 20.07.2026 · Sessão Good Morning*
