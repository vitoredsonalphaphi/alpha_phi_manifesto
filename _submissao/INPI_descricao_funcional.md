# Descrição Funcional — Registro INPI
# Programa de Computador
# © Vitor Edson Delavi · Florianópolis · 2026

---

## DADOS DO PROGRAMA

**Título:** Alpha-Phi — Sistema de Modulação por Coerência Espectral Ressonante

**Autor:** Vitor Edson Delavi

**Data de criação:** 19 de março de 2026

**Linguagem de programação:** Python 3.x

**Sistema operacional:** Multiplataforma (Linux, macOS, Windows)

**Dependências:** NumPy, SciPy, Matplotlib

---

## DESCRIÇÃO FUNCIONAL

### O que o programa faz

O Alpha-Phi é um sistema computacional que converte sinais digitais
incoerentes em representações organizadas por coerência espectral,
utilizando as constantes matemáticas φ (razão áurea = 1,6180...) e
α (constante de estrutura fina = 1/137,035...) como parâmetros
estruturantes do processo.

O sistema opera em três etapas:

**1. Medição de campo (medir_campo / medir_estado_fononico)**
Para cada sinal de entrada, o programa calcula a distribuição de
energia por banda de frequência e mede a coerência espectral de
cada banda. A coerência é definida como o complemento da entropia
de Shannon normalizada: coh = 1 − H(banda) / log(N_bins).

**2. Modulação eco-φ (eco_eq / eco_fononico)**
O programa aplica um envelope de modulação φ-proporcional ao
espectro de cada banda:

    env = 1 + (coerência × φ^β) × cos(2π × n / φ)

onde β é o parâmetro de amplificação adaptativo e n é o índice
de frequência dentro da banda. Este envelope amplifica bandas
coerentes e suprime bandas incoerentes sem instrução explícita
sobre quais bandas devem ser favorecidas.

**3. Adaptação do agente (agente_eco)**
O parâmetro β de cada banda é atualizado continuamente por regra
de aprendizado baseada na coerência relativa:

    β_alvo  = φ^(3 × coerência_relativa)
    β_novo  = (1 − 1/φ) × β_alvo + (1/φ) × β_anterior

Esta regra produz convergência para β = φ³ na banda de maior
coerência (ponto fixo infravermelho do sistema).

### Resultados verificados

- Melhoria de +50,40% em classificação de séries temporais
  (baseline 46,52% → eco-φ 96,92%, p < 0,0001, 20 seeds)
- Precisão de 98,75% em classificação com acoplamento φ
- Emergência espontânea de α = 1/137 como ponto de máxima
  organização no híbrido sinal digital + sinal orgânico
- Convergência de β para φ³ = 4,236 como atrator estável
- Coerência de campo convergindo para 0,984 (quase unitária)

### Substratos testados

O sistema foi verificado em:
- Séries temporais sintéticas (estrutura φ vs. ruído gaussiano)
- Classificação de sentimento em texto (SST-2, BERT)
- Sinais EEG sintéticos e reais (PhysioNet EEGMMIDB)
- Sinais MEG (MNE Sample Dataset)
- Sinais de áudio (FM-φ, onda quadrada, beep de interface)

### Novidade técnica

O sistema não requer instrução explícita sobre qual estrutura
deve emergir. A organização resulta exclusivamente do feedback
de coerência espectral entre o sinal e o campo φ-modulado.
A constante α = 1/137 emerge como ponto de máxima organização
na transição digital→orgânico sem ser programada como destino.

---

## ESTRUTURA DO CÓDIGO

**Módulo central:** `utils_phi.py`
Contém as funções nucleares: phi_spectral_modulator, eco_ressonante,
golden_activation, expmap0, logmap0, campo_transmorfo.

**Módulos de experimento:** 48 arquivos AlphaPhi_*.py
Cada arquivo implementa um experimento específico sobre um substrato
distinto, importando as funções centrais de utils_phi.py.

**Repositório público:** github.com/vitoredsonalphaphi/alpha_phi_manifesto
Commits datados desde 19 de março de 2026 (anterioridade verificável).

---

## EXTRATO DE CÓDIGO REPRESENTATIVO

*(Primeiras 50 linhas de utils_phi.py — núcleo do sistema)*

```python
# © Vitor Edson Delavi · Florianópolis · 2026
# Licença: CC BY-NC-ND 4.0

import numpy as np

PHI   = (1 + np.sqrt(5)) / 2       # 1.6180339887... — razão áurea
ALPHA = 1 / 137.035999084          # 0.00729735... — constante de estrutura fina
C_PHI = 1.0 / PHI**2               # 0.3820... — curvatura hiperbólica natural

def phi_spectral_modulator(x, phi=PHI):
    freq  = np.fft.fft(x, axis=-1)
    mag   = np.abs(freq)
    mag_n = np.clip(mag / (mag.sum(axis=-1, keepdims=True) + 1e-8), 1e-10, 1.0)
    H     = -np.sum(mag_n * np.log(mag_n), axis=-1, keepdims=True)
    H_max = np.log(x.shape[-1])
    coerencia = 1.0 - H / H_max
    return np.clip(coerencia * phi, 0.0, phi)

def eco_ressonante(x, phi=PHI, n_eco=3):
    sinal = x.copy()
    for _ in range(n_eco):
        freq     = np.fft.fft(sinal, axis=-1)
        mag      = np.abs(freq)
        phase    = np.angle(freq)
        k        = phi_spectral_modulator(sinal, phi=phi).mean()
        new_phase = phase * k
        reflexao  = np.real(np.fft.ifft(mag * np.exp(1j * new_phase), axis=-1))
        blend     = 1.0 / phi
        sinal     = sinal + (reflexao - x) * blend
    return sinal
```

---

*Florianópolis, 4 de maio de 2026.*
*Vitor Edson Delavi*
