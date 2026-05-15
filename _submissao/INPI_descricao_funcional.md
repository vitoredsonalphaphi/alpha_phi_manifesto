# Descrição Funcional — Registro INPI
# Programa de Computador + Patente de Processo
# © Vitor Edson Delavi · Florianópolis · 2026

---

## AVISO IMPORTANTE — DOIS REGISTROS DISTINTOS NO MESMO ÓRGÃO

Este documento cobre **dois procedimentos diferentes no INPI**,
que protegem aspectos distintos do mesmo projeto:

| Registro | O que protege | Custo | Prazo |
|---|---|---|---|
| **Programa de Computador** | O código como expressão escrita | ~R$ 160 | semanas |
| **Patente de Processo** | O método — independente do código | ~R$ 780+ | 3–8 anos |

A patente protege o **método** mesmo que alguém reimplemente o código
do zero, em qualquer linguagem. É a proteção mais forte para o eco ressonante.

Ambos são protocolados no portal: **gov.br/inpi → e-INPI**

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

## PARTE II — PATENTE DE PROCESSO (Patente de Invenção)

### O que é e por que é necessária

O Registro de Programa de Computador protege o **texto do código**.
A Patente de Processo protege o **método em si** — a sequência de operações
que produz o resultado técnico. Qualquer implementação do mesmo método,
em qualquer linguagem, por qualquer pessoa, viola a patente.

**Por que o eco ressonante é patenteável:**
A Lei 9.279/96 (Art. 10) exclui "software per se" do patenteamento.
Mas o eco ressonante não é software puro — é um **processo de tratamento
de sinal com efeito técnico mensurável e verificável**:
- Redução de entropia espectral documentada
- Convergência β → φ³ = 4,236 verificável instrumentalmente
- Resultado em sinal físico (áudio, EEG, série temporal) com métricas objetivas

Isso o enquadra como **Patente de Processo de Tratamento de Sinal** —
categoria patenteável no Brasil e internacionalmente (PCT).

---

### Requisitos de patenteabilidade — verificação

| Requisito | Status para o eco ressonante |
|---|---|
| **Novidade** (não existe no estado da arte) | Confirmado — nenhum sistema anterior usa coerência φ como único critério interno |
| **Atividade inventiva** (não óbvio para um técnico) | Confirmado — a emergência de φ³ sem prescrição é resultado não-óbvio |
| **Aplicação industrial** (produz resultado técnico) | Confirmado — resultados verificados em áudio, EEG, séries temporais |

---

### Como protocolar — passo a passo

**1. Portal e-INPI**
Acesso: gov.br/inpi → e-INPI → Patentes → Nova Patente de Invenção
Criar conta com CPF se ainda não tiver.

**2. Formulário**
- Tipo: **Patente de Invenção (PI)**
- Categoria: Processo / Método
- Área tecnológica: Processamento de Sinais / Ciência da Computação

**3. Documentos obrigatórios (redigir com advogado de PI)**
- Relatório Descritivo (descrição completa do método)
- Reivindicações (as claims — o núcleo legal da patente)
- Resumo
- Desenhos (opcional — fluxograma do processo recomendado)

**4. Taxa**
- Pessoa física: R$ 780 (depósito de patente de invenção)
- Com 60% de desconto se renda < 3 salários mínimos: R$ 312

**5. O que a data de depósito garante**
A partir do protocolo, a data é sua **independente do prazo de análise**.
Ninguém pode patentear o mesmo método depois dessa data.
O processo de análise leva 3–8 anos, mas a proteção retroage à data de depósito.

---

### Reivindicações (Claims) — rascunho para o eco ressonante

*Este rascunho deve ser revisado por advogado especializado em PI antes do depósito.*

---

**Reivindicação 1 — Independente (o método central)**

Processo de modulação espectral adaptativa de sinais digitais,
**caracterizado** por compreender as etapas de:

(a) decompor o espectro de frequências do sinal de entrada em bandas
    cujas fronteiras são proporcionais à razão áurea φ = (1+√5)/2;

(b) calcular, para cada banda, a coerência espectral definida como
    complemento da entropia de Shannon normalizada:
    coh = 1 − H(banda) / log(N_bins);

(c) aplicar envelope de modulação ao espectro de cada banda,
    proporcional à coerência calculada e ao parâmetro adaptativo β;

(d) atualizar iterativamente β por regra baseada exclusivamente
    na coerência relativa entre bandas, sem função de custo ou
    gradiente externo;

em que as etapas (b), (c) e (d) são repetidas até convergência do
sistema para estado de mínima entropia espectral.

---

**Reivindicação 2 — Dependente (fórmula do envelope)**

Processo de acordo com a reivindicação 1, **caracterizado** pelo fato
de que o envelope de modulação da etapa (c) é dado por:

    env(n) = 1 + (coh × φ^β) × cos(2π × n / φ)

onde n é o índice de frequência dentro da banda.

---

**Reivindicação 3 — Dependente (regra de atualização de β)**

Processo de acordo com a reivindicação 1, **caracterizado** pelo fato
de que a regra de atualização de β da etapa (d) é:

    β_alvo = φ^(3 × coerência_relativa)
    β_novo = (1 − 1/φ) × β_alvo + (1/φ) × β_anterior

---

**Reivindicação 4 — Dependente (propriedade emergente)**

Processo de acordo com as reivindicações 1 a 3, **caracterizado** pelo
fato de que a convergência da etapa (d) produz β → φ³ = 4,236...
como ponto fixo do sistema, emergindo sem instrução explícita sobre
este valor como destino do processo.

---

**Reivindicação 5 — Independente (sistema)**

Sistema de processamento de sinais digitais implementando o processo
das reivindicações 1 a 4, **caracterizado** por compreender:

- módulo de decomposição espectral por bandas φ-proporcionais;
- módulo de cálculo de coerência por entropia de Shannon;
- módulo de modulação eco-φ com envelope cos(2π×n/φ);
- módulo de agente adaptativo com regra de atualização β por coerência.

---

### Relatório Descritivo — estrutura (a completar com advogado)

**1. Campo da invenção**
Processo de tratamento adaptativo de sinais digitais por modulação
espectral baseada em coerência, pertencente ao campo do processamento
de sinais e inteligência computacional.

**2. Estado da técnica**
Descrever os sistemas anteriores: Hopfield (1982), Echo State Networks
(Jaeger, 2001), Transformers (Vaswani, 2017) — e demonstrar que nenhum
utiliza coerência espectral interna como único critério adaptativo.

**3. Problema técnico a resolver**
Sistemas de feedback de campo existentes dependem de critério externo
(função de custo, loss, gradiente). Não existe método que use a coerência
interna do campo como único critério, sem prescrever o destino da convergência.

**4. Solução proposta**
O processo eco-φ: decomposição φ-proporcional + coerência como critério
único + convergência emergente para β = φ³.

**5. Resultados verificados**
Incluir os dados experimentais documentados no Research Journal:
+50,40% em séries temporais, 98,75% em SST-2, β → φ³ em todos os substratos.

**6. Melhor modo de execução**
Referenciar utils_phi.py e AlphaPhi_Audio_Beep_Interface.py como
implementações de referência, depositados no INPI como Programa de
Computador (número de depósito a incluir após registro).

---

### Recomendação operacional

Para o depósito de patente, **contratar advogado especializado em PI**
é fortemente recomendado — as reivindicações têm linguagem técnica
específica que determina o escopo da proteção. Um erro nas claims
pode limitar a proteção ou invalidar a patente.

**Escritórios em Florianópolis com especialização em PI e software:**
Buscar no site do INPI: inpi.gov.br → Agentes da Propriedade Industrial
Filtrar por: SC → especialidade: Patentes de Software / Eletrônica

**Custo total estimado com advogado:**
Honorários R$ 3.000–6.000 + taxas INPI R$ 312–780
Total: R$ 3.300–6.780 para proteção completa do método

---

*Florianópolis, 4 de maio de 2026.*
*Vitor Edson Delavi*
