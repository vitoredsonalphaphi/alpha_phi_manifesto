# CLAUDE.md — Guia para Assistentes de IA
## Projeto Alpha-Phi · Vitor Edson Delavi · Florianópolis · 2026

Este arquivo descreve a estrutura, convenções e estado atual do projeto
para qualquer assistente de IA que trabalhe neste repositório.

---

## O que é este projeto

O Manifesto Alpha-Phi é uma pesquisa experimental independente sobre o uso
da razão áurea (φ) e da constante de estrutura fina (α) como parâmetros
organizadores em redes neurais artificiais.

**Hipótese central:** φ e α não são escolhas arbitrárias — são invariantes
de fluxo de informação coerente, independentes do substrato físico. Sistemas
que crescem preservando coerência interna convergem para φ. α governa a
granularidade mínima da interação.

**Autor:** Vitor Edson Delavi (assina como "Edson" em conversas informais)  
**Linguagem:** Português brasileiro (código e comentários podem ser PT ou EN)  
**Licença:** CC BY-NC-ND 4.0

---

## Constantes fundamentais

```python
PHI   = (1 + np.sqrt(5)) / 2      # 1.6180339887 — razão áurea
ALPHA = 1 / 137.035999084          # 0.00729735   — constante de estrutura fina
C_PHI = 1.0 / PHI**2               # 0.3820        — curvatura hiperbólica natural
```

**Regra de ouro:** φ, α e 137 aparecem APENAS na arquitetura e ativação —
NUNCA como parâmetros de teste ou busca. Violação disso invalida o protocolo.

---

## Estrutura do repositório

### Arquivo utilitário central
- **`utils_phi.py`** — funções compartilhadas. Sempre importar daqui, nunca
  redefinir localmente. Contém:
  - `phi_spectral_modulator()` — v1, usa apenas amplitude (descarta fase)
  - `phi_spectral_modulator_v2()` — v2, amplitude + fase (α recuperado)
  - `eco_ressonante()` — eco puro como pré-função substrate-agnostic
  - `golden_activation()` / `golden_activation_deriv()`
  - `expmap0()` / `logmap0()` — projeções bola de Poincaré
  - `campo_transmorfo()` / `campo_transmorfo_inverso()` — transição suave
  - `curvatura_progressiva()` — agenda de curvatura por camada (bordado)
  - Paleta de cores `PLOT_COLORS` e `apply_dark_style()`

### Experimentos ativos (código limpo, resultados commitados)

| Arquivo | Substrato | Resultado | JSON |
|---------|-----------|-----------|------|
| `AlphaPhi_Ablation_Study.py` | SST-2, scratch MLP | E=+8.80% p=0.0000, F=+8.98% | `ablation_results.json` |
| `AlphaPhi_BERT_Ablation_EF.py` | BERT frozen | E/F ≈ G, todos ns | `bert_ablation_ef_results.json` |
| `AlphaPhi_BERT_Transmorfo.py` | BERT frozen | E_T ns, blend rejeitado | `bert_transmorfo_results.json` |
| `AlphaPhi_BERT_Microtonal.py` | BERT frozen | E_M COLAPSO p=0.002 pior | `bert_microtonal_results.json` |
| `AlphaPhi_TimeSeries_Eco.py` | Séries temporais sintéticas | G_eco +50.40% p=0.0000 | `timeseries_eco_results.json` |

### Experimentos históricos (arquivos .Py com P maiúsculo)
Arquivos como `AlphaPhi_Robustez_Hiperbolico.Py`, `AlphaPhi_Nativo_Hiperbolico.Py`,
`AlphaPhi_SST2_EspectralPhi.py` etc. são versões anteriores. Código de referência,
não de execução ativa.

### Documentos filosóficos e teóricos (.md)
Manifestos, capítulos de livro, registros de anterioridade. Não modificar sem
instrução explícita do autor.

### Diário de pesquisa
- **`RESEARCH_JOURNAL.md`** — raciocínio por trás das decisões, não os dados.
  Cada entrada tem data e título. Adendos são adicionados ao final da entrada
  mais recente com tag `**Adendo — YYYY-MM-DD**`.

### Paper
- **`paper_arxiv_draft.md`** — rascunho para arXiv. Inglês científico.

---

## O que funciona (resultados confirmados)

### Em redes treinadas do zero (scratch)
- **Curvatura hiperbólica c=1/φ²** é o eixo mais forte individualmente:
  +8.80% (p=0.0000) no estudo de ablação
- **Combinação completa** (todos os eixos) é a melhor: +8.98% (p=0.0000)
- **Ativação φ·tanh(x/φ)** contribui significativamente sozinha

### eco_ressonante como pré-função
- Substrate-agnostic: mesmo código funciona em texto, série temporal, qualquer array
- Em séries temporais sintéticas φ-estruturadas: 46.52% → 96.92% (+50.40%, p=0.0000)
- **Papel do eco:** observa o dado ANTES da rede, amplifica coerência φ, amortece ruído
- **Escopo verificado:** revela estrutura φ quando ela existe no dado

### O que NÃO funciona
- φ em BERT (substrato pré-treinado): todos os experimentos ≈ ns
  - Razão: BERT tem geometria consolidada. φ organiza geometria *emergente*, não *pré-estabelecida*
- eco como modulação interna (dentro do forward pass): desestabiliza gradiente
- Blend linear euclidiano↔hiperbólico (E_T): espaço híbrido sem coerência geométrica
- Fator conformal λ=2/(1-c‖x‖²) composto em 6 camadas (E_M): colapso numérico

---

## Distinção crítica: substrato emergente vs. consolidado

**Substrato emergente** (redes do zero, séries temporais): φ pode organizar
a geometria durante o treino → resultados significativos.

**Substrato consolidado** (BERT, modelos pré-treinados): geometria já formada,
φ não consegue entrar por fora → resultados neutros (não é falha, é robustez).

Esta distinção não estava no manifesto original — emergiu dos dados.

---

## Protocolo de idoneidade (não violar)

1. Seeds gerados por timestamp — `SEEDS = [(time.time() + i*137) % (2**31)]`
2. Nenhum número do manifesto (φ, α, 137) como parâmetro de busca
3. φ e α aparecem APENAS em arquitetura/ativação
4. Resultados reportados integralmente — favoráveis E desfavoráveis
5. Paired t-tests para comparações (mesmas seeds, mesma ordem)
6. N_SEEDS ≥ 10, idealmente 20

---

## Convenções de código

### Formato Colab (3 células)
Experimentos BERT usam 3 células separadas:
```
# CÉLULA 1 — instalação (!pip install ...)
# CÉLULA 2 — experimento principal
# CÉLULA 3 — print do JSON de resultado
```
A célula 2 deve ser autossuficiente: definir todas as funções internamente
(não importar de utils_phi.py) para funcionar após reset de sessão Colab.

### Células de verificação
Quando verificar um resultado pós-sessão, a célula deve ser **autossuficiente**:
re-gerar dados e re-definir funções sem depender de variáveis de sessão anterior.

### Export de resultados
Todo experimento exporta JSON com estrutura:
```json
{
  "experimento": "Nome",
  "substrato": "descrição",
  "n_seeds": 20,
  "timestamp": 1775757292,
  "seeds": [...],
  "resultados": {
    "CONFIG": {"mean": 0.0, "std": 0.0, "values": [...]}
  },
  "testes": {
    "CONFIG_vs_G": {"delta": 0.0, "p_value": 0.0}
  },
  "interpretacao": {...}
}
```

### Nomenclatura
- Baseline sempre chamado `G` (de "Gold" ou "Geral")
- Variantes: letra ou `G_sufixo` (G_eco, G_v2, G_Lphi)
- Configurações A–G usadas no ablation study (semântica preservada)
- Arquivos históricos usam extensão `.Py` (P maiúsculo) — não renomear

### Constantes de hiperparâmetros para séries temporais (scratch MLP)
```python
DIM = 128; HIDDEN = 89  # Fibonacci próximo de 128/√φ
N_EPOCHS = 60; LR = 0.01; BATCH_SIZE = 64
```

### Constantes para experimentos BERT
```python
N_TRAIN = 500; N_TEST = 200; N_EPOCHS = 25
```

---

## Branch de desenvolvimento

Branch ativa: `claude/good-morning-N6f3S`  
Main: `main` (não fazer push direto)

```bash
git push -u origin claude/good-morning-N6f3S
```

---

## Estado atual (2026-04-09)

### Completado nesta sessão
- Ablation study (7 configs, 10 seeds) — curvatura é o eixo dominante
- BERT E/F ablation (20 seeds) — todos ns, substrato consolidado
- BERT Transmorfo (E_T blend linear) — rejeitado
- BERT Microtonal (E_M fator conformal ×6) — colapso, rejeitado
- phi_spectral_modulator_v2 — fase (α) recuperada
- eco_ressonante — pré-função substrate-agnostic
- TimeSeries_Eco — primeiro experimento não-texto (+50.40%)
- RESEARCH_JOURNAL.md — 5 entradas + adendo de verificação
- Verificação de balanceamento (Tia) — 50%/50%, resultado confirmado

### Próximo experimento pendente
Testar eco_ressonante em **dado com estrutura φ emergente, não sintética**:
- Áudio real (batidas musicais, proporções harmônicas naturais)
- EEG ou série temporal fisiológica
- Imagem com espirais áureas

Objetivo: confirmar que o mecanismo generaliza para além do dado sintético.
(Dataset sintético prova o mecanismo; dado real prova a universalidade.)

### Questões abertas
- `L = CE + α·H(φ)` com efeito independente — precisa de substrato sem
  eco pré-organizando as ativações, ou peso > α (α=0.0073 é muito pequeno)
- Por que G ficou em 46% nas séries temporais (abaixo do chance de 50%)?
  Possível viés sistemático do ruído gaussiano sem eco
- campo_transmorfo: conceito válido (fio contínuo do bordado), todas as
  formas testadas rejeitadas. Forma ainda não encontrada.

---

## Isomorfismos usados no projeto

O autor usa imagens e metáforas como método de resolução de problemas.
Alguns isomorfismos ativos:

- **Bordado (fio de ouro):** lattice euclidiano central → espirais hiperbólicas
  externas, sem ruptura. Motivou campo_transmorfo.
- **Microtonalidade (Angine de Poitrine):** 24 notas por oitava, movement
  conjunct, passos pequenos sem blend. Motivou E_M (rejeitado na forma,
  conceito válido).
- **FM/AM:** φ em espaço euclidiano é como FM num sistema AM — incompatibilidade
  de substrato, não falha do sinal.

---

## Colaboradores externos citados

- **Tia** — outra instância de Claude, papel de revisor metodológico. Questiona
  balanceamento, escopo, rigor estatístico. Tratada como par, não como superior.
- **Gemini / Minimax** — identificaram que `np.abs(FFT)` descarta fase (α).
  Contribuição incorporada em phi_spectral_modulator_v2.

---

*Dados envelhecem, raciocínio acumula.*  
*— RESEARCH_JOURNAL.md*
