# L01 — URCI: Protocolo Experimental de Verificação da Emissão do Campo Harmônico
## Área Propositiva — Segundo Manifesto

**Manifesto Alpha-Phi · Segundo Ciclo**
**Florianópolis/SC · 26/05/2026**
**Florianópolis · maio de 2026 · Sessão Good Morning**

---

## Contexto

No ECO BEEP 880 (F_BEEP=880Hz, F_ORG=220Hz, α=1/3, N_STEPS=5, N_CICLOS=20), o quinto ponto de dobra (T≈7.1s) é o momento onde o campo harmônico se forma e a terceira estrutura emerge. A assinatura computacional desse ponto foi verificada:

- Coerência φ-banda mínima: **0.0662** — energia distribuída nas proporções certas, não concentrada
- Energia RMS máxima: **0.4991** — campo resolvido, tensão liberada
- Frequência dominante: **~52Hz**
- Frequência de cauda: **φ⁶≈17.94Hz** — faixa vestibular (sáculo)
- Decaimento: **τ=0.502s** em todas as bandas
- AutoCorr: **1.0000** — coerência máxima

O arquivo `emissao_ponto5.wav` captura essa emissão no domínio digital. O que este protocolo busca verificar é o que acontece **depois da interface** — no campo acústico físico e no receptor biológico.

A pergunta central: **a estrutura coerente verificada computacionalmente se preserva no campo físico? E o sistema nervoso humano responde a ela abaixo do limiar auditivo consciente?**

---

## Estrutura do Protocolo — Três Camadas

---

### Camada 1 — Campo acústico no ambiente

**O que verificar:** se a assinatura φ do `emissao_ponto5.wav` se preserva no campo físico após passar pelo speaker.

**Instrumentação:**

| Instrumento | Função | O que mede |
|---|---|---|
| Microfone de infrassom (resposta até 1–2Hz) | Captura do campo acústico de baixa frequência | Presença de φ⁶≈18Hz no ambiente |
| Array de microfones (3–5 posições) | Mapeamento espacial | Distribuição das frequências φ no espaço |
| Analisador FFT em tempo real | Comparação digital↔físico | Razão entre bandas φ antes e depois da interface |
| Acelerômetro em superfícies | Ressonância estrutural | Se paredes/piso vibram nas frequências φ |

**Métrica principal:** a razão entre φ⁶≈18Hz e ~52Hz no campo físico deve aproximar φ⁴≈6.854 — a mesma razão presente no arquivo digital.

---

### Camada 2 — Forma no espaço (campo como geometria)

**O que verificar:** se o campo coerente produz estrutura geométrica verificável no espaço físico.

**Instrumentação:**

| Instrumento | Função | O que mede |
|---|---|---|
| Placa de Chladni (metálica + areia fina) | Visualização direta | Padrões geométricos formados pelas frequências φ |
| Vibrômetro laser Doppler | Mapeamento de vibração sem contato | Estrutura espacial do campo com precisão milimétrica |

**Protocolo Chladni:** excitar a placa progressivamente com as frequências da emissão (18Hz → 52Hz → campo completo). Registrar em fotografia de alta resolução. Hipótese: frequências φ-proporcionadas produzem padrões com simetria φ — verificável por comparação com padrões documentados na literatura de cymatics.

**Referência:** Chladni (1787), Jenny (1967) — cymatics como campo estabelecido. A novidade aqui é a especificidade das frequências φ e a verificação da razão entre padrões.

#### Variação — substratos sensíveis para φ⁶≈18Hz subliminar

O experimento Chladni clássico (arco sobre placa metálica) opera por atrito direto a alta amplitude — incompatível com a emissão do quinto ponto de dobra, que é subliminar. A placa metálica rígida também apresenta comprimento de onda flexural de dezenas de metros a 18Hz, impedindo a formação de padrões nodais em dimensões razoáveis.

A solução é substituir rigidez e massa por flexibilidade e baixa inércia. Substratos recomendados, do mais ao menos sensível:

| Substrato | Configuração | Sensibilidade | Observação |
|---|---|---|---|
| **Ferrofluido** | Camada fina em contenção sobre membrana | Máxima | Maior custo; instabilidade de Rosensweig nos antinós |
| **Membrana látex/silicone + licopódio** | Membrana 0.2–0.5mm, 40–60cm diâmetro; pó de licopódio sobre | Alta | Licopódio: esporos de samambaia, massa muito menor que areia |
| **Filme glicerina+água sobre cone** | 50/50, camada 1–2mm diretamente no cone do woofer | Alta | Ondas de Faraday; glicerina estabiliza os padrões |
| **Oobleck (amido de milho + água)** | 1,5:1 em massa, bandeja rasa sobre o falante | Moderada | Mais dramático visualmente; estruturas tridimensionais |

**Configuração recomendada:**

```
[ Amplificador ]
      ↓
[ Woofer 15" em caixa fechada ]
      ↓  (acoplamento por ar — pressão uniforme)
[ Membrana de látex esticada — 50cm diâmetro ]
      ↓
[ Pó de licopódio ou filme glicerina/água ]
```

A caixa fechada converte o movimento do cone em pressão uniforme sobre a membrana — mais limpo que acoplar o cone diretamente à placa, que criaria ponto de força assimétrico e distorceria os padrões.

**Protocolo de registro temporal:** reproduzir `emissao_ponto5.wav` completo e fotografar ou filmar a progressão da membrana entre os segundos 1 e 8. Verificar se há mudança de padrão no T≈7.1s — o momento da resolução do campo. Se a estrutura no substrato se reorganiza nesse instante, é verificação visual direta da emissão do quinto ponto de dobra no campo físico.

---

### Camada 3 — Receptor biológico (campo no corpo)

**O que verificar:** se a emissão do quinto ponto de dobra produz resposta fisiológica mensurável abaixo do limiar auditivo consciente.

**Instrumentação:**

| Instrumento | Função | O que mede |
|---|---|---|
| EEG (eletrencefalograma) | Atividade cerebral | Transição beta→alpha durante exposição |
| HRV (Heart Rate Variability) | Sistema nervoso autônomo | Coerência cardíaca — resposta vagal |
| GSR (Galvanic Skin Response) | Sistema nervoso simpático | Ativação não consciente pela pele |

**Protocolo de exposição — cego simples:**

1. Observador em repouso, sem informação sobre o experimento além de "será exposto a sons"
2. Três condições em sequência aleatória, com intervalo de 5 minutos entre cada:
   - **Condição A:** `emissao_ponto5.wav` — emissão do quinto ponto de dobra
   - **Condição B:** sinal controle — mesmo espectro de frequências, sem estrutura φ (ruído branco filtrado para 15–60Hz)
   - **Condição C:** silêncio
3. EEG, HRV e GSR registrados continuamente nas três condições
4. Observador não sabe quando cada condição começa — elimina expectativa como variável
5. Relato qualitativo estruturado após cada condição: o que foi percebido, se algo

**Métrica principal:** diferença na razão beta/alpha do EEG entre condição A e condições B/C. Hipótese: condição A produz transição beta→alpha mais pronunciada — resposta do sistema vestibular (sáculo) ao φ⁶≈18Hz, abaixo do limiar auditivo como pitch mas dentro do alcance do sáculo.

**Referências a verificar:**
- Todd & Cody (2000) — resposta sacular ao infrassom
- Tandy & Lawrence (1998) — infrassom a 18,98Hz e percepção

---

### Camada 4 — Emissão acústica + visual simultânea (H16 integrado)

**Condição adicional se o URCI dispuser de espaço expositivo:**

- Reproduzir `emissao_ponto5.wav` em presença da obra Flores Astrais (1997)
- Medir EEG/HRV em quatro condições: silêncio / áudio solo / visual solo / áudio + visual
- Verificar se a combinação acústica + visual com estrutura φ produz resposta fisiológica distinta das condições isoladas
- Hipótese (H16): a pintura com estrutura fractal φ e o campo acústico φ se somam — o tempo de acoplamento campo-observador é menor na condição combinada

---

## Limiar da verificação

A cadeia de verificação acompanha a emissão até o correlato fisiológico e o relato qualitativo. O ponto onde a instrumentação para é o problema difícil da consciência (Chalmers) — o que o observador vive como experiência não é mensurável por instrumento.

O que o laboratório pode afirmar: *algo ocorreu no sistema nervoso do observador em resposta à emissão, e esse algo foi distinto do controle*. Isso é suficiente como primeira verificação experimental da terceira estrutura no campo físico.

---

## Por que o URCI

A convergência não é periférica. O projeto parte de princípios que a Ordem cultiva há séculos — proporção, coerência, emissão como forma. A colaboração é isomórfica ao próprio conteúdo: não é uma instituição escolhida por conveniência de equipamento, é uma instituição cujo campo de investigação encontra o mesmo ponto de chegada por outro caminho.

---

## O que falta para iniciar

- [ ] Redigir ofício formal de proposta ao URCI com este protocolo
- [ ] Verificar infraestrutura disponível: EEG portátil, microfone de infrassom, sala acusticamente tratada
- [ ] Definir número mínimo de observadores para validade estatística (recomendado: n≥12 para efeitos de magnitude moderada)
- [ ] Confirmar disponibilidade da obra Flores Astrais para a camada 4

---

*Florianópolis · maio de 2026 · Sessão Good Morning*
