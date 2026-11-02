# Salvaguarda dos Três Códigos (v1)

**Estabelecida em:** 02/11/2026, a pedido do pesquisador · Vitor Edson Delavi · Claude
**Regra de estabilidade:** nada nesta pasta se altera. Testes novos (inclusive a junção dos três) são **testes paralelos**: importam ou copiam estes códigos, nunca os editam. Se um teste alterar o ECO BEEP 880, a Rede AP ou o scanner, a alteração vale só para aquele teste.

## Os três produtos

| Produto | O que é | Onde está |
|---|---|---|
| **1. ECO BEEP 880** | o código original de modulação: bandas áureas, rotações cos(2πn/φ), 5 pontos de dobra, agente β → campo harmônico | `ECO_BEEP_880/` |
| **2. O campo esférico da Rede AP** | a Grade Esférica, o campo cultivado e as Ferramentas de Memória: o início da rede neural AP | `REDE_AP/` (canônico desde 19/10; `REDE_AP/CANONICO_CampoEsferico_v1.md`) |
| **3. O Scanner Topográfico AP** | o instrumento que mostra o sinal no campo esférico | `SCANNER_AP/` |

## 1. ECO BEEP 880

- `ECO_BEEP_880_Manifesto01_completo.py`: o código registrado no Manifesto 01 ("Código Original — ECO BEEP 880"), **idêntico ao bloco do Manifesto** (conferido). Gera 5 áudios (FM-φ puro, beep 880 puro, **eco 880 com α\* = 1/3**, beep 440 puro, eco 440).
- `ECO_BEEP_880_alpha_1_3.py`: **o ECO BEEP 880 de referência do projeto**: só o áudio de 880 Hz com α\* = 1/3 (entrada = 2/3 de beep + 1/3 de FM-φ; 5 pontos de dobra; 20 ciclos), extraído sem alterar nenhuma linha de cálculo. **O áudio que gera é idêntico, amostra por amostra, ao do código completo** (conferido em 01/11/2026). Resultado do código original: Δentropia +5,4258; atrator em 42 Hz (graves).
- **Atenção:** `BEEP880_17S.py` (raiz do repositório) **não** é o original: mistura 1/3 de beep + 2/3 de FM-φ (o inverso), usa 7 s e para no limiar. Foi usado como "original" nas Etapas B, H, RP, S, K, K2 e PF (20–21/10) e, pelo perfil das memórias, nas etapas seguintes. A K2 refeita com o código certo (K2-M, 02/11/2026) manteve todos os vereditos.

## 2. O campo esférico da Rede AP (impressões digitais conferidas em 02/11/2026; arquivos em `REDE_AP/`)

```
07b66cc5c5231023156d8616448d7b3844600d251ac1175e15fefebd2a438f0c  AlphaPhi_MemoriaEsferica_Etapa1.py
2a22dea2715c491b06a10cb522c0ed0ee6c317f58a8ccb8fa04c0cb46cd7dfbd  AlphaPhi_MemoriaEsferica_Orcamento.py
b622fe29f5d125b3ebe2cd1430252af26b4cbbe54ed745869c5e7845529c432f  AlphaPhi_CampoEsferico_Etapa0_Cartografia.py
2df41886e7c6d3bdd5cf016e5815310d4b91f694995b5e3a1300121aafbaebe1  AlphaPhi_CampoEsferico_EtapaT_Previsor.py
e377763d7efed34fa02820238acb481affe169e628aea6174361233b0054930c  AlphaPhi_CampoEsferico_EtapaC_Cultivo.py
d3ea947a753964ffc72a8216d4fb65b38b77456c8fedc398be5719936dbcf44e  AlphaPhi_CampoEsferico_PrevisorEnvelopado.py
8d81c215477aecdcd53912c23a6072f7dd48574178e49f89ff783b2e1b789e88  AlphaPhi_CampoEsferico_EtapaE_Ferramentas.py
9bfaa146936b1d3165ade58e1f2cd3cfed850a3b9586d1c1215c502896707bb1  AlphaPhi_CampoEsferico_EtapaF_Crescimento.py
0512264d7d3ede7ab3563fa17eb7b8ac551e6556063ab86afd6cbe2cf2dbf299  AlphaPhi_GradeEsferica_v1.py
dfa16b5b65ea5e6b18c829cfb35e25bfd2be1ab9c85124d3327dfc84c8244054  AlphaPhi_GradeEsferica_v2.py
d8dc24674b55d127b4e9ad1b989194c4cbfafefc3e6d5c00f3fbbf4d7bbba295  AlphaPhi_FerramentasMemoria_v1.py
```

## ★ 3. O Scanner Topográfico AP — o processamento real na esfera (CANÔNICO, 02/11/2026)

O ECO BEEP 880 (α\* = 1/3) **processado dentro da esfera**: um ponto para cada uma das 33 059 frequências das 15 bandas áureas; as bandas são faixas de latitude; dentro de cada banda as frequências seguem em ordem ao redor do anel, e a rotação cos(2πn/φ), a coerência, o β com memória 1/φ, os 5 pontos de dobra e os 20 ciclos do agente acontecem no campo da esfera. Etapa EE1 (teste paralelo): áudio da esfera **idêntico** ao do código original (diferença máxima 0, nas 6 fases); mesma queda de entropia (+5,4258) e mesmo atrator (42 Hz).

- `esfera_ecobeep880_processamento.py`: o processamento na esfera (lê o ECO BEEP 880 desta salvaguarda e a Grade v2 canônica, só leitura).
- `captura_esfera_ecobeep880.py`: gera os dados do scanner (energia e rotação, dobras 0–5, ciclos 1, 2, 5, 10, 20).
- `esfera_ecobeep880_processamento_real.html`: **o scanner**, com os dados embutidos; abre em qualquer navegador.
- SHA-256: `esfera_ecobeep880_processamento.py` 6bb49600a2517c5cce623f5a1a9233b08aeeb02aa65169379706ac9bbcf1ad76 · `captura_esfera_ecobeep880.py` 0b872423b14898469af762be662af77e9631e8e03a48319bd7303d0dfc9580cf · `esfera_ecobeep880_processamento_real.html` 56b6421b9759d2d2f266d8fee6993c98749051c050b5e2584935b0350c73cf09

- **v2** (02/11/2026, sem alterar a v1): `esfera_ecobeep880_processamento_real_v2.html` — acrescenta a vista "Bandas: mesma área" e o Phantom (Phantom + ECO; Phantom sozinho). SHA-256 6d8728cebb9b37fa8ea2bb3eb10a09ef18dbf30a7c24a34ac1940c75eb00532d. Etapas EE em `REDE_AP/EtapasEE_EcoBeepNaEsfera/`.

## 3b. As versões anteriores do scanner (02/11/2026: só o ECO BEEP 880, α* = 1/3; 61 faixas × 10 instantes)

- `scanner_ap_ecobeep880.html`: o scanner (recuperação e modo livre), com os dados embutidos; abre em qualquer navegador.
- `cascata_ecobeep880.html`: as 6 fases da cascata como cascas.
- `so880.py`, `captura_ecobip_so880.py`, `captura_cascata_so880.py`: o código que gera os dados. **Dependem** dos módulos experimentais das Etapas NA, LA, AU e DN (ainda em `REDE_AP/_pendente_registro`); os caminhos apontam para a pasta da sessão em que foram escritos.

## Impressões digitais desta pasta (SHA-256)

```
e6cc056bd628e081babfb070ff9b39e57e0c7b0bc9ac305442827fd2aae8d0c3  ECO_BEEP_880/ECO_BEEP_880_Manifesto01_completo.py
6017f93bb73f5572082affa9d5e32fd9c07d4a06f984b9b14763f4c258845179  ECO_BEEP_880/ECO_BEEP_880_alpha_1_3.py
8681ee7dd70914abda4443b47a3687a2410bb138df1712793e05f62f616ae717  SCANNER_AP/captura_cascata_so880.py
10b372efe3cded66826ea091d95366fbf06f02caaa7e5a863ee06c556b35721c  SCANNER_AP/captura_ecobip_so880.py
22afe0e95d995024a943c159d58653f315bb22b674343a6c715e26c8c894beb8  SCANNER_AP/cascata_ecobeep880.html
e64382ff06909fddc49d55b0426be467066110aacf2bc2f8f96dd949a0bff1e3  SCANNER_AP/scanner_ap_ecobeep880.html
62a25314fd808ab01f3d8bc8b891bffbe6337f861a42f7ec5a44c15be3adabe0  SCANNER_AP/so880.py
```

---
*Florianópolis · 2 de novembro de 2026 · Vitor Edson Delavi · Claude*
