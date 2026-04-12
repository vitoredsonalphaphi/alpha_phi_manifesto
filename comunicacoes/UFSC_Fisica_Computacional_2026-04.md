# Ofício — Física Computacional / Processamento de Sinais · UFSC
**Data:** Abril de 2026
**De:** Vitor Edson Delavi · Florianópolis
**Para:** Departamento de Física / Engenharia — CFM ou CTC/UFSC
**Assunto:** Pré-processamento adaptativo por coerência espectral coletiva

---

Boa tarde,

Meu nome é Vitor Edson Delavi, sou pesquisador independente em
Florianópolis. Estou realizando experimentos computacionais sobre
pré-processamento de sinais para redes neurais e gostaria de
apresentar um resultado e fazer uma pergunta que pode ser de
interesse para a física computacional.

**O resultado:**

Desenvolvemos um método de pré-processamento denominado
*eco fonônico ressonante*. Em vez de aplicar uma transformação com
parâmetro fixo, o método mede a coerência espectral coletiva do
batch de dados — a "temperatura da rede", por analogia a fônons —
e calibra o parâmetro de rotação de fase automaticamente.

Em dois substratos distintos:
- Séries temporais com estrutura harmônica: +2.65% vs parâmetro
  fixo (p=0.0018, 20 seeds)
- Harmônicos musicais naturais (sem estrutura φ nos dados): +1.15%
  vs parâmetro fixo (p<0.001, 20 seeds)

O método é auto-calibrável: o parâmetro k emerge da coerência
espectral coletiva do batch, sem instrução externa.

**A pergunta aberta:**

O método funciona porque estatísticas espectrais coletivas do batch
carregam informação que a análise amostra-por-amostra não captura.
Isso levanta uma questão: experimentos de física que lidam com
grandes volumes de eventos ruidosos — detecção de partículas, ondas
gravitacionais, CMB — se beneficiariam de calibração de
pré-processamento por coerência espectral coletiva dos eventos?

O código está em repositório público com protocolo de idoneidade
documentado:
github.com/vitoredsonalphaphi/alpha_phi_manifesto

Estaria disponível para uma conversa breve sobre se esta pergunta
tem relevância no contexto da pesquisa da UFSC.

Grato,

**Vitor Edson Delavi**
Florianópolis · 2026
github.com/vitoredsonalphaphi/alpha_phi_manifesto
