# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

# utils_phi.py — Utilitários compartilhados Alpha Phi
# Vitor Edson Delavi · Florianópolis · 2026

import numpy as np

# ── Constantes fundamentais — núcleo do projeto αφ ───────────────────────
PHI   = (1 + np.sqrt(5)) / 2       # 1.6180339887... — razão áurea
                                    # Proporção do padrão organizador que precede a estrutura.
                                    # Operador de coerência. Substrato de φ·tanh(x/φ).

ALPHA = 1 / 137.035999084          # 0.00729735... — constante de estrutura fina
                                    # Granularidade mínima da interação.
                                    # Regula a penalidade de coerência: L = CE + α·H(φ)
                                    # Taxa de perturbação mínima natural no Quarto Eixo.
                                    # α e φ juntos: a dupla que nomeia e fundamenta o projeto.

LOG_ALPHA = np.log(1.0 / ALPHA)    # log(137) ≈ 4.920 — régua de entropia nativa

C_PHI = 1.0 / PHI**2               # 0.3820... — curvatura hiperbólica natural
                                    # Ponto de dobra do microponto do Quarto Eixo.

# ── Modulador Espectral φ ─────────────────────────────────────────────────
def phi_spectral_modulator(x, phi=PHI):
    """
    Campo morfogenético digital — análogo ao campo de Levin.
    Identifica a frequência informacional do dado e
    retorna o modulador φ para calibrar o gradiente.

    x: (batch, dim) — embeddings dos dados
    returns: (batch, 1) — fator de modulação por amostra
    """
    freq         = np.fft.fft(x, axis=-1)
    energia      = np.abs(freq)
    energia_norm = energia / (energia.sum(axis=-1, keepdims=True) + 1e-8)
    # Clip antes do log para evitar NaN quando energia_norm ≈ 0
    energia_norm = np.clip(energia_norm, 1e-10, 1.0)
    entropia     = -np.sum(energia_norm * np.log(energia_norm), axis=-1, keepdims=True)
    entropia_norm = entropia / np.log(x.shape[-1])
    coerencia    = 1.0 - entropia_norm
    return phi * np.tanh(coerencia * phi)

# ── Modulador Espectral αφ v2 — fase recuperada + eco ressonante ──────────
#
# Contribuição Gemini/Minimax (2026-04-08):
#   phi_spectral_modulator (v1): usa np.abs → descarta fase → descarta α
#   Amplitude = estrutura = φ (o que o sinal é)
#   Fase      = intenção  = α (para onde o sinal vai)
#   np.abs silencia α — descarta metade do nome do projeto.
#
# v2: amplitude + fase → plano complexo → eco ressonante → convergência
#   complex_mod = amplitude · e^(j·α·φ) — rotação no plano complexo
#   φ dita a curvatura, α dita a trajetória.
#
# Substrate-agnostic: FFT opera sobre qualquer array numérico.
#   Texto, áudio, imagem, EEG, série temporal — mesmo código.
#   A pergunta ao dado é universal: "sua trajetória ressoa com φ?"

def phi_spectral_modulator_v2(x, phi=PHI, alpha=ALPHA, n_eco=3):
    """
    Modulador espectral αφ com fase recuperada e eco ressonante.
    Substrate-agnostic: opera sobre qualquer array numérico (batch, dim).

    Ciclo:
      1. Projeção: FFT → amplitude (φ) + fase (α)
      2. Rotação:  complex_mod = amplitude · e^(j · α · φ · 137)
                  (137 ≈ 1/α em unidades naturais — escala da intenção)
      3. Reflexão: IFFT → sinal modulado no domínio original
      4. Eco:      resíduo = reflexão - original → reinjeção por φ
      5. Convergência: sinal com coerência φ estabiliza; ruído diverge

    returns: (batch, 1) — fator de modulação por amostra
    """
    x = np.asarray(x, dtype=float)
    sinal = x.copy()

    for _ in range(n_eco):
        freq      = np.fft.fft(sinal, axis=-1)
        amplitude = np.abs(freq)                    # estrutura (φ)
        fase      = np.angle(freq)                  # intenção (α)

        # Rotação no plano complexo: α como operador de trajetória
        # 1/ALPHA ≈ 137 — escala natural da constante de estrutura fina
        nova_fase      = fase * (phi * alpha * (1.0 / alpha))  # = fase * phi
        sinal_complexo = amplitude * np.exp(1j * nova_fase)

        # Reflexão: retorno ao domínio original
        reflexao = np.real(np.fft.ifft(sinal_complexo, axis=-1))

        # Eco: resíduo reinjetado por φ (atrator de ressonância)
        eco   = reflexao - x
        sinal = sinal + (eco / phi)

    # Extrai coerência do sinal convergido
    freq_final  = np.fft.fft(sinal, axis=-1)
    energia     = np.abs(freq_final)
    e_norm      = np.clip(energia / (energia.sum(axis=-1, keepdims=True) + 1e-8), 1e-10, 1.0)
    entropia    = -np.sum(e_norm * np.log(e_norm), axis=-1, keepdims=True)
    coerencia   = 1.0 - entropia / np.log(x.shape[-1])
    return phi * np.tanh(coerencia * phi)


def eco_ressonante(x, phi=PHI, n_eco=3):
    """
    Eco puro — sem extração de modulação.
    Retorna o sinal convergido após n_eco ciclos de ressonância φ.
    Útil como pré-função: filtra dado antes de qualquer processamento.

    Dado com estrutura φ-coerente converge.
    Ruído (sem coerência) diverge e é amortecido por φ.

    Substrate-agnostic: texto, áudio, imagem, série temporal.
    """
    x = np.asarray(x, dtype=float)
    sinal = x.copy()

    for _ in range(n_eco):
        freq      = np.fft.fft(sinal, axis=-1)
        amplitude = np.abs(freq)
        fase      = np.angle(freq)

        nova_fase      = fase * phi
        sinal_complexo = amplitude * np.exp(1j * nova_fase)
        reflexao       = np.real(np.fft.ifft(sinal_complexo, axis=-1))

        eco   = reflexao - x
        sinal = sinal + (eco / phi)

    return sinal


# ── Eco Adaptativo — pré-função da pré-função (Eco 0) ────────────────────
#
# Arquitetura em 3 camadas — a cidade antes do endereço antes da casa:
#
#   analisar_campo(x)          → Pré-função 0: entende o sistema/campo
#   selecionar_parametros(...) → Pré-função 1: determina os parâmetros de filtro
#   eco_adaptativo(x)          → Pré-função 2: filtra o dado com parâmetros do campo
#
# O eco_ressonante original usa φ fixo — não conhece o sistema onde opera.
# O eco_adaptativo conhece o campo primeiro, depois escolhe o parâmetro.

def analisar_campo(x):
    """
    Pré-função 0 — analisa o campo/sistema antes de qualquer filtro.
    Caracteriza o espectro de frequência e retorna métricas do substrato.

    Retorna dict com:
      H_alpha       — entropia espectral normalizada (0=concentrado, 1=disperso)
      conc_espectral — fração de energia nos componentes dominantes
      n_amostras, n_dim — dimensões do dado
    """
    x = np.asarray(x, dtype=float)
    freq    = np.fft.fft(x, axis=-1)
    energia = np.abs(freq)
    e_norm  = np.clip(energia / (energia.sum(axis=-1, keepdims=True) + 1e-8), 1e-10, 1.0)
    H       = -np.sum(e_norm * np.log(e_norm), axis=-1)
    H_alpha = float(np.clip(H / max(np.log(x.shape[-1]), LOG_ALPHA), 0.0, 1.0).mean())

    k      = max(1, x.shape[-1] // 8)
    e_sort = np.sort(energia, axis=-1)[:, ::-1]
    conc   = float((e_sort[:, :k].sum(axis=-1) / (energia.sum(axis=-1) + 1e-8)).mean())

    return {
        'H_alpha':        H_alpha,
        'conc_espectral': conc,
        'n_amostras':     x.shape[0],
        'n_dim':          x.shape[-1],
    }


def selecionar_parametros(campo_info):
    """
    Pré-função 1 — seleciona parâmetros de filtro com base na análise do campo.

    H_alpha baixo  (< 0.3) → espectro concentrado → φ já presente → reforça com φ
    H_alpha médio  (< 0.6) → φ modulado por α     → ajuste fino
    H_alpha alto   (≥ 0.6) → espectro disperso    → φ² como força restauradora maior

    Retorna dict com fator_fase, fator_eco, n_eco, modo.
    """
    H    = campo_info['H_alpha']
    conc = campo_info['conc_espectral']

    if H < 0.3:
        return {'fator_fase': PHI,        'fator_eco': PHI, 'n_eco': 2, 'modo': 'phi'}
    elif H < 0.6:
        ajuste = PHI * (1.0 + ALPHA * (1.0 - conc))
        return {'fator_fase': ajuste,     'fator_eco': PHI, 'n_eco': 3, 'modo': 'phi_alpha'}
    else:
        return {'fator_fase': PHI ** 2,   'fator_eco': PHI, 'n_eco': 5, 'modo': 'phi2'}


def eco_adaptativo(x, params=None):
    """
    Pré-função 2 — eco-ressonante com parâmetros determinados pelo campo.

    Se params não fornecido: chama analisar_campo → selecionar_parametros primeiro.
    Retorna: (sinal_convergido, params_usados)

    Substrate-agnostic. Parâmetro não é fixo — emerge do substrato.
    """
    x = np.asarray(x, dtype=float)

    if params is None:
        campo  = analisar_campo(x)
        params = selecionar_parametros(campo)

    fator_fase = params['fator_fase']
    fator_eco  = params['fator_eco']
    n          = params['n_eco']

    sinal = x.copy()
    for _ in range(n):
        freq           = np.fft.fft(sinal, axis=-1)
        amplitude      = np.abs(freq)
        fase           = np.angle(freq)
        nova_fase      = fase * fator_fase
        sinal_complexo = amplitude * np.exp(1j * nova_fase)
        reflexao       = np.real(np.fft.ifft(sinal_complexo, axis=-1))
        eco            = reflexao - x
        sinal          = sinal + (eco / fator_eco)

    return sinal, params


# ── Ativações ─────────────────────────────────────────────────────────────
def golden_activation(x, phi=PHI):
    """Ativação φ — versão euclidiana."""
    return phi * np.tanh(x / phi)

def golden_activation_deriv(x, phi=PHI):
    return 1.0 - np.tanh(x / phi)**2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# ── Gradiente ─────────────────────────────────────────────────────────────
def clip_grad(g, max_norm=1.0):
    norm = np.linalg.norm(g)
    if norm > max_norm:
        g = g * max_norm / norm
    return g

# ── Arquitetura Fibonacci ─────────────────────────────────────────────────
def fibonacci_sequence(n_terms, start=55, phi=PHI):
    fibs = [start]
    a, b = start, int(round(start * phi))
    for _ in range(n_terms - 1):
        fibs.append(b)
        a, b = b, int(a + b)
    return fibs

# ── Espaço Hiperbólico ────────────────────────────────────────────────────
def expmap0(v, c=C_PHI):
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.clip(v_norm, 1e-8, None)
    tanh_v = np.tanh(np.clip(np.sqrt(c) * v_norm, -15, 15))
    return tanh_v * v / (np.sqrt(c) * v_norm)

def logmap0(y, c=C_PHI):
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    max_norm = (1.0/np.sqrt(c)) - 1e-5
    y_norm = np.clip(y_norm, 1e-8, max_norm)
    return np.arctanh(np.clip(np.sqrt(c) * y_norm, -1+1e-8, 1-1e-8)) * y / (np.sqrt(c) * y_norm)

def conformal_factor(x, c=C_PHI):
    x_norm2 = np.sum(x**2, axis=-1, keepdims=True)
    x_norm2 = np.clip(x_norm2, 0, (1.0/c) - 1e-5)
    return 2.0 / (1.0 - c * x_norm2 + 1e-8)

def normalize_activation(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + 1e-8) * np.sqrt(x.shape[-1])

def golden_activation_hyperbolic(x, c=C_PHI, phi=PHI):
    """Ativação φ — versão hiperbólica (opera no espaço de Poincaré)."""
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x_norm = np.clip(x_norm, 1e-8, None)
    novo_raio = phi * np.tanh(x_norm / phi)
    max_norm  = (1.0/np.sqrt(c)) - 1e-5
    novo_raio = np.clip(novo_raio, 1e-8, max_norm)
    return novo_raio * x / x_norm

# ── Campo Transmorfo — metamorfose geométrica Euclidiano → Hiperbólico ───
#
# Isomorfismo do bordado:
#   lattice central (malha de losangos) → espaço euclidiano    c ≈ 0
#   laços de transição (figura-∞)       → zona de curvatura emergente
#   espirais externas                   → espaço hiperbólico pleno  c = C_PHI
#
# "Transmorfo": o espaço muda de forma — não é perturbação, é metamorfose.
# O fio é contínuo — sem corte. A transição é progressiva.
# expmap0 direto = corte abrupto = quebra de gradiente = obstáculo BERT.
# campo_transmorfo = fio contínuo = gradiente preservado.

def curvatura_progressiva(layer_idx, total_layers, c_target=C_PHI, phi=PHI):
    """
    Agenda de curvatura por camada: c cresce de 0 até c_target.
    Modulada por φ: lenta no início (respeita geometria pré-existente),
    rápida no fim (entrega espaço hiperbólico pleno).

    Bordado: lattice central (layer 0) → espirais externas (layer final).
    """
    t = layer_idx / max(total_layers - 1, 1)  # [0, 1]
    t_phi = t ** phi                            # aceleração modulada por φ
    return c_target * t_phi


def campo_transmorfo(x, layer_idx, total_layers, c_target=C_PHI, phi=PHI):
    """
    Transição suave de um ponto x do espaço euclidiano ao hiperbólico.

    Hipótese: em vez de projetar abruptamente (expmap0 no final),
    cada camada aplica uma curvatura crescente modulada por φ.
    Isso preserva a geometria pré-estabelecida (ex: BERT) nas camadas
    iniciais e introduz curvatura gradualmente nas camadas finais.

    Retorna x_transformado no espaço com curvatura c(layer_idx).
    """
    c = curvatura_progressiva(layer_idx, total_layers, c_target, phi)

    if c < 1e-6:
        return x, c  # camada inicial: euclidiano puro

    x_hyp = expmap0(x, c=c)

    # Interpolação linear suave entre euclidiano e hiperbólico
    # alpha = 0 → puro euclidiano  |  alpha = 1 → puro hiperbólico
    alpha = (layer_idx / max(total_layers - 1, 1)) ** phi
    x_out = (1.0 - alpha) * x + alpha * x_hyp

    return x_out, c


def campo_transmorfo_inverso(x, layer_idx, total_layers, c_target=C_PHI, phi=PHI):
    """
    Retorno do espaço hiperbólico ao euclidiano (para backprop ou leitura).
    Inverso de campo_transmorfo: logmap0 com curvatura progressiva.
    """
    c = curvatura_progressiva(layer_idx, total_layers, c_target, phi)

    if c < 1e-6:
        return x

    x_euclid = logmap0(x, c=c)
    alpha = (layer_idx / max(total_layers - 1, 1)) ** phi
    return (1.0 - alpha) * x + alpha * x_euclid


# ── Paleta de cores para plots ────────────────────────────────────────────
PLOT_COLORS = {
    "gold":  "#DAA520",
    "gold2": "#FF8C00",
    "blue":  "#4169E1",
    "gray":  "#888888",
    "bg":    "#0d1117",
    "panel": "#161b22",
    "text":  "#8B949E",
    "title": "#E6EDF3",
}

def apply_dark_style(fig, axes):
    """Aplica estilo escuro padrão Alpha Phi aos plots."""
    fig.patch.set_facecolor(PLOT_COLORS["bg"])
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PLOT_COLORS["panel"])
        ax.tick_params(colors=PLOT_COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color("#30363d")
