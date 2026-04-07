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
