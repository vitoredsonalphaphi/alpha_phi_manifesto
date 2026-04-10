# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

# ALPHA PHI — Estudo de Ablação Completo
# Vitor Edson Delavi · Florianópolis · 2026
#
# Pergunta central:
# Qual eixo do Alpha-Phi carrega o efeito?
#   - Arquitetura Fibonacci?
#   - Ativação φ·tanh(x/φ)?
#   - Modulação espectral φ?
#   - Modulação espectral aleatória?
#   - Curvatura hiperbólica c=1/φ²?
#   - Todos juntos?
#
# Motivação:
# Experimentos BERT (v4, v6) mostraram que modulação espectral φ ≈ aleatória.
# Este estudo isola cada contribuição em rede treinada do zero (sem pré-treino),
# para distinguir o papel geométrico de φ do papel como constante escalar.
#
# Protocolo:
# - N_SEEDS seeds por timestamp — ninguém escolhe os valores
# - φ, α, 137 aparecem apenas na arquitetura/ativação — nunca como parâmetros de teste
# - Resultados reportados integralmente — favoráveis ou não

import logging
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from utils_phi import (
    PHI, ALPHA, C_PHI,
    phi_spectral_modulator,
    golden_activation, golden_activation_deriv,
    golden_activation_hyperbolic,
    relu, relu_deriv, sigmoid, clip_grad,
    fibonacci_sequence,
    expmap0, logmap0, conformal_factor, normalize_activation,
    PLOT_COLORS, apply_dark_style,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers não instalado. Execute: pip install sentence-transformers"
    ) from e

try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError(
        "datasets não instalado. Execute: pip install datasets"
    ) from e

print(f"φ = {PHI:.10f}")
print(f"α = {ALPHA:.10f}  (1/137 — granularidade mínima)")
print(f"c = {C_PHI:.10f}  (1/φ² — curvatura hiperbólica nativa)")
print("=" * 70)
print("ABLAÇÃO COMPLETA — Isolando cada eixo do Alpha-Phi")
print("Redes treinadas do zero · SST-2 · MiniLM embeddings")
print("=" * 70)

# ── Configurações dos 7 experimentos ─────────────────────────────────────
#
# Matriz de ablação:
#   Config | Arquitetura | Ativação     | Modulação   | Geometria
#   -------|-------------|--------------|-------------|----------
#   A      | Fibonacci   | φ·tanh       | nenhuma     | Euclidiana  ← geometria pura
#   B      | Uniforme    | φ·tanh       | nenhuma     | Euclidiana  ← só ativação
#   C      | Uniforme    | ReLU         | φ-espectral | Euclidiana  ← só modulação φ
#   D      | Uniforme    | ReLU         | aleatória   | Euclidiana  ← controle modulação
#   E      | Uniforme    | ReLU         | nenhuma     | c=1/φ²      ← só curvatura
#   F      | Fibonacci   | φ·tanh       | φ-espectral | c=1/φ²      ← tudo junto
#   G      | Uniforme    | ReLU         | nenhuma     | Euclidiana  ← baseline

CONFIGS = {
    'A_geo_phi':      {'arch': 'fibonacci',  'activation': 'golden', 'modulation': 'none',    'geometry': 'euclidean',
                       'label': 'A: Fibonacci + φ·tanh',           'color': PLOT_COLORS['gold']},
    'B_act_phi':      {'arch': 'uniform',    'activation': 'golden', 'modulation': 'none',    'geometry': 'euclidean',
                       'label': 'B: Uniforme + φ·tanh',            'color': PLOT_COLORS['gold2']},
    'C_mod_phi':      {'arch': 'uniform',    'activation': 'relu',   'modulation': 'phi',     'geometry': 'euclidean',
                       'label': 'C: Modulação φ-espectral',        'color': PLOT_COLORS['blue']},
    'D_mod_random':   {'arch': 'uniform',    'activation': 'relu',   'modulation': 'random',  'geometry': 'euclidean',
                       'label': 'D: Modulação aleatória (controle)','color': '#9B59B6'},
    'E_curv_phi':     {'arch': 'uniform',    'activation': 'relu',   'modulation': 'none',    'geometry': 'hyperbolic',
                       'label': 'E: Curvatura c=1/φ²',             'color': '#E74C3C'},
    'F_full_phi':     {'arch': 'fibonacci',  'activation': 'golden', 'modulation': 'phi',     'geometry': 'hyperbolic',
                       'label': 'F: Todos os eixos φ',             'color': '#2ECC71'},
    'G_baseline':     {'arch': 'uniform',    'activation': 'relu',   'modulation': 'none',    'geometry': 'euclidean',
                       'label': 'G: Baseline convencional',        'color': PLOT_COLORS['gray']},
}

# ── Rede Neural com suporte a todos os modos ─────────────────────────────
class AblationNet:
    def __init__(self, layer_sizes, config, seed=42):
        np.random.seed(seed)
        self.config   = config
        self.geometry = config['geometry']
        self.act_type = config['activation']
        self.mod_type = config['modulation']
        self.weights  = []
        self.biases   = []
        self._spectral_mod_cache = None

        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            if self.geometry == 'hyperbolic':
                scale    = np.sqrt(2.0 / fan_in) * 0.5
                W        = np.random.randn(fan_in, fan_out) * scale
                norm     = np.linalg.norm(W, axis=0, keepdims=True)
                max_norm = (1.0 / np.sqrt(C_PHI)) * 0.9
                W        = W * np.minimum(1.0, max_norm / np.maximum(norm, 1e-8))
            elif self.act_type == 'golden':
                scale = np.sqrt(1.0 / (fan_in * PHI))
                W     = np.random.randn(fan_in, fan_out) * scale
            else:
                scale = np.sqrt(2.0 / fan_in)
                W     = np.random.randn(fan_in, fan_out) * scale

            self.weights.append(W)
            self.biases.append(np.zeros(fan_out))

    def _activate(self, z, is_out):
        if is_out:
            return sigmoid(z)
        if self.act_type == 'golden':
            return golden_activation(z)
        return relu(z)

    def _activate_deriv(self, z):
        if self.act_type == 'golden':
            return golden_activation_deriv(z)
        return relu_deriv(z)

    def forward(self, X):
        self.pre_acts = []
        self.acts     = [X]

        # Cache da modulação espectral — evita recomputar FFT no backward
        if self.mod_type == 'phi':
            self._spectral_mod_cache = phi_spectral_modulator(X)
        elif self.mod_type == 'random':
            # Controle: escalar aleatório na mesma faixa do modulador φ
            # Faixa observada: [PHI*tanh(0), PHI*tanh(PHI)] ≈ [0.76, 1.52]
            rng_mod = np.random.uniform(0.76, 1.52, size=(X.shape[0], 1))
            self._spectral_mod_cache = rng_mod
        else:
            self._spectral_mod_cache = None

        cur = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            is_out = (i == len(self.weights) - 1)

            if self.geometry == 'hyperbolic' and not is_out:
                x_tang  = logmap0(cur)
                z       = x_tang @ W + b
                z_hyp   = expmap0(z)
                cur     = golden_activation_hyperbolic(z_hyp)
                cur_log = normalize_activation(logmap0(cur))
                cur     = expmap0(cur_log)
            elif self.geometry == 'hyperbolic' and is_out:
                cur = logmap0(cur)
                z   = cur @ W + b
                cur = sigmoid(z)
            else:
                z   = cur @ W + b
                cur = self._activate(z, is_out)

            self.pre_acts.append(z)
            self.acts.append(cur)
        return cur

    def backward(self, X, y, lr=0.1):
        m = X.shape[0]

        # Learning rate efetivo — usa cache do forward
        if self._spectral_mod_cache is not None:
            lr_eff = lr * float(self._spectral_mod_cache.mean())
        else:
            lr_eff = lr

        delta = self.acts[-1] - y.reshape(-1, 1)

        for i in reversed(range(len(self.weights))):
            is_out = (i == len(self.weights) - 1)
            dW     = self.acts[i].T @ delta / m
            db     = delta.mean(axis=0)
            dW     = clip_grad(dW, 1.0)
            db     = clip_grad(db, 1.0)

            if self.geometry == 'hyperbolic' and not is_out:
                lam    = conformal_factor(self.weights[i])
                grad_R = clip_grad(dW * (lam.mean() ** 0.5), 1.0)
                self.weights[i] -= lr_eff * grad_R
                norm = np.linalg.norm(self.weights[i], axis=0, keepdims=True)
                max_norm = (1.0 / np.sqrt(C_PHI)) * 0.9
                self.weights[i] *= np.minimum(1.0, max_norm / np.maximum(norm, 1e-8))
            else:
                self.weights[i] -= lr_eff * dW

            self.biases[i] -= lr_eff * clip_grad(db, 1.0)

            if i > 0:
                delta_back = delta @ self.weights[i].T
                if self.geometry == 'hyperbolic':
                    lam   = conformal_factor(self.acts[i])
                    d_act = 1.0 - np.tanh(
                        np.linalg.norm(self.acts[i], axis=-1, keepdims=True) / PHI) ** 2
                    delta = clip_grad(delta_back * d_act * (lam ** 0.5 + 1e-8), 10.0)
                else:
                    delta = delta_back * self._activate_deriv(self.pre_acts[i - 1])

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ── Carregar dados ────────────────────────────────────────────────────────
print("\nCarregando SST-2...")
try:
    dataset = load_dataset('glue', 'sst2')
except Exception as e:
    raise RuntimeError("Falha ao carregar SST-2. Verifique conexão com a internet.") from e

try:
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    raise RuntimeError("Falha ao carregar SentenceTransformer.") from e

N_TRAIN = 5000
print(f"Gerando embeddings ({N_TRAIN} amostras)...")
X_train_raw = encoder.encode(dataset['train']['sentence'][:N_TRAIN],
                              show_progress_bar=True, batch_size=64)
y_train     = np.array(dataset['train']['label'][:N_TRAIN])
X_val_raw   = encoder.encode(dataset['validation']['sentence'],
                              show_progress_bar=True, batch_size=64)
y_val       = np.array(dataset['validation']['label'])

mean = X_train_raw.mean(0); std = X_train_raw.std(0) + 1e-8
X_train = (X_train_raw - mean) / std
X_val   = (X_val_raw   - mean) / std
print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

# ── Hiperparâmetros ───────────────────────────────────────────────────────
INPUT_DIM  = 384
N_EPOCHS   = 20
LR         = 0.1
BATCH_SIZE = 128
N_SEEDS    = 10   # aumentar para 20 para publicação

fib_layers     = fibonacci_sequence(3, start=55)  # [55, 89, 144]
uniform_layers = [144, 144, 144]

print(f"\nFibonacci: {fib_layers}")
print(f"Uniforme : {uniform_layers}")
print(f"Seeds    : {N_SEEDS} (por timestamp)")
print(f"Configs  : {len(CONFIGS)}")
print("=" * 70)


def get_layer_sizes(arch, input_dim=INPUT_DIM):
    hidden = fib_layers if arch == 'fibonacci' else uniform_layers
    return [input_dim] + hidden + [1]


def train_and_eval(config_key, seed):
    cfg     = CONFIGS[config_key]
    layers  = get_layer_sizes(cfg['arch'])
    net     = AblationNet(layers, cfg, seed=seed)
    n_batch = max(1, len(X_train) // BATCH_SIZE)
    history = []

    for epoch in range(N_EPOCHS):
        idx    = np.random.RandomState(seed + epoch).permutation(len(X_train))
        Xs, ys = X_train[idx], y_train[idx]
        for Xb, yb in zip(np.array_split(Xs, n_batch), np.array_split(ys, n_batch)):
            try:
                net.forward(Xb)
                net.backward(Xb, yb, lr=LR)
            except (ValueError, RuntimeError, FloatingPointError) as e:
                logging.warning("Batch falhou [%s seed=%d epoch=%d]: %s",
                                config_key, seed, epoch, e)
        try:
            acc = net.accuracy(X_val, y_val)
        except (ValueError, RuntimeError, FloatingPointError):
            acc = 0.5
        history.append(acc)

    return history


# ── Experimento principal ─────────────────────────────────────────────────
print("\nGerando seeds por timestamp...")
seeds = [int(time.time() * 1000) % 100000 + i * 1337 for i in range(N_SEEDS)]
print(f"Seeds: {seeds}")
print()

results = {k: [] for k in CONFIGS}   # lista de acurácias finais por config
history = {k: [] for k in CONFIGS}   # histórico de épocas por seed

for s_idx, seed in enumerate(seeds):
    print(f"── Seed {s_idx+1}/{N_SEEDS} ({seed}) ──────────────────────────")
    row = {}
    for key in CONFIGS:
        hist   = train_and_eval(key, seed)
        final  = hist[-1]
        results[key].append(final)
        history[key].append(hist)
        row[key] = final
        print(f"  {CONFIGS[key]['label'][:35]:35s}: {final*100:.2f}%")
    print()

# ── Estatísticas ──────────────────────────────────────────────────────────
print("=" * 70)
print("  RESULTADOS — ABLAÇÃO COMPLETA")
print("=" * 70)

stats_out = {}
for key, cfg in CONFIGS.items():
    vals     = results[key]
    mu       = np.mean(vals)
    sigma    = np.std(vals)
    stats_out[key] = {'mean': mu, 'std': sigma, 'values': vals}
    print(f"  {cfg['label'][:40]:40s}: {mu*100:.2f}% ± {sigma*100:.2f}%")

print()
print("── Testes t pareados vs Baseline (G) ──────────────────────────────")
baseline = results['G_baseline']
for key in CONFIGS:
    if key == 'G_baseline':
        continue
    t_stat, p_val = stats.ttest_rel(results[key], baseline)
    delta  = np.mean(results[key]) - np.mean(baseline)
    marker = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
    print(f"  {CONFIGS[key]['label'][:35]:35s} vs G: Δ={delta*100:+.2f}%  p={p_val:.4f}  {marker}")

print()
print("── Pergunta chave: C (mod φ) vs D (mod aleatória) ─────────────────")
t_cd, p_cd = stats.ttest_rel(results['C_mod_phi'], results['D_mod_random'])
delta_cd   = np.mean(results['C_mod_phi']) - np.mean(results['D_mod_random'])
print(f"  C vs D: Δ={delta_cd*100:+.3f}%  p={p_cd:.4f}  "
      f"{'φ ≠ aleatório' if p_cd < 0.05 else 'φ ≈ aleatório'}")

print()
print("── Pergunta chave: A (Fibonacci+φ·tanh) vs B (Uniforme+φ·tanh) ───")
t_ab, p_ab = stats.ttest_rel(results['A_geo_phi'], results['B_act_phi'])
delta_ab   = np.mean(results['A_geo_phi']) - np.mean(results['B_act_phi'])
print(f"  A vs B: Δ={delta_ab*100:+.3f}%  p={p_ab:.4f}  "
      f"{'Fibonacci ≠ uniforme' if p_ab < 0.05 else 'Fibonacci ≈ uniforme com mesma ativação'}")

print()
print("── Pergunta chave: F (tudo φ) vs G (baseline) ─────────────────────")
t_fg, p_fg = stats.ttest_rel(results['F_full_phi'], results['G_baseline'])
delta_fg   = np.mean(results['F_full_phi']) - np.mean(results['G_baseline'])
print(f"  F vs G: Δ={delta_fg*100:+.3f}%  p={p_fg:.4f}  "
      f"{'Combinação φ supera baseline' if p_fg < 0.05 else 'Combinação φ ≈ baseline'}")

# ── Exportação JSON ───────────────────────────────────────────────────────
export = {
    "experimento": "AlphaPhi_Ablation_Study",
    "data": datetime.datetime.now().isoformat(),
    "seeds": seeds,
    "hiperparametros": {
        "n_epochs": N_EPOCHS, "lr": LR,
        "batch_size": BATCH_SIZE, "n_seeds": N_SEEDS,
    },
    "configuracoes": {k: v['label'] for k, v in CONFIGS.items()},
    "resultados": {
        k: {
            "mean": round(float(stats_out[k]['mean']), 6),
            "std":  round(float(stats_out[k]['std']),  6),
            "values": [round(float(v), 6) for v in stats_out[k]['values']],
        }
        for k in CONFIGS
    },
    "testes_estatisticos": {
        "C_vs_D_phi_vs_random": {"delta": round(float(delta_cd), 6), "p": round(float(p_cd), 4)},
        "A_vs_B_fibonacci_vs_uniform": {"delta": round(float(delta_ab), 6), "p": round(float(p_ab), 4)},
        "F_vs_G_full_phi_vs_baseline": {"delta": round(float(delta_fg), 6), "p": round(float(p_fg), 4)},
    },
}
json_path = "alphaphi_ablation_results.json"
try:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos: {json_path}")
except OSError as e:
    logging.warning("Falha ao salvar JSON: %s", e)

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
apply_dark_style(fig, axes)

# Plot 1 — Barras de acurácia final com desvio padrão
means  = [stats_out[k]['mean'] * 100 for k in CONFIGS]
stds   = [stats_out[k]['std']  * 100 for k in CONFIGS]
colors = [CONFIGS[k]['color']        for k in CONFIGS]
labels = [CONFIGS[k]['label'].split(':')[0] for k in CONFIGS]  # só a letra

bars = axes[0].bar(labels, means, color=colors, alpha=0.85,
                   yerr=stds, capsize=5, error_kw={'color': 'white', 'linewidth': 1.5})
axes[0].axhline(stats_out['G_baseline']['mean'] * 100,
                color='white', lw=0.8, linestyle='--', alpha=0.5)
axes[0].set_ylim(min(means) - 3, max(means) + 3)
axes[0].set_ylabel("Acurácia (%)", color=PLOT_COLORS['text'])
axes[0].set_title("Acurácia Final por Configuração", color=PLOT_COLORS['title'], fontweight='bold')
for bar, val, std in zip(bars, means, stds):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                 f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=7)
axes[0].grid(True, alpha=0.2)

# Plot 2 — Histórico de épocas (média sobre seeds)
epochs_x = np.arange(1, N_EPOCHS + 1)
for key, cfg in CONFIGS.items():
    mean_hist = np.mean(history[key], axis=0) * 100
    axes[1].plot(epochs_x, mean_hist, "-", color=cfg['color'],
                 lw=2 if key in ['F_full_phi', 'G_baseline'] else 1.2,
                 alpha=0.9, label=cfg['label'].split(':')[0])
axes[1].axhline(50, color='red', lw=0.6, linestyle='--')
axes[1].set_xlabel("Época", color=PLOT_COLORS['text'])
axes[1].set_ylabel("Acurácia (%)", color=PLOT_COLORS['text'])
axes[1].set_title("Convergência por Configuração", color=PLOT_COLORS['title'], fontweight='bold')
axes[1].legend(facecolor=PLOT_COLORS['panel'], labelcolor=PLOT_COLORS['title'], fontsize=7)
axes[1].grid(True, alpha=0.2)

# Plot 3 — C vs D: φ vs aleatório (a pergunta chave)
axes[2].scatter(results['D_mod_random'],
                results['C_mod_phi'],
                color=PLOT_COLORS['blue'], alpha=0.8, s=60, zorder=3)
lim_vals = results['C_mod_phi'] + results['D_mod_random']
lim_min  = min(lim_vals) - 0.005
lim_max  = max(lim_vals) + 0.005
axes[2].plot([lim_min, lim_max], [lim_min, lim_max],
             color='white', lw=0.8, linestyle='--', alpha=0.5, label='φ = aleatório')
axes[2].set_xlabel("Modulação Aleatória (D)", color=PLOT_COLORS['text'])
axes[2].set_ylabel("Modulação φ (C)",         color=PLOT_COLORS['text'])
axes[2].set_title(f"C vs D: p={p_cd:.3f}",   color=PLOT_COLORS['title'], fontweight='bold')
axes[2].legend(facecolor=PLOT_COLORS['panel'], labelcolor=PLOT_COLORS['title'], fontsize=8)
axes[2].grid(True, alpha=0.2)

fig.suptitle(
    f"ALPHA PHI — Ablação Completa · {N_SEEDS} seeds · LR={LR} · Florianópolis 2026\n"
    f"Fibonacci{fib_layers}  |  Substrato: MiniLM-L6-v2 sem pré-treino de fine-tuning",
    color=PLOT_COLORS['gold'], fontsize=10, fontweight='bold'
)
plt.tight_layout()
png_path = "alphaphi_ablation_study.png"
try:
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=PLOT_COLORS['bg'])
    print(f"Gráfico salvo: {png_path}")
except OSError as e:
    logging.warning("Falha ao salvar gráfico: %s", e)
plt.show()
print("\nα como threshold — próximo experimento:")
print(f"  np.clip(phi_spectral_modulator(x), ALPHA, PHI)")
print(f"  floor = {ALPHA:.6f}  ceiling = {PHI:.6f}")
print("alpha-phi")
