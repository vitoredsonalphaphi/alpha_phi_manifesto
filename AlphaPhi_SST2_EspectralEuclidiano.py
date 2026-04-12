# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

# ALPHA PHI — SST-2 Modulação Espectral φ no Espaço Euclidiano
# Vitor Edson Delavi · Florianópolis · 2026
#
# Pergunta central:
# A modulação espectral φ — que funcionou no hiperbólico —
# melhora também redes no espaço euclidiano convencional?
#
# Se sim: aplicação imediata em qualquer rede existente.
# Sem precisar reconstruir nada.

import logging
import numpy as np
import matplotlib.pyplot as plt

from utils_phi import (
    PHI, ALPHA,
    phi_spectral_modulator,
    golden_activation, golden_activation_deriv,
    relu, relu_deriv, sigmoid, clip_grad,
    fibonacci_sequence,
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

print(f"phi   = {PHI:.10f}")
print(f"alpha = {ALPHA:.10f}  (constante de estrutura fina — 1/137)")
print("=" * 60)
print("SST-2 — Modulação Espectral φ no Espaço Euclidiano")
print("Pergunta: φ-espectral melhora redes convencionais?")
print("=" * 60)

fib_layers     = fibonacci_sequence(3, start=55)
uniform_layers = [144, 144, 144]
print(f"Camadas Fibonacci : {fib_layers}")
print(f"Camadas Uniforme  : {uniform_layers}")

# ── Redes ─────────────────────────────────────────────────────────────────
class NeuralNet:
    def __init__(self, layer_sizes, mode='phi', seed=42):
        np.random.seed(seed)
        self.mode = mode
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes)-1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            if mode in ['phi', 'phi_spectral']:
                scale = np.sqrt(1.0 / (fan_in * PHI))
            else:
                scale = np.sqrt(2.0 / fan_in)
            self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros(fan_out))

    def forward(self, X):
        self.pre_acts = []
        self.acts     = [X]
        cur = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            is_out = (i == len(self.weights)-1)
            z = cur @ W + b
            if is_out:
                cur = sigmoid(z)
            elif self.mode in ['phi', 'phi_spectral']:
                cur = golden_activation(z)
            else:
                cur = relu(z)
            self.pre_acts.append(z)
            self.acts.append(cur)
        return cur

    def backward(self, X, y, lr=0.1):
        m = X.shape[0]

        # Modulação espectral φ — calibra LR pela frequência do dado
        if self.mode == 'phi_spectral':
            mod = phi_spectral_modulator(X)
            lr_efetivo = lr * mod.mean()
        else:
            lr_efetivo = lr

        delta = self.acts[-1] - y.reshape(-1, 1)

        for i in reversed(range(len(self.weights))):
            dW = self.acts[i].T @ delta / m
            db = delta.mean(axis=0)

            dW = clip_grad(dW, 1.0)
            db = clip_grad(db, 1.0)

            self.weights[i] -= lr_efetivo * dW
            self.biases[i]  -= lr_efetivo * db

            if i > 0:
                delta_back = delta @ self.weights[i].T
                if self.mode in ['phi', 'phi_spectral']:
                    d_act = golden_activation_deriv(self.pre_acts[i-1])
                else:
                    d_act = relu_deriv(self.pre_acts[i-1])
                delta = delta_back * d_act

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def loss(self, X, y):
        out = np.clip(self.forward(X).flatten(), 1e-9, 1-1e-9)
        return -np.mean(y*np.log(out) + (1-y)*np.log(1-out))

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

# ── Carregar SST-2 ────────────────────────────────────────────────────────
print("\nCarregando SST-2...")
try:
    dataset = load_dataset('glue', 'sst2')
except Exception as e:
    raise RuntimeError(
        "Falha ao carregar SST-2. Verifique conexão com a internet ou cache HuggingFace."
    ) from e

try:
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    raise RuntimeError(
        "Falha ao carregar SentenceTransformer. Verifique conexão ou memória disponível."
    ) from e

N_TRAIN = 5000
print(f"Gerando embeddings ({N_TRAIN} amostras)...")
X_train = encoder.encode(dataset['train']['sentence'][:N_TRAIN],
                         show_progress_bar=True, batch_size=64)
y_train = np.array(dataset['train']['label'][:N_TRAIN])
X_val   = encoder.encode(dataset['validation']['sentence'],
                         show_progress_bar=True, batch_size=64)
y_val   = np.array(dataset['validation']['label'])

mean = X_train.mean(0); std = X_train.std(0) + 1e-8
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std

print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

# ── Diagnóstico espectral ─────────────────────────────────────────────────
mods = phi_spectral_modulator(X_train[:100])
print(f"\nModulador φ-espectral:")
print(f"  Média={mods.mean():.4f}  Mín={mods.min():.4f}  Máx={mods.max():.4f}")
print(f"  LR efetivo médio com base=0.1: {0.1*mods.mean():.4f}")

# ── Treino ────────────────────────────────────────────────────────────────
INPUT_DIM  = 384
N_EPOCHS   = 20
LR         = 0.1
BATCH_SIZE = 128
SEED       = 137

arch_fib = [INPUT_DIM] + fib_layers     + [1]
arch_uni = [INPUT_DIM] + uniform_layers + [1]

# Quatro redes — todas com LR=0.1
net_phi_spec = NeuralNet(arch_fib, mode='phi_spectral', seed=SEED)  # φ + espectral
net_phi      = NeuralNet(arch_fib, mode='phi',          seed=SEED)  # φ puro
net_conv_sp  = NeuralNet(arch_uni, mode='phi_spectral', seed=SEED)  # conv + espectral
net_conv     = NeuralNet(arch_uni, mode='conventional', seed=SEED)  # conv puro

hist = {'phi_spec':[], 'phi':[], 'conv_sp':[], 'conv':[]}
n_batches = max(1, len(X_train) // BATCH_SIZE)

print(f"\n{'Época':>5} | {'φ+Espectral':>11} | {'φ Puro':>8} | {'Conv+Esp':>9} | {'Conv':>7}")
print("-" * 52)

for epoch in range(1, N_EPOCHS+1):
    idx = np.random.permutation(len(X_train))
    Xs, ys = X_train[idx], y_train[idx]
    for Xb, yb in zip(np.array_split(Xs, n_batches), np.array_split(ys, n_batches)):
        net_phi_spec.forward(Xb); net_phi_spec.backward(Xb, yb, lr=LR)
        net_phi.forward(Xb);      net_phi.backward(Xb,      yb, lr=LR)
        net_conv_sp.forward(Xb);  net_conv_sp.backward(Xb,  yb, lr=LR)
        net_conv.forward(Xb);     net_conv.backward(Xb,     yb, lr=LR)

    a1 = net_phi_spec.accuracy(X_val, y_val)
    a2 = net_phi.accuracy(X_val, y_val)
    a3 = net_conv_sp.accuracy(X_val, y_val)
    a4 = net_conv.accuracy(X_val, y_val)

    hist['phi_spec'].append(a1)
    hist['phi'].append(a2)
    hist['conv_sp'].append(a3)
    hist['conv'].append(a4)
    print(f"{epoch:>5} | {a1:>11.4f} | {a2:>8.4f} | {a3:>9.4f} | {a4:>7.4f}")

# ── Resultados ────────────────────────────────────────────────────────────
f1 = net_phi_spec.accuracy(X_val, y_val)
f2 = net_phi.accuracy(X_val, y_val)
f3 = net_conv_sp.accuracy(X_val, y_val)
f4 = net_conv.accuracy(X_val, y_val)

print("=" * 65)
print("  RESULTADOS — ESPECTRAL φ NO ESPAÇO EUCLIDIANO")
print("=" * 65)
print(f"  φ Fibonacci + Espectral φ : {f1*100:.2f}%  ← nova proposta")
print(f"  φ Fibonacci puro          : {f2*100:.2f}%")
print(f"  Convencional + Espectral φ: {f3*100:.2f}%  ← espectral em conv")
print(f"  Convencional puro         : {f4*100:.2f}%")
print()
print(f"  Ganho espectral em φ Fib  : {(f1-f2)*100:+.2f}%")
print(f"  Ganho espectral em Conv   : {(f3-f4)*100:+.2f}%")
print(f"  φ Fib + Espectral vs Conv : {(f1-f4)*100:+.2f}%")
print("=" * 65)

if f3 > f4:
    print("\n  MODULAÇÃO ESPECTRAL φ MELHORA REDES CONVENCIONAIS!")
    print("  → Aplicação imediata em qualquer arquitetura existente")
else:
    print("\n  Modulação espectral φ não melhora redes convencionais")
    print("  → Benefício específico para arquiteturas φ-nativas")

# ── Exportação JSON ───────────────────────────────────────────────────────
import json, datetime
resultados = {
    "experimento": "AlphaPhi_SST2_EspectralEuclidiano",
    "data": datetime.datetime.now().isoformat(),
    "hiperparametros": {"n_epochs": N_EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "seed": SEED},
    "acuracia_final": {
        "phi_fibonacci_espectral": round(f1, 6),
        "phi_fibonacci_puro": round(f2, 6),
        "convencional_espectral": round(f3, 6),
        "convencional_puro": round(f4, 6),
    },
    "historico": {k: [round(v, 6) for v in vs] for k, vs in hist.items()},
}
json_path = "alphaphi_sst2_espectral_euclidiano.json"
try:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"Resultados salvos: {json_path}")
except OSError as e:
    logging.warning("Falha ao salvar JSON: %s", e)

# ── Plots ─────────────────────────────────────────────────────────────────
GOLD  = PLOT_COLORS["gold"]
GOLD2 = PLOT_COLORS["gold2"]
BLUE  = PLOT_COLORS["blue"]
GRAY  = PLOT_COLORS["gray"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
apply_dark_style(fig, axes)

epochs_x = np.arange(1, N_EPOCHS+1)
axes[0].plot(epochs_x, [v*100 for v in hist['phi_spec']],
             "o-", color=BLUE,  lw=2.5, label=f"φ Fib + Espectral ({f1*100:.1f}%)")
axes[0].plot(epochs_x, [v*100 for v in hist['phi']],
             "s-", color=GOLD,  lw=1.5, label=f"φ Fib puro ({f2*100:.1f}%)")
axes[0].plot(epochs_x, [v*100 for v in hist['conv_sp']],
             "^-", color=GOLD2, lw=1.5, label=f"Conv + Espectral ({f3*100:.1f}%)")
axes[0].plot(epochs_x, [v*100 for v in hist['conv']],
             "v-", color=GRAY,  lw=1.5, label=f"Conv puro ({f4*100:.1f}%)")
axes[0].axhline(50, color='red', lw=0.8, linestyle='--')
axes[0].set_xlabel("Época", color=PLOT_COLORS["text"])
axes[0].set_ylabel("Acurácia (%)", color=PLOT_COLORS["text"])
axes[0].set_title("SST-2 — Espectral φ Euclidiano", color=PLOT_COLORS["title"], fontweight="bold")
axes[0].legend(facecolor=PLOT_COLORS["panel"], labelcolor=PLOT_COLORS["title"], fontsize=7)
axes[0].grid(True, alpha=0.2)

barras = [f1*100, f2*100, f3*100, f4*100]
cores  = [BLUE, GOLD, GOLD2, GRAY]
labels = ['φ Fib\n+Espectral','φ Fib\npuro','Conv\n+Espectral','Conv\npuro']
bars   = axes[1].bar(labels, barras, color=cores, alpha=0.85)
axes[1].axhline(50, color='red', lw=0.8, linestyle='--')
axes[1].set_ylim(40, 100)
axes[1].set_ylabel("Acurácia (%)", color=PLOT_COLORS["text"])
axes[1].set_title("Comparativo Final", color=PLOT_COLORS["title"], fontweight="bold")
for bar, val in zip(bars, barras):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.1f}%', ha='center', va='bottom',
                 color='white', fontsize=9, fontweight='bold')
axes[1].grid(True, alpha=0.2)

fig.suptitle(
    f"ALPHA PHI — Espectral φ Euclidiano · LR={LR} igual para todos\n"
    f"Fibonacci{fib_layers}  Seed={SEED}  Florianopolis 2026",
    color=GOLD, fontsize=10, fontweight="bold"
)
plt.tight_layout()
png_path = "alphaphi_sst2_espectral_euclidiano.png"
try:
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=PLOT_COLORS["bg"])
    print(f"Grafico salvo: {png_path}")
except OSError as e:
    logging.warning("Falha ao salvar gráfico: %s", e)
plt.show()
print("alpha-phi")
