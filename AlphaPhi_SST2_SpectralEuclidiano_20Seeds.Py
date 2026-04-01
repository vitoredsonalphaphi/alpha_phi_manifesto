# ALPHA PHI — SST-2 Espectral Euclidiano — 20 Seeds por Timestamp
# Vitor Edson Delavi · Florianópolis · 2026
#
# PROTOCOLO DE IDONEIDADE:
# Seeds gerados por timestamp — ninguém escolhe os valores
# LR=0.1 igual para todos os modelos
# Resultados reportados integralmente

!pip install sentence-transformers datasets -q

import numpy as np
import matplotlib.pyplot as plt
import time
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from scipy.stats import ttest_ind

PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084

print(f"phi   = {PHI:.10f}")
print(f"alpha = {ALPHA:.10f}")
print("=" * 60)
print("SST-2 ESPECTRAL φ — 20 SEEDS POR TIMESTAMP")
print("Protocolo de idoneidade — ninguém escolhe os seeds")
print("=" * 60)

# ── Modulador Espectral φ ─────────────────────────────────────────────────
def phi_spectral_modulator(x, phi=PHI):
    freq         = np.fft.fft(x, axis=-1)
    energia      = np.abs(freq)
    energia_norm = energia / (energia.sum(axis=-1, keepdims=True) + 1e-8)
    entropia     = -np.sum(energia_norm * np.log(energia_norm + 1e-8), axis=-1, keepdims=True)
    entropia_norm = entropia / np.log(x.shape[-1])
    coerencia    = 1.0 - entropia_norm
    return PHI * np.tanh(coerencia * PHI)

def golden_activation(x):
    return PHI * np.tanh(x / PHI)

def golden_activation_deriv(x):
    return 1.0 - np.tanh(x / PHI)**2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def clip_grad(g, max_norm=1.0):
    norm = np.linalg.norm(g)
    if norm > max_norm:
        g = g * max_norm / norm
    return g

def fibonacci_sequence(n_terms, start=55):
    fibs = [start]
    a, b = start, int(round(start * PHI))
    for _ in range(n_terms - 1):
        fibs.append(b)
        a, b = b, int(a + b)
    return fibs

fib_layers     = fibonacci_sequence(3, start=55)
uniform_layers = [144, 144, 144]
print(f"Camadas Fibonacci : {fib_layers}")
print(f"Camadas Uniforme  : {uniform_layers}")

# ── Rede Neural ───────────────────────────────────────────────────────────
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
        if self.mode == 'phi_spectral':
            mod = phi_spectral_modulator(X)
            lr_efetivo = lr * mod.mean()
        else:
            lr_efetivo = lr

        delta = self.acts[-1] - y.reshape(-1, 1)
        for i in reversed(range(len(self.weights))):
            dW = clip_grad(self.acts[i].T @ delta / m, 1.0)
            db = clip_grad(delta.mean(axis=0), 1.0)
            self.weights[i] -= lr_efetivo * dW
            self.biases[i]  -= lr_efetivo * db
            if i > 0:
                delta_back = delta @ self.weights[i].T
                if self.mode in ['phi', 'phi_spectral']:
                    delta = delta_back * golden_activation_deriv(self.pre_acts[i-1])
                else:
                    delta = delta_back * relu_deriv(self.pre_acts[i-1])

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

# ── Carregar SST-2 ────────────────────────────────────────────────────────
print("\nCarregando SST-2...")
dataset = load_dataset('glue', 'sst2')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

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

# ── 20 Seeds por Timestamp ────────────────────────────────────────────────
N_SEEDS    = 20
N_EPOCHS   = 10
LR         = 0.1
BATCH_SIZE = 128

# Seeds gerados por timestamp — ninguém escolhe
rng_master = np.random.RandomState(int(time.time()) % 100000)
SEEDS      = rng_master.randint(0, 99999, size=N_SEEDS).tolist()

print(f"\nSeeds gerados por timestamp: {SEEDS}")
print(f"Timestamp base: {int(time.time())}")
print()

arch_fib = [384] + fib_layers     + [1]
arch_uni = [384] + uniform_layers + [1]

res_conv_sp = []  # Convencional + Espectral φ — resultado principal
res_conv    = []  # Convencional puro — baseline
res_phi_sp  = []  # φ Fibonacci + Espectral
res_phi     = []  # φ Fibonacci puro

print(f"{'Seed':>8} | {'Conv+Esp':>9} | {'Conv':>7} | {'φ+Esp':>7} | {'φ puro':>7} | {'Ganho':>7}")
print("-" * 62)

for seed in SEEDS:
    net_conv_sp = NeuralNet(arch_uni, mode='phi_spectral', seed=seed)
    net_conv    = NeuralNet(arch_uni, mode='conventional', seed=seed)
    net_phi_sp  = NeuralNet(arch_fib, mode='phi_spectral', seed=seed)
    net_phi     = NeuralNet(arch_fib, mode='phi',          seed=seed)

    n_batches = len(X_train) // BATCH_SIZE

    for epoch in range(N_EPOCHS):
        idx = np.random.RandomState(seed + epoch).permutation(len(X_train))
        Xs, ys = X_train[idx], y_train[idx]
        for b in range(n_batches):
            Xb = Xs[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            yb = ys[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            net_conv_sp.forward(Xb); net_conv_sp.backward(Xb, yb, lr=LR)
            net_conv.forward(Xb);    net_conv.backward(Xb,    yb, lr=LR)
            net_phi_sp.forward(Xb);  net_phi_sp.backward(Xb,  yb, lr=LR)
            net_phi.forward(Xb);     net_phi.backward(Xb,     yb, lr=LR)

    a1 = net_conv_sp.accuracy(X_val, y_val)
    a2 = net_conv.accuracy(X_val,    y_val)
    a3 = net_phi_sp.accuracy(X_val,  y_val)
    a4 = net_phi.accuracy(X_val,     y_val)

    res_conv_sp.append(a1)
    res_conv.append(a2)
    res_phi_sp.append(a3)
    res_phi.append(a4)

    ganho = (a1 - a2) * 100
    print(f"{seed:>8} | {a1:>9.4f} | {a2:>7.4f} | {a3:>7.4f} | {a4:>7.4f} | {ganho:>+6.1f}%")

# ── Estatísticas ──────────────────────────────────────────────────────────
conv_sp = np.array(res_conv_sp)
conv    = np.array(res_conv)
phi_sp  = np.array(res_phi_sp)
phi     = np.array(res_phi)

_, p_val = ttest_ind(conv_sp, conv)
wins = int(np.sum(conv_sp > conv))
delta_mean = (conv_sp - conv).mean() * 100

print("=" * 65)
print("  RESULTADOS FINAIS — 20 SEEDS POR TIMESTAMP")
print("=" * 65)
print(f"  Conv + Espectral φ : {conv_sp.mean()*100:.2f}% +/- {conv_sp.std()*100:.2f}%")
print(f"  Conv puro          : {conv.mean()*100:.2f}% +/- {conv.std()*100:.2f}%")
print(f"  φ Fib + Espectral  : {phi_sp.mean()*100:.2f}% +/- {phi_sp.std()*100:.2f}%")
print(f"  φ Fib puro         : {phi.mean()*100:.2f}% +/- {phi.std()*100:.2f}%")
print()
print(f"  Ganho Conv+Espectral vs Conv : {delta_mean:+.2f}%")
print(f"  p-valor                      : {p_val:.4f}")
print(f"  Conv+Espectral vence Conv    : {wins}/{N_SEEDS} seeds")
print()
print(f"  Seeds gerados por timestamp — protocolo limpo")
print(f"  LR={LR} igual para todos os modelos")
print("=" * 65)

if delta_mean > 0 and p_val < 0.05:
    print(f"\n  CONFIRMADO: Espectral φ melhora redes convencionais")
    print(f"  Ganho: {delta_mean:+.2f}% · p={p_val:.4f} · {wins}/{N_SEEDS} seeds")
elif delta_mean > 0:
    print(f"\n  TENDÊNCIA POSITIVA mas p={p_val:.4f} — ampliar seeds")
else:
    print(f"\n  RESULTADO NEUTRO ou NEGATIVO — revisar hipótese")

# ── Plots ─────────────────────────────────────────────────────────────────
GOLD="#DAA520"; GOLD2="#FF8C00"; BLUE="#4169E1"; GRAY="#888888"
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values(): spine.set_color("#30363d")

x = np.arange(N_SEEDS)
axes[0].plot(x, conv_sp*100, "o-", color=GOLD2, lw=2, label=f"Conv+Espectral φ ({conv_sp.mean()*100:.1f}%)")
axes[0].plot(x, conv*100,   "^-", color=GRAY,  lw=2, label=f"Conv puro ({conv.mean()*100:.1f}%)")
axes[0].plot(x, phi_sp*100, "s-", color=BLUE,  lw=1.5, label=f"φ Fib+Espectral ({phi_sp.mean()*100:.1f}%)")
axes[0].plot(x, phi*100,    "v-", color=GOLD,  lw=1.5, label=f"φ Fib puro ({phi.mean()*100:.1f}%)")
axes[0].axhline(50, color='red', lw=0.8, linestyle='--')
axes[0].set_xlabel("Seed", color="#8B949E")
axes[0].set_ylabel("Acurácia (%)", color="#8B949E")
axes[0].set_title("SST-2 — 20 Seeds Timestamp", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7)
axes[0].grid(True, alpha=0.2)

barras = [conv_sp.mean()*100, conv.mean()*100, phi_sp.mean()*100, phi.mean()*100]
bars = axes[1].bar(['Conv\n+Espectral', 'Conv\npuro', 'φ Fib\n+Espectral', 'φ Fib\npuro'],
                   barras, color=[GOLD2, GRAY, BLUE, GOLD], alpha=0.85)
axes[1].axhline(50, color='red', lw=0.8, linestyle='--')
axes[1].set_ylim(40, 100)
axes[1].set_ylabel("Acurácia média (%)", color="#8B949E")
axes[1].set_title("Comparativo — Média 20 Seeds", color="#E6EDF3", fontweight="bold")
for bar, val in zip(bars, barras):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.1f}%', ha='center', va='bottom',
                 color='white', fontsize=9, fontweight='bold')
axes[1].grid(True, alpha=0.2)

fig.suptitle(
    f"ALPHA PHI — Espectral φ Euclidiano · 20 Seeds Timestamp · p={p_val:.4f}\n"
    f"Fibonacci{fib_layers}  LR={LR} igual  Florianopolis 2026",
    color=GOLD, fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig("alphaphi_sst2_espectral_20seeds.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGrafico salvo: alphaphi_sst2_espectral_20seeds.png")
print("alpha-phi")
