# ALPHA PHI — SST-2 Modulação Espectral φ no Espaço Euclidiano
# Vitor Edson Delavi · Florianópolis · 2026
#
# Pergunta central:
# A modulação espectral φ — que funcionou no hiperbólico —
# melhora também redes no espaço euclidiano convencional?
#
# Se sim: aplicação imediata em qualquer rede existente.
# Sem precisar reconstruir nada.

!pip install sentence-transformers datasets -q

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084

print(f"phi   = {PHI:.10f}")
print(f"alpha = {ALPHA:.10f}")
print("=" * 60)
print("SST-2 — Modulação Espectral φ no Espaço Euclidiano")
print("Pergunta: φ-espectral melhora redes convencionais?")
print("=" * 60)

# ── Modulador Espectral φ ─────────────────────────────────────────────────
def phi_spectral_modulator(x, phi=PHI):
    """
    Assinatura vibracional do dado.
    Frequência informacional — independente do substrato físico.
    Análogo ao campo morfogenético de Levin aplicado ao dado.
    """
    freq         = np.fft.fft(x, axis=-1)
    energia      = np.abs(freq)
    energia_norm = energia / (energia.sum(axis=-1, keepdims=True) + 1e-8)
    entropia     = -np.sum(energia_norm * np.log(energia_norm + 1e-8), axis=-1, keepdims=True)
    entropia_norm = entropia / np.log(x.shape[-1])
    coerencia    = 1.0 - entropia_norm
    return PHI * np.tanh(coerencia * PHI)

# ── Ativações ─────────────────────────────────────────────────────────────
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
n_batches = len(X_train) // BATCH_SIZE

print(f"\n{'Época':>5} | {'φ+Espectral':>11} | {'φ Puro':>8} | {'Conv+Esp':>9} | {'Conv':>7}")
print("-" * 52)

for epoch in range(1, N_EPOCHS+1):
    idx = np.random.permutation(len(X_train))
    Xs, ys = X_train[idx], y_train[idx]
    for b in range(n_batches):
        Xb = Xs[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        yb = ys[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
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

# ── Plots ─────────────────────────────────────────────────────────────────
GOLD="#DAA520"; GOLD2="#FF8C00"; BLUE="#4169E1"; GRAY="#888888"
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values(): spine.set_color("#30363d")

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
axes[0].set_xlabel("Época", color="#8B949E")
axes[0].set_ylabel("Acurácia (%)", color="#8B949E")
axes[0].set_title("SST-2 — Espectral φ Euclidiano", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=7)
axes[0].grid(True, alpha=0.2)

barras = [f1*100, f2*100, f3*100, f4*100]
cores  = [BLUE, GOLD, GOLD2, GRAY]
labels = ['φ Fib\n+Espectral','φ Fib\npuro','Conv\n+Espectral','Conv\npuro']
bars   = axes[1].bar(labels, barras, color=cores, alpha=0.85)
axes[1].axhline(50, color='red', lw=0.8, linestyle='--')
axes[1].set_ylim(40, 100)
axes[1].set_ylabel("Acurácia (%)", color="#8B949E")
axes[1].set_title("Comparativo Final", color="#E6EDF3", fontweight="bold")
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
plt.savefig("alphaphi_sst2_espectral_euclidiano.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGrafico salvo: alphaphi_sst2_espectral_euclidiano.png")
print("alpha-phi")
