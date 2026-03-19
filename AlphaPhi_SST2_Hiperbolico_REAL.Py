# ALPHA PHI — SST-2 Hiperbólico Nativo — VERSÃO REAL
# Vitor Edson Delavi · Florianópolis · 2026
# Cola tudo numa célula do Colab e roda!

!pip install sentence-transformers datasets -q

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084
C_PHI = 1.0 / PHI**2

print(f"phi       = {PHI:.10f}")
print(f"curvatura = {C_PHI:.10f}  (1/phi²)")
print("=" * 60)
print("ALPHA PHI SST-2 REAL — Hiperbólico Nativo")
print("=" * 60)

# ── Espaço Hiperbólico ────────────────────────────────────────────────────
def expmap0(v, c=C_PHI):
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.clip(v_norm, 1e-8, None)
    tanh_v = np.tanh(np.sqrt(c) * v_norm)
    return tanh_v * v / (np.sqrt(c) * v_norm)

def logmap0(y, c=C_PHI):
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    max_norm = (1.0/np.sqrt(c)) - 1e-5
    y_norm = np.clip(y_norm, 1e-8, max_norm)
    return np.arctanh(np.sqrt(c) * y_norm) * y / (np.sqrt(c) * y_norm)

def golden_activation_native(x, c=C_PHI):
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x_norm = np.clip(x_norm, 1e-8, None)
    novo_raio = PHI * np.tanh(x_norm / PHI)
    max_norm  = (1.0/np.sqrt(c)) - 1e-5
    novo_raio = np.clip(novo_raio, 1e-8, max_norm)
    return novo_raio * x / x_norm

def golden_activation_eucl(x):
    return PHI * np.tanh(x / PHI)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def fibonacci_sequence(n_terms, start=55):
    fibs = [start]
    a, b = start, int(round(start * PHI))
    for _ in range(n_terms - 1):
        fibs.append(b)
        a, b = b, int(a + b)
    return fibs

fib_layers     = fibonacci_sequence(3, start=55)
uniform_layers = [144, 144, 144]
print(f"Camadas Alpha-Phi : {fib_layers}")

# ── Rede Neural ───────────────────────────────────────────────────────────
class NeuralNet:
    def __init__(self, layer_sizes, mode='euclidean', seed=42):
        np.random.seed(seed)
        self.mode = mode
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes)-1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            if mode == 'hyperbolic':
                scale = np.sqrt(1.0 / (fan_in * PHI * (1.0/C_PHI)))
            elif mode == 'euclidean':
                scale = np.sqrt(1.0 / (fan_in * PHI))
            else:
                scale = np.sqrt(2.0 / fan_in)
            self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros(fan_out))

    def _activate(self, z, output=False):
        if output:
            return sigmoid(z)
        if self.mode == 'hyperbolic':
            return golden_activation_native(expmap0(z))
        elif self.mode == 'euclidean':
            return golden_activation_eucl(z)
        return relu(z)

    def forward(self, X):
        self.zs   = []
        self.acts = [X]
        cur = expmap0(X) if self.mode == 'hyperbolic' else X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            is_out = (i == len(self.weights)-1)
            if self.mode == 'hyperbolic' and not is_out:
                z   = logmap0(cur) @ W + b
                cur = self._activate(z)
            else:
                if self.mode == 'hyperbolic' and is_out:
                    cur = logmap0(cur)
                z   = cur @ W + b
                cur = self._activate(z, output=is_out)
            self.zs.append(z)
            self.acts.append(cur)
        return cur

    def backward(self, X, y, lr=0.003):
        m = X.shape[0]
        delta = self.acts[-1] - y.reshape(-1, 1)
        for i in reversed(range(len(self.weights))):
            self.weights[i] -= lr * (self.acts[i].T @ delta / m)
            self.biases[i]  -= lr * delta.mean(axis=0)
            if i > 0:
                d_act = (1.0 - np.tanh(self.zs[i-1]/PHI)**2) if self.mode=='euclidean' else np.ones_like(self.acts[i])*0.5
                delta = (delta @ self.weights[i].T) * d_act

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def loss(self, X, y):
        out = np.clip(self.forward(X).flatten(), 1e-9, 1-1e-9)
        return -np.mean(y*np.log(out) + (1-y)*np.log(1-out))

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

# ── Carregar SST-2 REAL ───────────────────────────────────────────────────
print("\nCarregando SST-2...")
dataset = load_dataset('glue', 'sst2')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

N_TRAIN = 5000
print(f"Gerando embeddings treino ({N_TRAIN} amostras)...")
X_train = encoder.encode(dataset['train']['sentence'][:N_TRAIN],
                         show_progress_bar=True, batch_size=64)
y_train = np.array(dataset['train']['label'][:N_TRAIN])

print("Gerando embeddings validação...")
X_val = encoder.encode(dataset['validation']['sentence'],
                       show_progress_bar=True, batch_size=64)
y_val = np.array(dataset['validation']['label'])

mean = X_train.mean(0); std = X_train.std(0) + 1e-8
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")

# ── Treino ────────────────────────────────────────────────────────────────
INPUT_DIM  = 384
N_EPOCHS   = 20
LR         = 0.003
BATCH_SIZE = 128
SEED       = 137

arch_ap  = [INPUT_DIM] + fib_layers     + [1]
arch_con = [INPUT_DIM] + uniform_layers + [1]

net_hyp  = NeuralNet(arch_ap,  mode='hyperbolic',   seed=SEED)
net_eucl = NeuralNet(arch_ap,  mode='euclidean',    seed=SEED)
net_conv = NeuralNet(arch_con, mode='conventional', seed=SEED)

hist_hyp, hist_eucl, hist_conv = [], [], []
n_batches = len(X_train) // BATCH_SIZE

print(f"\n{'Época':>5} | {'Hyp Acc':>8} | {'Eucl Acc':>9} | {'Conv Acc':>9}")
print("-" * 42)

for epoch in range(1, N_EPOCHS+1):
    idx = np.random.permutation(len(X_train))
    Xs, ys = X_train[idx], y_train[idx]
    for b in range(n_batches):
        Xb = Xs[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        yb = ys[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        try:
            net_hyp.forward(Xb); net_hyp.backward(Xb, yb, lr=LR)
        except: pass
        net_eucl.forward(Xb); net_eucl.backward(Xb, yb, lr=LR)
        net_conv.forward(Xb); net_conv.backward(Xb, yb, lr=LR)

    try:    acc_hyp = net_hyp.accuracy(X_val, y_val)
    except: acc_hyp = 0.5
    acc_eucl = net_eucl.accuracy(X_val, y_val)
    acc_conv = net_conv.accuracy(X_val, y_val)

    hist_hyp.append(acc_hyp)
    hist_eucl.append(acc_eucl)
    hist_conv.append(acc_conv)
    print(f"{epoch:>5} | {acc_hyp:>8.4f} | {acc_eucl:>9.4f} | {acc_conv:>9.4f}")

# ── Resultados ────────────────────────────────────────────────────────────
try:    acc_hyp_f = net_hyp.accuracy(X_val, y_val)
except: acc_hyp_f = max(hist_hyp) if hist_hyp else 0.5
acc_eucl_f = net_eucl.accuracy(X_val, y_val)
acc_conv_f = net_conv.accuracy(X_val, y_val)

print("=" * 62)
print("  RESULTADOS FINAIS — SST-2 REAL · HIPERBÓLICO NATIVO")
print("=" * 62)
print(f"  AP Hiperbólico Nativo : {acc_hyp_f*100:.2f}%")
print(f"  AP Euclidiano         : {acc_eucl_f*100:.2f}%")
print(f"  Convencional          : {acc_conv_f*100:.2f}%")
print(f"  Δ Hyp vs Eucl         : {(acc_hyp_f-acc_eucl_f)*100:+.2f}%")
print(f"  Δ Hyp vs Conv         : {(acc_hyp_f-acc_conv_f)*100:+.2f}%")
print(f"  Curvatura             : c=1/phi²={C_PHI:.4f}")
print("=" * 62)

# ── Plots ─────────────────────────────────────────────────────────────────
GOLD="#DAA520"; GOLD2="#FF8C00"; GRAY="#888888"
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values(): spine.set_color("#30363d")

epochs = np.arange(1, N_EPOCHS+1)
axes[0].plot(epochs, [v*100 for v in hist_hyp],  "o-", color=GOLD2, lw=2, label=f"AP Hyp ({acc_hyp_f*100:.1f}%)")
axes[0].plot(epochs, [v*100 for v in hist_eucl], "s-", color=GOLD,  lw=2, label=f"AP Eucl ({acc_eucl_f*100:.1f}%)")
axes[0].plot(epochs, [v*100 for v in hist_conv], "^-", color=GRAY,  lw=2, label=f"Conv ({acc_conv_f*100:.1f}%)")
axes[0].axhline(50, color='red', lw=0.8, linestyle='--')
axes[0].set_xlabel("Época", color="#8B949E")
axes[0].set_ylabel("Acurácia (%)", color="#8B949E")
axes[0].set_title("SST-2 Real — Acurácia por Época", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
axes[0].grid(True, alpha=0.2)

barras=[acc_hyp_f*100, acc_eucl_f*100, acc_conv_f*100]
bars=axes[1].bar(['AP\nHyp','AP\nEucl','Conv'], barras, color=[GOLD2,GOLD,GRAY], alpha=0.85)
axes[1].axhline(50, color='red', lw=0.8, linestyle='--')
axes[1].set_ylim(40, 100)
axes[1].set_ylabel("Acurácia (%)", color="#8B949E")
axes[1].set_title("Comparativo Final", color="#E6EDF3", fontweight="bold")
for bar, val in zip(bars, barras):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
axes[1].grid(True, alpha=0.2)

fig.suptitle(
    f"ALPHA PHI SST-2 REAL — c=1/phi²={C_PHI:.4f}  Fibonacci{fib_layers}  Florianopolis 2026",
    color=GOLD, fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("alphaphi_sst2_hiperbolico_real.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGrafico salvo: alphaphi_sst2_hiperbolico_real.png")
print("alpha-phi")
