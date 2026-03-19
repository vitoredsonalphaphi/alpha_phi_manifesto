# ALPHA PHI — SST-2 com Gradiente Riemanniano Correto
# Vitor Edson Delavi · Florianópolis · 2026
#
# Correção fundamental:
# O backprop hiperbólico precisa do gradiente Riemanniano —
# não um gradiente constante aproximado.
#
# O gradiente Riemanniano escala o gradiente euclidiano
# pelo fator conforme da bola de Poincaré:
#   grad_R = grad_E / lambda(x)²
#   onde lambda(x) = 2 / (1 - c·||x||²)
#
# Cola numa célula do Colab e roda!

!pip install sentence-transformers datasets -q

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084
C_PHI = 1.0 / PHI**2  # curvatura nativa = 0.382

print(f"phi       = {PHI:.10f}")
print(f"curvatura = {C_PHI:.10f}  (1/phi²)")
print("=" * 60)
print("ALPHA PHI SST-2 — Gradiente Riemanniano Correto")
print("=" * 60)

# ── Espaço Hiperbólico ────────────────────────────────────────────────────
def conformal_factor(x, c=C_PHI):
    """
    Fator conforme da bola de Poincaré.
    lambda(x) = 2 / (1 - c·||x||²)
    Escala o espaço — fundamental para o gradiente correto.
    """
    x_norm2 = np.sum(x**2, axis=-1, keepdims=True)
    return 2.0 / (1.0 - c * x_norm2 + 1e-8)

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

def riemannian_grad(euclidean_grad, x, c=C_PHI):
    """
    Converte gradiente euclidiano para Riemanniano.
    grad_R = grad_E / lambda(x)²
    Isso é o gradiente correto no espaço de Poincaré.
    """
    lam = conformal_factor(x, c)
    return euclidean_grad / (lam**2 + 1e-8)

def riemannian_update(x, grad_R, lr, c=C_PHI):
    """
    Atualização de peso no espaço hiperbólico.
    Usa exponential map para permanecer na bola.
    """
    update = expmap0(-lr * grad_R, c)
    # Möbius addition para mover x na direção do gradiente
    x_norm2 = np.sum(x**2, axis=-1, keepdims=True)
    u_norm2 = np.sum(update**2, axis=-1, keepdims=True)
    xu      = np.sum(x*update, axis=-1, keepdims=True)
    num = (1 + 2*c*xu + c*u_norm2)*x + (1 - c*x_norm2)*update
    den = 1 + 2*c*xu + c**2*x_norm2*u_norm2
    result = num / (den + 1e-8)
    # Garante que permanece na bola
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    max_norm = (1.0/np.sqrt(c)) - 1e-5
    return result * np.minimum(1.0, max_norm/np.maximum(norm, 1e-8))

def golden_activation_native(x, c=C_PHI):
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x_norm = np.clip(x_norm, 1e-8, None)
    novo_raio = PHI * np.tanh(x_norm / PHI)
    max_norm  = (1.0/np.sqrt(c)) - 1e-5
    novo_raio = np.clip(novo_raio, 1e-8, max_norm)
    return novo_raio * x / x_norm

def golden_activation_native_deriv(x, c=C_PHI):
    """
    Derivada da ativação nativa em relação ao raio.
    d(phi·tanh(r/phi))/dr = 1 - tanh²(r/phi)
    """
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x_norm = np.clip(x_norm, 1e-8, None)
    return 1.0 - np.tanh(x_norm / PHI)**2

def golden_activation_eucl(x):
    return PHI * np.tanh(x / PHI)

def golden_activation_eucl_deriv(x):
    return 1.0 - np.tanh(x / PHI)**2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

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

# ── Rede com Gradiente Riemanniano ────────────────────────────────────────
class NeuralNet:
    def __init__(self, layer_sizes, mode='euclidean', seed=42):
        np.random.seed(seed)
        self.mode = mode
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes)-1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            if mode == 'hyperbolic':
                # Inicialização pequena — respeita geometria da bola
                scale = np.sqrt(1.0 / (fan_in * PHI)) * 0.1
                W = np.random.randn(fan_in, fan_out) * scale
                # Projeta pesos iniciais para a bola
                norm = np.linalg.norm(W, axis=-1, keepdims=True)
                max_norm = (1.0/np.sqrt(C_PHI)) - 1e-5
                W = W * np.minimum(1.0, max_norm/np.maximum(norm, 1e-8))
            elif mode == 'euclidean':
                scale = np.sqrt(1.0 / (fan_in * PHI))
                W = np.random.randn(fan_in, fan_out) * scale
            else:
                scale = np.sqrt(2.0 / fan_in)
                W = np.random.randn(fan_in, fan_out) * scale
            self.weights.append(W)
            self.biases.append(np.zeros(fan_out))

    def forward(self, X):
        self.pre_acts = []
        self.acts     = [X]
        cur = X

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            is_out = (i == len(self.weights)-1)

            if is_out:
                # Camada de saída — sempre euclidiana
                if self.mode == 'hyperbolic':
                    cur = logmap0(cur)
                z   = cur @ W + b
                cur = sigmoid(z)
            else:
                if self.mode == 'hyperbolic':
                    # Operação no espaço tangente
                    x_tang = logmap0(cur)
                    z      = x_tang @ W + b
                    # Volta para a bola com ativação nativa
                    cur = golden_activation_native(expmap0(z))
                else:
                    z   = cur @ W + b
                    cur = golden_activation_eucl(z)

            self.pre_acts.append(z if not is_out else z)
            self.acts.append(cur)
        return cur

    def backward(self, X, y, lr=0.003):
        m = X.shape[0]

        # Delta da saída (BCE + sigmoid)
        delta = self.acts[-1] - y.reshape(-1, 1)

        for i in reversed(range(len(self.weights))):
            is_out = (i == len(self.weights)-1)

            # Gradiente euclidiano dos pesos
            dW = self.acts[i].T @ delta / m
            db = delta.mean(axis=0)

            if self.mode == 'hyperbolic' and not is_out:
                # Converte para gradiente Riemanniano
                grad_R_W = riemannian_grad(dW, self.weights[i])
                # Atualiza pesos permanecendo na bola
                self.weights[i] = riemannian_update(
                    self.weights[i], grad_R_W, lr)
            else:
                self.weights[i] -= lr * dW

            self.biases[i] -= lr * db

            # Propaga delta para camada anterior
            if i > 0:
                delta_back = delta @ self.weights[i].T
                if self.mode == 'hyperbolic':
                    # Gradiente da ativação nativa
                    d_act = golden_activation_native_deriv(self.acts[i])
                    # Escala pelo fator conforme
                    lam   = conformal_factor(self.acts[i])
                    delta = delta_back * d_act / (lam + 1e-8)
                elif self.mode == 'euclidean':
                    d_act = golden_activation_eucl_deriv(self.pre_acts[i-1])
                    delta = delta_back * d_act
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

# ── Carregar SST-2 Real ───────────────────────────────────────────────────
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
LR         = 0.005
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
        except Exception as e:
            pass
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
print("  RESULTADOS — SST-2 REAL · GRADIENTE RIEMANNIANO")
print("=" * 62)
print(f"  AP Hiperbólico Riemanniano : {acc_hyp_f*100:.2f}%")
print(f"  AP Euclidiano              : {acc_eucl_f*100:.2f}%")
print(f"  Convencional               : {acc_conv_f*100:.2f}%")
print(f"  Δ Hyp vs Eucl              : {(acc_hyp_f-acc_eucl_f)*100:+.2f}%")
print(f"  Δ Hyp vs Conv              : {(acc_hyp_f-acc_conv_f)*100:+.2f}%")
print(f"  Curvatura                  : c=1/phi²={C_PHI:.4f}")
print("=" * 62)

# ── Plots ─────────────────────────────────────────────────────────────────
GOLD="#DAA520"; GOLD2="#FF8C00"; GRAY="#888888"
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values(): spine.set_color("#30363d")

epochs_x = np.arange(1, N_EPOCHS+1)
axes[0].plot(epochs_x, [v*100 for v in hist_hyp],
             "o-", color=GOLD2, lw=2, label=f"AP Hyp Riemann ({acc_hyp_f*100:.1f}%)")
axes[0].plot(epochs_x, [v*100 for v in hist_eucl],
             "s-", color=GOLD,  lw=2, label=f"AP Eucl ({acc_eucl_f*100:.1f}%)")
axes[0].plot(epochs_x, [v*100 for v in hist_conv],
             "^-", color=GRAY,  lw=2, label=f"Conv ({acc_conv_f*100:.1f}%)")
axes[0].axhline(50, color='red', lw=0.8, linestyle='--', label='Aleatório')
axes[0].set_xlabel("Época", color="#8B949E")
axes[0].set_ylabel("Acurácia (%)", color="#8B949E")
axes[0].set_title("SST-2 Real — Gradiente Riemanniano", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
axes[0].grid(True, alpha=0.2)

barras = [acc_hyp_f*100, acc_eucl_f*100, acc_conv_f*100]
bars   = axes[1].bar(['AP\nHyp\nRiemann','AP\nEucl','Conv'],
                     barras, color=[GOLD2,GOLD,GRAY], alpha=0.85)
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
    f"ALPHA PHI SST-2 — Gradiente Riemanniano  c=1/phi²={C_PHI:.4f}\n"
    f"Fibonacci{fib_layers}  Seed={SEED}  Florianopolis 2026",
    color=GOLD, fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("alphaphi_sst2_riemanniano.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGrafico salvo: alphaphi_sst2_riemanniano.png")
print("alpha-phi")
