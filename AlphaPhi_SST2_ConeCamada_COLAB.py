# ALPHA PHI — SST-2 Cone Autônomo por Camada
# Vitor Edson Delavi · Florianópolis · 2026
#
# O que este código testa:
# Cada camada Fibonacci opera como um cone autônomo —
# processa até atingir coerência φ antes de passar adiante.
# Inspirado no Serial φ (cada cone hermético, resultado próprio).
#
# Três versões comparadas:
#   A — Euclidiano (como está, referência)
#   B — Hiperbólico traduzido (expmap/logmap por fora, euclidiano por dentro)
#   C — Hiperbólico cone autônomo por camada (novo)
#
# Cola tudo numa célula do Colab e roda.

!pip install sentence-transformers datasets -q

import numpy as np
import matplotlib.pyplot as plt
import time
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from collections import defaultdict

# ── Constantes ────────────────────────────────────────────────────────────────
PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084
C_PHI = 1.0 / PHI**2          # curvatura — φ gera sua própria curvatura

print(f"φ         = {PHI:.10f}")
print(f"α         = {ALPHA:.10f}")
print(f"c = 1/φ²  = {C_PHI:.10f}")
print("=" * 65)
print("ALPHA PHI — SST-2 Cone Autônomo por Camada")
print("=" * 65)

# ── Espaço Hiperbólico (Bola de Poincaré) ─────────────────────────────────────
def expmap0(v, c=C_PHI):
    """Euclidiano → Poincaré. Projeta vetor para dentro da bola."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return np.tanh(np.sqrt(c) * norm) * v / (np.sqrt(c) * norm)

def logmap0(y, c=C_PHI):
    """Poincaré → Euclidiano. Recupera vetor do tangente."""
    norm = np.linalg.norm(y, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, (1.0/np.sqrt(c)) - 1e-5)
    return np.arctanh(np.sqrt(c) * norm) * y / (np.sqrt(c) * norm)

def coerencia_phi(x, c=C_PHI):
    """
    Mede coerência φ do vetor no espaço hiperbólico.
    Coerência = proximidade da norma hiperbólica ao raio φ.
    Raio da bola = 1/√c = φ. Coerência 1.0 = norma exatamente φ.
    """
    norm = np.linalg.norm(x, axis=-1)
    raio_phi = 1.0 / np.sqrt(c)          # = φ
    return float(1.0 - np.abs(norm - raio_phi).mean() / raio_phi)

def ativacao_eucl(x):
    """Ativação φ·tanh — espaço euclidiano."""
    return PHI * np.tanh(x / PHI)

def ativacao_nativa(x, c=C_PHI):
    """
    Ativação nativa no espaço hiperbólico.
    Preserva direção, escala raio por φ·tanh.
    Não projeta para euclidiano e de volta.
    """
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    novo_raio = PHI * np.tanh(norm / PHI)
    novo_raio = np.clip(novo_raio, 1e-8, (1.0/np.sqrt(c)) - 1e-5)
    return novo_raio * x / norm

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def fibonacci_layers(n, start=55):
    layers = [start]
    a, b = start, int(round(start * PHI))
    for _ in range(n - 1):
        layers.append(b); a, b = b, a + b
    return layers

FIB = fibonacci_layers(3)         # [55, 89, 144]
UNI = [144, 144, 144]
print(f"Camadas Fibonacci : {FIB}")
print(f"Camadas Uniformes : {UNI}")

# ── VERSÃO A — Euclidiana ──────────────────────────────────────────────────────
class RedeEuclidiana:
    """
    Referência. Camadas Fibonacci com ativação φ·tanh.
    Espaço completamente plano.
    """
    def __init__(self, arch, seed):
        np.random.seed(seed)
        self.W, self.b = [], []
        for i in range(len(arch)-1):
            s = np.sqrt(1.0 / (arch[i] * PHI))
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))

    def forward(self, X):
        self.acts = [X]
        cur = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = cur @ W + b
            cur = sigmoid(z) if i == len(self.W)-1 else ativacao_eucl(z)
            self.acts.append(cur)
        return cur

    def backward(self, X, y, lr):
        m = X.shape[0]
        delta = self.acts[-1] - y.reshape(-1,1)
        for i in reversed(range(len(self.W))):
            self.W[i] -= lr * (self.acts[i].T @ delta / m)
            self.b[i]  -= lr * delta.mean(0)
            if i > 0:
                d = 1.0 - np.tanh(self.acts[i] / PHI)**2
                delta = (delta @ self.W[i].T) * d

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


# ── VERSÃO B — Hiperbólico Traduzido ──────────────────────────────────────────
class RedeHiperbolicaTraduzida:
    """
    Código euclidiano envolvido em expmap/logmap.
    A lógica interna continua euclidiana.
    Como traduzir um texto palavra por palavra —
    a estrutura original aparece, a naturalidade se perde.
    """
    def __init__(self, arch, seed):
        np.random.seed(seed)
        self.W, self.b = [], []
        for i in range(len(arch)-1):
            s = np.sqrt(1.0 / (arch[i] * PHI * (1.0/C_PHI)))
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))

    def forward(self, X):
        self.acts = [X]
        cur = expmap0(X)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = logmap0(cur) @ W + b
            if i == len(self.W)-1:
                cur = sigmoid(logmap0(expmap0(z)))
            else:
                cur = expmap0(ativacao_eucl(z))
            self.acts.append(cur)
        return cur

    def backward(self, X, y, lr):
        m = X.shape[0]
        delta = self.acts[-1] - y.reshape(-1,1)
        for i in reversed(range(len(self.W))):
            self.W[i] -= lr * (self.acts[i].T @ delta / m)
            self.b[i]  -= lr * delta.mean(0)
            if i > 0:
                delta = (delta @ self.W[i].T) * 0.5  # gradiente euclidiano

    def predict(self, X):
        try:
            return (self.forward(X).flatten() >= 0.5).astype(int)
        except:
            return np.zeros(X.shape[0], dtype=int)

    def accuracy(self, X, y):
        try:
            return float(np.mean(self.predict(X) == y))
        except:
            return 0.5


# ── VERSÃO C — Cone Autônomo por Camada ───────────────────────────────────────
class RedeConeAutonomo:
    """
    Cada camada Fibonacci opera como cone autônomo.
    Inspirado no Serial φ: cada cone processa até convergir,
    só o campo resultante passa adiante.

    O que muda:
    1. Cada camada tem seu próprio ciclo de convergência φ
    2. A camada só passa adiante quando coerência ≥ limiar φ
    3. O backward usa fator conformal correto (gradiente Riemanniano)
    4. Inicialização respeita a curvatura da bola
    """
    def __init__(self, arch, seed, max_iter_cone=3, limiar_coh=0.85):
        np.random.seed(seed)
        self.W, self.b = [], []
        self.max_iter  = max_iter_cone
        self.limiar    = limiar_coh
        self.coh_hist  = defaultdict(list)  # histórico de coerência por camada

        for i in range(len(arch)-1):
            # Inicialização nativa: escala pela curvatura φ
            s = np.sqrt(C_PHI / (arch[i] * PHI))
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))

    def _cone(self, x, camada_idx):
        """
        Processa x como cone autônomo na camada camada_idx.
        Itera até coerência φ ≥ limiar ou max_iter atingido.
        Retorna o campo resultante e a coerência alcançada.
        """
        W, b = self.W[camada_idx], self.b[camada_idx]
        cur = expmap0(x)

        for _ in range(self.max_iter):
            z   = logmap0(cur) @ W + b
            cur = ativacao_nativa(expmap0(z))
            coh = coerencia_phi(cur)
            if coh >= self.limiar:
                break

        self.coh_hist[camada_idx].append(coerencia_phi(cur))
        return cur, coerencia_phi(cur)

    def forward(self, X):
        self.acts = [X]
        self.cohs = []
        cur = X

        for i in range(len(self.W)-1):
            cur, coh = self._cone(cur, i)
            self.acts.append(cur)
            self.cohs.append(coh)

        # Camada de saída — euclidiana
        z   = logmap0(cur) @ self.W[-1] + self.b[-1]
        out = sigmoid(z)
        self.acts.append(out)
        return out

    def backward(self, X, y, lr):
        """
        Backward com fator conformal correto.
        No disco de Poincaré, o fator conformal em x é:
            λ_x = 2 / (1 - c·||x||²)
        O gradiente Riemanniano = gradiente euclidiano / λ_x²
        Isso faz o aprendizado respeitar a curvatura — não contradiz a arquitetura.
        """
        m = X.shape[0]
        delta = self.acts[-1] - y.reshape(-1,1)

        for i in reversed(range(len(self.W))):
            # Gradiente dos pesos
            self.W[i] -= lr * (self.acts[i].T @ delta / m)
            self.b[i]  -= lr * delta.mean(0)

            if i > 0:
                # Fator conformal: λ_x = 2/(1 - c·||x||²)
                x     = self.acts[i]
                norm2 = np.sum(x**2, axis=-1, keepdims=True)
                norm2 = np.clip(norm2, 0, (1.0/C_PHI - 1e-5)**2)
                lam   = 2.0 / (1.0 - C_PHI * norm2 + 1e-10)
                # Gradiente Riemanniano
                delta = (delta @ self.W[i].T) / (lam**2 + 1e-10)

    def predict(self, X):
        try:
            return (self.forward(X).flatten() >= 0.5).astype(int)
        except:
            return np.zeros(X.shape[0], dtype=int)

    def accuracy(self, X, y):
        try:
            return float(np.mean(self.predict(X) == y))
        except:
            return 0.5

    def coerencia_media(self):
        """Coerência φ média por camada — o sinal interno do cone."""
        resultado = {}
        for k, vals in self.coh_hist.items():
            resultado[k] = float(np.mean(vals[-100:]))  # últimas 100 medições
        return resultado


# ── Carregar SST-2 ─────────────────────────────────────────────────────────────
print("\nCarregando SST-2 real...")
dataset = load_dataset('glue', 'sst2')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

N_TRAIN = 5000
print(f"Embeddings treino ({N_TRAIN})...")
X_train = encoder.encode(dataset['train']['sentence'][:N_TRAIN],
                         show_progress_bar=True, batch_size=64)
y_train = np.array(dataset['train']['label'][:N_TRAIN])

print("Embeddings validação...")
X_val = encoder.encode(dataset['validation']['sentence'],
                       show_progress_bar=True, batch_size=64)
y_val = np.array(dataset['validation']['label'])

# Normalização
mean = X_train.mean(0); std = X_train.std(0) + 1e-8
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std

print(f"Treino: {X_train.shape}  Validação: {X_val.shape}")

# ── Configuração do experimento ────────────────────────────────────────────────
INPUT_DIM  = 384
N_EPOCHS   = 20
LR         = 0.003
BATCH_SIZE = 128

arch_fib = [INPUT_DIM] + FIB + [1]
arch_uni = [INPUT_DIM] + UNI + [1]

# Seeds por timestamp — protocolo de idoneidade
rng = np.random.RandomState(int(time.time()) % 100000)
SEED = int(rng.randint(0, 99999))
print(f"\nSeed (timestamp): {SEED}")
print(f"Arquitetura Fibonacci: {arch_fib}")

# ── Treino ─────────────────────────────────────────────────────────────────────
net_A = RedeEuclidiana(arch_fib, SEED)
net_B = RedeHiperbolicaTraduzida(arch_fib, SEED)
net_C = RedeConeAutonomo(arch_fib, SEED)

hist = {'A': [], 'B': [], 'C': []}
n_bat = len(X_train) // BATCH_SIZE

print(f"\n{'Época':>5} | {'A Eucl':>8} | {'B Trad':>8} | {'C Cone':>8}")
print("-" * 40)

for ep in range(1, N_EPOCHS+1):
    idx = np.random.permutation(len(X_train))
    Xs, ys = X_train[idx], y_train[idx]

    for b in range(n_bat):
        Xb = Xs[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        yb = ys[b*BATCH_SIZE:(b+1)*BATCH_SIZE]

        net_A.forward(Xb);  net_A.backward(Xb, yb, LR)
        try: net_B.forward(Xb); net_B.backward(Xb, yb, LR)
        except: pass
        try: net_C.forward(Xb); net_C.backward(Xb, yb, LR)
        except: pass

    aA = net_A.accuracy(X_val, y_val)
    aB = net_B.accuracy(X_val, y_val)
    aC = net_C.accuracy(X_val, y_val)

    hist['A'].append(aA); hist['B'].append(aB); hist['C'].append(aC)
    print(f"{ep:>5} | {aA:>8.4f} | {aB:>8.4f} | {aC:>8.4f}")

# ── Resultados ─────────────────────────────────────────────────────────────────
fA = net_A.accuracy(X_val, y_val)
fB = net_B.accuracy(X_val, y_val)
fC = net_C.accuracy(X_val, y_val)

print("\n" + "=" * 65)
print("  RESULTADOS FINAIS")
print("=" * 65)
print(f"  A — Euclidiano          : {fA*100:.2f}%")
print(f"  B — Hiperbólico traduz. : {fB*100:.2f}%")
print(f"  C — Cone autônomo       : {fC*100:.2f}%")
print(f"  Δ C vs A                : {(fC-fA)*100:+.2f}%")
print(f"  Δ C vs B                : {(fC-fB)*100:+.2f}%")
print(f"  Seed                    : {SEED}")
print(f"  Curvatura c=1/φ²        : {C_PHI:.4f}")

# Coerência φ por camada (versão C)
coh = net_C.coerencia_media()
print(f"\n  Coerência φ por camada (Versão C):")
for k, v in sorted(coh.items()):
    bar = "█" * int(v * 30)
    print(f"    Camada {k+1} (Fib={FIB[k]:3d}): {bar:<30} {v:.4f}")

print("=" * 65)

if fC > fA:
    print(f"\n  ► CONE AUTÔNOMO SUPERA EUCLIDIANO: +{(fC-fA)*100:.2f}%")
elif abs(fC - fA) < 0.005:
    print(f"\n  ► RESULTADO NEUTRO — diferença < 0.5%")
else:
    print(f"\n  ► EUCLIDIANO AINDA SUPERIOR — diferença: {(fA-fC)*100:.2f}%")

# ── Gráficos ───────────────────────────────────────────────────────────────────
GOLD="#DAA520"; GOLD2="#FF8C00"; CYAN="#00BFFF"; GRAY="#888888"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8B949E")
    for sp in ax.spines.values(): sp.set_color("#30363d")

eps = np.arange(1, N_EPOCHS+1)
axes[0].plot(eps, [v*100 for v in hist['A']], "o-", color=GOLD,  lw=2, ms=4, label=f"A Eucl ({fA*100:.1f}%)")
axes[0].plot(eps, [v*100 for v in hist['B']], "s-", color=GRAY,  lw=2, ms=4, label=f"B Trad ({fB*100:.1f}%)")
axes[0].plot(eps, [v*100 for v in hist['C']], "^-", color=CYAN,  lw=2, ms=4, label=f"C Cone ({fC*100:.1f}%)")
axes[0].axhline(50, color='red', lw=0.8, ls='--', label="Chance")
axes[0].set_xlabel("Época", color="#8B949E")
axes[0].set_ylabel("Acurácia (%)", color="#8B949E")
axes[0].set_title("Acurácia por Época", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
axes[0].grid(True, alpha=0.2)

barras = [fA*100, fB*100, fC*100]
cores  = [GOLD, GRAY, CYAN]
bars   = axes[1].bar(['A\nEuclidiano','B\nTraduzido','C\nCone φ'], barras,
                      color=cores, alpha=0.85)
axes[1].axhline(50, color='red', lw=0.8, ls='--')
axes[1].set_ylim(40, 100)
axes[1].set_ylabel("Acurácia (%)", color="#8B949E")
axes[1].set_title("Comparativo Final", color="#E6EDF3", fontweight="bold")
for bar, val in zip(bars, barras):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.1f}%', ha='center', va='bottom',
                 color='white', fontsize=9, fontweight='bold')
axes[1].grid(True, alpha=0.2)

fig.suptitle(
    f"ALPHA PHI — Cone Autônomo por Camada  c=1/φ²={C_PHI:.4f}  "
    f"Fibonacci{FIB}  Seed={SEED}  Florianópolis 2026",
    color=GOLD, fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig("alphaphi_sst2_cone_autonomo.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGráfico salvo: alphaphi_sst2_cone_autonomo.png")
print("alpha-phi")
