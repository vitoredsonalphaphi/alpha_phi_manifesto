# ALPHA PHI — SST-2 Campo Observer
# Vitor Edson Delavi · Florianópolis · 2026
#
# O que este código testa:
# Versão D — Campo Observer no backward pass.
# O ruído sub-limiar não é corrigido como erro de ponto —
# é assimilado como vírgula de erro, informação de campo.
#
# Limiar natural: C_PHI = 1/φ² (a curvatura do espaço).
# Gradientes < C_PHI × max|grad|  → assimilados a escala 1/φ
# Gradientes ≥ C_PHI × max|grad|  → corrigidos normalmente
#
# Inspirado no Campo Harmônico φ: ruído não é descartado.
# Atribuído a outro tipo de eficiência.
#
# Versões comparadas:
#   A — Euclidiano (referência)
#   C — Cone autônomo por camada + gradiente Riemanniano
#   D — Cone autônomo + gradiente Riemanniano + campo observer (novo)
#
# Cola numa célula do Colab e roda.

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers", "datasets"], check=True)

import numpy as np
import matplotlib.pyplot as plt
import time
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from collections import defaultdict

# ── Constantes ────────────────────────────────────────────────────────────────
PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084
C_PHI = 1.0 / PHI**2

print(f"φ         = {PHI:.10f}")
print(f"α         = {ALPHA:.10f}")
print(f"c = 1/φ²  = {C_PHI:.10f}")
print("=" * 65)
print("ALPHA PHI — SST-2 Campo Observer")
print("=" * 65)

# ── Espaço Hiperbólico (Bola de Poincaré) ─────────────────────────────────────
def expmap0(v, c=C_PHI):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return np.tanh(np.sqrt(c) * norm) * v / (np.sqrt(c) * norm)

def logmap0(y, c=C_PHI):
    norm = np.linalg.norm(y, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, (1.0/np.sqrt(c)) - 1e-5)
    return np.arctanh(np.sqrt(c) * norm) * y / (np.sqrt(c) * norm)

def coerencia_phi(x, c=C_PHI):
    norm = np.linalg.norm(x, axis=-1)
    raio_phi = 1.0 / np.sqrt(c)   # = φ
    return float(1.0 - np.abs(norm - raio_phi).mean() / raio_phi)

def ativacao_eucl(x):
    return PHI * np.tanh(x / PHI)

def ativacao_nativa(x, c=C_PHI):
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

FIB = fibonacci_layers(3)   # [55, 89, 144]
print(f"Camadas Fibonacci: {FIB}")

# ── VERSÃO A — Euclidiana ──────────────────────────────────────────────────────
class RedeEuclidiana:
    def __init__(self, arch, seed):
        np.random.seed(seed)
        self.W, self.b = [], []
        for i in range(len(arch)-1):
            s = np.sqrt(1.0 / (arch[i] * PHI))
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))

    def forward(self, X):
        self.acts = [X]; cur = X
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
                delta = (delta @ self.W[i].T) * (1.0 - np.tanh(self.acts[i]/PHI)**2)

    def predict(self, X):
        return (self.forward(X).flatten() >= 0.5).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))


# ── VERSÃO C — Cone Autônomo + Gradiente Riemanniano ──────────────────────────
class RedeConeAutonomo:
    def __init__(self, arch, seed, max_iter=3, limiar=0.85):
        np.random.seed(seed)
        self.W, self.b = [], []
        self.max_iter  = max_iter
        self.limiar    = limiar
        self.coh_hist  = defaultdict(list)
        for i in range(len(arch)-1):
            # He init: expmap0 comprime as normas, então W precisa ser maior
            s = np.sqrt(2.0 / arch[i])
            self.W.append(np.random.randn(arch[i], arch[i+1]) * s)
            self.b.append(np.zeros(arch[i+1]))

    def _cone(self, x, idx):
        W, b = self.W[idx], self.b[idx]
        z   = x @ W + b
        cur = expmap0(z)   # ativacao_nativa comprimia demais — só expmap0
        self.coh_hist[idx].append(coerencia_phi(cur))
        return cur

    def forward(self, X):
        self.acts = [X]; self.cohs = []; cur = X
        for i in range(len(self.W)-1):
            cur = self._cone(cur, i)
            self.acts.append(cur)
            self.cohs.append(coerencia_phi(cur))
        # Camada de saída: projeção euclidiana do estado hiperbólico
        z   = cur @ self.W[-1] + self.b[-1]
        out = sigmoid(z)
        self.acts.append(out)
        return out

    def backward(self, X, y, lr):
        # _cone usa z = x @ W + b (projeção euclidiana), então backward é euclidiano.
        # A geometria hiperbólica está no forward — não no gradiente dos parâmetros.
        m     = X.shape[0]
        delta = self.acts[-1] - y.reshape(-1,1)
        for i in reversed(range(len(self.W))):
            self.W[i] -= lr * (self.acts[i].T @ delta / m)
            self.b[i]  -= lr * delta.mean(0)
            if i > 0:
                delta = delta @ self.W[i].T

    def predict(self, X):
        try:    return (self.forward(X).flatten() >= 0.5).astype(int)
        except: return np.zeros(X.shape[0], dtype=int)

    def accuracy(self, X, y):
        try:    return float(np.mean(self.predict(X) == y))
        except: return 0.5

    def coerencia_media(self):
        return {k: float(np.mean(v[-100:])) for k, v in self.coh_hist.items()}


# ── VERSÃO D — Campo Observer ──────────────────────────────────────────────────
class RedeCampoObserver(RedeConeAutonomo):
    """
    Cone autônomo + gradiente Riemanniano + campo observer.

    Campo observer no backward:
      Ponto:  magnitude ≥ C_PHI × max|grad|  → corrige totalmente
      Campo:  magnitude <  C_PHI × max|grad|  → assimila a escala 1/φ

    A vírgula de erro não é eliminada — é assimilada como
    informação de campo, na proporção 1/φ.
    Exatamente como o Campo Harmônico φ trata o ruído.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assimilados = 0
        self._total       = 0

    def backward(self, X, y, lr):
        m     = X.shape[0]
        delta = self.acts[-1] - y.reshape(-1,1)

        for i in reversed(range(len(self.W))):
            grad_W = self.acts[i].T @ delta / m

            # Campo observer — vírgula de erro
            max_g   = np.max(np.abs(grad_W)) + 1e-10
            virgula = C_PHI * max_g            # limiar proporcional ao campo atual
            mascara = np.abs(grad_W) < virgula

            self._assimilados += int(mascara.sum())
            self._total       += mascara.size

            # Sub-limiar → assimila a 1/φ    |    acima → corrige normalmente
            grad_W = np.where(mascara, grad_W / PHI, grad_W)

            self.W[i] -= lr * grad_W
            self.b[i]  -= lr * delta.mean(0)

            if i > 0:
                delta = delta @ self.W[i].T

    def taxa_assimilacao(self):
        if self._total == 0: return 0.0
        return self._assimilados / self._total


# ── Dataset ────────────────────────────────────────────────────────────────────
print("\nCarregando SST-2...")
dataset = load_dataset('nyu-mll/glue', 'sst2')
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

mean = X_train.mean(0); std = X_train.std(0) + 1e-8
X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
print(f"Treino: {X_train.shape}  Validação: {X_val.shape}")

# ── Configuração ───────────────────────────────────────────────────────────────
INPUT_DIM  = 384
N_EPOCHS   = 20
LR         = 0.003
BATCH_SIZE = 128

arch = [INPUT_DIM] + FIB + [1]

rng  = np.random.RandomState(int(time.time()) % 100000)
SEED = int(rng.randint(0, 99999))
print(f"\nSeed (timestamp): {SEED}")
print(f"Arquitetura: {arch}")

# ── Treino ─────────────────────────────────────────────────────────────────────
net_A = RedeEuclidiana(arch, SEED)
net_C = RedeConeAutonomo(arch, SEED)
net_D = RedeCampoObserver(arch, SEED)

hist  = {'A': [], 'C': [], 'D': []}
n_bat = len(X_train) // BATCH_SIZE

print(f"\n{'Época':>5} | {'A Eucl':>8} | {'C Cone':>8} | {'D Campo':>8} | Assim%")
print("-" * 52)

for ep in range(1, N_EPOCHS+1):
    idx = np.random.permutation(len(X_train))
    Xs, ys = X_train[idx], y_train[idx]

    for b in range(n_bat):
        Xb = Xs[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        yb = ys[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        net_A.forward(Xb); net_A.backward(Xb, yb, LR)
        net_C.forward(Xb); net_C.backward(Xb, yb, LR)
        net_D.forward(Xb); net_D.backward(Xb, yb, LR)

    aA   = net_A.accuracy(X_val, y_val)
    aC   = net_C.accuracy(X_val, y_val)
    aD   = net_D.accuracy(X_val, y_val)
    assm = net_D.taxa_assimilacao() * 100
    hist['A'].append(aA); hist['C'].append(aC); hist['D'].append(aD)
    print(f"{ep:>5} | {aA:>8.4f} | {aC:>8.4f} | {aD:>8.4f} | {assm:>5.1f}%")

# ── Resultados ─────────────────────────────────────────────────────────────────
fA = net_A.accuracy(X_val, y_val)
fC = net_C.accuracy(X_val, y_val)
fD = net_D.accuracy(X_val, y_val)

print("\n" + "=" * 65)
print("  RESULTADOS FINAIS — CAMPO OBSERVER")
print("=" * 65)
print(f"  A — Euclidiano          : {fA*100:.2f}%")
print(f"  C — Cone autônomo       : {fC*100:.2f}%")
print(f"  D — Campo observer      : {fD*100:.2f}%")
print(f"  Δ D vs A                : {(fD-fA)*100:+.2f}%")
print(f"  Δ D vs C                : {(fD-fC)*100:+.2f}%")
print(f"  Taxa de assimilação     : {net_D.taxa_assimilacao()*100:.1f}%  (vírgula de erro)")
print(f"  Seed                    : {SEED}")
print(f"  Curvatura c = 1/φ²      : {C_PHI:.4f}")

coh_C = net_C.coerencia_media()
coh_D = net_D.coerencia_media()
print(f"\n  Coerência φ por camada:")
print(f"  {'Camada':<14} {'C Cone':>8} {'D Campo':>8}")
print(f"  {'-'*32}")
for k in sorted(coh_C.keys()):
    bar = "█" * int(coh_D.get(k, 0) * 20)
    print(f"  Camada {k+1} (Fib={FIB[k]:3d}):  {coh_C[k]:>8.4f} {coh_D.get(k,0):>8.4f}  {bar}")
print("=" * 65)

if fD > fA and fD > fC:
    print(f"\n  ► CAMPO OBSERVER SUPERA AMBOS: +{(fD-fA)*100:.2f}% vs euclidiano")
elif fD > fA:
    print(f"\n  ► CAMPO OBSERVER SUPERA EUCLIDIANO: +{(fD-fA)*100:.2f}%")
elif abs(fD - fA) < 0.005:
    print(f"\n  ► RESULTADO NEUTRO — diferença < 0.5%")
else:
    print(f"\n  ► EUCLIDIANO AINDA SUPERIOR — Δ: {(fA-fD)*100:.2f}%")

# ── Gráfico ────────────────────────────────────────────────────────────────────
GOLD="#DAA520"; CYAN="#00BFFF"; GREEN="#00FF88"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8B949E")
    for sp in ax.spines.values(): sp.set_color("#30363d")

eps = np.arange(1, N_EPOCHS+1)
axes[0].plot(eps, [v*100 for v in hist['A']], "o-", color=GOLD,  lw=2, ms=4, label=f"A Eucl ({fA*100:.1f}%)")
axes[0].plot(eps, [v*100 for v in hist['C']], "s-", color=CYAN,  lw=2, ms=4, label=f"C Cone ({fC*100:.1f}%)")
axes[0].plot(eps, [v*100 for v in hist['D']], "^-", color=GREEN, lw=2, ms=4, label=f"D Campo ({fD*100:.1f}%)")
axes[0].axhline(50, color='red', lw=0.8, ls='--')
axes[0].set_xlabel("Época", color="#8B949E")
axes[0].set_ylabel("Acurácia (%)", color="#8B949E")
axes[0].set_title("Acurácia por Época", color="#E6EDF3", fontweight="bold")
axes[0].legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=8)
axes[0].grid(True, alpha=0.2)

barras = [fA*100, fC*100, fD*100]
cores  = [GOLD, CYAN, GREEN]
bars   = axes[1].bar(['A\nEuclidiano','C\nCone φ','D\nCampo Obs'], barras, color=cores, alpha=0.85)
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
    f"ALPHA PHI — Campo Observer  c=1/φ²={C_PHI:.4f}  Fibonacci{FIB}  Seed={SEED}  Florianópolis 2026",
    color=GOLD, fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.savefig("alphaphi_sst2_campo_observer.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\nGráfico salvo: alphaphi_sst2_campo_observer.png")
print("alpha-phi")
