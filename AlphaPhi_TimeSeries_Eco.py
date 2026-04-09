# AlphaPhi_TimeSeries_Eco.py
# Primeiro experimento não-texto do projeto Alpha-Phi
#
# Substrato: séries temporais sintéticas (φ-estruturado vs ruído)
# Pré-função: eco_ressonante — substrate-agnostic
# Também testa: L = CE + α·H(φ) — função de perda nunca testada
#
# Configurações:
#   G       — baseline: MLP scratch, sem eco, loss CE padrão
#   G_eco   — eco_ressonante como pré-função antes da rede
#   G_v2    — phi_spectral_modulator_v2 como modulação interna
#   G_Lphi  — L = CE + α·H(φ), eco como pré-função
#
# Dado:
#   Classe 1: sinal com frequências em proporção φ (estrutura real)
#   Classe 0: ruído gaussiano puro
#
# Rede: scratch numpy (sem pre-treinamento)
# 20 seeds, paired t-tests

import numpy as np
from scipy import stats
import json, time

# ── Constantes αφ ─────────────────────────────────────────────────────────
PHI   = (1 + np.sqrt(5)) / 2
ALPHA = 1 / 137.035999084
C_PHI = 1.0 / PHI**2

# ── Config ─────────────────────────────────────────────────────────────────
TIMESTAMP  = int(time.time())
N_SEEDS    = 20
SEEDS      = [(TIMESTAMP + i * 137) % (2**31) for i in range(N_SEEDS)]
N_TRAIN    = 400   # por classe → 800 total
N_TEST     = 100   # por classe → 200 total
DIM        = 128   # dimensão do sinal
N_EPOCHS   = 60
LR         = 0.01
BATCH_SIZE = 64
HIDDEN     = 89    # Fibonacci próximo de 128/√φ

print(f"α = {ALPHA:.10f}")
print(f"φ = {PHI:.10f}")
print(f"c = {C_PHI:.10f}")
print(f"Seeds[0]: {SEEDS[0]}")

# ── Geração de dados ───────────────────────────────────────────────────────

def gerar_sinal_phi(n, dim=DIM, phi=PHI, noise=0.15, seed=None):
    """
    Classe 1: sinal com estrutura φ.
    Soma de senóides com frequências em proporção φ: f, φf, φ²f, φ³f.
    Fase aleatória por amostra — a estrutura de frequência é real,
    a posição de fase é livre (como φ na natureza: proporção, não posição).
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, dim)
    freqs = [phi**i for i in range(4)]
    sinais = []
    for _ in range(n):
        s = sum(np.sin(f * t + rng.uniform(0, 2*np.pi)) for f in freqs)
        s += noise * rng.randn(dim)
        s /= (np.abs(s).max() + 1e-8)
        sinais.append(s)
    return np.array(sinais)

def gerar_ruido(n, dim=DIM, seed=None):
    """Classe 0: ruído gaussiano puro — sem estrutura φ."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n, dim)
    x /= (np.abs(x).max(axis=-1, keepdims=True) + 1e-8)
    return x

def montar_dataset(n_por_classe, dim=DIM, seed=0):
    X1 = gerar_sinal_phi(n_por_classe, dim, seed=seed)
    X0 = gerar_ruido(n_por_classe, dim, seed=seed+1)
    X  = np.vstack([X1, X0])
    y  = np.array([1]*n_por_classe + [0]*n_por_classe)
    idx = np.random.RandomState(seed).permutation(len(y))
    return X[idx], y[idx]

print("Gerando dados...")
X_train, y_train = montar_dataset(N_TRAIN, seed=0)
X_test,  y_test  = montar_dataset(N_TEST,  seed=999)
print(f"Treino: {X_train.shape}  Teste: {X_test.shape}")

# ── Funções αφ (numpy) ─────────────────────────────────────────────────────

def golden_activation(x):
    return PHI * np.tanh(x / PHI)

def golden_activation_deriv(x):
    return 1.0 - np.tanh(x / PHI)**2

def relu(x):        return np.maximum(0, x)
def relu_deriv(x):  return (x > 0).astype(float)
def sigmoid(x):     return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def eco_ressonante(x, phi=PHI, n_eco=3):
    """
    Pré-função substrate-agnostic.
    Ciclos de eco: sinal φ-coerente converge, ruído diverge.
    """
    sinal = x.copy()
    for _ in range(n_eco):
        freq      = np.fft.fft(sinal, axis=-1)
        amplitude = np.abs(freq)
        fase      = np.angle(freq)
        nova_fase = fase * phi
        reflexao  = np.real(np.fft.ifft(amplitude * np.exp(1j * nova_fase), axis=-1))
        eco       = reflexao - x
        sinal     = sinal + eco / phi
    return sinal

def phi_spectral_mod_v2(x, phi=PHI, n_eco=3):
    """Modulador espectral v2 — retorna (batch,1) por amostra."""
    sinal = eco_ressonante(x, phi=phi, n_eco=n_eco)
    freq  = np.fft.fft(sinal, axis=-1)
    e     = np.clip(np.abs(freq) / (np.abs(freq).sum(-1, keepdims=True) + 1e-8), 1e-10, 1.0)
    h     = -np.sum(e * np.log(e), axis=-1, keepdims=True)
    coe   = 1.0 - h / np.log(x.shape[-1])
    return phi * np.tanh(coe * phi)

def entropia_phi(h_estados, phi=PHI):
    """H(φ): entropia da distribuição de ativações golden."""
    a = golden_activation(h_estados)
    p = np.abs(a) / (np.abs(a).sum(-1, keepdims=True) + 1e-8)
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=-1).mean()

# ── Rede MLP numpy ─────────────────────────────────────────────────────────

class MLP:
    """
    2 camadas: DIM → HIDDEN → 2
    Suporta: eco como pré-função, v2 como modulação, L=CE+α·H(φ)
    """
    def __init__(self, dim=DIM, hidden=HIDDEN, use_eco=False,
                 use_v2=False, use_Lphi=False, seed=0):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.randn(dim, hidden) * scale1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, 2) * scale2
        self.b2 = np.zeros(2)
        self.use_eco   = use_eco
        self.use_v2    = use_v2
        self.use_Lphi  = use_Lphi

    def forward(self, x):
        # Pré-função: eco_ressonante
        if self.use_eco:
            x = eco_ressonante(x)

        # Camada 1
        z1 = x @ self.W1 + self.b1
        h1 = golden_activation(z1)

        # Modulação v2 interna
        if self.use_v2:
            mod = phi_spectral_mod_v2(x)   # (batch, 1)
            h1  = h1 * mod

        # Camada 2
        z2    = h1 @ self.W2 + self.b2
        probs = softmax(z2)
        return probs, h1, z1, x

    def loss(self, probs, targets, h1_estados=None):
        """CE padrão ou L = CE + α·H(φ)"""
        n   = len(targets)
        ce  = -np.log(probs[np.arange(n), targets] + 1e-10).mean()
        if self.use_Lphi and h1_estados is not None:
            h_phi   = entropia_phi(h1_estados)
            # α como threshold: penalidade só acima da granularidade mínima
            penalidade = max(0.0, h_phi - ALPHA) * ALPHA
            return ce + penalidade
        return ce

    def backward(self, x_in, z1, h1, probs, targets):
        n    = len(targets)
        dz2  = probs.copy(); dz2[np.arange(n), targets] -= 1; dz2 /= n
        dW2  = h1.T @ dz2
        db2  = dz2.sum(0)
        dh1  = dz2 @ self.W2.T
        dz1  = dh1 * golden_activation_deriv(z1)
        dW1  = x_in.T @ dz1
        db1  = dz1.sum(0)
        # clip
        for g in [dW1, db1, dW2, db2]:
            np.clip(g, -1.0, 1.0, out=g)
        self.W1 -= LR * dW1
        self.b1 -= LR * db1
        self.W2 -= LR * dW2
        self.b2 -= LR * db2

    def train_eval(self, X_tr, y_tr, X_te, y_te):
        n = len(X_tr)
        for _ in range(N_EPOCHS):
            idx = np.random.permutation(n)
            for i in range(0, n, BATCH_SIZE):
                b  = idx[i:i+BATCH_SIZE]
                xb, yb = X_tr[b], y_tr[b]
                probs, h1, z1, x_in = self.forward(xb)
                l = self.loss(probs, yb, h1)
                self.backward(x_in, z1, h1, probs, yb)
        probs, _, _, _ = self.forward(X_te)
        return (probs.argmax(-1) == y_te).mean()

# ── Experimento ────────────────────────────────────────────────────────────

CONFIGS = {
    'G':      {'use_eco': False, 'use_v2': False, 'use_Lphi': False},
    'G_eco':  {'use_eco': True,  'use_v2': False, 'use_Lphi': False},
    'G_v2':   {'use_eco': False, 'use_v2': True,  'use_Lphi': False},
    'G_Lphi': {'use_eco': True,  'use_v2': False, 'use_Lphi': True},
}

results  = {k: [] for k in CONFIGS}
total    = len(CONFIGS) * N_SEEDS
run_n    = 0

print(f"\n{len(CONFIGS)} configs × {N_SEEDS} seeds = {total} runs\n")

for cfg, kwargs in CONFIGS.items():
    print(f"Config {cfg}:")
    for seed in SEEDS:
        np.random.seed(seed % (2**31))
        model = MLP(seed=seed % (2**31), **kwargs)
        acc   = model.train_eval(X_train, y_train, X_test, y_test)
        results[cfg].append(float(acc))
        run_n += 1
        print(f"  {seed%10000:04d}  {acc:.4f}  ({run_n}/{total})")
    arr = np.array(results[cfg])
    print(f"  → {cfg}: {arr.mean():.4f} ± {arr.std():.4f}\n")

# ── Testes estatísticos ────────────────────────────────────────────────────

print("── Testes estatísticos ──")
sts = {}
pares = [('G_eco','G'), ('G_v2','G'), ('G_Lphi','G'),
         ('G_eco','G_v2'), ('G_Lphi','G_eco')]
for a, b in pares:
    _, p = stats.ttest_rel(results[a], results[b])
    d    = np.mean(results[a]) - np.mean(results[b])
    sig  = "✓" if p < 0.05 else "ns"
    print(f"{a} vs {b}:  Δ={d:+.4f}  p={p:.4f}  {sig}")
    sts[f"{a}_vs_{b}"] = {"delta": round(float(d),6),
                           "p_value": round(float(p),6)}

# ── Export ─────────────────────────────────────────────────────────────────

export = {
    "experimento":  "TimeSeries_Eco",
    "substrato":    "séries_temporais_sintéticas_phi",
    "dado_classe1": "sinal φ: frequências em proporção φ¹·²·³·⁴",
    "dado_classe0": "ruído gaussiano puro",
    "n_seeds":      N_SEEDS,
    "n_epochs":     N_EPOCHS,
    "n_train_pc":   N_TRAIN,
    "n_test_pc":    N_TEST,
    "dim":          DIM,
    "hidden":       HIDDEN,
    "timestamp":    TIMESTAMP,
    "seeds":        SEEDS,
    "resultados": {
        k: {"mean":   float(np.mean(results[k])),
            "std":    float(np.std(results[k])),
            "values": results[k]}
        for k in results
    },
    "testes": sts,
    "nota": (
        f"Primeiro experimento não-texto do Alpha-Phi. "
        f"eco_ressonante como pré-função substrate-agnostic. "
        f"L = CE + α·H(φ) com α={ALPHA:.8f} como threshold (floor)."
    )
}

with open('timeseries_eco_results.json', 'w', encoding='utf-8') as f:
    json.dump(export, f, indent=2, ensure_ascii=False)

print("\nSalvo: timeseries_eco_results.json")

# ── CÉLULA 3 ───────────────────────────────────────────────────────────────
# import json
# with open('timeseries_eco_results.json') as f:
#     d = json.load(f)
# print(json.dumps(d, indent=2, ensure_ascii=False))
