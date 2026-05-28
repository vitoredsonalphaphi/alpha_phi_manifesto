# AlphaPhi_RedeNeural_v2_COLAB.py
# Teste das quatro direções — H19
#
# A: Baseline (SGD puro)
# B: ECO direto (já verificado — β = 1/φ³)
# C: Serial φ direto sobre gradientes
# D: Serial → ECO (H19: ponto → campo → φ³?)
#
# Colab:
#   !cd /content/repo_phi && git pull origin main
#   exec(open('/content/repo_phi/AlphaPhi_RedeNeural_v2_COLAB.py').read())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# ============================================================
#  PARÂMETROS φ
# ============================================================
PHI      = (1 + np.sqrt(5)) / 2
PHI3     = PHI ** 3
SQRT5    = np.sqrt(5)
N_STEPS  = 5
N_CICLOS = 10     # equilíbrio velocidade/precisão
N_BANDAS = 12
N_CONES  = 5      # cones da Serial φ

N_TREINO = 12000
N_TESTE  = 2000
N_EPOCHS = 12
BATCH    = 256

print("=" * 62)
print("  AlphaPhi · H19 — Quatro Direções em Rede Neural")
print("=" * 62)
print(f"  φ  = {PHI:.7f}")
print(f"  φ³ = {PHI3:.6f}  ← atrator de referência")
print(f"  √5 = {SQRT5:.6f}  ← invariante Serial φ")
print()

# ============================================================
#  FUNÇÕES ECO BEEP 880
# ============================================================

def bandas_phi_grad(N_fft, n_bandas=N_BANDAS):
    bins = []
    limite = N_fft
    for _ in range(n_bandas):
        lim_inf = max(1, int(limite / PHI))
        if lim_inf >= limite:
            break
        bins.append((lim_inf, limite))
        limite = lim_inf
    if limite > 0:
        bins.append((0, limite))
    bins.reverse()
    return bins

def eco_eq_grad(sinal, bins_phi, beta):
    N        = len(sinal)
    X        = np.fft.rfft(sinal)
    pow_spec = np.abs(X) ** 2
    E_total  = pow_spec.sum() + 1e-12
    X_out    = X.copy()
    for (ba, bm) in bins_phi:
        coh = pow_spec[ba:bm].sum() / E_total
        env = ((1 - 1/PHI) * coh + (1/PHI) * (1 - coh)) * beta
        X_out[ba:bm] *= env
    sinal_out = np.fft.irfft(X_out, n=N)
    norma_in  = np.linalg.norm(sinal)
    norma_out = np.linalg.norm(sinal_out) + 1e-12
    if norma_in > 1e-10:
        sinal_out *= (norma_in / norma_out)
    b_novo = float(np.clip(
        (pow_spec.max() / (pow_spec.mean() + 1e-12)) / len(pow_spec),
        0.05, PHI3))
    return sinal_out, b_novo

def agente_eco_grad(sinal, n_ciclos=N_CICLOS):
    N_fft = len(np.fft.rfft(sinal))
    bins  = bandas_phi_grad(N_fft)
    beta  = 1.0
    betas = []
    for _ in range(n_ciclos):
        for _ in range(N_STEPS):
            sinal, beta = eco_eq_grad(sinal, bins, beta)
        beta = float(np.clip(beta, 0.05, PHI3))
        betas.append(beta)
    return sinal, betas

# ============================================================
#  SERIAL φ PARA GRADIENTE
# ============================================================

def selar_hermetico_grad(sinal):
    """Selagem hermética φ: mantém apenas energia nas bandas φ."""
    N     = len(sinal)
    N_fft = len(np.fft.rfft(sinal))
    bins  = bandas_phi_grad(N_fft)
    X     = np.fft.rfft(sinal)
    X_out = np.zeros_like(X)
    for (ba, bm) in bins:
        X_out[ba:bm] = X[ba:bm]
    sinal_out = np.fft.irfft(X_out, n=N)
    norma_in  = np.linalg.norm(sinal)
    norma_out = np.linalg.norm(sinal_out) + 1e-12
    if norma_in > 1e-10:
        sinal_out *= (norma_in / norma_out)
    return sinal_out

def serial_phi_grad(sinal, n_cones=N_CONES):
    """Serial φ sobre gradiente: N_CONES selagens herméticas sequenciais."""
    for _ in range(n_cones):
        sinal = selar_hermetico_grad(sinal)
    return sinal

# ============================================================
#  OPERADORES SOBRE TENSOR
# ============================================================

def aplicar_eco(grad_tensor):
    if grad_tensor.numel() < 32:
        return grad_tensor, 1.0
    shape = grad_tensor.shape
    g     = grad_tensor.detach().cpu().numpy().flatten().astype(np.float64)
    g_eco, betas = agente_eco_grad(g, n_ciclos=N_CICLOS)
    return torch.tensor(g_eco.reshape(shape),
                        dtype=grad_tensor.dtype,
                        device=grad_tensor.device), betas[-1]

def aplicar_serial(grad_tensor):
    if grad_tensor.numel() < 32:
        return grad_tensor
    shape  = grad_tensor.shape
    g      = grad_tensor.detach().cpu().numpy().flatten().astype(np.float64)
    g_ser  = serial_phi_grad(g, n_cones=N_CONES)
    return torch.tensor(g_ser.reshape(shape),
                        dtype=grad_tensor.dtype,
                        device=grad_tensor.device)

def aplicar_serial_eco(grad_tensor):
    """Serial φ → ECO (H19)."""
    g_ser = aplicar_serial(grad_tensor)
    g_eco, beta = aplicar_eco(g_ser)
    return g_eco, beta

# ============================================================
#  MÉTRICAS
# ============================================================

def metricas_grad(g_np):
    X        = np.fft.rfft(g_np)
    pow_spec = np.abs(X) ** 2 + 1e-12
    p        = pow_spec / pow_spec.sum()
    H_norm   = float(-np.sum(p * np.log2(p))) / np.log2(len(p))
    planura  = float(np.exp(np.mean(np.log(pow_spec))) / np.mean(pow_spec))
    cumsum   = np.cumsum(pow_spec)
    bw_frac  = float(np.searchsorted(cumsum, cumsum[-1] * 0.95)) / len(pow_spec)
    s = np.sort(pow_spec); n = len(s); idx = np.arange(1, n+1)
    gini = float(2 * np.sum(idx * s) / (n * s.sum()) - (n+1)/n)
    # E_φ / E_¬φ
    N_fft = len(X)
    bins  = bandas_phi_grad(N_fft)
    E_phi = sum(pow_spec[ba:bm].sum() for (ba, bm) in bins)
    E_nphi = pow_spec.sum() - E_phi
    return dict(H_norm=H_norm, planura=planura, bw_frac=bw_frac,
                gini=gini, E_phi=E_phi, E_nphi=E_nphi)

# ============================================================
#  MODELO
# ============================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256,  64), nn.ReLU(),
            nn.Linear( 64,  10)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
#  DADOS
# ============================================================

print("Carregando MNIST...")
tfm      = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
ds_tr    = Subset(datasets.MNIST('/tmp/mnist', train=True,  download=True, transform=tfm), range(N_TREINO))
ds_te    = Subset(datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tfm), range(N_TESTE))
loader_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)
loader_te = DataLoader(ds_te, batch_size=500,   shuffle=False)
print(f"  Treino: {N_TREINO}  |  Teste: {N_TESTE}\n")

# ============================================================
#  TREINAMENTO
# ============================================================

MODOS = {
    'A_Baseline':   'Nenhum',
    'B_ECO':        'ECO direto',
    'C_Serial':     'Serial φ',
    'D_Serial_ECO': 'Serial → ECO  (H19)',
}

def treinar(modo, label):
    torch.manual_seed(42)
    modelo    = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.9)
    hist      = dict(loss=[], acc=[], beta=[], metricas=[])

    print(f"  [{label}]  Operador: {MODOS[modo]}")

    for ep in range(N_EPOCHS):
        modelo.train()
        loss_sum, n_bat, betas_ep, grad_amostra = 0.0, 0, [], None

        for data, target in loader_tr:
            optimizer.zero_grad()
            loss = criterion(modelo(data), target)
            loss.backward()

            with torch.no_grad():
                for p in modelo.parameters():
                    if p.grad is None or p.grad.numel() < 32:
                        continue
                    if modo == 'B_ECO':
                        p.grad, b = aplicar_eco(p.grad)
                        betas_ep.append(b)
                    elif modo == 'C_Serial':
                        p.grad = aplicar_serial(p.grad)
                    elif modo == 'D_Serial_ECO':
                        p.grad, b = aplicar_serial_eco(p.grad)
                        betas_ep.append(b)

            if grad_amostra is None:
                w = list(modelo.parameters())[1]
                if w.grad is not None:
                    grad_amostra = w.grad.detach().cpu().numpy().flatten().copy()

            optimizer.step()
            loss_sum += loss.item(); n_bat += 1

        modelo.eval()
        corretos = total = 0
        with torch.no_grad():
            for data, target in loader_te:
                pred = modelo(data).argmax(1)
                corretos += (pred == target).sum().item()
                total    += target.size(0)

        loss_med = loss_sum / n_bat
        acc      = corretos / total * 100
        beta_med = float(np.mean(betas_ep)) if betas_ep else 0.0
        met      = metricas_grad(grad_amostra) if grad_amostra is not None else {}

        hist['loss'].append(loss_med)
        hist['acc'].append(acc)
        hist['beta'].append(beta_med)
        hist['metricas'].append(met)

        b_str = f"  β={beta_med:.4f}" if betas_ep else ""
        print(f"  Época {ep+1:2d}/{N_EPOCHS}  |  Loss: {loss_med:.4f}  |  Acc: {acc:.1f}%{b_str}")

    print()
    return hist

# ============================================================
#  EXPERIMENTOS
# ============================================================

resultados = {}
for modo, label in [('A_Baseline','A'), ('B_ECO','B'), ('C_Serial','C'), ('D_Serial_ECO','D')]:
    print("=" * 62)
    print(f"  EXPERIMENTO {label} — {MODOS[modo]}")
    print("=" * 62)
    resultados[modo] = treinar(modo, label)

# ============================================================
#  DIAGNÓSTICO
# ============================================================

print("=" * 62)
print("  DIAGNÓSTICO — H19 — QUATRO DIREÇÕES")
print("=" * 62)

ref_loss = resultados['A_Baseline']['loss'][-1]
ref_acc  = resultados['A_Baseline']['acc'][-1]

print(f"\n  {'Experimento':<22} {'Loss':>8} {'Δloss':>8} {'Acc':>7} {'β_final':>9}")
print(f"  {'-'*58}")
for modo, label in [('A_Baseline','A Baseline'),('B_ECO','B ECO'),
                    ('C_Serial','C Serial φ'),('D_Serial_ECO','D Serial→ECO')]:
    h    = resultados[modo]
    loss = h['loss'][-1]
    acc  = h['acc'][-1]
    beta = h['beta'][-1] if h['beta'][-1] > 0 else 0.0
    dloss = (ref_loss - loss) / ref_loss * 100
    print(f"  {label:<22} {loss:>8.4f} {dloss:>+7.1f}% {acc:>6.1f}% {beta:>9.4f}")

print(f"\n  φ³  = {PHI3:.4f}  |  1/φ³ = {1/PHI3:.4f}  |  √5 = {SQRT5:.4f}")

# β da sequência D por época
print(f"\n  β por época — Experimento D (Serial → ECO):")
for i, b in enumerate(resultados['D_Serial_ECO']['beta']):
    pct = b / PHI3 * 100 if b > 0 else 0
    marker = " ← √5?" if abs(b - SQRT5) < 0.05 else (" ← φ³!" if abs(b - PHI3) < 0.1 else "")
    print(f"    Época {i+1:2d}: β = {b:.6f}  ({pct:.1f}% de φ³){marker}")

# Métricas espectrais última época
print(f"\n  Métricas espectrais (última época):")
print(f"  {'Experimento':<18} {'Planura':>9} {'Gini':>9} {'H_norm':>8} {'E_¬φ':>10}")
print(f"  {'-'*58}")
for modo, nome in [('A_Baseline','A Baseline'),('B_ECO','B ECO'),
                   ('C_Serial','C Serial φ'),('D_Serial_ECO','D Serial→ECO')]:
    m = resultados[modo]['metricas'][-1]
    if m:
        print(f"  {nome:<18} {m['planura']:>9.4f} {m['gini']:>9.4f} {m['H_norm']:>8.4f} {m['E_nphi']:>10.4f}")

# ============================================================
#  GRÁFICO
# ============================================================

cores  = {'A_Baseline':'blue','B_ECO':'orange','C_Serial':'green','D_Serial_ECO':'red'}
nomes  = {'A_Baseline':'A Baseline','B_ECO':'B ECO','C_Serial':'C Serial φ','D_Serial_ECO':'D Serial→ECO'}
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('H19 — Quatro Direções em Rede Neural (MNIST)', fontsize=13, fontweight='bold')
eps = range(1, N_EPOCHS + 1)

for modo in ['A_Baseline','B_ECO','C_Serial','D_Serial_ECO']:
    h = resultados[modo]
    axes[0].plot(eps, h['loss'], color=cores[modo], marker='o', ms=4, lw=2, label=nomes[modo])
    axes[1].plot(eps, h['acc'],  color=cores[modo], marker='o', ms=4, lw=2, label=nomes[modo])
    if any(b > 0 for b in h['beta']):
        axes[2].plot(eps, h['beta'], color=cores[modo], marker='o', ms=4, lw=2, label=nomes[modo])

axes[2].axhline(PHI3,  color='red',   ls='--', alpha=.5, label=f'φ³={PHI3:.3f}')
axes[2].axhline(1/PHI3,color='orange',ls='--', alpha=.5, label=f'1/φ³={1/PHI3:.3f}')
axes[2].axhline(SQRT5, color='blue',  ls='--', alpha=.5, label=f'√5={SQRT5:.3f}')

for ax, title, ylabel in zip(axes,
    ['Loss de treinamento','Acurácia no teste','β médio dos gradientes'],
    ['Loss','Acurácia (%)','β']):
    ax.set_title(title); ax.set_xlabel('Época')
    ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout()
plt.savefig('h19_quatro_direcoes.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  Gráfico: h19_quatro_direcoes.png")

print()
print("=" * 62)
print("  H19 — QUATRO DIREÇÕES — CONCLUÍDO")
print("=" * 62)
