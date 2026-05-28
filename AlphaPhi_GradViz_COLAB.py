# AlphaPhi_GradViz_COLAB.py
# Visualização Espectral do Gradiente — Método da Retrocausalidade
# O que o ambiente do gradiente já nos mostra antes de qualquer ferramenta?
#
# Colab:
#   !cd /content/repo_phi && git pull origin main
#   exec(open('/content/repo_phi/AlphaPhi_GradViz_COLAB.py').read())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

PHI      = (1 + np.sqrt(5)) / 2
PHI3     = PHI ** 3
SQRT5    = np.sqrt(5)
N_BANDAS = 12
N_TREINO = 2000
BATCH    = 256
N_EPOCHS = 8

print("=" * 62)
print("  AlphaPhi · Visualização Espectral do Gradiente")
print("  Método da Retrocausalidade — O que já está lá?")
print("=" * 62)
print(f"  φ  = {PHI:.7f}")
print(f"  φ³ = {PHI3:.6f}   ← atrator domínio áudio")
print(f"  1/φ³= {1/PHI3:.6f}  ← atrator observado nos gradientes")
print(f"  √5 = {SQRT5:.6f}   ← invariante Serial φ")
print()

# ============================================================
#  FUNÇÕES φ
# ============================================================

def bandas_phi(N_fft, n_bandas=N_BANDAS):
    bins = []; limite = N_fft
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

def analisar(g):
    """Análise espectral φ completa — sem aplicar nenhuma transformação."""
    X       = np.fft.rfft(g)
    ps      = np.abs(X) ** 2 + 1e-12
    N_fft   = len(ps)
    bins    = bandas_phi(N_fft)
    E_total = ps.sum()
    E_phi   = sum(ps[ba:bm].sum() for (ba, bm) in bins)
    beta    = float(np.clip((ps.max() / ps.mean()) / N_fft, 0.05, PHI3))
    p       = ps / E_total
    H_norm  = float(-np.sum(p * np.log2(p))) / np.log2(N_fft)
    s       = np.sort(ps); n = len(s); idx = np.arange(1, n + 1)
    gini    = float(2 * np.sum(idx * s) / (n * s.sum()) - (n + 1) / n)
    planura = float(np.exp(np.mean(np.log(ps))) / np.mean(ps))
    return dict(beta=beta, e_phi_frac=E_phi / E_total, H_norm=H_norm,
                gini=gini, planura=planura, ps=ps, bins=bins)

# ============================================================
#  MODELO E DADOS
# ============================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256,  64), nn.ReLU(),
            nn.Linear( 64,  10))
    def forward(self, x):
        return self.net(x)

print("Carregando MNIST...")
tfm = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
ds  = Subset(datasets.MNIST('/tmp/mnist', train=True, download=True,
                             transform=tfm), range(N_TREINO))
loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
print(f"  {N_TREINO} amostras  |  {N_EPOCHS} épocas\n")

# ============================================================
#  COLETA — gradientes por camada por época
# ============================================================

torch.manual_seed(42)
modelo    = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.9)

CAMADAS  = ['W1(784→256)', 'W2(256→64)', 'W3(64→10)']
IDX_PAR  = [0, 2, 4]          # pesos grandes (sem biases)
CORES    = {'W1(784→256)': 'blue', 'W2(256→64)': 'green', 'W3(64→10)': 'red'}

historico = {c: dict(beta=[], e_phi=[], H_norm=[], gini=[], planura=[])
             for c in CAMADAS}
snapshots = {}   # ps e bins de W1 por época

print("Treinando e capturando gradientes — sem nenhuma ferramenta φ...")
params = list(modelo.parameters())

for ep in range(N_EPOCHS):
    modelo.train()
    acum = {c: [] for c in CAMADAS}

    for data, target in loader:
        optimizer.zero_grad()
        loss = criterion(modelo(data), target)
        loss.backward()
        for c, idx in zip(CAMADAS, IDX_PAR):
            p = params[idx]
            if p.grad is not None:
                acum[c].append(p.grad.detach().cpu().numpy().flatten().copy())
        optimizer.step()

    for c in CAMADAS:
        if acum[c]:
            g = np.concatenate(acum[c])
            m = analisar(g)
            historico[c]['beta'].append(m['beta'])
            historico[c]['e_phi'].append(m['e_phi_frac'])
            historico[c]['H_norm'].append(m['H_norm'])
            historico[c]['gini'].append(m['gini'])
            historico[c]['planura'].append(m['planura'])
            if c == 'W1(784→256)':
                snapshots[ep] = (m['ps'].copy(), m['bins'])

    b = historico['W1(784→256)']['beta'][-1]
    e = historico['W1(784→256)']['e_phi'][-1]
    h = historico['W1(784→256)']['H_norm'][-1]
    print(f"  Época {ep+1}/{N_EPOCHS}  W1: β={b:.4f}  E_φ={e*100:.1f}%  H={h:.4f}")

print()

# ============================================================
#  RELATÓRIO — RETROCAUSALIDADE
# ============================================================

print("=" * 62)
print("  O QUE O GRADIENTE JÁ MOSTRA — SEM NENHUMA FERRAMENTA φ")
print("=" * 62)
print(f"\n  {'Camada':<16} {'β_ep1':>7} {'β_epN':>7} {'E_φ%_1':>8} {'E_φ%_N':>8} {'H_ep1':>7} {'Gini1':>7}")
print(f"  {'-'*62}")
for c in CAMADAS:
    h = historico[c]
    if h['beta']:
        print(f"  {c:<16} {h['beta'][0]:>7.4f} {h['beta'][-1]:>7.4f} "
              f"{h['e_phi'][0]*100:>7.1f}% {h['e_phi'][-1]*100:>7.1f}% "
              f"{h['H_norm'][0]:>7.4f} {h['gini'][0]:>7.4f}")

print(f"\n  Referências φ:")
print(f"    φ³      = {PHI3:.4f}  ← atrator domínio áudio (ECO sobre som)")
print(f"    1/φ³    = {1/PHI3:.4f}  ← atrator observado nos gradientes (ECO na rede neural)")
print(f"    √5      = {SQRT5:.4f}  ← invariante entrada da Serial φ")
print(f"    E_φ unif= {100/N_BANDAS:.1f}%   ← distribuição uniforme (12 bandas)")

# Verificar posição de β em relação a 1/φ³
print(f"\n  β observado vs 1/φ³:")
for c in CAMADAS:
    h = historico[c]
    if h['beta']:
        b1 = h['beta'][0]; bN = h['beta'][-1]
        frac1 = b1 / (1/PHI3) * 100; fracN = bN / (1/PHI3) * 100
        print(f"    {c:<16}  β_ep1={b1:.4f} ({frac1:.0f}% de 1/φ³)  β_epN={bN:.4f} ({fracN:.0f}% de 1/φ³)")

# ============================================================
#  INTERPRETAÇÃO φ — RETROCAUSALIDADE
# ============================================================

print(f"\n  INTERPRETAÇÃO — Retrocausalidade φ:")
b_w1_medio = float(np.mean(historico['W1(784→256)']['beta']))
e_w1_medio = float(np.mean(historico['W1(784→256)']['e_phi']))
print(f"    β médio W1 = {b_w1_medio:.4f}  |  E_φ% médio = {e_w1_medio*100:.1f}%")
dist_phi3  = abs(b_w1_medio - PHI3)
dist_1phi3 = abs(b_w1_medio - 1/PHI3)
if dist_1phi3 < dist_phi3:
    print(f"    β natural do gradiente está PRÓXIMO de 1/φ³ (Δ={dist_1phi3:.4f})")
    print(f"    → O gradiente já habita o atrator inverso φ")
else:
    print(f"    β natural do gradiente está PRÓXIMO de φ³ (Δ={dist_phi3:.4f})")
    print(f"    → O gradiente já habita o atrator direto φ³")

# ============================================================
#  GRÁFICOS
# ============================================================

fig = plt.figure(figsize=(18, 10))
fig.suptitle('Retrocausalidade — Estrutura φ Natural dos Gradientes (MLP/MNIST)',
             fontsize=13, fontweight='bold')
eps_x = range(1, N_EPOCHS + 1)

# --- β por época ---
ax1 = fig.add_subplot(2, 4, 1)
for c in CAMADAS:
    ax1.plot(eps_x, historico[c]['beta'], color=CORES[c], marker='o', ms=4, lw=2, label=c)
ax1.axhline(PHI3,   color='darkred',  ls='--', alpha=.5, lw=1.5, label=f'φ³={PHI3:.3f}')
ax1.axhline(1/PHI3, color='orange',   ls='--', alpha=.7, lw=2,   label=f'1/φ³={1/PHI3:.3f}')
ax1.axhline(SQRT5,  color='purple',   ls='--', alpha=.4, lw=1,   label=f'√5={SQRT5:.3f}')
ax1.set_title('β natural por época'); ax1.set_xlabel('Época'); ax1.set_ylabel('β')
ax1.legend(fontsize=6); ax1.grid(alpha=.3)

# --- E_φ% por época ---
ax2 = fig.add_subplot(2, 4, 2)
for c in CAMADAS:
    ax2.plot(eps_x, [v*100 for v in historico[c]['e_phi']],
             color=CORES[c], marker='o', ms=4, lw=2, label=c)
ax2.axhline(100/N_BANDAS, color='gray', ls='--', alpha=.5,
            label=f'Uniforme ({100/N_BANDAS:.0f}%)')
ax2.set_title('E_φ% natural por época')
ax2.set_xlabel('Época'); ax2.set_ylabel('% energia em bandas φ')
ax2.legend(fontsize=6); ax2.grid(alpha=.3)

# --- H_norm por época ---
ax3 = fig.add_subplot(2, 4, 3)
for c in CAMADAS:
    ax3.plot(eps_x, historico[c]['H_norm'], color=CORES[c], marker='o', ms=4, lw=2, label=c)
ax3.set_title('Entropia H por época')
ax3.set_xlabel('Época'); ax3.set_ylabel('Shannon H (norm)')
ax3.legend(fontsize=6); ax3.grid(alpha=.3)

# --- Gini por época ---
ax4 = fig.add_subplot(2, 4, 4)
for c in CAMADAS:
    ax4.plot(eps_x, historico[c]['gini'], color=CORES[c], marker='o', ms=4, lw=2, label=c)
ax4.set_title('Gini espectral por época')
ax4.set_xlabel('Época'); ax4.set_ylabel('Gini')
ax4.legend(fontsize=6); ax4.grid(alpha=.3)

# --- Espectros W1 — épocas 1, 3, 5, N ---
eps_snap = [0, 2, 4, N_EPOCHS - 1]
tit_snap = ['Época 1', 'Época 3', 'Época 5', f'Época {N_EPOCHS}']
for i, (ep_idx, titulo) in enumerate(zip(eps_snap, tit_snap)):
    if ep_idx not in snapshots:
        continue
    ax = fig.add_subplot(2, 4, 5 + i)
    ps, bins = snapshots[ep_idx]
    freqs    = np.arange(len(ps))
    ps_db    = 10 * np.log10(ps)
    ax.plot(freqs, ps_db, 'k-', lw=0.5, alpha=0.8)
    cores_b  = plt.cm.plasma(np.linspace(0.1, 0.9, len(bins)))
    for (ba, bm), cb in zip(bins, cores_b):
        ax.axvspan(ba, bm, alpha=0.13, color=cb)
    b_val = historico['W1(784→256)']['beta'][ep_idx]
    e_val = historico['W1(784→256)']['e_phi'][ep_idx]
    ax.set_title(f'W1 — {titulo}\nβ={b_val:.4f}  E_φ={e_val*100:.1f}%', fontsize=8)
    ax.set_xlabel('Bin FFT'); ax.set_ylabel('Potência (dB)')
    ax.grid(alpha=.2)

plt.tight_layout()
plt.savefig('gradiente_retrocausalidade.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  Gráfico: gradiente_retrocausalidade.png")

print()
print("=" * 62)
print("  RETROCAUSALIDADE — CONCLUÍDO")
print("=" * 62)
