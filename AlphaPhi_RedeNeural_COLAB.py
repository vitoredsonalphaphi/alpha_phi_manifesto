# AlphaPhi_RedeNeural_COLAB.py
# ECO BEEP 880 aplicado a gradientes de rede neural
# Etapa 1 — Experimento base: ECO φ vs Baseline
#
# Colab:
#   !git clone -b main https://github.com/vitoredsonalphaphi/alpha_phi_manifesto repo_phi
#   exec(open('/content/repo_phi/AlphaPhi_RedeNeural_COLAB.py').read())

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
ALPHA_CO = 1/3        # coeficiente de coerência
N_STEPS  = 5          # dobras por ciclo
N_CICLOS = 20         # ciclos do agente completo (modo preciso)
N_BANDAS = 12         # bandas φ sobre o gradiente

N_TREINO = 12000      # amostras de treino (subconjunto MNIST para velocidade)
N_TESTE  = 2000
N_EPOCHS = 12
BATCH    = 256

print("=" * 62)
print("  AlphaPhi · ECO BEEP 880 em Rede Neural")
print("=" * 62)
print(f"  φ  = {PHI:.7f}")
print(f"  φ³ = {PHI3:.6f}  ← atrator de referência")
print(f"  √5 = {SQRT5:.6f}  ← invariante Serial φ")
print()

# ============================================================
#  FUNÇÕES ECO BEEP 880 — DOMÍNIO GRADIENTE
# ============================================================

def bandas_phi_grad(N_fft, n_bandas=N_BANDAS):
    """Bandas φ-geométricas no espaço de coeficientes FFT do gradiente.
    Sem Hz — divisão geométrica direta do espaço espectral."""
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
    """Envelope de coerência φ por banda — preserva norma L2 do gradiente."""
    N = len(sinal)
    X = np.fft.rfft(sinal)
    pow_spec = np.abs(X) ** 2
    E_total  = pow_spec.sum() + 1e-12
    X_out    = X.copy()

    for (ba, bm) in bins_phi:
        E_banda = pow_spec[ba:bm].sum()
        coh     = E_banda / E_total
        # envelope φ: coerente amplifica, incoerente atenua
        env = ((1 - 1/PHI) * coh + (1/PHI) * (1 - coh)) * beta
        X_out[ba:bm] *= env

    sinal_out = np.fft.irfft(X_out, n=N)

    # Preservar norma L2 — não alterar magnitude, apenas estrutura
    norma_in  = np.linalg.norm(sinal)
    norma_out = np.linalg.norm(sinal_out) + 1e-12
    if norma_in > 1e-10:
        sinal_out *= (norma_in / norma_out)

    # Atualizar beta (mesmo critério do domínio acústico)
    b_novo = (pow_spec.max() / (pow_spec.mean() + 1e-12)) / len(pow_spec)
    b_novo = float(np.clip(b_novo, 0.05, PHI3))
    return sinal_out, b_novo

def agente_eco_grad(sinal, n_ciclos=N_CICLOS):
    """Agente ECO completo para gradiente 1D — retorna sinal e histórico β."""
    N_fft  = len(np.fft.rfft(sinal))
    bins   = bandas_phi_grad(N_fft)
    beta   = 1.0
    betas  = []
    for _ in range(n_ciclos):
        for _ in range(N_STEPS):
            sinal, beta = eco_eq_grad(sinal, bins, beta)
        beta = float(np.clip(beta, 0.05, PHI3))
        betas.append(beta)
    return sinal, betas

def eco_gradiente_tensor(grad_tensor, n_ciclos=N_CICLOS):
    """Aplica ECO BEEP 880 a um tensor de gradiente PyTorch."""
    if grad_tensor.numel() < 32:
        return grad_tensor, 1.0   # muito pequeno — não processar
    shape = grad_tensor.shape
    g_np  = grad_tensor.detach().cpu().numpy().flatten().astype(np.float64)
    g_eco, betas = agente_eco_grad(g_np, n_ciclos=n_ciclos)
    g_out = torch.tensor(g_eco.reshape(shape),
                         dtype=grad_tensor.dtype,
                         device=grad_tensor.device)
    return g_out, betas[-1]

# ============================================================
#  MÉTRICAS ESPECTRAIS DO GRADIENTE
# ============================================================

def metricas_grad(g_np):
    X        = np.fft.rfft(g_np)
    pow_spec = np.abs(X) ** 2 + 1e-12
    p        = pow_spec / pow_spec.sum()
    H_norm   = float(-np.sum(p * np.log2(p))) / np.log2(len(p))
    planura  = float(np.exp(np.mean(np.log(pow_spec))) / np.mean(pow_spec))
    cumsum   = np.cumsum(pow_spec)
    bw_frac  = float(np.searchsorted(cumsum, cumsum[-1] * 0.95)) / len(pow_spec)
    s        = np.sort(pow_spec); n = len(s); idx = np.arange(1, n+1)
    gini     = float(2 * np.sum(idx * s) / (n * s.sum()) - (n+1)/n)
    return dict(H_norm=H_norm, planura=planura, bw_frac=bw_frac, gini=gini)

# ============================================================
#  MODELO — MLP
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
#  DADOS — MNIST (subconjunto)
# ============================================================

print("Carregando MNIST...")
tfm = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
ds_tr = datasets.MNIST('/tmp/mnist', train=True,  download=True, transform=tfm)
ds_te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tfm)
ds_tr = Subset(ds_tr, range(N_TREINO))
ds_te = Subset(ds_te, range(N_TESTE))
loader_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)
loader_te = DataLoader(ds_te, batch_size=500,   shuffle=False)
print(f"  Treino: {N_TREINO} amostras  |  Teste: {N_TESTE} amostras")
print()

# ============================================================
#  LOOP DE TREINAMENTO
# ============================================================

def treinar(usar_eco, label):
    torch.manual_seed(42)
    modelo    = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.9)

    hist = dict(loss=[], acc=[], beta=[], metricas=[])
    print(f"  [{label}]  ECO nos gradientes: {'SIM' if usar_eco else 'NÃO'}")

    for ep in range(N_EPOCHS):
        modelo.train()
        loss_sum, n_bat, betas_ep = 0.0, 0, []
        grad_amostra = None

        for data, target in loader_tr:
            optimizer.zero_grad()
            loss = criterion(modelo(data), target)
            loss.backward()

            if usar_eco:
                with torch.no_grad():
                    for p in modelo.parameters():
                        if p.grad is not None:
                            p.grad, beta_f = eco_gradiente_tensor(p.grad, N_CICLOS)
                            betas_ep.append(beta_f)

            # Captura gradiente da camada principal para métricas
            if grad_amostra is None:
                w = list(modelo.parameters())[1]
                if w.grad is not None:
                    grad_amostra = w.grad.detach().cpu().numpy().flatten().copy()

            optimizer.step()
            loss_sum += loss.item(); n_bat += 1

        # Avaliação
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

        beta_str = f"  β={beta_med:.4f}" if usar_eco else ""
        print(f"  Época {ep+1:2d}/{N_EPOCHS}  |  Loss: {loss_med:.4f}  |  Acc: {acc:.1f}%{beta_str}")

    print()
    return hist

# ============================================================
#  EXPERIMENTO
# ============================================================

print("=" * 62)
print("  EXPERIMENTO A — Baseline (SGD padrão)")
print("=" * 62)
hist_base = treinar(usar_eco=False, label="Baseline")

print("=" * 62)
print("  EXPERIMENTO B — ECO BEEP 880 nos gradientes")
print("=" * 62)
hist_eco = treinar(usar_eco=True, label="ECO φ")

# ============================================================
#  DIAGNÓSTICO
# ============================================================

def epocas_para(hist, threshold):
    for i, a in enumerate(hist['acc']):
        if a >= threshold:
            return i + 1
    return None

print("=" * 62)
print("  DIAGNÓSTICO — ECO BEEP 880 em Rede Neural")
print("=" * 62)

print(f"\n  Loss final:")
print(f"    Baseline : {hist_base['loss'][-1]:.4f}")
print(f"    ECO φ    : {hist_eco['loss'][-1]:.4f}")
d_loss = (hist_base['loss'][-1] - hist_eco['loss'][-1]) / hist_base['loss'][-1] * 100
print(f"    Δ        : {d_loss:+.1f}%  (positivo = ECO melhor)")

print(f"\n  Acurácia final (teste):")
print(f"    Baseline : {hist_base['acc'][-1]:.1f}%")
print(f"    ECO φ    : {hist_eco['acc'][-1]:.1f}%")

for thresh in [85, 90, 92]:
    eb = epocas_para(hist_base, thresh)
    ee = epocas_para(hist_eco,  thresh)
    sb = str(eb) if eb else f">{N_EPOCHS}"
    se = str(ee) if ee else f">{N_EPOCHS}"
    print(f"\n  Épocas para {thresh}% acc:")
    print(f"    Baseline : {sb}")
    print(f"    ECO φ    : {se}")

# Métricas espectrais dos gradientes
if hist_base['metricas'] and hist_eco['metricas']:
    mb = hist_base['metricas'][-1]
    me = hist_eco['metricas'][-1]
    print(f"\n  Métricas espectrais do gradiente (última época):")
    print(f"    {'Métrica':<22} {'Baseline':>10} {'ECO φ':>10}  Δ")
    print(f"    {'-'*52}")
    for k, nome in [('H_norm',  'Shannon H (norm)'),
                    ('planura', 'Planura espectral'),
                    ('bw_frac', 'BW_95 (fração)   '),
                    ('gini',    'Gini espectral   ')]:
        vb = mb.get(k, 0); ve = me.get(k, 0)
        delta = ve - vb
        print(f"    {nome:<22} {vb:>10.4f} {ve:>10.4f}  {delta:+.4f}")

# β convergência
if hist_eco['beta']:
    print(f"\n  β médio dos gradientes por época (ECO φ):")
    for i, b in enumerate(hist_eco['beta']):
        frac = b / PHI3 * 100
        print(f"    Época {i+1:2d}: β = {b:.6f}  ({frac:.1f}% de φ³)")
    print(f"  φ³ referência : {PHI3:.6f}")
    pct_final = hist_eco['beta'][-1] / PHI3 * 100
    print(f"  β final = {pct_final:.1f}% de φ³")

# ============================================================
#  GRÁFICO
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('ECO BEEP 880 em Rede Neural (MNIST)', fontsize=13, fontweight='bold')
eps = range(1, N_EPOCHS + 1)

axes[0].plot(eps, hist_base['loss'], 'b-o', label='Baseline', lw=2, ms=5)
axes[0].plot(eps, hist_eco['loss'],  'r-o', label='ECO φ',    lw=2, ms=5)
axes[0].set_title('Loss de treinamento'); axes[0].set_xlabel('Época')
axes[0].set_ylabel('Cross-entropy loss'); axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(eps, hist_base['acc'], 'b-o', label='Baseline', lw=2, ms=5)
axes[1].plot(eps, hist_eco['acc'],  'r-o', label='ECO φ',    lw=2, ms=5)
axes[1].set_title('Acurácia no teste'); axes[1].set_xlabel('Época')
axes[1].set_ylabel('Acurácia (%)'); axes[1].legend(); axes[1].grid(alpha=.3)

if hist_eco['beta']:
    axes[2].plot(eps, hist_eco['beta'], 'g-o', lw=2, ms=5, label='β gradiente')
    axes[2].axhline(PHI3,  color='r', ls='--', alpha=.7, label=f'φ³={PHI3:.3f}')
    axes[2].axhline(SQRT5, color='b', ls='--', alpha=.7, label=f'√5={SQRT5:.3f}')
    axes[2].set_title('β médio dos gradientes'); axes[2].set_xlabel('Época')
    axes[2].set_ylabel('β'); axes[2].legend(); axes[2].grid(alpha=.3)

plt.tight_layout()
plt.savefig('rede_neural_eco.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  Gráfico: rede_neural_eco.png")

print()
print("=" * 62)
print("  ECO BEEP 880 EM REDE NEURAL — CONCLUÍDO")
print("=" * 62)
print()
print("  Playback Colab:")
print("  from IPython.display import Image")
print("  Image('rede_neural_eco.png')")
