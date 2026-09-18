# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_NN_Estagio3_Autoencoder.py
Vitor Edson Delavi · Florianópolis · 2026

ESTÁGIO III — AUTOENCODER SOBRE COMPLEXIDADE DE FUNÇÕES

Uma rede neural é uma consequência de complexidade de função.
Para testá-la, inserimos complexidade real: cinco tipos de sinal
com estruturas distintas, misturados em proporções aleatórias.

Tipos de sinal (complexidade de funções):
  φ-estruturado:
    1. EcoBIP       — quadrada + FM_φ (sinal nativo do Alpha-Phi)
    2. Harmônico φ  — superposição com razões de frequência = φ
  Não-φ:
    3. Chirp        — varredura linear de frequência
    4. Harmônico arbitrário — razões irracionais ≠ φ
    5. Ruído branco — sem estrutura

Arquitetura — autoencoder Fibonacci completo:
  Encoder: 55 → 34 → 21 → 13 → 8 → 5 → 3  (com SeptroProjecao)
  Latente: 3D — o campo destilado (SEAL)
  Decoder: 3 → 5 → 8 → 13 → 21 → 34 → 55  (trajetória Sépstro invertida)

Loss:
  L_total = L_reconstrução + SEAL · L_campo
  L_campo = Σ |Coh_real - Coh_target|²  (por camada do encoder)

Verificação B (pós-treinamento):
  AP vs Xavier — qual tem menor erro de reconstrução
  sobre sinais φ-estruturados vs. sinais não-φ?
  Hipótese: o campo φ confere vantagem específica sobre sua própria geometria.

Duas redes treinadas na mesma tarefa, mesmos dados:
  RedeAP   — arquitetura Alpha-Phi (este arquivo)
  RedeXav  — mesma topologia, inicialização Xavier, sem Sépstro
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constantes ─────────────────────────────────────────────────────────────────
PHI      = (1 + np.sqrt(5)) / 2
ALPHA    = 1 / 137.035999
ALPHA_OP = 1 / 3
SEAL     = 1 / PHI
THETA_R  = np.arctan(2)
SR       = 44100
BASE     = 880

# ── Arquitetura ────────────────────────────────────────────────────────────────
DIMS_ENC = [55, 34, 21, 13, 8, 5, 3]
DIMS_DEC = [3,   5,  8, 13, 21, 34, 55]
N_ENC    = len(DIMS_ENC) - 1   # 6 camadas encoder
N_DEC    = len(DIMS_DEC) - 1   # 6 camadas decoder

# ── Hiperparâmetros ────────────────────────────────────────────────────────────
N_EPOCHS    = 300
BATCH_SIZE  = 64
LR          = 1e-3
N_TRAIN     = 2000    # amostras de treinamento
N_TEST_PHI  = 200     # amostras de teste φ-estruturado
N_TEST_ARB  = 200     # amostras de teste não-φ


# ══════════════════════════════════════════════════════════════════════════════
# GERADOR DE SINAIS — complexidade de funções
# ══════════════════════════════════════════════════════════════════════════════

def _normaliza(sig):
    m = np.max(np.abs(sig)) + 1e-8
    return sig / m

def gerar_ecobip(n=1024):
    t = np.linspace(0, 1, n, endpoint=False)
    q = np.sign(np.sin(2*np.pi*BASE*t))
    f = np.sin(2*np.pi*BASE*t + PHI*np.sin(2*np.pi*(BASE/4)*t))
    return _normaliza((1-ALPHA_OP)*q + ALPHA_OP*f)

def gerar_harmonico_phi(n=1024):
    """Superposição de senóides com razões de frequência φ."""
    t = np.linspace(0, 1, n, endpoint=False)
    f0 = BASE * (0.5 + np.random.rand())
    sig = sum(
        (SEAL**k) * np.sin(2*np.pi * f0 * (PHI**k) * t + np.random.rand()*2*np.pi)
        for k in range(5)
        if f0 * (PHI**k) < SR/2
    )
    return _normaliza(sig)

def gerar_chirp(n=1024):
    """Varredura linear de frequência."""
    t  = np.linspace(0, 1, n, endpoint=False)
    f0 = 200 + np.random.rand()*600
    f1 = 2000 + np.random.rand()*3000
    phi0 = np.random.rand() * 2*np.pi
    return _normaliza(np.sin(2*np.pi*(f0 + (f1-f0)*t/2)*t + phi0))

def gerar_harmonico_arb(n=1024):
    """Superposição com razões irracionais não-φ (e, π, √2, √3, √5/2)."""
    t  = np.linspace(0, 1, n, endpoint=False)
    f0 = 300 + np.random.rand()*400
    razoes = [1.0, np.e/2, np.pi/3, np.sqrt(2), np.sqrt(3)/1.2]
    sig = sum(
        (0.7**k) * np.sin(2*np.pi*f0*r*t + np.random.rand()*2*np.pi)
        for k, r in enumerate(razoes)
        if f0*r < SR/2
    )
    return _normaliza(sig)

def gerar_ruido(n=1024):
    return _normaliza(np.random.randn(n))

GERADORES_PHI = [gerar_ecobip, gerar_harmonico_phi]
GERADORES_ARB = [gerar_chirp, gerar_harmonico_arb, gerar_ruido]

def comprimir(sig, n=55):
    idx = np.linspace(0, len(sig)-1, n).astype(int)
    return sig[idx].astype(np.float32)

def gerar_amostra(phi_dominant=None):
    """
    Gera um sinal como mistura aleatória de todos os tipos.
    phi_dominant=True  → componente φ domina (teste B)
    phi_dominant=False → componente não-φ domina (teste B)
    phi_dominant=None  → mistura aleatória livre (treinamento)
    """
    if phi_dominant is True:
        w_phi = 0.6 + 0.4 * np.random.rand()
        w_arb = 1.0 - w_phi
    elif phi_dominant is False:
        w_arb = 0.6 + 0.4 * np.random.rand()
        w_phi = 1.0 - w_arb
    else:
        w_phi = np.random.rand()
        w_arb = 1.0 - w_phi

    sig_phi = np.mean([g() for g in GERADORES_PHI], axis=0) * w_phi
    sig_arb = np.mean([g() for g in GERADORES_ARB], axis=0) * w_arb
    sig = _normaliza(sig_phi + sig_arb)
    return comprimir(sig, 55)

def gerar_dataset(n, phi_dominant=None):
    X = np.stack([gerar_amostra(phi_dominant) for _ in range(n)])
    return torch.tensor(X)


# ══════════════════════════════════════════════════════════════════════════════
# SÉPSTRO E PROJEÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def medir_coh(v: torch.Tensor) -> float:
    a = torch.abs(v.flatten()) + 1e-10
    p = a / a.sum()
    H     = -torch.sum(p * torch.log(p))
    H_max = torch.log(torch.tensor(float(max(len(a), 2))))
    return float(1.0 - H / H_max)

def target_coh(nivel, n_total, inverso=False):
    r = nivel / max(n_total - 1, 1)
    if inverso:
        r = 1.0 - r
    return ALPHA + r * (1.0 - 2.0 * ALPHA)

class SeptroProj(nn.Module):
    def __init__(self, nivel, n_total, inverso=False):
        super().__init__()
        self.target = target_coh(nivel, n_total, inverso)
        self.coh_depois = 0.0

    def forward(self, x):
        coh = medir_coh(x)
        delta = self.target - coh
        fator = delta * SEAL
        norm  = torch.norm(x) + 1e-8
        xn    = x / norm
        xp    = xn * (1.0 + fator * torch.abs(xn))
        xp    = xp / (torch.norm(xp) + 1e-8) * norm
        self.coh_depois = medir_coh(xp)
        return xp


# ══════════════════════════════════════════════════════════════════════════════
# ARQUITETURA — Autoencoder Alpha-Phi
# ══════════════════════════════════════════════════════════════════════════════

class CamadaAP(nn.Module):
    def __init__(self, d_in, d_out, nivel, n_total, inverso=False):
        super().__init__()
        self.fc   = nn.Linear(d_in, d_out)
        self.proj = SeptroProj(nivel, n_total, inverso)
        self._init()

    def _init(self):
        escala = PHI ** (-float(self.proj.target * N_ENC))
        nn.init.normal_(self.fc.weight, 0.0, max(escala, 0.01))
        nn.init.constant_(self.fc.bias, ALPHA)

    def forward(self, x):
        return self.proj(torch.tanh(self.fc(x)))


class AutoencoderAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([
            CamadaAP(DIMS_ENC[i], DIMS_ENC[i+1], nivel=i, n_total=N_ENC)
            for i in range(N_ENC)
        ])
        self.decoder = nn.ModuleList([
            CamadaAP(DIMS_DEC[i], DIMS_DEC[i+1], nivel=i, n_total=N_DEC, inverso=True)
            for i in range(N_DEC)
        ])
        self.latente = None

    def encode(self, x):
        for c in self.encoder:
            x = c(x)
        self.latente = x
        return x

    def decode(self, z):
        for c in self.decoder:
            z = c(z)
        return z

    def forward(self, x):
        return self.decode(self.encode(x))

    def loss_campo(self):
        total = 0.0
        for c in self.encoder:
            total += (c.proj.coh_depois - c.proj.target) ** 2
        for c in self.decoder:
            total += (c.proj.coh_depois - c.proj.target) ** 2
        return total / (N_ENC + N_DEC)


class AutoencoderXavier(nn.Module):
    """Mesma topologia, inicialização Xavier, sem Sépstro — linha de base."""
    def __init__(self):
        super().__init__()
        dims = DIMS_ENC + DIMS_DEC[1:]
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# LOOP DE TREINAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def treinar(modelo, X_train, is_ap=True, verbose=True):
    opt  = optim.Adam(modelo.parameters(), lr=LR)
    mse  = nn.MSELoss()
    hist = []

    n = len(X_train)
    for ep in range(N_EPOCHS):
        idx   = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            batch = X_train[idx[i:i+BATCH_SIZE]]
            opt.zero_grad()
            rec = modelo(batch)
            l_rec = mse(rec, batch)
            l_campo = torch.tensor(modelo.loss_campo()) if is_ap else torch.tensor(0.0)
            loss = l_rec + SEAL * l_campo
            loss.backward()
            opt.step()
            ep_loss += l_rec.item() * len(batch)

        ep_loss /= n
        hist.append(ep_loss)
        if verbose and (ep+1) % 50 == 0:
            tag = "AP" if is_ap else "Xav"
            print(f"  [{tag}] época {ep+1:>3}/{N_EPOCHS}  L_rec={ep_loss:.6f}")

    return hist


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════

torch.manual_seed(42)
np.random.seed(42)

print("Gerando dados...")
X_train  = gerar_dataset(N_TRAIN)
X_phi    = gerar_dataset(N_TEST_PHI, phi_dominant=True)
X_arb    = gerar_dataset(N_TEST_ARB, phi_dominant=False)
print(f"  Treino: {X_train.shape}  |  Teste φ: {X_phi.shape}  |  Teste arb: {X_arb.shape}")

print("\nTreinando AutoencoderAP...")
ae_ap  = AutoencoderAP()
hist_ap = treinar(ae_ap, X_train, is_ap=True)

print("\nTreinando AutoencoderXavier (linha de base)...")
ae_xav = AutoencoderXavier()
hist_xav = treinar(ae_xav, X_train, is_ap=False)

# ── Verificação B — φ-sensibilidade ───────────────────────────────────────────
mse_fn = nn.MSELoss()

with torch.no_grad():
    ap_phi = mse_fn(ae_ap(X_phi),  X_phi).item()
    ap_arb = mse_fn(ae_ap(X_arb),  X_arb).item()
    xv_phi = mse_fn(ae_xav(X_phi), X_phi).item()
    xv_arb = mse_fn(ae_xav(X_arb), X_arb).item()

print("\n" + "="*60)
print("Verificação B — Sensibilidade φ")
print("="*60)
print(f"                   φ-estruturado    não-φ    razão (φ/arb)")
print(f"  AutoencoderAP :   {ap_phi:.6f}    {ap_arb:.6f}    {ap_phi/ap_arb:.4f}")
print(f"  AutoencoderXav:   {xv_phi:.6f}    {xv_arb:.6f}    {xv_phi/xv_arb:.4f}")
print()
if ap_phi < xv_phi:
    vantagem = (xv_phi - ap_phi) / xv_phi * 100
    print(f"  → AP reconstrói sinais φ-estruturados {vantagem:.1f}% melhor que Xavier.")
else:
    print(f"  → Sem vantagem AP sobre Xavier em sinais φ nesta execução.")
if ap_phi < ap_arb:
    print(f"  → AP mais preciso em φ do que em arbitrário: campo sensível à geometria.")

# ── Espaço latente ─────────────────────────────────────────────────────────────
with torch.no_grad():
    lat_phi = ae_ap.encode(X_phi).numpy()
    lat_arb = ae_ap.encode(X_arb).numpy()

print(f"\nEspaço latente 3D:")
print(f"  Centróide φ   : {lat_phi.mean(0).round(4)}")
print(f"  Centróide arb : {lat_arb.mean(0).round(4)}")
dist = np.linalg.norm(lat_phi.mean(0) - lat_arb.mean(0))
print(f"  Distância entre centróides: {dist:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Curva de Loss — AP vs Xavier',
        'Verificação B — Erro de Reconstrução por Tipo',
        'Espaço Latente 3D — Separação φ vs. Arbitrário',
        'Reconstrução — Amostra φ (AP vs Xavier)'
    ),
    specs=[
        [{'type':'xy'},    {'type':'xy'}],
        [{'type':'scene'}, {'type':'xy'}],
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# Painel 1 — loss
ep_ax = list(range(1, N_EPOCHS+1))
fig.add_trace(go.Scatter(x=ep_ax, y=hist_ap,  mode='lines', name='AP',
                         line=dict(color='#FFD700', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=ep_ax, y=hist_xav, mode='lines', name='Xavier',
                         line=dict(color='#4A90D9', width=2)), row=1, col=1)

# Painel 2 — barras φ-sensibilidade
categorias = ['φ-estruturado', 'Não-φ']
fig.add_trace(go.Bar(name='AP',    x=categorias, y=[ap_phi, ap_arb],
                     marker_color='#FFD700'), row=1, col=2)
fig.add_trace(go.Bar(name='Xavier', x=categorias, y=[xv_phi, xv_arb],
                     marker_color='#4A90D9'), row=1, col=2)

# Painel 3 — espaço latente 3D
fig.add_trace(go.Scatter3d(
    x=lat_phi[:,0], y=lat_phi[:,1], z=lat_phi[:,2],
    mode='markers', name='φ-estruturado',
    marker=dict(size=2.5, color='#FFD700', opacity=0.6)
), row=2, col=1)
fig.add_trace(go.Scatter3d(
    x=lat_arb[:,0], y=lat_arb[:,1], z=lat_arb[:,2],
    mode='markers', name='Não-φ',
    marker=dict(size=2.5, color='#4A90D9', opacity=0.6)
), row=2, col=1)

# Painel 4 — reconstrução
idx_ex = 0
amostra = X_phi[idx_ex:idx_ex+1]
with torch.no_grad():
    rec_ap  = ae_ap(amostra).numpy().flatten()
    rec_xav = ae_xav(amostra).numpy().flatten()
orig = amostra.numpy().flatten()
nx   = list(range(55))

fig.add_trace(go.Scatter(x=nx, y=orig,    mode='lines', name='Original',
                         line=dict(color='white',   width=2)),        row=2, col=2)
fig.add_trace(go.Scatter(x=nx, y=rec_ap,  mode='lines', name='AP rec',
                         line=dict(color='#FFD700', width=1.5, dash='dot')), row=2, col=2)
fig.add_trace(go.Scatter(x=nx, y=rec_xav, mode='lines', name='Xav rec',
                         line=dict(color='#4A90D9', width=1.5, dash='dot')), row=2, col=2)

fig.update_layout(
    template='plotly_dark',
    height=900,
    title=dict(
        text='Alpha-Phi NN — Estágio III · Autoencoder + Verificação φ',
        font=dict(size=15)
    ),
    barmode='group'
)
fig.update_xaxes(title_text='Época',    row=1, col=1)
fig.update_xaxes(title_text='Tipo',     row=1, col=2)
fig.update_yaxes(title_text='MSE',      row=1, col=1)
fig.update_yaxes(title_text='MSE',      row=1, col=2)
fig.update_xaxes(title_text='Neurônio', row=2, col=2)

# Câmera do espaço latente
fig.update_scenes(
    camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
    xaxis_title='Lat 0', yaxis_title='Lat 1', zaxis_title='Lat 2'
)

fig.show()

print("\nEstágio III — Autoencoder Alpha-Phi — concluído.")
print(f"  Arquitetura: {' → '.join(str(d) for d in DIMS_ENC)} | "
      f"{' → '.join(str(d) for d in DIMS_DEC)}")
print(f"  Complexidade de funções: {len(GERADORES_PHI)} tipos φ + "
      f"{len(GERADORES_ARB)} tipos não-φ em mistura aleatória")
print(f"  Próximo: STTG sobre self.ativacoes — Grade R no campo latente")
