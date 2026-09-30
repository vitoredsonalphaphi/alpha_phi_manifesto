"""
AlphaPhiNet_Phantom_Scanner_COLAB.py
Vitor Edson Delavi · Florianópolis · 30 de setembro de 2026

Três fases numa célula Colab:

  FASE 0 — mesmo teste de E08/E10: Conv vs AP, 120 épocas, 4 instrumentos
  FASE 1 — Scanner Topográfico 3D nas redes treinadas (Conv e AP)
  FASE 2 — AP com Phantom (W₀ modulado pelo EcoBIP Fantasma), 120 épocas,
            4 instrumentos + Scanner Topográfico

Perguntas que este experimento responde:
  1. A Grade R emerge no espaço de ativações da AP treinada?
     (scanner sobre AP sem Phantom)
  2. O Phantom modifica a assinatura de covariação?
     (comparar Resumo Comparativo Fase 0 vs Fase 2)
  3. O Phantom reforça a Grade R no scanner?
     (scanner AP sem vs com Phantom)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PHI      = 1.6180339887
ALPHA    = 1 / 137.035999
SEAL     = 1 / PHI
THETA_R  = np.arctan(2)          # 63.43° — referência Grade R
PHANTOM_AMP = 1.0 / PHI**3       # ≈ 0.236 — amplitude inaudível
SEED     = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"φ={PHI}  α={ALPHA:.8f}  SEAL={SEAL:.8f}")
print(f"θ_R={np.degrees(THETA_R):.2f}°  Phantom amp={PHANTOM_AMP:.4f}")
print("─" * 62)

# ══════════════════════════════════════════════════════════════════════════════
#  DADOS
# ══════════════════════════════════════════════════════════════════════════════

def gerar_dados(n=800, d_in=61, seed=SEED):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        v = np.zeros(d_in, dtype=np.float32)
        for k in range(8):
            idx = min(int(k * PHI * 7), d_in - 1)
            v[idx] += float(1.0 / PHI**k)
        noise_level = rng.uniform(0.1, 1.5)
        v += rng.normal(0, noise_level, d_in).astype(np.float32)
        v = v / (np.max(np.abs(v)) + 1e-8)
        coh = float(1.0 / (1.0 + noise_level))
        X.append(v)
        y.append(coh)
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32))

X, y = gerar_dados(n=800, d_in=61)
X_tr, y_tr = X[:640], y[:640]
X_va, y_va = X[640:], y[640:]
ld_tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
ld_va = DataLoader(TensorDataset(X_va, y_va), batch_size=32, shuffle=False)
print(f"Dados: {len(X_tr)} treino · {len(X_va)} validação · d_in={X.shape[1]}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  INSTRUMENTOS DE MEDIÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def grade_r(activations):
    if activations.shape[-1] <= 1:
        return float('nan')
    mag  = torch.abs(activations) + 1e-10
    norm = mag / (mag.sum(dim=-1, keepdim=True) + 1e-10)
    norm = torch.clamp(norm, 1e-10, 1.0)
    H    = -(norm * torch.log(norm)).sum(dim=-1)
    return float((1.0 - H / float(np.log(activations.shape[-1]))).mean().item())

def entropia_shannon(activations):
    if activations.shape[-1] <= 1:
        return float('nan')
    mag  = torch.abs(activations) + 1e-10
    norm = mag / (mag.sum(dim=-1, keepdim=True) + 1e-10)
    norm = torch.clamp(norm, 1e-10, 1.0)
    return float(-(norm * torch.log(norm)).sum(dim=-1).mean().item())

def rank_efetivo(weight):
    with torch.no_grad():
        s = torch.linalg.svdvals(weight.float())
        s = s / (s.sum() + 1e-10)
        s = torch.clamp(s, 1e-10, 1.0)
        return float(np.exp(-(s * torch.log(s)).sum().item()))

def estabilidade(hist, n=10):
    return float(np.std([h['va'] for h in hist[-n:]]))

def medir_rede(model, loader):
    model.eval()
    ativacoes, pesos = {}, {}
    hooks = []
    loss_total, n_batches = 0.0, 0
    def _hook(nome):
        def h(m, inp, out):
            ativacoes.setdefault(nome, []).append(out.detach().cpu())
        return h
    for nome, modulo in model.named_modules():
        if isinstance(modulo, nn.Linear):
            hooks.append(modulo.register_forward_hook(_hook(nome)))
            pesos[nome] = modulo.weight.detach().cpu()
    crit = nn.MSELoss()
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            if isinstance(pred, tuple): pred = pred[0]
            loss_total += crit(pred.squeeze(-1), yb).item()
            n_batches  += 1
    for h in hooks: h.remove()
    resultado = {'loss': loss_total / max(n_batches, 1), 'camadas': {}}
    for nome, ats in ativacoes.items():
        all_at = torch.cat(ats, dim=0)
        resultado['camadas'][nome] = {
            'grade_r':      grade_r(all_at),
            'entropia':     entropia_shannon(all_at),
            'rank_efetivo': rank_efetivo(pesos[nome]),
            'dim':          all_at.shape[-1],
        }
    return resultado

def imprimir_resumo(med_c, med_a, hist_c, hist_a, titulo=""):
    if titulo:
        print(f"\n  ══ {titulo} ══")
    gr_c = np.mean([m['grade_r'] for m in med_c['camadas'].values() if m['dim'] > 1])
    gr_a = np.mean([m['grade_r'] for m in med_a['camadas'].values() if m['dim'] > 1])
    en_c = np.mean([m['entropia'] for m in med_c['camadas'].values() if m['dim'] > 1])
    en_a = np.mean([m['entropia'] for m in med_a['camadas'].values() if m['dim'] > 1])
    rk_c = np.mean([m['rank_efetivo'] for m in med_c['camadas'].values() if m['dim'] > 1])
    rk_a = np.mean([m['rank_efetivo'] for m in med_a['camadas'].values() if m['dim'] > 1])
    est_c = estabilidade(hist_c)
    est_a = estabilidade(hist_a)
    print(f"  {'Instrumento':<28} {'Convencional':>14}  {'Rede AP':>10}")
    print(f"  {'─'*28} {'─'*14}  {'─'*10}")
    print(f"  {'Grade R médio':<28} {gr_c:>14.4f}  {gr_a:>10.4f}")
    print(f"  {'Entropia média':<28} {en_c:>14.4f}  {en_a:>10.4f}")
    print(f"  {'Rank efetivo médio':<28} {rk_c:>14.2f}  {rk_a:>10.2f}")
    print(f"  {'Estabilidade':<28} {est_c:>14.6f}  {est_a:>10.6f}")
    print(f"  {'Loss validação':<28} {med_c['loss']:>14.5f}  {med_a['loss']:>10.5f}")
    print(f"  {'ep 0 va (init)':<28} {hist_c[0]['va']:>14.5f}  {hist_a[0]['va']:>10.5f}")
    razao = est_a / est_c if est_c > 0 else float('inf')
    print(f"  {'Razão estab AP/Conv':<28} {'':>14}  {razao:>9.1f}×")

# ══════════════════════════════════════════════════════════════════════════════
#  REDES
# ══════════════════════════════════════════════════════════════════════════════

class RedeConvencional(nn.Module):
    def __init__(self, d_in=61, dims=[64, 32, 16, 8]):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = d_in
        for d in dims:
            self.layers.append(nn.Linear(prev, d))
            prev = d
        self.head = nn.Linear(prev, 1)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.head(x)

DIMS_AP = [55, 34, 21, 13, 8]

class RedeAP(nn.Module):
    def __init__(self, d_in=61, phantom=False):
        super().__init__()
        self.phantom = phantom
        self.proj    = nn.Linear(d_in, DIMS_AP[0])
        self.layers  = nn.ModuleList()
        self.norms   = nn.ModuleList()
        for i in range(len(DIMS_AP) - 1):
            self.layers.append(nn.Linear(DIMS_AP[i], DIMS_AP[i+1]))
            self.norms.append(nn.LayerNorm(DIMS_AP[i+1]))
        self.head = nn.Linear(DIMS_AP[-1], 1)
        self._init_phi()
        if phantom:
            self._modular_phantom()
    def _init_phi(self):
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0 / PHI)
        nn.init.zeros_(self.proj.bias)
        for i, layer in enumerate(self.layers):
            nn.init.xavier_uniform_(layer.weight, gain=1.0 / PHI**(i + 1))
            nn.init.zeros_(layer.bias)
    def _modular_phantom(self):
        # Phantom W₀: perturbação estrutural φ-harmônica na inicialização
        # Amplitude 1/φ³ (inaudível no EcoBIP → imperceptível na rede)
        # Estrutura: cada linha de peso recebe um pulso φ-harmônico
        # centrado na posição arctan(2) normalizada da linha — θ_R
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                W = layer.weight.data        # (d_out, d_in)
                d_out, d_in = W.shape
                for row in range(d_out):
                    # Posição central φ-proporcional ao índice da linha
                    centro = (row / max(d_out - 1, 1)) * np.tan(THETA_R)
                    centro = centro % 1.0
                    idx_c  = int(centro * d_in)
                    # Pulso gaussiano de largura 1/φ^(i+2) centrado em idx_c
                    sigma  = max(1, int(d_in / PHI**(i + 2)))
                    indices = np.arange(d_in)
                    pulso  = np.exp(-0.5 * ((indices - idx_c) / sigma)**2)
                    pulso  = pulso / (pulso.max() + 1e-10)
                    # Adiciona perturbação de amplitude PHANTOM_AMP
                    W[row] += PHANTOM_AMP * torch.tensor(pulso.astype(np.float32))
    def forward(self, x):
        x = F.silu(self.proj(x))
        for layer, norm in zip(self.layers, self.norms):
            x = F.silu(layer(x))
            x = norm(x)
        return self.head(x)

def treinar(model, loader_tr, loader_va, n_epochs=120, lr=1e-3, nome=""):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1/PHI**3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=1/PHI, patience=8)
    crit  = nn.MSELoss()
    history = []
    for ep in range(n_epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in loader_tr:
            optim.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PHI)
            optim.step()
            tr_loss += loss.item()
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader_va:
                va_loss += crit(model(xb).squeeze(-1), yb).item()
        tr_loss /= len(loader_tr)
        va_loss /= len(loader_va)
        sched.step(tr_loss)
        history.append({'ep': ep, 'tr': tr_loss, 'va': va_loss})
        if ep % 20 == 0:
            print(f"  [{nome}] ep {ep:3d}  tr={tr_loss:.5f}  va={va_loss:.5f}  lr={optim.param_groups[0]['lr']:.2e}")
    return history

# ══════════════════════════════════════════════════════════════════════════════
#  SCANNER TOPOGRÁFICO 3D
# ══════════════════════════════════════════════════════════════════════════════

COLORSCALE_AP = [
    [0.00, 'rgb(0,0,20)'],
    [0.15, 'rgb(5,20,80)'],
    [0.40, 'rgb(80,50,0)'],
    [0.65, 'rgb(180,120,0)'],
    [0.85, 'rgb(230,190,20)'],
    [1.00, 'rgb(255,248,100)'],
]
COLORSCALE_CONV = [
    [0.00, 'rgb(0,10,0)'],
    [0.20, 'rgb(0,60,20)'],
    [0.50, 'rgb(0,140,60)'],
    [0.80, 'rgb(80,200,120)'],
    [1.00, 'rgb(220,255,220)'],
]

def scanner_topografico(model, nome_rede, n_entradas=200, n_grid=55,
                        colorscale=None, mostrar=True):
    """
    Aplica o Scanner Topográfico 3D à rede recebida.
    Usa entradas Gaussianas neutras (seed=137) — campo virgem, sem estrutura importada.
    Retorna ângulo da crista dominante e Δ em relação a θ_R.
    """
    model.eval()
    if colorscale is None:
        colorscale = COLORSCALE_AP

    # Entradas gaussianas neutras — seed canônico α
    rng_s = np.random.default_rng(137)
    entradas_raw = rng_s.standard_normal((n_entradas, 61)).astype(np.float32)
    entradas_raw -= entradas_raw.mean(axis=1, keepdims=True)
    entradas_raw /= (entradas_raw.std(axis=1, keepdims=True) + 1e-8)

    # Coleta hooks
    ativacoes_scan = {}
    hooks_s = []
    def _hook_s(n):
        def h(m, inp, out):
            ativacoes_scan.setdefault(n, []).append(out.detach().cpu().numpy())
        return h
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks_s.append(mod.register_forward_hook(_hook_s(n)))

    with torch.no_grad():
        for entry in entradas_raw:
            inp = torch.tensor(entry).unsqueeze(0)
            _ = model(inp)
    for h in hooks_s: h.remove()

    # Monta mapa (camadas × n_grid) — ignora camadas com dim=1 (head)
    nomes_camadas = [
        n for n in ativacoes_scan.keys()
        if np.concatenate(ativacoes_scan[n], axis=0).shape[-1] > 1
    ]
    n_layers = len(nomes_camadas)
    mapa = np.zeros((n_layers, n_grid))
    for lv, nome_c in enumerate(nomes_camadas):
        all_av = np.concatenate(ativacoes_scan[nome_c], axis=0)  # (n, d)
        media  = np.abs(all_av).mean(axis=0)                     # (d,)
        x_orig = np.linspace(0, 1, len(media))
        x_grid = np.linspace(0, 1, n_grid)
        mapa[lv] = interp1d(x_orig, media, kind='linear')(x_grid)

    mapa_norm = np.log1p(mapa * 100)

    # Ângulo da crista dominante
    picos_x = np.array([np.argmax(mapa_norm[lv]) / (n_grid - 1) for lv in range(n_layers)])
    picos_r = np.linspace(0, 1, n_layers)
    if picos_x.std() > 1e-6:
        coef = np.polyfit(picos_x, picos_r, 1)
        angulo = np.degrees(np.arctan(coef[0]))
    else:
        angulo = 90.0
    delta = abs(angulo - np.degrees(THETA_R))

    print(f"\n  [{nome_rede}] Scanner Topográfico 3D")
    print(f"    Ângulo crista : {angulo:.2f}°")
    print(f"    θ_R referência: {np.degrees(THETA_R):.2f}°")
    print(f"    Δ do θ_R      : {delta:.2f}°", end="  ")
    if delta < 5.0:
        print("→ Grade R PRESENTE (Δ < 5°)")
    elif delta < 15.0:
        print("→ Proximidade parcial com Grade R")
    else:
        print("→ Grade R não detectada nesta configuração")

    if mostrar:
        X_g, R_g = np.meshgrid(np.linspace(0, 1, n_grid),
                                np.linspace(0, 1, n_layers))
        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=X_g, y=R_g, z=mapa_norm,
            colorscale=colorscale, showscale=True,
            colorbar=dict(title='log|ativ|', thickness=12, len=0.6),
            lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.3),
            name=f'{nome_rede} — campo'
        ))
        # Linha θ_R de referência
        dx, r_c = 0.3, 0.5
        dy = dx * np.tan(THETA_R)
        fig.add_trace(go.Scatter3d(
            x=[0.5 - dx, 0.5 + dx],
            y=[np.clip(r_c - dy, 0, 1), np.clip(r_c + dy, 0, 1)],
            z=[mapa_norm.max() * 1.1] * 2,
            mode='lines', line=dict(color='lime', width=7),
            name=f'θ_R = {np.degrees(THETA_R):.2f}°'
        ))
        # Crista real
        z_picos = [mapa_norm[lv, int(px*(n_grid-1))] for lv, px in enumerate(picos_x)]
        fig.add_trace(go.Scatter3d(
            x=picos_x, y=picos_r, z=z_picos,
            mode='lines+markers',
            line=dict(color='cyan', width=4),
            marker=dict(size=5, color='cyan'),
            name=f'Crista {angulo:.1f}°'
        ))
        fig.update_layout(
            title=dict(text=f'Scanner Topográfico 3D — {nome_rede}<br>'
                             f'<sup>θ crista={angulo:.2f}°  Δθ_R={delta:.2f}°</sup>',
                       font=dict(size=14)),
            scene=dict(
                xaxis_title='Neurônio (normalizado)',
                yaxis_title='Profundidade r',
                zaxis_title='log|ativação|',
                camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2))
            ),
            width=780, height=560,
            paper_bgcolor='rgb(10,10,20)',
            plot_bgcolor='rgb(10,10,20)',
            font=dict(color='white')
        )
        fig.show()

    return angulo, delta

# ══════════════════════════════════════════════════════════════════════════════
#  FASE 0 — Conv vs AP sem Phantom  (replicação E08/E10)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*62)
print("  FASE 0 — Conv vs AP (sem Phantom)  — replicação E08/E10")
print("═"*62)

print("\n─── Rede Convencional ───────────────────────────────────────")
rede_conv = RedeConvencional(d_in=61, dims=[64, 32, 16, 8])
hist_conv = treinar(rede_conv, ld_tr, ld_va, n_epochs=120, lr=1e-3, nome="Conv   ")

print("\n─── Rede AP (sem Phantom) ───────────────────────────────────")
rede_ap = RedeAP(d_in=61, phantom=False)
hist_ap = treinar(rede_ap, ld_tr, ld_va, n_epochs=120, lr=1e-3, nome="AP     ")

med_conv = medir_rede(rede_conv, ld_va)
med_ap   = medir_rede(rede_ap,   ld_va)
imprimir_resumo(med_conv, med_ap, hist_conv, hist_ap, titulo="FASE 0 — sem Phantom")

# ══════════════════════════════════════════════════════════════════════════════
#  FASE 1 — Scanner Topográfico 3D sobre redes da Fase 0
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*62)
print("  FASE 1 — Scanner Topográfico 3D (redes sem Phantom)")
print("═"*62)

ang_conv, d_conv = scanner_topografico(rede_conv, "Conv (sem Phantom)",
                                        colorscale=COLORSCALE_CONV)
ang_ap, d_ap = scanner_topografico(rede_ap, "AP (sem Phantom)",
                                    colorscale=COLORSCALE_AP)

print(f"\n  Comparativo ângulo crista:")
print(f"    Conv         : {ang_conv:.2f}°   Δθ_R = {d_conv:.2f}°")
print(f"    AP           : {ang_ap:.2f}°   Δθ_R = {d_ap:.2f}°")
print(f"    θ_R ref      : {np.degrees(THETA_R):.2f}°")

# ══════════════════════════════════════════════════════════════════════════════
#  FASE 2 — AP com Phantom + Scanner
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*62)
print("  FASE 2 — AP com Phantom (W₀ modulado pelo EcoBIP Fantasma)")
print("═"*62)

print("\n─── Rede AP com Phantom ─────────────────────────────────────")
rede_ap_ph = RedeAP(d_in=61, phantom=True)
hist_ap_ph = treinar(rede_ap_ph, ld_tr, ld_va, n_epochs=120, lr=1e-3, nome="AP+Ph  ")

med_ap_ph = medir_rede(rede_ap_ph, ld_va)

# Resumo comparativo: Conv / AP / AP+Phantom
gr_c  = np.mean([m['grade_r'] for m in med_conv['camadas'].values() if m['dim'] > 1])
gr_a  = np.mean([m['grade_r'] for m in med_ap['camadas'].values() if m['dim'] > 1])
gr_ap2= np.mean([m['grade_r'] for m in med_ap_ph['camadas'].values() if m['dim'] > 1])
en_c  = np.mean([m['entropia'] for m in med_conv['camadas'].values() if m['dim'] > 1])
en_a  = np.mean([m['entropia'] for m in med_ap['camadas'].values() if m['dim'] > 1])
en_a2 = np.mean([m['entropia'] for m in med_ap_ph['camadas'].values() if m['dim'] > 1])
rk_c  = np.mean([m['rank_efetivo'] for m in med_conv['camadas'].values() if m['dim'] > 1])
rk_a  = np.mean([m['rank_efetivo'] for m in med_ap['camadas'].values() if m['dim'] > 1])
rk_a2 = np.mean([m['rank_efetivo'] for m in med_ap_ph['camadas'].values() if m['dim'] > 1])
est_c = estabilidade(hist_conv)
est_a = estabilidade(hist_ap)
est_a2= estabilidade(hist_ap_ph)

print(f"\n  ── Resumo Comparativo Fase 0 vs Fase 2 ─────────────────────────")
print(f"  {'Instrumento':<28} {'Conv':>10}  {'AP':>10}  {'AP+Phantom':>12}")
print(f"  {'─'*28} {'─'*10}  {'─'*10}  {'─'*12}")
print(f"  {'Grade R médio':<28} {gr_c:>10.4f}  {gr_a:>10.4f}  {gr_ap2:>12.4f}")
print(f"  {'Entropia média':<28} {en_c:>10.4f}  {en_a:>10.4f}  {en_a2:>12.4f}")
print(f"  {'Rank efetivo médio':<28} {rk_c:>10.2f}  {rk_a:>10.2f}  {rk_a2:>12.2f}")
print(f"  {'Estabilidade':<28} {est_c:>10.6f}  {est_a:>10.6f}  {est_a2:>12.6f}")
print(f"  {'Loss validação':<28} {med_conv['loss']:>10.5f}  {med_ap['loss']:>10.5f}  {med_ap_ph['loss']:>12.5f}")
print(f"  {'ep 0 va (init)':<28} {hist_conv[0]['va']:>10.5f}  {hist_ap[0]['va']:>10.5f}  {hist_ap_ph[0]['va']:>12.5f}")

print("\n─── Scanner Fase 2 — AP com Phantom ────────────────────────")
ang_ap_ph, d_ap_ph = scanner_topografico(rede_ap_ph, "AP + Phantom",
                                          colorscale=COLORSCALE_AP)

# ── Quadro final ──────────────────────────────────────────────────────────────
print("\n" + "═"*62)
print("  QUADRO FINAL — Comparativo Scanner θ_R")
print("═"*62)
print(f"  {'Rede':<22} {'θ crista':>10}  {'Δ θ_R':>8}  {'Grade R?':>12}")
print(f"  {'─'*22} {'─'*10}  {'─'*8}  {'─'*12}")
for nome_r, ang, dlt in [
    ("Conv (sem Phantom)",   ang_conv,  d_conv),
    ("AP  (sem Phantom)",    ang_ap,    d_ap),
    ("AP  (com Phantom)",    ang_ap_ph, d_ap_ph),
]:
    if dlt < 5.0:
        g = "PRESENTE"
    elif dlt < 15.0:
        g = "Parcial"
    else:
        g = "—"
    print(f"  {nome_r:<22} {ang:>10.2f}°  {dlt:>7.2f}°  {g:>12}")
print(f"  {'θ_R referência':<22} {np.degrees(THETA_R):>10.2f}°")
print(f"\n  φ={PHI}  α={ALPHA:.8f}  Phantom amp={PHANTOM_AMP:.4f}")
