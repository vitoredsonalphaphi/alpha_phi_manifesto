"""
AlphaPhiNet_CampoFI_COLAB.py
Vitor Edson Delavi · Florianópolis · 01 de outubro de 2026

Campo FI (φ-field) — primeira implantação do campo Riemanniano φ na Rede AP.

O que muda em relação ao baseline euclidiano (AlphaPhiNet_AnguloPhantom):
  - Gradientes precondicionados pela métrica G_φ antes de cada step do otimizador
  - G_φ diagonal com escala 1/φ^(i+1) por camada Fibonacci
  - O espaço de parâmetros deixa de ser euclidiano — distâncias medidas em φ

Hipótese:
  Com o campo FI instalado, a crista da Rede AP deve se aproximar de θ_R (63.43°)
  sem necessitar de Phantom — porque o gradiente já se move no espaço icosaédrico.

Comparação central:
  1. Conv_eucl       — baseline euclidiano
  2. AP_eucl         — AP sem campo FI (como nos experimentos E11–E13)
  3. AP_FI           — AP com campo φ instalado
  4. AP_FI + Ph−45°  — campo φ + melhor Phantom (E12: +48.30°, Δ=15.14°)

Métricas coletadas:
  - Ângulo da crista (scanner topográfico 3D)
  - Grade R médio
  - Loss de validação
  - Alinhamento com bases icosaédricas φ (projetar_fi)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
import plotly.graph_objects as go

# ── Constantes ─────────────────────────────────────────────────────────────────

PHI         = 1.6180339887
ALPHA       = 1 / 137.035999
SEAL        = 1 / PHI
THETA_R     = np.arctan(2)        # 63.43° — ângulo de referência AP
PHANTOM_AMP = 1.0 / PHI**3        # ≈ 0.236
SEED        = 42
DIMS_AP     = [55, 34, 21, 13, 8]

# Vetores de base do icosaedro φ (normalizados)
_b = np.array([
    [1.0,     1/PHI,   1/PHI**2],
    [1/PHI,   1/PHI**2, 1.0    ],
    [1/PHI**2, 1.0,    1/PHI   ]
], dtype=np.float64)
PHI_BASES = _b / np.linalg.norm(_b, axis=1, keepdims=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"φ={PHI}  α={ALPHA:.8f}  SEAL={SEAL:.6f}")
print(f"θ_R={np.degrees(THETA_R):.2f}°  Phantom amp={PHANTOM_AMP:.4f}")
print(f"Bases icosaédricas φ (normalizadas):")
for i, b in enumerate(PHI_BASES):
    print(f"  v{i+1} = [{b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}]")
print("─" * 62)

# ── Dados ──────────────────────────────────────────────────────────────────────

def gerar_dados(n=800, d_in=61, seed=SEED):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        v = np.zeros(d_in, dtype=np.float32)
        for k in range(8):
            idx = min(int(k * PHI * 7), d_in - 1)
            v[idx] += float(1.0 / PHI**k)
        nl = rng.uniform(0.1, 1.5)
        v += rng.normal(0, nl, d_in).astype(np.float32)
        v /= np.max(np.abs(v)) + 1e-8
        X.append(v)
        y.append(float(1.0 / (1.0 + nl)))
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32))

X, y = gerar_dados()
ld_tr = DataLoader(TensorDataset(X[:640], y[:640]), batch_size=32, shuffle=True)
ld_va = DataLoader(TensorDataset(X[640:], y[640:]), batch_size=32, shuffle=False)
print(f"Dados: 640 treino · 160 validação · d_in=61\n")

# ── Instrumentos ───────────────────────────────────────────────────────────────

def grade_r(act):
    if act.shape[-1] <= 1: return float('nan')
    mag  = torch.abs(act) + 1e-10
    norm = torch.clamp(mag / mag.sum(dim=-1, keepdim=True), 1e-10, 1.0)
    H    = -(norm * torch.log(norm)).sum(dim=-1)
    return float((1.0 - H / np.log(act.shape[-1])).mean().item())

def projetar_fi(act):
    """Alinhamento médio das ativações com os 3 vetores de base φ."""
    a = act.numpy() if isinstance(act, torch.Tensor) else act
    a = a.mean(axis=0)
    if len(a) < 3:
        return np.zeros(3)
    a3 = a[:3] / (np.linalg.norm(a[:3]) + 1e-8)
    return PHI_BASES @ a3  # (3,) — alinhamento com v1, v2, v3

def medir(model, loader):
    model.eval()
    ativ, loss_t, nb = {}, 0.0, 0
    hooks = []
    def _h(n):
        def h(m, i, o): ativ.setdefault(n, []).append(o.detach().cpu())
        return h
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_h(n)))
    crit = nn.MSELoss()
    with torch.no_grad():
        for xb, yb in loader:
            loss_t += crit(model(xb).squeeze(-1), yb).item()
            nb += 1
    for h in hooks: h.remove()
    res = {'loss': loss_t / max(nb, 1), 'gr': {}, 'fi': {}}
    for n, ats in ativ.items():
        a = torch.cat(ats, dim=0)
        res['gr'][n] = {'grade_r': grade_r(a), 'dim': a.shape[-1]}
        res['fi'][n] = projetar_fi(a)
    return res

def gr_medio(med):
    vals = [v['grade_r'] for v in med['gr'].values()
            if v['dim'] > 1 and not np.isnan(v['grade_r'])]
    return float(np.mean(vals)) if vals else float('nan')

def fi_medio(med):
    """Alinhamento φ médio entre todas as camadas."""
    projs = [v for v in med['fi'].values() if v is not None and len(v) == 3]
    return float(np.mean([np.max(np.abs(p)) for p in projs])) if projs else float('nan')

# ── Redes ──────────────────────────────────────────────────────────────────────

class RedeConvencional(nn.Module):
    def __init__(self, d_in=61, dims=[64, 32, 16, 8]):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = d_in
        for d in dims:
            self.layers.append(nn.Linear(prev, d))
            prev = d
        self.head = nn.Linear(prev, 1)
        for l in self.layers:
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)
    def forward(self, x):
        for l in self.layers: x = F.relu(l(x))
        return self.head(x)

class RedeAP(nn.Module):
    def __init__(self, d_in=61, phantom_deg=None):
        super().__init__()
        self.proj   = nn.Linear(d_in, DIMS_AP[0])
        self.layers = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for i in range(len(DIMS_AP) - 1):
            self.layers.append(nn.Linear(DIMS_AP[i], DIMS_AP[i+1]))
            self.norms.append(nn.LayerNorm(DIMS_AP[i+1]))
        self.head = nn.Linear(DIMS_AP[-1], 1)
        self._init_phi()
        if phantom_deg is not None:
            self._phantom(phantom_deg)
    def _init_phi(self):
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0 / PHI)
        nn.init.zeros_(self.proj.bias)
        for i, l in enumerate(self.layers):
            nn.init.xavier_uniform_(l.weight, gain=1.0 / PHI**(i + 1))
            nn.init.zeros_(l.bias)
    def _phantom(self, angulo_deg):
        ang_rad = np.radians(angulo_deg)
        tan_val = np.tan(ang_rad)
        if not np.isfinite(tan_val) or abs(tan_val) > 1e4:
            tan_val = np.sign(tan_val if np.isfinite(tan_val) else 1) * 1e4
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                W = layer.weight.data
                d_out, d_in = W.shape
                sigma = max(1, int(d_in / PHI**(i + 2)))
                indices = np.arange(d_in)
                for row in range(d_out):
                    centro = (row / max(d_out - 1, 1)) * tan_val
                    idx_c  = int((centro % 1.0) * d_in)
                    pulso  = np.exp(-0.5 * ((indices - idx_c) / sigma)**2)
                    pulso /= pulso.max() + 1e-10
                    W[row] += PHANTOM_AMP * torch.tensor(pulso.astype(np.float32))
    def forward(self, x):
        x = F.silu(self.proj(x))
        for l, n in zip(self.layers, self.norms):
            x = F.silu(l(x)); x = n(x)
        return self.head(x)

# ── Campo FI — Precondicionador Icosaédrico φ ──────────────────────────────────

class PhiPrecondicionador:
    """
    Aplica a métrica icosaédrica φ aos gradientes antes de cada step.

    Cada neurônio (linha da matriz de pesos) é atribuído a uma das três
    direções icosaédricas pelo índice modular:
      índice % 3 == 0 → v₁ = coerência   → escala 1.0     (movimento livre)
      índice % 3 == 1 → v₂ = entropia    → escala SEAL     (movimento moderado)
      índice % 3 == 2 → v₃ = atrator     → escala SEAL²    (movimento restrito)

    Resultado: neurônios de coerência adaptam livremente; neurônios de atrator
    são os mais resistentes à mudança — a estrutura estável do campo.

    Diferença da versão anterior (errada): antes escalava por CAMADA (profundidade
    Fibonacci), que apenas diminuía o learning rate das camadas profundas sem
    implementar a geometria icosaédrica. Agora escala por DIREÇÃO (tipo de neurônio).
    """
    def __init__(self):
        self.escalas = torch.tensor([1.0, SEAL, SEAL**2])  # v₁, v₂, v₃

    def aplicar(self, model):
        with torch.no_grad():
            for module in model.modules():
                if not isinstance(module, nn.Linear):
                    continue
                if module.weight.grad is None:
                    continue
                G = module.weight.grad  # (d_out, d_in)
                d_out = G.shape[0]
                # Aplica escala icosaédrica por tipo de neurônio de saída (linha)
                for tipo in range(3):
                    idx = torch.arange(tipo, d_out, 3)
                    if len(idx):
                        G[idx, :] *= self.escalas[tipo].item()
                # Bias: mesma escala do tipo de neurônio
                if module.bias is not None and module.bias.grad is not None:
                    tipos = torch.arange(d_out) % 3
                    module.bias.grad.mul_(self.escalas[tipos])

# ── Treinamento ────────────────────────────────────────────────────────────────

def treinar(model, n_epochs=120, lr=1e-3, nome="", campo_fi=False):
    precond = PhiPrecondicionador() if campo_fi else None
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1/PHI**3)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=1/PHI, patience=8)
    crit = nn.MSELoss()
    hist = []
    for ep in range(n_epochs):
        model.train()
        tl = 0.0
        for xb, yb in ld_tr:
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward()
            if precond is not None:
                precond.aplicar(model)     # ← campo FI icosaédrico ANTES do clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), PHI)
            opt.step()
            tl += loss.item()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in ld_va:
                vl += crit(model(xb).squeeze(-1), yb).item()
        tl /= len(ld_tr); vl /= len(ld_va)
        sch.step(tl)
        hist.append({'ep': ep, 'tr': tl, 'va': vl})
        if ep % 30 == 0:
            fi_tag = "[FI]" if campo_fi else "    "
            print(f"  {fi_tag} [{nome}] ep {ep:3d}  tr={tl:.5f}  va={vl:.5f}"
                  f"  lr={opt.param_groups[0]['lr']:.2e}")
    return hist

# ── Scanner Topográfico 3D ─────────────────────────────────────────────────────

COLORSCALE = {
    'conv': [[0,'rgb(0,10,0)'],[0.4,'rgb(0,100,40)'],[1,'rgb(180,255,180)']],
    'ap':   [[0,'rgb(0,0,20)'],[0.4,'rgb(80,50,0)'],[1,'rgb(255,248,100)']],
    'fi':   [[0,'rgb(0,0,30)'],[0.4,'rgb(60,0,100)'],[1,'rgb(200,100,255)']],
}

def scanner(model, nome_rede, cor='ap', n_ent=200, n_grid=55, mostrar=True):
    model.eval()
    rng_s = np.random.default_rng(137)
    ents  = rng_s.standard_normal((n_ent, 61)).astype(np.float32)
    ents -= ents.mean(axis=1, keepdims=True)
    ents /= ents.std(axis=1, keepdims=True) + 1e-8

    ativ_s = {}
    hooks  = []
    def _h(n):
        def h(m, i, o): ativ_s.setdefault(n, []).append(o.detach().cpu().numpy())
        return h
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_h(n)))
    with torch.no_grad():
        for e in ents:
            _ = model(torch.tensor(e).unsqueeze(0))
    for h in hooks: h.remove()

    nomes_c = [n for n in ativ_s
               if np.concatenate(ativ_s[n], axis=0).shape[-1] > 1]
    n_layers = len(nomes_c)
    mapa = np.zeros((n_layers, n_grid))
    for lv, nc in enumerate(nomes_c):
        a   = np.abs(np.concatenate(ativ_s[nc], axis=0)).mean(axis=0)
        x0  = np.linspace(0, 1, len(a))
        xg  = np.linspace(0, 1, n_grid)
        mapa[lv] = interp1d(x0, a, kind='linear')(xg)

    mapa_n = np.log1p(mapa * 100)
    picos  = np.array([np.argmax(mapa_n[lv]) / (n_grid - 1) for lv in range(n_layers)])
    picos_r = np.linspace(0, 1, n_layers)
    angulo  = np.degrees(np.arctan(np.polyfit(picos, picos_r, 1)[0])) \
              if picos.std() > 1e-6 else 90.0
    delta   = abs(angulo - np.degrees(THETA_R))

    tag = "Grade R PRESENTE" if delta < 5 else ("Parcial" if delta < 15 else "—")
    print(f"  [{nome_rede}]  θ={angulo:.2f}°  Δθ_R={delta:.2f}°  {tag}")

    if mostrar:
        X_g, R_g = np.meshgrid(np.linspace(0,1,n_grid), np.linspace(0,1,n_layers))
        fig = go.Figure()
        fig.add_trace(go.Surface(x=X_g, y=R_g, z=mapa_n,
            colorscale=COLORSCALE[cor], showscale=False,
            lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5)))
        dx = 0.3; rc = 0.5; dy = dx * np.tan(THETA_R)
        fig.add_trace(go.Scatter3d(x=[.5-dx,.5+dx], y=[max(0,rc-dy),min(1,rc+dy)],
            z=[mapa_n.max()*1.1]*2, mode='lines', line=dict(color='lime',width=6),
            name=f'θ_R=63.43°'))
        z_p = [mapa_n[lv, int(p*(n_grid-1))] for lv,p in enumerate(picos)]
        fig.add_trace(go.Scatter3d(x=picos, y=picos_r, z=z_p,
            mode='lines+markers', line=dict(color='cyan',width=4),
            marker=dict(size=5,color='cyan'), name=f'Crista {angulo:.1f}°'))
        fig.update_layout(
            title=dict(text=f'{nome_rede}<br><sup>θ={angulo:.2f}°  Δθ_R={delta:.2f}°</sup>',
                       font=dict(size=13)),
            scene=dict(xaxis_title='Neurônio', yaxis_title='Profundidade r',
                       zaxis_title='log|ativ|',
                       camera=dict(eye=dict(x=1.5,y=-1.8,z=1.2))),
            width=720, height=520,
            paper_bgcolor='rgb(8,8,18)', font=dict(color='white'))
        fig.show()
    return angulo, delta

# ══════════════════════════════════════════════════════════════════════════════
#  TREINO — 4 redes: Conv_eucl · AP_eucl · AP_FI · AP_FI+Ph−45°
# ══════════════════════════════════════════════════════════════════════════════

print("═"*62)
print("  Treinando 4 redes — 120 épocas cada")
print("  Campo FI = precondicionador Riemanniano φ nos gradientes")
print("═"*62)

print("\n[1/4] Conv euclidiana (baseline)")
rede_conv = RedeConvencional()
hist_conv = treinar(rede_conv, nome="Conv     ", campo_fi=False)

print("\n[2/4] AP euclidiana (sem campo FI — referência E11–E13)")
rede_ap = RedeAP(phantom_deg=None)
hist_ap  = treinar(rede_ap, nome="AP_eucl  ", campo_fi=False)

print("\n[3/4] AP com campo FI instalado")
rede_fi = RedeAP(phantom_deg=None)
hist_fi  = treinar(rede_fi, nome="AP_FI    ", campo_fi=True)

print("\n[4/4] AP com campo FI + Phantom −45° (melhor Phantom de E12)")
rede_fi_m45 = RedeAP(phantom_deg=-45.0)
hist_fi_m45 = treinar(rede_fi_m45, nome="AP_FI−45°", campo_fi=True)

# ── Métricas ──────────────────────────────────────────────────────────────────

redes = [
    (rede_conv,    hist_conv,    "Conv     ", 'conv', False),
    (rede_ap,      hist_ap,      "AP_eucl  ", 'ap',   False),
    (rede_fi,      hist_fi,      "AP_FI    ", 'fi',   True ),
    (rede_fi_m45,  hist_fi_m45,  "AP_FI−45°", 'fi',  True ),
]
meds = [(nome, medir(r, ld_va), hist, fi_flag)
        for r, hist, nome, _, fi_flag in redes]

print("\n" + "═"*62)
print("  Grade R · Alinhamento φ · Loss por rede")
print("═"*62)
print(f"  {'Rede':<12} {'Campo':>6}  {'Grade R':>8}  {'Alin. φ':>8}  {'Loss va':>8}")
print(f"  {'─'*12} {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
for nome, med, hist, fi_flag in meds:
    gr  = gr_medio(med)
    fi  = fi_medio(med)
    tag = "[FI]" if fi_flag else "    "
    print(f"  {nome:<12} {tag}  {gr:>8.4f}  {fi:>8.4f}  {med['loss']:>8.5f}")

# ── Scanner Topográfico 3D ─────────────────────────────────────────────────────

print("\n" + "═"*62)
print("  Scanner Topográfico 3D — 4 redes")
print(f"  θ_R referência = {np.degrees(THETA_R):.2f}°")
print("═"*62)

resultados_scan = []
for r, hist, nome, cor, fi_flag in redes:
    ang, dlt = scanner(r, nome, cor=cor, mostrar=True)
    resultados_scan.append((nome, ang, dlt, fi_flag))

# ── Quadro final ──────────────────────────────────────────────────────────────

print("\n" + "═"*62)
print("  QUADRO FINAL — Ângulo da Crista vs θ_R")
print("═"*62)
print(f"  {'Rede':<12} {'Campo':>6}  {'θ crista':>10}  {'Δ θ_R':>8}  {'Status':>16}")
print(f"  {'─'*12} {'─'*6}  {'─'*10}  {'─'*8}  {'─'*16}")
for nome, ang, dlt, fi_flag in resultados_scan:
    if dlt < 5:    status = "Grade R PRESENTE"
    elif dlt < 15: status = "Parcial"
    else:          status = "—"
    tag = "[FI]" if fi_flag else "    "
    print(f"  {nome:<12} {tag}  {ang:>10.2f}°  {dlt:>7.2f}°  {status:>16}")
print(f"  {'θ_R ref':<12} {'':>6}  {np.degrees(THETA_R):>10.2f}°")

# Diferença FI vs euclidiana
ap_ang  = next(a for n,a,d,f in resultados_scan if "AP_eucl" in n)
fi_ang  = next(a for n,a,d,f in resultados_scan if "AP_FI  " in n or n.strip()=="AP_FI")
delta_fi = fi_ang - ap_ang
print(f"\n  Δ crista (AP_FI − AP_eucl) = {delta_fi:+.2f}°")
print(f"  {'↑ campo FI aproximou θ_R' if delta_fi > 0 else '↓ campo FI afastou θ_R'}")

print(f"\n  φ={PHI}  α={ALPHA:.8f}  SEAL={SEAL:.6f}")
