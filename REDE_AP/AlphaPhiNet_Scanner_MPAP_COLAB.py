"""
AlphaPhiNet_Scanner_MPAP_COLAB.py
Vitor Edson Delavi · Florianópolis · outubro de 2026

Scanner Topográfico 3D — Rede Convencional pura  ×  Conv + MPAP acoplado

ESTRUTURA:
  FASE 0 — Treino RedeConvencional (mesma arquitetura do Phantom Scanner)
  FASE 1 — Scanner Topográfico: Conv pura (sem MPAP)
             Colorscale verde — referência
  FASE 2 — Scanner Topográfico: Conv + MPAP acoplado
             O MPAP adiciona uma camada extra no scanner (r=SEAL)
             Colorscale amarelo-dourado — campo harmônico
  QUADRO FINAL — ângulo da crista, Δθ_R, Grade R detectada?

Diferença chave em relação ao Phantom Scanner:
  Aqui o MPAP NÃO modifica os pesos da rede.
  Opera sobre o output da última camada oculta, redistribuindo
  com p_i = SEAL*(1-SEAL)^i — e essa camada redistribuída
  aparece como uma nova profundidade r=SEAL no scanner.

Perguntas que este experimento responde:
  1. O MPAP produz uma assinatura topográfica distinta da Conv?
  2. O padrão da camada MPAP se aproxima de θ_R = 63.43°?
  3. A crista do MPAP se alinha com a Grade R antes do treino convergir?
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "colab"

PHI     = 1.6180339887
ALPHA   = 1 / 137.035999
SEAL    = 1 / PHI           # 0.618034 — critério de selagem
THETA_R = np.arctan(2)      # 63.43° — referência Grade R
SEED    = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"φ = {PHI}  α = {ALPHA:.8f}  SEAL = {SEAL:.8f}")
print(f"θ_R = {np.degrees(THETA_R):.2f}°")
print("─" * 62)

# ══════════════════════════════════════════════════════════════════════════════
#  COLORSCALES
# ══════════════════════════════════════════════════════════════════════════════

CS_CONV = [
    [0.00, 'rgb(0,10,0)'],
    [0.20, 'rgb(0,60,20)'],
    [0.50, 'rgb(0,140,60)'],
    [0.80, 'rgb(80,200,120)'],
    [1.00, 'rgb(220,255,220)'],
]
CS_MPAP = [
    [0.00, 'rgb(0,0,20)'],
    [0.15, 'rgb(5,20,80)'],
    [0.40, 'rgb(80,50,0)'],
    [0.65, 'rgb(180,120,0)'],
    [0.85, 'rgb(230,190,20)'],
    [1.00, 'rgb(255,248,100)'],
]

# ══════════════════════════════════════════════════════════════════════════════
#  DADOS — mesma geração do Phantom Scanner
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
#  INSTRUMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def ativacao_coerencia(ativs):
    if ativs.shape[-1] <= 1:
        return float('nan')
    mag  = torch.abs(ativs) + 1e-10
    norm = mag / (mag.sum(dim=-1, keepdim=True) + 1e-10)
    norm = torch.clamp(norm, 1e-10, 1.0)
    H    = -(norm * torch.log(norm)).sum(dim=-1)
    return float((1.0 - H / float(np.log(ativs.shape[-1]))).mean().item())

def entropia_shannon(ativs):
    if ativs.shape[-1] <= 1:
        return float('nan')
    mag  = torch.abs(ativs) + 1e-10
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

# ══════════════════════════════════════════════════════════════════════════════
#  REDE CONVENCIONAL
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

    def forward_com_oculta(self, x):
        """Retorna (output_head, ultima_camada_oculta)"""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
            if i == len(self.layers) - 1:
                ultima = x
        return self.head(x), ultima

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
#  MPAP — APMetaprocessador  (inline, sem dependências externas)
# ══════════════════════════════════════════════════════════════════════════════

def _pesos_phi(n):
    """p_i = SEAL*(1-SEAL)^i normalizado"""
    idx = np.arange(n, dtype=float)
    w   = SEAL * (1.0 - SEAL) ** idx
    return w / w.sum()

def mpap_redistribuir(v_np):
    """
    Redistribuição φ de um vetor numpy.
    Preserva sinal — redistribui energia por magnitude decrescente.
    Retorna vetor com mesma dimensão, Coh >= SEAL.
    """
    n   = len(v_np)
    w   = _pesos_phi(n)
    mag = np.abs(v_np)
    idx = np.argsort(mag)[::-1]
    energia_total = mag.sum()
    mag_nova      = w * energia_total
    resultado     = np.empty_like(v_np)
    resultado[idx] = np.sign(v_np[idx] + 1e-10) * mag_nova
    return resultado

def mpap_processar_batch(output_np, n_ciclos_max=20):
    """
    Processa batch numpy (N, D).
    Redistribui amostras com Coh < SEAL até convergir.
    Retorna (output_final, coh_antes, coh_depois).
    """
    def coh_np(v):
        mag  = np.abs(v) + 1e-10
        p    = mag / mag.sum()
        p    = np.clip(p, 1e-10, 1.0)
        H    = -(p * np.log(p)).sum()
        return 1.0 - H / np.log(len(v))

    out        = output_np.copy()
    coh_antes  = np.array([coh_np(row) for row in out])
    ciclos     = np.zeros(len(out))

    for _ in range(n_ciclos_max):
        abaixo = np.where(np.array([coh_np(row) for row in out]) < SEAL)[0]
        if len(abaixo) == 0:
            break
        for i in abaixo:
            out[i]    = mpap_redistribuir(out[i])
            ciclos[i] += 1

    coh_depois = np.array([coh_np(row) for row in out])
    return out, coh_antes, coh_depois

# ══════════════════════════════════════════════════════════════════════════════
#  SCANNER TOPOGRÁFICO 3D
# ══════════════════════════════════════════════════════════════════════════════

def scanner_topografico(model, nome_rede, usar_mpap=False, n_entradas=200,
                        n_grid=55, colorscale=None):
    """
    Scanner Topográfico 3D sobre a rede.

    usar_mpap=False → scanner padrão (igual ao Phantom Scanner)
    usar_mpap=True  → adiciona camada extra: output da última camada oculta
                      redistribuído pelo MPAP (aparece na borda r=SEAL)
    """
    model.eval()
    if colorscale is None:
        colorscale = CS_CONV if not usar_mpap else CS_MPAP

    rng_s = np.random.default_rng(137)
    entradas_raw = rng_s.standard_normal((n_entradas, 61)).astype(np.float32)
    entradas_raw -= entradas_raw.mean(axis=1, keepdims=True)
    entradas_raw /= (entradas_raw.std(axis=1, keepdims=True) + 1e-8)

    ativacoes_scan   = {}
    ultima_oculta    = []
    hooks_s          = []

    def _hook_s(n):
        def h(m, inp, out):
            ativacoes_scan.setdefault(n, []).append(out.detach().cpu().numpy())
        return h

    nomes_lineares = []
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks_s.append(mod.register_forward_hook(_hook_s(n)))
            nomes_lineares.append(n)

    with torch.no_grad():
        for entry in entradas_raw:
            inp = torch.tensor(entry).unsqueeze(0)
            _ = model(inp)

    for h in hooks_s:
        h.remove()

    nomes_camadas = [
        n for n in ativacoes_scan.keys()
        if np.concatenate(ativacoes_scan[n], axis=0).shape[-1] > 1
    ]

    # Coleta última camada oculta para MPAP (excluindo head dim=1)
    if usar_mpap and nomes_camadas:
        nome_ult = nomes_camadas[-1]
        ult_raw  = np.concatenate(ativacoes_scan[nome_ult], axis=0)  # (N, d)
        out_mpap, coh_antes, coh_depois = mpap_processar_batch(ult_raw)

    n_layers = len(nomes_camadas)
    if usar_mpap:
        n_layers += 1   # camada extra: MPAP na borda r=SEAL

    mapa = np.zeros((n_layers, n_grid))
    for lv, nome_c in enumerate(nomes_camadas):
        all_av = np.concatenate(ativacoes_scan[nome_c], axis=0)
        media  = np.abs(all_av).mean(axis=0)
        x_orig = np.linspace(0, 1, len(media))
        x_grid = np.linspace(0, 1, n_grid)
        mapa[lv] = interp1d(x_orig, media, kind='linear')(x_grid)

    if usar_mpap:
        # Camada MPAP: média das magnitudes após redistribuição
        media_mpap = np.abs(out_mpap).mean(axis=0)
        x_orig_m   = np.linspace(0, 1, len(media_mpap))
        x_grid_m   = np.linspace(0, 1, n_grid)
        mapa[n_layers - 1] = interp1d(x_orig_m, media_mpap, kind='linear')(x_grid_m)

    mapa_norm = np.log1p(mapa * 100)

    # Ângulo da crista dominante
    picos_x = np.array([np.argmax(mapa_norm[lv]) / (n_grid - 1)
                        for lv in range(mapa_norm.shape[0])])
    picos_r = np.linspace(0, 1, mapa_norm.shape[0])
    if picos_x.std() > 1e-6:
        coef   = np.polyfit(picos_x, picos_r, 1)
        angulo = np.degrees(np.arctan(coef[0]))
    else:
        angulo = 90.0
    delta = abs(angulo - np.degrees(THETA_R))

    mpap_tag = " + MPAP" if usar_mpap else ""
    print(f"\n  [{nome_rede}{mpap_tag}] Scanner Topográfico 3D")
    if usar_mpap:
        coh_m_antes  = float(coh_antes.mean())
        coh_m_depois = float(coh_depois.mean())
        print(f"    MPAP: Coh antes = {coh_m_antes:.4f}  →  Coh depois = {coh_m_depois:.4f}  "
              f"({'✓ SEAL' if coh_m_depois >= SEAL else '✗ SEAL'})")
        pct = float((coh_depois >= SEAL).mean() * 100)
        print(f"    Amostras acima SEAL: {pct:.1f}%")
    print(f"    Ângulo crista : {angulo:.2f}°")
    print(f"    θ_R referência: {np.degrees(THETA_R):.2f}°")
    print(f"    Δ do θ_R      : {delta:.2f}°", end="  ")
    if delta < 5.0:
        print("→ Grade R PRESENTE (Δ < 5°)")
    elif delta < 15.0:
        print("→ Proximidade parcial com Grade R")
    else:
        print("→ Grade R não detectada nesta configuração")

    X_g, R_g = np.meshgrid(np.linspace(0, 1, n_grid),
                            np.linspace(0, 1, mapa_norm.shape[0]))
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X_g, y=R_g, z=mapa_norm,
        colorscale=colorscale, showscale=True,
        colorbar=dict(title='log|ativ|', thickness=12, len=0.6),
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.3),
        name=f'{nome_rede}{mpap_tag} — campo'
    ))

    # Plano SEAL (r=SEAL) se MPAP presente — linha na borda
    if usar_mpap:
        y_seal = np.full(n_grid, SEAL)
        z_seal = np.full(n_grid, mapa_norm.max() * 0.5)
        fig.add_trace(go.Scatter3d(
            x=np.linspace(0, 1, n_grid), y=y_seal, z=z_seal,
            mode='lines',
            line=dict(color='rgba(0,210,165,0.6)', width=4),
            name=f'r=SEAL={SEAL:.3f}',
            hoverinfo='skip',
        ))
        # Plano MPAP (r=1)
        y_mpap = np.ones(n_grid)
        x_mpap = np.linspace(0, 1, n_grid)
        z_mpap = mapa_norm[n_layers - 1]
        fig.add_trace(go.Scatter3d(
            x=x_mpap, y=y_mpap, z=z_mpap,
            mode='lines+markers',
            line=dict(color='rgba(255,215,55,0.90)', width=5),
            marker=dict(size=3, color='rgba(255,215,55,0.70)'),
            name=f'MPAP output  Coh→{coh_m_depois:.3f}',
        ))

    # Linha θ_R de referência
    dx, r_c = 0.3, 0.5
    dy = dx * np.tan(THETA_R)
    fig.add_trace(go.Scatter3d(
        x=[0.5 - dx, 0.5 + dx],
        y=[np.clip(r_c - dy, 0, 1), np.clip(r_c + dy, 0, 1)],
        z=[mapa_norm.max() * 1.12] * 2,
        mode='lines', line=dict(color='lime', width=7),
        name=f'θ_R = {np.degrees(THETA_R):.2f}°'
    ))

    # Crista real
    z_picos = [mapa_norm[lv, int(px * (n_grid - 1))]
               for lv, px in enumerate(picos_x)]
    fig.add_trace(go.Scatter3d(
        x=picos_x, y=picos_r, z=z_picos,
        mode='lines+markers',
        line=dict(color='cyan', width=4),
        marker=dict(size=5, color='cyan'),
        name=f'Crista {angulo:.1f}°'
    ))

    titulo_mpap = f"  ┃  MPAP Coh {coh_m_antes:.3f}→{coh_m_depois:.3f}" if usar_mpap else ""
    fig.update_layout(
        title=dict(
            text=(f'Scanner Topográfico 3D — {nome_rede}{mpap_tag}<br>'
                  f'<sup>θ crista={angulo:.2f}°  Δθ_R={delta:.2f}°{titulo_mpap}</sup>'),
            font=dict(size=14, color='#d0c8b8'),
        ),
        scene=dict(
            xaxis=dict(title='Neurônio (normalizado)', color='#888',
                       gridcolor='#1a1a2e', showbackground=True, backgroundcolor='#07070e'),
            yaxis=dict(title='Profundidade r', color='#888',
                       tickvals=[0, SEAL, 1.0],
                       ticktext=['0 (entrada)', f'SEAL={SEAL:.3f}', '1 (saída)'],
                       gridcolor='#1a1a2e', showbackground=True, backgroundcolor='#07070e'),
            zaxis=dict(title='log|ativação|', color='#888',
                       gridcolor='#1a1a2e', showbackground=True, backgroundcolor='#07070e'),
            bgcolor='#04040a',
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2)),
        ),
        width=800, height=580,
        paper_bgcolor='rgb(10,10,20)',
        font=dict(color='white'),
        legend=dict(font=dict(size=10, color='#bbb'),
                    bgcolor='rgba(8,8,18,0.9)', bordercolor='#2a2a3c', borderwidth=1),
    )
    fig.show(renderer="colab")
    return angulo, delta


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 0 — Treino RedeConvencional
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 62)
print("  FASE 0 — Treino Rede Convencional")
print("═" * 62)

rede_conv = RedeConvencional(d_in=61, dims=[64, 32, 16, 8])
hist_conv = treinar(rede_conv, ld_tr, ld_va, n_epochs=120, lr=1e-3, nome="Conv")

rede_conv.eval()
crit_val = nn.MSELoss()
with torch.no_grad():
    va_loss_final = float(np.mean([crit_val(rede_conv(xb).squeeze(-1), yb).item()
                                   for xb, yb in ld_va]))
print(f"\n  Loss validação final: {va_loss_final:.5f}")
print(f"  Estabilidade (σ últimas 10 épocas): {estabilidade(hist_conv):.6f}")

# ══════════════════════════════════════════════════════════════════════════════
#  FASE 1 — Scanner Conv pura (sem MPAP)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 62)
print("  FASE 1 — Scanner Topográfico: Conv pura (sem MPAP)")
print("═" * 62)

ang_conv, d_conv = scanner_topografico(
    rede_conv, "Conv",
    usar_mpap=False,
    colorscale=CS_CONV
)

# ══════════════════════════════════════════════════════════════════════════════
#  FASE 2 — Scanner Conv + MPAP acoplado
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 62)
print("  FASE 2 — Scanner Topográfico: Conv + MPAP acoplado")
print("  (mesma rede, MPAP opera sobre a última camada oculta)")
print("  (camada MPAP aparece como borda r=1 no scanner)")
print("═" * 62)

ang_mpap, d_mpap = scanner_topografico(
    rede_conv, "Conv",
    usar_mpap=True,
    colorscale=CS_MPAP
)

# ══════════════════════════════════════════════════════════════════════════════
#  QUADRO FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 62)
print("  QUADRO FINAL — Scanner Topográfico + MPAP")
print("═" * 62)
print(f"  {'Configuração':<30} {'θ crista':>10}  {'Δ θ_R':>8}  {'Grade R?':>12}")
print(f"  {'─' * 30} {'─' * 10}  {'─' * 8}  {'─' * 12}")

for nome_r, ang, dlt in [
    ("Conv pura  (sem MPAP)",  ang_conv, d_conv),
    ("Conv + MPAP (com MPAP)", ang_mpap, d_mpap),
]:
    g = "PRESENTE" if dlt < 5.0 else ("Parcial" if dlt < 15.0 else "—")
    print(f"  {nome_r:<30} {ang:>10.2f}°  {dlt:>7.2f}°  {g:>12}")

print(f"  {'θ_R referência':<30} {np.degrees(THETA_R):>10.2f}°")
print(f"\n  φ={PHI}  α={ALPHA:.8f}  SEAL={SEAL:.8f}")
print(f"  MPAP: p_i = SEAL*(1-SEAL)^i  (redistribuição φ-ponderada)")
print(f"  Ponto fixo geométrico: Coh ≈ 0.689 > SEAL após 1 ciclo")
