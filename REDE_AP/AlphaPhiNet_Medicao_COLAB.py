"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AlphaPhi · Verificação dos Instrumentos de Medição da Rede AP             ║
║  Vitor Edson Delavi · Florianópolis · 2026                                 ║
║  REDE_AP/AlphaPhiNet_Medicao_COLAB.py                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Objetivo: verificar se os instrumentos de medição propostos para a Rede AP
funcionam e produzem leituras comparáveis com a rede convencional.

Instrumentos verificados:
  1. Grade R por ciclo (coerência φ nas ativações)
  2. Entropia Shannon por camada
  3. Rank efetivo das matrizes de peso
  4. Estabilidade do treino (variação da loss entre ciclos)

Tarefa de referência: regressão sintética — duas redes na mesma tarefa,
mesmas métricas, leituras lado a lado.
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CÉLULA ÚNICA — cole tudo no Colab e execute                    ║
# ╚══════════════════════════════════════════════════════════════════╝

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Constantes ────────────────────────────────────────────────────────────────
PHI    = 1.6180339887
ALPHA  = 1 / 137.035999
SEAL   = 1 / PHI
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"φ={PHI}  α={ALPHA:.8f}  SEAL={SEAL:.8f}")
print("─" * 60)

# ══════════════════════════════════════════════════════════════════════════════
#  TAREFA SINTÉTICA
#  Sinal de entrada: mistura de harmônicos φ + ruído
#  Target: coerência teórica do sinal (Grade R esperado)
#  Ambas as redes aprendem a prever o quão coerente é o sinal de entrada.
# ══════════════════════════════════════════════════════════════════════════════

def gerar_dados(n=800, d_in=61, seed=SEED):
    """
    Gera pares (entrada, target) para a tarefa de referência.

    Entrada: vetor de magnitudes espectrais em φ-bandas + ruído
    Target:  coerência teórica — quanto o sinal ressoa com φ
             (valor em [0,1], maior = mais ressonante)
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        # Sinal base: harmônicos φ com amplitudes decrescentes
        v = np.zeros(d_in, dtype=np.float32)
        for k in range(8):
            idx = min(int(k * PHI * 7), d_in - 1)
            v[idx] += float(1.0 / PHI**k)

        # Ruído gaussiano
        noise_level = rng.uniform(0.1, 1.5)
        v += rng.normal(0, noise_level, d_in).astype(np.float32)
        v = v / (np.max(np.abs(v)) + 1e-8)

        # Coerência teórica: razão sinal/total — diminui com ruído
        coh = float(1.0 / (1.0 + noise_level))
        X.append(v)
        y.append(coh)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y

X, y = gerar_dados(n=800, d_in=61)
X_tr, y_tr = X[:640], y[:640]
X_va, y_va = X[640:], y[640:]

ds_tr = TensorDataset(X_tr, y_tr)
ds_va = TensorDataset(X_va, y_va)
ld_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
ld_va = DataLoader(ds_va, batch_size=32, shuffle=False)

print(f"Dados: {len(X_tr)} treino · {len(X_va)} validação · d_in={X.shape[1]}")
print("─" * 60)

# ══════════════════════════════════════════════════════════════════════════════
#  INSTRUMENTOS DE MEDIÇÃO AP
# ══════════════════════════════════════════════════════════════════════════════

def ativacao_coerencia(activations: torch.Tensor) -> float:
    """
    Coerência de ativação via Sépstro: 1 - H_normalizada das magnitudes absolutas.
    Mede concentração energética, não geometria Grade R.
    """
    mag  = torch.abs(activations) + 1e-10
    norm = mag / (mag.sum(dim=-1, keepdim=True) + 1e-10)
    norm = torch.clamp(norm, 1e-10, 1.0)
    H    = -(norm * torch.log(norm)).sum(dim=-1)
    H_max = float(np.log(activations.shape[-1]))
    coh  = (1.0 - H / H_max).mean().item()
    return float(coh)

def entropia_shannon(activations: torch.Tensor) -> float:
    """Entropia Shannon normalizada das ativações."""
    mag  = torch.abs(activations) + 1e-10
    norm = mag / (mag.sum(dim=-1, keepdim=True) + 1e-10)
    norm = torch.clamp(norm, 1e-10, 1.0)
    H    = -(norm * torch.log(norm)).sum(dim=-1).mean().item()
    return float(H)

def rank_efetivo(weight: torch.Tensor) -> float:
    """
    Rank efetivo da matriz de pesos via entropia dos valores singulares.
    Mede quantas dimensões do espaço estão realmente sendo usadas.
    Alta compressão φ → rank efetivo menor = espaço bem organizado.
    """
    with torch.no_grad():
        s = torch.linalg.svdvals(weight.float())
        s = s / (s.sum() + 1e-10)
        s = torch.clamp(s, 1e-10, 1.0)
        H = -(s * torch.log(s)).sum().item()
        return float(np.exp(H))

def medir_rede(model, loader, device='cpu'):
    """
    Roda a rede em modo eval sobre o loader e coleta todos os instrumentos.
    Retorna: dict com Grade R por camada, entropia, rank efetivo, loss média.
    """
    model.eval()
    ativacoes  = {}   # camada → lista de tensores
    pesos      = {}   # camada → weight tensor
    loss_total = 0.0
    n_batches  = 0

    hooks = []

    def _fazer_hook(nome):
        def hook(module, inp, out):
            if nome not in ativacoes:
                ativacoes[nome] = []
            ativacoes[nome].append(out.detach().cpu())
        return hook

    # Registra hooks em todas as camadas lineares
    for nome, modulo in model.named_modules():
        if isinstance(modulo, nn.Linear):
            hooks.append(modulo.register_forward_hook(_fazer_hook(nome)))
            pesos[nome] = modulo.weight.detach().cpu()

    criterion = nn.MSELoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.squeeze(-1)
            loss_total += criterion(pred, yb).item()
            n_batches  += 1

    for h in hooks:
        h.remove()

    # Concatena ativações por camada e computa métricas
    resultado = {'loss': loss_total / max(n_batches, 1), 'camadas': {}}

    for nome, ats in ativacoes.items():
        all_at = torch.cat(ats, dim=0)
        gr = ativacao_coerencia(all_at)
        en = entropia_shannon(all_at)
        rk = rank_efetivo(pesos[nome])
        resultado['camadas'][nome] = {
            'ativ_coh':      gr,
            'entropia':      en,
            'rank_efetivo':  rk,
            'dim':           all_at.shape[-1],
        }

    return resultado

# ══════════════════════════════════════════════════════════════════════════════
#  REDE CONVENCIONAL (baseline euclidiano)
# ══════════════════════════════════════════════════════════════════════════════

class RedeConvencional(nn.Module):
    """
    Rede padrão: Linear → ReLU → Linear → ReLU → ... → Linear
    Dimensões arbitrárias, sem φ-estrutura.
    Serve como linha de base para comparar os instrumentos de medição.
    """
    def __init__(self, d_in=61, dims=[64, 32, 16, 8]):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = d_in
        for d in dims:
            self.layers.append(nn.Linear(prev, d))
            prev = d
        self.head = nn.Linear(prev, 1)

        # Inicialização Xavier padrão
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.head(x)

# ══════════════════════════════════════════════════════════════════════════════
#  REDE AP — PhiAttractorNetwork simplificada para esta tarefa
#  (adaptada para produzir escalar único, sem features de banda)
# ══════════════════════════════════════════════════════════════════════════════

class RedeAP(nn.Module):
    """
    Versão da PhiAttractorNetwork adaptada para a tarefa de referência.
    Mantém as propriedades φ essenciais:
      - Dimensões Fibonacci (55→34→21→13→8)
      - Compressão 1/φ por camada
      - Inicialização φ^-(i+1)
      - Ativação SiLU (suave, sem saturação abrupta)
      - LayerNorm após cada camada
    Não usa atrator temporal (sem dados sequenciais nesta tarefa básica).
    """
    def __init__(self, d_in=61):
        super().__init__()
        # Sequência Fibonacci que parte de d_in
        DIMS_AP = [55, 34, 21, 13, 8]

        self.proj   = nn.Linear(d_in, DIMS_AP[0])
        self.layers = nn.ModuleList()
        self.norms  = nn.ModuleList()

        for i in range(len(DIMS_AP) - 1):
            self.layers.append(nn.Linear(DIMS_AP[i], DIMS_AP[i+1]))
            self.norms.append(nn.LayerNorm(DIMS_AP[i+1]))

        self.head = nn.Linear(DIMS_AP[-1], 1)

        # Inicialização φ-informada
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0 / PHI)
        for i, layer in enumerate(self.layers):
            scale = 1.0 / PHI**(i + 1)
            nn.init.xavier_uniform_(layer.weight, gain=scale)
            nn.init.zeros_(layer.bias)

        n = sum(p.numel() for p in self.parameters())
        print(f"RedeAP: {d_in} → {DIMS_AP} → 1  |  {n:,} parâmetros")

    def forward(self, x):
        x = F.silu(self.proj(x))
        for layer, norm in zip(self.layers, self.norms):
            x = F.silu(layer(x))
            x = norm(x)
        return self.head(x)

# ══════════════════════════════════════════════════════════════════════════════
#  TREINO
# ══════════════════════════════════════════════════════════════════════════════

def treinar(model, loader_tr, loader_va, n_epochs=60, lr=1e-3, nome=""):
    """Treina e retorna histórico de loss por ciclo."""
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=1/PHI**3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, factor=1/PHI, patience=8
    )
    crit = nn.MSELoss()
    history = []

    for ep in range(n_epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in loader_tr:
            optim.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PHI)
            optim.step()
            tr_loss += loss.item()

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader_va:
                pred = model(xb).squeeze(-1)
                va_loss += crit(pred, yb).item()

        tr_loss /= len(loader_tr)
        va_loss /= len(loader_va)
        sched.step(tr_loss)
        history.append({'ep': ep, 'tr': tr_loss, 'va': va_loss})

        if ep % 10 == 0:
            lr_c = optim.param_groups[0]['lr']
            print(f"  [{nome}] ep {ep:3d}  tr={tr_loss:.5f}  "
                  f"va={va_loss:.5f}  lr={lr_c:.2e}")

    return history

# ══════════════════════════════════════════════════════════════════════════════
#  EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════

print("\n─── Rede Convencional ───────────────────────────────────────────────────")
rede_conv = RedeConvencional(d_in=61, dims=[64, 32, 16, 8])
n_conv = sum(p.numel() for p in rede_conv.parameters())
print(f"RedeConv: 61 → [64,32,16,8] → 1  |  {n_conv:,} parâmetros")

hist_conv = treinar(rede_conv, ld_tr, ld_va, n_epochs=120, lr=1e-3, nome="Conv")

print("\n─── Rede AP ─────────────────────────────────────────────────────────────")
rede_ap = RedeAP(d_in=61)
hist_ap = treinar(rede_ap, ld_tr, ld_va, n_epochs=120, lr=1e-3, nome="AP  ")

# ══════════════════════════════════════════════════════════════════════════════
#  MEDIÇÃO — INSTRUMENTOS AP SOBRE AMBAS AS REDES
# ══════════════════════════════════════════════════════════════════════════════

print("\n─── Instrumentos de Medição ─────────────────────────────────────────────")

med_conv = medir_rede(rede_conv, ld_va)
med_ap   = medir_rede(rede_ap,   ld_va)

print(f"\n  Loss validação   — Conv: {med_conv['loss']:.5f}  "
      f"| AP: {med_ap['loss']:.5f}")
print()

# Mostra Grade R, Entropia e Rank por camada
print(f"  {'Camada':<30} {'Dim':>5}  {'Grade R':>8}  {'Entropia':>9}  "
      f"{'Rank Ef.':>9}")
print(f"  {'─'*30} {'─'*5}  {'─'*8}  {'─'*9}  {'─'*9}")

print("  [CONVENCIONAL]")
for nome, m in med_conv['camadas'].items():
    print(f"  {nome:<30} {m['dim']:>5}  "
          f"{m['ativ_coh']:>8.4f}  {m['entropia']:>9.4f}  "
          f"{m['rank_efetivo']:>9.2f}")

print("  [REDE AP]")
for nome, m in med_ap['camadas'].items():
    print(f"  {nome:<30} {m['dim']:>5}  "
          f"{m['ativ_coh']:>8.4f}  {m['entropia']:>9.4f}  "
          f"{m['rank_efetivo']:>9.2f}")

# ── Estabilidade do treino (variação da loss nos últimos 10 ciclos) ──────────
def estabilidade(hist, n=10):
    losses = [h['va'] for h in hist[-n:]]
    return float(np.std(losses))

est_conv = estabilidade(hist_conv)
est_ap   = estabilidade(hist_ap)

print(f"\n  Estabilidade (std loss últimos 10 ciclos):")
print(f"    Convencional : {est_conv:.6f}")
print(f"    Rede AP      : {est_ap:.6f}")

# ── Coerência de ativação global (média sobre todas as camadas) ───────────────
gr_conv = np.mean([m['ativ_coh'] for m in med_conv['camadas'].values() if m['dim'] > 1])
gr_ap   = np.mean([m['ativ_coh'] for m in med_ap['camadas'].values() if m['dim'] > 1])
en_conv = np.mean([m['entropia'] for m in med_conv['camadas'].values() if m['dim'] > 1])
en_ap   = np.mean([m['entropia'] for m in med_ap['camadas'].values() if m['dim'] > 1])
rk_conv = np.mean([m['rank_efetivo'] for m in med_conv['camadas'].values() if m['dim'] > 1])
rk_ap   = np.mean([m['rank_efetivo'] for m in med_ap['camadas'].values() if m['dim'] > 1])

print(f"\n  ── Resumo Comparativo ────────────────────────────────────────────")
print(f"  {'Instrumento':<30} {'Convencional':>14}  {'Rede AP':>10}")
print(f"  {'─'*30} {'─'*14}  {'─'*10}")
print(f"  {'Grade R médio':<30} {gr_conv:>14.4f}  {gr_ap:>10.4f}")
print(f"  {'Entropia média':<30} {en_conv:>14.4f}  {en_ap:>10.4f}")
print(f"  {'Rank efetivo médio':<30} {rk_conv:>14.2f}  {rk_ap:>10.2f}")
print(f"  {'Estabilidade (menor=melhor)':<30} {est_conv:>14.6f}  {est_ap:>10.6f}")
print(f"  {'Loss validação':<30} {med_conv['loss']:>14.5f}  {med_ap['loss']:>10.5f}")
print()

# ── Interpretação orientativa ────────────────────────────────────────────────
print("  ── Interpretação ─────────────────────────────────────────────────")
print("  Grade R > convencional → ativações AP mais coerentes com φ")
print("  Entropia < convencional → AP organiza mais, desperdicia menos")
print("  Rank efetivo menor     → AP usa menos dimensões (compressão real)")
print("  Estabilidade menor     → AP converge com menos oscilação")
print()
print("  Nota: Grade R mede concentração de ativações (não coerência-φ direta).")
print("  Conv alta = ReLU esparsifica. AP menor = SiLU distribui — outra assinatura.")
print(f"\n  φ={PHI}  α={ALPHA:.8f}")

# ── Curva de convergência (assinatura de covariação) ─────────────────────────
print("\n─── Curva de Convergência (loss validação por época) ────────────────────")
print(f"  {'Época':>6}  {'Conv va':>10}  {'AP va':>10}  {'Δ (AP-Conv)':>12}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*12}")
for hc, ha in zip(hist_conv[::10], hist_ap[::10]):
    delta = ha['va'] - hc['va']
    sinal = "+" if delta >= 0 else ""
    print(f"  {hc['ep']:>6}  {hc['va']:>10.5f}  {ha['va']:>10.5f}  {sinal}{delta:>11.5f}")

# Convergência final
print(f"\n  Convergência final (épocas 110-120):")
est_conv_final = estabilidade(hist_conv, n=10)
est_ap_final   = estabilidade(hist_ap,   n=10)
print(f"    Conv std últimas 10 épocas: {est_conv_final:.6f}")
print(f"    AP   std últimas 10 épocas: {est_ap_final:.6f}")
razao = est_ap_final / est_conv_final if est_conv_final > 0 else float('inf')
print(f"    Razão AP/Conv: {razao:.1f}×  (1.0 = convergência equivalente)")
