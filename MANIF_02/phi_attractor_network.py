"""
phi_attractor_network.py
Rede Neural φ-Atratora — Estrutura Base
Alpha-Phi · MANIF_02

Proposta: rede neural onde o atrator φ está embutido na arquitetura
desde a construção — não é aprendido durante o treinamento.
A coerência harmônica é uma propriedade estrutural invariante.

Princípio central:
    A rede não descobre coerência. Ela foi construída a partir de coerência.
    Cada função agregada herda o protocolo do atrator.

Arquitetura:
    Entrada    : 4 × N_BANDAS + 1 features por φ-banda
    Projeção   : Linear → 89 (Fibonacci)
    Camadas    : 89 → 55 → 34 → 21 → 13 → 8  (sequência Fibonacci)
    Checkpoint : coerência φ medida em cada camada
    Atrator    : estado acumulado φ-ponderado, propaga entre frames
    Saída      : α* ∈ [1e-4, 0.35]  (ponto de emergência por fonema)

φ opera em:
    - dimensões das camadas  (compressão 1/φ por transição)
    - inicialização de pesos (escala φ^-i por camada)
    - pesos da loss          (φ^-i por componente)
    - decaimento do LR       (fator 1/φ em cada plateau)
    - estado do atrator      (média φ-ponderada: 0.618 memória + 0.382 novo)

Conexão com o Manifesto Alpha-Phi (p. 116):
    "O eco-φ detecta coerência em qualquer campo que se propague,
     reflita e interfira — independente do substrato."
    Esta rede é a implementação computacional direta dessa afirmação.

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Constantes ────────────────────────────────────────────────
PHI     = (1 + np.sqrt(5)) / 2   # 1.6180339887
ALPHA_0 = 1 / 137.035999         # piso EM — ponto de partida do atrator
ALPHA_MAX = 0.35                  # teto de busca (distorção aceitável)

# ── Sequência Fibonacci da arquitetura ────────────────────────
# 89 → 55 → 34 → 21 → 13 → 8
# Razão entre camadas: 55/89 ≈ 34/55 ≈ 21/34 ≈ 13/21 ≈ 8/13 ≈ 1/φ
DIMS = [89, 55, 34, 21, 13, 8]


# ── Extração de features por φ-banda ─────────────────────────

def extrair_features_phi(frame, campo_bip, bins_phi, alpha_prev=ALPHA_0):
    """
    Extrai features compactas por φ-banda de um frame e seu campo_bip.

    Estrutura (4 × N_BANDAS + 1):
        [0        : N_BANDAS  ]  magnitudes médias por banda (frame)
        [N_BANDAS : 2*N_BANDAS]  magnitudes médias por banda (campo_bip)
        [2*N      : 3*N_BANDAS]  coerências por banda (via eco_eq local)
        [3*N      : 4*N_BANDAS]  fases médias por banda (frame)
        [-1]                     alpha_prev (estado angular propagado)

    Nota: N_BANDAS ≈ 15 para SR=44100Hz (φ-espaçamento de 20Hz a 22050Hz).
    A projeção de entrada da rede mapeia para 89 (Fibonacci).

    Parâmetros:
        frame      : np.ndarray (WIN_SIZE,) — frame temporal
        campo_bip  : np.ndarray (WIN_SIZE,) — saída do eco_eq sobre o frame
        bins_phi   : lista de (b_lo, b_hi, f_lo, f_hi) por φ-banda
        alpha_prev : float — α* do frame anterior (estado angular)

    Retorna:
        feat : np.ndarray (4*N_BANDAS+1,) float32
    """
    N = len(bins_phi)
    F_frame = np.fft.rfft(frame)
    F_bip   = np.fft.rfft(campo_bip)

    mag_frame  = np.zeros(N, dtype=np.float32)
    mag_bip    = np.zeros(N, dtype=np.float32)
    coh_banda  = np.zeros(N, dtype=np.float32)
    phase_mean = np.zeros(N, dtype=np.float32)

    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        Fb = F_frame[b_lo:b_hi]
        if len(Fb) == 0:
            continue
        mag = np.abs(Fb)

        mag_frame[i] = float(mag.mean())
        mag_bip[i]   = float(np.abs(F_bip[b_lo:b_hi]).mean())

        # Coerência espectral da banda: 1 - H_normalizada
        an    = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
        H     = -np.sum(an * np.log(an))
        H_max = np.log(max(len(an), 2))
        coh_banda[i] = float(1.0 - H / H_max)

        phase_mean[i] = float(np.angle(Fb).mean())

    # Normaliza cada bloco
    def norm01(v):
        mn, mx = v.min(), v.max()
        return (v - mn) / (mx - mn + 1e-12)

    def norm11(v):
        m = np.max(np.abs(v)) + 1e-12
        return v / m

    feat = np.concatenate([
        norm01(mag_frame),          # magnitudes frame    [0:N]
        norm01(mag_bip),            # magnitudes campo    [N:2N]
        coh_banda,                  # coerências          [2N:3N] já em [0,1]
        norm11(phase_mean),         # fases               [3N:4N]
        [float(alpha_prev)]         # estado angular      [-1]
    ]).astype(np.float32)

    return feat


# ── Checkpoint de Coerência φ ─────────────────────────────────

class CoherenceCheckpoint(nn.Module):
    """
    Mede coerência φ no espaço de ativações de uma camada.

    Coerência = 1 - H_normalizada das magnitudes absolutas das ativações.

    Alta coerência → ativações concentradas (estrutura emergiu).
    Baixa coerência → ativações difusas (ainda é ruído).

    O checkpoint não modifica as ativações — apenas as observa.
    É o "Scanner" da rede: mede sem interferir.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.log_dim = float(np.log(dim))  # H_max para esta dimensão

    def forward(self, x):
        # x: (batch, dim)
        mag      = torch.abs(x) + 1e-10
        mag_norm = mag / (mag.sum(dim=-1, keepdim=True) + 1e-10)
        mag_norm = torch.clamp(mag_norm, 1e-10, 1.0)
        H        = -torch.sum(mag_norm * torch.log(mag_norm), dim=-1)
        coerencia = 1.0 - H / self.log_dim   # (batch,) ∈ [0, 1]
        return coerencia


# ── Rede Neural φ-Atratora ────────────────────────────────────

class PhiAttractorNetwork(nn.Module):
    """
    Rede com atrator φ embutido desde a construção.

    O atrator não é um parâmetro aprendido.
    Ele é uma propriedade da arquitetura: cada camada contrai o espaço
    de representação por 1/φ, preservando coerência em cada transição.

    O estado do atrator acumula coerência ao longo do forward pass
    e propaga entre frames consecutivos (memória angular).

    Esta memória é o que permite à rede predizer α* sem busca exaustiva:
    ela "conhece" a trajetória do campo porque carrega seu histórico.
    """

    def __init__(self, n_bandas, dims=None):
        super().__init__()
        if dims is None:
            dims = DIMS

        self.phi  = PHI
        self.dims = dims
        self.n_bandas = n_bandas

        # Projeção de entrada: features brutas → espaço Fibonacci
        input_raw = 4 * n_bandas + 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_raw, dims[0]),
            nn.LayerNorm(dims[0])
        )

        # Camadas Fibonacci + checkpoints de coerência
        self.layers      = nn.ModuleList()
        self.checkpoints = nn.ModuleList()
        self.norms       = nn.ModuleList()

        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i+1]
            self.layers.append(nn.Linear(d_in, d_out))
            self.checkpoints.append(CoherenceCheckpoint(d_out))
            self.norms.append(nn.LayerNorm(d_out))

        # Cabeça de saída: α* ∈ [1e-4, ALPHA_MAX]
        self.head = nn.Sequential(
            nn.Linear(dims[-1], 1),
            nn.Sigmoid()
        )

        self._init_phi_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"PhiAttractorNetwork")
        print(f"  input_raw={input_raw} → proj → {dims}")
        print(f"  parâmetros: {n_params:,}")
        print(f"  razões entre camadas:")
        for i in range(len(dims) - 1):
            r = dims[i+1] / dims[i]
            print(f"    {dims[i]:3d} → {dims[i+1]:3d} : {r:.4f}  "
                  f"(1/φ = {1/PHI:.4f})")

    def _init_phi_weights(self):
        """
        Inicialização φ-informada:
        camada i tem escala φ^-(i+1) — cada camada mais profunda
        começa com pesos menores, refletindo a contração Fibonacci.
        """
        for i, layer in enumerate(self.layers):
            scale = float(1.0 / self.phi ** (i + 1))
            nn.init.xavier_uniform_(layer.weight, gain=scale)
            nn.init.zeros_(layer.bias)

    def forward(self, x, attractor_state=None):
        """
        Parâmetros:
            x               : (batch, 4*n_bandas+1) — features φ-banda
            attractor_state : (batch,) — estado do frame anterior, ou None

        Retorna:
            alpha_pred      : (batch,) — α* ∈ [1e-4, ALPHA_MAX]
            new_attractor   : (batch,) — estado para o próximo frame
            coerencias      : list[(batch,)] — coerência por camada

        Estado do atrator:
            att = (1/φ) × att_anterior + (1 - 1/φ) × coerência_nova
            Memória (0.618) domina sobre novidade (0.382) —
            o atrator é conservador, como φ.
        """
        batch = x.shape[0]

        # Inicializa atrator no piso EM se primeiro frame
        if attractor_state is None:
            attractor_state = torch.full(
                (batch,), float(ALPHA_0), device=x.device
            )

        wm = 1.0 / self.phi        # peso memória = 0.618
        wn = 1.0 - 1.0 / self.phi  # peso novidade = 0.382

        # Projeção de entrada
        x = self.input_proj(x)

        coerencias = []

        for layer, ckpt, norm in zip(self.layers, self.checkpoints, self.norms):
            x   = F.silu(layer(x))    # SiLU: suave, sem saturação abrupta
            x   = norm(x)
            coh = ckpt(x)             # coerência desta camada
            coerencias.append(coh)

            # Atualiza atrator: memória domina novidade (φ-ponderado)
            attractor_state = wm * attractor_state + wn * coh

        # α* escalado: Sigmoid → [0,1] → [1e-4, ALPHA_MAX]
        alpha_raw  = self.head(x).squeeze(-1)
        alpha_pred = 1e-4 + alpha_raw * (ALPHA_MAX - 1e-4)

        return alpha_pred, attractor_state, coerencias


# ── Loss φ-Composta ───────────────────────────────────────────

class PhiLoss(nn.Module):
    """
    Loss com quatro componentes, pesos decrescentes por potências de φ:

        L = 1.000 × L_alpha    (erro de predição de α*)
          + 0.618 × L_coh      (coerência crescente por camada)
          + 0.382 × L_att      (consistência do atrator entre frames)
          + 0.236 × L_mono     (α* dentro do domínio)

    L_alpha : MSE entre α*_predito e α*_target (ground truth da busca hermética)
    L_coh   : penaliza coerência decrescente entre camadas consecutivas
    L_att   : penaliza variação brusca do atrator entre frames
    L_mono  : penaliza predições fora de [1e-4, ALPHA_MAX]
    """
    def __init__(self):
        super().__init__()
        self.w = [1.0, 1/PHI, 1/PHI**2, 1/PHI**3]

    def forward(self, alpha_pred, alpha_target,
                coerencias, attractor_curr, attractor_prev=None):

        # L1: erro de predição de α*
        L_alpha = F.mse_loss(alpha_pred, alpha_target)

        # L2: coerência deve crescer camada a camada
        L_coh = torch.tensor(0.0, device=alpha_pred.device)
        for i in range(len(coerencias) - 1):
            delta = coerencias[i+1] - coerencias[i]
            L_coh = L_coh + F.relu(-delta).mean()  # penaliza queda

        # L3: consistência do atrator entre frames consecutivos
        L_att = torch.tensor(0.0, device=alpha_pred.device)
        if attractor_prev is not None:
            L_att = torch.abs(attractor_curr - attractor_prev).mean()

        # L4: α* não deve sair do domínio
        L_mono = F.relu(alpha_pred - ALPHA_MAX).mean()

        total = (self.w[0] * L_alpha +
                 self.w[1] * L_coh   +
                 self.w[2] * L_att   +
                 self.w[3] * L_mono)

        return total, {
            'L_alpha': L_alpha.item(),
            'L_coh':   L_coh.item(),
            'L_att':   L_att.item(),
            'L_mono':  L_mono.item()
        }


# ── Dataset ───────────────────────────────────────────────────

class PhiFrameDataset(Dataset):
    """
    Dataset de frames com targets α* gerados pela busca hermética v2.0.

    Os targets são o ground truth: α* encontrado pelo hermético 2
    (corr(eco_res(track1, α), campo_bip)²) frame a frame.

    A rede aprende a predizer esses targets diretamente,
    sem realizar a busca — apenas por reconhecimento estrutural.

    IMPORTANTE: NÃO embaralhar durante o treinamento.
    A ordem temporal é essencial — o atrator tem memória dos frames
    anteriores, e esse histórico é parte do sinal de aprendizado.
    """
    def __init__(self, features, alphas_target):
        assert len(features) == len(alphas_target), \
            "features e alphas_target devem ter o mesmo comprimento"
        self.features = torch.tensor(
            np.array(features), dtype=torch.float32
        )
        self.targets = torch.tensor(
            np.array(alphas_target), dtype=torch.float32
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]


# ── Treinamento ───────────────────────────────────────────────

def treinar(model, dataset, n_epochs=100, lr=1e-3, batch_size=64,
            device='cpu'):
    """
    Treinamento com propagação temporal do estado do atrator.

    NÃO embaralha — a ordem dos frames é o sinal de memória do atrator.
    LR decai por 1/φ em cada plateau (patience=8 épocas).
    Gradient clipping com norma máxima = φ.

    Retorna: list de dicts com métricas por época.
    """
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1 / PHI**3   # regularização φ-proporcional
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=1/PHI, patience=8, verbose=True
    )
    criterion = PhiLoss()
    history   = []

    for epoch in range(n_epochs):
        model.train()
        ep_loss = ep_err = 0.0
        att_state = att_prev = None

        for feat, target in loader:
            feat   = feat.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            alpha_pred, att_state, cohs = model(feat, att_state)

            loss, bd = criterion(
                alpha_pred, target, cohs,
                att_state,
                att_prev
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), PHI)
            optimizer.step()

            att_prev  = att_state.detach().clone()
            att_state = att_state.detach()

            ep_loss += loss.item()
            ep_err  += torch.abs(alpha_pred.detach() - target).mean().item()

        n       = len(loader)
        avg_loss = ep_loss / n
        avg_err  = ep_err  / n
        history.append({'epoch': epoch, 'loss': avg_loss, 'err': avg_err})
        scheduler.step(avg_loss)

        if epoch % 10 == 0:
            lr_curr = optimizer.param_groups[0]['lr']
            print(f"  ep {epoch:3d}  loss={avg_loss:.6f}  "
                  f"|Δα|={avg_err:.6f}  lr={lr_curr:.2e}")

    return history


# ── Inferência ────────────────────────────────────────────────

def predizer_alpha(model, features_seq, device='cpu'):
    """
    Dado uma sequência de feature vectors (n_frames × input_raw),
    prediz a trajetória de α* sem busca exaustiva.

    Esta função substitui busca_hermetica_v1_2 em produção.

    Velocidade estimada: ~1000× mais rápido que a busca (1 forward
    vs 30 chamadas de eco_res por frame).

    Parâmetros:
        model        : PhiAttractorNetwork treinado
        features_seq : np.ndarray (n_frames, input_raw)

    Retorna:
        alphas : np.ndarray (n_frames,) — α* predito por frame
        atts   : np.ndarray (n_frames,) — trajetória do atrator
    """
    model.eval()
    model = model.to(device)
    alphas = []
    atts   = []
    att    = None

    with torch.no_grad():
        for feat in features_seq:
            f = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            alpha_pred, att, _ = model(f, att)
            alphas.append(float(alpha_pred.cpu()))
            atts.append(float(att.cpu().mean()))

    return np.array(alphas), np.array(atts)


# ── Verificação da arquitetura ────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  PhiAttractorNetwork — Verificação da Arquitetura")
    print("  Alpha-Phi · MANIF_02")
    print("=" * 60)

    # N_BANDAS típico para SR=44100Hz (φ-espaçamento 20Hz → 22050Hz)
    N_BANDAS_DEMO = 15
    model = PhiAttractorNetwork(n_bandas=N_BANDAS_DEMO, dims=DIMS)

    print(f"\nInput raw: 4×{N_BANDAS_DEMO}+1 = {4*N_BANDAS_DEMO+1} features")
    print(f"PHI = {PHI:.10f}")
    print(f"ALPHA_0 = {ALPHA_0:.8f}  (1/137)")

    # Teste forward com batch de 4 frames
    x_demo = torch.randn(4, 4 * N_BANDAS_DEMO + 1)
    alpha_pred, att_state, cohs = model(x_demo)

    print(f"\n── Forward pass (batch=4) ──")
    print(f"α* predito : {alpha_pred.detach().numpy().round(6)}")
    print(f"Atrator    : {att_state.detach().numpy().round(6)}")
    print(f"Coerências por camada:")
    for i, (d, c) in enumerate(zip(DIMS[1:], cohs)):
        print(f"  camada {i+1}: dim={d:3d}  "
              f"coerência_média={c.mean().item():.4f}")

    # Verifica razões entre camadas
    print(f"\n── Razões de compressão (target: 1/φ = {1/PHI:.4f}) ──")
    for i in range(len(DIMS) - 1):
        r = DIMS[i+1] / DIMS[i]
        delta = abs(r - 1/PHI)
        ok = "✓" if delta < 0.01 else "~"
        print(f"  {DIMS[i]:3d} → {DIMS[i+1]:3d} : {r:.4f}  {ok}")

    # Teste de propagação temporal (simula 3 frames consecutivos)
    print(f"\n── Propagação temporal do atrator (3 frames) ──")
    att = None
    for fi in range(3):
        x_f = torch.randn(1, 4 * N_BANDAS_DEMO + 1)
        ap, att, _ = model(x_f, att)
        print(f"  frame {fi}: α*={float(ap):.6f}  "
              f"atrator={float(att.mean()):.6f}")

    print(f"\n✓ Arquitetura verificada.")
    print(f"  Próximo passo: gerar dataset com targets α* da busca")
    print(f"  hermética v2.0 e treinar sobre frames do áudio Gemini.")
    print("=" * 60)
