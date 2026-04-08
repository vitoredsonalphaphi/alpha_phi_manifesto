# AlphaPhi_BERT_Ablation_EF.py
# Teste de E (curvatura c=1/φ²) e F (todos os eixos) no bert-base-uncased
# 20 seeds — etapa necessária antes de testar campo_transmorfo
#
# Ordem metodológica (Tia, 2026-04-08):
#   1. Confirmar E e F com 20 seeds no bert-base-uncased  ← ESTE EXPERIMENTO
#   2. Se E/F não superam G → campo_transmorfo como hipótese seguinte
#   3. Se E/F superam G → curvatura já ajuda sem transição suave
#
# Colab — 3 células:
#   Célula 1: instalação
#   Célula 2: experimento principal (cole este arquivo)
#   Célula 3: print do JSON

# ════════════════════════════════════════════════════════
# CÉLULA 1 — Cole e execute primeiro
# ════════════════════════════════════════════════════════
# !pip install -q transformers datasets torch scipy

# ════════════════════════════════════════════════════════
# CÉLULA 2 — Experimento principal
# ════════════════════════════════════════════════════════

import numpy as np
import json
import time
from scipy import stats

try:
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel
    from datasets import load_dataset
except ImportError as e:
    raise ImportError(
        f"Dependência ausente: {e}\n"
        "Execute a Célula 1 antes: !pip install transformers datasets torch scipy"
    )

# ── Constantes αφ ─────────────────────────────────────────────────────────
PHI   = (1 + np.sqrt(5)) / 2       # 1.6180339887 — razão áurea
ALPHA = 1 / 137.035999084          # granularidade mínima — metade do nome αφ
C_PHI = 1.0 / PHI**2              # 0.3820 — curvatura hiperbólica nativa

# ── Seeds derivadas do timestamp ─────────────────────────────────────────
TIMESTAMP = int(time.time())
N_SEEDS   = 20
SEEDS     = [(TIMESTAMP + i * 137) % (2**31) for i in range(N_SEEDS)]

# ── Hiperparâmetros ────────────────────────────────────────────────────────
N_TRAIN    = 500
N_TEST     = 200
N_EPOCHS   = 25
LR         = 1e-3
BATCH_SIZE = 64
MAX_LEN    = 64   # SST-2: frases curtas

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"α = {ALPHA:.10f}")
print(f"φ = {PHI:.10f}")
print(f"c = {C_PHI:.10f}")
print(f"Device: {DEVICE}")
print(f"Seeds: {SEEDS[:3]}... (total: {N_SEEDS})")

# ── Funções geométricas (torch) ───────────────────────────────────────────

def expmap0_t(v, c=C_PHI):
    """Projeção Euclidiano → Poincaré (expmap na origem)."""
    c_t = torch.tensor(c, dtype=v.dtype, device=v.device)
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
    tanh_v = torch.tanh(torch.clamp(torch.sqrt(c_t) * v_norm, -15, 15))
    return tanh_v * v / (torch.sqrt(c_t) * v_norm)

def logmap0_t(y, c=C_PHI):
    """Projeção Poincaré → Euclidiano (logmap na origem)."""
    c_t = torch.tensor(c, dtype=y.dtype, device=y.device)
    y_norm = torch.norm(y, dim=-1, keepdim=True)
    max_norm = (1.0 / torch.sqrt(c_t)) - 1e-5
    y_norm = torch.clamp(y_norm, 1e-8, max_norm.item())
    return torch.arctanh(
        torch.clamp(torch.sqrt(c_t) * y_norm, -1 + 1e-8, 1 - 1e-8)
    ) * y / (torch.sqrt(c_t) * y_norm)

def phi_spectral_mod_t(x, phi=PHI):
    """Modulador espectral φ — análogo ao campo de Levin."""
    freq      = torch.fft.fft(x.float(), dim=-1)
    energia   = torch.abs(freq)
    soma      = energia.sum(dim=-1, keepdim=True) + 1e-8
    e_norm    = torch.clamp(energia / soma, 1e-10, 1.0)
    entropia  = -(e_norm * torch.log(e_norm)).sum(dim=-1, keepdim=True)
    e_norm_sc = entropia / np.log(x.shape[-1])
    coerencia = 1.0 - e_norm_sc
    return phi * torch.tanh(coerencia * phi)

class GoldenAct(nn.Module):
    """Ativação φ·tanh(x/φ) — satura em ±φ."""
    def forward(self, x):
        return PHI * torch.tanh(x / PHI)

# ── Cabeçotes de classificação ────────────────────────────────────────────

class HeadG(nn.Module):
    """
    G — Baseline euclidiano.
    768 → 256 (ReLU) → 2
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.net(x)


class HeadE(nn.Module):
    """
    E — Curvatura c=1/φ² (mesmo capacity que G, geometria diferente).
    768 → 256 (ReLU) → expmap0 → logmap0 → 2
    Testa: a projeção hiperbólica ajuda BERT mesmo com transição abrupta?
    """
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(768, 256)
        self.norm  = nn.LayerNorm(256)   # estabilidade numérica pré-expmap
        self.fc2   = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.norm(x)
        # Projeção hiperbólica (expmap0 abrupto — o que queremos testar)
        x_hyp  = expmap0_t(x)
        x_back = logmap0_t(x_hyp)
        return self.fc2(x_back)


class HeadF(nn.Module):
    """
    F — Todos os eixos Alpha-Phi:
      Fibonacci (768→144→89→55) + GoldenAct + modulação φ + curvatura.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 144)
        self.fc2 = nn.Linear(144, 89)
        self.fc3 = nn.Linear(89, 55)
        self.out = nn.Linear(55, 2)
        self.act = GoldenAct()

    def forward(self, x):
        mod = phi_spectral_mod_t(x)           # modulação espectral φ

        x = self.act(self.fc1(x)) * mod       # Fibonacci + φ·tanh + mod
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))

        # Curvatura c=1/φ²
        x_hyp  = expmap0_t(x)
        x_back = logmap0_t(x_hyp)
        return self.out(x_back)


# ── Passo 1: embeddings BERT (uma vez, reutilizados em todos os seeds) ────

print("\nCarregando bert-base-uncased...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
bert.eval()

print("Carregando SST-2...")
sst2       = load_dataset('glue', 'sst2')
train_data = sst2['train'].shuffle(seed=42).select(range(N_TRAIN))
val_data   = sst2['validation'].shuffle(seed=42).select(range(N_TEST))

def get_embeddings(dataset, batch_size=64):
    """Extrai embeddings do token [CLS] (BERT frozen)."""
    embeddings, labels = [], []
    texts = dataset['sentence']
    lbls  = dataset['label']

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors='pt'
        ).to(DEVICE)
        with torch.no_grad():
            out = bert(**enc)
            cls = out.last_hidden_state[:, 0, :]   # [CLS] token
        embeddings.append(cls.cpu())
        labels.extend(lbls[i:i + batch_size])

    return torch.cat(embeddings, dim=0), torch.tensor(labels)

print("Computando embeddings de treino...")
X_train, y_train = get_embeddings(train_data)
print("Computando embeddings de teste...")
X_test,  y_test  = get_embeddings(val_data)
print(f"Embeddings: treino={X_train.shape}, teste={X_test.shape}")

# ── Passo 2: loop de treinamento ──────────────────────────────────────────

def train_eval(head_class, seed, X_tr, y_tr, X_te, y_te):
    """Treina um cabeçote sobre embeddings fixos; retorna acurácia."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**31))

    model     = head_class().to(DEVICE)
    X_tr_d    = X_tr.to(DEVICE)
    y_tr_d    = y_tr.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    n         = len(X_tr_d)

    for _ in range(N_EPOCHS):
        perm = torch.randperm(n)
        X_sh, y_sh = X_tr_d[perm], y_tr_d[perm]

        model.train()
        for i in range(0, n, BATCH_SIZE):
            xb, yb = X_sh[i:i + BATCH_SIZE], y_sh[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_te.to(DEVICE)).argmax(dim=-1)
        acc   = (preds == y_te.to(DEVICE)).float().mean().item()
    return acc

# ── Passo 3: execução ─────────────────────────────────────────────────────

CONFIGS = {'E': HeadE, 'F': HeadF, 'G': HeadG}
results = {k: [] for k in CONFIGS}
total   = len(CONFIGS) * N_SEEDS

print(f"\nIniciando: {len(CONFIGS)} configs × {N_SEEDS} seeds = {total} runs\n")

run = 0
for cfg_name, head_cls in CONFIGS.items():
    print(f"Config {cfg_name}:")
    for seed in SEEDS:
        acc = train_eval(head_cls, seed, X_train, y_train, X_test, y_test)
        results[cfg_name].append(acc)
        run += 1
        print(f"  seed={seed % 10000:04d}  acc={acc:.4f}  ({run}/{total})")

    accs = np.array(results[cfg_name])
    print(f"  → {cfg_name}: {accs.mean():.4f} ± {accs.std():.4f}\n")

# ── Passo 4: testes estatísticos ──────────────────────────────────────────

print("── Testes estatísticos (paired t-test) ──")
stat_tests = {}
for k in ['E', 'F']:
    t_val, p_val = stats.ttest_rel(results[k], results['G'])
    delta = float(np.mean(results[k])) - float(np.mean(results['G']))
    sig   = "✓ significativo" if p_val < 0.05 else "ns (não significativo)"
    print(f"{k} vs G:  Δ={delta:+.4f}  p={p_val:.4f}  {sig}")
    stat_tests[f"{k}_vs_G"] = {"delta": round(delta, 6), "p_value": round(float(p_val), 6)}

t_ef, p_ef = stats.ttest_rel(results['F'], results['E'])
d_ef = float(np.mean(results['F'])) - float(np.mean(results['E']))
print(f"F vs E:  Δ={d_ef:+.4f}  p={p_ef:.4f}  {'✓' if p_ef < 0.05 else 'ns'}")
stat_tests["F_vs_E"] = {"delta": round(d_ef, 6), "p_value": round(float(p_ef), 6)}

# ── Interpretação automática ───────────────────────────────────────────────
print("\n── Interpretação ──")
e_sig = stat_tests["E_vs_G"]["p_value"] < 0.05
f_sig = stat_tests["F_vs_G"]["p_value"] < 0.05

if e_sig or f_sig:
    print("Curvatura c=1/φ² AJUDA mesmo com expmap0 abrupto no BERT.")
    print("→ campo_transmorfo pode amplificar o efeito.")
else:
    print("Curvatura c=1/φ² NÃO ajuda com expmap0 abrupto no BERT.")
    print("→ campo_transmorfo é o próximo passo: transição suave pode resolver.")

# ── Export JSON ───────────────────────────────────────────────────────────

export = {
    "experimento":  "BERT_Ablation_EF",
    "substrato":    "bert-base-uncased",
    "dataset":      "SST-2",
    "bert_frozen":  True,
    "n_seeds":      N_SEEDS,
    "n_epochs":     N_EPOCHS,
    "lr":           LR,
    "batch_size":   BATCH_SIZE,
    "n_train":      N_TRAIN,
    "n_test":       N_TEST,
    "timestamp":    TIMESTAMP,
    "seeds":        SEEDS,
    "resultados": {
        k: {
            "mean":   float(np.mean(results[k])),
            "std":    float(np.std(results[k])),
            "values": [float(v) for v in results[k]]
        }
        for k in results
    },
    "testes_estatisticos": stat_tests,
    "interpretacao": {
        "E_ajuda_BERT": bool(e_sig),
        "F_ajuda_BERT": bool(f_sig),
        "proximo_passo": (
            "campo_transmorfo" if not (e_sig or f_sig)
            else "E/F confirmados — campo_transmorfo como extensão"
        )
    },
    "nota": (
        f"α={ALPHA:.10f} — granularidade mínima / metade do nome αφ. "
        f"Experimento conduzido por isomorfismo: bordado → campo_transmorfo."
    )
}

with open('bert_ablation_ef_results.json', 'w', encoding='utf-8') as f:
    json.dump(export, f, indent=2, ensure_ascii=False)

print("\nSalvo: bert_ablation_ef_results.json")

# ════════════════════════════════════════════════════════
# CÉLULA 3 — Alternativa ao download (cole em célula separada)
# ════════════════════════════════════════════════════════
# import json
# with open('bert_ablation_ef_results.json') as f:
#     dados = json.load(f)
# print(json.dumps(dados, indent=2, ensure_ascii=False))
