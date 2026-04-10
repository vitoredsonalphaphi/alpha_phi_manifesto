# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

# AlphaPhi_BERT_Transmorfo.py
# Hipótese campo_transmorfo: transição suave Euclidiano → Hiperbólico no BERT
#
# NOTA METODOLÓGICA:
#   A foto do bordado propôs o CONCEITO: uma linha que parte do euclidiano
#   e chega ao hiperbólico sem se romper. Não propôs a forma arquitetural.
#   Este script testa UMA leitura do estado transmorfo — não a única possível.
#   A complexidade da forma permanece em aberto.
#
# Comparação isolada:
#   G    — baseline: sem curvatura
#   E    — curvatura c=1/φ²: expmap0 ABRUPTO (hard jump)
#   E_T  — curvatura c=1/φ²: campo_transmorfo SUAVE (progressive)
#
#   E vs E_T isola: hard jump vs transição contínua (mesma curvatura, forma diferente)
#   E_T vs G isola: transição suave vs sem curvatura
#
# ════════════════════════════════════════════════════════
# CÉLULA 1
# !pip install -q transformers datasets torch scipy
# ════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════
# CÉLULA 2 — cole a partir daqui
# ════════════════════════════════════════════════════════

import numpy as np
import json
import time
from scipy import stats
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

# ── Constantes αφ ─────────────────────────────────────────────────────────
PHI   = (1 + 5**0.5) / 2
ALPHA = 1 / 137.035999084
C_PHI = 1.0 / PHI**2

# ── Seeds ─────────────────────────────────────────────────────────────────
TIMESTAMP = int(time.time())
N_SEEDS   = 20
SEEDS     = [(TIMESTAMP + i * 137) % (2**31) for i in range(N_SEEDS)]

# ── Config ─────────────────────────────────────────────────────────────────
N_TRAIN    = 500
N_TEST     = 200
N_EPOCHS   = 25
LR         = 1e-3
BATCH_SIZE = 64
MAX_LEN    = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"α={ALPHA:.6f}  φ={PHI:.6f}  c={C_PHI:.6f}")
print(f"Device: {DEVICE}  |  Seeds[0]: {SEEDS[0]}")

# ── Geometria ──────────────────────────────────────────────────────────────

def expmap0_t(v, c=C_PHI):
    c_t = torch.tensor(c, dtype=v.dtype, device=v.device)
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
    tanh_v = torch.tanh(torch.clamp(c_t**0.5 * v_norm, -15, 15))
    return tanh_v * v / (c_t**0.5 * v_norm)

def logmap0_t(y, c=C_PHI):
    c_t = torch.tensor(c, dtype=y.dtype, device=y.device)
    y_norm = torch.norm(y, dim=-1, keepdim=True)
    max_norm = (1.0 / c_t**0.5) - 1e-5
    y_norm = torch.clamp(y_norm, 1e-8, max_norm.item())
    return torch.arctanh(
        torch.clamp(c_t**0.5 * y_norm, -1+1e-8, 1-1e-8)
    ) * y / (c_t**0.5 * y_norm)

def curvatura_progressiva(layer_idx, total_layers, c_target=C_PHI):
    """c cresce de 0 até c_target, modulado por φ."""
    t = layer_idx / max(total_layers - 1, 1)
    return c_target * (t ** PHI)

def aplicar_transmorfo(x, layer_idx, total_layers):
    """
    Transição suave: blend entre euclidiano e hiperbólico.
    α(t) = t^φ — lento no início, rápido no fim.
    """
    c = curvatura_progressiva(layer_idx, total_layers)
    if c < 1e-6:
        return x
    x_hyp  = expmap0_t(x, c=c)
    x_back = logmap0_t(x_hyp, c=c)
    alpha  = (layer_idx / max(total_layers - 1, 1)) ** PHI
    return (1.0 - alpha) * x + alpha * x_back

# ── Cabeçotes ──────────────────────────────────────────────────────────────

class HeadG(nn.Module):
    """G — Baseline: sem curvatura."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.net(x)


class HeadE(nn.Module):
    """
    E — Curvatura c=1/φ², expmap0 ABRUPTO.
    Referência: o que já testamos. Hard jump no gradiente.
    """
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(768, 256)
        self.norm = nn.LayerNorm(256)
        self.fc2  = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.norm(x)
        x = logmap0_t(expmap0_t(x))   # salto abrupto
        return self.fc2(x)


class HeadE_T(nn.Module):
    """
    E_T — campo_transmorfo: transição SUAVE em 2 estágios.

    Estágio 1 (layer_idx=1, total=3): c ≈ 0.121 — curvatura emergente
    Estágio 2 (layer_idx=2, total=3): c = 0.382 — hiperbólico pleno

    NOTA: requer uma camada extra para criar os estágios de transição.
    Isso é o custo estrutural do estado transmorfo.
    A forma (2 estágios) é UMA leitura do conceito — não a única.
    """
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(768, 256)
        self.norm1 = nn.LayerNorm(256)
        self.fc2   = nn.Linear(256, 256)
        self.norm2 = nn.LayerNorm(256)
        self.out   = nn.Linear(256, 2)
        self.total = 3   # 3 pontos: entrada (0), estágio1 (1), estágio2 (2)

    def forward(self, x):
        # Estágio 1 — curvatura emergente
        x = torch.relu(self.fc1(x))
        x = self.norm1(x)
        x = aplicar_transmorfo(x, layer_idx=1, total_layers=self.total)

        # Estágio 2 — hiperbólico pleno
        x = torch.relu(self.fc2(x))
        x = self.norm2(x)
        x = aplicar_transmorfo(x, layer_idx=2, total_layers=self.total)

        return self.out(x)


# ── BERT + embeddings ──────────────────────────────────────────────────────

print("\nCarregando BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
bert.eval()

print("Carregando SST-2...")
sst2       = load_dataset('glue', 'sst2')
train_data = sst2['train'].shuffle(seed=42).select(range(N_TRAIN))
val_data   = sst2['validation'].shuffle(seed=42).select(range(N_TEST))

def get_embeddings(dataset):
    embs, labels = [], []
    for i in range(0, len(dataset['sentence']), 64):
        enc = tokenizer(
            dataset['sentence'][i:i+64], padding=True,
            truncation=True, max_length=MAX_LEN, return_tensors='pt'
        ).to(DEVICE)
        with torch.no_grad():
            embs.append(bert(**enc).last_hidden_state[:, 0, :].cpu())
        labels.extend(dataset['label'][i:i+64])
    return torch.cat(embs), torch.tensor(labels)

print("Computando embeddings (uma vez, reutilizados em todos os seeds)...")
X_train, y_train = get_embeddings(train_data)
X_test,  y_test  = get_embeddings(val_data)
print(f"Pronto: {X_train.shape}")

# ── Loop de treino ─────────────────────────────────────────────────────────

def train_eval(HeadClass, seed):
    torch.manual_seed(seed)
    model = HeadClass().to(DEVICE)
    Xtr   = X_train.to(DEVICE)
    ytr   = y_train.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()
    n     = len(Xtr)

    for _ in range(N_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            xb, yb = Xtr[perm[i:i+BATCH_SIZE]], ytr[perm[i:i+BATCH_SIZE]]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).argmax(-1)
        return (preds == y_test.to(DEVICE)).float().mean().item()

# ── Execução ───────────────────────────────────────────────────────────────

CONFIGS = {'G': HeadG, 'E': HeadE, 'E_T': HeadE_T}
results = {k: [] for k in CONFIGS}
total   = len(CONFIGS) * N_SEEDS
run     = 0

print(f"\n{len(CONFIGS)} configs × {N_SEEDS} seeds = {total} runs\n")

for cfg, HeadClass in CONFIGS.items():
    print(f"Config {cfg}:")
    for seed in SEEDS:
        acc = train_eval(HeadClass, seed)
        results[cfg].append(acc)
        run += 1
        print(f"  {seed%10000:04d}  {acc:.4f}  ({run}/{total})")
    arr = np.array(results[cfg])
    print(f"  → {cfg}: {arr.mean():.4f} ± {arr.std():.4f}\n")

# ── Testes estatísticos ────────────────────────────────────────────────────

print("── Testes estatísticos ──")
pares = [('E_T','G'), ('E_T','E'), ('E','G')]
stat_tests = {}
for a, b in pares:
    t, p = stats.ttest_rel(results[a], results[b])
    d    = np.mean(results[a]) - np.mean(results[b])
    sig  = "✓" if p < 0.05 else "ns"
    print(f"{a} vs {b}:  Δ={d:+.4f}  p={p:.4f}  {sig}")
    stat_tests[f"{a}_vs_{b}"] = {"delta": round(float(d), 6),
                                  "p_value": round(float(p), 6)}

# ── Interpretação automática ───────────────────────────────────────────────
print("\n── Interpretação ──")
et_vs_g = stat_tests["E_T_vs_G"]["p_value"] < 0.05
et_vs_e = stat_tests["E_T_vs_E"]["p_value"] < 0.05

if et_vs_g and et_vs_e:
    interpretacao = "campo_transmorfo supera G E supera E: transição suave é o mecanismo."
elif et_vs_g and not et_vs_e:
    interpretacao = "campo_transmorfo supera G mas não supera E: curvatura ajuda, forma indistinta."
elif not et_vs_g and et_vs_e:
    interpretacao = "campo_transmorfo não supera G mas supera E: sinal da transição, substrato resiste."
else:
    interpretacao = "campo_transmorfo não supera G nem E: hipótese desta forma não confirmada. Forma alternativa em aberto."

print(interpretacao)

# ── Export ─────────────────────────────────────────────────────────────────

export = {
    "experimento":   "BERT_Transmorfo",
    "substrato":     "bert-base-uncased",
    "bert_frozen":   True,
    "hipotese":      "campo_transmorfo — transição suave Euclidiano→Hiperbólico",
    "nota_conceito": (
        "A foto propôs o CONCEITO (linha contínua euclidiano→hiperbólico), "
        "não a forma. Este experimento testa UMA leitura. "
        "A forma da complexidade permanece em aberto."
    ),
    "n_seeds":    N_SEEDS,
    "n_epochs":   N_EPOCHS,
    "n_train":    N_TRAIN,
    "n_test":     N_TEST,
    "timestamp":  TIMESTAMP,
    "seeds":      SEEDS,
    "resultados": {
        k: {
            "mean":   float(np.mean(results[k])),
            "std":    float(np.std(results[k])),
            "values": [float(v) for v in results[k]]
        }
        for k in results
    },
    "testes_estatisticos": stat_tests,
    "interpretacao": interpretacao,
    "referencia_anterior": {
        "E_vs_G_20seeds": {"delta": 0.0037, "p_value": 0.1556, "resultado": "ns"}
    }
}

with open('bert_transmorfo_results.json', 'w', encoding='utf-8') as f:
    json.dump(export, f, indent=2, ensure_ascii=False)

print("\nSalvo: bert_transmorfo_results.json")

# ════════════════════════════════════════════════════════
# CÉLULA 3 — print do JSON
# import json
# with open('bert_transmorfo_results.json') as f:
#     dados = json.load(f)
# print(json.dumps(dados, indent=2, ensure_ascii=False))
# ════════════════════════════════════════════════════════
