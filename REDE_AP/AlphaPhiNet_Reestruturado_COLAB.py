"""
AlphaPhiNet_Reestruturado_COLAB.py
Reestruturação dos testes auditados em 03/10/2026 — cada teste tem um controle que pode falhar.

  Parte 1 · Instrumentos corrigidos: Coh_rel (sem dependência de ln n) e D_φ (detecta φ, não impõe)
  Parte 2 · Ablação do φ-init (isola dims, ativação, LayerNorm e gain) — 5 seeds, dataset real
  Parte 3 · α como âncora: residual LayerScale com γ0 = α, comparado com outros γ0
  Parte 4 · MPAP como regularizador (pressão φ) e como pós-processador de saída

Dataset: sklearn digits (1797 imagens 8x8, 10 classes) — roda offline no Colab.
Florianópolis · 03/10/2026 · Vitor Edson Delavi · Claude
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PHI   = 1.6180339887
ALPHA = 1 / 137.035999
SEAL  = 1 / PHI
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 40

# ── PARTE 1 · INSTRUMENTOS ────────────────────────────────────────────────────

def coh(v):
    m = np.abs(v) + 1e-12
    p = m / m.sum(-1, keepdims=True)
    return 1 - (-(p * np.log(p)).sum(-1)) / np.log(v.shape[-1])

def perfil_phi(n):
    w = SEAL * (1 - SEAL) ** np.arange(n)
    return w / w.sum()

def coh_star(n):
    return float(coh(perfil_phi(n)[None])[0])

def coh_rel(v):
    # Coh dividido pelo máximo φ alcançável naquela dimensão: remove o artefato 1 − 1.0757/ln(n)
    return coh(v) / coh_star(v.shape[-1])

def d_phi(v):
    # Jensen-Shannon entre o perfil ordenado de |v| e o perfil φ, em [0,1]. 0 = perfil φ exato.
    n = v.shape[-1]
    m = np.sort(np.abs(v), axis=-1)[..., ::-1] + 1e-12
    p = m / m.sum(-1, keepdims=True)
    q = np.broadcast_to(perfil_phi(n), p.shape)
    mm = 0.5 * (p + q)
    js = 0.5 * (p * np.log(p / mm)).sum(-1) + 0.5 * (q * np.log(q / mm)).sum(-1)
    return js / np.log(2)

def projecao_phi(v):
    n = v.shape[-1]
    idx = np.argsort(np.abs(v))[::-1]
    r = np.empty(n)
    r[idx] = np.sign(v[idx]) * perfil_phi(n) * np.abs(v).sum()
    return r

def mpap_antigo(v, max_c=10):
    c = 0
    while coh(v[None])[0] < SEAL and c < max_c:
        v = projecao_phi(v); c += 1
    return c

def mpap_ancorado(v, max_c=10, lam=SEAL):
    # passo suave em direção a φ + âncora α no sinal de origem (centro → fora)
    v0, c = v.copy(), 0
    while coh_rel(v[None])[0] < SEAL and c < max_c:
        v = ALPHA * v0 + (1 - ALPHA) * ((1 - lam) * v + lam * projecao_phi(v)); c += 1
    return v, c

def parte1():
    print("\n" + "═" * 72 + "\n  PARTE 1 · INSTRUMENTOS CORRIGIDOS\n" + "═" * 72)
    rng = np.random.default_rng(0)
    print("\n  Critério de parada: antigo (Coh ≥ SEAL) vs corrigido (Coh_rel ≥ SEAL)")
    print(f"  {'dim':>4} {'Coh*(n)':>8} {'ciclos antigo':>14} {'ciclos novo':>12} {'info preservada (corr)':>24}")
    for n in [8, 13, 16, 21, 34, 55]:
        v = rng.standard_normal(n)
        vn, cn = mpap_ancorado(v)
        corr = np.corrcoef(np.abs(v), np.abs(vn))[0, 1]
        print(f"  {n:>4} {coh_star(n):>8.4f} {mpap_antigo(v):>14} {cn:>12} {corr:>24.3f}")

    n, N = 64, 500
    t = np.linspace(0, 8 * np.pi, n)
    serial = np.sin(t * PHI) + 0.3 * np.sin(t * PHI ** 2)
    sinais = {
        "ruído branco":            rng.standard_normal((N, n)),
        "serial φ (seno)":         np.tile(serial, (N, 1)) + 0.05 * rng.standard_normal((N, n)),
        "geométrico r=0.5 (não-φ)": rng.permuted(np.tile(0.5 ** np.arange(n), (N, 1)), axis=1) * (1 + 0.05 * rng.standard_normal((N, n))),
        "lei de potência 1/k":     rng.permuted(np.tile(1 / np.arange(1, n + 1), (N, 1)), axis=1),
        "perfil φ + 5% ruído":     rng.permuted(np.tile(perfil_phi(n), (N, 1)), axis=1) * (1 + 0.05 * rng.standard_normal((N, n))),
    }
    print("\n  Detecção (D_φ: 0 = perfil φ, 1 = máximo distante) — um instrumento que pode dizer NÃO")
    print(f"  {'sinal':<28} {'Coh antigo':>10} {'Coh_rel':>8} {'D_φ':>8}")
    for k, s in sinais.items():
        print(f"  {k:<28} {coh(s).mean():>10.4f} {coh_rel(s).mean():>8.4f} {d_phi(s).mean():>8.4f}")

# ── DADOS ─────────────────────────────────────────────────────────────────────

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr = torch.tensor(sc.transform(Xtr), dtype=torch.float32); ytr = torch.tensor(ytr)
Xte = torch.tensor(sc.transform(Xte), dtype=torch.float32); yte = torch.tensor(yte)

# ── REDE GENÉRICA (todas as variantes saem daqui) ─────────────────────────────

class Rede(nn.Module):
    def __init__(s, dims, act="silu", ln=True, gains=None, gamma0=None, d_in=64, n_cls=10):
        super().__init__()
        s.act = F.silu if act == "silu" else F.relu
        s.gamma0 = gamma0
        s.lins, s.norms, s.skips = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        s.gammas = nn.ParameterList()
        d = d_in
        for i, h in enumerate(dims):
            lin = nn.Linear(d, h)
            nn.init.xavier_uniform_(lin.weight, gain=1.0 if gains is None else gains[i])
            nn.init.zeros_(lin.bias)
            s.lins.append(lin)
            s.norms.append(nn.LayerNorm(h) if ln else nn.Identity())
            if gamma0 is not None:
                sk = nn.Linear(d, h, bias=False); nn.init.orthogonal_(sk.weight)
                s.skips.append(sk)
                s.gammas.append(nn.Parameter(torch.full((h,), float(gamma0))))
            d = h
        s.head = nn.Linear(d, n_cls)

    def forward(s, x):
        for i, (lin, nm) in enumerate(zip(s.lins, s.norms)):
            f = nm(s.act(lin(x)))
            x = s.skips[i](x) + s.gammas[i] * f if s.gamma0 is not None else f
        return s.head(x), x

def g_phi(k):  return [1 / PHI ** (i + 1) for i in range(k)]

def d_phi_t(h):
    n = h.shape[-1]
    m = torch.sort(h.abs() + 1e-8, dim=-1, descending=True).values
    p = m / m.sum(-1, keepdim=True)
    q = torch.tensor(perfil_phi(n), dtype=torch.float32).expand_as(p)
    mm = 0.5 * (p + q)
    js = 0.5 * (p * torch.log(p / mm)).sum(-1) + 0.5 * (q * torch.log(q / mm)).sum(-1)
    return js / np.log(2)

@torch.no_grad()
def avaliar(model, Xs=None, ruido=0.0, seed=0):
    model.eval()
    Xs = Xte if Xs is None else Xs
    if ruido > 0:
        Xs = Xs + ruido * torch.randn(Xs.shape, generator=torch.Generator().manual_seed(seed))
    out, h = model(Xs)
    return F.cross_entropy(out, yte).item(), (out.argmax(1) == yte).float().mean().item(), out, h

def treinar(model, seed, lr=1e-3, reg=0.0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    loss0 = avaliar(model)[0]
    ep95 = None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            out, h = model(Xtr[idx])
            loss = F.cross_entropy(out, ytr[idx])
            if reg > 0:
                loss = loss + reg * d_phi_t(h).mean()
            if not torch.isfinite(loss):
                return dict(loss0=loss0, acc=float("nan"), ep95=None, divergiu=True)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep95 is None and avaliar(model)[1] >= 0.95:
            ep95 = ep + 1
    _, acc, _, _ = avaliar(model)
    return dict(loss0=loss0, acc=acc, ep95=ep95, divergiu=False)

def ms(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return f"{np.mean(xs):.4f}±{np.std(xs):.4f}" if xs else "—"

def rodar(nome, fabrica, **kw):
    rs = []
    for s in SEEDS:
        torch.manual_seed(s)
        rs.append(treinar(fabrica(), s, **kw))
    ep = [r["ep95"] for r in rs]
    return dict(nome=nome, loss0=ms([r["loss0"] for r in rs]), acc=ms([r["acc"] for r in rs]),
                ep95=ms([e for e in ep]) if any(e is not None for e in ep) else "—",
                div=sum(r["divergiu"] for r in rs))

def tabela(linhas):
    print(f"\n  {'variante':<36} {'loss época 0':>16} {'acc final':>16} {'épocas→95%':>14} {'diverg':>6}")
    for r in linhas:
        print(f"  {r['nome']:<36} {r['loss0']:>16} {r['acc']:>16} {r['ep95']:>14} {r['div']:>6}")

# ── PARTE 2 · ABLAÇÃO DO φ-INIT ───────────────────────────────────────────────

def parte2():
    print("\n" + "═" * 72 + "\n  PARTE 2 · ABLAÇÃO — o que causa a vantagem na época 0?\n" + "═" * 72)
    F4 = [55, 34, 21, 13]
    V = [
        ("Conv (64-32-16, ReLU, Xavier)",       lambda: Rede([64, 32, 16], "relu", False)),
        ("AP completa (Fib, SiLU, LN, 1/φ^k)",  lambda: Rede(F4, gains=g_phi(4))),
        ("AP sem φ-init (gain 1)",              lambda: Rede(F4)),
        ("AP gain constante 0.618",             lambda: Rede(F4, gains=[SEAL] * 4)),
        ("AP gain 0.5^k (decaimento não-φ)",    lambda: Rede(F4, gains=[0.5 ** (i + 1) for i in range(4)])),
        ("AP dims não-Fib (56-32-20-12)",       lambda: Rede([56, 32, 20, 12], gains=g_phi(4))),
        ("Conv + φ-init",                       lambda: Rede([64, 32, 16], "relu", False, gains=g_phi(3))),
    ]
    tabela([rodar(n, f) for n, f in V])
    print("\n  Leitura: se 'gain 0.618' e 'gain 0.5^k' empatam com 1/φ^k, a vantagem é de ESCALA, não de φ.")

# ── PARTE 3 · α COMO ÂNCORA ───────────────────────────────────────────────────

def parte3():
    print("\n" + "═" * 72 + "\n  PARTE 3 · α COMO ÂNCORA — residual x' = P·x + γ·f(x), γ0 variável\n" + "═" * 72)
    D6 = [89, 55, 34, 21, 13, 8]
    for lr in [1e-3, 1e-2]:
        print(f"\n  lr = {lr}  (rede profunda, 6 camadas Fibonacci)")
        linhas = [rodar("sem âncora (AP pura)", lambda: Rede(D6, gains=g_phi(6)), lr=lr)]
        for nome, g0 in [("γ0 = 1.0", 1.0), ("γ0 = 0.1", 0.1), ("γ0 = 0.01", 0.01),
                         ("γ0 = α = 1/137", ALPHA), ("γ0 = 0.001", 0.001)]:
            linhas.append(rodar(nome, lambda g0=g0: Rede(D6, gains=g_phi(6), gamma0=g0), lr=lr))
        tabela(linhas)
    print("\n  Leitura: o papel de âncora vale se γ0 pequeno ≥ sem âncora.")
    print("  O valor 1/137 especificamente só se sustenta se superar 0.01 e 0.001 — o que não se espera.")

# ── PARTE 4 · MPAP COMO PRESSÃO DE CAMPO ──────────────────────────────────────

def ece(probs, y, bins=10):
    conf, pred = probs.max(1)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.float().mean().item() * abs((pred[m] == y[m]).float().mean().item() - conf[m].mean().item())
    return e

def parte4():
    print("\n" + "═" * 72 + "\n  PARTE 4 · MPAP — pressão φ no treino e na saída\n" + "═" * 72)
    F4 = [55, 34, 21, 13]
    print(f"\n  {'regularização λ·D_φ':<22} {'acc limpo':>16} {'acc σ=0.5':>16} {'acc σ=1.0':>16} {'D_φ oculta':>16}")
    for lam in [0.0, 0.1, 1.0]:
        res = {k: [] for k in ["a0", "a5", "a10", "d"]}
        for s in SEEDS:
            torch.manual_seed(s)
            m = Rede(F4, gains=g_phi(4))
            treinar(m, s, reg=lam)
            _, a0, _, h = avaliar(m)
            res["a0"].append(a0); res["d"].append(d_phi_t(h).mean().item())
            res["a5"].append(avaliar(m, ruido=0.5, seed=s)[1])
            res["a10"].append(avaliar(m, ruido=1.0, seed=s)[1])
        print(f"  λ = {lam:<18} {ms(res['a0']):>16} {ms(res['a5']):>16} {ms(res['a10']):>16} {ms(res['d']):>16}")

    print("\n  MPAP sobre a saída (probabilidades), seed 0:")
    torch.manual_seed(0)
    m = Rede(F4, gains=g_phi(4)); treinar(m, 0)
    _, acc, out, _ = avaliar(m)
    p = F.softmax(out, 1)
    pm = torch.tensor(np.stack([np.abs(projecao_phi(v)) for v in p.numpy()]), dtype=torch.float32)
    print(f"    sem MPAP: acc={acc:.4f}  confiança média={p.max(1).values.mean():.4f}  ECE={ece(p, yte):.4f}")
    print(f"    com MPAP: acc={(pm.argmax(1) == yte).float().mean():.4f}  confiança média={pm.max(1).values.mean():.4f}  ECE={ece(pm, yte):.4f}")

    print("\n  Detecção na rede treinada (sem regularização): a AP se organiza sozinha em φ?")
    for nome, fab in [("Conv", lambda: Rede([64, 32, 16], "relu", False)), ("AP", lambda: Rede(F4, gains=g_phi(4)))]:
        torch.manual_seed(0); m = fab()
        d_antes = d_phi_t(avaliar(m)[3]).mean().item()
        treinar(m, 0)
        d_depois = d_phi_t(avaliar(m)[3]).mean().item()
        print(f"    {nome:<5} D_φ última camada: antes {d_antes:.4f} → depois {d_depois:.4f}")

if __name__ == "__main__":
    torch.set_num_threads(4)
    parte1(); parte2(); parte3(); parte4()
