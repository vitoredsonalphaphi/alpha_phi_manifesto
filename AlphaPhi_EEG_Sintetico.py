# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_EEG_Sintetico.py
Vitor Edson Delavi · Florianópolis · 2026

Pergunta:
    O eco ressonante fonônico funciona no domínio de frequências neurológicas?
    O campo coletivo detecta coerência espectral em sinais tipo-EEG?

Contexto:
    A visão original do projeto propunha um campo computacional ressonante
    que detecta estrutura por coerência — análogo ao sensor de porta automática,
    que não mede o objeto diretamente, mas mede a perturbação que ele causa
    no campo eletromagnético.

    EEG humano tem estrutura espectral mensurável:
        Alpha  (8-13 Hz): estado relaxado, olhos fechados — alta coerência
        Beta  (13-30 Hz): atenção, foco ativo — coerência moderada
        Theta  (4-8 Hz):  sonolência, meditação — coerência intermediária
        Noise (broadband): artefato, atividade de fundo — baixa coerência

    Se o eco fonônico discrimina Alpha de Noise no domínio EEG:
        o princípio de coerência espectral coletiva opera em frequências neurológicas.

    Se também discrimina Alpha de Beta:
        o campo detecta diferença de estado cognitivo por ressonância espectral —
        sem eletrodo implantado, sem instrução direta ao processador.

Substrato:
    EEG sintético realista — 1 segundo a 256 Hz por época.
    Frequências, amplitudes e SNR compatíveis com EEG humano real.

Tarefas:
    Tarefa 1: Alpha vs. Noise  — análogo direto ao teste original
    Tarefa 2: Alpha vs. Beta   — classificação de estado cognitivo

Protocolo: seeds por timestamp, resultados integralmente reportados.
"""

import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils_phi import PHI, ALPHA, C_PHI, golden_activation, clip_grad, sigmoid

# ── Parâmetros ────────────────────────────────────────────────────────────────
N_SEEDS  = 20
N_EPOCHS = 60
N_TRAIN  = 400
N_TEST   = 100
FS       = 256        # Hz — taxa de amostragem EEG padrão
DIM      = 256        # 1 segundo de sinal
HIDDEN   = 89
LR       = 0.01
N_ECO    = 3
K_MIN    = np.sqrt(2)

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print("Experimento: Eco Ressonante Fonônico — substrato EEG sintético")
print(f"fs={FS} Hz | época={DIM/FS:.2f}s | resolução FFT={FS/DIM:.2f} Hz/bin")
print(f"Timestamp: {TIMESTAMP}\n")

# ── Geração de sinais EEG sintéticos ─────────────────────────────────────────

def gerar_alpha(n, rng):
    """
    Estado Alpha (8-13 Hz): relaxamento, olhos fechados.
    Alta coerência espectral — pico dominante na banda alpha.
    SNR realista: amplitude do pico 3-10× o ruído de fundo.
    """
    t = np.arange(DIM) / FS
    X = np.zeros((n, DIM))
    for i in range(n):
        freq_dom = rng.uniform(8.0, 13.0)      # frequência dominante alpha
        amp_dom  = rng.uniform(20.0, 50.0)     # µV — amplitude típica alpha

        sinal = amp_dom * np.sin(2*np.pi*freq_dom*t + rng.uniform(0, 2*np.pi))

        # Harmônicos fisiológicos menores
        for mult in [0.5, 2.0, 3.0]:
            f_h = freq_dom * mult
            if 1.0 < f_h < FS/2:
                a_h = amp_dom * rng.uniform(0.05, 0.20)
                sinal += a_h * np.sin(2*np.pi*f_h*t + rng.uniform(0, 2*np.pi))

        # Ruído de fundo EEG (1/f + branco) — SNR realista
        ruido_amp = amp_dom * rng.uniform(0.15, 0.35)
        freqs_ruido = np.fft.rfftfreq(DIM, 1/FS)
        espectro_pink = np.zeros(len(freqs_ruido), dtype=complex)
        for fi in range(1, len(freqs_ruido)):
            espectro_pink[fi] = (rng.normal() + 1j*rng.normal()) / np.sqrt(freqs_ruido[fi])
        ruido_pink = np.fft.irfft(espectro_pink, n=DIM)
        ruido_pink = ruido_pink / (np.std(ruido_pink) + 1e-8) * ruido_amp

        sinal = sinal + ruido_pink
        X[i] = sinal / (np.std(sinal) + 1e-8)
    return X

def gerar_beta(n, rng):
    """
    Estado Beta (13-30 Hz): atenção ativa, foco cognitivo.
    Coerência moderada — energia distribuída na banda beta.
    Amplitude menor que alpha (fisiologicamente correto).
    """
    t = np.arange(DIM) / FS
    X = np.zeros((n, DIM))
    for i in range(n):
        # Beta: múltiplos componentes na banda (menos coerente que alpha)
        n_componentes = rng.integers(2, 5)
        sinal = np.zeros(DIM)
        for _ in range(n_componentes):
            freq_b = rng.uniform(13.0, 30.0)
            amp_b  = rng.uniform(5.0, 20.0)    # µV — menor que alpha
            sinal += amp_b * np.sin(2*np.pi*freq_b*t + rng.uniform(0, 2*np.pi))

        # Ruído de fundo maior que no alpha
        ruido_amp = np.std(sinal) * rng.uniform(0.3, 0.6)
        freqs_ruido = np.fft.rfftfreq(DIM, 1/FS)
        espectro_pink = np.zeros(len(freqs_ruido), dtype=complex)
        for fi in range(1, len(freqs_ruido)):
            espectro_pink[fi] = (rng.normal() + 1j*rng.normal()) / np.sqrt(freqs_ruido[fi])
        ruido_pink = np.fft.irfft(espectro_pink, n=DIM)
        ruido_pink = ruido_pink / (np.std(ruido_pink) + 1e-8) * ruido_amp

        sinal = sinal + ruido_pink
        X[i] = sinal / (np.std(sinal) + 1e-8)
    return X

def gerar_noise_eeg(n, rng):
    """
    Ruído de fundo EEG: sem estado dominante, sem coerência espectral.
    Mistura de 1/f e ruído branco — artefato de linha de base.
    """
    X = np.zeros((n, DIM))
    for i in range(n):
        freqs = np.fft.rfftfreq(DIM, 1/FS)
        espectro = np.zeros(len(freqs), dtype=complex)
        for fi in range(1, len(freqs)):
            espectro[fi] = (rng.normal() + 1j*rng.normal()) / np.sqrt(freqs[fi])
        sinal = np.fft.irfft(espectro, n=DIM)
        sinal += rng.normal(0, 0.3, DIM)
        X[i] = sinal / (np.std(sinal) + 1e-8)
    return X

def gerar_dados(seed, tarefa="alpha_vs_noise"):
    rng = np.random.default_rng(seed)
    n_tr, n_te = N_TRAIN // 2, N_TEST // 2

    if tarefa == "alpha_vs_noise":
        Xtr1, Xte1 = gerar_alpha(n_tr, rng), gerar_alpha(n_te, rng)
        Xtr0, Xte0 = gerar_noise_eeg(n_tr, rng), gerar_noise_eeg(n_te, rng)
    else:  # alpha_vs_beta
        Xtr1, Xte1 = gerar_alpha(n_tr, rng), gerar_alpha(n_te, rng)
        Xtr0, Xte0 = gerar_beta(n_tr, rng), gerar_beta(n_te, rng)

    X_tr = np.vstack([Xtr1, Xtr0])
    y_tr = np.array([1]*n_tr + [0]*n_tr, dtype=float)
    X_te = np.vstack([Xte1, Xte0])
    y_te = np.array([1]*n_te + [0]*n_te, dtype=float)

    idx_tr = rng.permutation(N_TRAIN)
    idx_te = rng.permutation(N_TEST)
    return X_tr[idx_tr], y_tr[idx_tr], X_te[idx_te], y_te[idx_te]

# ── Campo coletivo + Eco ───────────────────────────────────────────────────────

def medir_campo(X):
    freq_batch = np.fft.fft(X, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)
    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia   = -np.sum(amp_norm * np.log(amp_norm))
    coh_campo  = float(1.0 - entropia / np.log(X.shape[-1]))
    k          = K_MIN + (PHI - K_MIN) * coh_campo
    return k, coh_campo

def eco_v1(X):
    X = np.asarray(X, dtype=float)
    k, _ = medir_campo(X)
    s = X.copy()
    for _ in range(N_ECO):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) / k
    return s, k

def eco_v2(X):
    X = np.asarray(X, dtype=float)
    k, _ = medir_campo(X)
    s = X.copy()
    for _ in range(N_ECO):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) * PHI
    return s, k

# ── Rede neural ───────────────────────────────────────────────────────────────

def treinar(X_tr, y_tr, X_te, y_te, seed):
    dim_in = X_tr.shape[1]
    rng = np.random.default_rng(seed + 999)
    W1 = rng.normal(0, np.sqrt(2.0/dim_in),  (dim_in, HIDDEN))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, np.sqrt(2.0/HIDDEN), (HIDDEN, 1))
    b2 = np.zeros(1)
    for _ in range(N_EPOCHS):
        idx = rng.permutation(len(X_tr))
        for i in range(0, len(X_tr) - 31, 32):
            Xb, yb = X_tr[idx[i:i+32]], y_tr[idx[i:i+32]]
            z1 = Xb @ W1 + b1
            a1 = golden_activation(z1)
            yh = sigmoid(a1 @ W2 + b2).squeeze()
            dL = (yh - yb) / len(yb)
            dW2 = a1.T @ dL.reshape(-1,1)
            db2 = dL.sum(keepdims=True)
            dz1 = (dL.reshape(-1,1) * W2.T) * (1 - np.tanh(z1/PHI)**2)
            W1 -= LR * clip_grad(Xb.T @ dz1)
            b1 -= LR * np.clip(dz1.sum(axis=0), -1, 1)
            W2 -= LR * clip_grad(dW2)
            b2 -= LR * np.clip(db2, -1, 1)
    yh_te = sigmoid(golden_activation(X_te @ W1 + b1) @ W2 + b2).squeeze()
    return float(np.mean((yh_te >= 0.5) == y_te))

# ── Execução ──────────────────────────────────────────────────────────────────

resultados = {}

for tarefa, label in [("alpha_vs_noise", "Alpha vs. Noise"),
                      ("alpha_vs_beta",  "Alpha vs. Beta")]:

    print(f"\n{'='*60}")
    print(f"Tarefa: {label}")
    print(f"{'='*60}")
    print(f"{'Seed':<14} {'G':>8} {'V1(1/k)':>10} {'V2(φ)':>10} {'k_campo':>10}")
    print("-"*56)

    res_G, res_v1, res_v2, ks = [], [], [], []

    for seed in SEEDS:
        X_tr, y_tr, X_te, y_te = gerar_dados(seed, tarefa)

        # Baseline — sem eco
        acc_G = treinar(X_tr, y_tr, X_te, y_te, seed)

        # V1 — coupling = 1/k
        Xtr_v1, k_tr = eco_v1(X_tr)
        Xte_v1, _    = eco_v1(X_te)
        acc_v1 = treinar(Xtr_v1, y_tr, Xte_v1, y_te, seed)

        # V2 — coupling = φ
        Xtr_v2, _ = eco_v2(X_tr)
        Xte_v2, _ = eco_v2(X_te)
        acc_v2 = treinar(Xtr_v2, y_tr, Xte_v2, y_te, seed)

        res_G.append(acc_G); res_v1.append(acc_v1)
        res_v2.append(acc_v2); ks.append(k_tr)

        print(f"{seed:<14} {acc_G:>8.3f} {acc_v1:>10.3f} {acc_v2:>10.3f} {k_tr:>10.4f}")

    G   = np.array(res_G)
    V1  = np.array(res_v1)
    V2  = np.array(res_v2)
    k_m = float(np.mean(ks))

    _, p_v1 = stats.wilcoxon(V1, G)
    _, p_v2 = stats.wilcoxon(V2, G)
    _, p_v2v1 = stats.wilcoxon(V2, V1)

    print(f"\n{'Modo':<12} {'Média':>8} {'Δ vs G':>8} {'p-valor':>12}")
    print(f"  {'G':<10} {G.mean():>8.4f}")
    print(f"  {'V1 (1/k)':<10} {V1.mean():>8.4f} {V1.mean()-G.mean():>+8.4f} {p_v1:>12.6f}")
    print(f"  {'V2 (φ)':<10} {V2.mean():>8.4f} {V2.mean()-G.mean():>+8.4f} {p_v2:>12.6f}")
    print(f"\n  k_campo médio: {k_m:.5f}  (√2={np.sqrt(2):.5f}, φ={PHI:.5f})")
    print(f"  V2 vs V1: Δ={V2.mean()-V1.mean():+.4f}, p={p_v2v1:.6f}")

    sinal_v2 = "✅" if V2.mean() > G.mean() and p_v2 < 0.05 else "⚠️"
    print(f"  Eco V2 supera baseline: {sinal_v2}")

    resultados[tarefa] = {
        "label": label,
        "G":  {"mean": float(G.mean()),  "values": G.tolist()},
        "V1": {"mean": float(V1.mean()), "values": V1.tolist(), "p_vs_G": float(p_v1)},
        "V2": {"mean": float(V2.mean()), "values": V2.tolist(), "p_vs_G": float(p_v2)},
        "k_medio": k_m,
        "delta_V2_G":  float(V2.mean() - G.mean()),
        "delta_V2_V1": float(V2.mean() - V1.mean()),
        "p_V2vsV1": float(p_v2v1),
    }

# ── Visualização ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#0d1117")

# Espectros dos sinais (seed fixo para visualização)
rng_vis = np.random.default_rng(42)
sig_alpha = gerar_alpha(10, rng_vis).mean(axis=0)
sig_beta  = gerar_beta(10, rng_vis).mean(axis=0)
sig_noise = gerar_noise_eeg(10, rng_vis).mean(axis=0)
freqs     = np.fft.rfftfreq(DIM, 1/FS)

ax = axes[0, 0]
ax.set_facecolor("#161b22")
ax.plot(freqs[:60], np.abs(np.fft.rfft(sig_alpha))[:60],
        color="#DAA520", label="Alpha (8-13 Hz)", linewidth=1.5)
ax.plot(freqs[:60], np.abs(np.fft.rfft(sig_beta))[:60],
        color="#4169E1", label="Beta (13-30 Hz)", linewidth=1.5)
ax.plot(freqs[:60], np.abs(np.fft.rfft(sig_noise))[:60],
        color="#888888", label="Noise", linewidth=1.0, alpha=0.7)
ax.axvspan(8, 13, alpha=0.15, color="#DAA520")
ax.axvspan(13, 30, alpha=0.10, color="#4169E1")
ax.set_title("Espectro EEG sintético (média 10 épocas)", color="#E6EDF3")
ax.set_xlabel("Frequência (Hz)", color="#8B949E")
ax.set_ylabel("Amplitude", color="#8B949E")
ax.tick_params(colors="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=9)
for spine in ax.spines.values(): spine.set_color("#30363d")

# Acurácia por seed — Tarefa 1
ax = axes[0, 1]
ax.set_facecolor("#161b22")
r = resultados["alpha_vs_noise"]
ax.plot(range(N_SEEDS), r["G"]["values"],  'o-', color="#888888", label="G (sem eco)", lw=1.2)
ax.plot(range(N_SEEDS), r["V1"]["values"], 's-', color="#4169E1", label="V1 (1/k)", lw=1.5)
ax.plot(range(N_SEEDS), r["V2"]["values"], '^-', color="#DAA520", label="V2 (φ)", lw=1.8)
ax.axhline(r["G"]["mean"],  color="#888888", linestyle=':', alpha=0.5)
ax.axhline(r["V2"]["mean"], color="#DAA520", linestyle='--', alpha=0.7)
ax.set_title(f"Alpha vs. Noise | V2={r['V2']['mean']:.4f}", color="#E6EDF3")
ax.set_xlabel("seed", color="#8B949E"); ax.set_ylabel("acurácia", color="#8B949E")
ax.tick_params(colors="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=9)
for spine in ax.spines.values(): spine.set_color("#30363d")

# Acurácia por seed — Tarefa 2
ax = axes[1, 0]
ax.set_facecolor("#161b22")
r = resultados["alpha_vs_beta"]
ax.plot(range(N_SEEDS), r["G"]["values"],  'o-', color="#888888", label="G (sem eco)", lw=1.2)
ax.plot(range(N_SEEDS), r["V1"]["values"], 's-', color="#4169E1", label="V1 (1/k)", lw=1.5)
ax.plot(range(N_SEEDS), r["V2"]["values"], '^-', color="#DAA520", label="V2 (φ)", lw=1.8)
ax.axhline(r["G"]["mean"],  color="#888888", linestyle=':', alpha=0.5)
ax.axhline(r["V2"]["mean"], color="#DAA520", linestyle='--', alpha=0.7)
ax.set_title(f"Alpha vs. Beta | V2={r['V2']['mean']:.4f}", color="#E6EDF3")
ax.set_xlabel("seed", color="#8B949E"); ax.set_ylabel("acurácia", color="#8B949E")
ax.tick_params(colors="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=9)
for spine in ax.spines.values(): spine.set_color("#30363d")

# Comparativo geral
ax = axes[1, 1]
ax.set_facecolor("#161b22")
tarefas  = ["Alpha\nvs Noise", "Alpha\nvs Beta"]
modos    = ["G", "V1", "V2"]
cores    = ["#888888", "#4169E1", "#DAA520"]
x        = np.arange(len(tarefas))
width    = 0.25

for mi, (modo, cor) in enumerate(zip(modos, cores)):
    vals = [resultados["alpha_vs_noise"][modo]["mean"],
            resultados["alpha_vs_beta"][modo]["mean"]]
    ax.bar(x + mi*width - width, vals, width, label=modo, color=cor, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(tarefas, color="#8B949E")
ax.set_title("Comparativo geral — EEG sintético", color="#E6EDF3")
ax.set_ylabel("acurácia média", color="#8B949E")
ax.tick_params(colors="#8B949E")
ax.legend(facecolor="#161b22", labelcolor="#E6EDF3", fontsize=9)
for spine in ax.spines.values(): spine.set_color("#30363d")

plt.suptitle(
    "Eco Ressonante Fonônico — substrato EEG sintético (256 Hz)\n"
    f"G (baseline) | V1 (coupling=1/k) | V2 (coupling=φ={PHI:.3f})",
    color="#E6EDF3", fontsize=11
)
plt.tight_layout()
plt.savefig("eeg_sintetico_results.png", dpi=150, bbox_inches='tight',
            facecolor="#0d1117")
plt.close()
print("\nVisualização salva: eeg_sintetico_results.png")

# ── Salvar ────────────────────────────────────────────────────────────────────

saida = {
    "experimento": "EEG_Sintetico",
    "substrato": "EEG sintético realista (256 Hz, 1s/época)",
    "tarefas": ["alpha_vs_noise", "alpha_vs_beta"],
    "constantes": {"PHI": PHI, "K_MIN": float(K_MIN), "C_PHI": C_PHI},
    "n_seeds": N_SEEDS, "n_epochs": N_EPOCHS,
    "timestamp": TIMESTAMP, "seeds": SEEDS,
    "resultados": resultados,
}

with open("eeg_sintetico_results.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

print("Dados salvos: eeg_sintetico_results.json")
