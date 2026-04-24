# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_MEG_Sample.py
Vitor Edson Delavi · Florianópolis · 2026

Hipótese:
    O eco ressonante fonônico falhou no EEG de 109 sujeitos porque o sinal
    neural foi destruído pelo crânio antes de chegar ao eletrodo.
    MEG mede o campo magnético — atravessa o crânio sem distorção.

    Se o eco funciona em MEG mas não em EEG para o mesmo tipo de tarefa,
    a fronteira está confirmada: não é o princípio que falha — é o substrato
    de captura que determina o que pode ser lido.

Dataset:
    MNE Sample Dataset — download automático via MNE
    1 sujeito · 306 canais MEG (102 magnetômetros + 204 gradiômetros)
    Tarefa: classificar resposta auditiva vs visual
    Referência: Hämäläinen et al. — Elekta/NEUROMAG

Diferença crítica vs EEG:
    EEG: campo elétrico atravessa crânio (osso atenua ~100x, difunde espacialmente)
    MEG: campo magnético atravessa crânio sem distorção — sinal orgânico preservado

INSTRUÇÕES PARA GOOGLE COLAB:
    1. !pip install -q mne
    2. Cole e rode — download automático (~1.5GB, cached)
"""

# ── Instalação (rodar antes no Colab) ─────────────────────────────────────────
# !pip install -q mne

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import json
import time
from scipy import stats

try:
    import mne
    mne.set_log_level('WARNING')
except ImportError:
    raise ImportError("Execute: !pip install -q mne")

# ── Constantes ────────────────────────────────────────────────────────────────
PHI   = (1 + np.sqrt(5)) / 2    # 1.6180...
K_MIN = np.sqrt(2)               # 1.4142...
N_ECO          = 3
N_SEEDS        = 20
N_EPOCHS_TRAIN = 60
HIDDEN         = 89
LR             = 0.01
FS_ALVO        = 256             # resample para 256 Hz — igual aos experimentos anteriores
T_MIN, T_MAX   = 0.0, 1.0       # janela: 0 a 1 segundo após estímulo
FILTRO         = (8, 30)         # Alpha + Beta — mesma banda dos experimentos EEG

TIMESTAMP     = int(time.time())
SEEDS         = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print("AlphaPhi MEG — MNE Sample Dataset")
print(f"PHI={PHI:.6f}  K_MIN(√2)={K_MIN:.6f}")
print(f"Banda: {FILTRO[0]}-{FILTRO[1]} Hz | Janela: {T_MIN}-{T_MAX}s | fs_alvo={FS_ALVO} Hz")
print(f"Seeds: {N_SEEDS} | Timestamp: {TIMESTAMP}\n")

# ── Campo coletivo ─────────────────────────────────────────────────────────────
def medir_campo(X):
    freq_batch = np.fft.fft(X, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)
    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia   = -np.sum(amp_norm * np.log(amp_norm))
    coh        = float(1.0 - entropia / np.log(X.shape[-1]))
    k          = K_MIN + (PHI - K_MIN) * coh
    return k, coh

def eco_v1(X):
    X = np.asarray(X, dtype=float)
    k, coh = medir_campo(X)
    s = X.copy()
    for _ in range(N_ECO):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) / k
    return s, k, coh

def eco_v2(X):
    X = np.asarray(X, dtype=float)
    k, coh = medir_campo(X)
    s = X.copy()
    for _ in range(N_ECO):
        freq     = np.fft.fft(s, axis=-1)
        reflexao = np.real(np.fft.ifft(
            np.abs(freq) * np.exp(1j * np.angle(freq) * k), axis=-1))
        s = s + (reflexao - X) * PHI
    return s, k, coh

def golden_activation(x):
    return PHI * np.tanh(x / PHI)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def clip_grad(g, max_norm=1.0):
    norm = np.linalg.norm(g)
    return g * (max_norm / norm) if norm > max_norm else g

def treinar(X_tr, y_tr, X_te, y_te, seed):
    dim_in = X_tr.shape[1]
    rng = np.random.default_rng(seed + 999)
    W1  = rng.normal(0, np.sqrt(2.0 / dim_in), (dim_in, HIDDEN))
    b1  = np.zeros(HIDDEN)
    W2  = rng.normal(0, np.sqrt(2.0 / HIDDEN), (HIDDEN, 1))
    b2  = np.zeros(1)
    for _ in range(N_EPOCHS_TRAIN):
        idx = rng.permutation(len(X_tr))
        for i in range(0, len(X_tr) - 15, 16):
            Xb, yb = X_tr[idx[i:i+16]], y_tr[idx[i:i+16]]
            z1  = Xb @ W1 + b1
            a1  = golden_activation(z1)
            yh  = sigmoid(a1 @ W2 + b2).squeeze()
            dL  = (yh - yb) / len(yb)
            dW2 = a1.T @ dL.reshape(-1, 1)
            db2 = dL.sum(keepdims=True)
            dz1 = (dL.reshape(-1, 1) * W2.T) * (1 - np.tanh(z1 / PHI) ** 2)
            W1 -= LR * clip_grad(Xb.T @ dz1)
            b1 -= LR * np.clip(dz1.sum(axis=0), -1, 1)
            W2 -= LR * clip_grad(dW2)
            b2 -= LR * np.clip(db2, -1, 1)
    yh_te = sigmoid(golden_activation(X_te @ W1 + b1) @ W2 + b2).squeeze()
    return float(np.mean((yh_te >= 0.5) == y_te))

# ── Carregar dados MEG ─────────────────────────────────────────────────────────
def carregar_meg():
    print("Baixando MNE Sample Dataset (download automático ~1.5GB)...")
    data_path = mne.datasets.sample.data_path()
    raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

    # Seleciona apenas magnetômetros (102 canais)
    # Magnetômetros = campo magnético direto — mais análogos ao EEG em 1 canal
    raw.pick_types(meg='mag', eeg=False, stim=False, verbose=False)

    # Filtro de banda — mesma banda dos experimentos anteriores
    raw.filter(FILTRO[0], FILTRO[1], fir_design='firwin', verbose=False)

    # Resample para 256 Hz
    raw.resample(FS_ALVO, verbose=False)

    print(f"  Canais MEG magnetômetros: {len(raw.ch_names)}")
    print(f"  fs após resample: {raw.info['sfreq']} Hz")

    # Eventos
    events = mne.find_events(raw, stim_channel='STI 014', verbose=False)

    # Auditivo: eventos 1, 2, 3 | Visual: eventos 4, 5
    event_id_aud = {'auditory/left': 1, 'auditory/right': 2}
    event_id_vis = {'visual/left':   3, 'visual/right':   4}
    event_id_all = {**event_id_aud, **event_id_vis}

    epochs = mne.Epochs(
        raw, events, event_id=event_id_all,
        tmin=T_MIN, tmax=T_MAX,
        baseline=None, preload=True, verbose=False
    )

    # Extrair dados: média dos 102 magnetômetros → 1 sinal por época
    # Média espacial preserva coerência temporal, reduz ruído espacial
    data   = epochs.get_data()          # (n_epocas, n_canais, n_tempos)
    labels = epochs.events[:, 2]        # códigos de evento

    # Média dos magnetômetros → (n_epocas, n_tempos)
    X = data.mean(axis=1)
    X = X / (np.std(X, axis=1, keepdims=True) + 1e-8)

    # Binário: auditivo=1, visual=0
    y = np.array([1 if lbl in [1, 2] else 0 for lbl in labels], dtype=float)

    n_aud = int(y.sum())
    n_vis = int((1 - y).sum())
    print(f"  Épocas auditivas: {n_aud} | visuais: {n_vis}")
    print(f"  Dimensão por época: {X.shape[1]} pontos\n")

    # Balancear classes
    n_min = min(n_aud, n_vis)
    idx_aud = np.where(y == 1)[0][:n_min]
    idx_vis = np.where(y == 0)[0][:n_min]
    idx_bal = np.concatenate([idx_aud, idx_vis])

    return X[idx_bal], y[idx_bal], n_min

# ── Experimento principal ──────────────────────────────────────────────────────
def rodar():
    X, y, n_min = carregar_meg()

    n_tr  = int(0.7 * len(X))
    res_G, res_v1, res_v2, ks = [], [], [], []

    print(f"{'Seed':<8} {'G':>7} {'V1':>9} {'V2(φ)':>9} {'Δ':>8} {'k':>9}")
    print("─" * 56)

    for seed in SEEDS:
        rng      = np.random.default_rng(seed)
        idx      = rng.permutation(len(X))
        X_sh, y_sh = X[idx], y[idx]
        X_tr, y_tr = X_sh[:n_tr], y_sh[:n_tr]
        X_te, y_te = X_sh[n_tr:], y_sh[n_tr:]

        acc_G = treinar(X_tr, y_tr, X_te, y_te, seed)

        Xtr_v1, k_tr, _ = eco_v1(X_tr)
        Xte_v1,  _,  _  = eco_v1(X_te)
        acc_v1 = treinar(Xtr_v1, y_tr, Xte_v1, y_te, seed)

        Xtr_v2, _, _ = eco_v2(X_tr)
        Xte_v2, _, _ = eco_v2(X_te)
        acc_v2 = treinar(Xtr_v2, y_tr, Xte_v2, y_te, seed)

        res_G.append(acc_G)
        res_v1.append(acc_v1)
        res_v2.append(acc_v2)
        ks.append(k_tr)

        delta = acc_v2 - acc_G
        sig   = "✓" if delta > 0 else "✗"
        print(f"{seed % 100000:<8} {acc_G:>7.3f} {acc_v1:>9.3f} {acc_v2:>9.3f}"
              f" {delta:>+8.3f} {k_tr:>9.5f} {sig}")

    G  = np.array(res_G)
    V1 = np.array(res_v1)
    V2 = np.array(res_v2)
    k_m = float(np.mean(ks))

    _, p_v1   = stats.wilcoxon(V1, G)
    _, p_v2   = stats.wilcoxon(V2, G)
    _, p_v2v1 = stats.wilcoxon(V2, V1)

    n_pos = int(np.sum(V2 > G))

    print(f"\n{'═'*60}")
    print(f"  AlphaPhi MEG — MNE Sample · Auditivo vs Visual")
    print(f"  Magnetômetros (102 ch, média espacial) · {FILTRO[0]}-{FILTRO[1]} Hz")
    print(f"{'═'*60}")
    print(f"  Épocas por classe:  {n_min}")
    print(f"{'─'*60}")
    print(f"  G  (baseline): {G.mean():.4f}  ±{G.std():.4f}")
    print(f"  V1 (√2+campo): {V1.mean():.4f}  ±{V1.std():.4f}"
          f"  Δ={V1.mean()-G.mean():+.4f}  p={p_v1:.6f}")
    print(f"  V2 (φ):        {V2.mean():.4f}  ±{V2.std():.4f}"
          f"  Δ={V2.mean()-G.mean():+.4f}  p={p_v2:.6f}")
    print(f"{'─'*60}")
    print(f"  V2 > baseline:  {n_pos}/{N_SEEDS} seeds")
    print(f"  V2 vs V1: Δ={V2.mean()-V1.mean():+.4f}  p={p_v2v1:.6f}")
    print(f"  k_campo médio: {k_m:.5f}  (√2={np.sqrt(2):.5f}  φ={PHI:.5f})")

    conclusao = "CONFIRMA" if (V2.mean() > G.mean() and p_v2 < 0.05) else "NAO CONFIRMA"
    print(f"\n  Hipótese eco em MEG → {conclusao}")
    print(f"{'═'*60}\n")

    resultado = {
        "experimento":  "AlphaPhi_MEG_Sample",
        "substrato":    "MEG magnetômetros 102ch média espacial",
        "tarefa":       "auditivo vs visual",
        "filtro_hz":    list(FILTRO),
        "timestamp":    TIMESTAMP,
        "n_epocas_por_classe": n_min,
        "G_mean":       float(G.mean()),
        "V1_mean":      float(V1.mean()),
        "V2_mean":      float(V2.mean()),
        "delta_V2_G":   float(V2.mean() - G.mean()),
        "p_v2":         float(p_v2),
        "p_v1":         float(p_v1),
        "p_v2_vs_v1":   float(p_v2v1),
        "k_medio":      float(k_m),
        "conclusao":    conclusao,
    }

    with open("meg_sample_results.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("  Salvo: meg_sample_results.json")

    return resultado

rodar()
