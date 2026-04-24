# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_MEG_Frames.py
Vitor Edson Delavi · Florianópolis · 2026

Hipótese — Eco de Frames:
    O eco global lê o batch como campo único — uma entropia, um k, uma
    transformação igual para todos os pontos do sinal.

    Sinais neurais têm coerência que EVOLUI no tempo: ondas alpha que surgem,
    crescem e dissipam. Um sinal que começa caótico e termina coerente é
    indistinguível do inverso para o eco global.

    O eco de frames divide cada sinal em janelas temporais. Cada frame tem
    sua própria entropia e seu próprio k. A progressão k₁→k₂→k₃→k₄ é uma
    assinatura temporal de coerência — única para cada tipo de sinal.

    Hipótese: a progressão de k ao longo dos frames difere entre auditivo
    e visual no MEG, onde o eco global não encontrou diferença.

Novidade:
    - eco_frames: eco aplicado frame a frame, k independente por janela
    - vetor ks como feature adicional (assinatura temporal)
    - comparação: eco_global vs eco_frames vs baseline

Dataset:
    MNE Sample Dataset — download automático via MNE
    Canal de máxima variância | Auditivo vs Visual

INSTRUÇÕES PARA GOOGLE COLAB:
    !pip install -q mne
"""

# !pip install -q mne

import numpy as np
import json
import time
from scipy import stats
import mne
mne.set_log_level('WARNING')

# ── Constantes ────────────────────────────────────────────────────────────────
PHI   = (1 + np.sqrt(5)) / 2
K_MIN = np.sqrt(2)
N_ECO          = 3
N_SEEDS        = 20
N_EPOCHS_TRAIN = 200
LR             = 0.05
FILTRO         = (8, 30)
FS_ALVO        = 256
N_FRAMES       = 4      # divisão temporal do sinal

TIMESTAMP = int(time.time())
SEEDS     = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print("AlphaPhi MEG — Eco de Frames")
print(f"PHI={PHI:.6f}  K_MIN(√2)={K_MIN:.6f}")
print(f"Frames: {N_FRAMES} | Timestamp: {TIMESTAMP}\n")

# ── Campo e eco ───────────────────────────────────────────────────────────────
def medir_campo(X):
    f = np.fft.fft(X, axis=-1)
    a = np.abs(f).mean(axis=0)
    a = np.clip(a / (a.sum() + 1e-8), 1e-10, 1.0)
    e = -np.sum(a * np.log(a))
    c = float(1.0 - e / np.log(X.shape[-1]))
    return K_MIN + (PHI - K_MIN) * c, c

def eco_global(X):
    X = np.asarray(X, dtype=float)
    k, _ = medir_campo(X)
    s = X.copy()
    for _ in range(N_ECO):
        f = np.fft.fft(s, axis=-1)
        r = np.real(np.fft.ifft(np.abs(f) * np.exp(1j * np.angle(f) * k), axis=-1))
        s = s + (r - X) * PHI
    return s, k, [k] * N_FRAMES

def eco_frames(X, n_frames=N_FRAMES):
    """
    Eco aplicado frame a frame.
    Cada janela temporal tem sua própria leitura de campo (k independente).
    Retorna: sinal transformado + k_global (média) + lista de k por frame.
    """
    X  = np.asarray(X, dtype=float)
    N  = X.shape[-1]
    sz = N // n_frames
    s  = X.copy()
    ks = []

    for i in range(n_frames):
        ini = i * sz
        fim = ini + sz if i < n_frames - 1 else N   # último frame pega o resto
        trecho = s[:, ini:fim]
        k, _   = medir_campo(trecho)
        for _ in range(N_ECO):
            f  = np.fft.fft(trecho, axis=-1)
            r  = np.real(np.fft.ifft(
                     np.abs(f) * np.exp(1j * np.angle(f) * k), axis=-1))
            trecho = trecho + (r - X[:, ini:fim]) * PHI
        s[:, ini:fim] = trecho
        ks.append(k)

    k_medio = float(np.mean(ks))
    return s, k_medio, ks

# ── Features ──────────────────────────────────────────────────────────────────
def band_power(X, fs=256, band=(8, 13)):
    freqs = np.fft.rfftfreq(X.shape[-1], 1/fs)
    idx   = (freqs >= band[0]) & (freqs <= band[1])
    fft   = np.fft.rfft(X, axis=-1)
    return np.mean(np.abs(fft[:, idx]) ** 2, axis=-1, keepdims=True)

def extrair_features_base(X, fs=256):
    alpha = band_power(X, fs, (8, 13))
    beta  = band_power(X, fs, (13, 30))
    F     = np.log1p(np.hstack([alpha, beta]))
    return (F - F.mean(0)) / (F.std(0) + 1e-8)

def extrair_features_frames(X, ks_por_amostra, fs=256):
    """
    Features base (alpha+beta) + assinatura temporal de k por frame.
    ks_por_amostra: lista de listas — cada amostra tem N_FRAMES valores de k.
    """
    F_base = extrair_features_base(X, fs)
    K_sig  = np.array(ks_por_amostra)   # (n_amostras, n_frames)
    K_sig  = (K_sig - K_sig.mean(0)) / (K_sig.std(0) + 1e-8)
    return np.hstack([F_base, K_sig])

# ── Classificador ─────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def treinar_lr(X_tr, y_tr, X_te, y_te, seed):
    rng = np.random.default_rng(seed + 999)
    w   = rng.normal(0, 0.01, (X_tr.shape[1], 1))
    b   = np.zeros(1)
    for _ in range(N_EPOCHS_TRAIN):
        yh  = sigmoid(X_tr @ w + b).squeeze()
        dL  = (yh - y_tr) / len(y_tr)
        w  -= LR * (X_tr.T @ dL.reshape(-1, 1))
        b  -= LR * dL.mean()
    return float(np.mean((sigmoid(X_te @ w + b).squeeze() >= 0.5) == y_te))

# ── Dados ─────────────────────────────────────────────────────────────────────
def carregar_meg():
    print("Carregando MNE Sample Dataset...")
    dp  = mne.datasets.sample.data_path()
    raw = mne.io.read_raw_fif(dp / 'MEG' / 'sample' / 'sample_audvis_raw.fif',
                               preload=True, verbose=False)
    events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
    raw.pick_types(meg='mag', eeg=False, stim=False, verbose=False)
    raw.filter(FILTRO[0], FILTRO[1], fir_design='firwin', verbose=False)
    raw.resample(FS_ALVO, verbose=False)

    epochs = mne.Epochs(raw, events,
                        event_id={'aud/L': 1, 'aud/R': 2, 'vis/L': 3, 'vis/R': 4},
                        tmin=0.0, tmax=0.5,
                        baseline=None, preload=True, verbose=False)
    data   = epochs.get_data()
    labels = epochs.events[:, 2]

    ch_var  = data.var(axis=(0, 2))
    best_ch = int(np.argmax(ch_var))
    print(f"  Canal: {raw.ch_names[best_ch]} | fs: {FS_ALVO} Hz")

    X = data[:, best_ch, :]
    X = X / (np.std(X, axis=1, keepdims=True) + 1e-8)
    y = np.array([1 if l in [1, 2] else 0 for l in labels], dtype=float)

    n_min   = min(int(y.sum()), int((1 - y).sum()))
    idx_bal = np.concatenate([np.where(y == 1)[0][:n_min],
                               np.where(y == 0)[0][:n_min]])
    print(f"  Épocas/classe: {n_min} | dim: {X.shape[1]} | frames: {N_FRAMES}\n")
    return X[idx_bal], y[idx_bal], n_min, raw.ch_names[best_ch]

# ── Pipeline ──────────────────────────────────────────────────────────────────
def rodar():
    X, y, n_min, ch = carregar_meg()
    n_tr = int(0.7 * len(X))

    res_G, res_EG, res_EF = [], [], []
    ks_global_list, ks_frames_list = [], []

    print(f"{'Seed':<8} {'G':>7} {'EcoGlob':>9} {'EcoFrames':>11} {'ΔFrames':>9} {'k_glob':>8}")
    print("─" * 60)

    for seed in SEEDS:
        rng        = np.random.default_rng(seed)
        idx        = rng.permutation(len(X))
        X_sh, y_sh = X[idx], y[idx]
        X_tr, y_tr = X_sh[:n_tr], y_sh[:n_tr]
        X_te, y_te = X_sh[n_tr:], y_sh[n_tr:]

        # ── Baseline ──────────────────────────────────────────────────────────
        Fg_tr = extrair_features_base(X_tr)
        Fg_te = extrair_features_base(X_te)
        acc_G = treinar_lr(Fg_tr, y_tr, Fg_te, y_te, seed)

        # ── Eco global ────────────────────────────────────────────────────────
        Xtr_eg, k_glob, _ = eco_global(X_tr)
        Xte_eg, _, _      = eco_global(X_te)
        Feg_tr = extrair_features_base(Xtr_eg)
        Feg_te = extrair_features_base(Xte_eg)
        acc_EG = treinar_lr(Feg_tr, y_tr, Feg_te, y_te, seed)

        # ── Eco frames ────────────────────────────────────────────────────────
        Xtr_ef, _, ks_tr = eco_frames(X_tr)
        Xte_ef, _, ks_te = eco_frames(X_te)

        # assinatura de k por amostra: cada amostra no batch teve o mesmo
        # k por frame (campo coletivo), mas queremos a progressão como feature
        ks_tr_por_amostra = [ks_tr] * len(X_tr)   # mesmo campo p/ todo o batch
        ks_te_por_amostra = [ks_tr] * len(X_te)   # usa k do treino p/ teste

        Fef_tr = extrair_features_frames(Xtr_ef, ks_tr_por_amostra)
        Fef_te = extrair_features_frames(Xte_ef, ks_te_por_amostra)
        acc_EF = treinar_lr(Fef_tr, y_tr, Fef_te, y_te, seed)

        res_G.append(acc_G); res_EG.append(acc_EG); res_EF.append(acc_EF)
        ks_global_list.append(k_glob); ks_frames_list.append(ks_tr)

        d_ef = acc_EF - acc_G
        print(f"{seed%100000:<8} {acc_G:>7.3f} {acc_EG:>9.3f} {acc_EF:>11.3f}"
              f" {d_ef:>+9.3f} {k_glob:>8.5f} {'✓' if d_ef>0 else '✗'}")

    G   = np.array(res_G)
    EG  = np.array(res_EG)
    EF  = np.array(res_EF)
    k_g = float(np.mean(ks_global_list))
    ks_f_medio = np.mean(ks_frames_list, axis=0)

    try:
        _, p_eg = stats.wilcoxon(EG, G)
        _, p_ef = stats.wilcoxon(EF, G)
        _, p_ef_eg = stats.wilcoxon(EF, EG)
    except Exception:
        p_eg = p_ef = p_ef_eg = float('nan')

    print(f"\n{'═'*65}")
    print(f"  AlphaPhi MEG — Eco de Frames | Canal: {ch}")
    print(f"{'═'*65}")
    print(f"  G  (baseline):   {G.mean():.4f}  ±{G.std():.4f}")
    print(f"  Eco Global:      {EG.mean():.4f}  Δ={EG.mean()-G.mean():+.4f}  p={p_eg:.6f}")
    print(f"  Eco Frames:      {EF.mean():.4f}  Δ={EF.mean()-G.mean():+.4f}  p={p_ef:.6f}")
    print(f"  Frames vs Global: Δ={EF.mean()-EG.mean():+.4f}  p={p_ef_eg:.6f}")
    print(f"{'─'*65}")
    print(f"  k_global médio:  {k_g:.5f}")
    print(f"  k por frame:     {' → '.join(f'{k:.4f}' for k in ks_f_medio)}")
    print(f"  progressão:      {'↑' if ks_f_medio[-1]>ks_f_medio[0] else '↓'}"
          f"  Δk = {ks_f_medio[-1]-ks_f_medio[0]:+.5f}")

    conclusao_ef  = "CONFIRMA" if (EF.mean() > G.mean()  and p_ef  < 0.05) else "NAO CONFIRMA"
    conclusao_eg  = "CONFIRMA" if (EG.mean() > G.mean()  and p_eg  < 0.05) else "NAO CONFIRMA"
    print(f"\n  Eco Global → {conclusao_eg}")
    print(f"  Eco Frames → {conclusao_ef}")
    print(f"{'═'*65}\n")

    resultado = {
        "experimento":   "AlphaPhi_MEG_Frames",
        "canal":         ch,
        "n_frames":      N_FRAMES,
        "timestamp":     TIMESTAMP,
        "G_mean":        float(G.mean()),
        "EcoGlobal_mean": float(EG.mean()),
        "EcoFrames_mean": float(EF.mean()),
        "delta_EF_G":    float(EF.mean() - G.mean()),
        "delta_EF_EG":   float(EF.mean() - EG.mean()),
        "p_ef":          float(p_ef),
        "p_eg":          float(p_eg),
        "k_global":      k_g,
        "k_por_frame":   [float(k) for k in ks_f_medio],
        "progressao_k":  float(ks_f_medio[-1] - ks_f_medio[0]),
        "conclusao_frames": conclusao_ef,
        "conclusao_global": conclusao_eg,
    }

    with open("meg_frames_results.json", "w") as f:
        json.dump(resultado, f, indent=2)
    print("  Salvo: meg_frames_results.json")
    return resultado

rodar()
