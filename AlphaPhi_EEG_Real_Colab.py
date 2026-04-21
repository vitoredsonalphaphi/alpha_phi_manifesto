# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_EEG_Real_Colab.py
Vitor Edson Delavi · Florianópolis · 2026

Cole este bloco no Google Colab após carregar os arquivos EDF.

Dataset: PhysioNet EEGMMIDB — EEG Motor Movement/Imagery Database
    S001R04.edf: imaginação de movimento — punho esquerdo (T1) vs direito (T2)
    S001R06.edf: repetição do mesmo protocolo

Tarefa: classificar T1 (imaginação esquerda) vs T2 (imaginação direita)
    por coerência espectral — sem instrução ao processador.

Instalação no Colab (rodar antes):
    !pip install -q mne
    !wget -q https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf -O S001R04.edf
    !wget -q https://physionet.org/files/eegmmidb/1.0.0/S001/S001R06.edf -O S001R06.edf
"""

# ── Dependências ──────────────────────────────────────────────────────────────
import numpy as np
import json
import mne
from scipy import stats
mne.set_log_level('WARNING')

# ── Constantes ────────────────────────────────────────────────────────────────
PHI   = (1 + np.sqrt(5)) / 2       # 1.6180...
K_MIN = np.sqrt(2)                  # 1.4142...
N_ECO = 3
N_SEEDS  = 20
N_EPOCHS_TRAIN = 60
HIDDEN   = 89
LR       = 0.01

# ── Carregar e preparar dados EEG reais ───────────────────────────────────────

def carregar_eeg(arquivos, canal='C3', epoch_len=1.0, filtro=(8, 30)):
    """
    Carrega arquivos EDF, filtra banda de interesse, extrai épocas por evento.

    canal    : canal EEG central de interesse (C3, Cz, C4 para motor)
    epoch_len: duração de cada época em segundos
    filtro   : banda de frequência (Hz) — (8,30) cobre Alpha + Beta
    """
    epochs_T1, epochs_T2 = [], []

    for arquivo in arquivos:
        raw = mne.io.read_raw_edf(arquivo, preload=True, verbose=False)
        fs  = raw.info['sfreq']

        # Filtro de banda
        raw.filter(filtro[0], filtro[1], fir_design='firwin', verbose=False)

        # Eventos: T1=imaginação esquerda, T2=imaginação direita
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Seleciona canal
        ch_idx = raw.ch_names.index(canal) if canal in raw.ch_names else 0
        data   = raw.get_data()[ch_idx]  # (n_amostras,)
        n_pts  = int(epoch_len * fs)

        for ev in events:
            onset  = ev[0]
            codigo = ev[2]
            if onset + n_pts > len(data):
                continue
            segmento = data[onset:onset + n_pts]
            segmento = segmento / (np.std(segmento) + 1e-8)
            # Reamostrar para DIM=256 se necessário
            if len(segmento) != 256:
                indices = np.linspace(0, len(segmento)-1, 256).astype(int)
                segmento = segmento[indices]
            if codigo == 1:    # T1 — esquerda
                epochs_T1.append(segmento)
            elif codigo == 2:  # T2 — direita
                epochs_T2.append(segmento)

    print(f"  Épocas T1 (esquerda): {len(epochs_T1)}")
    print(f"  Épocas T2 (direita):  {len(epochs_T2)}")
    return np.array(epochs_T1), np.array(epochs_T2)

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

# ── Ativações e utilitários ───────────────────────────────────────────────────

def golden_activation(x):
    return PHI * np.tanh(x / PHI)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def clip_grad(g, max_norm=1.0):
    norm = np.linalg.norm(g)
    return g * max_norm / norm if norm > max_norm else g

# ── Rede neural ───────────────────────────────────────────────────────────────

def treinar(X_tr, y_tr, X_te, y_te, seed):
    dim_in = X_tr.shape[1]
    rng = np.random.default_rng(seed + 999)
    W1 = rng.normal(0, np.sqrt(2.0/dim_in),  (dim_in, HIDDEN))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, np.sqrt(2.0/HIDDEN), (HIDDEN, 1))
    b2 = np.zeros(1)
    for _ in range(N_EPOCHS_TRAIN):
        idx = rng.permutation(len(X_tr))
        for i in range(0, len(X_tr) - 31, 32):
            Xb, yb = X_tr[idx[i:i+32]], y_tr[idx[i:i+32]]
            z1 = Xb @ W1 + b1
            a1 = golden_activation(z1)
            yh = sigmoid(a1 @ W2 + b2).squeeze()
            dL = (yh - yb) / len(yb)
            dW2 = a1.T @ dL.reshape(-1, 1)
            db2 = dL.sum(keepdims=True)
            dz1 = (dL.reshape(-1, 1) * W2.T) * (1 - np.tanh(z1/PHI)**2)
            W1 -= LR * clip_grad(Xb.T @ dz1)
            b1 -= LR * np.clip(dz1.sum(axis=0), -1, 1)
            W2 -= LR * clip_grad(dW2)
            b2 -= LR * np.clip(db2, -1, 1)
    yh_te = sigmoid(golden_activation(X_te @ W1 + b1) @ W2 + b2).squeeze()
    return float(np.mean((yh_te >= 0.5) == y_te))

# ── Pipeline principal ────────────────────────────────────────────────────────

def rodar_experimento(arquivos, canal='C3'):
    print(f"\nCarregando EEG real — canal {canal}")
    print("─" * 50)

    T1, T2 = carregar_eeg(arquivos, canal=canal)

    n_min = min(len(T1), len(T2))
    if n_min < 20:
        print(f"  ⚠️  Épocas insuficientes ({n_min} por classe). Tente outro canal.")
        return

    # Balancear classes
    T1, T2 = T1[:n_min], T2[:n_min]
    X = np.vstack([T1, T2])
    y = np.array([1]*n_min + [0]*n_min, dtype=float)

    print(f"  {n_min} épocas por classe | DIM={X.shape[1]}")
    print(f"  Total: {len(X)} épocas\n")

    # Split treino/teste (70/30)
    n_tr = int(0.7 * len(X))
    TIMESTAMP = int(__import__('time').time())
    SEEDS = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

    res_G, res_v1, res_v2, ks = [], [], [], []

    print(f"{'Seed':<6} {'G':>7} {'V1(1/k)':>9} {'V2(φ)':>9} {'k_campo':>9}")
    print("─" * 44)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        X_sh, y_sh = X[idx], y[idx]
        X_tr, y_tr = X_sh[:n_tr], y_sh[:n_tr]
        X_te, y_te = X_sh[n_tr:], y_sh[n_tr:]

        acc_G = treinar(X_tr, y_tr, X_te, y_te, seed)

        Xtr_v1, k_tr, _ = eco_v1(X_tr)
        Xte_v1,  _,  _ = eco_v1(X_te)
        acc_v1 = treinar(Xtr_v1, y_tr, Xte_v1, y_te, seed)

        Xtr_v2, _, _ = eco_v2(X_tr)
        Xte_v2, _, _ = eco_v2(X_te)
        acc_v2 = treinar(Xtr_v2, y_tr, Xte_v2, y_te, seed)

        res_G.append(acc_G); res_v1.append(acc_v1)
        res_v2.append(acc_v2); ks.append(k_tr)
        print(f"{seed % 10000:<6} {acc_G:>7.3f} {acc_v1:>9.3f} {acc_v2:>9.3f} {k_tr:>9.5f}")

    G, V1, V2 = np.array(res_G), np.array(res_v1), np.array(res_v2)
    _, p_v1  = stats.wilcoxon(V1, G)
    _, p_v2  = stats.wilcoxon(V2, G)
    _, p_v2v1= stats.wilcoxon(V2, V1)
    k_m = float(np.mean(ks))

    print(f"\n{'═'*50}")
    print(f"  G  (baseline): {G.mean():.4f}")
    print(f"  V1 (1/k):      {V1.mean():.4f}  Δ={V1.mean()-G.mean():+.4f}  p={p_v1:.6f}")
    print(f"  V2 (φ):        {V2.mean():.4f}  Δ={V2.mean()-G.mean():+.4f}  p={p_v2:.6f}")
    print(f"\n  k_campo médio: {k_m:.5f}  (√2={np.sqrt(2):.5f})")
    print(f"  V2 vs V1: Δ={V2.mean()-V1.mean():+.4f}, p={p_v2v1:.6f}")

    sig = "✅" if V2.mean() > G.mean() and p_v2 < 0.05 else "⚠️"
    print(f"  Eco V2 supera baseline: {sig}")

    return {
        "canal": canal, "n_epocas": n_min,
        "G": float(G.mean()), "V1": float(V1.mean()), "V2": float(V2.mean()),
        "p_v2": float(p_v2), "k_medio": k_m,
        "delta_V2_G": float(V2.mean()-G.mean()),
    }

# ── Executar ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    arquivos = ["S001R04.edf", "S001R06.edf"]

    # Teste em C3 (hemisfério esquerdo — motor direito)
    r_C3 = rodar_experimento(arquivos, canal='C3')

    # Teste em C4 (hemisfério direito — motor esquerdo)
    r_C4 = rodar_experimento(arquivos, canal='C4')

    # Teste em Cz (central)
    r_Cz = rodar_experimento(arquivos, canal='Cz')

    print("\n\n── Comparativo por canal ─────────────────────────────")
    print(f"{'Canal':<8} {'G':>8} {'V2(φ)':>8} {'Δ':>8} {'k':>10}")
    for r, ch in [(r_C3,'C3'), (r_C4,'C4'), (r_Cz,'Cz')]:
        if r:
            print(f"  {ch:<6} {r['G']:>8.4f} {r['V2']:>8.4f} "
                  f"{r['delta_V2_G']:>+8.4f} {r['k_medio']:>10.5f}")

    # Salvar
    with open("eeg_real_results.json", "w") as f:
        json.dump({"C3": r_C3, "C4": r_C4, "Cz": r_Cz}, f, indent=2)
    print("\nResultados salvos: eeg_real_results.json")
