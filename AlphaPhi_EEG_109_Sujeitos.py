# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_EEG_109_Sujeitos.py
Vitor Edson Delavi · Florianópolis · 2026

Hipótese:
    O eco ressonante fonônico (V2: acoplamento φ) melhora a classificação
    de estados cognitivos (imaginação motora esquerda vs direita) em sinais
    EEG de cérebros humanos reais — generalizando o resultado do EEG sintético
    (+42% Alpha vs Noise, +41% Alpha vs Beta) para substrato humano genuíno.

Dataset:
    PhysioNet EEGMMIDB — EEG Motor Movement/Imagery Database
    109 sujeitos · Runs R04 + R06 · Imaginação de movimento: esquerdo (T1) vs direito (T2)
    https://physionet.org/content/eegmmidb/1.0.0/

Protocolo:
    - Seeds por timestamp — nenhum valor manual
    - Resultados reportados integralmente (favoráveis e não favoráveis)
    - Sujeitos com épocas insuficientes documentados, não descartados silenciosamente
    - Resultados parciais salvos após cada sujeito

INSTRUÇÕES PARA GOOGLE COLAB:
    1. Rodar a Célula 1 (instalação)
    2. Rodar a Célula 2 (funções)
    3. Rodar a Célula 3 (pipeline 109 sujeitos) — ~20-30 min
    4. Rodar a Célula 4 (análise agregada)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CÉLULA 1 — Instalação (rodar uma vez)
# ═══════════════════════════════════════════════════════════════════════════════

# Cole e rode no Colab:
#
# !pip install -q mne
# import importlib, sys
# if 'mne' not in sys.modules: importlib.import_module('mne')

# ═══════════════════════════════════════════════════════════════════════════════
# CÉLULA 2 — Imports, constantes e funções base
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import json
import os
import time
import subprocess
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
N_SEEDS        = 10              # por sujeito — suficiente para Wilcoxon
N_EPOCHS_TRAIN = 40
HIDDEN         = 89
LR             = 0.01
CANAL          = 'C3'           # hemisfério contralateral ao movimento direito
FILTRO         = (8, 30)        # banda Alpha + Beta — estados cognitivos motores
URL_BASE       = "https://physionet.org/files/eegmmidb/1.0.0"

TIMESTAMP = int(time.time())
SEEDS_GLOBAIS = [TIMESTAMP + i * 137 for i in range(N_SEEDS)]

print(f"AlphaPhi EEG 109 Sujeitos")
print(f"PHI={PHI:.6f}  K_MIN(√2)={K_MIN:.6f}")
print(f"Canal: {CANAL} | Filtro: {FILTRO[0]}-{FILTRO[1]} Hz")
print(f"Seeds: {N_SEEDS} por sujeito | Timestamp: {TIMESTAMP}\n")

# ── Campo coletivo ─────────────────────────────────────────────────────────────
def medir_campo(X):
    freq_batch = np.fft.fft(X, axis=-1)
    amp_media  = np.abs(freq_batch).mean(axis=0)
    amp_norm   = np.clip(amp_media / (amp_media.sum() + 1e-8), 1e-10, 1.0)
    entropia   = -np.sum(amp_norm * np.log(amp_norm))
    coh        = float(1.0 - entropia / np.log(X.shape[-1]))
    k          = K_MIN + (PHI - K_MIN) * coh
    return k, coh

# ── Eco V1 (rotação 1/k) ──────────────────────────────────────────────────────
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

# ── Eco V2 (acoplamento φ) ────────────────────────────────────────────────────
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

# ── Ativações ─────────────────────────────────────────────────────────────────
def golden_activation(x):
    return PHI * np.tanh(x / PHI)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def clip_grad(g, max_norm=1.0):
    norm = np.linalg.norm(g)
    return g * (max_norm / norm) if norm > max_norm else g

# ── Rede neural ───────────────────────────────────────────────────────────────
def treinar(X_tr, y_tr, X_te, y_te, seed):
    dim_in = X_tr.shape[1]
    rng = np.random.default_rng(seed + 999)
    W1 = rng.normal(0, np.sqrt(2.0 / dim_in), (dim_in, HIDDEN))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, np.sqrt(2.0 / HIDDEN), (HIDDEN, 1))
    b2 = np.zeros(1)
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

# ── Carregar EDF ──────────────────────────────────────────────────────────────
def carregar_eeg(arquivos, canal=CANAL, filtro=FILTRO):
    epochs_T1, epochs_T2 = [], []
    for arq in arquivos:
        if not os.path.exists(arq):
            continue
        try:
            raw = mne.io.read_raw_edf(arq, preload=True, verbose=False)
            fs  = raw.info['sfreq']
            raw.filter(filtro[0], filtro[1], fir_design='firwin', verbose=False)
            events, _ = mne.events_from_annotations(raw, verbose=False)
            ch_idx = raw.ch_names.index(canal) if canal in raw.ch_names else 0
            data   = raw.get_data()[ch_idx]
            n_pts  = int(fs)  # 1 segundo
            for ev in events:
                onset, codigo = ev[0], ev[2]
                if onset + n_pts > len(data):
                    continue
                seg = data[onset:onset + n_pts]
                seg = seg / (np.std(seg) + 1e-8)
                if len(seg) != 256:
                    idx_r = np.linspace(0, len(seg) - 1, 256).astype(int)
                    seg   = seg[idx_r]
                if codigo == 1:
                    epochs_T1.append(seg)
                elif codigo == 2:
                    epochs_T2.append(seg)
        except Exception as e:
            print(f"    [aviso] {arq}: {e}")
    return np.array(epochs_T1) if epochs_T1 else np.empty((0, 256)), \
           np.array(epochs_T2) if epochs_T2 else np.empty((0, 256))

# ── Download de um sujeito ─────────────────────────────────────────────────────
def baixar_sujeito(sid):
    s = f"S{sid:03d}"
    arquivos = []
    for run in ["R04", "R06"]:
        nome  = f"{s}{run}.edf"
        if not os.path.exists(nome):
            url = f"{URL_BASE}/{s}/{nome}"
            ret = subprocess.run(
                ["wget", "-q", "--timeout=30", url, "-O", nome],
                capture_output=True
            )
            if ret.returncode != 0 or os.path.getsize(nome) < 1000:
                if os.path.exists(nome):
                    os.remove(nome)
                continue
        arquivos.append(nome)
    return arquivos

# ── Processar um sujeito ───────────────────────────────────────────────────────
def processar_sujeito(sid):
    arquivos = baixar_sujeito(sid)
    if not arquivos:
        return None

    T1, T2 = carregar_eeg(arquivos)

    # Limpar EDFs para poupar espaço em disco
    for arq in arquivos:
        if os.path.exists(arq):
            os.remove(arq)

    n_min = min(len(T1), len(T2))
    if n_min < 10:
        return {"sid": sid, "status": "insuficiente", "n_epocas": int(n_min)}

    T1, T2 = T1[:n_min], T2[:n_min]
    X = np.vstack([T1, T2])
    y = np.array([1] * n_min + [0] * n_min, dtype=float)
    n_tr = int(0.7 * len(X))

    res_G, res_v1, res_v2, ks = [], [], [], []

    for seed in SEEDS_GLOBAIS:
        rng   = np.random.default_rng(seed + sid * 10000)
        idx   = rng.permutation(len(X))
        X_sh, y_sh = X[idx], y[idx]
        X_tr, y_tr = X_sh[:n_tr], y_sh[:n_tr]
        X_te, y_te = X_sh[n_tr:], y_sh[n_tr:]

        acc_G = treinar(X_tr, y_tr, X_te, y_te, seed + sid)

        Xtr_v1, k_tr, _ = eco_v1(X_tr)
        Xte_v1,  _,  _  = eco_v1(X_te)
        acc_v1 = treinar(Xtr_v1, y_tr, Xte_v1, y_te, seed + sid)

        Xtr_v2, _, _ = eco_v2(X_tr)
        Xte_v2, _, _ = eco_v2(X_te)
        acc_v2 = treinar(Xtr_v2, y_tr, Xte_v2, y_te, seed + sid)

        res_G.append(acc_G)
        res_v1.append(acc_v1)
        res_v2.append(acc_v2)
        ks.append(k_tr)

    return {
        "sid":       sid,
        "status":    "ok",
        "n_epocas":  int(n_min),
        "G":         float(np.mean(res_G)),
        "V1":        float(np.mean(res_v1)),
        "V2":        float(np.mean(res_v2)),
        "k_medio":   float(np.mean(ks)),
        "delta_V2":  float(np.mean(res_v2) - np.mean(res_G)),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CÉLULA 3 — Pipeline 109 sujeitos
# ═══════════════════════════════════════════════════════════════════════════════

def rodar_109():
    resultados  = []
    arquivo_saida = "eeg_109_parcial.json"

    print(f"{'Suj':>4} {'n':>4} {'G':>7} {'V1':>7} {'V2(φ)':>7} {'Δ':>7} {'k':>8}")
    print("─" * 52)

    for sid in range(1, 110):
        r = processar_sujeito(sid)

        if r is None:
            print(f"  S{sid:03d} — download falhou")
            resultados.append({"sid": sid, "status": "download_falhou"})
        elif r["status"] == "insuficiente":
            print(f"  S{sid:03d} — épocas insuficientes ({r['n_epocas']})")
            resultados.append(r)
        else:
            sig = "✓" if r["delta_V2"] > 0 else "✗"
            print(
                f"  S{sid:03d} "
                f"{r['n_epocas']:>4} "
                f"{r['G']:>7.3f} "
                f"{r['V1']:>7.3f} "
                f"{r['V2']:>7.3f} "
                f"{r['delta_V2']:>+7.3f} "
                f"{r['k_medio']:>8.5f} {sig}"
            )
            resultados.append(r)

        # Salvar parcial após cada sujeito
        with open(arquivo_saida, "w") as f:
            json.dump({
                "timestamp": TIMESTAMP,
                "seeds":     SEEDS_GLOBAIS,
                "canal":     CANAL,
                "resultados": resultados
            }, f, indent=2)

    return resultados

resultados = rodar_109()

# ═══════════════════════════════════════════════════════════════════════════════
# CÉLULA 4 — Análise agregada
# ═══════════════════════════════════════════════════════════════════════════════

def analisar(resultados):
    validos = [r for r in resultados if r.get("status") == "ok"]
    n_ok    = len(validos)
    n_ins   = sum(1 for r in resultados if r.get("status") == "insuficiente")
    n_fail  = sum(1 for r in resultados if r.get("status") in ("download_falhou", None))

    if n_ok == 0:
        print("Nenhum sujeito válido processado.")
        return

    G_all  = np.array([r["G"]  for r in validos])
    V1_all = np.array([r["V1"] for r in validos])
    V2_all = np.array([r["V2"] for r in validos])
    k_all  = np.array([r["k_medio"] for r in validos])

    _, p_v1  = stats.wilcoxon(V1_all, G_all)
    _, p_v2  = stats.wilcoxon(V2_all, G_all)
    _, p_v2v1= stats.wilcoxon(V2_all, V1_all)

    n_v2_positivo = int(np.sum(V2_all > G_all))

    print(f"\n{'═'*60}")
    print(f"  AlphaPhi EEG — 109 Sujeitos · PhysioNet EEGMMIDB")
    print(f"  Canal: {CANAL} | Filtro: {FILTRO[0]}-{FILTRO[1]} Hz")
    print(f"{'═'*60}")
    print(f"  Sujeitos válidos:       {n_ok}/109")
    print(f"  Insuficientes:          {n_ins}")
    print(f"  Falhas de download:     {n_fail}")
    print(f"{'─'*60}")
    print(f"  G  (baseline):  {G_all.mean():.4f}  ±{G_all.std():.4f}")
    print(f"  V1 (√2+campo):  {V1_all.mean():.4f}  ±{V1_all.std():.4f}"
          f"  Δ={V1_all.mean()-G_all.mean():+.4f}  p={p_v1:.6f}")
    print(f"  V2 (φ):         {V2_all.mean():.4f}  ±{V2_all.std():.4f}"
          f"  Δ={V2_all.mean()-G_all.mean():+.4f}  p={p_v2:.6f}")
    print(f"{'─'*60}")
    print(f"  V2 supera baseline em: {n_v2_positivo}/{n_ok} sujeitos"
          f"  ({100*n_v2_positivo/n_ok:.1f}%)")
    print(f"  V2 vs V1: Δ={V2_all.mean()-V1_all.mean():+.4f}  p={p_v2v1:.6f}")
    print(f"  k_campo médio: {k_all.mean():.5f}  (√2={np.sqrt(2):.5f}  φ={PHI:.5f})")

    conclusao = "CONFIRMA" if (V2_all.mean() > G_all.mean() and p_v2 < 0.05) else "NAO CONFIRMA"
    print(f"\n  Hipótese: eco V2 supera baseline em EEG humano real → {conclusao}")
    print(f"{'═'*60}\n")

    resultado_final = {
        "experimento":    "AlphaPhi_EEG_109_Sujeitos",
        "canal":          CANAL,
        "filtro_hz":      list(FILTRO),
        "timestamp":      TIMESTAMP,
        "n_validos":      n_ok,
        "n_insuficientes": n_ins,
        "n_falhas":       n_fail,
        "G_mean":         float(G_all.mean()),
        "V1_mean":        float(V1_all.mean()),
        "V2_mean":        float(V2_all.mean()),
        "delta_V2_G":     float(V2_all.mean() - G_all.mean()),
        "p_v2":           float(p_v2),
        "p_v1":           float(p_v1),
        "p_v2_vs_v1":     float(p_v2v1),
        "n_v2_positivo":  n_v2_positivo,
        "k_medio":        float(k_all.mean()),
        "conclusao":      conclusao,
        "por_sujeito":    resultados,
    }

    with open("eeg_109_results.json", "w") as f:
        json.dump(resultado_final, f, indent=2)
    print("  Resultados salvos: eeg_109_results.json")

    return resultado_final

resultado_final = analisar(resultados)
