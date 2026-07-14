# AlphaPhi_Scanner_EEG_Repouso_COLAB.py
# Vitor Edson Delavi · Florianópolis · 2026
#
# SCANNER α-φ (OBSERVAÇÃO PURA) sobre EEG humano em repouso
# Substrato: PhysioNet EEGMMIDB · R01 (olhos abertos) · R02 (olhos fechados)
#
# ══════════════════════════════════════════════════════════════════════════════
# DECLARAÇÃO DE ESCOPO — PROTOCOLO ANTI-TENDENCIAMENTO
#
# CONTEXTO HISTÓRICO:
#   Abril/2026: eco ressonante testado em EEG real (109 sujeitos).
#   Resultado: NÃO CONFIRMA — eco destruiu informação lateral do substrato.
#   Diagnóstico: eco TRANSFORMA; não serve para observação pura.
#
# O QUE ESTE TESTE É:
#   Primeira verificação do Scanner α-φ em substrato humano em REPOUSO.
#   Instrumento: Scanner — OBSERVA, não transforma o sinal em nenhum momento.
#   T1 = R01 (olhos abertos) · T2 = R02 (olhos fechados).
#   Discriminação biologicamente mais robusta do EEG: ritmo Alpha (8–13 Hz)
#   emerge naturalmente em repouso com olhos fechados. Décadas de literatura.
#   Pergunta central: o Scanner detecta estrutura coerente no substrato humano?
#
# O QUE ESTE TESTE NÃO É:
#   Aplicação do eco ressonante (eco ressonante: AUSENTE).
#   Classificação de tarefa motora.
#   Afirmação prévia de que Alpha será a banda encontrada.
#
# DISTINÇÃO FUNDAMENTAL:
#   eco ressonante → pergunta "o sinal ressoa SE eu aplicar φ?" → TRANSFORMA
#   Scanner        → pergunta "o sinal JÁ ressoa por si mesmo?" → OBSERVA
#
# RESULTADOS IGUALMENTE VÁLIDOS:
#   Scanner encontra Alpha → diálogo código/substrato estabelecido
#   Scanner encontra outra banda → informa onde α reside neste substrato
#   meta_coh < 0.70 → substrato não convergiu nesta configuração
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mne", "scipy"],
               check=True)

import numpy as np
import time
import os
import subprocess as sp
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    mne.set_log_level('WARNING')
except ImportError:
    raise ImportError("Execute: !pip install -q mne")

# ── Constantes α-φ ────────────────────────────────────────────────────────────
PHI       = (1 + np.sqrt(5)) / 2        # 1.6180339887
ALPHA     = 1 / 137.035999084           # constante de estrutura fina
LOG_ALPHA = np.log(1.0 / ALPHA)         # log(137) ≈ 4.920

FS_EEG     = 160         # PhysioNet EEGMMIDB: 160 Hz
N_AMOSTRAS = 320         # 2 segundos → resolução espectral 0.5 Hz/bin
CANAL      = 'C3'        # hemisfério esquerdo — dominante em repouso
N_SUJ      = 10          # escopo exploratório: S001–S010
N_EPOCAS   = 30          # máximo de épocas por run por sujeito
URL_BASE   = "https://physionet.org/files/eegmmidb/1.0.0"

TIMESTAMP = int(time.time())
print("=" * 65)
print(f"φ = {PHI:.10f}")
print(f"α = {ALPHA:.10f}")
print(f"SCANNER α-φ — OBSERVAÇÃO PURA — EEG REPOUSO")
print(f"T1=R01 (olhos abertos)  T2=R02 (olhos fechados)")
print(f"Sujeitos: S001–S0{N_SUJ:02d}  |  Canal: {CANAL}")
print(f"Épocas: {N_AMOSTRAS} amostras ({N_AMOSTRAS/FS_EEG:.1f}s)  |  Seed: {TIMESTAMP}")
print(f"eco ressonante: AUSENTE — instrumento de observação pura")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# BANDAS φ-PROPORCIONAIS
# ══════════════════════════════════════════════════════════════════════════════

def gerar_bandas_phi_eeg(f_min=0.5, f_max=None, fs=FS_EEG):
    if f_max is None:
        f_max = fs / 2.0
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max:
            break
        f = f_next
    return bandas

def bandas_para_bins_eeg(bandas, n, fs):
    result = []
    for f_lo, f_hi in bandas:
        b_lo = max(0, int(f_lo / (fs / n)))
        b_hi = min(int(f_hi / (fs / n)) + 1, n // 2 + 1)
        result.append((b_lo, b_hi, f_lo, f_hi))
    return result

def nome_banda(f_lo, f_hi):
    centro = (f_lo + f_hi) / 2
    if centro < 4:   return "Delta"
    if centro < 8:   return "Theta"
    if centro < 13:  return "Alpha"
    if centro < 30:  return "Beta"
    if centro < 80:  return "Gamma"
    return "HiGamma"

# ══════════════════════════════════════════════════════════════════════════════
# SCANNER α-φ — OBSERVAÇÃO PURA
# ══════════════════════════════════════════════════════════════════════════════

class ScannerEEGRepouso:
    """
    Scanner α-φ com Capacitor de Software — substrato EEG em repouso.

    eco ressonante: AUSENTE.
    Instrumento de observação pura: mede coerência e discriminabilidade
    por banda φ-proporcional, sem transformar o sinal em nenhum momento.
    """

    PHI              = PHI
    ALPHA            = ALPHA
    LOG_ALPHA        = LOG_ALPHA
    META_COH_LIMIAR  = 0.70
    LIMIAR_SUBSTRATO = 1e-4
    N_SONDA_MIN      = 3
    N_SONDA_MAX      = 10

    def __init__(self, bandas, bins, fs=FS_EEG):
        self.bandas = bandas
        self.bins   = bins
        self.fs     = fs

        self._scores_ema    = None
        self._meta_coh_hist = []
        self._perfil_final  = None
        self._n_ciclos      = 0
        self._fase_otima    = None
        self._beta          = 1.0
        self.pronto         = False

        self._capacitance = 1.0
        self._resistance  = 1.0
        self._charge      = None

    def _coh_banda(self, ffts, b_lo, b_hi):
        """Coerência espectral — quão organizado é o campo nesta banda."""
        if b_hi <= b_lo:
            return 0.0
        mags   = np.abs(ffts[:, b_lo:b_hi]) + 1e-10
        mags_n = mags / mags.sum(axis=1, keepdims=True)
        H      = -np.sum(mags_n * np.log(mags_n + 1e-15), axis=1)
        H_max  = max(np.log(max(b_hi - b_lo, 2)), self.LOG_ALPHA)
        return float(1.0 - np.clip(H / H_max, 0.0, 1.0).mean())

    def _disc_banda(self, ffts_T1, ffts_T2, b_lo, b_hi):
        """Discriminabilidade — diferença de potência R01 vs R02 nesta banda."""
        if b_hi <= b_lo:
            return 0.0
        p1 = np.abs(ffts_T1[:, b_lo:b_hi]).mean(axis=1).mean()
        p2 = np.abs(ffts_T2[:, b_lo:b_hi]).mean(axis=1).mean()
        return float(abs(p1 - p2) / (p1 + p2 + 1e-10))

    def _meta_coh(self, scores):
        s      = np.array(scores) + 1e-10
        s_norm = s / s.sum()
        H_meta = float(-np.sum(s_norm * np.log(s_norm + 1e-15)))
        H_max  = np.log(max(len(s), 2))
        return float(1.0 - np.clip(H_meta / H_max, 0.0, 1.0))

    def _adjust_dielectric(self, spectral_entropy):
        self._capacitance = 1.0 + (spectral_entropy * self.PHI)

    def _process_pulse(self, s_arr):
        tau = self._capacitance * self._resistance * (self.PHI / self.ALPHA)
        a   = 1.0 / (tau + 1.0)
        if self._charge is None:
            self._charge = s_arr.copy()
        else:
            self._charge = a * s_arr + (1.0 - a) * self._charge
        return self._charge

    def escaneia(self, epochs_T1, epochs_T2):
        if self.pronto:
            return

        ffts_T1  = np.array([np.fft.rfft(e) for e in epochs_T1])
        ffts_T2  = np.array([np.fft.rfft(e) for e in epochs_T2])
        ffts_all = np.vstack([ffts_T1, ffts_T2])

        perfil, scores = [], []

        for i, (b_lo, b_hi, f_lo, f_hi) in enumerate(self.bins):
            # Mínimo de 6 bins: elimina viés de coh em bandas de baixa resolução
            # F03 (3 bins) e F04 (3 bins) têm coh inflado matematicamente
            if b_hi - b_lo < 6:
                continue
            coh  = self._coh_banda(ffts_all, b_lo, b_hi)
            disc = self._disc_banda(ffts_T1, ffts_T2, b_lo, b_hi)
            s    = coh * disc
            perfil.append({
                'fase': i + 1, 'f_lo': f_lo, 'f_hi': f_hi,
                'nome': nome_banda(f_lo, f_hi),
                'coh':  round(coh, 4),
                'disc': round(disc, 4),
                'score': round(s, 6),
            })
            scores.append(s)

        s_arr   = np.array(scores)
        mc_prev = self._meta_coh_hist[-1] if self._meta_coh_hist else 0.5
        self._adjust_dielectric(1.0 - mc_prev)
        self._scores_ema = self._process_pulse(s_arr)

        mc = self._meta_coh(self._scores_ema)
        self._meta_coh_hist.append(mc)
        self._n_ciclos += 1

        parar  = self._n_ciclos >= self.N_SONDA_MIN and mc >= self.META_COH_LIMIAR
        limite = self._n_ciclos >= self.N_SONDA_MAX

        if parar or limite:
            self._fase_otima   = int(np.argmax(self._scores_ema))
            self._perfil_final = perfil
            self.pronto        = True

    def relatorio(self):
        if self._perfil_final is None:
            return {"pronto": False}
        scores   = [p['score'] for p in self._perfil_final]
        mc_final = self._meta_coh_hist[-1] if self._meta_coh_hist else 0.0
        adequado = (max(scores) > self.LIMIAR_SUBSTRATO
                    and mc_final >= self.META_COH_LIMIAR)
        f_opt    = self._perfil_final[self._fase_otima]
        return {
            "pronto":       True,
            "fase_otima":   self._fase_otima + 1,
            "banda":        f_opt['nome'],
            "f_lo":         f_opt['f_lo'],
            "f_hi":         f_opt['f_hi'],
            "scores":       scores,
            "meta_coh":     round(mc_final, 4),
            "n_ciclos":     self._n_ciclos,
            "adequado":     adequado,
            "perfil":       self._perfil_final,
            "capacitancia": round(self._capacitance, 4),
            # banda de maior coh pura (independente de disc)
            "banda_coh_pura": self._perfil_final[
                int(np.argmax([p['coh'] for p in self._perfil_final]))
            ]['nome'],
        }

# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO — R01/R02: segmentação de gravação contínua de repouso
# ══════════════════════════════════════════════════════════════════════════════

def baixar_arquivo(suj_id, run):
    """Baixa um arquivo EDF individual via wget."""
    nome = f"S{suj_id:03d}{run}.edf"
    if not os.path.exists(nome):
        url = f"{URL_BASE}/S{suj_id:03d}/{nome}"
        ret = sp.run(["wget", "-q", url, "-O", nome], capture_output=True)
        if ret.returncode != 0 or os.path.getsize(nome) < 1000:
            os.remove(nome) if os.path.exists(nome) else None
            return None
    return nome if os.path.exists(nome) else None

def carregar_epochs_repouso(arquivo, canal=CANAL,
                             n_amostras=N_AMOSTRAS, n_max=N_EPOCAS):
    """
    Segmenta gravação contínua de repouso em épocas de n_amostras.
    Pula os primeiros 2 segundos (transitório de início de gravação).
    Não usa anotações — toda a gravação é a condição.
    """
    epocas = []
    if arquivo is None:
        return np.zeros((0, n_amostras))
    try:
        raw    = mne.io.read_raw_edf(arquivo, preload=True, verbose=False)
        fs     = int(raw.info['sfreq'])
        ch_idx = (raw.ch_names.index(canal)
                  if canal in raw.ch_names else 0)
        data   = raw.get_data()[ch_idx]
        inicio = 2 * fs          # pula 2 segundos iniciais
        for i in range(n_max):
            start = inicio + i * n_amostras
            end   = start + n_amostras
            if end > len(data):
                break
            seg = data[start:end]
            seg = seg / (np.std(seg) + 1e-8)
            epocas.append(seg)
    except Exception:
        pass
    return np.array(epocas) if epocas else np.zeros((0, n_amostras))

# ══════════════════════════════════════════════════════════════════════════════
# BANDAS φ-PROPORCIONAIS (resolução 0.5 Hz/bin com N=320)
# ══════════════════════════════════════════════════════════════════════════════

BANDAS = gerar_bandas_phi_eeg(f_min=0.5, fs=FS_EEG)
BINS   = bandas_para_bins_eeg(BANDAS, n=N_AMOSTRAS, fs=FS_EEG)

print(f"\nBandas φ-proporcionais ({len(BANDAS)} bandas, resolução 0.5 Hz/bin):")
for i, (b_lo, b_hi, f_lo, f_hi) in enumerate(BINS):
    n_bins = b_hi - b_lo
    status = "SKIP" if n_bins < 6 else f"{n_bins} bins"
    print(f"  F{i+1:02d}  {f_lo:6.2f}–{f_hi:6.2f} Hz"
          f"  [{nome_banda(f_lo, f_hi):7s}]"
          f"  bins {b_lo:3d}–{b_hi:3d}  ({status})")

print(f"\nIniciando observação sobre {N_SUJ} sujeitos...\n")

# ══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

resultados    = []
bandas_otimas = []
bandas_coh    = []

for suj_id in range(1, N_SUJ + 1):
    print(f"── S{suj_id:03d} ─────────────────────────────────────────────")

    arq_r01 = baixar_arquivo(suj_id, 'R01')
    arq_r02 = baixar_arquivo(suj_id, 'R02')

    if arq_r01 is None or arq_r02 is None:
        print(f"  S{suj_id:03d}: download falhou (R01={arq_r01} R02={arq_r02})")
        resultados.append({"suj": suj_id, "status": "download_falhou"})
        continue

    T1 = carregar_epochs_repouso(arq_r01)   # olhos abertos
    T2 = carregar_epochs_repouso(arq_r02)   # olhos fechados

    print(f"  Épocas brutas : T1(R01/abertos)={len(T1)}"
          f"  T2(R02/fechados)={len(T2)}")

    if len(T1) < 5 or len(T2) < 5:
        print(f"  S{suj_id:03d}: épocas insuficientes — registrado")
        resultados.append({"suj": suj_id, "status": "epochs_insuficientes"})
        continue

    scanner = ScannerEEGRepouso(BANDAS, BINS)
    rng     = np.random.default_rng(TIMESTAMP + suj_id * 137)

    # Balancear T1/T2
    n_min = min(len(T1), len(T2))
    if len(T1) > n_min:
        T1 = T1[rng.choice(len(T1), n_min, replace=False)]
    if len(T2) > n_min:
        T2 = T2[rng.choice(len(T2), n_min, replace=False)]
    print(f"  Balanceadas   : T1={len(T1)}  T2={len(T2)}")

    for ciclo in range(scanner.N_SONDA_MAX):
        if scanner.pronto:
            break
        idx1 = rng.choice(len(T1), min(20, len(T1)), replace=False)
        idx2 = rng.choice(len(T2), min(20, len(T2)), replace=False)
        scanner.escaneia(T1[idx1], T2[idx2])

    r = scanner.relatorio()
    if not r["pronto"]:
        print(f"  S{suj_id:03d}: Scanner não convergiu")
        resultados.append({"suj": suj_id, "status": "nao_convergiu"})
        continue

    tag = "OK" if r["adequado"] else "INADEQUADO"
    print(f"  Banda ótima   : {r['banda']}"
          f"  ({r['f_lo']:.2f}–{r['f_hi']:.2f} Hz)")
    print(f"  Coh pura máx  : {r['banda_coh_pura']}  "
          f"(banda mais coerente independente de disc)")
    print(f"  meta_coh      : {r['meta_coh']:.4f}"
          f"  ciclos={r['n_ciclos']}"
          f"  C={r['capacitancia']:.3f}  {tag}")

    print("  Scores por banda (coh · disc → S):")
    for p in r['perfil']:
        marker = " ◀ ótima" if p['fase'] == r['fase_otima'] else ""
        coh_bar = "█" * int(p['coh'] * 10)
        print(f"    {p['nome']:7s} ({p['f_lo']:5.1f}–{p['f_hi']:5.1f} Hz)"
              f"  coh={p['coh']:.3f} {coh_bar}"
              f"  disc={p['disc']:.3f}"
              f"  S={p['score']:.5f}{marker}")

    resultados.append({"suj": suj_id, "status": "ok", **r})
    bandas_otimas.append(r['banda'])
    bandas_coh.append(r['banda_coh_pura'])
    print()

# ══════════════════════════════════════════════════════════════════════════════
# SÍNTESE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("SÍNTESE — SCANNER α-φ · OBSERVAÇÃO PURA · EEG REPOUSO")
print("=" * 65)

ok = [r for r in resultados if r.get("status") == "ok"]
print(f"\nSujeitos processados : {len(resultados)}")
print(f"Convergidos (OK)     : {len(ok)}")
print(f"Não convergidos      : {sum(1 for r in resultados if r.get('status') == 'nao_convergiu')}")
print(f"Falha de dados       : {sum(1 for r in resultados if r.get('status') not in ('ok','nao_convergiu'))}")

if ok:
    mc_med    = np.mean([r['meta_coh']    for r in ok])
    c_med     = np.mean([r['capacitancia'] for r in ok])
    adequados = sum(r['adequado'] for r in ok)
    print(f"\nmeta_coh médio       : {mc_med:.4f}  (limiar={ScannerEEGRepouso.META_COH_LIMIAR})")
    print(f"capacitância média   : {c_med:.4f}")
    print(f"substrato adequado   : {adequados}/{len(ok)}")

if bandas_otimas:
    from collections import Counter

    print(f"\nBanda ótima Scanner (coh × disc):")
    for banda, n in Counter(bandas_otimas).most_common():
        pct = 100 * n / len(bandas_otimas)
        bar = "█" * n
        print(f"  {banda:8s} : {n:2d} sujeitos ({pct:.0f}%)  {bar}")

    print(f"\nBanda mais coerente (coh pura — sem disc):")
    for banda, n in Counter(bandas_coh).most_common():
        pct = 100 * n / len(bandas_coh)
        bar = "█" * n
        print(f"  {banda:8s} : {n:2d} sujeitos ({pct:.0f}%)  {bar}")

    banda_dom  = Counter(bandas_otimas).most_common(1)[0][0]
    banda_coh0 = Counter(bandas_coh).most_common(1)[0][0]

    print(f"\n{'─'*50}")
    if banda_dom == "Alpha" or banda_coh0 == "Alpha":
        print(f"  → Alpha detectado.")
        print(f"    Consistente com o ritmo de repouso olhos-fechados.")
        print(f"    Diálogo Scanner/substrato: observação confirma estrutura.")
        print(f"    Registrar como calibração — não como confirmação causal.")
    else:
        print(f"  → Banda dominante: {banda_dom} (Scanner)  /  {banda_coh0} (coh pura)")
        print(f"    Informa onde α reside neste substrato nesta configuração.")
        print(f"    Resultado igualmente válido — dado honesto sobre o campo.")

print(f"\nφ = {PHI:.6f}   α = {ALPHA:.10f}")
print(f"eco ressonante: AUSENTE — observação pura confirmada")
print("=" * 65)
print(f"\nalpha-phi | scanner | eeg-repouso | {TIMESTAMP}")
print("=" * 65)
