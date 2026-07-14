# AlphaPhi_EEG_Scanner_PhiBands_COLAB.py
# Vitor Edson Delavi · Florianópolis · 2026
#
# INTEGRAÇÃO: Eco-Beep (bandas φ-proporcionais) + Scanner α-φ (Capacitor de Software)
# SUBSTRATO : EEG real — PhysioNet EEGMMIDB (imaginação motora)
#
# ══════════════════════════════════════════════════════════════════════════════
# DECLARAÇÃO DE ESCOPO — PROTOCOLO ANTI-TENDENCIAMENTO
#
# O QUE ESTE TESTE É:
#   Verificação exploratória do Scanner α-φ com SoftwareCapacitor sobre
#   sinal EEG real. As "fases" do Scanner são bandas φ-proporcionais do
#   espectro de frequência (eco-beep) em vez de camadas de rede neural.
#   Escopo: 10 sujeitos (S001–S010), exploratório.
#
# O QUE ESTE TESTE NÃO É:
#   Confirmação de que α reside na banda Alpha cerebral.
#   Todos os resultados — favoráveis e desfavoráveis — são reportados.
#
# DIRETRIZ DE DOMÍNIO:
#   EEG é substrato de frequência direto — alinhado com a diretriz do Scanner.
#   Nenhum proxy semântico. A frequência é a informação primária.
#
# RESULTADOS IGUALMENTE VÁLIDOS:
#   α encontra residência em qualquer banda → informação real
#   Scanner não converge (meta_coh < 0.70) → substrato inadequado neste estado
#   Capacitor comporta-se diferente do esperado → dado honesto sobre limites
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mne", "scipy"],
               check=True)

import numpy as np
import time
import os
import subprocess as sp
from scipy import stats
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

FS_EEG    = 160        # PhysioNet EEGMMIDB: 160 Hz
CANAL     = 'C3'       # hemisfério contralateral ao movimento direito
N_SUJ     = 10         # escopo exploratório: S001–S010
URL_BASE  = "https://physionet.org/files/eegmmidb/1.0.0"
RUNS      = ['R04', 'R06']

TIMESTAMP = int(time.time())
print("=" * 65)
print(f"φ = {PHI:.10f}")
print(f"α = {ALPHA:.10f}")
print(f"SCANNER α-φ + ECO-BEEP — EEG PhiBands")
print(f"Sujeitos: S001–S0{N_SUJ:02d}  |  Canal: {CANAL}")
print(f"Timestamp (seed): {TIMESTAMP}")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# BANDAS φ-PROPORCIONAIS — ECO-BEEP adaptado para EEG
# ══════════════════════════════════════════════════════════════════════════════

def gerar_bandas_phi_eeg(f_min=0.5, f_max=None, fs=FS_EEG):
    """Bandas φ-proporcionais para EEG. f_max = Nyquist se não informado."""
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
    """Converte bandas Hz → índices de bins FFT."""
    result = []
    for f_lo, f_hi in bandas:
        b_lo = max(0, int(f_lo / (fs / n)))
        b_hi = min(int(f_hi / (fs / n)) + 1, n // 2 + 1)
        result.append((b_lo, b_hi, f_lo, f_hi))
    return result

def nome_banda(f_lo, f_hi):
    """Nomeia a banda por convenção EEG padrão."""
    centro = (f_lo + f_hi) / 2
    if centro < 4:   return "Delta"
    if centro < 8:   return "Theta"
    if centro < 13:  return "Alpha"
    if centro < 30:  return "Beta"
    if centro < 80:  return "Gamma"
    return "HiGamma"

# ══════════════════════════════════════════════════════════════════════════════
# SCANNER α-φ — adaptado para bandas φ-proporcionais de EEG
# ══════════════════════════════════════════════════════════════════════════════

class ScannerEEGPhiBands:
    """
    Scanner α-φ com Capacitor de Software — substrato EEG.

    Fases = bandas φ-proporcionais (eco-beep) em vez de camadas de rede neural.
    Coerência  = entropia espectral dentro de cada banda (análogo a coh_A).
    Discriminabilidade = separação de potência T1 vs T2 (análogo a disc_lin).
    """

    PHI              = PHI
    ALPHA            = ALPHA
    LOG_ALPHA        = LOG_ALPHA
    wm               = 1.0 / PHI
    wn               = 1.0 - 1.0 / PHI
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
        self._bm            = 1.0
        self.pronto         = False

        # ── Capacitor de Software (Eco-Ressonante) ────────────────────────────
        self._capacitance = 1.0
        self._resistance  = 1.0
        self._charge      = None

    # ── Coerência espectral por banda ─────────────────────────────────────────

    def _coh_banda(self, ffts, b_lo, b_hi):
        """
        Entropia espectral média dentro da banda, análogo a coh_A.
        ffts: array (n_epochs, n_bins) de amplitudes.
        """
        if b_hi <= b_lo:
            return 0.0
        mags     = np.abs(ffts[:, b_lo:b_hi]) + 1e-10
        mags_n   = mags / mags.sum(axis=1, keepdims=True)
        H        = -np.sum(mags_n * np.log(mags_n + 1e-15), axis=1)
        H_max    = max(np.log(max(b_hi - b_lo, 2)), self.LOG_ALPHA)
        return float(1.0 - np.clip(H / H_max, 0.0, 1.0).mean())

    # ── Discriminabilidade por banda ──────────────────────────────────────────

    def _disc_banda(self, ffts_T1, ffts_T2, b_lo, b_hi):
        """
        Separação de potência média T1 vs T2, análogo a disc_lin normalizado.
        """
        if b_hi <= b_lo:
            return 0.0
        p1 = np.abs(ffts_T1[:, b_lo:b_hi]).mean(axis=1).mean()
        p2 = np.abs(ffts_T2[:, b_lo:b_hi]).mean(axis=1).mean()
        return float(abs(p1 - p2) / (p1 + p2 + 1e-10))

    # ── Meta-coerência ────────────────────────────────────────────────────────

    def _meta_coh(self, scores):
        s      = np.array(scores) + 1e-10
        s_norm = s / s.sum()
        H_meta = float(-np.sum(s_norm * np.log(s_norm + 1e-15)))
        H_max  = np.log(max(len(s), 2))
        return float(1.0 - np.clip(H_meta / H_max, 0.0, 1.0))

    # ── Capacitor de Software ─────────────────────────────────────────────────

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

    # ── Interface principal ───────────────────────────────────────────────────

    def escaneia(self, epochs_T1, epochs_T2):
        """
        Um ciclo de sondagem multi-banda.
        epochs_T1, epochs_T2: arrays (n_epochs, n_amostras).
        """
        if self.pronto:
            return

        n = epochs_T1.shape[1]

        # FFT de todas as épocas
        ffts_T1 = np.array([np.fft.rfft(e) for e in epochs_T1])
        ffts_T2 = np.array([np.fft.rfft(e) for e in epochs_T2])
        ffts_all = np.vstack([ffts_T1, ffts_T2])

        perfil, scores = [], []

        for i, (b_lo, b_hi, f_lo, f_hi) in enumerate(self.bins):
            coh  = self._coh_banda(ffts_all, b_lo, b_hi)
            disc = self._disc_banda(ffts_T1, ffts_T2, b_lo, b_hi)
            s    = coh * disc
            perfil.append({
                'fase': i + 1, 'f_lo': f_lo, 'f_hi': f_hi,
                'nome': nome_banda(f_lo, f_hi),
                'coh': round(coh, 4), 'disc': round(disc, 4),
                'score': round(s, 6),
            })
            scores.append(s)

        # Capacitor de Software — integração RC adaptativa
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
            "beta_atual":   round(self._beta, 4),
            "capacitancia": round(self._capacitance, 4),
        }

# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DE DADOS — PhysioNet EEGMMIDB
# ══════════════════════════════════════════════════════════════════════════════

def baixar_sujeito(suj_id, runs=RUNS):
    """Baixa os arquivos EDF de um sujeito via wget."""
    arquivos = []
    for run in runs:
        nome = f"S{suj_id:03d}{run}.edf"
        if not os.path.exists(nome):
            url = f"{URL_BASE}/S{suj_id:03d}/{nome}"
            ret = sp.run(["wget", "-q", url, "-O", nome], capture_output=True)
            if ret.returncode != 0 or os.path.getsize(nome) < 1000:
                os.remove(nome) if os.path.exists(nome) else None
                continue
        if os.path.exists(nome):
            arquivos.append(nome)
    return arquivos

def carregar_epochs(arquivos, canal=CANAL, n_amostras=160):
    """
    Carrega épocas T1 (esquerda) e T2 (direita) dos arquivos EDF.
    Retorna arrays normalizados de n_amostras pontos cada.
    """
    T1, T2 = [], []
    for arq in arquivos:
        try:
            raw    = mne.io.read_raw_edf(arq, preload=True, verbose=False)
            fs     = raw.info['sfreq']
            events, _ = mne.events_from_annotations(raw, verbose=False)
            ch_idx = (raw.ch_names.index(canal)
                      if canal in raw.ch_names else 0)
            data   = raw.get_data()[ch_idx]
            n_pts  = int(fs)
            for ev in events:
                onset, codigo = ev[0], ev[2]
                if onset + n_pts > len(data):
                    continue
                seg = data[onset:onset + n_pts]
                seg = seg / (np.std(seg) + 1e-8)
                # reamostrar para n_amostras se necessário
                if len(seg) != n_amostras:
                    idx = np.linspace(0, len(seg) - 1, n_amostras).astype(int)
                    seg = seg[idx]
                if   codigo == 1: T1.append(seg)
                elif codigo == 2: T2.append(seg)
        except Exception:
            continue
    return np.array(T1), np.array(T2)

# ══════════════════════════════════════════════════════════════════════════════
# DEMO — Scanner α-φ + Eco-Beep sobre EEG real
# ══════════════════════════════════════════════════════════════════════════════

# Bandas φ-proporcionais para EEG
BANDAS = gerar_bandas_phi_eeg(f_min=0.5, fs=FS_EEG)
BINS   = bandas_para_bins_eeg(BANDAS, n=FS_EEG, fs=FS_EEG)

print(f"\nBandas φ-proporcionais ({len(BANDAS)} bandas, 0.5 Hz → {FS_EEG//2} Hz):")
for i, (b_lo, b_hi, f_lo, f_hi) in enumerate(BINS):
    print(f"  F{i+1:02d}  {f_lo:6.2f}–{f_hi:6.2f} Hz  [{nome_banda(f_lo, f_hi)}]"
          f"  bins {b_lo}–{b_hi}")

print(f"\nIniciando scan sobre {N_SUJ} sujeitos...\n")

resultados = []
bandas_otimas = []

for suj_id in range(1, N_SUJ + 1):
    print(f"── Sujeito S{suj_id:03d} ──────────────────────────────────")

    arquivos = baixar_sujeito(suj_id)
    if not arquivos:
        print(f"  S{suj_id:03d}: download falhou — registrado, não descartado")
        resultados.append({"suj": suj_id, "status": "download_falhou"})
        continue

    T1, T2 = carregar_epochs(arquivos)

    if len(T1) < 5 or len(T2) < 5:
        print(f"  S{suj_id:03d}: épocas insuficientes T1={len(T1)} T2={len(T2)}")
        resultados.append({"suj": suj_id, "status": "epochs_insuficientes",
                           "n_T1": len(T1), "n_T2": len(T2)})
        continue

    print(f"  Épocas: T1={len(T1)}  T2={len(T2)}")

    # Rodar Scanner por N_SONDA_MAX ciclos usando subconjuntos de épocas
    scanner = ScannerEEGPhiBands(BANDAS, BINS)
    rng = np.random.default_rng(TIMESTAMP + suj_id * 137)

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
    print(f"  Fase ótima : F{r['fase_otima']:02d} — {r['banda']}"
          f"  ({r['f_lo']:.2f}–{r['f_hi']:.2f} Hz)")
    print(f"  meta_coh   : {r['meta_coh']:.4f}  ciclos={r['n_ciclos']}"
          f"  C={r['capacitancia']:.3f}  β={r['beta_atual']:.3f}  {tag}")

    # Perfil completo de scores por banda
    print("  Scores por banda:")
    for p in r['perfil']:
        marker = " ◀ ótima" if p['fase'] == r['fase_otima'] else ""
        print(f"    F{p['fase']:02d} {p['nome']:8s}"
              f" ({p['f_lo']:5.1f}–{p['f_hi']:5.1f} Hz)"
              f"  coh={p['coh']:.3f}  disc={p['disc']:.3f}"
              f"  S={p['score']:.5f}{marker}")

    resultados.append({"suj": suj_id, "status": "ok", **r})
    bandas_otimas.append(r['banda'])
    print()

# ══════════════════════════════════════════════════════════════════════════════
# SÍNTESE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("SÍNTESE — SCANNER α-φ + ECO-BEEP sobre EEG")
print("=" * 65)

ok = [r for r in resultados if r.get("status") == "ok"]
print(f"\nSujeitos processados : {len(resultados)}")
print(f"Convergidos (OK)     : {len(ok)}")
print(f"Não convergidos      : {sum(1 for r in resultados if r.get('status') == 'nao_convergiu')}")
print(f"Falha de dados       : {sum(1 for r in resultados if r.get('status') != 'ok' and r.get('status') != 'nao_convergiu')}")

if ok:
    mc_med  = np.mean([r['meta_coh']    for r in ok])
    c_med   = np.mean([r['capacitancia'] for r in ok])
    adequados = sum(r['adequado'] for r in ok)
    print(f"\nmeta_coh médio       : {mc_med:.4f}  (limiar={ScannerEEGPhiBands.META_COH_LIMIAR})")
    print(f"capacitância média   : {c_med:.4f}")
    print(f"substrato adequado   : {adequados}/{len(ok)}")

if bandas_otimas:
    from collections import Counter
    contagem = Counter(bandas_otimas)
    print(f"\nDistribuição de bandas ótimas:")
    for banda, n in contagem.most_common():
        pct = 100 * n / len(bandas_otimas)
        print(f"  {banda:10s} : {n:2d} sujeitos ({pct:.0f}%)")

    banda_dom = contagem.most_common(1)[0][0]
    print(f"\n  Banda dominante: {banda_dom}")
    if banda_dom == "Alpha":
        print(f"  → α encontrou residência natural nas ondas Alpha cerebrais.")
        print(f"    Resultado notável — registro como observação, não confirmação.")
    else:
        print(f"  → α encontrou residência em {banda_dom}.")
        print(f"    Resultado igualmente válido — informa onde α reside neste substrato.")

print(f"\nφ = {PHI:.6f}   α = {ALPHA:.10f}")
print("=" * 65)
print(f"\nalpha-phi | eeg-phibands | scanner | {TIMESTAMP}")
print("=" * 65)
