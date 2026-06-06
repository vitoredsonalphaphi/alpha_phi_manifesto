"""
analise_longa_colab.py
─────────────────────────────────────────────────────────────────────
Analisador de gravação noturna longa (até 10h).
Detecta, classifica e mapeia eventos acústicos ao longo da noite.

Uso no Google Colab:
  1. Monte o Google Drive:
       from google.colab import drive; drive.mount('/content/drive')
  2. Ajuste CAMINHO_AUDIO abaixo
  3. Execute a célula

Dependências (instaladas automaticamente):
  numpy, scipy, av (PyAV para M4A)
─────────────────────────────────────────────────────────────────────
"""

import sys, os, subprocess, time
from collections import Counter, defaultdict

# ── Instalar dependências se necessário ───────────────────────────
def _instalar():
    pkgs = []
    try: import numpy
    except ImportError: pkgs.append("numpy")
    try: import scipy
    except ImportError: pkgs.append("scipy")
    try: import av
    except ImportError: pkgs.append("av")
    if pkgs:
        print(f"Instalando: {' '.join(pkgs)} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
        print("OK\n")

_instalar()

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, hilbert, butter, sosfiltfilt
from scipy.cluster.vq import kmeans, vq, whiten

# ══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO — ajuste aqui
# ══════════════════════════════════════════════════════════════════
CAMINHO_AUDIO = "/content/drive/MyDrive/gravacao_noturna.m4a"

JANELA_S     = 30    # janela de varredura em segundos
ATIV_MIN     = 5     # cliques mínimos para janela ser "ativa"
K_ESTADOS    = 4     # estados acústicos (A B C D)
BURST_GAP_MS = 100   # gap em ms para separar bursts
HP_FREQ      = 800   # filtro passa-alta (Hz)
WIN_MS       = 5     # janela por clique (ms)
MIN_IV_MS    = 3     # intervalo mínimo entre cliques (ms)
TKEO_THRESH  = 0.04
TKEO_PROM    = 0.03
NGRAM        = 4     # comprimento máximo de n-grama

# Sequências de referência — Campo Largo (Galeria do Meteorito)
REF_B5 = "AAAAAAABDBDDDDDBBBBBBBDDBBBBBBB"
REF_B6 = "DDDDDBBBBBBBBBBBBBBBBBBBBCCC"
REF_SEQ = REF_B5 + REF_B6

PHI   = 1.6180339887
ALPHA = 1 / 137.035999
SIMB  = list("ABCD")
BANDS = [(800,1500),(1500,3000),(3000,4500),(4500,6000),
         (6000,8000),(8000,12000),(12000,16000),(16000,20000)]

# ══════════════════════════════════════════════════════════════════
# CONVERSÃO DE ÁUDIO
# ══════════════════════════════════════════════════════════════════
def _converter_para_wav(src):
    dst = "/tmp/_anlonga_audio.wav"
    if os.path.exists(dst): os.remove(dst)
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "44100", "-ac", "1", dst],
        capture_output=True
    )
    if r.returncode == 0: return dst
    # fallback PyAV
    import av
    container = av.open(src)
    chunks = []
    sr_av = None
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray()
        if arr.ndim > 1: arr = arr.mean(axis=0)
        chunks.append(arr.astype(np.float32))
        if sr_av is None: sr_av = frame.sample_rate
    if not chunks: raise RuntimeError("Não foi possível decodificar o áudio")
    data = np.concatenate(chunks)
    data = (data / (max(np.abs(data).max(), 1e-6)) * 32767).astype(np.int16)
    wavfile.write(dst, sr_av, data)
    return dst

# ══════════════════════════════════════════════════════════════════
# PIPELINE ACÚSTICO
# ══════════════════════════════════════════════════════════════════
def tkeo(x): return x[1:-1]**2 - x[:-2]*x[2:]

def filtrar(d, sr):
    sos = butter(6, HP_FREQ, btype="high", fs=sr, output="sos")
    return sosfiltfilt(sos, d)

def detectar(d, sr):
    tk = np.abs(tkeo(d)); tk_n = tk / (tk.max() + 1e-10)
    p, _ = find_peaks(tk_n, height=TKEO_THRESH,
                      distance=int(MIN_IV_MS * 1e-3 * sr), prominence=TKEO_PROM)
    return p

def feature(chunk, sr):
    if len(chunk) < 16: return None
    spec = np.abs(np.fft.rfft(chunk, n=4096)) ** 2
    freqs = np.fft.rfftfreq(4096, 1/sr); tot = spec.sum() + 1e-30
    be = [spec[(freqs >= fl) & (freqs < fh)].sum() / tot for fl, fh in BANDS]
    an = hilbert(chunk)
    ifreq = np.diff(np.unwrap(np.angle(an))) / (2 * np.pi) * sr
    q4 = max(1, len(chunk) // 4)
    chirp = (np.median(np.abs(ifreq[-q4:])) - np.median(np.abs(ifreq[:q4]))) / 1e4 \
            if len(ifreq) >= 2 * q4 else 0.0
    mc = (freqs > 500) & (freqs < 18000)
    cent = float(np.sum(freqs[mc] * spec[mc]) / (spec[mc].sum() + 1e-30)) / 1e4
    pkf = np.argmax(np.abs(tkeo(chunk))) / max(1, len(chunk) - 2)
    return be + [chirp, cent, pkf]

def agrupar(peaks, sr):
    if not len(peaks): return []
    bs, cur = [], [peaks[0]]
    for p in peaks[1:]:
        if (p - cur[-1]) / sr > BURST_GAP_MS / 1000: bs.append(cur); cur = [p]
        else: cur.append(p)
    bs.append(cur)
    return bs

def classificar_janela(data, sr, cent_ref=None):
    """Retorna (peaks, labels, feats, bursts, cents)."""
    peaks = detectar(data, sr)
    if len(peaks) < ATIV_MIN: return None, None, None, None, None
    WIN = int(WIN_MS * 1e-3 * sr)
    feats, vp = [], []
    for p in peaks:
        s = max(0, p - WIN // 2); e = min(len(data), s + WIN)
        f = feature(data[s:e], sr)
        if f: feats.append(f); vp.append(p)
    if len(feats) < ATIV_MIN: return None, None, None, None, None
    F = np.array(feats); Fw = whiten(F)
    np.random.seed(42)
    if cent_ref is not None:
        cents = cent_ref; labs, _ = vq(Fw, cents)
    else:
        cents, _ = kmeans(Fw, K_ESTADOS, iter=300); labs, _ = vq(Fw, cents)
    cr = np.array([F[labs == k].mean(axis=0) if (labs == k).any()
                   else np.zeros(F.shape[1]) for k in range(K_ESTADOS)])
    order = np.argsort(cr[:, 2] * 0.3 + cr[:, 3] * 0.7)
    remap = {o: n for n, o in enumerate(order)}
    labs = np.array([remap[l] for l in labs])
    bs = agrupar(np.array(vp), sr)
    return np.array(vp), labs, F, bs, cents

# ══════════════════════════════════════════════════════════════════
# SIMILARIDADE COM REFERÊNCIA
# ══════════════════════════════════════════════════════════════════
def similaridade_lcs(a, b):
    """Longest Common Subsequence normalizado."""
    if not a or not b: return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n] / max(m, n)

def score_ngrama(seq, ref, n=3):
    """Fração de n-gramas da ref encontrados em seq."""
    if len(seq) < n or len(ref) < n: return 0.0
    ref_ng = set(''.join(ref[i:i+n]) for i in range(len(ref)-n+1))
    seq_ng = [''.join(seq[i:i+n]) for i in range(len(seq)-n+1)]
    if not seq_ng: return 0.0
    return sum(1 for g in seq_ng if g in ref_ng) / len(seq_ng)

# ══════════════════════════════════════════════════════════════════
# PERIODICIDADE φ e α
# ══════════════════════════════════════════════════════════════════
def checar_periodicidade(tempos_s):
    """Verifica se intervalos entre eventos seguem série φⁿ."""
    if len(tempos_s) < 3: return None
    ivs = np.diff(tempos_s)
    if len(ivs) < 2: return None
    ratios = ivs[1:] / (ivs[:-1] + 1e-10)
    dev_phi   = np.abs(ratios - PHI).mean()
    dev_1phi  = np.abs(ratios - 1/PHI).mean()
    dev_alpha = np.abs(ratios - ALPHA * 100).mean()  # escala α para comparação
    best = min(dev_phi, dev_1phi)
    return {
        "ratios": ratios.tolist(),
        "dev_phi": float(dev_phi),
        "dev_1phi": float(dev_1phi),
        "media_ratio": float(ratios.mean()),
        "sinal_phi": best < 0.3
    }

# ══════════════════════════════════════════════════════════════════
# ANÁLISE PRINCIPAL
# ══════════════════════════════════════════════════════════════════
def analisar_noite():
    print("="*65)
    print("  ANALISADOR DE GRAVAÇÃO NOTURNA — eco_decoder longa duração")
    print("="*65)

    if not os.path.exists(CAMINHO_AUDIO):
        print(f"\n[ERRO] Arquivo não encontrado: {CAMINHO_AUDIO}")
        print("  Monte o Google Drive e ajuste CAMINHO_AUDIO no início do código.")
        return

    ext = os.path.splitext(CAMINHO_AUDIO)[1].lower()
    print(f"\nArquivo : {os.path.basename(CAMINHO_AUDIO)}")
    print(f"Tamanho : {os.path.getsize(CAMINHO_AUDIO)/1e6:.1f} MB")

    if ext in (".wav",):
        wav_path = CAMINHO_AUDIO
        print("Formato : WAV (sem conversão)\n")
    else:
        print(f"Convertendo {ext.upper()} → WAV ...")
        t0 = time.time()
        wav_path = _converter_para_wav(CAMINHO_AUDIO)
        print(f"Conversão OK em {time.time()-t0:.1f}s — {wav_path}\n")

    print("Carregando WAV...", end=" ", flush=True)
    sr, raw = wavfile.read(wav_path)
    if raw.ndim > 1: raw = raw[:, 0]
    raw = raw.astype(np.float64)
    if raw.max() > 1.0: raw /= np.iinfo(np.int16).max
    dur_total = len(raw) / sr
    horas = int(dur_total // 3600)
    mins  = int((dur_total % 3600) // 60)
    print(f"OK — {dur_total:.0f}s ({horas}h{mins:02d}m) | SR={sr}Hz")

    print(f"\nVarrendo em janelas de {JANELA_S}s...")
    n_janelas = int(np.ceil(dur_total / JANELA_S))
    eventos = []          # (t_inicio_s, seq, n_bursts, n_clicks, sim_ref)
    timeline_hora = defaultdict(list)  # hora → lista de atividade

    t_ini_global = time.time()
    cent_global = None    # centroides aprendidos na primeira janela ativa

    for jj in range(n_janelas):
        t_ini = jj * JANELA_S
        t_fim = min(t_ini + JANELA_S, dur_total)
        s0 = int(t_ini * sr); s1 = int(t_fim * sr)
        chunk = raw[s0:s1]

        if len(chunk) < sr: continue  # < 1s, pular

        filt = filtrar(chunk, sr)
        vp, labs, F, bs, cents = classificar_janela(filt, sr, cent_global)

        hora = int(t_ini // 3600)
        if vp is None:
            timeline_hora[hora].append("·")
            if (jj + 1) % 50 == 0:
                elapsed = time.time() - t_ini_global
                print(f"  [{jj+1:4d}/{n_janelas}] t={t_ini/3600:.2f}h  "
                      f"eventos={len(eventos)}  {elapsed:.0f}s decorridos")
            continue

        # Aproveitar os centroides da primeira janela ativa como referência
        if cent_global is None:
            cent_global = cents
            print(f"  Centroides de referência definidos na janela t={t_ini:.0f}s")

        seq = [SIMB[l] for l in labs]
        seq_str = "".join(seq)
        sim_b5  = similaridade_lcs(seq, list(REF_B5))
        sim_b6  = similaridade_lcs(seq, list(REF_B6))
        sim_ref = max(sim_b5, sim_b6)
        ng3     = score_ngrama(seq, list(REF_SEQ), n=3)

        ev = {
            "t_s": t_ini,
            "seq": seq_str,
            "n_cl": len(vp),
            "n_bs": len(bs) if bs else 0,
            "sim": sim_ref,
            "ng3": ng3,
            "dist": Counter(seq),
        }
        eventos.append(ev)
        timeline_hora[hora].append("█" if sim_ref > 0.5 else "▓" if sim_ref > 0.3 else "░")

        elapsed = time.time() - t_ini_global
        if (jj + 1) % 50 == 0 or sim_ref > 0.4:
            flag = " ◄ ALTO" if sim_ref > 0.4 else ""
            print(f"  [{jj+1:4d}/{n_janelas}] t={t_ini/3600:.2f}h  "
                  f"cl={len(vp):3d}  bs={len(bs)}  sim={sim_ref:.2f}{flag}  {elapsed:.0f}s")

    print(f"\nVarredura concluída em {time.time()-t_ini_global:.1f}s")
    print(f"Janelas ativas: {len(eventos)}/{n_janelas}")

    # ── RELATÓRIO ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  MAPA DA NOITE  [·=silêncio  ░=baixo  ▓=médio  █=alto match]")
    print(f"{'='*65}")
    max_h = max(timeline_hora.keys()) if timeline_hora else 0
    for h in range(max_h + 1):
        bar = "".join(timeline_hora[h]) if h in timeline_hora else ""
        ativos = bar.count("░") + bar.count("▓") + bar.count("█")
        print(f"  {h:02d}h [{bar[:80]:<80}] ativ={ativos}")

    if not eventos:
        print("\n  Nenhum evento detectado na gravação.")
        print("="*65)
        return

    # ── TOP EVENTOS ───────────────────────────────────────────────
    top_sim = sorted(eventos, key=lambda e: e["sim"], reverse=True)[:10]
    print(f"\n{'─'*65}\n  TOP {min(10,len(top_sim))} EVENTOS — maior similaridade com Campo Largo\n{'─'*65}")
    for i, ev in enumerate(top_sim, 1):
        hh = int(ev["t_s"] // 3600); mm = int((ev["t_s"] % 3600) // 60); ss = int(ev["t_s"] % 60)
        print(f"  {i:2d}. {hh:02d}h{mm:02d}m{ss:02d}s  cl={ev['n_cl']:3d}  bs={ev['n_bs']}  "
              f"sim={ev['sim']:.3f}  ng3={ev['ng3']:.3f}")
        print(f"      seq: {ev['seq'][:60]}{'...' if len(ev['seq'])>60 else ''}")

    # ── DISTRIBUIÇÃO TEMPORAL ─────────────────────────────────────
    print(f"\n{'─'*65}\n  DENSIDADE DE EVENTOS POR HORA\n{'─'*65}")
    ev_por_hora = defaultdict(int)
    for ev in eventos: ev_por_hora[int(ev["t_s"] // 3600)] += 1
    max_ev = max(ev_por_hora.values()) if ev_por_hora else 1
    for h in sorted(ev_por_hora):
        bar = "█" * int(ev_por_hora[h] / max_ev * 40)
        print(f"  {h:02d}h  {bar:<40} {ev_por_hora[h]:3d} janelas ativas")

    # ── PERIODICIDADE φ ───────────────────────────────────────────
    tempos_eventos = sorted([ev["t_s"] for ev in eventos])
    per = checar_periodicidade(tempos_eventos)
    print(f"\n{'─'*65}\n  PERIODICIDADE ENTRE EVENTOS\n{'─'*65}")
    if per:
        print(f"  Razão média entre intervalos consecutivos: {per['media_ratio']:.4f}")
        print(f"  Desvio de φ={PHI:.4f}  : {per['dev_phi']:.4f}")
        print(f"  Desvio de 1/φ={1/PHI:.4f}: {per['dev_1phi']:.4f}")
        if per["sinal_phi"]:
            print(f"  ► SINAL φ DETECTADO — periodicidade consistente com razão áurea")
        else:
            print(f"  ► Sem periodicidade φ clara (desvio > 0.3)")
    else:
        print("  Poucos eventos para análise de periodicidade.")

    # ── N-GRAMAS GLOBAIS ──────────────────────────────────────────
    seq_global = "".join(ev["seq"] for ev in eventos)
    print(f"\n{'─'*65}\n  N-GRAMAS MAIS FREQUENTES (gravação completa)\n{'─'*65}")
    print(f"  Total de símbolos: {len(seq_global)}")
    for n in range(2, min(NGRAM + 1, len(seq_global))):
        top = Counter([seq_global[i:i+n] for i in range(len(seq_global)-n+1)]).most_common(5)
        if not top or top[0][1] < 3: break
        print(f"  {n}-gram: " + "  ".join(f"'{g}'×{c}" for g, c in top))

    # ── DISTRIBUIÇÃO DE ESTADOS ───────────────────────────────────
    dist_global = Counter(seq_global)
    n_gl = len(seq_global) or 1
    print(f"\n{'─'*65}\n  DISTRIBUIÇÃO GLOBAL DE ESTADOS\n{'─'*65}")
    for s in SIMB:
        pct = dist_global.get(s, 0) / n_gl * 100
        bar = "█" * int(pct / 2)
        print(f"  {s}: {bar:<50} {pct:4.1f}%  (n={dist_global.get(s,0)})")

    # ── COMPARAÇÃO COM CAMPO LARGO ────────────────────────────────
    sim_gl_b5 = similaridade_lcs(list(seq_global[:200]), list(REF_B5))
    sim_gl_b6 = similaridade_lcs(list(seq_global[:200]), list(REF_B6))
    print(f"\n{'─'*65}\n  COMPARAÇÃO COM REFERÊNCIA CAMPO LARGO\n{'─'*65}")
    print(f"  Ref B5 (A→B→D): {REF_B5}")
    print(f"  Ref B6 (D→B→C): {REF_B6}")
    print(f"  Similaridade B5 (200 primeiros símbolos da noite): {sim_gl_b5:.3f}")
    print(f"  Similaridade B6 (200 primeiros símbolos da noite): {sim_gl_b6:.3f}")
    if max(sim_gl_b5, sim_gl_b6) > 0.5:
        print("  ► PADRÃO COMPATÍVEL com referência Campo Largo")
    else:
        print("  ► Padrão divergente da referência Campo Largo")

    print(f"\n{'='*65}\n  FIM DO RELATÓRIO NOTURNO\n{'='*65}")
    return eventos

# ── EXECUTAR ──────────────────────────────────────────────────────
resultado = analisar_noite()
