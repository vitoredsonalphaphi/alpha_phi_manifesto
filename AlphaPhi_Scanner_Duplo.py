# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_Scanner_Duplo.py
Vitor Edson Delavi · Florianópolis · 2026

Protocolo E302 — Dois Scanners Coadjuvantes:

  Scanner 1: Topográfico Lissajous
    T(ω,τ) + ∇S (existente) + Lissajous inter-bandas (novo)
    Diagnóstico: acoplamento entre φ-bandas adjacentes

  Scanner 2: Cepstral Topográfico
    T_φ 9×9 + S(k)/C(k) (existente) + evolução temporal T_φ (novo)
    Diagnóstico: auto-acoplamento espectro-cepstral por frame

Protocolo: scan_duplo(sinal) ativa os dois juntos.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# ── Constantes ────────────────────────────────────────────────────────────────
PHI        = 1.6180339887
ALPHA      = 1 / 137.035999
ALPHA_STAR = 1 / 3          # α* efetivo para áudio
SEAL       = 1 / PHI
FS         = 22050
N_WIN      = 2048
HOP        = 256
N_BAND     = 9
N_BINS     = 100

# Paleta Alpha-Phi
_AP_COLORS = ['#0E1E48', '#1E4A90', '#1A4A80', '#C8960A', '#F5D55A', '#FFE89A']
CMAP_AP    = LinearSegmentedColormap.from_list('alphaphi', _AP_COLORS)


# ── Utilitários ───────────────────────────────────────────────────────────────

def _norm(x):
    mx = np.max(np.abs(x))
    return x / mx if mx > 1e-12 else x


def phi_bands(n_fft, fs):
    fr   = np.fft.rfftfreq(n_fft, 1 / fs)
    fmax = fs / 2
    fmin = fs / n_fft
    f    = fmin * PHI
    bands = []
    while f < fmax:
        fh = min(f * PHI, fmax)
        bl = int(np.searchsorted(fr, f))
        bh = int(np.searchsorted(fr, fh))
        if bh > bl:
            bands.append((bl, bh, f, fh))
        if fh >= fmax:
            break
        f = fh
    return bands


def _style_ax(ax):
    ax.set_facecolor('#0D1525')
    for sp in ax.spines.values():
        sp.set_color('#2E4055')
    ax.tick_params(colors='#2E4055', labelsize=5)


# ── Métricas base ─────────────────────────────────────────────────────────────

def grade_r_matriz(sig):
    """
    Coerência por φ-banda por frame.
    Retorna array (n_frames, n_band).
    """
    bands = phi_bands(N_WIN, FS)[:N_BAND]
    n_band = len(bands)
    win    = np.hanning(N_WIN)
    n_sig  = len(sig)
    rows   = []
    step   = HOP * 4      # subsample para velocidade
    for p in range(0, n_sig - N_WIN + 1, step):
        frame = sig[p:p + N_WIN] * win
        F     = np.fft.rfft(frame)
        row   = []
        for bl, bh, _, __ in bands:
            if bh <= bl or bh > len(F):
                row.append(0.0)
                continue
            mag = np.abs(F[bl:bh])
            if len(mag) < 2:
                row.append(0.0)
                continue
            an  = np.clip(mag / (mag.sum() + 1e-8), 1e-10, 1.0)
            row.append(float(1.0 - (-np.sum(an * np.log(an))) / np.log(len(an))))
        rows.append(row)
    if not rows:
        return np.zeros((1, n_band))
    return np.array(rows)   # (n_frames, n_band)


def tensor_topo(x, n_bins=N_BINS):
    """T(ω,τ) = S(ω) ⊗ C(τ) — produto externo espectro × cepstro."""
    F        = np.fft.rfft(x)
    esp      = np.abs(F)[1:n_bins + 1]
    log_spec = np.log(np.abs(F) + 1e-9)
    cep      = np.abs(np.fft.irfft(log_spec))[1:n_bins + 1]
    Sn       = (esp - esp.min()) / (esp.max() - esp.min() + 1e-9)
    Cn       = (cep - cep.min()) / (cep.max() - cep.min() + 1e-9)
    return np.outer(Sn, Cn)


def tensor_cepstral_phi(x):
    """
    T_φ(k,m) = S_banda_k × C_banda_m  →  matriz 9×9 φ-geométrica.
    Retorna (T_phi, S_banda, C_banda).
    """
    n_fft  = len(x)
    F      = np.fft.rfft(x)
    amp    = np.abs(F)
    amp_n  = amp / (amp.max() + 1e-12)
    log_s  = np.log(amp_n + 1e-9)
    cep    = np.abs(np.fft.irfft(log_s, n_fft))
    cep_n  = cep / (cep.max() + 1e-12)
    bands  = phi_bands(n_fft, FS)[:N_BAND]
    S_b    = np.zeros(N_BAND)
    C_b    = np.zeros(N_BAND)
    for i, (bl, bh, _, __) in enumerate(bands):
        if i >= N_BAND:
            break
        if bh > bl and bh <= len(amp_n):
            S_b[i] = amp_n[bl:bh].mean()
        ql, qh = bl, min(bh, n_fft // 2)
        if qh > ql:
            C_b[i] = cep_n[ql:qh].mean()
    return np.outer(S_b, C_b), S_b, C_b


# ── Scanner 1: Topográfico Lissajous ─────────────────────────────────────────

def scanner_topografico_lissajous(sig, titulo="Scanner Topográfico Lissajous"):
    """
    Painel 1: T(ω,τ)
    Painel 2: ∇S (respiração das células)
    Painéis 3+: Lissajous inter-bandas para cada par de φ-bandas adjacentes
    """
    T        = tensor_topo(sig)
    G        = T - gaussian_filter(T, sigma=2.5)
    coh_mat  = grade_r_matriz(sig)      # (n_frames, n_band)
    n_frames, n_band = coh_mat.shape
    n_pairs  = min(n_band - 1, 8)
    n_cols   = max(n_pairs, 2)

    fig = plt.figure(figsize=(max(n_cols * 1.8, 12), 9), facecolor='#060610')
    fig.suptitle(titulo, color='white', fontsize=10, fontweight='bold', y=1.00)

    outer = mgs.GridSpec(2, 1, figure=fig, hspace=0.52, height_ratios=[1, 1])

    # ── Linha 0: T(ω,τ) e ∇S ──────────────────────────────────────────────
    row0 = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.30)

    ax_t = fig.add_subplot(row0[0])
    _style_ax(ax_t)
    im_t = ax_t.imshow(np.log1p(T * 100).T, aspect='auto', origin='lower',
                       cmap='inferno', interpolation='bilinear')
    ax_t.set_title('T(ω,τ) — Acoplamento Espectro·Cepstro', color='white', fontsize=8)
    ax_t.set_xlabel('ω', color='#666666', fontsize=7)
    ax_t.set_ylabel('τ', color='#666666', fontsize=7)
    fig.colorbar(im_t, ax=ax_t, fraction=0.04).ax.tick_params(colors='#2E4055', labelsize=4)

    ax_g = fig.add_subplot(row0[1])
    _style_ax(ax_g)
    vmax = max(float(np.percentile(np.abs(G), 98)), 1e-9)
    im_g = ax_g.imshow(G.T, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax_g.set_title('∇S — Respiração  (verm=vértice · azul=alívio)', color='#ffcc00', fontsize=8)
    ax_g.set_xlabel('ω', color='#666666', fontsize=7)
    ax_g.set_ylabel('τ', color='#666666', fontsize=7)
    fig.colorbar(im_g, ax=ax_g, fraction=0.04).ax.tick_params(colors='#2E4055', labelsize=4)

    # ── Linha 1: Lissajous inter-bandas ───────────────────────────────────
    row1 = mgs.GridSpecFromSubplotSpec(1, n_pairs, subplot_spec=outer[1], wspace=0.35)
    cores = plt.cm.plasma(np.linspace(0, 1, max(n_frames, 2)))

    acoplamentos = {}
    for k in range(n_pairs):
        ax = fig.add_subplot(row1[k])
        _style_ax(ax)
        x = coh_mat[:, k]
        y = coh_mat[:, k + 1]
        for i in range(max(n_frames - 1, 1)):
            ax.plot(x[i:i + 2], y[i:i + 2], color=cores[i], lw=0.9, alpha=0.75)
        r = (float(np.corrcoef(x, y)[0, 1])
             if x.std() > 1e-6 and y.std() > 1e-6 else 0.0)
        acoplamentos[f'B{k}·B{k+1}'] = r
        cor_r = '#4AFFE8' if abs(r) > 0.6 else ('#F5D55A' if abs(r) > 0.3 else '#E84040')
        ax.set_title(f'B{k}·B{k+1}\nr={r:+.2f}', color=cor_r, fontsize=7, pad=2)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        # Círculo de referência (acoplamento ideal)
        theta = np.linspace(0, 2 * np.pi, 120)
        ax.plot(0.5 + 0.45 * np.cos(theta), 0.5 + 0.45 * np.sin(theta),
                '--', color='#4AFFE8', lw=0.5, alpha=0.25)
        ax.set_xlabel(f'coh B{k}', color='#2E4055', fontsize=5)
        ax.set_ylabel(f'coh B{k+1}', color='#2E4055', fontsize=5)

    out = 'Scanner1_Topografico_Lissajous.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#060610')
    plt.close(fig)
    print(f"[Scanner 1] → {out}")
    return out, coh_mat, acoplamentos


# ── Scanner 2: Cepstral Topográfico ──────────────────────────────────────────

def scanner_cepstral_topografico(sig, titulo="Scanner Cepstral Topográfico"):
    """
    Painel 1: T_φ(k,m) 9×9
    Painel 2: S(k) e C(k) por banda φ
    Painel 3: diagonal T_φ (auto-acoplamento)
    Painel 4: evolução temporal T_φ(k,k) por frame (novo)
    """
    # T_φ a partir de um frame representativo (meio do sinal)
    mid  = max(0, len(sig) // 2 - N_WIN // 2)
    mid  = min(mid, len(sig) - N_WIN)
    frame_rep = sig[mid:mid + N_WIN] * np.hanning(N_WIN)
    T_phi, S_b, C_b = tensor_cepstral_phi(frame_rep)

    # Evolução temporal: diagonal T_φ por frame
    n_sig      = len(sig)
    win_arr    = np.hanning(N_WIN)
    frame_step = HOP * 16
    diag_evol  = []
    for p in range(0, n_sig - N_WIN + 1, frame_step):
        f   = sig[p:p + N_WIN] * win_arr
        Tf, _, _ = tensor_cepstral_phi(f)
        diag_evol.append(np.diag(Tf))
    evol = np.array(diag_evol).T if diag_evol else np.zeros((N_BAND, 1))

    fig = plt.figure(figsize=(15, 9), facecolor='#070810')
    fig.suptitle(titulo, color='#C8BBAA', fontsize=10, fontweight='bold', y=1.00)

    outer = mgs.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.30)
    ks    = np.arange(N_BAND)

    # Painel 1: T_φ 9×9
    ax1 = fig.add_subplot(outer[0, 0])
    _style_ax(ax1)
    im1 = ax1.imshow(T_phi, aspect='auto', cmap=CMAP_AP, origin='lower')
    ax1.set_title('T_φ(k,m) — 9×9 bandas φ', color='#C8BBAA', fontsize=8)
    ax1.set_xticks(ks)
    ax1.set_yticks(ks)
    ax1.set_xticklabels([f'C{i}' for i in ks], color='#2E4055', fontsize=5)
    ax1.set_yticklabels([f'S{i}' for i in ks], color='#2E4055', fontsize=5)
    fig.colorbar(im1, ax=ax1, fraction=0.04).ax.tick_params(colors='#2E4055', labelsize=4)

    # Painel 2: S(k) e C(k)
    ax2 = fig.add_subplot(outer[0, 1])
    _style_ax(ax2)
    ax2.plot(ks, S_b, 'o-', color='#F5D55A', lw=1.5, ms=5, label='S(k) espectro')
    ax2.plot(ks, C_b, 's--', color='#4AFFE8', lw=1.5, ms=4, label='C(k) cepstro')
    ax2.set_title('S(k) e C(k) por banda φ', color='#C8BBAA', fontsize=8)
    ax2.legend(facecolor='#0D1525', edgecolor='#2E4055',
               labelcolor='#C8BBAA', fontsize=7)
    ax2.set_xlabel('Banda k', color='#2E4055', fontsize=7)

    # Painel 3: diagonal T_φ
    ax3 = fig.add_subplot(outer[0, 2])
    _style_ax(ax3)
    diag = np.diag(T_phi)
    ax3.plot(ks, diag, 'o-', color='#F5D55A', lw=1.5, ms=5)
    ax3.set_title('T_φ(k,k) — Auto-acoplamento', color='#C8BBAA', fontsize=8)
    ax3.set_xlabel('Banda k', color='#2E4055', fontsize=7)

    # Painel 4: evolução temporal (nova camada topográfica)
    ax4 = fig.add_subplot(outer[1, :])
    _style_ax(ax4)
    if evol.shape[1] > 1:
        for k in range(N_BAND):
            color = CMAP_AP(k / max(N_BAND - 1, 1))
            ax4.plot(evol[k], lw=1.2, color=color, alpha=0.85, label=f'B{k}')
        ax4.set_title(
            'Evolução Temporal — T_φ(k,k) por frame OLA  '
            '(topografia cepstral ao longo do tempo)',
            color='#C8BBAA', fontsize=8)
        ax4.set_xlabel('Frame OLA', color='#2E4055', fontsize=7)
        ax4.set_ylabel('Auto-acoplamento', color='#2E4055', fontsize=7)
        ax4.legend(facecolor='#0D1525', edgecolor='#2E4055',
                   labelcolor='#C8BBAA', fontsize=6, ncol=5, loc='upper right')
    else:
        ax4.text(0.5, 0.5, 'sinal curto demais para evolução temporal',
                 color='#C8BBAA', ha='center', va='center', transform=ax4.transAxes)

    out = 'Scanner2_Cepstral_Topografico.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#070810')
    plt.close(fig)
    print(f"[Scanner 2] → {out}")
    return out


# ── Protocolo E302: ambos juntos ──────────────────────────────────────────────

def scan_duplo(sig, nome="sinal"):
    """
    Protocolo E302: sempre que acionado, roda os dois scanners.
    Scanner 1 → Topográfico Lissajous (inter-bandas)
    Scanner 2 → Cepstral Topográfico  (evolução temporal)
    """
    print(f"\n{'═' * 64}")
    print(f"  Scanner Duplo Alpha-Phi · {nome}")
    print('═' * 64)

    out1, coh_mat, acoplamentos = scanner_topografico_lissajous(
        sig, f"Scanner Topográfico Lissajous · {nome}")
    out2 = scanner_cepstral_topografico(
        sig, f"Scanner Cepstral Topográfico · {nome}")

    # ── Relatório de acoplamento Lissajous ────────────────────────────────
    print("\n  Acoplamento inter-bandas (Pearson r):")
    for par, r in acoplamentos.items():
        if abs(r) > 0.6:
            estado = '✓ acoplado'
        elif abs(r) > 0.3:
            estado = '~ parcial '
        else:
            estado = '○ independente'
        print(f"  {par}: r={r:+.3f}  {estado}")

    # ── Grade R global ────────────────────────────────────────────────────
    grade_global = float(coh_mat.mean())
    print(f"\n  Grade R global (média): {grade_global:.4f}")
    print(f"  Frames analisados: {coh_mat.shape[0]}  ·  Bandas: {coh_mat.shape[1]}")
    print(f"\n  → {out1}")
    print(f"  → {out2}")
    print()
    return out1, out2


# ── Sinais de teste ───────────────────────────────────────────────────────────

def gerar_hdr(seed=42):
    """Sinal HDR 3-épocas: E1(0.05) E2(1.00) E3(0.02) — ruído rosa."""
    N_SINAL = int(FS * 12.0)
    N_EP    = N_SINAL // 3
    rng     = np.random.default_rng(seed)
    def pink(n):
        w  = rng.standard_normal(n)
        F  = np.fft.rfft(w)
        fr = np.fft.rfftfreq(n)
        fr[0] = 1
        F /= np.sqrt(fr)
        F[0] = 0
        s = np.fft.irfft(F, n)
        return s / (np.max(np.abs(s)) + 1e-12)
    sig = np.concatenate([0.05 * pink(N_EP), 1.00 * pink(N_EP), 0.02 * pink(N_EP)])
    return sig / (np.max(np.abs(sig)) + 1e-12)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    hdr = gerar_hdr(seed=42)
    scan_duplo(hdr, nome="HDR REDE-AP seed=42")
