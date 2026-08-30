# AlphaPhi_Scanner_v2.py
# Scanner Topográfico v2 — Espectrograma φ-escalado + Nomenclatura Automática Máxima
# Vitor Edson Delavi · Florianópolis · 2026
# © CC BY-NC-ND 4.0

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, label as nd_label
from scipy.signal import hilbert, find_peaks, stft, butter, filtfilt
from scipy.stats import entropy as scipy_entropy
import warnings
warnings.filterwarnings('ignore')

# ─── Constantes ───────────────────────────────────────────────────────────────
PHI     = 1.6180339887
ALPHA   = 1 / 137.035999
SEAL    = 1 / PHI
THETA_R = np.arctan(2.0)          # 63.43° — Grade Romboédrica
SR      = 44100
DUR     = 10.0
BASE    = 880.0

# ─── Geração de sinais ────────────────────────────────────────────────────────
def gerar_ecobeep(sr=SR, dur=DUR):
    t  = np.linspace(0, dur, int(sr*dur), endpoint=False)
    fm = np.sin(2*np.pi*220*t + PHI*np.sin(2*np.pi*(220/PHI)*t))
    sq = np.sign(np.sin(2*np.pi*BASE*t))
    x  = (1-ALPHA)*sq + ALPHA*fm
    return t, x / np.max(np.abs(x))

def gerar_quadrada(sr=SR, dur=DUR):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    return t, np.sign(np.sin(2*np.pi*BASE*t)).astype(float)

# ─── STFT genuinamente 2D com janela φ-escalada ───────────────────────────────
def stft_phi(x, sr=SR):
    win = int(sr / BASE * 2 * PHI)        # 2 ciclos de 880Hz × φ
    win = max(512, min(win, 4096))
    hop = win // 4
    f, t, Zxx = stft(x, fs=sr, window='hann', nperseg=win, noverlap=win-hop)
    return f, t, np.abs(Zxx)**2

# ─── Métricas ─────────────────────────────────────────────────────────────────
def phi_score(x, sr=SR, n=1024):
    F = np.abs(np.fft.rfft(x, n=n*2))[:n]
    fq = np.fft.rfftfreq(n*2, 1/sr)[:n]
    tot = F.sum() + 1e-9
    b0  = fq[np.argmax(F)] or BASE
    s   = sum(F[np.argmin(np.abs(fq - b0*PHI**k))]
              + F[np.argmin(np.abs(fq - b0/PHI**k))]
              for k in range(1, 8))
    return s / tot

def grad_entropia(T, sigma=2.5):
    return T - gaussian_filter(T, sigma=sigma)

def delta_z_banda(x, sr=SR, fl=135, fh=360):
    b, a = butter(4, [fl/(sr/2), fh/(sr/2)], btype='band')
    xf   = filtfilt(b, a, x)
    inst = np.diff(np.unwrap(np.angle(hilbert(xf))))
    return float(np.std(inst) / (np.abs(inst).mean() + 1e-9))

def energia_bandas_phi(S, freqs, n=7):
    """Energia média por banda φ-harmônica."""
    res = []
    for k in range(n):
        fc = BASE / PHI**k
        bw = fc * (PHI - 1) / 2
        m  = (freqs >= fc-bw) & (freqs <= fc+bw)
        res.append((fc, S[m].mean() if m.any() else 0.0))
    return res

# ─── Nomenclatura automática de frequência ────────────────────────────────────
def nomear_freq(f, base=BASE, tol=0.04):
    if f < 1:
        return "DC"
    # Relações φ
    for k in range(-5, 6):
        fh = base * PHI**k
        if abs(f - fh) / fh < tol:
            tag = f"φ^{k:+d}" if k else "base"
            return f"{f:.0f}Hz [{tag}]"
    # Múltiplos inteiros
    for k in range(2, 16):
        if abs(f - base*k) / (base*k) < tol:
            return f"{f:.0f}Hz [{k}×base]"
        if abs(f - base/k) / (base/k) < tol:
            return f"{f:.0f}Hz [base/{k}]"
    # Harmônicos α
    if abs(f - base*ALPHA*10) / (base*ALPHA*10) < 0.1:
        return f"{f:.0f}Hz [α-banda]"
    return f"{f:.0f}Hz"

def classificar_crista(fc, base=BASE):
    if fc < 5:    return "Sub-DC"
    if abs(fc-base)/base < 0.05:   return "Fundamental 880"
    for k in range(2, 12):
        if abs(fc-base*k)/(base*k) < 0.05: return f"Harmônico {k}×880"
    for k in range(1, 7):
        fh = base * PHI**k
        if abs(fc-fh)/fh < 0.06: return f"φ^+{k} ({fh:.0f}Hz)"
        fh = base / PHI**k
        if abs(fc-fh)/fh < 0.06: return f"φ^-{k} ({fh:.0f}Hz)"
    return f"Livre ({fc:.0f}Hz)"

# ─── Detecção de picos 2D ─────────────────────────────────────────────────────
def detectar_picos(S, freqs, times, n=12):
    S_log = np.log1p(S * 100)
    order = np.argsort(S_log.ravel())[::-1]
    picos, usado = [], np.zeros(S_log.shape, bool)
    df_min = max(1, len(freqs)//18)
    dt_min = max(1, len(times)//18)
    for idx in order:
        fi, ti = np.unravel_index(idx, S_log.shape)
        if usado[fi, ti]: continue
        picos.append({'fi':fi,'ti':ti,
                      'f':freqs[fi],'t':times[ti],
                      'amp':S[fi,ti],
                      'nome':nomear_freq(freqs[fi])})
        usado[max(0,fi-df_min):fi+df_min,
              max(0,ti-dt_min):ti+dt_min] = True
        if len(picos) >= n: break
    return picos

def detectar_cristas(S, freqs):
    S_n = (S - S.min()) / (S.max()-S.min()+1e-9)
    thr = S_n.mean() + S_n.std()*0.8
    lab, n = nd_label(S_n > thr)
    cristas = []
    for i in range(1, n+1):
        m = lab == i
        if m.sum() < 8: continue
        fi = int(np.where(m)[0].mean())
        fc = freqs[fi] if fi < len(freqs) else 0
        cristas.append({'fi':fi,'fc':fc,'area':m.sum(),
                        'tipo':classificar_crista(fc)})
    cristas.sort(key=lambda x: -x['area'])
    return cristas[:5]

def linha_theta_r(freqs, times):
    fn = (freqs - freqs[0]) / (freqs[-1]-freqs[0]+1e-9)
    tv = np.tan(THETA_R) * fn * (times[-1]-times[0]) + times[0]
    ok = (tv >= times[0]) & (tv <= times[-1])
    return freqs[ok], tv[ok]

# ─── Painel de métricas numéricas ─────────────────────────────────────────────
def painel_metricas(ax, x, titulo, cor):
    ps = phi_score(x)
    dz = delta_z_banda(x)
    F  = np.abs(np.fft.rfft(x, n=2048))
    fq = np.fft.rfftfreq(2048, 1/SR)
    dom_f = fq[np.argmax(F)]
    tot   = F.sum() + 1e-9

    linhas = [
        f"PHI_score  : {ps:.6f}",
        f"ΔZ (banda) : {dz:.6f}",
        f"Freq dom.  : {dom_f:.1f} Hz",
        "",
        "φ-Harmônicos [% energia]:",
    ]
    for k in range(-3, 5):
        fh  = BASE * PHI**k
        idx = np.argmin(np.abs(fq - fh))
        pct = F[idx]/tot*100
        tag = f"φ^{k:+d}"
        bar = "█"*int(min(pct,30)/1.5) if pct > 0.2 else "·"
        linhas.append(f"  {tag:6s} {fh:7.1f}Hz  {pct:5.2f}%  {bar}")
    linhas += ["", "Bandas φ [energia média]:"]
    _, _, Sv = stft_phi(x)
    fv, _ = np.fft.rfftfreq(2048, 1/SR), None
    fv2, tv2, Sv2 = stft_phi(x)
    fmx = fv2 <= 5000
    fv2 = fv2[fmx]; Sv2 = Sv2[fmx]
    for fc, e in energia_bandas_phi(Sv2, fv2, n=6):
        bar = "▪"*int(min(e*1e6,20))
        linhas.append(f"  {fc:6.0f}Hz  E={e:.2e}  {bar}")

    ax.axis('off')
    ax.set_facecolor('#050510')
    ax.text(0.03, 0.99, titulo, transform=ax.transAxes,
            color=cor, fontsize=8.5, fontweight='bold', va='top')
    ax.text(0.03, 0.91, "\n".join(linhas), transform=ax.transAxes,
            color='#DDDDDD', fontsize=6.8, va='top', family='monospace',
            linespacing=1.55)

# ─── FIGURA PRINCIPAL ──────────────────────────────────────────────────────────
def scanner_v2():
    _, eco  = gerar_ecobeep()
    _, quad = gerar_quadrada()
    sinais  = [('EcoBIP 880Hz', eco, '#FFD700'),
               ('Quadrada Pura', quad, '#4488FF')]

    fig = plt.figure(figsize=(24, 15), facecolor='#030308')
    fig.suptitle(
        'Scanner Topográfico v2  ·  Espectrograma φ-escalado  ·  Nomenclatura Automática',
        color='white', fontsize=13, fontweight='bold', y=0.995)

    outer = gridspec.GridSpec(2, 1, fig, hspace=0.42, top=0.965, bottom=0.03)

    CORES_PICO = ['#FF4444','#FF8800','#FFEE00','#88FF00','#00FFAA',
                  '#00CCFF','#8844FF','#FF44CC','#FFFFFF','#FF9966',
                  '#66FF66','#FF6666']

    for row, (nome, x, cor) in enumerate(sinais):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 5, subplot_spec=outer[row],
            wspace=0.38, hspace=0.50,
            width_ratios=[3.5, 1.2, 1.2, 1.2, 2.0])

        # ── STFT ─────────────────────────────────────────────────────────────
        freqs, times, S = stft_phi(x)
        fmask = freqs <= 5500
        fv, Sv = freqs[fmask], S[fmask]
        Sl     = np.log1p(Sv * 100)
        gradS  = grad_entropia(Sl, sigma=3.0)
        ext    = [times[0], times[-1], fv[0], fv[-1]]

        picos   = detectar_picos(Sv, fv, times, n=12)
        cristas = detectar_cristas(Sv, fv)
        fr, tr  = linha_theta_r(fv, times)

        # ── Plot A: espectrograma principal (2 linhas, col 0) ─────────────────
        ax_m = fig.add_subplot(inner[:, 0])
        ax_m.set_facecolor('#050510')

        im = ax_m.imshow(Sl, aspect='auto', origin='lower',
                         extent=ext, cmap='inferno', alpha=0.93)

        # ∇S contorno (respiração=azul, vértice=vermelho)
        ax_m.contourf(times, fv, gradS, levels=[-99,0],
                      colors=['#0044AA'], alpha=0.18)
        ax_m.contourf(times, fv, gradS, levels=[0,99],
                      colors=['#AA2200'], alpha=0.18)
        ax_m.contour(times, fv, gradS, levels=[0],
                     colors=['#00FF88'], linewidths=0.6, alpha=0.55)

        # Linha θ_R
        if len(fr) > 1:
            ax_m.plot(tr, fr, '--', color='#00FF88', lw=1.4, alpha=0.8,
                      label=f'θ_R = {np.degrees(THETA_R):.1f}°  (Grade R)')

        # φ-harmônicos horizontais
        for k in range(-3, 5):
            fh = BASE * PHI**k
            if fv[0] < fh < fv[-1]:
                ax_m.axhline(fh, color='#AA88FF', lw=0.5, alpha=0.45, ls=':')
                ax_m.text(times[-1]*0.995, fh+15, f'φ^{k:+d}',
                          color='#AA88FF', fontsize=5.5, ha='right', va='bottom')

        # Harmônicos inteiros de BASE
        for k in range(1, 7):
            fh = BASE * k
            if fh < fv[-1]:
                ax_m.axhline(fh, color='#FF8866', lw=0.4, alpha=0.3, ls=':')

        # Banda α
        fa = BASE * ALPHA * 137
        if fv[0] < fa < fv[-1]:
            ax_m.axhline(fa, color='#FF8844', lw=0.6, alpha=0.5, ls='-.')
            ax_m.text(times[0]+0.1, fa+15, 'α-ref', color='#FF8844', fontsize=5.5)

        # Marcadores de picos
        for i, p in enumerate(picos):
            c = CORES_PICO[i % len(CORES_PICO)]
            ax_m.plot(p['t'], p['f'], 'o', color=c, ms=5.5,
                      markeredgecolor='white', markeredgewidth=0.3, zorder=12)
            dx = 0.25 if p['t'] < times[-1]*0.7 else -0.25
            ax_m.annotate(f" {p['nome']}",
                          xy=(p['t'], p['f']),
                          xytext=(p['t']+dx, p['f']+60),
                          color=c, fontsize=5.2, fontweight='bold',
                          arrowprops=dict(arrowstyle='->', color=c, lw=0.5),
                          zorder=13)

        # Marcadores de cristas
        for cr in cristas[:4]:
            ax_m.axhline(cr['fc'], color='#00FFFF', lw=0.7, alpha=0.35, ls='--')
            ax_m.text(times[0]+0.05, cr['fc']+20,
                      f"◄ {cr['tipo']}", color='#00FFFF', fontsize=5.5)

        ax_m.set_xlabel('Tempo (s)', color='#AAAAAA', fontsize=8)
        ax_m.set_ylabel('Frequência (Hz)', color='#AAAAAA', fontsize=8)
        ax_m.set_title(f'{nome}  —  Espectrograma φ-escalado + Nomenclatura Automática',
                       color=cor, fontsize=9, fontweight='bold', pad=4)
        ax_m.tick_params(colors='#777777', labelsize=7)
        ax_m.legend(fontsize=6.5, facecolor='#0A0A18', edgecolor='#333355',
                    labelcolor='white', loc='upper right')
        cb = plt.colorbar(im, ax=ax_m, shrink=0.55, pad=0.01)
        cb.set_label('log(energia)', color='#888888', fontsize=7)
        cb.ax.yaxis.set_tick_params(color='#666666', labelsize=6)

        # ── Plot B: perfil espectral médio (linha 0, col 1) ───────────────────
        ax_sp = fig.add_subplot(inner[0, 1])
        ax_sp.set_facecolor('#050510')
        esp_med = Sv.mean(axis=1)
        ax_sp.fill_betweenx(fv, 0, esp_med, color=cor, alpha=0.45)
        ax_sp.plot(esp_med, fv, color=cor, lw=0.9)
        for k in range(-2, 4):
            fh = BASE*PHI**k
            if fv[0] < fh < fv[-1]:
                ax_sp.axhline(fh, color='#AA88FF', lw=0.5, alpha=0.5, ls=':')
        ax_sp.set_title('Perfil Espectral\n(média temporal)', color='#CCCCCC', fontsize=7)
        ax_sp.set_xlabel('Energia', color='#888888', fontsize=6)
        ax_sp.set_ylabel('Hz', color='#888888', fontsize=6)
        ax_sp.tick_params(colors='#555555', labelsize=5.5)
        for sp in ax_sp.spines.values(): sp.set_edgecolor('#222233')

        # ── Plot C: ∇S temporal (linha 0, col 2) ─────────────────────────────
        ax_gs = fig.add_subplot(inner[0, 2])
        ax_gs.set_facecolor('#050510')
        gs_t = gradS.mean(axis=0)
        ax_gs.fill_between(times, 0, gs_t, where=gs_t>0,
                           color='#FF4444', alpha=0.55, label='Vértice')
        ax_gs.fill_between(times, 0, gs_t, where=gs_t<0,
                           color='#4488FF', alpha=0.55, label='Respiração')
        ax_gs.axhline(0, color='#888888', lw=0.5)
        pks, _ = find_peaks(np.abs(gs_t), height=gs_t.std()*0.8)
        for pk in pks[:6]:
            ax_gs.axvline(times[pk], color='#FFD700', lw=0.5, alpha=0.7)
            ax_gs.text(times[pk], gs_t.max()*0.85, f'{times[pk]:.1f}s',
                       color='#FFD700', fontsize=5, rotation=90, va='top')
        ax_gs.set_title('∇S Temporal\n(pontos de dobra)', color='#CCCCCC', fontsize=7)
        ax_gs.set_xlabel('Tempo (s)', color='#888888', fontsize=6)
        ax_gs.legend(fontsize=5.5, facecolor='#0A0A18', labelcolor='white')
        ax_gs.tick_params(colors='#555555', labelsize=5.5)
        for sp in ax_gs.spines.values(): sp.set_edgecolor('#222233')

        # ── Plot D: energia por banda φ (linha 0, col 3) ──────────────────────
        ax_bnd = fig.add_subplot(inner[0, 3])
        ax_bnd.set_facecolor('#050510')
        bndas = energia_bandas_phi(Sv, fv, n=7)
        fcs   = [b[0] for b in bndas]
        ens   = [b[1] for b in bndas]
        bars  = ax_bnd.bar(range(len(ens)), ens, color=cor, alpha=0.72,
                           edgecolor='#333333', linewidth=0.4)
        # destacar barra máxima
        mx = np.argmax(ens)
        bars[mx].set_edgecolor('#FFFFFF')
        bars[mx].set_linewidth(1.2)
        ax_bnd.set_xticks(range(len(fcs)))
        ax_bnd.set_xticklabels([f'φ^-{k}\n{fc:.0f}Hz'
                                 for k, (fc,_) in enumerate(bndas)],
                                fontsize=5, color='#AAAAAA')
        ax_bnd.set_title('Energia por Banda φ\n(destacado=máximo)', color='#CCCCCC', fontsize=7)
        ax_bnd.set_ylabel('E média', color='#888888', fontsize=6)
        ax_bnd.tick_params(colors='#555555', labelsize=5.5)
        for sp in ax_bnd.spines.values(): sp.set_edgecolor('#222233')

        # ── Plot E: mapa ∇S 2D (linha 1, col 1) ──────────────────────────────
        ax_gs2 = fig.add_subplot(inner[1, 1])
        ax_gs2.set_facecolor('#050510')
        ax_gs2.imshow(gradS, aspect='auto', origin='lower',
                      extent=ext, cmap='RdBu_r', alpha=0.9)
        if len(fr) > 1:
            ax_gs2.plot(tr, fr, '--', color='#00FF88', lw=1.0, alpha=0.85,
                        label='θ_R')
        # Marcar vértices da Grade R (interseções dos picos com θ_R)
        for p in picos[:5]:
            if len(fr) > 1:
                t_interp = np.interp(p['f'], fr, tr) if fr[0] <= p['f'] <= fr[-1] else None
                if t_interp and abs(p['t'] - t_interp) < (times[-1]-times[0])*0.1:
                    ax_gs2.plot(p['t'], p['f'], '*', color='#FFD700',
                                ms=7, zorder=10, label='Vértice Grade R')
        ax_gs2.set_title('∇S — Mapa 2D\nazul=respiração · vermelho=vértice',
                         color='#CCCCCC', fontsize=7)
        ax_gs2.set_xlabel('Tempo (s)', color='#888888', fontsize=6)
        ax_gs2.set_ylabel('Hz', color='#888888', fontsize=6)
        ax_gs2.tick_params(colors='#555555', labelsize=5.5)
        ax_gs2.legend(fontsize=5.5, facecolor='#0A0A18', labelcolor='white',
                      loc='upper right')

        # ── Plot F: espectrograma diferença (linha 1, col 2-3) ───────────────
        ax_diff = fig.add_subplot(inner[1, 2:4])
        ax_diff.set_facecolor('#050510')
        # Diferença normalizada do espectrograma vs. espectrograma médio
        Sl_n   = (Sl - Sl.min()) / (Sl.max()-Sl.min()+1e-9)
        Sl_med = Sl.mean(axis=1, keepdims=True)
        dif    = Sl - Sl_med
        ax_diff.imshow(dif, aspect='auto', origin='lower',
                       extent=ext, cmap='seismic', alpha=0.9,
                       vmin=-dif.std()*2, vmax=dif.std()*2)
        if len(fr) > 1:
            ax_diff.plot(tr, fr, '--', color='#00FF88', lw=1.0, alpha=0.7)
        ax_diff.set_title('Desvio local vs. perfil médio\n(vermelho=acima média · azul=abaixo)',
                          color='#CCCCCC', fontsize=7)
        ax_diff.set_xlabel('Tempo (s)', color='#888888', fontsize=6)
        ax_diff.set_ylabel('Hz', color='#888888', fontsize=6)
        ax_diff.tick_params(colors='#555555', labelsize=5.5)

        # ── Plot G: painel de métricas (col 4, span 2 linhas) ─────────────────
        ax_mt = fig.add_subplot(inner[:, 4])
        painel_metricas(ax_mt, x, f'MÉTRICAS  ·  {nome}', cor)

    fig.savefig('scanner_v2_resultado.png', dpi=155,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Salvo: scanner_v2_resultado.png")
    plt.show()
    return fig


if __name__ == '__main__':
    print(f"φ  = {PHI}")
    print(f"α  = {ALPHA:.8f}")
    print(f"θ_R = {np.degrees(THETA_R):.2f}°")
    print(f"SR  = {SR} Hz | DUR = {DUR}s")
    print()
    scanner_v2()
