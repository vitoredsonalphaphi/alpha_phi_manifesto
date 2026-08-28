# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_ScannerCepstral.py
Vitor Edson Delavi · Florianópolis · 2026

Scanner espectro-cepstral: o tensor de acoplamento T(ω,τ) = S(ω) ⊗ C(τ)

Identifica sementes de germinação fractal no espaço dual superfície-profundidade.

Estrutura:
    1. Espectro S(ω)         — superfície de frequências
    2. Cepstro C(τ)          — eco interno (quefrência)
    3. T_phi(k,m) = S_k ⊗ Cm — acoplamento em bandas φ-geométricas (9×9)
    4. Diagnóstico fractal    — D constante, Sépstro dual, vetor de expansão
    5. Visualização           — mapa topográfico de fases (matplotlib)

Sinal canônico Alpha-Phi:
    x(t) = sin(2π·f₀·t) + φ⁻¹·sin(2π·f₀·φ·t) + φ⁻²·sin(2π·f₀·φ²·t)
    Pesos: 1, 1/φ, 1/φ² — progressão φ incorporada na amplitude

Dimensão fractal φ-natural: D_φ = log(2)/log(φ) ≈ 1.440
"""

import math
import numpy as np

PHI   = (1 + math.sqrt(5)) / 2
ALPHA = 1 / 137.035999
SEAL  = 1 / PHI
N_BAND = 9
D_PHI  = math.log(2) / math.log(PHI)   # ≈ 1.440 — dimensão fractal φ-natural


# ── Geração de sinais de teste ────────────────────────────────────────────────

def sinal_phi(f0=220.0, fs=16000, duracao=1.0, ruido=0.03, seed=137) -> np.ndarray:
    """
    Sinal canônico Alpha-Phi: componentes em progressão φ.
    Pesos: 1, φ⁻¹=0.618, φ⁻²=0.382 — a semente fractal já está na amplitude.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duracao, int(fs * duracao), endpoint=False)
    x   = (1.0          * np.sin(2 * np.pi * f0 * t)
           + SEAL        * np.sin(2 * np.pi * f0 * PHI * t)
           + SEAL ** 2   * np.sin(2 * np.pi * f0 * PHI ** 2 * t))
    x  += ruido * rng.standard_normal(len(t))
    return x


def sinal_ruido(fs=16000, duracao=1.0, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(fs * duracao))


def sinal_ecobip(f0=880.0, fs=16000, duracao=1.0, seed=880) -> np.ndarray:
    """EcoBIP: harmônico base + eco_fononico simulado (spread φ nas amplitudes)."""
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duracao, int(fs * duracao), endpoint=False)
    x   = sum(SEAL ** k * np.sin(2 * np.pi * f0 * (k + 1) * t)
              for k in range(5))
    x  += 0.04 * rng.standard_normal(len(t))
    return x


# ── Espectro e Cepstro ────────────────────────────────────────────────────────

def computar_espectro(x: np.ndarray) -> tuple:
    """
    Retorna (espectro_norm, freqs_hz, fft_complexo) do sinal x.
    espectro_norm: amplitude normalizada por banda, em [0, 1].
    """
    fft   = np.fft.rfft(x)
    amp   = np.abs(fft)
    norm  = amp / (amp.max() + 1e-12)
    freqs = np.fft.rfftfreq(len(x), 1 / 16000)
    return norm, freqs, fft


def computar_cepstro(espectro_norm: np.ndarray) -> np.ndarray:
    """
    Cepstro real: C(τ) = |IFFT(log S(ω))|
    A quefrência τ = 1/f — índice k corresponde a 1/(k×Δf) segundos.
    """
    log_s  = np.log(espectro_norm + 1e-9)
    cepstro = np.abs(np.fft.irfft(log_s))
    return cepstro / (cepstro.max() + 1e-12)


# ── Bandas φ-geométricas ──────────────────────────────────────────────────────

def bandas_phi(n_bins: int) -> list:
    """
    Fronteiras das N_BAND bandas φ-geométricas em índices de bin.
    Idêntica ao ScannerTool — mesma geometria, domínio diferente.
    """
    limites = [0]
    atual   = max(1, int(n_bins * ALPHA))
    while atual < n_bins and len(limites) < N_BAND + 1:
        limites.append(atual)
        atual = min(n_bins, int(atual * PHI))
    limites.append(n_bins)
    return limites


def agregar_por_banda(vetor: np.ndarray) -> np.ndarray:
    """Agrega vetor em N_BAND bandas φ-geométricas (média por banda)."""
    n_bins  = len(vetor)
    limites = bandas_phi(n_bins)
    resultado = np.zeros(N_BAND)
    for k in range(min(N_BAND, len(limites) - 1)):
        lo, hi = limites[k], limites[k + 1]
        if hi > lo:
            resultado[k] = vetor[lo:hi].mean()
    return resultado


# ── Tensor de Acoplamento T_φ ─────────────────────────────────────────────────

def tensor_acoplamento(x: np.ndarray) -> dict:
    """
    Computa T_phi(k, m) = S_banda_k × C_quefbanda_m   →  matriz 9×9

    Cada célula (k,m) = acoplamento entre a k-ésima banda espectral
    e a m-ésima banda cepstral. Alta célula = candidata a semente fractal.

    Também retorna T_raw (produto externo sobre N_bins uniformes)
    para visualização de alta resolução.

    Retorna dict com:
        T_phi:      np.ndarray (9,9) — tensor φ-geométrico
        T_raw:      np.ndarray (N,N) — tensor de alta resolução
        S_banda:    np.ndarray (9,)  — espectro por banda φ
        C_banda:    np.ndarray (9,)  — cepstro por banda φ
        coh_phi:    np.ndarray (9,9) — Coh por célula (Sépstro dual)
        entr_phi:   np.ndarray (9,9) — Entr por célula (= 1 - coh_phi)
        dim_phi:    float            — dimensão fractal estimada
        sementes:   list             — (k, m, probabilidade) das sementes ativas
        vetor_exp:  np.ndarray (8,)  — ΔT na diagonal (vetor de expansão)
        S_raw:      np.ndarray       — espectro normalizado (alta resolução)
        C_raw:      np.ndarray       — cepstro normalizado (alta resolução)
    """
    S_raw, freqs, _ = computar_espectro(x)
    C_raw           = computar_cepstro(S_raw)

    # Bandas φ em cada domínio
    S_banda = agregar_por_banda(S_raw)
    C_banda = agregar_por_banda(C_raw)

    # Tensor de acoplamento 9×9
    T_phi = np.outer(S_banda, C_banda)

    # Tensor de alta resolução (Gemini original — primeiros N_bins uniformes)
    N_raw = min(200, len(S_raw))
    T_raw = np.outer(S_raw[:N_raw], C_raw[:N_raw])

    # ── Sépstro no espaço dual ────────────────────────────────────────────────
    # Normaliza T_phi para que cada coluna tenha soma ≤ 1 (lei de conservação)
    total = T_phi.sum() + 1e-12
    coh_phi  = np.clip(T_phi / total * N_BAND, ALPHA, 1 - ALPHA)
    entr_phi = 1.0 - coh_phi

    # ── Dimensão fractal estimada via lei de potência na diagonal ─────────────
    diag = np.diag(T_phi)
    diag = np.where(diag > 1e-12, diag, 1e-12)
    # Ajuste log-log: log(T_k) = -D × log(k+1) + const
    ks = np.arange(1, N_BAND + 1, dtype=float)
    if diag.sum() > 1e-6:
        coeffs  = np.polyfit(np.log(ks), np.log(diag), 1)
        dim_phi = float(np.clip(-coeffs[0], 0.5, 3.0))
    else:
        dim_phi = D_PHI

    # ── Vetor de Expansão Potencial ───────────────────────────────────────────
    # ΔT(k) = T_phi[k+1,k+1] - T_phi[k,k] — crescimento na diagonal
    vetor_exp = np.diff(np.diag(T_phi))   # (8,)

    # ── Sementes fractais ─────────────────────────────────────────────────────
    # Máximos locais em T_phi com Coh > threshold
    sementes = []
    threshold = coh_phi.mean() + coh_phi.std()
    for k in range(N_BAND):
        for m in range(N_BAND):
            if coh_phi[k, m] > threshold:
                prob = float((coh_phi[k, m] - threshold) /
                             (coh_phi.max() - threshold + 1e-8))
                sementes.append((k, m, prob))
    sementes.sort(key=lambda s: -s[2])

    return {
        'T_phi':    T_phi,
        'T_raw':    T_raw,
        'S_banda':  S_banda,
        'C_banda':  C_banda,
        'coh_phi':  coh_phi,
        'entr_phi': entr_phi,
        'dim_phi':  dim_phi,
        'sementes': sementes,
        'vetor_exp': vetor_exp,
        'S_raw':    S_raw[:N_raw],
        'C_raw':    C_raw[:N_raw],
    }


# ── Diagnóstico fractal ───────────────────────────────────────────────────────

def diagnostico(resultado: dict) -> str:
    """
    Aplica os três critérios de diagnóstico fractal de Entrada 198:
    1. Invariância de escala — D ≈ D_φ?
    2. Conservação do Sépstro dual — Coh + Entr = 1 em cada célula?
    3. Vetor de expansão — ΔT > 0?
    """
    r = resultado
    D   = r['dim_phi']
    sep = np.allclose(r['coh_phi'] + r['entr_phi'], 1.0, atol=1e-6)
    exp = float(r['vetor_exp'].mean())
    n_sem = len(r['sementes'])

    linhas = [
        "=" * 60,
        "Scanner Cepstral — Diagnóstico Fractal Alpha-Phi",
        f"  D_medida  = {D:.4f}   D_φ_natural = {D_PHI:.4f}",
        f"  |D − D_φ| = {abs(D - D_PHI):.4f}   {'✓ atrator φ' if abs(D - D_PHI) < 0.2 else '○ fora do atrator'}",
        f"  Sépstro dual OK: {sep}   (Coh+Entr=1 em todas as células)",
        f"  Vetor de expansão médio: {exp:+.6f}   "
          f"{'↑ germinando' if exp > 1e-6 else ('↓ dissipando' if exp < -1e-6 else '→ estacionário')}",
        f"  Sementes ativas: {n_sem}",
    ]
    if r['sementes']:
        top = r['sementes'][:3]
        linhas.append("  Top sementes (banda_S, banda_C, prob):")
        for k, m, p in top:
            linhas.append(f"    (N{k}, N{m}) prob={p:.3f}")
    linhas.append("=" * 60)
    return "\n".join(linhas)


# ── Visualização ──────────────────────────────────────────────────────────────

def visualizar(resultado: dict, titulo: str = "Scanner Cepstral Alpha-Phi"):
    """
    Gera o mapa topográfico de fases em três painéis:
      Painel 1: T_raw — acoplamento de alta resolução (cores Alpha-Phi)
      Painel 2: T_phi (9×9) — bandas φ com marcação das sementes
      Painel 3: Diagonal T_phi + D_φ de referência
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("[visualizar] matplotlib não disponível — use pip install matplotlib")
        return None

    # Paleta Alpha-Phi: azul profundo → âmbar → ouro
    COLORS = ['#0E1E48', '#1E4A90', '#1A4A80', '#C8960A', '#F5D55A', '#FFE89A']
    cmap   = mcolors.LinearSegmentedColormap.from_list('alphaphi', COLORS)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             facecolor='#070810', constrained_layout=True)
    for ax in axes:
        ax.set_facecolor('#0D1525')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2E4055')

    # ── Painel 1: T_raw ───────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(resultado['T_raw'], aspect='auto', cmap=cmap, origin='lower',
                   interpolation='bilinear')
    ax.set_title('T(ω,τ) — Alta Resolução', color='#C8BBAA', fontsize=10)
    ax.set_xlabel('Quefrência τ (eco interior)', color='#2E4055', fontsize=8)
    ax.set_ylabel('Frequência ω (superfície)', color='#2E4055', fontsize=8)
    ax.tick_params(colors='#2E4055', labelsize=7)
    fig.colorbar(im, ax=ax, label='S(ω)⊗C(τ)', fraction=0.04).ax.yaxis.label.set_color('#2E4055')

    # ── Painel 2: T_phi 9×9 ───────────────────────────────────────────────────
    ax = axes[1]
    im2 = ax.imshow(resultado['T_phi'], aspect='auto', cmap=cmap, origin='lower')
    ax.set_title('T_φ(k,m) — Bandas φ (9×9)', color='#C8BBAA', fontsize=10)
    ax.set_xlabel('Banda Cepstral (m)', color='#2E4055', fontsize=8)
    ax.set_ylabel('Banda Espectral (k)', color='#2E4055', fontsize=8)
    ax.set_xticks(range(N_BAND)); ax.set_yticks(range(N_BAND))
    ax.set_xticklabels([f'N{i}' for i in range(N_BAND)], color='#2E4055', fontsize=7)
    ax.set_yticklabels([f'N{i}' for i in range(N_BAND)], color='#2E4055', fontsize=7)
    # Marcar sementes
    for k, m, p in resultado['sementes'][:6]:
        ax.plot(m, k, 'o', color='#6B3AFF', markersize=4 + p * 8, alpha=0.8)
        if p > 0.5:
            ax.plot(m, k, '+', color='#4AFFE8', markersize=8, lw=1.5)
    # Diagonal (auto-similaridade)
    ax.plot(range(N_BAND), range(N_BAND), '--', color='rgba(74,255,232,0.3)',
            lw=0.8, alpha=0.4)
    fig.colorbar(im2, ax=ax, fraction=0.04).ax.yaxis.label.set_color('#2E4055')

    # ── Painel 3: diagonal + referência D_φ ───────────────────────────────────
    ax = axes[2]
    diag = np.diag(resultado['T_phi'])
    ks   = np.arange(N_BAND)
    ax.plot(ks, diag, 'o-', color='#F5D55A', lw=1.5, ms=5, label='T_φ diagonal')
    # Referência D_φ: potência de lei com D=D_φ
    if diag[0] > 1e-10:
        ref = diag[0] * np.power(ks + 1, -D_PHI) * (1 ** D_PHI)
        ax.plot(ks, ref, '--', color='#4AFFE8', lw=1, alpha=0.7,
                label=f'D_φ = {D_PHI:.3f} (ref.)')
    ax.axhline(0, color='#2E4055', lw=0.5)
    D = resultado['dim_phi']
    ax.set_title(f'Diagonal T_φ · D_medida={D:.3f}', color='#C8BBAA', fontsize=10)
    ax.set_xlabel('Banda k', color='#2E4055', fontsize=8)
    ax.tick_params(colors='#2E4055', labelsize=7)
    ax.legend(facecolor='#0D1525', edgecolor='#2E4055',
              labelcolor='#C8BBAA', fontsize=7)
    # Vetor de expansão
    ax2 = ax.twinx()
    ax2.bar(ks[:-1] + 0.5, resultado['vetor_exp'],
            color=['#E84040' if v < 0 else '#4AFFE8' for v in resultado['vetor_exp']],
            alpha=0.35, width=0.7, label='ΔT (expansão)')
    ax2.set_ylabel('ΔT (expansão)', color='#2E4055', fontsize=7)
    ax2.tick_params(colors='#2E4055', labelsize=6)
    ax2.axhline(0, color='#2E4055', lw=0.4, ls=':')

    fig.suptitle(titulo, color='#C8BBAA', fontsize=12, y=1.01)

    out = titulo.replace(' ', '_') + '.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#070810')
    print(f"[visualizar] salvo em: {out}")
    plt.close(fig)
    return out


# ── Demo comparativa ──────────────────────────────────────────────────────────

def demo():
    sinais = {
        'Sinal φ (220Hz)':  sinal_phi(f0=220),
        'EcoBIP (880Hz)':   sinal_ecobip(f0=880),
        'Ruído Branco':     sinal_ruido(),
    }

    for nome, x in sinais.items():
        print(f"\n{'─'*60}")
        print(f"  {nome}")
        r = tensor_acoplamento(x)
        print(diagnostico(r))
        visualizar(r, titulo=f"Scanner Cepstral · {nome}")


if __name__ == '__main__':
    demo()
