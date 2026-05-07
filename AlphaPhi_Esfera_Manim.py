"""
AlphaPhi_Esfera_Manim.py
Esfera Animada — Manim · Beep 880Hz · α*=1/3

A esfera dourada responde ao campo eco em tempo real:
  • cor: azul (coh baixa) → dourada (coh alta)
  • anéis φ: opacidade = amplitude espectral por faixa
  • pontos de dobra P/S/T: pulso expansivo + halo
  • trajetória β: linha crescente na base

© Vitor Edson Delavi · Florianópolis · 2026
Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
"""

# ══════════════════════════════════════════════════════════════
# CÉLULA COLAB — cole tudo de uma vez
# Instala Manim, processa o áudio, renderiza, exibe
# ══════════════════════════════════════════════════════════════

import subprocess, sys, os

print("Instalando Manim (pode demorar 1-2 min na primeira vez)...")
subprocess.run([sys.executable, '-m', 'pip', 'install', 'manim', '-q'], check=True)
print("✓ Manim instalado")

# ── constantes ────────────────────────────────────────────────
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.io import wavfile

PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
F_BEEP     = 880.0
F_ORG      = 220.0
F_M        = F_ORG / PHI
BETA_FM    = PHI
ALPHA_STAR = 1.0 / 3.0
DURACAO    = 1.5
N_STEPS    = 5
N_CICLOS   = 20
FPS_ANIM   = 24

# ── funções eco ───────────────────────────────────────────────
def normalizar(s):
    m = np.max(np.abs(s))
    return s / m if m > 1e-12 else s

def gerar_bandas_phi(f_min=20.0, f_max=22050.0):
    bandas, f = [], f_min
    while f < f_max:
        f_next = min(f * PHI, f_max)
        bandas.append((f, f_next))
        if f_next >= f_max: break
        f = f_next
    return bandas

def bandas_para_bins(bandas, n):
    return [(max(0, int(f_lo/(FS/n))),
             min(int(f_hi/(FS/n))+1, n//2+1),
             f_lo, f_hi)
            for f_lo, f_hi in bandas]

N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)
N_BANDAS = len(BINS_PHI)

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    w_mem, w_now = 1.0/PHI, 1.0 - 1.0/PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi   = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb   = F[b_lo:b_hi]
        mag  = np.abs(Fb); phase = np.angle(Fb)
        an   = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh  = float(1.0 - (-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        ce   = (w_now*coh + w_mem*float(coh_mem[i])
                if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk   = np.arange(len(Fb))
        env  = np.clip(1.0 + (ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag*env)*np.exp(1j*phase)
    r = np.fft.irfft(F_out, n=N)
    return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

def cascata_eq(sinal, beta_bands, bins_phi):
    cas, s = [sinal], sinal.copy()
    cm = np.zeros(len(bins_phi)); cf = np.zeros(len(bins_phi))
    for _ in range(N_STEPS):
        se, cohs = eco_eq(s, bins_phi, beta_bands, cm)
        cm, cf = cohs, cohs
        se = normalizar(se); cas.append(se); s = se.copy()
    return cas, cf

def agente_eco_full(sinal, bins_phi, n_ciclos=20):
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    sinais, betas = [], []
    for _ in range(n_ciclos):
        cas, cohs = cascata_eq(sinal, beta, bins_phi)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
        sinais.append(cas[-1]); betas.append(beta.copy())
    return sinais, betas

# ── gerar sinal ────────────────────────────────────────────────
print("Processando sinal eco 880Hz (α*=1/3)…")
t_sig  = np.linspace(0, DURACAO, N_SINAL, endpoint=False)
beep   = normalizar(np.sign(np.sin(2*np.pi*F_BEEP*t_sig)))
fm     = normalizar(np.sin(2*np.pi*F_ORG*t_sig + BETA_FM*np.sin(2*np.pi*F_M*t_sig)))
x_mix  = normalizar((1.0-ALPHA_STAR)*beep + ALPHA_STAR*fm)
sinais_ciclos, betas_ciclos = agente_eco_full(x_mix, BINS_PHI, N_CICLOS)
eco    = sinais_ciclos[-1]
print(f"✓ {N_CICLOS} ciclos processados")

s16 = np.int16(np.clip(normalizar(eco), -1, 1)*32767)
wavfile.write('/content/beep880_eco_final.wav', FS, s16)
print("✓ beep880_eco_final.wav")

# ── camadas P/S/T ──────────────────────────────────────────────
def blp(s, c, o=4):
    b, a = butter(o, c/(FS/2), btype='low'); return filtfilt(b, a, s)

P = eco.copy()
S = blp(np.abs(hilbert(P)), 200.0)
T = blp(np.abs(hilbert(S)), 50.0)

# ── pontos de dobra ────────────────────────────────────────────
def ponto_dobra(sig, ms=50):
    w = int(FS*ms/1000); step = w//10
    var = np.array([np.var(sig[i:i+w]) for i in range(0, len(sig)-w, step)])
    return np.argmin(var)*step

idx_P, idx_S, idx_T = ponto_dobra(P), ponto_dobra(S), ponto_dobra(T)
t_P, t_S, t_T = idx_P/FS, idx_S/FS, idx_T/FS
print(f"✓ Pontos de dobra: P={t_P*1e3:.0f}ms  S={t_S*1e3:.0f}ms  T={t_T*1e3:.0f}ms")

# ── dados espectrais por frame de animação ─────────────────────
N_FRAMES = int(DURACAO * FPS_ANIM)
hop      = max(1, N_SINAL // N_FRAMES)
spec_frames = np.zeros((N_FRAMES, N_BANDAS))
coh_frames  = np.zeros(N_FRAMES)

phi_freqs = [F_BEEP*(PHI**k) for k in range(1,5) if F_BEEP*(PHI**k) < FS/2]

for fi in range(N_FRAMES):
    chunk = eco[fi*hop : min((fi+1)*hop, N_SINAL)]
    if len(chunk) < 8: continue
    nf = max(128, len(chunk))
    X  = np.abs(np.fft.rfft(chunk, n=nf))
    fr = np.fft.rfftfreq(nf, 1/FS)
    for bi, (_, _, flo, fhi) in enumerate(BINS_PHI):
        mask = (fr >= flo) & (fr < fhi)
        spec_frames[fi, bi] = X[mask].mean() if mask.any() else 0.0
    tot = np.sum(X**2) + 1e-12
    php = sum(np.sum(X[(fr>=f*0.92)&(fr<=f*1.08)]**2) for f in phi_freqs)
    coh_frames[fi] = php / tot

# normalizar por frame
for fi in range(N_FRAMES):
    mx = spec_frames[fi].max()
    if mx > 1e-10: spec_frames[fi] /= mx
coh_max = coh_frames.max()
if coh_max > 1e-10: coh_frames /= coh_max

# β trajetória interpolada para frames
beta_max_ciclos = [float(b.max()) for b in betas_ciclos]
beta_frames = np.interp(
    np.linspace(0, N_CICLOS-1, N_FRAMES),
    np.arange(N_CICLOS), beta_max_ciclos)

# salvar
np.savez('/content/alphaphi_data.npz',
         spec_frames=spec_frames, coh_frames=coh_frames,
         beta_frames=beta_frames,
         fold_times=np.array([t_P, t_S, t_T]),
         PHI=np.array([PHI]), fps=np.array([FPS_ANIM]),
         n_bandas=np.array([N_BANDAS]), duracao=np.array([DURACAO]))
print(f"✓ Dados pré-computados: {N_FRAMES} frames  {N_BANDAS} faixas φ")

# ══════════════════════════════════════════════════════════════
# CENA MANIM
# ══════════════════════════════════════════════════════════════
SCENE = r'''
from manim import *
import numpy as np

d           = np.load('/content/alphaphi_data.npz')
spec_frames = d['spec_frames']
coh_frames  = d['coh_frames']
beta_frames = d['beta_frames']
fold_times  = d['fold_times']
PHI         = float(d['PHI'][0])
FPS         = int(d['fps'][0])
N_BANDAS    = int(d['n_bandas'][0])
DURACAO     = float(d['duracao'][0])
N_FRAMES    = len(spec_frames)

COR_P = '#00FF88'
COR_S = '#FFB800'
COR_T = '#FF4466'
BG    = '#04040E'

N_RINGS = min(6, N_BANDAS)

class EsferaAlphaPhi(ThreeDScene):

    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=70*DEGREES, theta=-50*DEGREES)

        # ── tracker de tempo ────────────────────────────────
        tr = ValueTracker(0)

        def fi():
            return min(int(tr.get_value() * FPS), N_FRAMES - 1)

        # ── esfera central ──────────────────────────────────
        sphere = Sphere(radius=1.25, resolution=(22, 22))
        sphere.set_fill(GOLD_A, opacity=0.85)
        sphere.set_stroke(color=GOLD, width=0.4, opacity=0.5)

        def upd_sphere(mob):
            c = float(coh_frames[fi()])
            mob.set_fill(
                color=interpolate_color(BLUE_E, GOLD, c),
                opacity=0.75 + 0.20 * c
            )
        sphere.add_updater(upd_sphere)

        # glow externo
        glow = Sphere(radius=1.45, resolution=(10, 10))
        glow.set_fill(GOLD_E, opacity=0.06)
        glow.set_stroke(width=0)

        def upd_glow(mob):
            c = float(coh_frames[fi()])
            mob.set_fill(opacity=0.04 + 0.14 * c)
        glow.add_updater(upd_glow)

        # ── anéis espectrais φ ──────────────────────────────
        rings = []
        ring_colors = [GOLD, YELLOW_A, GREEN_A, TEAL_A, BLUE_A, PURPLE_A]
        for k in range(N_RINGS):
            r   = 1.55 + k * 0.18
            cor = ring_colors[k % len(ring_colors)]
            rng = Circle(radius=r, color=cor)
            rng.set_stroke(width=1.2, opacity=0.25)
            rng.rotate(PI/2, axis=RIGHT)
            band_idx = k
            def make_upd(bi):
                def upd(mob):
                    amp = float(spec_frames[fi(), bi]) if bi < N_BANDAS else 0.0
                    mob.set_stroke(opacity=0.10 + 0.70 * amp)
                    mob.set_stroke(width=0.8 + 1.5 * amp)
                return upd
            rng.add_updater(make_upd(band_idx))
            rings.append(rng)

        # ── eixo central (coluna de luz) ─────────────────────
        eixo = Line3D(start=[0,0,-2.2], end=[0,0,2.2],
                      radius=0.015, color=WHITE)
        eixo.set_opacity(0.18)

        # ── trajetória β (base, 2D fixo na câmera) ──────────
        axes_b = Axes(
            x_range=[0, N_FRAMES, N_FRAMES//4],
            y_range=[1.0, PHI**3 + 0.3, 1.0],
            x_length=5.5, y_length=1.0,
            axis_config={'color':'#1A1A3A','stroke_width':1},
        ).to_edge(DOWN, buff=0.08)

        ref_phi3 = DashedLine(
            axes_b.c2p(0, PHI**3), axes_b.c2p(N_FRAMES, PHI**3),
            color=GOLD_A, stroke_width=1.2, dash_length=0.06
        )
        lbl_phi3 = Text(f'φ³={PHI**3:.3f}', font_size=11, color=GOLD_A
                        ).next_to(axes_b.c2p(N_FRAMES, PHI**3), RIGHT, buff=0.05)

        beta_line = VMobject(color=COR_S, stroke_width=1.8)
        beta_pts  = []

        def upd_beta(mob):
            f = fi()
            while len(beta_pts) <= f:
                beta_pts.append(axes_b.c2p(len(beta_pts),
                                            float(beta_frames[len(beta_pts)])))
            if len(beta_pts) >= 2:
                mob.set_points_as_corners(beta_pts[:f+1])
        beta_line.add_updater(upd_beta)

        # ── rótulos fixos na câmera ──────────────────────────
        titulo = Text(
            'AlphaPhi · Esfera 880Hz · α*=1/3',
            font_size=18, color=WHITE
        ).to_edge(UP, buff=0.12)

        lbl_coh  = DecimalNumber(0, num_decimal_places=3,
                                 font_size=14, color=COR_P
                                 ).to_corner(UL, buff=0.25).shift(DOWN*0.5)
        lbl_beta = DecimalNumber(1, num_decimal_places=3,
                                 font_size=14, color=COR_S
                                 ).next_to(lbl_coh, DOWN, buff=0.12)
        pre_coh  = Text('coh ', font_size=14, color=COR_P
                        ).next_to(lbl_coh,  LEFT, buff=0.05)
        pre_beta = Text('β   ', font_size=14, color=COR_S
                        ).next_to(lbl_beta, LEFT, buff=0.05)

        lbl_coh.add_updater(lambda m: m.set_value(float(coh_frames[fi()])))
        lbl_beta.add_updater(lambda m: m.set_value(float(beta_frames[fi()])))

        for mob in [titulo, lbl_coh, lbl_beta, pre_coh, pre_beta,
                    axes_b, ref_phi3, lbl_phi3, beta_line]:
            self.add_fixed_in_frame_mobjects(mob)

        # ── montar cena ──────────────────────────────────────
        self.add(glow, sphere, eixo, *rings)
        self.play(
            Create(sphere), FadeIn(glow), FadeIn(eixo),
            FadeIn(titulo), FadeIn(axes_b), FadeIn(ref_phi3),
            run_time=0.8
        )

        # ── rotação de câmera ─────────────────────────────────
        self.begin_ambient_camera_rotation(rate=0.22)

        # ── animar pontos de dobra ────────────────────────────
        fold_labels = [
            (fold_times[0], 'P', COR_P),
            (fold_times[1], 'S', COR_S),
            (fold_times[2], 'T', COR_T),
        ]
        fold_labels.sort(key=lambda x: x[0])

        for ft, name, cor in fold_labels:
            dt = ft - tr.get_value()
            if dt > 0:
                self.play(tr.animate.set_value(ft),
                          run_time=dt, rate_func=linear)

            # evento: halo expansivo
            halo = Circle(radius=1.3, color=cor)
            halo.set_stroke(width=3.5, opacity=0.95)
            halo.rotate(PI/2, axis=RIGHT)
            self.add(halo)

            lbl_ev = Text(f'PONTO DE DOBRA  {name}',
                          font_size=16, color=cor, weight=BOLD)
            lbl_ev.move_to(DOWN * 1.9)
            self.add_fixed_in_frame_mobjects(lbl_ev)

            self.play(
                halo.animate.scale(3.2).set_stroke(opacity=0),
                sphere.animate.set_fill(opacity=0.98),
                run_time=0.55, rate_func=smooth
            )
            self.remove(halo)
            self.play(FadeOut(lbl_ev), run_time=0.25)

        # ── resto da animação até o fim ───────────────────────
        restante = DURACAO - tr.get_value()
        if restante > 0.05:
            self.play(tr.animate.set_value(DURACAO),
                      run_time=restante, rate_func=linear)

        # ── pausa final dourada ───────────────────────────────
        self.stop_ambient_camera_rotation()
        self.play(
            sphere.animate.set_fill(color=GOLD, opacity=0.97).scale(1.08),
            glow.animate.set_fill(opacity=0.22),
            run_time=1.0, rate_func=smooth
        )
        self.wait(1.2)
'''

with open('/content/esfera_scene.py', 'w') as f:
    f.write(SCENE)
print("✓ Cena Manim escrita")

# ── renderizar ─────────────────────────────────────────────────
print("\nRenderizando (qualidade média, ~4-8 min)…")
result = subprocess.run(
    [sys.executable, '-m', 'manim', '-qm', '--disable_caching',
     '/content/esfera_scene.py', 'EsferaAlphaPhi'],
    capture_output=True, text=True
)
if result.stdout: print(result.stdout[-1500:])
if result.returncode != 0:
    print("ERRO:")
    print(result.stderr[-1500:])

# ── exibir ─────────────────────────────────────────────────────
import glob
from IPython.display import Video, Audio, display

videos = sorted(glob.glob('/content/media/videos/**/*.mp4', recursive=True))
if videos:
    print(f"\n✓ Vídeo: {videos[-1]}")
    display(Video(videos[-1], embed=True, width=760))
else:
    # fallback: tentar qualidade baixa
    print("Vídeo não encontrado — tentando qualidade baixa…")
    subprocess.run(
        [sys.executable, '-m', 'manim', '-ql', '--disable_caching',
         '/content/esfera_scene.py', 'EsferaAlphaPhi'],
        capture_output=True, text=True
    )
    videos = sorted(glob.glob('/content/media/videos/**/*.mp4', recursive=True))
    if videos:
        display(Video(videos[-1], embed=True, width=760))

print("\nÁudio:")
display(Audio('/content/beep880_eco_final.wav'))
