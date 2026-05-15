"""
AlphaPhi_Pacotes_Rede.py
Teste do campo harmônico em pacotes de rede binários simulados.

Metodologia:
  1. Gerar pacotes binários sintéticos (header estruturado + payload variável)
  2. Codificar como sinal NRZ float (-1/+1)
  3. Processar com agente eco-φ
  4. Adicionar ruído de canal (SNR variável)
  5. Decodificar de volta para binário
  6. Medir taxa de erro de bit (BER) — convencional vs campo harmônico

Observação honesta: o resultado pode mostrar melhora, piora ou nenhuma diferença.

© Vitor Edson Delavi · Florianópolis · 2026
"""

import numpy as np
import zlib
import json

PHI        = (1 + np.sqrt(5)) / 2
FS         = 44100
DURACAO    = 1.5
N_STEPS    = 5
N_CICLOS   = 20

def normalizar(s):
    m = np.max(np.abs(s)); return s / m if m > 1e-12 else s

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
             min(int(f_hi/(FS/n))+1, n//2+1), f_lo, f_hi)
            for f_lo, f_hi in bandas]

N_SINAL  = int(FS * DURACAO)
BANDAS   = gerar_bandas_phi()
BINS_PHI = bandas_para_bins(BANDAS, N_SINAL)

def eco_eq(x, bins_phi, beta_bands, coh_mem=None):
    beta_bands = np.atleast_1d(np.asarray(beta_bands, dtype=float))
    if coh_mem is not None:
        coh_mem = np.atleast_1d(np.asarray(coh_mem, dtype=float))
    N, F = len(x), np.fft.rfft(x)
    F_out, cohs = F.copy(), []
    wm, wn = 1.0/PHI, 1.0 - 1.0/PHI
    for i, (b_lo, b_hi, _, _) in enumerate(bins_phi):
        bi   = float(beta_bands[i]) if i < len(beta_bands) else 1.0
        Fb   = F[b_lo:b_hi]
        mag  = np.abs(Fb); phase = np.angle(Fb)
        an   = np.clip(mag/(mag.sum()+1e-8), 1e-10, 1.0)
        coh  = float(1.0-(-np.sum(an*np.log(an)))/np.log(max(len(an),2)))
        ce   = (wn*coh + wm*float(coh_mem[i])
                if (coh_mem is not None and i < len(coh_mem)) else coh)
        cohs.append(coh)
        nk   = np.arange(len(Fb))
        env  = np.clip(1.0+(ce*PHI**bi)*np.cos(2*np.pi*nk/PHI), 0.05, None)
        F_out[b_lo:b_hi] = (mag*env)*np.exp(1j*phase)
    r = np.fft.irfft(F_out, n=N)
    return r/(np.max(np.abs(r))+1e-10), np.array(cohs)

def agente_eco(sinal, bins_phi, n_ciclos=20, coh_mem_inicial=None):
    nb = len(bins_phi)
    beta = np.ones(nb); bm = beta.copy()
    wm, wn = 1.0/PHI, 1.0-1.0/PHI
    cm = coh_mem_inicial if coh_mem_inicial is not None else np.zeros(nb)
    for _ in range(n_ciclos):
        s = sinal.copy()
        for _ in range(N_STEPS):
            s, cohs = eco_eq(s, bins_phi, beta, cm)
            cm = cohs; s = normalizar(s)
        cr   = (cohs-cohs.min())/(cohs.max()-cohs.min()+1e-10)
        ba   = PHI**(3*cr)
        beta = wn*ba + wm*bm; bm = beta.copy()
        beta = np.clip(beta, 0.05, PHI**3)
    return beta, s, cohs

# ── geração de pacotes sintéticos ────────────────────────────

def gerar_pacote(n_bits=1024, seed=None):
    """
    Simula pacote de rede:
      - 20% header: bits estruturados (padrões repetitivos como IP/TCP)
      - 80% payload: conteúdo misto (texto compressível + dados aleatórios)
    """
    rng = np.random.default_rng(seed)
    n_header  = n_bits // 5
    n_payload = n_bits - n_header

    # header: padrões estruturados (simula campos IP/TCP)
    header = np.zeros(n_header, dtype=np.int8)
    # campos fixos (versão, protocolo, flags)
    for i in range(0, n_header, 8):
        padrao = rng.integers(0, 2, size=4)  # primeiros 4 bits: campo fixo
        header[i:i+4] = padrao
        header[i+4:i+8] = rng.integers(0, 2, size=min(4, n_header-i-4))

    # payload: 50% texto compressível + 50% aleatório
    n_texto = n_payload // 2
    n_rand  = n_payload - n_texto
    texto   = np.tile(rng.integers(0, 2, size=64), n_texto // 64 + 1)[:n_texto]
    aleato  = rng.integers(0, 2, size=n_rand, dtype=np.int8)

    return np.concatenate([header, texto, aleato]).astype(np.int8)

def bits_para_sinal_nrz(bits):
    """Codificação NRZ: 0→-1, 1→+1"""
    return (bits.astype(float) * 2 - 1)

def sinal_para_bits_nrz(sinal):
    """Decodificação NRZ: <0→0, ≥0→1"""
    return (sinal >= 0).astype(np.int8)

def adicionar_ruido(sinal, snr_db):
    """Adiciona ruído gaussiano com SNR em dB."""
    potencia_sinal = np.mean(sinal**2)
    potencia_ruido = potencia_sinal / (10 ** (snr_db / 10))
    ruido = np.random.randn(len(sinal)) * np.sqrt(potencia_ruido)
    return sinal + ruido

def taxa_erro_bits(original, recuperado):
    n = min(len(original), len(recuperado))
    return float(np.sum(original[:n] != recuperado[:n])) / n

def entropia_bits(bits):
    p1 = np.mean(bits)
    p0 = 1 - p1
    if p1 <= 0 or p0 <= 0: return 0.0
    return float(-(p0*np.log2(p0) + p1*np.log2(p1)))

def compressao(bits):
    dados = np.packbits(bits).tobytes()
    return len(zlib.compress(dados, 9)) / len(dados)

# ── adaptador: bits → sinal FS → campo harmônico → bits ─────

def aplicar_campo_harmonico(bits):
    """
    Adapta bits para o campo harmônico:
    1. NRZ → sinal float
    2. Redimensiona para N_SINAL
    3. Processa com agente eco-φ
    4. Redimensiona de volta
    5. Decodifica NRZ
    """
    sinal_nrz = bits_para_sinal_nrz(bits)
    n_orig = len(sinal_nrz)

    # interpola para N_SINAL
    idx_orig = np.linspace(0, n_orig-1, N_SINAL)
    sinal_fs = np.interp(idx_orig, np.arange(n_orig), sinal_nrz)
    sinal_fs = normalizar(sinal_fs)

    # campo harmônico
    _, sinal_proc, _ = agente_eco(sinal_fs, BINS_PHI, N_CICLOS)

    # volta para tamanho original
    idx_back = np.linspace(0, N_SINAL-1, n_orig)
    sinal_rec = np.interp(idx_back, np.arange(N_SINAL), sinal_proc)

    return sinal_rec

# ── experimento principal ─────────────────────────────────────

print("=" * 66)
print("  AlphaPhi · Campo Harmônico em Pacotes de Rede Binários")
print("  Observação honesta — resultado pode ser positivo ou negativo")
print("=" * 66)

N_PACOTES = 10
N_BITS    = N_SINAL  # mesmo tamanho do sinal FS para precisão máxima
SNR_LEVELS = [3, 6, 10, 15, 20]  # dB — do mais ruidoso ao mais limpo

print(f"\n  {N_PACOTES} pacotes · {N_BITS} bits/pacote")
print(f"  SNR testado: {SNR_LEVELS} dB")
print(f"\n  Gerando e processando pacotes…\n")

resultados_snr = []

for snr in SNR_LEVELS:
    ber_conv_list  = []
    ber_phi_list   = []
    comp_orig_list = []
    comp_phi_list  = []
    ent_orig_list  = []
    ent_phi_list   = []

    for k in range(N_PACOTES):
        # 1. gerar pacote
        bits_orig = gerar_pacote(N_BITS, seed=k)

        # 2. sinal convencional NRZ
        sinal_conv = bits_para_sinal_nrz(bits_orig)

        # 3. sinal com campo harmônico
        sinal_phi = aplicar_campo_harmonico(bits_orig)

        # 4. adicionar ruído de canal
        sinal_conv_ruido = adicionar_ruido(sinal_conv, snr)
        sinal_phi_ruido  = adicionar_ruido(sinal_phi,  snr)

        # 5. decodificar
        bits_conv_rec = sinal_para_bits_nrz(sinal_conv_ruido)
        bits_phi_rec  = sinal_para_bits_nrz(sinal_phi_ruido)

        # 6. medir
        ber_conv_list.append(taxa_erro_bits(bits_orig, bits_conv_rec))
        ber_phi_list.append(taxa_erro_bits(bits_orig, bits_phi_rec))
        comp_orig_list.append(compressao(bits_orig))
        comp_phi_list.append(compressao(bits_phi_rec))
        ent_orig_list.append(entropia_bits(bits_orig))
        ent_phi_list.append(entropia_bits(bits_phi_rec))

    ber_conv = np.mean(ber_conv_list)
    ber_phi  = np.mean(ber_phi_list)
    delta_ber = ber_conv - ber_phi  # positivo = φ melhor

    resultados_snr.append({
        'snr_db':    snr,
        'ber_conv':  round(ber_conv,  6),
        'ber_phi':   round(ber_phi,   6),
        'delta_ber': round(delta_ber, 6),
        'comp_orig': round(np.mean(comp_orig_list), 4),
        'comp_phi':  round(np.mean(comp_phi_list),  4),
        'ent_orig':  round(np.mean(ent_orig_list),  4),
        'ent_phi':   round(np.mean(ent_phi_list),   4),
    })

# ── resultados ───────────────────────────────────────────────
print(f"  {'SNR':>6}  {'BER conv':>10}  {'BER φ':>10}  {'Δ BER':>10}  {'φ melhor?':>10}")
print(f"  {'-'*58}")

for r in resultados_snr:
    melhor = "SIM ✓" if r['delta_ber'] > 0 else ("NÃO ✗" if r['delta_ber'] < 0 else "IGUAL")
    print(f"  {r['snr_db']:>4}dB  "
          f"{r['ber_conv']:>10.6f}  "
          f"{r['ber_phi']:>10.6f}  "
          f"{r['delta_ber']:>+10.6f}  "
          f"{melhor:>10}")

print(f"\n  {'SNR':>6}  {'Comp orig':>10}  {'Comp φ':>10}  {'Ent orig':>10}  {'Ent φ':>10}")
print(f"  {'-'*58}")
for r in resultados_snr:
    print(f"  {r['snr_db']:>4}dB  "
          f"{r['comp_orig']:>10.4f}  "
          f"{r['comp_phi']:>10.4f}  "
          f"{r['ent_orig']:>10.4f}  "
          f"{r['ent_phi']:>10.4f}")

# ── interpretação ─────────────────────────────────────────────
print("\n" + "=" * 66)
print("  INTERPRETAÇÃO HONESTA")
print("=" * 66)

melhoras = sum(1 for r in resultados_snr if r['delta_ber'] > 1e-6)
pioras   = sum(1 for r in resultados_snr if r['delta_ber'] < -1e-6)
iguais   = len(resultados_snr) - melhoras - pioras

print(f"""
  Em {len(resultados_snr)} níveis de SNR testados:
  → Campo φ melhor : {melhoras} casos
  → Campo φ pior   : {pioras} casos
  → Sem diferença  : {iguais} casos
""")

if melhoras > pioras:
    print("  → O campo harmônico mostrou VANTAGEM na maioria dos casos.")
    print("    Hipótese de eficiência de canal tem suporte neste teste.")
elif pioras > melhoras:
    print("  → O campo harmônico mostrou DESVANTAGEM na maioria dos casos.")
    print("    A transformação introduz distorção que aumenta erros.")
    print("    Hipótese NÃO confirmada — requer reformulação.")
else:
    print("  → Resultado NEUTRO — campo φ não melhora nem piora BER.")
    print("    A vantagem pode estar em outra dimensão (compressão, latência).")

print(f"\n  φ³ = {PHI**3:.4f}")
print("=" * 66)

with open('/home/user/alpha_phi_manifesto/pacotes_rede_results.json', 'w', encoding='utf-8') as f:
    json.dump({'phi3': PHI**3, 'resultados': resultados_snr}, f, indent=2)

print("  Resultados salvos: pacotes_rede_results.json")
