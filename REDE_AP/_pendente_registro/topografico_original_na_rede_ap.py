# O scanner topográfico ORIGINAL (AlphaPhi_Scanner_TopogColab.py) aplicado ao sinal digital da Rede AP esférica.
# As funções do scanner são extraídas do arquivo original e usadas SEM alteração (caminho dos sinais não sonoros, o mesmo dos textos:
# STFT com SR simbólico 1, janela 256, passo 64; altura log1p(S·1e6); cor do ECO = PLV). Só o sinal de entrada muda:
# o valor de cada neurônio, lido na ordem do endereço canônico da Grade v2 (índice do ângulo áureo), volta após volta.
import numpy as np, ast, importlib.util, json
import os
H = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(H, '..', '..', 'AlphaPhi_Scanner_TopogColab.py')
arv = ast.parse(open(SRC).read()); ns = {}
pre = "import numpy as np\nfrom scipy.signal import stft\nfrom scipy.ndimage import gaussian_filter, laplace\n"
consts = [n for n in arv.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('PHI','ALPHA','SEAL','THETA_R','SR','BASE') for t in n.targets)]
funcs = [n for n in arv.body if isinstance(n, ast.FunctionDef) and n.name in ('_norm','stft_texto','stft_texto_cplx','compute_plv','compute_teto','compute_lap','downsample_grid')]
exec(pre + ast.unparse(ast.Module(body=consts + funcs, type_ignores=[])), ns)
print('funções originais carregadas:', [f.name for f in funcs])
SP = H + '/'
s = importlib.util.spec_from_file_location('LA', SP + 'AlphaPhi_CampoEsferico_EtapaLA_LigacoesAlturas.py'); LA = importlib.util.module_from_spec(s); s.loader.exec_module(LA)
NA, FM = LA.NA, LA.FM
def recuperar_hist(J, X, S0, rng, g, ancora=True, previsor=True):   # a mesma dinâmica da Etapa NA (N3), guardando cada volta
    S = S0.copy(); hesc = np.abs(S0 @ J).mean(1, keepdims=True); mu_t = NA.wmean(X).mean(); sd_t = NA.wstd(X).mean(); H = [S.copy()]
    for v in range(FM.MAXV):
        extra = FM.FORCA_PREVISOR * FM.previsor(S, X) if previsor else np.zeros_like(S); mud = 0.
        for i in rng.permutation(NA.N):
            novo = np.tanh(g * (S @ J[:, i] / hesc[:, 0] + extra[:, i])); mud = max(mud, float(np.abs(novo - S[:, i]).max())); S[:, i] = novo
        if ancora:
            m = NA.wmean(S); S = S - 0.5 * (m - mu_t)[:, None]; m = NA.wmean(S); sd = NA.wstd(S)
            S = np.clip(m[:, None] + (S - m[:, None]) * (1 + 0.5 * (sd_t / np.maximum(sd, 1e-9) - 1))[:, None], -1, 1)
        H.append(S.copy())
        if mud < FM.TOL: break
    return H
s0 = 190; X = NA.memorias(s0); S0 = NA.pista(X, 1.0, s0); mem = 0
sinais = {}
for nome, J in (('Rede AP · regra delta', LA.ligacao_delta(X)), ('Rede AP · regra de Hebb', FM.ligacoes(NA.P2, X))):
    g = NA.ganho_ajustado(J, X); H = recuperar_hist(J, X, S0, np.random.default_rng([2_800_000 + s0, 10]), g)
    while len(H) < 51: H.append(H[-1])            # completa as 50 voltas com o estado final (a rede parou)
    sinais[nome] = np.concatenate([h[mem] for h in H])      # 51 × 610 valores, na ordem do endereço
    print(nome, 'voltas reais', len([1 for _ in H]), 'amostras', len(sinais[nome]))
sinais['A memória guardada (repetida)'] = np.tile(X[mem], 51)
out = {}
for nome, sig in sinais.items():
    sig = sig - sig.mean()
    fv, tv, S = ns['stft_texto'](sig); _, _, Zc = ns['stft_texto_cplx'](sig)
    Sl = np.log1p(S * 1e6); PLV = ns['compute_plv'](Zc); Teto = ns['compute_teto'](Sl); Lap, _ = ns['compute_lap'](Sl)
    f1, t1, Sl1 = ns['downsample_grid'](fv, tv, Sl); _, _, P1 = ns['downsample_grid'](fv, tv, PLV)
    _, _, T1 = ns['downsample_grid'](fv, tv, Teto); _, _, L1 = ns['downsample_grid'](fv, tv, Lap)
    out[nome] = {'f': np.round(f1, 4).tolist(), 't': np.round(t1 / 610, 3).tolist(), 'Sl': np.round(Sl1, 3).tolist(), 'PLV': np.round(P1, 3).tolist(),
                 'Teto': np.round(T1, 3).tolist(), 'Lap': np.round(L1, 4).tolist()}
    print(nome, 'grade', Sl1.shape)
json.dump(out, open(SP + 'topografico_rede_ap.json', 'w'), separators=(',', ':'), ensure_ascii=False)
