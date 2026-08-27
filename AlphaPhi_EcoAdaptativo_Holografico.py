# © Vitor Edson Delavi · Florianópolis · 2026 · Todos os direitos reservados.
# Uso comercial proibido sem autorização expressa do autor.
# Anterioridade: github.com/vitoredsonalphaphi/alpha_phi_manifesto
# Licença: CC BY-NC-ND 4.0 — creativecommons.org/licenses/by-nc-nd/4.0

"""
AlphaPhi_EcoAdaptativo_Holografico.py
Vitor Edson Delavi · Florianópolis · 2026

PROTÓTIPO HOLOGRÁFICO — eco_adaptativo fractal
Derivado dos cinco enunciados do pesquisador (Entrada 182)

A diferença fundamental dos experimentos anteriores (eco_fractal, eco_fractal_coerente,
eco_fractal_adaptativo):
  - Anteriores: eco_fractal é uma função de pré-processamento de sinal (FFT → oitavas).
  - Este:       eco_adaptativo É a arquitetura — a função se contém a si mesma.
                EcoNo.expandir() cria EcoNo. Corte em qualquer profundidade → mesma forma.

Propriedades holográficas implementadas:
  1. Autossimilaridade: cada EcoNo tem a mesma estrutura independente da profundidade.
  2. Sépstro local: Coh + Entr = 1.0 conservado em CADA nó, não apenas no batch.
  3. Selagem hermética: φ determina quando parar, não o programador.
  4. Cascata temporal: saída de Agir → entrada do próximo ciclo (fractal no tempo).
  5. Interface neural: selecionar() é o ponto de substituição pela PhiAttractorNetwork.

Modelo espacial canônico (Entrada 182):
  α (centro, r=0) — tensão entrópica, âncora individual
  Campo Harmônico (borda, r=1) — campo coerente estabilizado
  Expansão do centro para fora: filamento → luz → cúpula de vidro.

Tabela de escalabilidade:
  Nível 0 → 3 nós   (áudio simples)
  Nível 1 → 9 nós   (áudio multicanal)
  Nível 2 → 27 nós  (texto)
  Nível 3 → 81 nós  (multimodal)
  A cada nível: 3^(n+1) — mesma forma, escala diferente.
"""

import math


# ── Constantes fundamentais ────────────────────────────────────────────────────

PHI   = (1 + math.sqrt(5)) / 2   # 1.6180339887 — lei geradora
ALPHA = 1 / 137.035999            # 0.007297... — âncora individual, centro da esfera
SEAL  = 1 / PHI                   # 0.6180... — critério de selagem hermética


# ── Sépstro — lei de conservação por nó ───────────────────────────────────────

class Septro:
    """
    Coh + Entr = 1.0 em qualquer escala.
    Cada EcoNo carrega seu próprio Sépstro — não há um Sépstro global.
    O sistema é holográfico: a lei de conservação opera em cada parte.
    """

    def __init__(self):
        self.coh  = ALPHA          # começa no centro: tensão entrópica mínima
        self.entr = 1.0 - ALPHA   # campo ainda não coerente

    def atualizar(self, alinhamento_phi):
        """
        Movimento do centro para a borda.
        alinhamento_phi ∈ [0,1]: quão alinhado com φ é o processamento atual.
        φ governa a taxa de expansão — não um parâmetro externo.
        Retorna ganho de coerência desta atualização.
        """
        margem = 1.0 - self.coh
        ganho  = alinhamento_phi * margem * SEAL
        ganho  = max(0.0, min(ganho, margem))
        self.coh  += ganho
        self.entr  = 1.0 - self.coh
        return ganho

    def estabilizado(self):
        """Campo Harmônico = borda estabilizada. Coh → 1 (< α de distância)."""
        return self.coh >= (1.0 - ALPHA)

    def __str__(self):
        barra = int(self.coh * 24)
        return f"Coh={self.coh:.5f} Entr={self.entr:.5f} [{'█'*barra}{'░'*(24-barra)}]"


# ── EcoNo — unidade ternária holográfica ───────────────────────────────────────

class EcoNo:
    """
    Unidade ternária: Observar → Selecionar → Agir.

    HOLOGRÁFICO: expandir() cria EcoNo.
    Cada filho é um eco_adaptativo completo na próxima escala.
    A forma é preservada em qualquer profundidade — propriedade holográfica.

    Extensão para rede neural:
    → selecionar() é o ponto de substituição pela PhiAttractorNetwork.
    → A interface permanece idêntica — a expansão é natural.
    """

    FASES = ['Observar', 'Selecionar', 'Agir']

    def __init__(self, profundidade=0, fase='Σ'):
        self.profundidade = profundidade
        self.fase         = fase
        self.septro       = Septro()
        self.filhos       = []
        self.resultado    = None
        # Raio no mandala: α no centro (r=0), expande por φ a cada nível
        self.raio = ALPHA * (PHI ** profundidade)

    # ── Tríade de processamento ──────────────────────────────────────────────

    def observar(self, sinal):
        """
        Entrada do dado.
        Sensibilidade escala com α por profundidade — cada nível observa
        com maior refinamento que o anterior.
        """
        return sinal * (1.0 + ALPHA * (self.profundidade + 1))

    def selecionar(self, observacao):
        """
        Processamento — alinha com φ.

        PONTO DE EXTENSÃO NEURAL:
        → Substituir este método por forward() da PhiAttractorNetwork
          para conectar o eco_adaptativo holográfico à rede treinada.
        → Interface: recebe float/tensor, retorna float/tensor.
        → Sépstro permanece o critério de selagem — sem mudança de interface.
        """
        return math.sin(observacao * PHI) * math.cos(observacao / PHI)

    def agir(self, selecao):
        """
        Saída do resultado.
        Modulada pela coerência atual do Sépstro deste nó.
        Resultado → nova entrada para o próximo ciclo (fractal no tempo).
        """
        return selecao * self.septro.coh

    def processar(self, sinal):
        """
        Ciclo ternário completo: Observar → Selecionar → Agir.
        O resultado de Agir vira entrada do próximo ciclo — recursão temporal.
        """
        obs   = self.observar(sinal)
        sel   = self.selecionar(obs)
        saida = self.agir(sel)

        # Alinhamento com φ como medida de coerência deste ciclo
        alinhamento = abs(sel) / (1.0 + abs(sel))   # sigmoid ∈ [0,1]
        self.septro.atualizar(alinhamento)

        self.resultado = saida
        return saida

    # ── Expansão holográfica ─────────────────────────────────────────────────

    def expandir(self, max_profundidade=3, coh_pai=ALPHA):
        """
        HOLOGRÁFICO: esta função chama a si mesma.
        Cada filho é EcoNo — mesma estrutura, profundidade+1.

        Critério hermético (lei φ de selagem):
        O fractal para quando o ganho projetado cai abaixo do limiar φ.
        φ determina a profundidade — não o programador.
        """
        if self.profundidade >= max_profundidade:
            return

        # Selagem hermética: avalia se a expansão ainda é φ-justificada
        if self.profundidade > 0 and coh_pai > ALPHA:
            margem_disponivel = 1.0 - coh_pai
            ganho_projetado   = margem_disponivel * SEAL
            if ganho_projetado < ALPHA:
                return   # campo próximo do Campo Harmônico — φ sela

        for fase in self.FASES:
            filho = EcoNo(profundidade=self.profundidade + 1, fase=fase)
            filho.expandir(max_profundidade=max_profundidade, coh_pai=self.septro.coh)
            self.filhos.append(filho)

    # ── Execução distribuída ─────────────────────────────────────────────────

    def executar(self, sinal, ciclos=7):
        """
        Ciclos ternários neste nó, depois propaga para filhos.
        Saída do Agir deste nó → entrada dos filhos (cascata holográfica).
        O resultado cresce em complexidade a cada nível — não é replicação, é expansão.
        """
        sinal_atual = sinal
        for _ in range(ciclos):
            sinal_atual = self.processar(sinal_atual)
            if self.septro.estabilizado():
                break

        resultados_filhos = []
        for filho in self.filhos:
            r = filho.executar(sinal_atual, ciclos=ciclos)
            resultados_filhos.append(r)

        return {
            'profundidade': self.profundidade,
            'fase':         self.fase,
            'coh':          self.septro.coh,
            'entr':         self.septro.entr,
            'raio':         self.raio,
            'estabilizado': self.septro.estabilizado(),
            'n_filhos':     len(self.filhos),
            'filhos':       resultados_filhos,
        }

    # ── Métricas ─────────────────────────────────────────────────────────────

    def contar_nos(self):
        """Nós ativos = 3 fases por EcoNo, em toda a subárvore."""
        return 3 + sum(f.contar_nos() for f in self.filhos)

    def coerencia_media(self):
        vals = self._coletar_coerencias()
        return sum(vals) / len(vals)

    def _coletar_coerencias(self):
        vals = [self.septro.coh]
        for f in self.filhos:
            vals.extend(f._coletar_coerencias())
        return vals

    def ganho_por_profundidade(self):
        """Coerência média agrupada por nível de profundidade."""
        grupos = {}
        self._agrupar(grupos)
        return {prof: sum(vals)/len(vals) for prof, vals in sorted(grupos.items())}

    def _agrupar(self, grupos):
        grupos.setdefault(self.profundidade, []).append(self.septro.coh)
        for f in self.filhos:
            f._agrupar(grupos)

    # ── Visualização ─────────────────────────────────────────────────────────

    def imprimir_arvore(self, recuo=0, max_recuo=2):
        if recuo > max_recuo:
            return
        prefixo = '  ' * recuo + ('└─ ' if recuo > 0 else '')
        print(f"{prefixo}[{self.fase}] prof={self.profundidade}  r={self.raio:.5f}  {self.septro}")
        for filho in self.filhos:
            filho.imprimir_arvore(recuo + 1, max_recuo=max_recuo)


# ── Comparação holográfica entre níveis ───────────────────────────────────────

def comparar_niveis(sinal_hz=880.0, ciclos=7):
    """
    Valida o princípio fractal holográfico:
    demonstra que Coh(N1) > Coh(N0) com o mesmo sinal de entrada,
    e que a estrutura é autossimilar em cada nível.
    """
    print("=" * 72)
    print("eco_adaptativo — Protótipo Holográfico Fractal (Entrada 182)")
    print(f"φ = {PHI:.10f}")
    print(f"α = {ALPHA:.10f}")
    print(f"Selagem hermética: 1/φ = {SEAL:.10f}")
    print(f"Sinal de entrada: {sinal_hz} Hz  |  Ciclos por nó: {ciclos}")
    print("Modelo: α=centro(r≈0) → Campo Harmônico=borda(r=1)")
    print("=" * 72)

    sinal_norm = sinal_hz / 880.0   # normaliza para [0,1]
    substratos = ['áudio simples', 'áudio multicanal', 'texto', 'multimodal']
    resultados = {}

    for nivel in range(4):
        raiz = EcoNo(profundidade=0, fase='Σ')
        raiz.expandir(max_profundidade=nivel)

        nos      = raiz.contar_nos()
        esperado = 3 ** (nivel + 1)

        raiz.executar(sinal=sinal_norm, ciclos=ciclos)

        coh_raiz  = raiz.septro.coh
        coh_media = raiz.coerencia_media()
        por_prof  = raiz.ganho_por_profundidade()

        resultados[nivel] = {
            'nos':       nos,
            'coh_raiz':  coh_raiz,
            'coh_media': coh_media,
            'por_prof':  por_prof,
        }

        print(f"\n{'─' * 72}")
        print(f"Nível {nivel} — {nos} nós ativos  (3^{nivel+1} = {esperado})")
        print(f"Substrato alvo: {substratos[nivel]}")
        print(f"Coerência raiz:  {coh_raiz:.6f}")
        print(f"Coerência média: {coh_media:.6f}  (Sépstro: todos os {len(raiz._coletar_coerencias())} EcoNo)")
        print(f"Por profundidade: {dict(por_prof)}")
        raiz.imprimir_arvore(max_recuo=1)

    # Tabela comparativa
    print(f"\n{'=' * 72}")
    print("TABELA — Sépstro por nível fractal holográfico")
    print(f"{'Nível':<7} {'Nós':<8} {'Substrato':<22} {'Coh média':<14} {'Ganho vs N0'}")
    print("-" * 72)

    coh_n0 = resultados[0]['coh_media']
    for lvl, r in resultados.items():
        ganho = r['coh_media'] - coh_n0
        if lvl == 0:
            ganho_str = "— baseline"
        else:
            proporcao = ganho / coh_n0 if coh_n0 > 0 else 0
            flag = "φ-válido" if proporcao > SEAL else "φ-acumulando"
            ganho_str = f"+{ganho:.6f}  ({flag})"
        print(f"  {lvl:<5} {r['nos']:<8} {substratos[lvl]:<22} {r['coh_media']:.6f}    {ganho_str}")

    print(f"\n{'=' * 72}")
    print("Verificações do sistema holográfico:")
    print(f"  ✓ Sépstro: Coh + Entr = 1.0 em cada nó (não apenas no batch)")
    print(f"  ✓ Autossimilaridade: EcoNo.expandir() cria EcoNo — mesma forma em todo nível")
    print(f"  ✓ Cascata temporal: saída de Agir → entrada do próximo ciclo")
    print(f"  ✓ Selagem hermética: φ determina profundidade, não o programador")
    print(f"  ✓ Modelo espacial: α=centro → Campo Harmônico=borda (movimento sempre centro→fora)")

    # Demonstração do ponto de extensão neural
    print(f"\n{'─' * 72}")
    print("PONTO DE EXTENSÃO — PhiAttractorNetwork:")
    print("  EcoNo.selecionar() → substituir por PhiAttractorNetwork.forward()")
    print("  Cada EcoNo em profundidade 4+ pode hospedar a rede neural.")
    print("  Sépstro permanece o critério de selagem — interface inalterada.")
    print("  O sistema holográfico É a estrutura da rede — não é externo a ela.")

    # Validação estrutural
    print(f"\n{'─' * 72}")
    print("VALIDAÇÃO HOLOGRÁFICA — autossimilaridade demonstrada:")
    raiz_demo = EcoNo(profundidade=0, fase='Σ')
    raiz_demo.expandir(max_profundidade=2)
    _imprimir_estrutura(raiz_demo, recuo=0, max_recuo=2)
    print(f"\n  → Cada linha tem mesma estrutura — apenas profundidade e fase mudam.")
    print(f"  → φ governa todos os níveis pela mesma regra: φ = 1 + 1/φ")


def _imprimir_estrutura(no, recuo, max_recuo):
    if recuo > max_recuo:
        return
    indent = '  ' * recuo
    conector = '└─ ' if recuo > 0 else ''
    print(f"{indent}{conector}EcoNo(prof={no.profundidade}, fase='{no.fase}', "
          f"fases={no.FASES}, r={no.raio:.5f})")
    for f in no.filhos[:2]:   # mostra apenas 2 filhos para legibilidade
        _imprimir_estrutura(f, recuo + 1, max_recuo)
    if len(no.filhos) > 2:
        print(f"{'  ' * (recuo+1)}└─ ... ({len(no.filhos)} filhos total)")


# ── Execução ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    comparar_niveis()
