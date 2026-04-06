# NOTA: Este é o protótipo original — versão histórica.
# Seed fixo np.random.seed(137) — substituído por seeds aleatórios.
# Versão corrigida: AlphaPhi_Original_RobustezEstrutura.ipynb
# Vitor Edson Delavi · Florianópolis · 2026
"""
ALPHA PHI — Protótipo Experimental
Vitor Edson · Florianópolis · 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy

PHI = (1 + np.sqrt(5)) / 2          # 1.6180339887

print(f"φ = {PHI:.10f}")

def golden_activation(x):
    return PHI * np.tanh(x / PHI)

def relu(x):
    return np.maximum(0, x)

def fibonacci_sequence(n_terms, start=8):
    fibs = [start]
    a, b = start, int(start * PHI)
    for _ in range(n_terms - 1):
        fibs.append(b)
        a, b = b, int(a + b)
    return fibs

fib_layers    = fibonacci_sequence(5, start=8)
uniform_layers = [34, 34, 34, 34, 34]

print(f"Fibonacci: {fib_layers}")
print(f"Uniforme:  {uniform_layers}")

class AlphaPhiNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            scale   = np.sqrt(1.0 / (fan_in * PHI))
            W = np.random.randn(fan_in, layer_sizes[i+1]) * scale
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        self.activations = [x]
        current = x
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = current @ W + b
            current = golden_activation(z)
            self.activations.append(current)
        z = current @ self.weights[-1] + self.biases[-1]
        self.activations.append(z)
        return z

    def weight_entropy(self):
        all_w = np.concatenate([W.flatten() for W in self.weights])
        hist, _ = np.histogram(all_w, bins=50, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        return scipy_entropy(hist)

class ConventionalNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            scale  = np.sqrt(2.0 / fan_in)
            W = np.random.randn(fan_in, layer_sizes[i+1]) * scale
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        self.activations = [x]
        current = x
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = current @ W + b
            current = relu(z)
            self.activations.append(current)
        z = current @ self.weights[-1] + self.biases[-1]
        self.activations.append(z)
        return z

    def weight_entropy(self):
        all_w = np.concatenate([W.flatten() for W in self.weights])
        hist, _ = np.histogram(all_w, bins=50, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        return scipy_entropy(hist)

rng = np.random.default_rng(137)
np.random.seed(137)  # mantido para retrocompatibilidade com resultados publicados

INPUT_DIM  = 16
OUTPUT_DIM = 4
N_SAMPLES  = 200

alphaphi_arch     = [INPUT_DIM] + fib_layers + [OUTPUT_DIM]
conventional_arch = [INPUT_DIM] + uniform_layers + [OUTPUT_DIM]

net_ap  = AlphaPhiNetwork(alphaphi_arch)
net_con = ConventionalNetwork(conventional_arch)

t = np.linspace(0, 4 * np.pi, N_SAMPLES)
X = np.column_stack([np.sin(t * (i+1) / PHI) for i in range(INPUT_DIM)])
X = (X - X.mean()) / X.std()

net_ap.forward(X)
net_con.forward(X)

entropy_ap  = net_ap.weight_entropy()
entropy_con = net_con.weight_entropy()

var_ap  = [np.var(a) for a in net_ap.activations]
var_con = [np.var(a) for a in net_con.activations]

grad_ap  = np.mean(np.abs(np.diff(var_ap)))
grad_con = np.mean(np.abs(np.diff(var_con)))

delta = ((entropy_con - entropy_ap) / entropy_con) * 100

print(f"\nRESULTADOS ALPHA PHI")
print(f"Entropia Alpha Phi:    {entropy_ap:.4f}")
print(f"Entropia Convencional: {entropy_con:.4f}")
print(f"Reducao de entropia:   {delta:.1f}%")
print(f"Estabilidade Alpha Phi:    {grad_ap:.4f}")
print(f"Estabilidade Convencional: {grad_con:.4f}")
print(f"Melhora estabilidade: {((grad_con-grad_ap)/grad_con)*100:.1f}%")
