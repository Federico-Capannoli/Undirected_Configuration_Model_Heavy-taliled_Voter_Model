"""
Undirected Voter Model on Configuration Graphs
Discordant Edges vs Opinion Density

This script simulates the voter model on undirected networks generated via the
configuration model (CM). Degree sequences can be either regular or heavy-tailed
(Pareto-distributed). The evolution of discordant edges and opinion density is
tracked until consensus, and empirical results are compared with a theoretical
parabolic prediction.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy as sp
import pandas as pd


def deg_sequence_CM(k, alpha, k_min, regular=False):
    """
    Generate a degree sequence for the undirected configuration model.

    Parameters
    ----------
    k : int
        Number of nodes.
    alpha : float
        Pareto tail exponent.
    k_min : int
        Minimum degree.
    regular : bool, optional
        If True, generate a k_min-regular degree sequence.

    Returns
    -------
    degrees : np.ndarray
        Valid degree sequence with even total degree.
    """
    if regular:
        degrees = np.ones(k) * k_min
        return degrees.astype(int)

    while True:
        degrees = np.floor(np.random.pareto(alpha, k)) + k_min
        if not sum(degrees) % 2:
            return degrees.astype(int)


###################################################################################################
# Simulation parameters
###################################################################################################

alpha = 0.8
k_min = 4
regular = True

n = 5000
U_values = [0.2, 0.4, 0.6, 0.8]


###################################################################################################
# Voter-model simulation
###################################################################################################

avg_deg = []
total_discordant_edges = {U: [] for U in U_values}
total_blues = {U: [] for U in U_values}

for U in U_values:

    fraction_discordant_edges = []
    fraction_blue_vertices = []

    degree_sequence = deg_sequence_CM(n, alpha, k_min, regular)
    G = nx.configuration_model(degree_sequence)

    while not nx.is_connected(G):
        G = nx.configuration_model(degree_sequence)

    M = len(G.edges())

    node_colors = {i: "red" if random.random() < U else "blue" for i in range(n)}
    discordant_edges = {(u, v) for u, v in G.edges() if node_colors[u] != node_colors[v]}

    fraction_discordant_edges.append(len(discordant_edges) / M)

    blues = sum(1 for color in node_colors.values() if color == "blue")
    fraction_blue_vertices.append(blues / n)

    time = 0
    nodes = list(G.nodes)

    while discordant_edges:

        u = random.choice(nodes)
        v = random.choice(list(G.neighbors(u)))

        if node_colors[u] != node_colors[v]:

            if node_colors[v] == "blue" and node_colors[u] == "red":
                blues += 1
            else:
                blues -= 1

            node_colors[u] = node_colors[v]

            for neighbor in G.neighbors(u):
                edge = (u, neighbor) if u < neighbor else (neighbor, u)
                if node_colors[u] == node_colors[neighbor]:
                    discordant_edges.discard(edge)
                else:
                    discordant_edges.add(edge)

            for neighbor in G.neighbors(v):
                edge = (v, neighbor) if v < neighbor else (neighbor, v)
                if node_colors[v] == node_colors[neighbor]:
                    discordant_edges.discard(edge)
                else:
                    discordant_edges.add(edge)

        fraction_blue_vertices.append(blues / n)
        fraction_discordant_edges.append(len(discordant_edges) / M)
        time += 1

        if len(discordant_edges) == 0:
            avg_deg.append((1 / n) * sum(degree_sequence))
            total_discordant_edges[U].append(fraction_discordant_edges)
            total_blues[U].append(fraction_blue_vertices)


###################################################################################################
# Theoretical comparison and visualization
###################################################################################################

avg_deg_total = np.mean(avg_deg)
theta_d = (avg_deg_total - 2) / (avg_deg_total - 1)

max_len = max([max(map(len, total_discordant_edges[U])) for U in U_values])
parabola = [2 * theta_d * x * (1 - x) for x in np.linspace(0, 1, max_len)]

colors = ['yellow', 'blue', 'green', 'orange']

plt.figure()
for i, U in enumerate(U_values):
    plt.scatter(
        total_blues[U],
        total_discordant_edges[U],
        color=colors[i],
        s=10
    )

plt.xlabel('fraction blue vertices')
plt.ylabel('fraction discordant edges')

if regular:
    plt.title(f"{k_min}-regular CM")
else:
    plt.title(f"CM with alpha = {alpha}")

plt.show()
