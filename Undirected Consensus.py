"""
Undirected Voter Model on Configuration Graphs
Consensus Time Scaling

This script studies the scaling of consensus times for the voter model on
undirected configuration model graphs with heavy-tailed degree distributions.
Consensus times are measured across different system sizes and compared with
theoretical scaling predictions depending on the Pareto exponent alpha.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy as sp
import pandas as pd


def deg_sequence_CM(k, alpha, k_min):
    """
    Generate a valid degree sequence for the undirected configuration model
    using a Pareto distribution.

    Parameters
    ----------
    k : int
        Number of nodes.
    alpha : float
        Pareto tail exponent.
    k_min : int
        Minimum degree.

    Returns
    -------
    degrees : np.ndarray
        Degree sequence with even total degree.
    """
    while True:
        degrees = np.floor(np.random.pareto(alpha, k)) + k_min
        if not sum(degrees) % 2:
            return degrees.astype(int)


###################################################################################################
# Parameters
###################################################################################################

alpha = 3
k_min = 5
U = 0.5  # Initial proportion of red opinions

n_min = 100
n_max = 3000
number_of_iterations_n = 15

iteration_VM = 10
iteration_Graph = 10
iteration_Degree_sequence = 1

n_values = np.linspace(n_min, n_max, number_of_iterations_n).astype(int)


###################################################################################################
# Storage structures
###################################################################################################

All_Consensus_times = {n: [] for n in n_values}
Consensus_times_degree_sequences = {n: [] for n in n_values}
Average_Consensus_times = []


###################################################################################################
# Main simulation loop
###################################################################################################

for n in n_values:

    Consensus_times_Degree_Sequence_average = []

    for Iter_deg_seq in range(iteration_Degree_sequence):

        Consensus_times_Graph_average = []
        degree_sequence = deg_sequence_CM(n, alpha, k_min)

        for Iter_graph in range(iteration_Graph):

            G = nx.configuration_model(degree_sequence)
            while not nx.is_connected(G):
                G = nx.configuration_model(degree_sequence)

            Consensus_times_VM = []

            for Iter_VM in range(iteration_VM):

                node_colors = {
                    i: "red" if random.random() < U else "blue"
                    for i in range(n)
                }

                discordant_edges = {
                    (u, v) for u, v in G.edges()
                    if node_colors[u] != node_colors[v]
                }

                time = 0
                nodes = list(G.nodes)

                while discordant_edges:

                    u = random.choice(nodes)
                    v = random.choice(list(G.neighbors(u)))

                    if node_colors[u] != node_colors[v]:
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

                    time += 1

                    if len(discordant_edges) == 0:
                        consensus = True

                Consensus_times_VM.append(time / n)
                All_Consensus_times[n].append(time / n)

                print(
                    f'n = {n}/{n_max}, '
                    f'Iteration Deg_Seq: {Iter_deg_seq + 1}/{iteration_Degree_sequence}, '
                    f'Iteration Graph: {Iter_graph + 1}/{iteration_Graph}, '
                    f'Iteration VM: {Iter_VM + 1}/{iteration_VM}, '
                    f'Time: {time / n}'
                )

            Consensus_times_Graph_average.append(np.mean(Consensus_times_VM))

        Consensus_times_Degree_Sequence_average.append(
            np.mean(Consensus_times_Graph_average)
        )
        Consensus_times_degree_sequences[n].append(
            np.mean(Consensus_times_Graph_average)
        )

    Average_Consensus_times.append(
        np.mean(Consensus_times_Degree_Sequence_average)
    )


###################################################################################################
# Theoretical scaling regimes and visualization
###################################################################################################

nu = alpha + 1

if 0 < alpha < 1:

    Consensus_times_VM_avg = {n: [] for n in n_values}
    for n in n_values:
        lst = All_Consensus_times[n]
        Consensus_times_VM_avg[n] = [
            sum(lst[i:i + 10]) / 10 for i in range(0, len(lst), 10)
        ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [All_Consensus_times[n] for n in n_values],
        positions=n_values,
        widths=0.1 * n_values
    )

    plt.xlabel('N')
    plt.ylabel('Average Consensus Time')
    plt.plot(n_values, Average_Consensus_times, 'o', label='Average Consensus')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.title(f'CM - Alpha = {alpha}')
    plt.show()


elif 1 < alpha < 2:

    theoretical_values = n_values ** (2 * (nu - 2) / (nu - 1))

    x = n_values ** (2 * (nu - 2) / (nu - 1))
    reg_slope = np.sum(x * Average_Consensus_times) / np.sum(x ** 2)

    Consensus_times_VM_avg = {n: [] for n in n_values}
    for n in n_values:
        lst = All_Consensus_times[n]
        Consensus_times_VM_avg[n] = [
            sum(lst[i:i + 10]) / 10 for i in range(0, len(lst), 10)
        ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [All_Consensus_times[n] for n in n_values],
        positions=n_values,
        widths=0.1 * n_values
    )

    plt.xlabel('N')
    plt.ylabel('Average Consensus Time')

    plt.plot(n_values, Average_Consensus_times, 'o')
    plt.plot(
        n_values,
        theoretical_values * reg_slope,
        '-',
        color="green",
        label=r'$N^{\frac{2(\alpha - 2)}{(\alpha - 1)}}$'
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.title(f'CM - Alpha = {alpha}')
    plt.show()


elif alpha > 2:

    theoretical_values = n_values
    x = n_values
    reg_slope = np.sum(x * Average_Consensus_times) / np.sum(x ** 2)

    Consensus_times_VM_avg = {n: [] for n in n_values}
    for n in n_values:
        lst = All_Consensus_times[n]
        Consensus_times_VM_avg[n] = [
            sum(lst[i:i + 10]) / 10 for i in range(0, len(lst), 10)
        ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [All_Consensus_times[n] for n in n_values],
        positions=n_values,
        widths=0.1 * n_values
    )

    plt.xlabel('N')
    plt.ylabel('Average Consensus Time')

    plt.plot(n_values, Average_Consensus_times, 'o')
    plt.plot(
        n_values,
        theoretical_values * reg_slope,
        '-',
        color="green",
        label=r'$N$'
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.title(f'CM - Alpha = {alpha}')
    plt.show()
