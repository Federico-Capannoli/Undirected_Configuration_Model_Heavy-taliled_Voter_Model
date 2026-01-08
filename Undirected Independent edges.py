"""
Voter Model on Erdős–Rényi Random Graphs
Consensus Time Scaling

This script simulates the voter model on Erdős–Rényi graphs G(n, p = l/n).
Consensus times are measured for increasing system sizes and compared with
theoretical scaling predictions depending on the connectivity regime.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy as sp
import pandas as pd


###################################################################################################
# Parameters
###################################################################################################

alpha = 1.5
k_min = 5

U = 0.5  # Initial proportion of red opinions

# System size range
n_min = 100
n_max = 1000
number_of_iterations_n = 10

# Average degree parameter (p = l / n)
l = 1

# Iterations
iteration_VM = 10
iteration_Graph = 10
iteration_Degree_sequence = 1

n_values = np.linspace(n_min, n_max, number_of_iterations_n).astype(int)


###################################################################################################
# Storage structures
###################################################################################################

All_Consensus_times = {n: [] for n in n_values}  # Store all consensus times for each n
Consensus_times_degree_sequences = {n: [] for n in n_values}
Average_Consensus_times = []


###################################################################################################
# Main simulation loop
###################################################################################################

for n in n_values:

    Consensus_times_Degree_Sequence_average = []

    for Iter_deg_seq in range(iteration_Degree_sequence):

        Consensus_times_Graph_average = []

        for Iter_graph in range(iteration_Graph):

            # Generate Erdős–Rényi graph
            G = nx.erdos_renyi_graph(n, l / n)

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

                    if len(list(G.neighbors(u))) > 0:
                        v = random.choice(list(G.neighbors(u)))

                        if node_colors[u] != node_colors[v]:
                            node_colors[u] = node_colors[v]

                            # Update discordant edges dynamically
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

                    # Check for consensus
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
# Theoretical scaling and visualization
###################################################################################################

if l == 1:
    x = n_values ** (2 / 3)
elif l > 1:
    x = n_values
else:
    x = np.log(np.array(n_values))

theoretical_values = x

# Log–log regression (not plotted by default)
log_n_values = np.log10(n_values)
log_Average_Consensus_times = np.log10(Average_Consensus_times)

weights = np.linspace(1, 100, len(log_n_values))
W = np.diag(weights)

X = np.vstack([log_n_values, np.ones(len(log_n_values))]).T
y = log_Average_Consensus_times

beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
regression_slope, regression_intercept = beta

# Linear regression coefficient for theoretical scaling
reg_slope = np.sum(x * Average_Consensus_time*_
