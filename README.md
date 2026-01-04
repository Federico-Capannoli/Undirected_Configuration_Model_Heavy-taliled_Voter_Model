# Voter Model on Heavy-Tailed Undirected Networks

This repository contains Python simulations of **opinion dynamics and consensus formation** on **complex undirected networks** with **heavy-tailed degree distributions**.  
The project explores how network topology (in particular, heavy-tailed in- and out-degree distributions) affects consensus time and opinion evolution.

The code is intended for **academic / educational use**, especially for students or researchers interested in:
- Complex networks
- Voter models and consensus dynamics
- Configuration models
- Heavy-tailed (Pareto-like) degree distributions

---

## Project Structure

The repository currently includes the following scripts:

- **`Undirected Wright-Fisher diffusion.py`**  
  Studies the diffusion approximation of opinion dynamics on directed networks. The script investigates the evolution of opinion density and its relationship with theoretical predictions derived from Wright–Fisher–type diffusion limits in the presence of degree heterogeneity.

---

## References
All the projects in this repository are based and were used to develope cutting-edge research in the field of applied mathematics, physics, complex systems and information theory. The main references are

[Voter model on heterogeneous directed networks](https://arxiv.org/abs/2506.12169) by Luca Avena, Federico Capannoli, Diego Garlaschelli and Rajat Subhra Hazra.

[Meeting, coalescence and consensus time on random directed graphs](https://projecteuclid.org/journals/annals-of-applied-probability/volume-34/issue-5/Meeting-coalescence-and-consensus-time-on-random-directed-graphs/10.1214/24-AAP2087.short) by Luca Avena, Federico Capannoli, Rajat Subhra Hazra and Matteo Quattropani.

[Evolution of discordant edges in the voter model on random sparse digraphs](https://projecteuclid.org/journals/electronic-journal-of-probability/volume-30/issue-none/Evolution-of-discordant-edges-in-the-voter-model-on-random/10.1214/24-EJP1265.pdf) by Federico Capannoli.


---

## Model Overview

The simulations are based on variants of the **voter model**, where:
- Each node represents an agent with a binary opinion.
- The network is **undirected**, meaning influence is symmetric.
- Degrees are drawn from **heavy-tailed (Pareto) distributions**, allowing for hubs and strong heterogeneity.
- Opinion updates follow stochastic rules based on neighbors’ states.

---

## Requirements

The code is written in **Python 3** and uses standard scientific libraries:

- `numpy`
- `scipy`
- `networkx`
- `matplotlib`

You can install the dependencies with:

```bash
pip install numpy scipy networkx matplotlib

