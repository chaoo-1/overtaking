# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import network_sir as sir


# Open the file containing the data and read it into a graph
with open("email.txt", 'rb') as fh:
    G = nx.read_weighted_edgelist(fh)

G.remove_edges_from(nx.selfloop_edges(G))
# Get the size of the network (number of nodes)
netsize = len(G.nodes())


# Parameter settings
tmax = 50
gamma = 1  # Recovery rate
tau = 0.15  # Infection rate

# Print network characteristics
def print_network_stats():
    assort = nx.degree_assortativity_coefficient(G)
    print(f"Network size: {netsize}, Assortativity: {assort:.2f}")

def centrality_sort(g, rank):
    """Calculate and return the sorted centrality measures for nodes"""
    def sort_centrality(cent_dict):
        sorted_items = sorted(((k, v) for v, k in cent_dict.items()), reverse=rank)
        return [(n, d) for d, n in sorted_items]

    betweeness = sort_centrality(nx.betweenness_centrality(g))
    eigenvector = sort_centrality(nx.eigenvector_centrality_numpy(g))
    degree = [(n, d) for d, n in sorted(((d, n) for n, d in g.degree()), reverse=rank)]
    kcore = sort_centrality(nx.core_number(g))
    
    return betweeness, degree, eigenvector, kcore 

centrality = centrality_sort(g=G, rank=True)


# Function to create subplots for results
def plot_subplot(ax, results, plotnum, i, metrics=['BC', 'DC', 'EC', 'KC']):
    colors = ['royalblue', 'r']
    labels = [f'Max_{metrics[plotnum-1]}', f'Min_{metrics[plotnum-1]}'] if i == 0 else [None, None]
    
    for idx, (t, _, _, R) in enumerate(results):
        plt.plot(t, R, colors[idx], label=labels[idx])
    
    plt.axis([0, 22, 0, 0.79])
    plt.legend(loc=4, fontsize=12, frameon=False)
    
    # Set axis labels
    if plotnum in [1, 3]:
        plt.ylabel('R(t)', fontsize=14, style='italic')
    if plotnum > 2:
        plt.xlabel('t', fontsize=14, style='italic')
    if plotnum in [2, 4]:
        plt.yticks([])
    if plotnum < 3:
        plt.xticks([])
        
# Function to simulate SIR model
def run_sir_simulation(seeds):
    model = sir.NetworkSIR(G, beta=tau, gamma=gamma, seeds=seeds)
    t, S, I, R = model.simulate(max_time=tmax)

    return t, S, I, np.array(R) / netsize

# Main function to run SIR simulations and plot results
def run_overtake_sir(num):
    plt.figure(figsize=(8, 5))   
    num=int(num)
    for plotnum, n in enumerate(centrality, 1):
        # Select nodes
        max_node = [n for n, v in n[:num]]
        min_node = [n for n, v in n[-num:]]
        seed_nodes = [max_node, min_node]
        
        # Run multiple simulations
        for i in range(10):
            results = [run_sir_simulation(seeds) for seeds in seed_nodes]
            ax = plt.subplot(2, 2, plotnum)
            plot_subplot(ax, results, plotnum, i)
    
    plt.subplots_adjust(wspace=0, hspace=0)


#print_network_stats()

# Run the main program
seed_ratio = 0.1  # Proportion of seed nodes in the network
run_overtake_sir(num=seed_ratio*netsize) 
plt.show()

