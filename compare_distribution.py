
# -*- coding: utf-8 -*-

import networkx as nx
import json
import matplotlib.pyplot as plt
import network_sir as sir

tmax = 200
simu_num = 500

cir_size = 30

with open("email.txt", 'rb') as fh:
    G = nx.read_weighted_edgelist(fh)
    
G.remove_edges_from(nx.selfloop_edges(G))
netsize = len(G.nodes())

def centrality_sort(g, rank):
    def sort_centrality(cent_dict):
        return [(n, d) for d, n in sorted(((v, k) for k, v in cent_dict.items()), reverse=rank)]

    return (
        sort_centrality(nx.betweenness_centrality(g)),
        [(n, d) for d, n in sorted(((d, n) for n, d in g.degree()), reverse=rank)],
        sort_centrality(nx.eigenvector_centrality_numpy(g)),
        sort_centrality(nx.core_number(g))
    )

def centra_nodes(num, g):
    centrality = centrality_sort(g, True)
    max_nodeset = []
    min_nodeset = []
    
    for n in centrality:
        max_nodeset.append([node for node, _ in n[:num]])
        min_nodeset.append([node for node, _ in reversed(n[-num:])])
    
    return {'max': max_nodeset, 'min': min_nodeset}

def run_overtake(seeds_ratio):
    nodeset_num = round(seeds_ratio * netsize)
    seed_nodes = centra_nodes(nodeset_num, G)
    dc_nodes = [seed_nodes['max'][1], seed_nodes['min'][1]]
    
    for nodes in dc_nodes:
        results = []

        for _ in range(simu_num):
            model = sir.NetworkSIR(G, beta=0.25, gamma=1, seeds=nodes)
            _, _, _, R = model.simulate(max_time=tmax)
            #print((int(R[-1])))
            results.append(int(R[-1]))   
            
        count_result = {}
        for r in results:
            count_result[r] = results.count(r) / len(results)
            
        scale = [s/netsize for s in count_result.keys()]
        pro = list(count_result.values())

        ax = plt.subplot(1, 1, 1)
        color = 'royalblue' if nodes == dc_nodes[0] else 'red'
        label = 'Max_deg' if nodes == dc_nodes[0]  else 'Min_deg' 
        plt.scatter(scale, pro, s=cir_size, facecolors='none', edgecolors=color, label=label)
        
        
        plt.axis([-0.05, 1, 0.001, 2])
        plt.yscale("log")
        plt.xlabel('R', fontsize=14, fontdict={'fontstyle': 'italic'})
        plt.ylabel('P', fontsize=14, fontdict={'fontstyle': 'italic'})
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.tick_params(length=5, width=1, grid_alpha=0.2)
    plt.legend(loc='best', ncol=2, fontsize=10, frameon=False)
    
run_overtake(seeds_ratio=0.1)
plt.show()

