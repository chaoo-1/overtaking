
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class NetworkSIR:
    def __init__(self, network, beta, gamma, seeds):
        self.G = network
        self.beta = beta
        self.gamma = gamma
        self.ini_seeds = seeds
        
        # Initialize all nodes as susceptible
        self.status = {n: 'S' for n in self.G.nodes()}
        self.infected = set()
        self.recovered = set()
        
        # Susceptible neighbors of infected nodes
        self.infectious_edges = dict()
        
        # Initialize infected nodes
        # patient_zero = np.random.choice(list(self.G.nodes()))
        for seed in self.ini_seeds:           
            self._infect_node(seed)

        # Record time series
        self.t = [0.0]
        self.s = [len(self.G) - 1]
        self.i = [1]
        self.r = [0]

    def _infect_node(self, node):
        """Mark the node as infected and update data structures"""
        self.status[node] = 'I'
        self.infected.add(node)
        neighbors = list(self.G.neighbors(node))
        
        # Maintain the list of susceptible neighbors
        self.infectious_edges[node] = [
            n for n in neighbors if self.status[n] == 'S']
        
        # Update the neighbor list for other infected nodes
        for n in neighbors:
            if self.status[n] == 'I' and n != node:
                try:
                    self.infectious_edges[n].remove(node)
                except ValueError:
                    pass

    def _recover_node(self, node):
        """Mark the node as recovered and update data structures"""
        self.status[node] = 'R'
        self.infected.remove(node)
        self.recovered.add(node)
        del self.infectious_edges[node]
        
        # Update the infection opportunities for neighboring nodes
        for n in self.G.neighbors(node):
            if self.status[n] == 'I':
                try:
                    self.infectious_edges[n].remove(node)
                except ValueError:
                    pass

    def _get_total_rate(self):
        """Calculate the total event rate"""
        infection_rate = self.beta * sum(
            len(v) for v in self.infectious_edges.values()
        )
        recovery_rate = self.gamma * len(self.infected)
        return infection_rate + recovery_rate

    def simulate(self, max_time=50):
        """Execute the Gillespie algorithm"""
        current_time = 0.0
        
        while current_time < max_time and self.infected:
            total_rate = self._get_total_rate()
            if total_rate == 0:
                break
                
            # Calculate the next event time
            dt = np.random.exponential(1 / total_rate)
            current_time += dt
            
            # Choose event type
            infection_rate = self.beta * sum(
                len(v) for v in self.infectious_edges.values()
            )
            rand = np.random.random() * total_rate
            
            if rand < infection_rate:  # Infection event
                # Select infector based on weights
                weights = [len(v) for v in self.infectious_edges.values()]
                infector = np.random.choice(
                    list(self.infectious_edges.keys()), 
                    p=np.array(weights) / sum(weights)
                )
                
                # Choose the infected node
                susceptible = self.infectious_edges[infector]
                if susceptible:
                    new_infected = np.random.choice(susceptible)
                    self._infect_node(new_infected)
                
            else:  # Recovery event
                recovered = np.random.choice(list(self.infectious_edges.keys()))
                self._recover_node(recovered)
            
            # Record state
            self.t.append(current_time)
            self.s.append(sum(1 for s in self.status.values() if s == 'S'))
            self.i.append(len(self.infected))
            self.r.append(len(self.recovered))
            
        return self.t, self.s, self.i, self.r

