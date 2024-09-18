import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def initialize_nodes(n, xm, ym, zm, Eo):
    nodes = np.array([{
        'id': i,
        'x': random.uniform(0, xm),
        'y': random.uniform(0, ym),
        'z': random.uniform(0, zm),
        'energy': Eo,
        'type': 'N', 
        'neighbors': [],
        'link_quality': random.uniform(0, 1)
    } for i in range(n)])
    return nodes

def broadcast_hello(node):
    return {
        'type': 'HELLO',
        'id': node['id'],
        'x': node['x'],
        'y': node['y'],
        'z': node['z'],
        'energy': node['energy']
    }

def send_ack(receiver, sender):
    distance = np.sqrt(sum((receiver[coord] - sender[coord])**2 for coord in 'xyz'))
    return {
        'type': 'ACK',
        'id': receiver['id'],
        'energy': receiver['energy'],
        'distance': distance,
        'link_quality': random.uniform(0.5, 1)  # Simulated link quality
    }

def initialize_network(nodes):
    for node in nodes:
        hello = broadcast_hello(node)
        for other in nodes:
            if other['id'] != node['id']:
                if np.sqrt(sum((node[coord] - other[coord])**2 for coord in 'xyz')) <= 10:  # Communication range
                    ack = send_ack(other, node)
                    node['neighbors'].append({
                        'id': ack['id'],
                        'energy': ack['energy'],
                        'distance': ack['distance'],
                        'link_quality': ack['link_quality']
                    })
    return nodes

def calculate_holding_time(q_value, max_holding_time=0.1):
    return (1 - q_value) * max_holding_time

def select_next_hop(current_node, nodes, Q):
    neighbors = current_node['neighbors']
    if not neighbors:
        return None
    
    for neighbor in neighbors:
        neighbor['holding_time'] = calculate_holding_time(Q[current_node['id']][neighbor['id']])
    
    neighbors.sort(key=lambda x: x['holding_time'])
    return neighbors[0]['id']

def update_q_value(Q, current_node, next_node, reward, alpha, gamma):
    Q[current_node][next_node] = (1 - alpha) * Q[current_node][next_node] + \
                                 alpha * (reward + gamma * np.max(Q[next_node]))
    return Q

def calculate_reward(current_node, next_node, nodes, packet_size, vsound):
    if next_node is None:
        return -10  # Penalty for no available next hop
    
    if nodes[next_node]['energy'] < 0.1 * nodes[next_node]['initial_energy']:
        return -5  # Penalty for selecting a low-energy node
    
    distance = np.sqrt(sum((nodes[current_node][coord] - nodes[next_node][coord])**2 for coord in 'xyz'))
    delay = packet_size / (1000 * 8) + distance / vsound
    energy_efficiency = 1 / (nodes[current_node]['energy'] - nodes[next_node]['energy'])
    
    return energy_efficiency - 0.1 * delay

def run_qlro_simulation(n, max_rounds, alpha, gamma):
    xm, ym, zm = 100, 100, 100
    Eo = 5  # Initial energy
    vsound = 1500  # Speed of sound in water (m/s)
    packet_size = 2000  # bits
    
    nodes = initialize_nodes(n, xm, ym, zm, Eo)
    nodes = initialize_network(nodes)
    
    Q = np.zeros((n, n))
    
    total_delay = 0
    packets_sent = 0
    packets_received = 0
    total_energy_consumption = 0
    network_lifetime = 0
    
    for round in range(max_rounds):
        for node in nodes:
            if node['energy'] <= 0.1 * Eo:
                node['neighbors'] = []  # Remove low-energy nodes from neighbor lists
        
        for i in range(n-1):  # Exclude the sink node
            if nodes[i]['energy'] > 0:
                current_node = i
                path = [current_node]
                
                while current_node != n-1:  # Until we reach the sink node
                    next_node = select_next_hop(nodes[current_node], nodes, Q)
                    if next_node is None:
                        break
                    
                    reward = calculate_reward(current_node, next_node, nodes, packet_size, vsound)
                    Q = update_q_value(Q, current_node, next_node, reward, alpha, gamma)
                    
                    # Simulate packet transmission
                    distance = np.sqrt(sum((nodes[current_node][coord] - nodes[next_node][coord])**2 for coord in 'xyz'))
                    delay = packet_size / (1000 * 8) + distance / vsound
                    energy_consumed = 0.00001 * packet_size * distance**2
                    
                    nodes[current_node]['energy'] -= energy_consumed
                    total_energy_consumption += energy_consumed
                    total_delay += delay
                    
                    current_node = next_node
                    path.append(current_node)
                
                packets_sent += 1
                if path[-1] == n-1:
                    packets_received += 1
        
        # Check network lifetime
        if sum(node['energy'] > 0 for node in nodes) > 0:
            network_lifetime = round + 1
        
        # Check if all nodes are dead
        if sum(node['energy'] <= 0 for node in nodes) == n:
            break
    
    avg_delay = total_delay / packets_received if packets_received > 0 else 0
    packet_delivery_ratio = packets_received / packets_sent if packets_sent > 0 else 0
    
    return avg_delay, packet_delivery_ratio, network_lifetime, total_energy_consumption

def main():
    node_ranges = [5, 10, 15, 20]
    alpha, gamma = 0.5, 0.9
    max_rounds = 100
    
    results = []
    
    for n in node_ranges:
        avg_delay, pdr, lifetime, energy = run_qlro_simulation(n, max_rounds, alpha, gamma)
        results.append((n, avg_delay, pdr, lifetime, energy))
    
    # Plot results
    plot_results(results)

def plot_results(results):
    nodes, delays, pdrs, lifetimes, energies = zip(*results)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(nodes, delays, marker='o')
    plt.title('Average End-to-End Delay')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Delay (s)')
    
    plt.subplot(2, 2, 2)
    plt.plot(nodes, pdrs, marker='o')
    plt.title('Packet Delivery Ratio')
    plt.xlabel('Number of Nodes')
    plt.ylabel('PDR')
    
    plt.subplot(2, 2, 3)
    plt.plot(nodes, lifetimes, marker='o')
    plt.title('Network Lifetime')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Rounds')
    
    plt.subplot(2, 2, 4)
    plt.plot(nodes, energies, marker='o')
    plt.title('Total Energy Consumption')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Energy (J)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()