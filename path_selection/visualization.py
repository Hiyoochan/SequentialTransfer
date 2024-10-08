import matplotlib.pyplot as plt
import networkx as nx
import csv
import matplotlib.font_manager as fm
import pandas as pd
import random


# Load your graph from CSV
df = pd.read_csv('/home/workspace/sequential/graph/results/graph.csv')  # replace with your actual filename

# Create a new graph from the DataFrame
G = nx.Graph()
        
domain_sizes = {'01': 80, '18': 65, '04': 45, '21': 40, '06': 34, '13': 34, '20': 33}
cluster_colors = {'0': 'red', '1': 'orange', '2': 'blue', '3': 'green', '4': 'xkcd:sky blue', '5': '#C79FEF', '6': '#FF81C0', '7': 'yellowgreen','8':'brown'}

# Read the CSV file of cluster labels
cluster_file = '/home/workspace/sequential/graph/result/clusters.csv'
node_labels = {}
with open(cluster_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        node = row[1]
        label = row[0]
        node_labels[node] = label

# Read the CSV file of centroids
centroid_file = '/home/workspace/sequential/graph/result/centroids.csv'
centroids = {}
with open(centroid_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        cluster_label = row[0]
        centroid_node = row[1]
        centroids[cluster_label] = centroid_node

for index, row in df.iterrows():
    node_a = row['Node A']
    node_b = row['Node B']
    state = row['State']
    distance = row['Distance']
    
    # Extract the domain from the node name
    domain1 = node_a.strip().split('-')[0]
    domain2 = node_b.strip().split('-')[0]

    cluster1 = node_labels.get(node_a)
    cluster2 = node_labels.get(node_b)

    if state != "unconnected" and distance <= 1:
        # Add nodes to the graph with their sizes based on the domain
        G.add_node(node_a, size=domain_sizes.get(domain1, 0), color=cluster_colors.get(cluster1, 'black'))
        G.add_node(node_b, size=domain_sizes.get(domain2, 0), color=cluster_colors.get(cluster2, 'black'))
        G.add_edge(node_a, node_b, weight=float(distance))
        


# Extract node sizes from the graph
sizes = [data['size'] for _, data in G.nodes(data=True)]
# Extract node colors from the graph
colors = [data['color'] for _, data in G.nodes(data=True)]
# Extract edge weights from the graph
edge_weights = nx.get_edge_attributes(G, 'weight')

# Draw the graph
pos = nx.spring_layout(G)  # Choose a layout algorithm

plt.figure(figsize=(40, 50))



nx.draw_networkx_edges(G, pos, width=list(edge_weights.values()), alpha=0.7)


nx.draw_networkx_labels(G, pos, font_size=20 )

# Draw node sizes as circles
ax = plt.gca()
ax.set_aspect('equal')

for node, (x, y) in pos.items():
    node_size = G.nodes[node]['size']
    node_color = G.nodes[node]['color']
    if node in centroids.values():
        circle = plt.Circle((x, y), node_size / 2800,facecolor=node_color, alpha=0.8, edgecolor='black', linewidth=5)
    else:
        circle = plt.Circle((x, y), node_size / 2800, color=node_color, alpha=0.8)
    ax.add_patch(circle)


plt.axis('off')
plt.tight_layout()

# Display the graph
plt.show()
plt.savefig('graph_cluster.png')
#plt.savefig('graph.svg', format='svg')
print('Draw the graph done!')