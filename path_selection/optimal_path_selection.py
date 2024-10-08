import csv
import pandas as pd
import os
import networkx as nx

# files
centroid_file = '/home/workspace/sequential/graph/centroids.csv'
cluster_file = '/home/workspace/sequential/graph/clusters.csv'
edge_file = '/home/workspace/sequential/graph/graph.csv'
edge_to_target_file = '/home/workspace/sequential/optimization/edge_to_target.csv'
output_file = 'optimal_path.csv'
target_domain_path = "/home/workspace/dataset/z_train/target"

target_task = '16-t2-NCR'


def girvan_newman(G):
    working_graph = G.copy()
    
    # Continue removing edges until more than one component is created
    while nx.number_connected_components(working_graph) == 1:
        edge_betweenness = nx.edge_betweenness_centrality(working_graph)
        max_betweenness = max(edge_betweenness.values())
        
        # Remove all edges with maximum betweenness
        for edge, betweenness in edge_betweenness.items():
            if betweenness == max_betweenness:
                working_graph.remove_edge(*edge)
                
    return list(nx.connected_components(working_graph))

def extended_girvan_newman(G, target_number_of_communities=None):
    target_number_of_communities = target_number_of_communities or G.number_of_nodes()


    communities_list = [list(nx.connected_components(G))]


    while len(communities_list[-1]) < target_number_of_communities and G.number_of_edges() > 0:
        betweenness = nx.edge_betweenness_centrality(G)

        edge_to_remove = max(betweenness, key=betweenness.get)

        G.remove_edge(*edge_to_remove)

        new_communities = list(nx.connected_components(G))
        if len(new_communities) > len(communities_list[-1]):
            communities_list.append(new_communities)

    if target_number_of_communities is not None:
        for communities in reversed(communities_list):
            if len(communities) <= target_number_of_communities:
                return communities 

    return communities_list[-1] 


# Load data
data = pd.read_csv(edge_file) 

# Create a graph from the loaded data
G = nx.Graph()

for index, row in data.iterrows():
    node_a = row['Node A']
    node_b = row['Node B']
    state = row['State']
    # distance = float('inf') if state == "unconnected" else row['Distance']
    
    # if state != "unconnected":  # we don't actually add 'unconnected' edges
    #     G.add_edge(node_a, node_b, weight=distance)
    
    distance = float(row['Distance'])  

    if state.strip() != "unconnected":
        G.add_edge(node_a, node_b, weight=distance)

desired_number_of_communities = 9
resulting_communities = extended_girvan_newman(G, target_number_of_communities=desired_number_of_communities)
community_dict = {}

for idx, community in enumerate(resulting_communities, start=1):
    for node in community:
        community_dict[node] = idx
    print(f"Community {idx}: {community}")

community_df = pd.DataFrame(list(community_dict.items()), columns=['Node', 'Community'])
community_df.to_csv(cluster_file, index=False)
    

print("Communities have been written to cluster_file")


nodes_df = pd.read_csv(cluster_file)
edges_df = pd.read_csv(edge_file)

domain_sizes = {'01': 120, '18': 90, '04': 60, '21': 35, '06': 34, '13': 34, '20': 33}

# Function to extract the domain from a node identifier
def get_domain(node):
    return node.split('-')[0]

# Assign weights to the nodes based on the domain sizes
nodes_df['weight'] = nodes_df['Node'].apply(lambda node: domain_sizes.get(get_domain(node), 1))


total_weighted_distances = {node: 0 for node in nodes_df['Node']}


for _, row in edges_df.iterrows():
    node_a, node_b, distance = row['Node A'], row['Node B'], row['Distance']
    weight_a = nodes_df.loc[nodes_df['Node'] == node_a, 'weight'].iloc[0]
    weight_b = nodes_df.loc[nodes_df['Node'] == node_b, 'weight'].iloc[0]
    weighted_distance = distance / (weight_a + weight_b)
    
    total_weighted_distances[node_a] += weighted_distance
    total_weighted_distances[node_b] += weighted_distance  # since the graph is undirected

centroids = {}
for cluster in pd.unique(nodes_df['Cluster']):
    cluster_nodes = nodes_df[nodes_df['Cluster'] == cluster]['Node'].tolist()
    

    centroid = min(cluster_nodes, key=lambda node: total_weighted_distances[node])
    centroids[cluster] = centroid


results_df = pd.DataFrame(list(centroids.items()), columns=['Cluster', 'Centroid_Node'])
results_df.to_csv(centroid_file, index=False)

print(f"Centroids calculation is complete. Results are saved to {centroid_file}")

top_n_paths = 1
df_centroids = pd.read_csv(centroid_file) 
centroids = pd.Series(df_centroids['Centroid_Node'].values, index=df_centroids['Cluster']).to_dict()

df_clusters = pd.read_csv(cluster_file) 
clusters = pd.Series(df_clusters['Cluster'].values, index=df_clusters['Node']).to_dict()

df_edges = pd.read_csv(edge_file)
edge_distances = {}
for _, row in df_edges.iterrows():
    node_a = row['Node A']
    node_b = row['Node B']
    distance = float(row['Distance'])  
    edge_distances[(node_a, node_b)] = distance
    edge_distances[(node_b, node_a)] = distance


df_edges_to_target = pd.read_csv(edge_to_target_file)
target_edge_distances = {}
for _, row in df_edges_to_target.iterrows():
    node_a = row['Node A']
    node_b = row['Node B']
    distance = float(row['Distance'])  
    target_edge_distances[(node_a, node_b)] = distance
    target_edge_distances[(node_b, node_a)] = distance


# Define the cost function
def cost_function(vk, vi, vt):
    vk_vt_distance = target_edge_distances.get((vk,vt), float('inf'))
    
    vk_vi_distance = edge_distances.get((vk, vi), float('inf'))
    
    vi_vt_distance = target_edge_distances.get((vi,vt), float('inf'))
    
    return vk_vt_distance + vk_vi_distance + vi_vt_distance

def transf(vk, vi, vt):
    
    vk_vi_distance = edge_distances.get((vk, vi), float('inf'))
    
    vi_vt_distance = target_edge_distances.get((vi,vt), float('inf'))
    
    return vk_vi_distance + vi_vt_distance


# Perform optimization
optimization_results = []

for target_task in os.listdir(target_domain_path):
    for cluster_index, vk in centroids.items():
        cluster_nodes = [node for node, cluster in clusters.items() if cluster == cluster_index]
        for vi in cluster_nodes:
            if vk != vi:
                path_cost = cost_function(vk, vi,target_task)
                path_length = transf(vk,vi,target_task)
                result_entry = {
                'initial_source': vk,
                'intermediate_source': vi,
                'path_cost': path_cost,
                'path_length': path_length
                }
                optimization_results.append(result_entry)
    results_df = pd.DataFrame(optimization_results)


    results_df = results_df.sort_values('path_cost')

    # Get the top N paths
    top_paths_df = results_df.head(top_n_paths)

    file_exists = os.path.isfile(output_file) and os.path.getsize(output_file) > 0

    with open(output_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Node A", "Node B", "Target task", "Cost", "Path length"])
        if not file_exists:
                writer.writeheader()
                
        for index, result in top_paths_df.iterrows():
            writer.writerow({
                "Node A": result['initial_source'],
                "Node B": result['intermediate_source'],
                "Target task": target_task,
                "Cost": f"{result['path_cost']:.4f}",  
                "Path length": f"{result['path_length']:.4f}"  
            })
            
    # Print the top paths
    for index, path in top_paths_df.iterrows():
        vk = path['initial_source']
        vi = path['intermediate_source']
        cost = path['path_cost']
        transferability = path['path_length'] 

        print(f"(Node A: {vk}, Node B: {vi}) for target task {target_task} with cost: {cost:.4f}, transfer path length: {transferability:.4f}")

