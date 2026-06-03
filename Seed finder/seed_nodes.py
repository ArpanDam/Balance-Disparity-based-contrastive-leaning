# import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import networkx as nx
import argparse

def select_initial_seed_set(node_embeddings, second_order_neighbors, k, beta):
    # Initialize the seed set and variables to track influence
    seed_set = set()
    no_embeddings = set()
    node_influence = defaultdict(set)  # Tracks which nodes each node influences

    # Filter second-order neighbors by similarity > beta and track influence
    for node, neighbors in second_order_neighbors.items():
        if node not in node_embeddings:
            no_embeddings.add(node)
            continue

        filtered_neighbors = set()
        for neighbor in neighbors:
            if neighbor not in node_embeddings:
                no_embeddings.add(neighbor)
                continue

            similarity = cosine_similarity(node_embeddings[node].reshape(1, -1), node_embeddings[neighbor].reshape(1, -1))[0][0]
            if similarity > beta:
                filtered_neighbors.add(neighbor)

        node_influence[node] = filtered_neighbors

    # Iteratively select the top-k influential nodes
    influenced_nodes = set()
    while len(seed_set) < k:
        # Select the node with the maximum number of unique influences
        best_node = None
        max_influence = 0

        for node, influences in node_influence.items():
            # Exclude already influenced nodes
            unique_influences = influences - influenced_nodes
            if len(unique_influences) > max_influence:
                best_node = node
                max_influence = len(unique_influences)

        if best_node is None:  # In case no more nodes can be selected
            break

        # Add the best node to the seed set
        seed_set.add(best_node)
        # Update the set of influenced nodes
        influenced_nodes.update(node_influence[best_node])

        # Remove the influenced nodes from all other nodes' influence lists
        for node in node_influence:
            node_influence[node] -= influenced_nodes

    # print(f"Nodes with missing embeddings: {no_embeddings}")
    return list(seed_set)


def get_second_order_neighbors(edge_index):
    # Create a directed graph using NetworkX
    G = nx.DiGraph()
    G.add_edges_from(edge_index)
    
    # Initialize a dictionary to store second-order neighbors
    second_order_neighbors = defaultdict(set)
    
    # For each node, find second-order neighbors
    for node in G.nodes():
        first_order = set(G.successors(node))  # First-order neighbors
        second_order = set()
        
        for neighbor in first_order:
            second_order.update(G.successors(neighbor))  # Second-order neighbors
            
        # Remove the original node from its own second-order neighbors, if present
        second_order.discard(node)
        
        second_order_neighbors[node] = second_order
    
    return second_order_neighbors


if __name__ == "__main__":
    # Set up argument parser to accept 'k' and 'beta' as command-line arguments
    parser = argparse.ArgumentParser(description="Select top-k influential authors.")
    parser.add_argument("k", type=int, help="Number of seed nodes to select.")
    parser.add_argument("beta", type=float, help="Similarity threshold for neighbor selection.")

    args = parser.parse_args()

    # Load embeddings and edge index
    with open('dict_node_embedding', 'rb') as file:  # here node embeddings
        features = pickle.load(file)

    with open('original_edge_index.pkl', 'rb') as file:  # here edge index
        edge_index = pickle.load(file)

    # Get second-order neighbors
    second_order_neighbors = get_second_order_neighbors(edge_index)

    # Select seed nodes using the values from command line arguments
    seed_nodes = select_initial_seed_set(node_embeddings=features, second_order_neighbors=second_order_neighbors, k=args.k, beta=args.beta)

    print("Selected influential authors:", seed_nodes)