# visualizer/network_visualizer.py
import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(layers, connections):
    """
    Visualizes the neural network structure.
    
    Args:
      layers: A list of tuples (layer_name, number_of_neurons)
      connections: A list of tuples (source_layer, target_layer, connection_parameters)
    """
    G = nx.DiGraph()
    
    # Add nodes for each neuron in each layer
    for layer_name, count in layers:
        for i in range(count):
            node_name = f"{layer_name}_{i}"
            G.add_node(node_name, layer=layer_name)
    
    # Add edges connecting neurons from source to target layers
    for src_layer, dst_layer, params in connections:
        src_nodes = [n for n, attr in G.nodes(data=True) if attr['layer'] == src_layer]
        dst_nodes = [n for n, attr in G.nodes(data=True) if attr['layer'] == dst_layer]
        for src in src_nodes:
            for dst in dst_nodes:
                G.add_edge(src, dst, **params)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=50, node_color="skyblue", arrowsize=5)
    plt.show()

if __name__ == "__main__":
    # Example usage of the network visualizer
    layers = [("Input", 10), ("Hidden", 5), ("Output", 2)]
    connections = [
        ("Input", "Hidden", {"weight": "xavier"}),
        ("Hidden", "Output", {"weight": "xavier"})
    ]
    visualize_network(layers, connections)
