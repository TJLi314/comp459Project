import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, MSELoss
import torch.nn.functional as F
import os
import random
import pickle
import networkx as nx

SAVE_DIR = "./trained_networks"  # Where to save the trained models
NUM_NETWORKS = 500   


# MLP model designed for 0-9 digit classification
class SmallMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        super(SmallMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def make_NN():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(SAVE_DIR, exist_ok=True)


    # Prepare MNIST Data 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train and Save Multiple Networks 
    for i in range(NUM_NETWORKS):
        # Vary random seed
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        random.seed(seed)

        # Randomly vary hyperparameters if needed
        hidden_size = random.choice([32, 64, 128])

        # Create model
        model = SmallMLP(hidden_size=hidden_size).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train for a few epochs
        model.train()
        for epoch in range(5):  # Quick training
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Save the model weights + architecture info
        save_data = {
            "seed": seed,
            "hidden_size": hidden_size,
            "state_dict": model.state_dict()
        }
        with open(os.path.join(SAVE_DIR, f"mlp_{i}.pkl"), "wb") as f:
            pickle.dump(save_data, f)

        print(f"Saved model {i} with hidden size {hidden_size} and seed {seed}.")

    print("Done generating dataset!")

def mlp_to_graph(mlp_path):
    with open(mlp_path, "rb") as f:
        saved = pickle.load(f)
    
    state_dict = saved['state_dict']
    hidden_size = saved['hidden_size']

    # Initialize directed graph
    G = nx.DiGraph()

    # Layer sizes
    input_size = 784
    output_size = 10

    # Nodes: Add input, hidden, and output bias nodes
    for i in range(input_size):
        G.add_node(f"in_{i}")

    for i in range(hidden_size):
        G.add_node(f"hidden_{i}", bias=state_dict['fc1.bias'][i].item())

    for i in range(output_size):
        G.add_node(f"out_{i}", bias=state_dict['fc2.bias'][i].item())

    # Edges: input → hidden
    fc1_weights = state_dict['fc1.weight']  # Shape: [hidden, input]
    for i in range(hidden_size):
        for j in range(input_size):
            weight = fc1_weights[i, j].item()
            G.add_edge(f"in_{j}", f"hidden_{i}", weight=weight)

    # Edges: hidden → output
    fc2_weights = state_dict['fc2.weight']  # Shape: [output, hidden]
    for i in range(output_size):
        for j in range(hidden_size):
            weight = fc2_weights[i, j].item()
            G.add_edge(f"hidden_{j}", f"out_{i}", weight=weight)

    return G

def convert_networkx_to_pyg_data(G, mask_ratio=0.1):
    # Map node type strings to one-hot vectors
    type_to_vec = {"input": [1, 0, 0], "hidden": [0, 1, 0], "output": [0, 0, 1]}
    node_feats = []
    node_id_map = {nid: i for i, nid in enumerate(G.nodes())}

    for nid in G.nodes:
        bias = G.nodes[nid].get("bias", 0.0)
        ntype = G.nodes[nid].get("type", "hidden")
        type_vec = type_to_vec.get(ntype, [0, 1, 0])
        node_feats.append([bias] + type_vec)

    edge_index = []
    edge_attr = []
    target_edges = []
    target_weights = []

    for u, v, data in G.edges(data=True):
        i, j = node_id_map[u], node_id_map[v]
        edge_index.append([i, j])
        weight = data.get("weight", 0.0)
        
        if random.random() < mask_ratio:
            # Masked edge: we will try to predict this
            edge_attr.append([0.0])  # Set dummy or zeroed edge feature
            target_edges.append(len(edge_index) - 1)
            target_weights.append(weight)
        else:
            edge_attr.append([weight])  # Normal edge

    return Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        target_edge_index=torch.tensor(target_edges, dtype=torch.long),
        target_weights=torch.tensor(target_weights, dtype=torch.float)
    )

class EdgeGNN(MessagePassing):
    def __init__(self, node_in_dim=4, edge_in_dim=1, hidden_dim=32):
        super().__init__(aggr='add')  # Aggregate messages via sum
        self.node_mlp = Linear(node_in_dim, hidden_dim)
        self.edge_mlp = Linear(edge_in_dim, hidden_dim)
        self.message_mlp = Linear(2 * hidden_dim, hidden_dim)
        self.update_mlp = Linear(hidden_dim, hidden_dim)

        # Head for predicting edge weights
        self.edge_pred_mlp = torch.nn.Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, target_edge_index):
        x = self.node_mlp(x)
        edge_attr = self.edge_mlp(edge_attr)
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        self.cached_node_features = x  # Save for edge prediction
        return self.predict_edge_weights(x, edge_index, target_edge_index)

    def message(self, x_i, x_j, edge_attr):
        # Combine source, target, and edge info
        msg = torch.cat([x_i, edge_attr], dim=-1)
        return F.relu(self.message_mlp(msg))

    def update(self, aggr_out):
        return self.update_mlp(aggr_out)

    def predict_edge_weights(self, x, edge_index, target_edge_index):
        # Predict weight for each masked edge
        src = edge_index[0, target_edge_index]
        dst = edge_index[1, target_edge_index]
        h_src = x[src]
        h_dst = x[dst]
        return self.edge_pred_mlp(torch.cat([h_src, h_dst], dim=-1)).squeeze()
    


def train(model, data_list, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(data_list)
        for data in data_list:
            model.train()
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index, data.edge_attr, data.target_edge_index)
            loss = loss_fn(pred, data.target_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")


# Predicts the missing edges in networkx graphs, add them to said graph, then
# returns a list of graphs with complete networks

def complete_graphs(model, test_data):
    model.eval()
    completed_graphs = []
    
    with torch.no_grad():
        for data in test_data:
            pred_weights = model(data.x, data.edge_index, data.edge_attr, data.target_edge_index)
            
            completed_edge_index = data.edge_index.clone()
            completed_edge_attr = data.edge_attr.clone()

            for idx, edge_idx in enumerate(data.target_edge_index):
                src, dst = data.edge_index[:, edge_idx].tolist()
                weight = pred_weights[idx].item()
                
                completed_edge_index = torch.cat([completed_edge_index, torch.tensor([[src], [dst]])], dim=-1)
                completed_edge_attr = torch.cat([completed_edge_attr, torch.tensor([[weight]])], dim=0)

            completed_graphs.append(Data(
                x=data.x,
                edge_index=completed_edge_index,
                edge_attr=completed_edge_attr
            ))

    return completed_graphs

# Turns pyg graph representation back into SmallMLP representation

def convert_pyg_to_mlp(graph, input_size=784, output_size=10):
    node_features = graph.x
    edge_index = graph.edge_index
    edge_weights = graph.edge_attr

    input_nodes = [i for i, feat in enumerate(node_features) if feat[1:4].tolist() == [1, 0, 0]]
    hidden_nodes = [i for i, feat in enumerate(node_features) if feat[1:4].tolist() == [0, 1, 0]]
    output_nodes = [i for i, feat in enumerate(node_features) if feat[1:4].tolist() == [0, 0, 1]]

    adjacency = {}
    for src, dst in edge_index.t().tolist():
        weight = edge_weights[src][0].item()  # Get the weight for this edge
        if dst not in adjacency:
            adjacency[dst] = {}
        adjacency[dst][src] = weight

    mlp = SmallMLP(input_size=input_size, hidden_size=len(hidden_nodes), output_size=output_size)

    with torch.no_grad():
        for i, hidden_idx in enumerate(hidden_nodes):
            for j, input_idx in enumerate(input_nodes):
                mlp.fc1.weight.data[i, j] = adjacency[hidden_idx].get(input_idx, 0.0)
        for i, hidden_idx in enumerate(hidden_nodes):
            mlp.fc1.bias.data[i] = node_features[hidden_idx][0]  # bias is first feature
        
        # Set weights for fc2 (hidden to output)
        for i, output_idx in enumerate(output_nodes):
            for j, hidden_idx in enumerate(hidden_nodes):
                mlp.fc2.weight.data[i, j] = adjacency[output_idx].get(hidden_idx, 0.0)
        
        # Set biases for fc2
        for i, output_idx in enumerate(output_nodes):
            mlp.fc2.bias.data[i] = node_features[output_idx][0]  # bias is first feature
    
    return mlp


if __name__ == "__main__":
    # Convert saved NNs to pyg data
    graph_data = []
    for i in range(NUM_NETWORKS):
        print("Converting neural net " + str(i))
        path = os.path.join(SAVE_DIR, f"mlp_{i}.pkl")
        G = mlp_to_graph(path)
        pyg_data = convert_networkx_to_pyg_data(G, mask_ratio=0.1)
        graph_data.append(pyg_data)
    
    # Graph data train test split

    graph_data.shuffle(graph_data)

    split_index = int(0.8 * len(graph_data))
    train_graph = graph_data[:split_index]
    test_graph = graph_data[split_index:]

    # Train GNN
    model = EdgeGNN()
    print("Training GNN")
    train(model, train_graph)
    print("Done training")

    # Predicting for the test graphs, then 
    pred_graphs = complete_graphs(model, test_graph)
    completed_mlp = [convert_pyg_to_mlp(graph) for graph in pred_graphs]
