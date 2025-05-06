import pickle
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch.utils.data import Subset
from torch.utils.data import random_split
import matplotlib.pyplot as plt

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

class MLPForwardLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
        self.activation = activation

    def forward(self, x, edge_index, edge_weight, bias_tensor):
        # propagate handles message passing
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        out += bias_tensor
        out = self.linear(out)

        # Optional activation
        if self.activation:
            out = self.activation(out)

        return out

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)

class GNNMLPSimulator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_to_hidden = MLPForwardLayer(1, 1, activation=F.relu)
        self.hidden_to_output = MLPForwardLayer(1, 1, activation=None)

    def forward(self, x_input, edge_index, edge_attr, node_types, bias_vec):
        # Split node features
        input_mask = node_types == 0
        hidden_mask = node_types == 1
        output_mask = node_types == 2

        bias_tensor = bias_vec.unsqueeze(1)

        # Initial activations
        x = torch.zeros_like(bias_tensor)
        input_mask = node_types == 0
        x[input_mask] = x_input[:input_mask.sum()]
        # x[input_mask] = x_input

        # Step 1: Input → Hidden
        edge_mask_1 = (node_types[edge_index[0]] == 0) & (node_types[edge_index[1]] == 1)
        edge_index_1 = edge_index[:, edge_mask_1]
        edge_attr_1 = edge_attr[edge_mask_1]
        x_hidden = self.input_to_hidden(x, edge_index_1, edge_attr_1, bias_tensor)
        x[hidden_mask] = x_hidden[hidden_mask]

        # Step 2: Hidden → Output
        edge_mask_2 = (node_types[edge_index[0]] == 1) & (node_types[edge_index[1]] == 2)
        edge_index_2 = edge_index[:, edge_mask_2]
        edge_attr_2 = edge_attr[edge_mask_2]
        x_output = self.hidden_to_output(x, edge_index_2, edge_attr_2, bias_tensor)

        return x_output[output_mask]

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

def convert_networkx_to_pyg_data(G, mnist_input=None):
    type_map = {"in": 0, "hidden": 1, "out": 2}
    node_feats = []
    node_types = []
    biases = []
    node_id_map = {nid: i for i, nid in enumerate(G.nodes())}

    for nid in G.nodes:
        bias = G.nodes[nid].get("bias", 0.0)
        node_type = nid.split("_")[0]  # "in", "hidden", "out"
        node_types.append(type_map.get(node_type, 1))
        biases.append(bias)
        node_feats.append([0.0])  # Placeholder for activation, filled later

    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):
        i, j = node_id_map[u], node_id_map[v]
        edge_index.append([i, j])
        edge_attr.append([data.get("weight", 0.0)])

    # x_input = torch.tensor(mnist_input.view(-1, 1), dtype=torch.float) if mnist_input is not None else torch.zeros((len(node_feats), 1))
    x_input = mnist_input.view(-1, 1).clone().detach()
    
    return Data(
        x=x_input,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float).squeeze(),
        node_types=torch.tensor(node_types, dtype=torch.long),
        bias_vec=torch.tensor(biases, dtype=torch.float)
    )

def train_gnn_simulator(model, gnn, dataloader, optimizer, device):
    model.eval()
    gnn.train()
    loss_fn = torch.nn.MSELoss()

    for epoch in range(30):
        total_loss = 0
        print(f"Starting epoch {epoch}")
        for data, label in dataloader:
            
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            # Ground truth from original model
            gt = model(data).detach()

            # Build graph from MLP and image
            G = mlp_to_graph("./trained_networks/mlp_0.pkl")
            pyg_data = convert_networkx_to_pyg_data(G, mnist_input=data[0])

            pyg_data = pyg_data.to(device)

            pred = gnn(
                x_input=pyg_data.x,
                edge_index=pyg_data.edge_index,
                edge_attr=pyg_data.edge_attr,
                node_types=pyg_data.node_types,
                bias_vec=pyg_data.bias_vec
            )

            pred = pred.view(1, -1)
            
            loss = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss:.4f}")
        
def evaluate_gnn_simulator(mlp, gnn, dataloader, device):
    G = mlp_to_graph("./trained_networks/mlp_0.pkl")
    gnn.eval()
    total_loss = 0

    mlp_predicted = []
    mlp_vectors = []
    gnn_predicted = []
    gnn_vectors = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            
            mlp_out = mlp(x.view(x.size(0), -1))
            mlp_vectors.append(mlp_out)
            predicted_digit = mlp_out.argmax(dim=1).item()
            mlp_predicted.append(predicted_digit)
            
            pyg_data = convert_networkx_to_pyg_data(G, mnist_input=x[0])
            
            gnn_out = gnn(x_input=pyg_data.x,
                edge_index=pyg_data.edge_index,
                edge_attr=pyg_data.edge_attr,
                node_types=pyg_data.node_types,
                bias_vec=pyg_data.bias_vec
            )
            gnn_out = gnn_out.view(1, -1)
            gnn_vectors.append(gnn_out)
            
            predicted_digit = gnn_out.argmax(dim=1).item()
            gnn_predicted.append(predicted_digit)
            
            loss = torch.nn.functional.mse_loss(gnn_out, mlp_out)
            total_loss += loss.item()
            
    print(f"Test MSE loss: {total_loss / len(dataloader):.6f}")
    return mlp_predicted, mlp_vectors, gnn_predicted, gnn_vectors

def prediction_comparison_graph(mlp_predicted, gnn_predicted):
    x = list(range(1, len(mlp_predicted) + 1))  # Input indices

    plt.figure(figsize=(10, 5))
    plt.scatter(x, mlp_predicted, marker='o', label='MLP Prediction', color='blue')
    plt.scatter(x, gnn_predicted, marker='x', label='GNN Prediction', color='orange')

    plt.title("Predicted Digits from MLP vs GNN")
    plt.xlabel("Input Index")
    plt.ylabel("Predicted Digit")
    plt.yticks(range(10))  # Digits 0 through 9
    plt.legend()
    plt.grid(True)
    plt.show()

def vector_comparison_graph(mlp_vectors, gnn_vectors):
    cos_sims = []

    for mlp_out, gnn_out in zip(mlp_vectors, gnn_vectors):
        mlp_vec = mlp_out.squeeze()     # shape: [10]
        gnn_vec = gnn_out.squeeze()     # shape: [10]
        
        cos_sim = F.cosine_similarity(mlp_vec.unsqueeze(0), gnn_vec.unsqueeze(0)).item()
        cos_sims.append(cos_sim)

    # Plotting
    # plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cos_sims)+1), cos_sims, marker='o', linestyle="None")
    plt.title("Cosine Similarity Between MLP and GNN Outputs")
    plt.xlabel("Input Index")
    plt.ylabel("Cosine Similarity")
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load one MLP
    mlp_info = pickle.load(open("./trained_networks/mlp_0.pkl", "rb"))
    hidden_size = mlp_info["hidden_size"]
    model = SmallMLP(hidden_size=hidden_size).to(device)
    model.load_state_dict(mlp_info["state_dict"])
    
    # Initialize GNN
    gnn = GNNMLPSimulator(input_size=784, hidden_size=hidden_size, output_size=10).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-3)

    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    subset_size = 125  # or any smaller number
    mnist_subset = Subset(mnist, range(subset_size))

    # Train/test split
    train_size = int(0.8 * subset_size)
    test_size = subset_size - train_size
    train_dataset, test_dataset = random_split(mnist_subset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Train
    # print("Starting training")
    # train_gnn_simulator(model, gnn, train_loader, optimizer, device)

    # # Save model
    # torch.save(gnn.state_dict(), "gnn_forward_op_30.pt")
    
    # Load in model
    gnn.load_state_dict(torch.load("gnn_forward_op_30.pt"))
    gnn.eval()  # Set to evaluation mode if testing
    print("Loaded gnn")
    
    mlp_predicted, mlp_vectors, gnn_predicted, gnn_vectors = evaluate_gnn_simulator(model, gnn, test_loader, device)
    
    prediction_comparison_graph(mlp_predicted, gnn_predicted)
    vector_comparison_graph(mlp_vectors, gnn_vectors)
    