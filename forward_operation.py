import pickle
import networkx as nx
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

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
    def __init__(self, activation=None):
        super().__init__(aggr='add')
        self.activation = activation

    def forward(self, x, edge_index, edge_weight, bias=None):
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        if bias is not None:
            out += bias
        if self.activation:
            out = self.activation(out)
        return out

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)

class GNNMLPSimulator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_to_hidden = MLPForwardLayer(activation=F.relu)
        self.hidden_to_output = MLPForwardLayer(activation=None)

    def forward(self, x_input, edge_index, edge_attr, node_types, bias_vec):
        # Split node features
        input_mask = node_types == 0
        hidden_mask = node_types == 1
        output_mask = node_types == 2

        bias_tensor = bias_vec.unsqueeze(1)

        # Initial activations
        x = torch.zeros_like(bias_tensor)
        x[input_mask] = x_input

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
    type_map = {"input": 0, "hidden": 1, "output": 2}
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

    x_input = torch.tensor(mnist_input.view(-1, 1), dtype=torch.float) if mnist_input is not None else torch.zeros((len(node_feats), 1))
    
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

    for epoch in range(5):
        total_loss = 0
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            # Ground truth from original model
            gt = model(data).detach()

            # Build graph from MLP and image
            G = mlp_to_graph("path_to_mlp.pkl")
            pyg_data = convert_networkx_to_pyg_data(G, mnist_input=data[0])

            pyg_data = pyg_data.to(device)

            pred = gnn(
                x_input=pyg_data.x,
                edge_index=pyg_data.edge_index,
                edge_attr=pyg_data.edge_attr,
                node_types=pyg_data.node_types,
                bias_vec=pyg_data.bias_vec
            )

            loss = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} Loss: {total_loss:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load one MLP
    mlp_info = pickle.load(open("path_to_mlp.pkl", "rb"))
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
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=1, shuffle=True)

    # Train
    train_gnn_simulator(model, gnn, dataloader, optimizer, device)
