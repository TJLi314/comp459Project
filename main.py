import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import random
import pickle

def make_NN():
    NUM_NETWORKS = 50         
    SAVE_DIR = "./trained_networks"  # Where to save the trained models
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(SAVE_DIR, exist_ok=True)

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
        for epoch in range(3):  # Quick training
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


if __name__ == "__main__":
    make_NN()