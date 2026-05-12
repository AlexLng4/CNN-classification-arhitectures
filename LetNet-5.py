import json
import os
import torch
from torch import nn
import tensorflow as tf
import torch.nn.functional as F

def load_data():
    mnsit = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnsit.load_data()
    return x_train, y_train, x_test, y_test


def load_rbf_centers(json_path):
    """Load the fixed 7x12 bitmap centers from JSON. Returns tensor (10, 84)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    centers = []
    for i in range(10):
        centers.append(data["centers"][str(i)])
    return torch.tensor(centers, dtype=torch.float32)  # (10, 84)


class LeNetGaussianConnections(torch.nn.Module):
    """RBF output layer with FIXED centers (non-trainable), faithful to the paper."""

    def __init__(self, n_input, n_output, centers_tensor):
        super(LeNetGaussianConnections, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        # register_buffer = part of the model state but NOT a trainable parameter
        self.register_buffer('centers', centers_tensor)  # (n_output, n_input)

    def forward(self, input):

        extend_input = input[:,None, :]
        extended_centers_tensor = self.centers[None, :, :]
        distances = (extend_input - extended_centers_tensor)**2
        media = torch.sum(distances, dim=2)
        return media


class LeNetLoss(nn.Module):
    """
    Original LeNet-5 loss from the paper:
        L = d_correct + log(sum_j exp(-d_j))

    Where d_j are the RBF distances. This encourages:
      - d_correct to be small (input close to correct prototype)
      - all other d_j to be large (input far from wrong prototypes)
    """

    def __init__(self):
        super(LeNetLoss, self).__init__()

    # Penalized log-likelihood
    def forward(self, distances, labels):

        number_of_batches = distances.shape[0]
        rows = torch.arange(number_of_batches)

        # specify the rows in order to map correctly
        correct_distances = distances[rows, labels]

        # "Penalized log-likelihood" / "log-sum-exp penalty loss" (LeCun et al., 1998)
        # in theory we should avoid the correct distance index but for now we keep it like this.
        # if the distances (excluding the correct one) are very big (that is good) we use the inverse fraction to get in
        # [0,1]. meaning big distance will go to 0. vice versa small distance for missmatch index go to high and penalize
        # the score.

        j = torch.tensor(0.1, device=distances.device)

        # Formula corectată:
        # torch.exp(-j) acum funcționează pentru că -j este un Tensor
        penalty = torch.log(torch.exp(-j) + torch.sum(torch.exp(-distances), dim=1))

        loss = correct_distances + penalty

        return loss.mean()


class LeNetSubsampling(nn.Module):
    def __init__(self, in_channels):
        super(LeNetSubsampling, self).__init__()

        self.weight = nn.Parameter(torch.randn(in_channels, 1, 1))
        self.bias = nn.Parameter(torch.randn(in_channels, 1, 1))

    def forward(self, x):
        # in the original paper of LeNet5, the author is using a trainable pooling
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # affine projection of the pooled value
        x = self.weight * x + self.bias
        # also do the activation function here for simplicity
        x = torch.tanh(x)
        return x


class LeNet5(torch.nn.Module):
    def __init__(self, centers_tensor):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pooling2 = LeNetSubsampling(in_channels=6)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pooling4 = LeNetSubsampling(16)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=120, out_features=84)

        # RBF layer with fixed bitmap centers (non-trainable)
        self.rbf = LeNetGaussianConnections(n_input=84, n_output=10, centers_tensor=centers_tensor)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))    # -> (batch, 6, 24, 24)
        x = self.pooling2(x)              # -> (batch, 6, 12, 12)
        x = torch.tanh(self.conv3(x))     # -> (batch, 16, 8, 8)
        x = self.pooling4(x)              # -> (batch, 16, 4, 4)
        x = torch.tanh(self.conv5(x))     # -> (batch, 120, 1, 1)
        x = self.flatten(x)               # -> (batch, 120)
        x = torch.tanh(self.dense1(x))    # -> (batch, 84)
        distances = self.rbf(x)           # -> (batch, 10) — distances to prototypes
        return distances


if __name__ == "__main__":

    # ---------- data ----------------------------------------------------------
    x_train, y_train, x_test, y_test = load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0
    x_test  = torch.tensor(x_test,  dtype=torch.float32).unsqueeze(1) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset  = torch.utils.data.TensorDataset(x_test, y_test)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=256)

    # ---------- model / loss --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load fixed RBF centers from JSON
    json_path = os.path.join(os.path.dirname(__file__), "rbf_centers.json")
    centers_tensor = load_rbf_centers(json_path)

    model = LeNet5(centers_tensor).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = LeNetLoss()

    # ---------- training loop -------------------------------------------------
    # Prediction: the class with the SMALLEST distance wins
    Epochs = 30
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"[INFO] Batch size: 64, Batches per epoch: {len(train_loader)}")
    print(f"[INFO] Starting training for {Epochs} epochs...\n")

    for epoch in range(Epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            distances = model(batch_x)
            loss = criterion(distances, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 200 == 0:
                print(f"  [Epoch {epoch+1}/{Epochs}] Batch {batch_idx+1}/{len(train_loader)} — Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                distances = model(batch_x)
                predictions = distances.argmin(dim=1)
                correct += (predictions == batch_y).sum().item()

        accuracy = correct / len(test_dataset) * 100
        print(f"Epoch {epoch+1}/{Epochs} — Avg Loss: {avg_loss:.4f} — Test Accuracy: {accuracy:.2f}% ({correct}/{len(test_dataset)})")
        print()

    print("Training complete!")
