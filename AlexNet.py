import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# implement LRN
class AlexNetLRN(nn.Module):
    def __init__(self):
        super(AlexNetLRN, self).__init__()
        # this class is implementation from the AlexNet paper for batch type of batch normalization
        #  k = 2, n = 5, α = 10−4, and β = 0.75
        self.register_buffer('k', torch.tensor(2.0))
        self.register_buffer('n', torch.tensor(5.0))
        self.register_buffer('alpha', torch.tensor(1e-4))
        self.register_buffer('beta', torch.tensor(0.75))

    def forward(self, x):
        # x shape: (batch, C, H, W)
        C = x.shape[1]
        half_n = int(self.n.item()) // 2
        # pad channels dimension so we can sum over the neighborhood
        x_sq = x * x
        # pad along channel dim (dim=1): (0,0,0,0,half_n,half_n) pads last, second-last, third-last dims
        x_sq_padded = torch.nn.functional.pad(x_sq, (0, 0, 0, 0, half_n, half_n))
        # sum over sliding window of size n across channels
        scale = torch.zeros_like(x)
        for i in range(int(self.n.item())):
            scale += x_sq_padded[:, i:i + C, :, :]
        scale = (self.k + self.alpha * scale).pow(self.beta)
        return x / scale



class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        # expected input is 227x227x3
        # after the first 2 conv layers we will apply an old method for normalization of the big data, LRN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        # -> 55x55x96
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # -> 27x27x96
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # -> 27x27x256
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # -> 13x13x256
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # -> 13x13x384
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # -> 13x13x384
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # -> 13x13x256
        self.pooling5= nn.MaxPool2d(kernel_size=3, stride=2)
        # -> 6x6x256

        # flattening to 9216
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)

        # initialize LRN variable
        self.lrn = AlexNetLRN()

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = self.lrn(x)
        x = self.pooling1(x)

        x = torch.relu(self.conv2(x))
        x = self.lrn(x)
        x = self.pooling2(x)

        x = torch.relu(self.conv3(x))

        x = torch.relu(self.conv4(x))

        x = torch.relu(self.conv5(x))
        x = self.pooling5(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

if __name__=="__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms - resize to 227x227 as expected by AlexNet
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load cats vs dogs dataset
    train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
    test_dataset = datasets.ImageFolder(root='data/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize network - change output to 2 classes (cats vs dogs)
    net = AlexNet()
    # Replace last layer for binary classification
    net.fc3 = nn.Linear(in_features=4096, out_features=2)
    net = net.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Evaluation
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%\n")

    # Save model
    torch.save(net.state_dict(), 'alexnet_cats_dogs.pth')
    print("Model saved to alexnet_cats_dogs.pth")
