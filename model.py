import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import argparse




args = (10, 2, 600)

# Define a simple feedforward neural network## fully connected layers class
class fcn(nn.Module):
    def __init__(self, num_classes, num_layers, num_neurons):
        super(fcn, self).__init__()
        layers = []
        layers.append(nn.Linear(28*28, num_neurons))
        layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(num_neurons, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Set up the training and test data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = fcn(*args)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#save model_init.pt
torch.save(model.state_dict(), './fc_2_2_600_55000_0_mnist/model_init.pt')

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the model
save_path = './fc_2_2_600_55000_0_mnist/model.pt'
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

