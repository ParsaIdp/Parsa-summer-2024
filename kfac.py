import torch
import torchvision

from backpack import backpack, extend

from backpack.extensions import KFAC

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import random
import math



BATCH_SIZE = 64
STEP_SIZE = 0.01
DAMPING = 1.0
MAX_ITER = 100
torch.manual_seed(0)

mnist_loader = torch.utils.data.dataloader.DataLoader(
    torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,)
            )
        ])),
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 2, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Flatten(),
    torch.nn.Linear(288, 10)
)


loss_function = torch.nn.CrossEntropyLoss()
def get_accuracy(output, targets):
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()

class DiagGGNOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters, step_size, damping):
        super().__init__(
            parameters,
            dict(step_size=step_size, damping=damping)
        )

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                step_direction = p.grad / (p.diag_ggn_mc + group["damping"])
                p.data.add_(-group["step_size"], step_direction)
        return loss
extend(model)
extend(loss_function)

optimizer = DiagGGNOptimizer(
    model.parameters(),
    step_size=STEP_SIZE,
    damping=DAMPING
)

#use adam optimizer to train
optimizer_train = optim.Adam(model.parameters(), lr=0.001)


# save the initil weights of the model
initial_model = copy.deepcopy(model)

#train the model on mnist 

def train_model(model, optimizer, loss_function, dataloader, max_iter):
    model.train()
    for epoch in range(max_iter):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{max_iter}, Loss: {loss.item()}")
    return model

model = train_model(model, optimizer_train, loss_function, mnist_loader, MAX_ITER) 

#save the final weights of the model
final_model = copy.deepcopy(model)


def fisher_information(model, dataloader):
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    fisher_information = torch.zeros(num_params, num_params)

    iteration_counter = 0

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = loss_function(outputs, labels)

        with backpack(KFAC()):
            loss.backward()

        current_position = 0
        for param in model.parameters():
            kfac_info = param.kfac

            if len(kfac_info) == 1:
                size = kfac_info[0].shape[0]
                end_position = current_position + size

                if end_position > num_params:
                    break

        
                fisher_information[current_position:end_position, current_position:end_position] += kfac_info[0]
                current_position = end_position

            else:
                A, G = kfac_info[0], kfac_info[1]


                kron_prod = torch.kron(A, G)

                size = kron_prod.shape[0]
                end_position = current_position + size

                if end_position > num_params:
                    break

                
                fisher_information[current_position:end_position, current_position:end_position] += kron_prod
                current_position = end_position  # Update current_position

        iteration_counter += 1

        # Reset current_position every 4 iterations
        if iteration_counter % 4 == 0:
            current_position = 0

    fisher_information /= len(dataloader.dataset)

    return fisher_information

#calculate epsilon as the inverse of the variance of weights of initial model
def calculate_epsilon(model):
    epsilon = 0
    for param in model.parameters():
        epsilon += torch.var(param).item()
    return 1/epsilon

epsilon = calculate_epsilon(initial_model)



# Calculate Fisher Information Matrix
fisher_info = fisher_information(model, mnist_loader)
print("Fisher Information Matrix:")
print(fisher_info)

#calculate the eigenvalues of the fisher information matrix
eigenvalues, _ = torch.linalg.eig(fisher_info)
eigenvalues = eigenvalues.real

#sort the eigenvalues in descending order
sorted_eigenvalues = torch.sort(eigenvalues, descending=True).values
print("Sorted Eigenvalues:")
print(sorted_eigenvalues)

#calculate the number of eigenvalues that are greater than epsilon divided by 2*no. of samples - 1
def calculate_ranks(eigenvalues, epsilon, num_samples):
    ranks = 0
    for eigenvalue in eigenvalues:
        if eigenvalue > epsilon / (2 * num_samples - 1):
            ranks += 1
        else:
            break
    return ranks

effective_dim = calculate_ranks(sorted_eigenvalues, epsilon, len(mnist_loader.dataset))
print("Effective Dimension:")
print(effective_dim)

#calculate the sum of the log(2(n-1)eigenvalues/ epsilon + 1) where n is the number of samples for first effective_dim eigenvalues
def calculate_sne(eigenvalues, epsilon, num_samples, effective_dim):
    trace = 0
    for i in range(effective_dim):
        trace += 1 + math.log(2 * (num_samples - 1) * eigenvalues[i] / epsilon + 1)
    return trace

sne = calculate_sne(sorted_eigenvalues, epsilon, len(mnist_loader.dataset), effective_dim)

print("SNE:")
print(sne)

#the smallest eigenvalue above the threshold epsilon/2(n-1) is the effective dimension
lambda_r = eigenvalues[effective_dim - 1]


#slope of the curve of eigenvalues in log scale
def calculate_c(A, r, eigenvalues):
    
    # Extract the r-th eigenvalue
    lambda_r = eigenvalues[r-1]  # Python is 0-indexed, so r-1
    
    def satisfies_condition(c_prime):
        for i in range(len(eigenvalues) - 1, r - 2, -1):
            if eigenvalues[i] > lambda_r * np.exp(-c_prime * (i - r)):
                return False
        return True
    
    # Binary search for the supremum of c_prime that satisfies the condition
    c_min, c_max = 0, 10  # Initialize bounds for c_prime
    epsilon = 1e-6  # Precision for stopping criteria
    
    while c_max - c_min > epsilon:
        c_prime = (c_min + c_max) / 2
        if satisfies_condition(c_prime):
            c_min = c_prime
        else:
            c_max = c_prime
    
    return c_min

c = calculate_c(fisher_info, effective_dim, sorted_eigenvalues)

print("C:")
print(c)

#calculate the PAC-Bayes bound as sne + 2/c + epsilon(final_weight - initial_weight)^2 / 4n - 1

def calculate_pac_bayes_bound(sne, c, epsilon, final_model, initial_model, num_samples):
   # Calculate the norm of the difference between the parameters of the final and initial models
    # Flatten the parameters before subtracting
    diff_norm = torch.norm(torch.cat([p1.data.view(-1) - p2.data.view(-1) for p1, p2 in zip(final_model.parameters(), initial_model.parameters())])) ** 2
    bound = (sne + (2 / c) + epsilon * diff_norm) / (4 * num_samples - 1)
    return bound

pac_bayes_bound = calculate_pac_bayes_bound(sne, c, epsilon, final_model, initial_model, len(mnist_loader.dataset))

print("PAC-Bayes Bound:")
print(pac_bayes_bound)






    
