import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class FeedForwardNN(nn.Module):
    def __init__(self, layer_sizes, learning_rate, optimiser, activation_function):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        self.optimiser = optimiser(params=self.parameters(), lr = self.learning_rate)
    def forward(self, data):
        for layer in self.layers:
            data = self.activation_function(layer(data))
        return data
        
def main():
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    
    batch_size = 500
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    n_iter = 2000
    n_epochs = int(n_iter/(len(train_dataset)/batch_size))

    layers = [28*28, 100, 100, 10]
    model = FeedForwardNN(layers, 0.2, torch.optim.SGD, nn.Tanh())
    optimiser = model.optimiser
    criterion = nn.CrossEntropyLoss()
    
    try:
        device = torch.accelerator.current_accelertor()
    except:
        device = torch.device("cpu")
    print("Current Device is: {0}".format(device))
    model.to(device)
    COUNT = 0
    for epoch in range(1, n_epochs):
        for i, (D, labels) in enumerate(train_loader):
            D = D.view(-1, 28*28).requires_grad_() #format the data
            optimiser.zero_grad()
            outputs = model(D)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            COUNT += 1
            if COUNT%100 == 0:
                correct = 0
                total = 0
                for T, L in test_loader:
                    T = T.view(-1, 28*28).requires_grad_() #format the data in the same manner 
                    outputs = model(T)
                    _, predicted = torch.max(outputs.data, 1)
                    total += L.size(0)
                    correct += (predicted == L).sum()
                accuracy = round(100*correct/total, 4)
                print('Iteration: {} Loss: {} Accuracy: {}%'.format(COUNT, loss.item(), accuracy))

if __name__ == "__main__":
    main()
