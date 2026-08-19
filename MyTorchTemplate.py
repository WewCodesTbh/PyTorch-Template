import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class LinearNN(nn.Module):
    def __init__(self, layer_sizes, activation_function):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        self.activ_func = activation_function
    def forward(self, data):
        for layer in self.layers:
            data = self.activ_func(layer(data))
        return data

class ConvolutionalNN(nn.Module):
    def __init__(self, layer_sizes, convolution_sizes, convolution_widths, activation_function):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(convolution_sizes)):
            self.layers.append(nn.Conv2d(convolution_sizes[i-1], convolution_sizes[i], convolution_widths[i-1]))
        self.layers.append(nn.Linear(convolution_sizes[::-1][0], layer_sizes[0]*convolution_widths[::-1][0], convolution_widths[::-1][0]))
        for i in range(2, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        if len(convolution_sizes) != len(convolution_widths):
            raise Exception("Number of specified convolution sizes must match the number of specified convolution widths")
        if layer_sizes[0]/convolution_widths[::-1][0] != int(layer_sizes[0]/convolution_widths[::-1][0]):
            raise Exception("Final convolution layer is incompatible with first linear layer, the first linear layer's dimension must be a multiple of the kernel width of the final convolution layer")
        self.activ_func = activation_function
    def forward(self, data):
        output = data
        for layer in self.layers[:len(self.layers)-1]: #TODO: allow arbitrary ordering of linear and convolutional layers
            output = self.activ_func(layer(data))
        output = self.layers[::-1][0](data)
        return output
        
def trainLinear(train_dataset, test_dataset, layers, activation = nn.LeakyReLU(), batch_size = 500, iterations = 1000, training_rate = 0.2, momentum = 0.01, optimiser = torch.optim.SGD, criterion = nn.CrossEntropyLoss()):

    model = LinearNN(layers, activation)
    opti = optimiser(model.parameters(), lr = training_rate)
    opti.zero_grad()

    try:
        device = torch.accelerator.current_accelertor()
    except:
        device = torch.device("cpu")
        
    n_epochs = int(iterations/(len(train_dataset)/batch_size))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print("Current Device is: {0}".format(device))
    model.to(device)
    COUNT = 0
    for epoch in range(1, n_epochs):
        for i, (D, labels) in enumerate(train_loader):
            D = D.view(-1, 28*28).requires_grad_() #format the data
            opti.zero_grad()
            outputs = model(D)
            loss = criterion(outputs, labels)
            loss.backward()
            opti.step()
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
                accuracy = round(100*int(correct)/total, 5)
                print('Iteration: {} Loss: {} Accuracy: {}%'.format(COUNT, round(loss.item(), 5), accuracy))
    return model

def trainConvolutional(train_dataset, test_dataset, convolutions, layers, activation = nn.LeakyReLU(), batch_size = 500, iterations = 1000, training_rate = 0.2, momentum = 0.01, optimiser = torch.optim.SGD, criterion = nn.CrossEntropyLoss()):
    '''not yet implemented'''
    model = ConvolutionalNN(layers, convolutions[0], convolutions[1], activation)
    print(model)
    opti = optimiser(model.parameters(), lr = training_rate)
    n_epochs = int(iterations/(len(train_dataset)/batch_size))

    try:
        device = torch.accelerator.current_accelertor()
    except:
        device = torch.device("cpu")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print("Current Device is: {0}".format(device))
    model.to(device)
    COUNT = 0
    for epoch in range(1, n_epochs):
        for i, (D, labels) in enumerate(train_loader):
            D = D.view(-1, 28, 28).requires_grad_() #format the data
            print(D[0].shape)
            opti.zero_grad()
            outputs = model(D)
            loss = criterion(outputs, labels)
            loss.backward()
            opti.step()
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
                accuracy = round(100*int(correct)/total, 5)
                print('Iteration: {} Loss: {} Accuracy: {}%'.format(COUNT, round(loss.item(), 5), accuracy))
    return model

    
    
