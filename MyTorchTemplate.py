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
    '''half implemented, throws an error when trying to train'''
    def __init__(self, input_shape, conv_layers, linear_layers, activation_function):
        super().__init__()
        in_channels, height, width = input_shape
        self.activ_func = activation_function
        if len({len(x) for x in conv_layers}) == 1:
            pass
        else:
            raise Warning("Convolutional parameter lists must have equal lengths, not {0} {1} {2}".format(*[len(x) for x in conv_layers]))
        self.conv_layers = nn.ModuleList()
        for i in range(len(conv_layers[0])):
            self.conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=conv_layers[0][i], kernel_size=conv_layers[1][i], stride=conv_layers[2][i]))
            in_channels = conv_layers[0][i]
            height =(height-conv_layers[1][i])//conv_layers[2][i]+1
            width = (width-conv_layers[1][i])//conv_layers[2][i]+1
        linear_input = in_channels*height*width
        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(linear_input, linear_layers[0]))
        for i in range(1, len(linear_layers)):
            self.lin_layers.append(nn.Linear(linear_layers[i-1], linear_layers[i]))

    def forward(self, data):
        for layer in self.conv_layers:
            data = self.activ_func(layer(data))
        data = torch.flatten(data, start_dim=1)
        for layer in self.lin_layers:
            data = self.activ_func(layer(data))
        return data
        
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
                CONFUSION_MATRIX = [[0 for _ in range(layers[::-1][0])] for _ in range(layers[::-1][0])]
                correct = 0
                total = 0
                for T, L in test_loader:
                    T = T.view(-1, 28*28).requires_grad_() #format the data in the same manner 
                    outputs = model(T)
                    _, predicted = torch.max(outputs.data, 1)
                    total += L.size(0)
                    correct += (predicted == L).sum()
                    for i in range(len(L)):
                        CONFUSION_MATRIX[L[i]][predicted[i]] += 1
                accuracy = round(100*int(correct)/total, 5)
                print('Iteration: {} Loss: {} Accuracy: {}%'.format(COUNT, round(loss.item(), 5), accuracy))
                print("Confusion Matrix: ")
                print(torch.tensor(CONFUSION_MATRIX))
    return model, CONFUSION_MATRIX

def trainConvolutional(train_dataset, test_dataset, input_shape, conv_layers, lin_layers, activation = nn.LeakyReLU(), batch_size = 500, iterations = 1000, training_rate = 0.2, momentum = 0.01, optimiser = torch.optim.SGD, criterion = nn.CrossEntropyLoss()):
    '''not yet implemented'''
    model = ConvolutionalNN(input_shape, conv_layers, lin_layers, activation)
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
            opti.zero_grad()
            outputs = model(D) #throws an error here, says there's 500 channels but expects one. But when I manually pass a batch through the example NN it works fine? Very odd
            loss = criterion(outputs, labels)
            loss.backward()
            opti.step()
            COUNT += 1
            if COUNT%100 == 0:
                correct = 0
                total = 0
                for T, L in test_loader:
                    T = T.view(-1, 28, 28).requires_grad_() #format the data in the same manner 
                    outputs = model(T)
                    _, predicted = torch.max(outputs.data, 1)
                    total += L.size(0)
                    correct += (predicted == L).sum()
                accuracy = round(100*int(correct)/total, 5)
                print('Iteration: {} Loss: {} Accuracy: {}%'.format(COUNT, round(loss.item(), 5), accuracy))
    return model

    
    
