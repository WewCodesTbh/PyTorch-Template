import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pygame
import random
import MyTorchTemplate
from time import *

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
        

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    
batch_size = 500
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
n_iter = 2000

layers = [28*28, 100, 100, 10]
t = time()
'''print("training using nn.Tanh()")
model = MyTorchTemplate.train(train_dataset, test_dataset, layers, nn.Tanh(), batch_size, n_iter)
print("training took {0}s".format(round(time()-t, 4)))'''
print("training using nn.LeakyReLU()")
model = MyTorchTemplate.train(train_dataset, test_dataset, layers, nn.LeakyReLU(), batch_size, n_iter)
print("training took {0}s".format(round(time()-t, 4)))

print("training complete. Draw some shapes! Left click to place a pixel, right click to erase, c to clear and x twice to evaluate")
mouse_data = [False, False, False]

pygame.init()
gameDisplay = pygame.display.set_mode((28,28))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                gameDisplay.fill((0,0,0))
            if event.key == pygame.K_x:
                gameDisplay.lock()
                pxarray = [] #could use the in built pixelarray function in pygame, but the display is so small
                #the ease of conversion from inbuilt python arrays to numpy arrays outweights the speed up
                for i in range(28):
                    for j in range(28):
                        pxarray.append(gameDisplay.get_at((i,j))[0]//255) #monochrome so we only take the R channel
                gameDisplay.unlock()
                guess = model(torch.Tensor(pxarray)).detach().numpy().tolist()
                for a in range(28):
                    for b in range(28):
                        gameDisplay.set_at((b, a), [255*pxarray[a*28+b], 0, 0])
                pygame.display.update()
                print(guess)
                for i in range(len(guess)):
                    if guess[i] == max(guess):
                        print(i)
                
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_data[event.button-1] = True
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_data[event.button-1] = False
        
        loc = pygame.mouse.get_pos()
        cols = [(255, 255, 255), (127, 127, 127), (0,0,0)]
        for button in enumerate(mouse_data):
            if button[1]:
                for i in range(-1, 1):
                    for j in range(-1, 1): #brush size = 2 to match training data
                        pos = [max(0,int(loc[0]+i)), max(0,round(loc[1]+j))]
                        gameDisplay.set_at(pos, [i for i in cols[button[0]]])
        pygame.display.update()
            
