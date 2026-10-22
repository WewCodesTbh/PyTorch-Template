import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pygame
import random
import MyTorchTemplate
from time import *
        

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

batch_size = 500
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
n_iter = 1200

layers = [28*28, 100, 100, 10]
t = time()
model = MyTorchTemplate.trainLinear(train_dataset, test_dataset, layers, nn.LeakyReLU(), batch_size, n_iter)
print("training took {0}s".format(round(time()-t, 4)))

print("training complete. Draw some shapes! Left click to place a pixel, right click to erase, c to clear and x to evaluate")
print("after evaluating, press v to calculate the loss via cross-entropy!")
mouse_data = [False, False, False]

pygame.init()

SCALE_FACTOR = 11 #How much larger should the drawing area be than the MNIST data set

gameDisplay = pygame.display.set_mode((28*SCALE_FACTOR,28*SCALE_FACTOR))

def compress_image(array, scale_factor):
    '''very basic image compression, requires an odd scale factor'''
    pxarray = [] #could use the in built pixelarray function in pygame
    r = (scale_factor-1)//2
    for i in range(r, len(array), scale_factor):
        row = []
        for j in range(r, len(array[i]), scale_factor):
            avg = []
            for x_offset in range(-r, r):
                for y_offset in range(-r, r):
                    try:
                        avg.append(array[i+x_offset][j+y_offset])
                    except IndexError:
                        pass
            avg = sum(avg)/len(avg)
            row.append(avg)
        pxarray.append(row)
    return pxarray

def cross_entropy(dist1, dist2):
    return -sum([dist1[i]*np.log(dist2[i]) for i in range(min(len(dist1), len(dist2)))])
                    
BRUSH_SIZE = round(SCALE_FACTOR*0.9) #seems to work the best from experimentation
guess = [0,0,0,0,0,0,0,0,0,0]
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
                uncompressed_wrapped = []
                for i in range(28*SCALE_FACTOR):
                    r = []
                    for j in range(28*SCALE_FACTOR):
                        r.append(gameDisplay.get_at((j,i))[0]/255) #monochrome so we only take the R channel
                    uncompressed_wrapped.append(r)
                gameDisplay.unlock()
                compressed_wrapped = compress_image(uncompressed_wrapped, SCALE_FACTOR)
                compressed = []
                for row in compressed_wrapped:
                    compressed += row
                
                guess = model(torch.Tensor(compressed)).detach().numpy().tolist()
                print([round(g, 5) for g in guess])
                for i in range(len(guess)):
                    if guess[i] == max(guess):
                        print(i)
            if event.key == pygame.K_v:
                try:
                    label = int(input("what number did you draw?: "))
                except ValueError:
                    print("label is too large!")
                    continue
                try:
                    label_dist = [0]*10
                    label_dist[label] = 1
                except IndexError:
                    print("only input one digit!")
                    continue
                #we normalise guess first to compare to this distribution
                try:
                    guess_normalised = [max(g/sum(guess), 0.01) for g in guess]
                except ZeroDivisionError:
                    guess_normalised = guess
                print("cross entropy approximation: {0}".format(cross_entropy(label_dist, guess_normalised)))
                
                
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_data[event.button-1] = True
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_data[event.button-1] = False
        
        loc = pygame.mouse.get_pos()
        cols = [(255, 255, 255), (127, 127, 127), (0,0,0)]
        for button in enumerate(mouse_data):
            if button[1]:
                for i in range(-BRUSH_SIZE, BRUSH_SIZE-1):
                    for j in range(-BRUSH_SIZE, BRUSH_SIZE-1): #brush size = 2 to match training data
                        pos = [max(0,int(loc[0]+i)), max(0,round(loc[1]+j))]
                        gameDisplay.set_at(pos, [i for i in cols[button[0]]])
        pygame.display.update()
            
