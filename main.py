import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
from pathlib import Path
import MNIST
import layers as ly


### FUNCTIONS
def softmax(y):
    e_y = np.exp(y)
    return e_y / np.sum(e_y)

def MSE(a, y):
    return np.linalg.norm(a-y)**2/len(a)


### LOAD DATA
data = MNIST.load("LeNet5/MNIST_10%/train", "LeNet5/MNIST_10%/test")
train_data = data[0]
test_data = data[1]


### HYPERPARAMETERS
eta = 0.1                   # learning rate
num_epoch = 2               # number of training epochs
batch_size = 20             # size of training batches
num_batches = int(len(train_data)/batch_size)


### NETWORK STRUCTURE
# initialise weights (Kaiming initialization) 
F1 = np.random.normal(0,np.sqrt(2/1024),size=(6,1,5,5))
F3 = np.random.normal(0,np.sqrt(2/1176),size=(16,6,5,5))
W5 = np.random.normal(0,np.sqrt(2/400),size=(120,400))
W6 = np.random.normal(0,np.sqrt(2/120),size=(84,120))
W7 = np.random.normal(0,np.sqrt(2/84),size=(10,84))

# layers
L1_conv = ly.ConvLayer(F1)
L2_pool = ly.PoolLayer(2)
L3_conv = ly.ConvLayer(F3)
L4_pool = ly.PoolLayer(2)
L5_fc = ly.FCLayer(W5)
L6_fc = ly.FCLayer(W6)
L7_fc = ly.FCLayer(W7)


### TRAINING
batches = MNIST.batch(train_data, batch_size)
for i, batch in enumerate(batches):
    batch_loss = 0.0
    for j, example in enumerate(batch):

        # forward
        Z1 = L1_conv.forward(example[0])
        Z2 = L2_pool.forward(Z1)
        Z3 = L3_conv.forward(Z2)
        Z4 = L4_pool.forward(Z3)
        a4 = np.ravel(Z4)
        a5 = L5_fc.forward(a4)
        a6 = L6_fc.forward(a5)
        a7 = L7_fc.forward(a6)

        # calculate loss
        output = softmax(a7)
        loss = MSE(output, example[1])
        batch_loss += loss

        # backward
        dCda6 = L7_fc.backward(a6, output-example[1])
        dCda5 = L6_fc.backward(a5, dCda6)
        dCda4 = L5_fc.backward(a4, dCda5)
        dCdZ4 = np.reshape(dCda4, np.shape(Z4))
        dCdZ3 = L4_pool.backward(dCdZ4)
        dCdZ2 = L3_conv.backward(Z2, dCdZ3)
        dCdZ1 = L2_pool.backward(dCdZ2)
        dCdZ0 = L1_conv.backward(example[0], dCdZ1)

    L1_conv.update(eta)
    L3_conv.update(eta)
    L5_fc.update(eta)
    L6_fc.update(eta)
    L7_fc.update(eta)
    batch_loss /= batch_size
    print(f"batch {i+1} of {num_batches}: loss = {batch_loss:.4f}")



### TESTING
scores = np.zeros(10)
totals = np.zeros(10)

record = []
for j, example in enumerate(test_data):

    # forward
    Z1 = L1_conv.forward(example[0])
    Z2 = L2_pool.forward(Z1)
    Z3 = L3_conv.forward(Z2)
    Z4 = L4_pool.forward(Z3)
    a4 = np.ravel(Z4)
    a5 = L5_fc.forward(a4)
    a6 = L6_fc.forward(a5)
    a7 = L7_fc.forward(a6)

    guess = np.argmax(a7)
    label = example[1]
    totals[label] += 1
    if guess == label:
        scores[label] += 1



### RESULTS
print("")
print("ACCURACY:")
for k in range(10):
    print(f"{k}: {scores[k]/totals[k]*100}%")
print(f"TOTAL: {np.sum(scores)/np.sum(totals)*100}%")