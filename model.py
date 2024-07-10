import numpy as np
import scipy.signal as sg



# rectified linear unit activation function
def ReLU(X):
    return (X + np.abs(X))/2

# derivative of rectified linear unit function
def dReLU(X):
    return np.heaviside(X,0)



class Conv:
    def __init__(self, C, N, f, m, padding=0):
        # C = number of input channels
        # N = size of (square) input
        # f = number of filters
        # m = size of (square) filters
        # p = padding size
        self.filters = np.random.normal(0, np.sqrt(2/(C*N*N)), size=(f, C, m, m))
        self.bias = np.zeros(f)
        self.F_grads = []
        self.b_grads = []
        self.pad = padding
        self.dim = (f, C, m, m)
        self.down = None

    def forward(self, a):
        if self.pad > 0:
            a = np.pad(a, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)), constant_values=0)

        self.down = a     # store downstream activations for backpropagation

        maps = []
        for x in range(self.dim[0]):
            filter = self.filters[x,:,:,:]      # select current filter from filters array
            Z = sg.correlate(a, filter, mode='valid') + self.bias[x]    # cross-correlate filter and input to produce feature map
            maps.append(Z)      # append to list of feature maps for stacking
        return ReLU(np.concatenate(maps))

    def backward(self, dCdA):

        ### calculate dAdZ (f_L, N_L, N_L)
        maps = []
        for x in range(self.dim[0]):
            filter = self.filters[x,:,:,:]                                      # select current filter from filters array
            Z = sg.correlate(self.down, filter, mode='valid') + self.bias[x]       # cross-correlate filter and input to produce feature map
            maps.append(Z)
        dAdZ = dReLU(np.concatenate(maps))

        # calculate dCdZ  (f_L, N_L, N_L)
        dCdZ = dCdA * dAdZ

        ### LOCAL GRADIENTS
        ### filter gradient dCdF
        dCdF = np.zeros(self.dim)
        for x in range(self.dim[0]):
            dCdZ_x = np.array([dCdZ[x,:,:]])
            for y in range(self.dim[1]):
                A_down_y = np.array([self.down[y,:,:]])
                dCdF[x,y,:,:] = sg.correlate(A_down_y, dCdZ_x, mode='valid')
        self.F_grads.append(dCdF)

        ### bias gradient dCdb
        dCdb = np.sum(dCdZ, axis=(1,2))
        self.b_grads.append(dCdb)

        # DOWNSTREAM GRADIENTS
        A_dim = self.down.shape
        dCdA_down = np.zeros(A_dim)
        pad = self.dim[2]-1
        dCdZ_pad = np.pad(dCdZ, pad, constant_values=0)[pad:-pad,:,:]   # pad non-channel dimensions by m_L-1
        for y in range(self.dim[1]):
            F_y = self.filters[:,y,:,:]
            dCdA_down[y,:,:] = sg.convolve(dCdZ_pad, F_y, mode='valid')

        return dCdA_down

    def update(self, lr):
        # weights update
        F_grad = np.mean(np.stack(self.F_grads), axis=0)
        self.filters -= lr * F_grad
        self.F_grads = []
        # bias update
        b_grad = np.mean(np.stack(self.b_grads), axis=0)
        self.bias -= lr * b_grad
        self.b_grads = []



class FC:
    def __init__(self, N, M):
        # N = size of input
        # M = size of output
        self.weights = np.random.normal(0,np.sqrt(2/N**2), size=(M, N))
        self.bias = np.zeros(M)
        self.W_grads = []
        self.b_grads = []
        self.down = None

    def forward(self, a):
        if len(a.shape) > 1: a = a.reshape(-1)
        self.down = a
        z = self.weights @ a + self.bias
        return ReLU(z)

    def backward(self, dCda):

        dadz = dReLU(self.weights @ self.down + self.bias)
        dCdz = dadz * dCda
        dCdW = np.transpose(np.outer(self.down, dCdz))

        self.W_grads.append(dCdW)
        self.b_grads.append(dCdz)

        dCda_down = np.transpose(self.weights) @ dCdz

        return dCda_down

    def update(self, lr):
        # weights update
        W_grad = np.mean(np.stack(self.W_grads), axis=0)
        self.weights -= lr * W_grad
        self.W_grads = []
        # bias update
        b_grad = np.mean(np.stack(self.b_grads), axis=0)
        self.bias -= lr * b_grad
        self.b_grads = []



class MaxPool:
    def __init__(self, size):
        self.size = size
        self.mask = None
        self.dim = None

    def forward(self, input):
        C, M, M = np.shape(input)
        if M % self.size != 0:
            print("ERROR: Input width and height must be a multiple of MaxPool size.")
            return None
        else:
            P = int(M / self.size)
            pool = np.zeros((C, P, P))
            self.dim = pool.shape
            self.mask = np.zeros(input.shape)

            for y in range(C):
                for i in range(P):
                    for j in range(P):
                        I = self.size * i
                        J = self.size * j
                        patch = input[y, I:I+self.size, J:J+self.size]
                        pool[y, i, j] = np.max(patch)
                        mask_idx = np.unravel_index(patch.argmax(), patch.shape)
                        self.mask[y, I + mask_idx[0], J + mask_idx[1]] = 1

            return pool

    def backward(self, dCdZ):
        if len(dCdZ.shape) < len(self.dim): dCdZ = dCdZ.reshape(self.dim)
        dCdZ_2 = np.repeat(np.repeat(dCdZ, 2, axis=1), 2, axis=2)
        return dCdZ_2 * self.mask

    

class LeNet5:
    def __init__(self):
        self.conv1 = Conv(1, 28, 6, 5, padding=2)
        self.pool1 = MaxPool(2)
        self.conv2 = Conv(6, 14, 16, 5)
        self.pool2 = MaxPool(2)
        self.fc1 = FC(400, 120)
        self.fc2 = FC(120, 84)
        self.fc3 = FC(84, 10)

    def forward(self, x):
        z1 = self.pool1.forward(self.conv1.forward(x))
        z2 = self.pool2.forward(self.conv2.forward(z1))
        a1 = self.fc1.forward(z2)
        a2 = self.fc2.forward(a1)
        y = self.fc3.forward(a2)
        return y

    def backward(self, dCdy):
        dCda2 = self.fc3.backward(dCdy)
        dCda1 = self.fc2.backward(dCda2)
        dCdz2 = self.fc1.backward(dCda1)
        dCdz1 = self.conv2.backward(self.pool2.backward(dCdz2))
        dCdx = self.conv1.backward(self.pool1.backward(dCdz1))

    def update(self, lr):
        self.conv1.update(lr)
        self.conv2.update(lr)
        self.fc1.update(lr)
        self.fc2.update(lr)
        self.fc3.update(lr)

    def save(self, path):
        self.conv1.save(path)
        self.conv2.save(path)
        self.fc1.save(path)
        self.fc2.save(path)
        self.fc3.save(path)

    def load(self, path):
        self.conv1.load(path)
        self.conv2.load(path)
        self.fc1.load(path)
        self.fc2.load(path)
        self.fc3.load(path)

































class MaxPoolOld:
    def __init__(self, size):
        self.size = size
        self.mask = None
        self.dim = None

    def forward(self, input):
        C, M, M = np.shape(input)
        if M % self.size != 0:
            print("ERROR: Input width and height must be a multiple of MaxPool size.")
            return None

        else:
            P = int(M / self.size)

            pool_channels = []
            masks = []
            for y in range(C):
                channel = input[y,:,:]
                mask_y = np.zeros((M, M))
                pool_y = np.zeros((P, P))
                for i in range(P):
                    for j in range(P):
                        patch = channel[self.size*i:self.size*(i+1), self.size*j:self.size*(j+1)]
                        mask_index = np.zeros((self.size**2))
                        pool_y[i, j] = np.max(patch)
                        mask_index[np.argmax(patch)] = 1
                        mask_patch = np.reshape(mask_index, (self.size, self.size))
                        mask_y[self.size*i:self.size*(i+1), self.size*j:self.size*(j+1)] = mask_patch
                pool_channels.append(pool_y)
                masks.append(mask_y)

            self.mask = np.stack(masks)
            pool = np.stack(pool_channels)

            self.dim = pool.shape
            return pool

    def backward(self, dCdZ):
        if len(dCdZ.shape) < len(self.dim): dCdZ = dCdZ.reshape(self.dim)
        dCdZ_2 = np.repeat(np.repeat(dCdZ, 2, axis=1), 2, axis=2)
        return dCdZ_2 * self.mask