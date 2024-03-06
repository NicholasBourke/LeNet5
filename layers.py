import numpy as np
import scipy.signal as sg



# rectified linear unit activation function
def ReLU(X):
    return (X + np.abs(X))/2

# derivative of rectified linear unit function
def dReLU(X):
    return np.heaviside(X,0)



class ConvLayer:
    """
    Convolutional Layer L
    Attributes:
        filters tensor size=[f,C,m,m] (pre-initialised)
        bias vector size=[f] (optional)
            C = number of input channels
            f = number of filters
            m = size of filters (square)
    """
    def __init__(self, filters, bias=None):
        self.filters = filters
        if bias is None:
            bias = np.zeros(filters.shape[0])
        self.bias = bias
        self.F_grads = []
        self.b_grads = []

    def forward(self, input):
        """
        Forward Propagation
            Input: layer L-1 activations tensor size=[C,M,M]
                C = number of channels
                M = size of each channel (square)
            Output: layer L activations tensor size=[f,N,N]
                f = number of filters = number of channels
                N = size of each channel (square)
        """
        maps = []
        for x in range(self.filters.shape[0]):
            filter = self.filters[x,:,:,:]                                      # select current filter from filters array
            Z = sg.correlate(input, filter, mode='valid') + self.bias[x]        # cross-correlate filter and input to produce feature map
            maps.append(Z)                                                      # append to list of feature maps for stacking

        return ReLU(np.concatenate(maps))
    
    def backward(self, A_down, dCdA):
        """
        Backward Propagation
            Inputs:
                A_down: layer L-1 activations tensor size=[C,M,M]
                    C = number of channels
                    M = size of each channel (square)
                dCdA: layer L activation gradients tensor size=[f,N,N]
                    f = number of filters = number of channels
                    N = size of each channel (square)
            Output:
                dCdA_down: layer L-1 activation gradients tensor size=[C,M,M]
                    C = number of channels
                    M = size of each channel (square)
        """

        ### Local Gradients (for upgrade)
        F_dim = np.shape(self.filters)
        dCdF = np.zeros(F_dim)
        dCdb = np.zeros(F_dim[0])
        dCdZ_maps = []
        for x in range(F_dim[0]):
            F_x = self.filters[x,:,:,:]
            Z_x = sg.correlate(A_down, F_x, mode='valid') + self.bias[x]
            dAdZ_x = dReLU(Z_x)
            dCdA_x = np.array([dCdA[x,:,:]])
            dCdZ_x = dCdA_x * dAdZ_x
            dCdZ_maps.append(dCdZ_x)
            # calculate gradient matrix dC/dF_xy (gradients for channel y of filter x)
            for y in range(F_dim[1]):
                A_down_y = np.array([A_down[y,:,:]])
                dCdF[x,y,:,:] = sg.correlate(A_down_y, dCdZ_x, mode='valid')
        self.F_grads.append(dCdF)

        # calculate bias gradient dC/db
        dCdZ = np.concatenate(dCdZ_maps)
        dCdb = np.sum(dCdZ, axis=(1,2))
        self.b_grads.append(dCdb)

        ### Downstream Gradients (to backpropagate to layer L-1)
        A_dim = np.shape(A_down)
        dCdA_down = np.zeros(A_dim)
        # dCdZ_pad = pad(dCdZ, F_dim[2]-1)
        p = F_dim[2]-1
        dCdZ_pad = np.pad(dCdZ, ((0,0),(p,p),(p,p)), constant_values=0)
        for y in range(F_dim[1]):
            F_y = self.filters[:,y,:,:]
            dCdA_down[y,:,:] = sg.convolve(dCdZ_pad, F_y, mode='valid')

        return dCdA_down
    
    def update(self, learning_rate):
        """
        Gradient Descent Update
            1. collect gradients (calculated over batch of training examples) in self.F_grads, self.b_grads 
            2. stack into array and reset
            3. average over all examples (first axis of array)
            4. use to update parameters
        """
        
        # filters update
        F_grad_array = np.stack(self.F_grads)
        self.F_grads = []
        F_grad = np.mean(F_grad_array, axis=0)
        self.filters -= learning_rate * F_grad

        # bias update
        b_grad_array = np.stack(self.b_grads)
        self.b_grads = []
        b_grad = np.mean(b_grad_array, axis=0)
        self.bias -= learning_rate * b_grad

        return None



class FCLayer:
    """
    Fully Connected Layer L
    Attributes:
        weights tensor size=[N,M] (pre-initialised)
        bias vector size=[N] (optional)
            M = input size
            N = output size
    """
    def __init__(self, weights, bias=None):
        self.weights = weights
        if bias is None:
            bias = np.zeros(weights.shape[0])
        self.bias = bias
        self.W_grads = []
        self.b_grads = []

    def forward(self, input):
        """
        Forward Propagation
            Input: layer L-1 activations vector size=[M]
            Output: layer L activations vector size=[N]
        """
        z = self.weights @ input + self.bias
        return ReLU(z)
    
    def backward(self, a_down, dCda):
        """
        Backward Propagation
            Inputs:
                a_down: layer L-1 activations vector size=[M]
                dCda: layer L activation gradients vector size=[N]
            Outputs:
                dCda_down: layer L-1 activation gradients vector size=[M]
        """

        dadz = dReLU(self.weights @ a_down + self.bias)
        dCdz = dadz * dCda
        dCdW = np.transpose(np.outer(a_down, dCdz))

        self.W_grads.append(dCdW)
        self.b_grads.append(dCdz)

        dCda_down = np.transpose(self.weights) @ dCdz

        return dCda_down
    
    def update(self, learning_rate):
        """
        Gradient Descent Update
            1. collect gradients (calculated over batch of training examples) in self.F_grads, self.b_grads 
            2. stack into array and reset
            3. average over all examples (first axis of array)
            4. use to update parameters
        """

        # weights update
        W_grad_array = np.stack(self.W_grads)
        self.W_grads = []
        W_grad = np.mean(W_grad_array, axis=0)
        self.weights -= learning_rate * W_grad

        # bias update
        b_grad_array = np.stack(self.b_grads)
        self.b_grads = []
        b_grad = np.mean(b_grad_array, axis=0)
        self.bias -= learning_rate * b_grad



class PoolLayer:
    """
    Max-Pooling Layer L
    Attributes:
        patch size = stride size
    """
    def __init__(self, size):
        self.size = size
        self.mask = None

    def forward(self, input):
        """
        Forward Propagation
            Input: layer L-1 activations tensor size=[C,M,M]
                C = number of channels
                M = size of each channel (square)
            Output: layer L max-pooling tensor size=[C,N,N]
                N = size of each pooled channel (square)
        """

        C = np.shape(input)[0]
        M = np.shape(input)[1]
        if M % self.size != 0:
            print("ERROR: Input width and height must be a multiple of MaxPool size.")
            return None

        P = int(M/self.size)

        pool_channels = []
        masks = []
        for y in range(C):
            channel = input[y,:,:]
            mask_y = np.zeros((M,M))
            pool_y = np.zeros((P,P))
            for i in range(P):
                for j in range(P):
                    patch = channel[self.size*i:self.size*(i+1),self.size*j:self.size*(j+1)]
                    mask_index = np.zeros((self.size**2))
                    pool_y[i,j] = np.max(patch)
                    mask_index[np.argmax(patch)] = 1
                    mask_patch = np.reshape(mask_index, (self.size,self.size))
                    mask_y[self.size*i:self.size*(i+1),self.size*j:self.size*(j+1)] = mask_patch
            pool_channels.append(pool_y)
            masks.append(mask_y)
        self.mask = np.stack(masks)
        pool = np.stack(pool_channels)

        return pool
    
    def backward(self, dCdZ):
        dCdZ_2 = np.repeat(np.repeat(dCdZ, 2, axis=1), 2, axis=2)
        return dCdZ_2 * self.mask
    


class PoolLayer2:
    def __init__(self, size):
        self.size = size
        self.mask = None

    def forward(self, input):

        s = self.size
        C,M,N = np.shape(input)
        if M % s != 0 or N % s != 0:
            print("ERROR: Input width and height must be a multiple of MaxPool size.")
            return None

        P = int(M/s)
        Q = int(N/s)

        mask = np.zeros((C,M,N))
        pool_channels = []
        for y in range(C):
            channel = input[y,:,:]
            pool = np.zeros((P,Q))
            for i in range(P):
                for j in range(Q):
                    patch = channel[s*i:s*(i+1),s*j:s*(j+1)]
                    pool[i,j] = patch.max()
                    index = np.argwhere(patch==patch.max())
                    mask[y,s*i+index[0][0],s*j+index[0][1]] = 1
            pool_channels.append(pool)
        self.mask = mask

        return np.stack(pool_channels)
    
    def backward(self, dCdZ):
        dCdZ_2 = np.repeat(np.repeat(dCdZ, 2, axis=1), 2, axis=2)
        return dCdZ_2 * self.mask