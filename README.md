LeNet-5 convolutional neural network as based on "Gradient-based Learning Applied to Document Recognition" (Y.LeCun, L.Bottou, Y.Bengio, P.Haffner; 1998).

Object-oriented programming in Python without the use of deep learning packages (ie. no PyTorch etc.). Packages used for calculations limited to NumPy and SciPy.

No autograd or similar, gradient expressions determined by hand and backpropagation coded as layer class methods in the same way as the forward propagation usually is.

Fully-connected, convolutional, and max-pooling layers scripted in layers.py module and imported into main.py.

MNIST dataset loaded from .png images (function imported from MNIST.py).

Uses ReLU instead of sigmoid activation function used in original paper (coded in layer classes).

Loss function is mean-squared error (MSE) as in original paper.

Uses Kaiming He initialisation method to initialise weights/filters.

Quite slow on available hardware (CPU) and so no optimization attempted besides a little adjusing to the batch size. Stil reasonably accurate despite this (~84%).