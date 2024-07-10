# LeNet-5

LeNet-5, a convolutional neural network described in _Gradient-based Learning Applied to Document Recognition_ (LeCun et al, 1998), implemented in Python using only NumPy and SciPy, ie. without assistance from any deep learning packages such as PyTorch. The aim was to construct (and potentially train) a working model from first principles in order to gain a deeper understanding of the underlying mathematical processes.

An object-oriented approach was used, with classes for fully-connected, convolutional, and max-pooling layers, and the model itself. Each class has methods for forward/backward propagation and gradient update, with the necessary expressions for each determined by hand and hard-coded.

Earlier versions had a self-coded convolution function, however for efficiency it was decided to use SciPy's signal.correlate function instead as the first version was too computationally expensive.