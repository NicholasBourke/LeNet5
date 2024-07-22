# LeNet-5

LeNet-5, a convolutional neural network described in _Gradient-based Learning Applied to Document Recognition_ (LeCun et al, 1998), implemented in Python using only NumPy and SciPy, ie. without assistance from any deep learning packages such as PyTorch. The aim was to construct (and potentially train) a working model from first principles in order to gain a deeper understanding of the underlying mathematical processes.

An object-oriented approach was used, with classes for fully-connected, convolutional, and max-pooling layers, and the model itself. Each class has methods for forward/backward propagation and gradient update, with the necessary expressions for each determined by hand and hard-coded.

The first version had a self-coded convolution function, however for computational efficiency it was decided to use SciPy's signal.correlate function instead.

The model has recently been extensively rewritten to be cleaner, more interpretable, and hopefully more efficient. This current version is yet to be trained (or as yet, fully tested), however the previous version achieved an accuracy of 83.7% on the MNIST dataset with almost no hyperparameter fine-tuning. The intention is to train the current version after more careful hyperparameter selection to attain a higher accuracy.