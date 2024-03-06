import numpy as np
from PIL import Image
from pathlib import Path
import random

def load(trainfolder, testfolder):
    """
    "trainfolder" and "testfolder" are strings denoting the folder containing the 0-9 image folders (from the current working directory).

    Returns MNIST data as the tuple (train_data, test_data).
    "train_data" is a list of tuples (X,Y), where X is a (1,32,32) numpy array of the image (normalised and padded) and Y is a (10,1) numpy array of the label.
    "test_data" is a list of tuples (X,Y), where X is a (1,32,32) numpy array of the image (normalised and padded) and Y is a scalar label.
    """

    ### LOAD TRAINING DATA
    train_folders = []
    for digit in range(10):
        folder = Path.cwd() / trainfolder / str(digit)
        train_folders.append(folder)

    train_data = []
    labels = np.identity(10)

    for digit, folder in enumerate(train_folders):
        print(f"loading training data #{digit}")
        for image_file in folder.glob("*.png"):
            image = np.asarray(Image.open(image_file))
            image = image * 0.99/255 + 0.01
            image = np.expand_dims(np.pad(image, 2, constant_values=0), axis=0)
            data = (image, labels[digit])
            train_data.append(data)

    ### LOAD TESTING DATA
    test_folders = []
    for digit in range(10):
        folder = Path.cwd() / testfolder / str(digit)
        test_folders.append(folder)

    test_data = []
    print(f"loading test data")
    for digit, folder in enumerate(test_folders):
        for image_file in folder.glob("*.png"):
            image = np.asarray(Image.open(image_file))
            image = image * 0.99/255 + 0.01
            image = np.expand_dims(np.pad(image, 2, constant_values=0), axis=0)
            data = (image, digit)
            test_data.append(data)

    return (train_data, test_data)

def batch(data, batch_size):
    n = len(data)
    random.shuffle(data)
    batches = [data[k:k+batch_size] for k in range(0, n, batch_size)]
    return batches

def display(image):
    """
    Take numpy array ''image'' and display as heatmap, along with label
    """
    return None

