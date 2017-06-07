import numpy as np # For array
import struct # For load struct of MINIST data


#########################################################################
### Load MNIST data
def getMNISTdata(folder_location):
    dataset = ['train', 't10k']
    image_extension = '-images.idx3-ubyte'
    label_extension = '-labels.idx1-ubyte'
    
    # Load images data for training and testing
    for data in dataset:
        with open(folder_location + data + image_extension, 'rb') as limage:
            data_description = struct.unpack(">IIII", limage.read(16))
            if data == 'train':
                X_train = np.fromfile(limage, dtype=np.uint8)\
                            .reshape(data_description[1], 784)
            elif data == 't10k':
                X_test = np.fromfile(limage, dtype=np.uint8)\
                            .reshape(data_description[1], 784)
    
    # Load labels data for training and testing
    for label in dataset:
        with open(folder_location + label + label_extension, 'rb') as llabel:
            data_description = struct.unpack('>II', llabel.read(8))
            if label == 'train':
                y_train = np.fromfile(llabel, dtype=np.uint8)
            elif label == 't10k':
                y_test = np.fromfile(llabel, dtype=np.uint8)
    
return X_train, X_test, y_train, y_test


###########################################################################
folder_location = '/home/TrisZaska/Downloads/MNIST/'
X_train, X_test, y_train, y_test = getMNISTdata(folder_location)
