import numpy as np
from tensorflow import keras
from config import VALIDATION_SIZE, RANDOM_SEED

class LoadCifar10:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
    
    def load_and_prepare_data(self):
        print("Loading the CIFAR-10 dataset from keras")
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalizing to [0, 1] cause the range is initially [0, 255] / a byte
        x_train_full = x_train_full.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatenning the labels
        y_train_full = y_train_full.flatten()
        y_test = y_test.flatten()
        
        # Splitting into training (training + validation) & testing
        np.random.seed(RANDOM_SEED)
        indices = np.random.permutation(len(x_train_full))
        
        val_indices = indices[:VALIDATION_SIZE]
        train_indices = indices[VALIDATION_SIZE:]
        
        self.x_train = x_train_full[train_indices]
        self.y_train = y_train_full[train_indices]
        self.x_val = x_train_full[val_indices]
        self.y_val = y_train_full[val_indices]
        self.x_test = x_test
        self.y_test = y_test
        
        assert(self.x_train.shape[0] == 40000)
        assert(self.x_val.shape[0] == 10000)
        assert(self.x_test.shape[0] == 10000)
        
        return self
    
    def get_data_splits(self):
        assert(self.x_train is not None)

        return {
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_val': self.x_val,
            'y_val': self.y_val,
            'x_test': self.x_test,
            'y_test': self.y_test
        }
    
    def get_test_subset(self, size=1000):
        assert(self.x_test is not None)

        return self.x_test[:size], self.y_test[:size]