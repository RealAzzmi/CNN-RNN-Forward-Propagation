import numpy as np
from tensorflow import keras
from sklearn.metrics import f1_score
import os

class CNNCustomForwardPropagation:    
    def __init__(self, model_path: str, architecture_path: str):
        self.model_path = model_path
        self.architecture_path = architecture_path
        self.weights = {}
        self.biases = {}
        self.architecture = None
        self.load_model()
        
    def load_model(self):
        # Opening the compiled model architecture
        with open(self.architecture_path, 'r') as f:
            architecture_json = f.read()
        
        # Loading the weights and biases and extracting the weights & biases
        temp_model = keras.models.model_from_json(architecture_json)
        temp_model.load_weights(self.model_path)
        weighted_layer_count = 0
        for layer in temp_model.layers:
            if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                weights = layer.get_weights()
                # To check if it's convolutional or dense layer
                # In other words, check that it's not a pooling, dropout, or other type of layers that don't have weights/biases
                if len(weights) == 2:
                    self.weights[weighted_layer_count] = weights[0]
                    self.biases[weighted_layer_count] = weights[1]
                weighted_layer_count += 1
                
        # Storing it for forward pass later
        self.architecture = temp_model.get_config()

    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def convolutional_2d_forward(self, input_data, weights, bias, stride=1, padding='same'):
        batch_size, in_h, in_w, in_c = input_data.shape
        filter_h, filter_w, _, out_c = weights.shape
        
        # 1. Calculate the output dimensions and padding and apply padding if required
        if padding == 'same':
            out_h = in_h // stride
            out_w = in_w // stride
            pad_h = max(0, (out_h - 1) * stride + filter_h - in_h) // 2
            pad_w = max(0, (out_w - 1) * stride + filter_w - in_w) // 2
        elif padding == 'valid':
            out_h = (in_h - filter_h) // stride + 1
            out_w = (in_w - filter_w) // stride + 1
            pad_h = pad_w = 0
        else:
            print("Unknown padding")
            assert(False)

        if pad_h > 0 or pad_w > 0:
            input_data = np.pad(input_data, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        
        # 2. Calculate the result of the operation
        result = np.zeros((batch_size, out_h, out_w, out_c))
        for b in range(batch_size):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    h_end = h_start + filter_h
                    w_start = w * stride
                    w_end = w_start + filter_w
                    
                    input_slice = input_data[b, h_start:h_end, w_start:w_end, :]
                    for c in range(out_c):
                        result[b, h, w, c] = np.sum(input_slice * weights[:, :, :, c]) + bias[c]
        
        return result
    
    def max_pooling_2d_forward(self, input_data, pool_size=2, stride=2):
        # 1. Calculate the output dimensions
        batch_size, in_h, in_w, in_c = input_data.shape
        out_h = in_h // stride
        out_w = in_w // stride
        
        # 2. Calculate the result of the operation
        result = np.zeros((batch_size, out_h, out_w, in_c))
        for b in range(batch_size):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    
                    result[b, h, w, :] = np.max(input_data[b, h_start:h_end, w_start:w_end, :], axis=(0, 1))
        
        return result
    
    def average_pooling_2d_forward(self, input_data, pool_size=2, stride=2):
        # 1. Calculate the output dimensions
        batch_size, in_h, in_w, in_c = input_data.shape
        out_h = in_h // stride
        out_w = in_w // stride
        
        # 2. Calculate the result of the operation
        result = np.zeros((batch_size, out_h, out_w, in_c))
        for b in range(batch_size):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    
                    result[b, h, w, :] = np.mean(input_data[b, h_start:h_end, w_start:w_end, :], axis=(0, 1))
        
        return result
    
    def dense_forward(self, input_data, weights, bias):
        return np.dot(input_data, weights) + bias
    
    def global_average_pooling_2d_forward(self, input_data):
        return np.mean(input_data, axis=(1, 2))
    
    def flatten_forward(self, input_data):
        batch_size = input_data.shape[0]
        return input_data.reshape(batch_size, -1)
    
    def forward(self, x):
        current_output = x
        layer_idx = 0
        
        # Apply the current_output (initially x) to each layer in the architecture successively for each layer in the architecture
        for layer_config in self.architecture['layers']:
            layer_type = layer_config['class_name']
            
            if layer_type == 'Conv2D':
                if layer_idx in self.weights:
                    current_output = self.convolutional_2d_forward(current_output, self.weights[layer_idx], self.biases[layer_idx])
                    current_output = self.relu(current_output)
                    layer_idx += 1
            elif layer_type == 'MaxPooling2D':
                current_output = self.max_pooling_2d_forward(current_output)
            elif layer_type == 'AveragePooling2D':
                current_output = self.average_pooling_2d_forward(current_output)
            elif layer_type == 'GlobalAveragePooling2D':
                current_output = self.global_average_pooling_2d_forward(current_output)
            elif layer_type == 'Flatten':
                current_output = self.flatten_forward(current_output)
            elif layer_type == 'Dropout':
                pass
            elif layer_type == 'Dense':
                if layer_idx in self.weights:
                    current_output = self.dense_forward(current_output, self.weights[layer_idx], self.biases[layer_idx])
                    activation = layer_config['config'].get('activation', 'linear')
                    if activation == 'relu':
                        current_output = self.relu(current_output)
                    elif activation == 'softmax':
                        current_output = self.softmax(current_output)
                    layer_idx += 1
        
        return current_output
    
    def predict(self, x):
        return self.forward(x)


def test_cnn_custom_forward_propagation(model_name: str = "3_conv_layers", test_size: int = 1000):
    from load_cifar10 import LoadCifar10
    print("TESTING CUSTOM FORWARD PROPAGATION")
    
    data_loader = LoadCifar10().load_and_prepare_data()
    x_test_subset, y_test_subset = data_loader.get_test_subset(test_size)
    
    model_path = f"models/{model_name}.weights.h5"
    architecture_path = f"models/{model_name}.json"

    with open(architecture_path, 'r') as f:
        architecture_json = f.read()
    keras_model = keras.models.model_from_json(architecture_json)
    keras_model.load_weights(model_path)
    keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    keras_predictions = keras_model.predict(x_test_subset, verbose=0)
    keras_pred_classes = np.argmax(keras_predictions, axis=1)
    keras_f1 = f1_score(y_test_subset, keras_pred_classes, average='macro')
    
    custom_cnn = CNNCustomForwardPropagation(model_path, architecture_path)
    custom_predictions = custom_cnn.predict(x_test_subset)
    custom_pred_classes = np.argmax(custom_predictions, axis=1)
    custom_f1 = f1_score(y_test_subset, custom_pred_classes, average='macro')
    
    print(f"Keras F1 Score: {keras_f1:.4f}")
    print(f"Custom F1 Score: {custom_f1:.4f}")

    predictions_match = np.allclose(keras_predictions, custom_predictions, rtol=1e-5, atol=1e-6)
    if predictions_match:
        print("The custom forward propagation MATCH keras forward propagation.")
    else:
        print("The custom forward propagation DIDN'T MATCH keras forward propagation.")