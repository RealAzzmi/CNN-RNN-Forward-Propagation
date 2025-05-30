import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from neural_network import (
    NeuralNetwork, sigmoid, relu, tanh, softmax, linear,
    sigmoid_derivative, relu_derivative, tanh_derivative, 
    softmax_derivative, linear_derivative
)

class EmbeddingLayer: 
    def __init__(self, vocab_size, embedding_dim, mask_zero=True):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.mask_zero = mask_zero
        
        self.weights = self._initialize_weights()
        
        self.input_shape = None
        self.output_shape = None
        self.input_ids = None
        self.output = None
    
    def _initialize_weights(self):
        return np.random.uniform(-0.05, 0.05, (self.vocab_size, self.embedding_dim)).astype(np.float32)
        
    def load_weights(self, weights):
        self.weights = None
        self.weights = weights.copy().astype(np.float32)
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        batch_size, seq_length = x.shape
        self.input_shape = x.shape

        embeddings = np.zeros((batch_size, seq_length, self.embedding_dim), dtype=np.float32)

        for i in range(batch_size):
            for j in range(seq_length):
                token_id = int(x[i, j])
                if (not self.mask_zero or token_id != 0) and 0 <= token_id < self.vocab_size:
                    embeddings[i, j] = self.weights[token_id]

        self.input_ids = x
        self.output = embeddings
        self.output_shape = embeddings.shape
        return embeddings
    
    def backward(self, grad_output):
        grad_weights = np.zeros_like(self.weights, dtype=np.float32)
        batch_size, seq_length, _ = grad_output.shape
        
        for i in range(batch_size):
            for j in range(seq_length):
                token_id = int(self.input_ids[i, j])
                if token_id == 0 and self.mask_zero:
                    continue
                if 0 <= token_id < self.vocab_size:
                    grad_weights[token_id] += grad_output[i, j]
        
        return grad_weights

class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, activation='tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        limit = np.sqrt(6 / (input_size + hidden_size))
        self.W_ih = np.random.uniform(-limit, limit, (input_size, hidden_size)).astype(np.float32)
        self.W_hh = np.random.uniform(-limit, limit, (hidden_size, hidden_size)).astype(np.float32)
        self.b = np.zeros((1, hidden_size), dtype=np.float32)

        self.cache = {}
        
    def load_weights(self, W_ih, W_hh, b):
        self.W_ih = W_ih.copy()
        self.W_hh = W_hh.copy()
        self.b = b.copy()
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2
    
    def forward_step(self, x_t, h_prev):
        linear = np.dot(x_t, self.W_ih) + np.dot(h_prev, self.W_hh) + self.b
        
        if self.activation == 'tanh':
            h_t = self.tanh(linear)
        else:
            h_t = linear
            
        self.cache = {
            'x_t': x_t,
            'h_prev': h_prev,
            'linear': linear,
            'h_t': h_t
        }
        
        return h_t
    
    def backward_step(self, grad_h_t):
        x_t = self.cache['x_t']
        h_prev = self.cache['h_prev']
        linear = self.cache['linear']
        
        if self.activation == 'tanh':
            grad_linear = grad_h_t * self.tanh_derivative(linear)
        else:
            grad_linear = grad_h_t
        
        grad_W_ih = np.dot(x_t.T, grad_linear)
        grad_W_hh = np.dot(h_prev.T, grad_linear)
        grad_b = np.sum(grad_linear, axis=0)
        
        grad_x_t = np.dot(grad_linear, self.W_ih.T)
        grad_h_prev = np.dot(grad_linear, self.W_hh.T)
        
        return grad_x_t, grad_h_prev, grad_W_ih, grad_W_hh, grad_b

class SimpleRNNLayer:
    def __init__(self, input_size, hidden_size, return_sequences=False, 
                 bidirectional=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        
        self.forward_cell = SimpleRNNCell(input_size, hidden_size)
        
        if bidirectional:
            self.backward_cell = SimpleRNNCell(input_size, hidden_size)
        else:
            self.backward_cell = None
        
        self.cache = {}
        
    def load_weights(self, weights_dict):
        if self.bidirectional:
            self.forward_cell.load_weights(
                weights_dict['forward_W_ih'],
                weights_dict['forward_W_hh'],
                weights_dict['forward_b']
            )
            self.backward_cell.load_weights(
                weights_dict['backward_W_ih'],
                weights_dict['backward_W_hh'],
                weights_dict['backward_b']
            )
        else:
            self.forward_cell.load_weights(
                weights_dict['W_ih'],
                weights_dict['W_hh'],
                weights_dict['b']
            )
    
    def forward(self, x):
        if len(x.shape) == 2:
            batch_size, input_size = x.shape
            seq_length = 1
            x = x.reshape(batch_size, seq_length, input_size)
        else:
            batch_size, seq_length, input_size = x.shape
        
        h_forward = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        if self.bidirectional:
            h_backward = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        
        # Forward Direction
        forward_outputs = []
        for t in range(seq_length):
            x_t = x[:, t, :].astype(np.float32)  
            h_forward = self.forward_cell.forward_step(x_t, h_forward)
            forward_outputs.append(h_forward.copy())
        
        # Backward Direction (if bidirectional)
        if self.bidirectional:
            backward_outputs = []
            for t in range(seq_length-1, -1, -1):
                x_t = x[:, t, :].astype(np.float32)  
                h_backward = self.backward_cell.forward_step(x_t, h_backward)
                backward_outputs.append(h_backward.copy())
            backward_outputs.reverse()
        
        if self.return_sequences:
            if self.bidirectional:
                forward_stack = np.stack(forward_outputs, axis=1)  
                backward_stack = np.stack(backward_outputs, axis=1)  
                output = np.concatenate([forward_stack, backward_stack], axis=-1)
            else:
                output = np.stack(forward_outputs, axis=1)
        else:
            if self.bidirectional:
                output = np.concatenate([forward_outputs[-1], backward_outputs[0]], axis=-1)
            else:
                output = forward_outputs[-1]
        
        output = output.astype(np.float32)
        
        self.cache = {
            'x': x,
            'forward_outputs': forward_outputs,
        }
        if self.bidirectional:
            self.cache['backward_outputs'] = backward_outputs
        
        return output
    
    def backward(self, grad_output):
        x = self.cache['x']
        forward_outputs = self.cache['forward_outputs']
        batch_size, seq_length, input_size = x.shape

        grad_input = np.zeros_like(x)
        grad_W_ih_f = np.zeros_like(self.forward_cell.W_ih)
        grad_W_hh_f = np.zeros_like(self.forward_cell.W_hh)
        grad_b_f = np.zeros_like(self.forward_cell.b)
        grad_h_f = np.zeros((batch_size, self.hidden_size))

        if self.bidirectional:
            backward_outputs = self.cache['backward_outputs']
            grad_W_ih_b = np.zeros_like(self.backward_cell.W_ih)
            grad_W_hh_b = np.zeros_like(self.backward_cell.W_hh)
            grad_b_b = np.zeros_like(self.backward_cell.b)
            grad_h_b = np.zeros((batch_size, self.hidden_size))

        if self.return_sequences:
            if self.bidirectional:
                grad_forward = grad_output[:, :, :self.hidden_size]
                grad_backward = grad_output[:, :, self.hidden_size:]
            else:
                grad_forward = grad_output
        else:
            if self.bidirectional:
                grad_forward = np.zeros((batch_size, seq_length, self.hidden_size))
                grad_forward[:, -1, :] = grad_output[:, :self.hidden_size]
                grad_backward = np.zeros((batch_size, seq_length, self.hidden_size))
                grad_backward[:, 0, :] = grad_output[:, self.hidden_size:]
            else:
                grad_forward = np.zeros((batch_size, seq_length, self.hidden_size))
                grad_forward[:, -1, :] = grad_output

        for t in reversed(range(seq_length)):
            h_t = forward_outputs[t]
            h_prev = forward_outputs[t-1] if t > 0 else np.zeros_like(h_t)
            x_t = x[:, t, :]

            self.forward_cell.cache = {
                'x_t': x_t,
                'h_prev': h_prev,
                'linear': np.dot(x_t, self.forward_cell.W_ih) +
                          np.dot(h_prev, self.forward_cell.W_hh) +
                          self.forward_cell.b,
                'h_t': h_t
            }

            grad_h_t = grad_forward[:, t, :] + grad_h_f
            dx, dh_prev, dW_ih, dW_hh, db = self.forward_cell.backward_step(grad_h_t)

            grad_input[:, t, :] += dx
            grad_h_f = dh_prev
            grad_W_ih_f += dW_ih
            grad_W_hh_f += dW_hh
            grad_b_f += db

        if self.bidirectional:
            for t in range(seq_length):
                h_t = backward_outputs[t]
                h_prev = backward_outputs[t-1] if t > 0 else np.zeros_like(h_t)
                x_t = x[:, seq_length - 1 - t, :]  

                self.backward_cell.cache = {
                    'x_t': x_t,
                    'h_prev': h_prev,
                    'linear': np.dot(x_t, self.backward_cell.W_ih) +
                              np.dot(h_prev, self.backward_cell.W_hh) +
                              self.backward_cell.b,
                    'h_t': h_t
                }

                grad_h_t = grad_backward[:, seq_length - 1 - t, :] + grad_h_b
                dx, dh_prev, dW_ih, dW_hh, db = self.backward_cell.backward_step(grad_h_t)

                grad_input[:, seq_length - 1 - t, :] += dx
                grad_h_b = dh_prev
                grad_W_ih_b += dW_ih
                grad_W_hh_b += dW_hh
                grad_b_b += db

        grad_weights = {
            'forward_W_ih': grad_W_ih_f,
            'forward_W_hh': grad_W_hh_f,
            'forward_b': grad_b_f
        }

        if self.bidirectional:
            grad_weights.update({
                'backward_W_ih': grad_W_ih_b,
                'backward_W_hh': grad_W_hh_b,
                'backward_b': grad_b_b
            })

        return grad_input, grad_weights

class DropoutLayer:
    def __init__(self, dropout_rate=0.0):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, x):
        if self.training and self.dropout_rate > 0:
            self.mask = np.random.binomial(1, 1-self.dropout_rate, x.shape) / (1-self.dropout_rate)
            return x * self.mask
        return x
    
    def backward(self, grad_output):
        if self.training and self.dropout_rate > 0:
            return grad_output * self.mask
        return grad_output
    
    def set_training(self, training):
        self.training = training

class DenseLayer:
    def __init__(self, layer_sizes, activation_functions, output_activation):
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.output_activation = output_activation
        
        from neural_network import mean_squared_error
        
        self.nn = NeuralNetwork(
            layer_sizes=layer_sizes,
            hidden_layer_activation_functions=activation_functions,
            output_layer_activation_function=output_activation,
            loss_function=mean_squared_error,
            initialization_method="xavier",
            learning_rate=0.001,
            max_iter=1,
            batch_size=32,
            verbose=False
        )
        
        self.input_shape = None
        self.original_input_shape = None
        self.cache = {}
        
    def load_keras_weights(self, keras_weights_list):
        layer_idx = 1
        
        for weights, biases in keras_weights_list:
            if layer_idx < len(self.nn.weights):
                self.nn.weights[layer_idx] = weights.copy()
                self.nn.biases[layer_idx] = biases.reshape(1, -1).copy()
                layer_idx += 1
    
    def forward(self, x):
        self.original_input_shape = x.shape
        
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        
        self.input_shape = x.shape
        
        preactivations, activations = self.nn.forward_propagation(x)
        
        self.cache = {
            'input': x,
            'preactivations': preactivations,
            'activations': activations,
            'original_input_shape': self.original_input_shape
        }
        
        return activations[-1]
    
    def backward(self, grad_output):
        input_data = self.cache['input']
        preactivations = self.cache['preactivations']
        activations = self.cache['activations']
        
        current_grad = grad_output
        weight_gradients = {}
        
        num_layers = len(self.layer_sizes) - 1
        
        for layer_idx in range(num_layers, 0, -1):
            W = self.nn.weights[layer_idx]
            b = self.nn.biases[layer_idx]
            
            if layer_idx == num_layers:
                if self.output_activation == softmax:
                    dZ = current_grad
                else:
                    if self.output_activation == sigmoid:
                        dZ = current_grad * sigmoid_derivative(preactivations[layer_idx])
                    elif self.output_activation == tanh:
                        dZ = current_grad * tanh_derivative(preactivations[layer_idx])
                    elif self.output_activation == relu:
                        dZ = current_grad * relu_derivative(preactivations[layer_idx])
                    else:  
                        dZ = current_grad
            else:
                activation_func = self.activation_functions[layer_idx - 1]
                
                if activation_func == relu:
                    dZ = current_grad * relu_derivative(preactivations[layer_idx])
                elif activation_func == tanh:
                    dZ = current_grad * tanh_derivative(preactivations[layer_idx])
                elif activation_func == sigmoid:
                    dZ = current_grad * sigmoid_derivative(preactivations[layer_idx])
                else:  
                    dZ = current_grad
            
            A_prev = activations[layer_idx - 1]
            
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            
            weight_gradients[f'layer_{layer_idx}_weights'] = dW
            weight_gradients[f'layer_{layer_idx}_biases'] = db
            
            if layer_idx > 1:
                current_grad = np.dot(dZ, W.T)
            else:
                grad_input = np.dot(dZ, W.T)
        
        if len(self.cache['original_input_shape']) > 2:
            grad_input = grad_input.reshape(self.cache['original_input_shape'])
        
        return grad_input, weight_gradients

class RNNFromScratch:
    def __init__(self):
        self.layers = []
        self.layer_types = []
        self.training = True
        
    def add_embedding(self, vocab_size, embedding_dim, mask_zero=True):
        layer = EmbeddingLayer(vocab_size, embedding_dim, mask_zero)
        self.layers.append(layer)
        self.layer_types.append('embedding')
        
    def add_rnn(self, input_size, hidden_size, return_sequences=False, bidirectional=False):
        layer = SimpleRNNLayer(input_size, hidden_size, return_sequences, bidirectional)
        self.layers.append(layer)
        self.layer_types.append('rnn')
        
    def add_dropout(self, dropout_rate):
        layer = DropoutLayer(dropout_rate)
        self.layers.append(layer)
        self.layer_types.append('dropout')
        
    def add_dense(self, layer_sizes, activation_functions, output_activation):
        layer = DenseLayer(layer_sizes, activation_functions, output_activation)
        self.layers.append(layer)
        self.layer_types.append('dense')
    
    def forward(self, x, batch_size=None):
        if batch_size is not None and len(x) > batch_size:
            outputs = []
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                batch_output = self._forward_single_batch(batch_x)
                outputs.append(batch_output)
            return np.vstack(outputs)
        else:
            return self._forward_single_batch(x)
    
    def _forward_single_batch(self, x):
        current_input = x
        
        for layer, layer_type in zip(self.layers, self.layer_types):
            if layer_type == 'dropout':
                layer.set_training(self.training)
            current_input = layer.forward(current_input)
            
        return current_input
    
    def backward(self, grad_output):
        gradients = {}
        current_grad = grad_output
        
        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            layer_type = self.layer_types[i]
            
            if layer_type == 'dense':
                grad_input, grad_weights = layer.backward(current_grad)
                gradients[f'layer_{i}'] = grad_weights
                current_grad = grad_input
                
            elif layer_type == 'dropout':
                current_grad = layer.backward(current_grad)
                
            elif layer_type == 'rnn':
                grad_input, grad_weights = layer.backward(current_grad)
                gradients[f'layer_{i}'] = grad_weights
                current_grad = grad_input
                
            elif layer_type == 'embedding':
                grad_weights = layer.backward(current_grad)
                gradients[f'layer_{i}_weights'] = grad_weights
                current_grad = None
                break
        
        return gradients
    
    def update_weights(self, gradients, learning_rate=0.01):
        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == 'embedding':
                if f'layer_{i}_weights' in gradients:
                    layer.weights -= learning_rate * gradients[f'layer_{i}_weights']
                    
            elif layer_type == 'rnn':
                if f'layer_{i}' in gradients:
                    grad_dict = gradients[f'layer_{i}']
                    if 'forward_W_ih' in grad_dict:
                        layer.forward_cell.W_ih -= learning_rate * grad_dict['forward_W_ih']
                        layer.forward_cell.W_hh -= learning_rate * grad_dict['forward_W_hh']
                        layer.forward_cell.b -= learning_rate * grad_dict['forward_b']
                    if 'backward_W_ih' in grad_dict:
                        layer.backward_cell.W_ih -= learning_rate * grad_dict['backward_W_ih']
                        layer.backward_cell.W_hh -= learning_rate * grad_dict['backward_W_hh']
                        layer.backward_cell.b -= learning_rate * grad_dict['backward_b']
                    if 'W_ih' in grad_dict:  
                        layer.forward_cell.W_ih -= learning_rate * grad_dict['W_ih']
                        layer.forward_cell.W_hh -= learning_rate * grad_dict['W_hh']
                        layer.forward_cell.b -= learning_rate * grad_dict['b']
                        
            elif layer_type == 'dense':
                if f'layer_{i}' in gradients:
                    grad_dict = gradients[f'layer_{i}']
                    for layer_idx in range(1, len(layer.nn.weights)):
                        if f'layer_{layer_idx}_weights' in grad_dict:
                            layer.nn.weights[layer_idx] -= learning_rate * grad_dict[f'layer_{layer_idx}_weights']
                        if f'layer_{layer_idx}_biases' in grad_dict:
                            layer.nn.biases[layer_idx] -= learning_rate * grad_dict[f'layer_{layer_idx}_biases']
    
    def train_step(self, x_batch, y_batch, learning_rate=0.001):
        # Forward 
        predictions = self.forward(x_batch)
        
        # Compute Loss 
        epsilon = 1e-12
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        if len(y_batch.shape) == 1:
            num_classes = predictions.shape[1]
            y_one_hot = np.eye(num_classes)[y_batch]
        else:
            y_one_hot = y_batch
            
        loss = -np.mean(np.sum(y_one_hot * np.log(predictions), axis=1))
        
        grad_output = (predictions - y_one_hot) / len(x_batch)
        
        # Backward 
        gradients = self.backward(grad_output)
        
        # Update Weights
        self.update_weights(gradients, learning_rate)
        
        return loss
    
    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=10, 
            batch_size=32, learning_rate=0.001, verbose=True):
        history = {'loss': [], 'val_loss': []}
        
        self.set_training(True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Train on Batches
            for i in range(0, len(x_train), batch_size):
                batch_x = x_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]
                
                batch_loss = self.train_step(batch_x, batch_y, learning_rate)
                epoch_loss += batch_loss
                num_batches += 1
            
            epoch_loss /= num_batches
            history['loss'].append(epoch_loss)
            
            # Validation Loss
            if x_val is not None and y_val is not None:
                self.set_training(False)
                val_predictions = self.predict(x_val, batch_size)
                
                epsilon = 1e-12
                val_predictions = np.clip(val_predictions, epsilon, 1 - epsilon)
                
                if len(y_val.shape) == 1:
                    num_classes = val_predictions.shape[1]
                    y_val_one_hot = np.eye(num_classes)[y_val]
                else:
                    y_val_one_hot = y_val
                    
                val_loss = -np.mean(np.sum(y_val_one_hot * np.log(val_predictions), axis=1))
                history['val_loss'].append(val_loss)
                
                self.set_training(True)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
        return history
    
    def set_training(self, training=True):
        self.training = training
        for layer, layer_type in zip(self.layers, self.layer_types):
            if layer_type == 'dropout':
                layer.set_training(training)
    
    def predict(self, x, batch_size=32):
        self.set_training(False)
        predictions = self.forward(x, batch_size)
        return predictions
    
    def evaluate(self, x_test, y_test, batch_size=32):
        predictions = self.predict(x_test, batch_size)
        y_pred = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'predictions': y_pred,
            'probabilities': predictions
        }

def extract_keras_architecture(keras_model):
    architecture = {
        'layers': [],
        'total_layers': len(keras_model.layers),
        'input_shape': getattr(keras_model, 'input_shape', None),
        'output_shape': getattr(keras_model, 'output_shape', None)
    }
    
    for i, layer in enumerate(keras_model.layers):
        try:
            input_shape = layer.input_shape
        except (AttributeError, RuntimeError):
            input_shape = getattr(layer, 'input_spec', None)
            if input_shape and hasattr(input_shape, 'shape'):
                input_shape = input_shape.shape
            else:
                input_shape = None
        
        try:
            output_shape = layer.output_shape
        except (AttributeError, RuntimeError):
            output_shape = None
        
        try:
            trainable_params = layer.count_params()
        except:
            trainable_params = 0
            
        layer_info = {
            'index': i,
            'name': layer.name,
            'type': type(layer).__name__,
            'config': layer.get_config(),
            'input_shape': input_shape,
            'output_shape': output_shape,
            'trainable_params': trainable_params
        }
        
        if 'Embedding' in layer_info['type']:
            layer_info['vocab_size'] = layer_info['config']['input_dim']
            layer_info['embedding_dim'] = layer_info['config']['output_dim']
            layer_info['mask_zero'] = layer_info['config'].get('mask_zero', False)
            
        elif 'Bidirectional' in layer_info['type']:
            layer_info['bidirectional'] = True
            layer_info['rnn_type'] = type(layer.forward_layer).__name__
            layer_info['units'] = layer.forward_layer.units
            layer_info['return_sequences'] = layer.forward_layer.return_sequences
            
        elif layer_info['type'] in ['SimpleRNN']:
            layer_info['bidirectional'] = False
            layer_info['rnn_type'] = layer_info['type']
            layer_info['units'] = layer_info['config']['units']
            layer_info['return_sequences'] = layer_info['config'].get('return_sequences', False)
            
        elif 'Dense' in layer_info['type']:
            layer_info['units'] = layer_info['config']['units']
            layer_info['activation'] = layer_info['config']['activation']
            
        elif 'Dropout' in layer_info['type']:
            layer_info['rate'] = layer_info['config']['rate']
            
        architecture['layers'].append(layer_info)
    
    return architecture

def build_scratch_model_from_keras_auto(keras_model):
    architecture = extract_keras_architecture(keras_model)
    scratch_model = RNNFromScratch()
    
    print("Keras Model Architecture:")
    for layer_info in architecture['layers']:
        print(f"  {layer_info['index']}: {layer_info['type']} - {layer_info['name']}")
    
    dense_layers_info = [] 
    current_size = None
    
    for layer_info in architecture['layers']:
        layer_type = layer_info['type']
        
        if 'Embedding' in layer_type:
            scratch_model.add_embedding(
                vocab_size=layer_info['vocab_size'],
                embedding_dim=layer_info['embedding_dim'],
                mask_zero=layer_info['mask_zero']
            )
            current_size = layer_info['embedding_dim']
            print(f"  Added embedding: {layer_info['vocab_size']} -> {layer_info['embedding_dim']}")
            
        elif 'Bidirectional' in layer_type or layer_type in ['SimpleRNN']:
            is_bidirectional = layer_info.get('bidirectional', False)
            units = layer_info['units']
            return_sequences = layer_info.get('return_sequences', False)
            
            scratch_model.add_rnn(
                input_size=current_size,
                hidden_size=units,
                return_sequences=return_sequences,
                bidirectional=is_bidirectional
            )
            
            current_size = units * 2 if is_bidirectional else units
            print(f"  Added RNN: {units} units, bidirectional={is_bidirectional}, current_size={current_size}")
            
        elif 'Dropout' in layer_type:
            dropout_rate = layer_info['rate']
            scratch_model.add_dropout(dropout_rate)
            print(f"  Added dropout: {dropout_rate}")
            
        elif 'Dense' in layer_type:
            dense_layers_info.append({
                'units': layer_info['units'],
                'activation': layer_info['activation'],
                'input_size': current_size
            })
            current_size = layer_info['units']
    
    if dense_layers_info:
        print(f"  Building dense block with {len(dense_layers_info)} layers...")
        
        layer_sizes = [dense_layers_info[0]['input_size']]
        activation_functions = []
        
        for i, dense_info in enumerate(dense_layers_info):
            layer_sizes.append(dense_info['units'])
            print(f"    Dense layer {i+1}: {dense_info['input_size']} -> {dense_info['units']} ({dense_info['activation']})")
            
            activation_name = dense_info['activation']
            if activation_name == 'relu':
                activation_func = relu
            elif activation_name == 'tanh':
                activation_func = tanh
            elif activation_name == 'sigmoid':
                activation_func = sigmoid
            elif activation_name in ['softmax', 'linear']:
                activation_func = linear  # Handle output layer separately
            else:
                activation_func = linear
            
            if i < len(dense_layers_info) - 1:  # Hidden layers only
                activation_functions.append(activation_func)
        
        output_activation_name = dense_layers_info[-1]['activation']
        if output_activation_name == 'softmax':
            output_activation = softmax
        elif output_activation_name == 'sigmoid':
            output_activation = sigmoid
        elif output_activation_name == 'relu':
            output_activation = relu
        elif output_activation_name == 'tanh':
            output_activation = tanh
        else:
            output_activation = linear
        
        print(f"    Layer sizes: {layer_sizes}")
        print(f"    Hidden activations: {[f.__name__ if hasattr(f, '__name__') else str(f) for f in activation_functions]}")
        print(f"    Output activation: {output_activation.__name__ if hasattr(output_activation, '__name__') else str(output_activation)}")
        
        scratch_model.add_dense(layer_sizes, activation_functions, output_activation)
    
    print("Loading weights...")
    load_keras_weights_auto(scratch_model, keras_model, architecture)
    
    print("Scratch model building completed!")
    return scratch_model

def load_keras_weights_auto(scratch_model, keras_model, architecture):
    keras_layer_idx = 0
    scratch_layer_idx = 0
    
    for arch_layer in architecture['layers']:
        if keras_layer_idx >= len(keras_model.layers):
            break
            
        keras_layer = keras_model.layers[keras_layer_idx]
        
        try:
            layer_weights = keras_layer.get_weights()
            if len(layer_weights) == 0:
                keras_layer_idx += 1
                continue
        except:
            keras_layer_idx += 1
            continue
            
        if scratch_layer_idx >= len(scratch_model.layers):
            break
            
        layer_type = arch_layer['type']
        scratch_layer = scratch_model.layers[scratch_layer_idx]
        scratch_layer_type = scratch_model.layer_types[scratch_layer_idx]
        
        try:
            if 'Embedding' in layer_type and scratch_layer_type == 'embedding':
                weights = layer_weights[0]
                scratch_layer.load_weights(weights)
                print(f"   Loaded embedding weights: {weights.shape}")
                
            elif ('Bidirectional' in layer_type or layer_type in ['SimpleRNN']) and scratch_layer_type == 'rnn':
                if hasattr(keras_layer, 'forward_layer'): 
                    try:
                        forward_weights = keras_layer.forward_layer.get_weights()
                        backward_weights = keras_layer.backward_layer.get_weights()
                        
                        weights_dict = {
                            'forward_W_ih': forward_weights[0],
                            'forward_W_hh': forward_weights[1],
                            'forward_b': forward_weights[2],
                            'backward_W_ih': backward_weights[0],
                            'backward_W_hh': backward_weights[1],
                            'backward_b': backward_weights[2],
                        }
                        print(f"   Loaded bidirectional RNN weights")
                    except Exception as e:
                        print(f"   Warning: Could not load bidirectional weights: {e}")
                        if len(layer_weights) >= 6:
                            weights_dict = {
                                'forward_W_ih': layer_weights[0],
                                'forward_W_hh': layer_weights[1],
                                'forward_b': layer_weights[2],
                                'backward_W_ih': layer_weights[3],
                                'backward_W_hh': layer_weights[4],
                                'backward_b': layer_weights[5],
                            }
                        else:
                            print(f"   Error: Insufficient weights for bidirectional layer")
                            keras_layer_idx += 1
                            scratch_layer_idx += 1
                            continue
                else:  
                    if len(layer_weights) >= 3:
                        weights_dict = {
                            'W_ih': layer_weights[0],
                            'W_hh': layer_weights[1],
                            'b': layer_weights[2],
                        }
                        print(f"   Loaded unidirectional RNN weights")
                    else:
                        print(f"   Error: Insufficient weights for RNN layer")
                        keras_layer_idx += 1
                        scratch_layer_idx += 1
                        continue
                
                scratch_layer.load_weights(weights_dict)
                
            elif 'Dense' in layer_type and scratch_layer_type == 'dense':
                dense_weights = []
                temp_keras_idx = keras_layer_idx
                
                while temp_keras_idx < len(keras_model.layers):
                    temp_layer = keras_model.layers[temp_keras_idx]
                    temp_layer_type = type(temp_layer).__name__
                    
                    if 'Dense' in temp_layer_type:
                        try:
                            temp_weights = temp_layer.get_weights()
                            if len(temp_weights) >= 2:  # weights and biases
                                dense_weights.append(temp_weights)
                                print(f"   Collected dense layer weights: {temp_weights[0].shape}")
                        except:
                            pass
                        temp_keras_idx += 1
                    elif 'Dropout' in temp_layer_type:
                        temp_keras_idx += 1  # Skip dropout layers
                    else:
                        break
                
                if dense_weights:
                    scratch_layer.load_keras_weights(dense_weights)
                    print(f"   Loaded {len(dense_weights)} dense layers")
                
                keras_layer_idx = temp_keras_idx - 1
                
        except Exception as e:
            print(f"   Warning: Error loading weights for layer {keras_layer_idx}: {e}")
        
        keras_layer_idx += 1
        scratch_layer_idx += 1
        
        while (scratch_layer_idx < len(scratch_model.layers) and 
               scratch_model.layer_types[scratch_layer_idx] == 'dropout'):
            scratch_layer_idx += 1

def compare_keras_vs_scratch(keras_model, scratch_model, test_data, test_labels, batch_size=32):
    """
    Compare predictions between Keras model and from-scratch implementation
    """
    print("Comparing Keras vs From-Scratch Implementation...")
    
    # Get Keras predictions
    keras_predictions = keras_model.predict(test_data, batch_size=batch_size, verbose=0)
    keras_pred_classes = np.argmax(keras_predictions, axis=1)
    
    # Convert test_data to numpy array if it's a dataset
    if hasattr(test_data, 'numpy'):
        # Handle tf.data.Dataset
        test_array = []
        for batch in test_data:
            test_array.append(batch[0].numpy())
        test_array = np.vstack(test_array)
    else:
        test_array = test_data
    
    # Get from-scratch predictions
    scratch_predictions = scratch_model.predict(test_array, batch_size=batch_size)
    scratch_pred_classes = np.argmax(scratch_predictions, axis=1)
    
    # Compare predictions
    keras_accuracy = accuracy_score(test_labels, keras_pred_classes)
    keras_f1 = f1_score(test_labels, keras_pred_classes, average='macro')
    
    scratch_accuracy = accuracy_score(test_labels, scratch_pred_classes)
    scratch_f1 = f1_score(test_labels, scratch_pred_classes, average='macro')
    
    # Check prediction similarity
    prediction_match = np.mean(keras_pred_classes == scratch_pred_classes)
    prob_mse = np.mean((keras_predictions - scratch_predictions) ** 2)
    
    print(f"\nKeras Model:")
    print(f"  Accuracy: {keras_accuracy:.4f}")
    print(f"  Macro F1: {keras_f1:.4f}")
    
    print(f"\nFrom-Scratch Model:")
    print(f"  Accuracy: {scratch_accuracy:.4f}")
    print(f"  Macro F1: {scratch_f1:.4f}")
    
    print(f"\nComparison:")
    print(f"  Prediction Match Rate: {prediction_match:.4f}")
    print(f"  Probability MSE: {prob_mse:.6f}")
    
    return {
        'keras': {'accuracy': keras_accuracy, 'macro_f1': keras_f1},
        'scratch': {'accuracy': scratch_accuracy, 'macro_f1': scratch_f1},
        'prediction_match_rate': prediction_match,
        'probability_mse': prob_mse
    }