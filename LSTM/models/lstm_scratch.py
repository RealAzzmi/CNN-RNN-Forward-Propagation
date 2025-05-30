import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import json
import os
from sklearn.metrics import f1_score, classification_report

class ActivationFunctions:
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class EmbeddingLayer:
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = None
    
    def set_weights(self, weights: np.ndarray):
        self.weights = weights
        assert self.weights.shape == (self.vocab_size, self.embedding_dim)
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Weights not set. Call set_weights() first.")
        
        # Simple embedding lookup
        embeddings = self.weights[input_ids]
        return embeddings

class LSTMCell:    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight matrices
        self.W_f = None  # Forget gate weights
        self.W_i = None  # Input gate weights
        self.W_c = None  # Candidate values weights
        self.W_o = None  # Output gate weights
        
        # Bias vectors
        self.b_f = None  # Forget gate bias
        self.b_i = None  # Input gate bias
        self.b_c = None  # Candidate values bias
        self.b_o = None  # Output gate bias
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        self.W_f = weights['W_f']
        self.W_i = weights['W_i']
        self.W_c = weights['W_c']
        self.W_o = weights['W_o']
        self.b_f = weights['b_f']
        self.b_i = weights['b_i']
        self.b_c = weights['b_c']
        self.b_o = weights['b_o']
    
    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Concatenate input and previous hidden state
        concat_input = np.concatenate([x_t, h_prev], axis=1)
        
        # Forget gate
        f_t = ActivationFunctions.sigmoid(np.dot(concat_input, self.W_f) + self.b_f)
        
        # Input gate
        i_t = ActivationFunctions.sigmoid(np.dot(concat_input, self.W_i) + self.b_i)
        
        # Candidate values
        c_tilde_t = ActivationFunctions.tanh(np.dot(concat_input, self.W_c) + self.b_c)
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde_t
        
        # Output gate
        o_t = ActivationFunctions.sigmoid(np.dot(concat_input, self.W_o) + self.b_o)
        
        # Update hidden state
        h_t = o_t * ActivationFunctions.tanh(c_t)
        
        return h_t, c_t

class LSTMLayer:    
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        self.forward_cell = LSTMCell(input_size, hidden_size)
        if bidirectional:
            self.backward_cell = LSTMCell(input_size, hidden_size)
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        # Forward cell weights
        forward_weights = {key: weights[f'forward_{key}'] for key in ['W_f', 'W_i', 'W_c', 'W_o', 'b_f', 'b_i', 'b_c', 'b_o']}
        self.forward_cell.set_weights(forward_weights)
        
        # Backward cell weights (for bidirectional)
        if self.bidirectional:
            backward_weights = {key: weights[f'backward_{key}'] for key in ['W_f', 'W_i', 'W_c', 'W_o', 'b_f', 'b_i', 'b_c', 'b_o']}
            self.backward_cell.set_weights(backward_weights)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        batch_size, seq_length, _ = x.shape
        
        # Forward pass
        h_forward = np.zeros((batch_size, self.hidden_size))
        c_forward = np.zeros((batch_size, self.hidden_size))
        
        forward_outputs = []
        for t in range(seq_length):
            h_forward, c_forward = self.forward_cell.forward_step(x[:, t, :], h_forward, c_forward)
            
            # Apply mask if provided
            if mask is not None:
                mask_t = mask[:, t:t+1]  # (batch_size, 1)
                h_forward = h_forward * mask_t
                c_forward = c_forward * mask_t
            
            forward_outputs.append(h_forward)
        
        forward_outputs = np.stack(forward_outputs, axis=1)  # (batch_size, seq_length, hidden_size)
        
        if not self.bidirectional:
            return forward_outputs
        
        # Backward pass
        h_backward = np.zeros((batch_size, self.hidden_size))
        c_backward = np.zeros((batch_size, self.hidden_size))
        
        backward_outputs = []
        for t in range(seq_length - 1, -1, -1):
            h_backward, c_backward = self.backward_cell.forward_step(x[:, t, :], h_backward, c_backward)
            
            # Apply mask if provided
            if mask is not None:
                mask_t = mask[:, t:t+1]  # (batch_size, 1)
                h_backward = h_backward * mask_t
                c_backward = c_backward * mask_t
            
            backward_outputs.append(h_backward)
        
        backward_outputs = np.stack(backward_outputs[::-1], axis=1)  # (batch_size, seq_length, hidden_size)
        
        # Concatenate forward and backward outputs
        outputs = np.concatenate([forward_outputs, backward_outputs], axis=2)
        
        return outputs

class DenseLayer:    
    def __init__(self, input_size: int, output_size: int, activation: str = 'linear'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        self.weights = None
        self.bias = None
    
    def set_weights(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias
        assert self.weights.shape == (self.input_size, self.output_size)
        assert self.bias.shape == (self.output_size,)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise ValueError("Weights not set. Call set_weights() first.")
        
        # Linear transformation
        output = np.dot(x, self.weights) + self.bias
        
        # Apply activation
        if self.activation == 'relu':
            output = ActivationFunctions.relu(output)
        elif self.activation == 'softmax':
            output = ActivationFunctions.softmax(output)
        
        return output

class LSTMScratchModel:    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize layers
        self.embedding = EmbeddingLayer(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim']
        )
        
        # LSTM layers
        self.lstm_layers = []
        input_size = config['embedding_dim']
        
        for i in range(config['num_lstm_layers']):
            lstm_layer = LSTMLayer(
                input_size=input_size,
                hidden_size=config['lstm_units'],
                bidirectional=config['bidirectional']
            )
            self.lstm_layers.append(lstm_layer)
            
            # Update input size for next layer
            input_size = config['lstm_units'] * (2 if config['bidirectional'] else 1)
        
        # Dense layers
        self.dense = DenseLayer(
            input_size=input_size,
            output_size=config['dense_units'],
            activation='relu'
        )
        
        self.output_layer = DenseLayer(
            input_size=config['dense_units'],
            output_size=config['num_classes'],
            activation='softmax'
        )
    
    def load_weights_from_keras(self, keras_model: tf.keras.Model):
        print("Loading weights from Keras model...")
        
        # Get all layer weights
        layer_weights = {}
        for layer in keras_model.layers:
            if hasattr(layer, 'get_weights') and layer.get_weights():
                layer_weights[layer.name] = layer.get_weights()
        
        # Load embedding weights
        if 'embedding' in layer_weights:
            self.embedding.set_weights(layer_weights['embedding'][0])
            print("✓ Embedding weights loaded")
        
        # Load LSTM weights
        lstm_layer_count = 0
        for layer_name, weights in layer_weights.items():
            if 'lstm' in layer_name.lower():
                self._load_lstm_weights(weights, lstm_layer_count)
                lstm_layer_count += 1
                print(f"✓ LSTM layer {lstm_layer_count} weights loaded")
        
        # Load dense layer weights
        if 'dense' in layer_weights:
            dense_weights, dense_bias = layer_weights['dense']
            self.dense.set_weights(dense_weights, dense_bias)
            print("✓ Dense layer weights loaded")
        
        # Load output layer weights
        if 'output' in layer_weights:
            output_weights, output_bias = layer_weights['output']
            self.output_layer.set_weights(output_weights, output_bias)
            print("✓ Output layer weights loaded")
    
    def _load_lstm_weights(self, weights: List[np.ndarray], layer_idx: int):
        if self.config['bidirectional']:
            self._load_bidirectional_lstm_weights(weights, layer_idx)
        else:
            self._load_unidirectional_lstm_weights(weights, layer_idx)
    
    def _load_unidirectional_lstm_weights(self, weights: List[np.ndarray], layer_idx: int):

        # Keras LSTM weights: [kernel, recurrent_kernel, bias]
        kernel, recurrent_kernel, bias = weights
        
        input_size = kernel.shape[0]
        hidden_size = kernel.shape[1] // 4  # 4 gates
        
        # Split kernel into gates (input, forget, cell, output)
        W_i_x = kernel[:, :hidden_size]
        W_f_x = kernel[:, hidden_size:2*hidden_size]
        W_c_x = kernel[:, 2*hidden_size:3*hidden_size]
        W_o_x = kernel[:, 3*hidden_size:]
        
        # Split recurrent kernel into gates
        W_i_h = recurrent_kernel[:, :hidden_size]
        W_f_h = recurrent_kernel[:, hidden_size:2*hidden_size]
        W_c_h = recurrent_kernel[:, 2*hidden_size:3*hidden_size]
        W_o_h = recurrent_kernel[:, 3*hidden_size:]
        
        # Split bias into gates
        b_i = bias[:hidden_size]
        b_f = bias[hidden_size:2*hidden_size]
        b_c = bias[2*hidden_size:3*hidden_size]
        b_o = bias[3*hidden_size:]
        
        # Combine input and hidden weights for each gate
        lstm_weights = {
            'W_i': np.concatenate([W_i_x, W_i_h], axis=0),
            'W_f': np.concatenate([W_f_x, W_f_h], axis=0),
            'W_c': np.concatenate([W_c_x, W_c_h], axis=0),
            'W_o': np.concatenate([W_o_x, W_o_h], axis=0),
            'b_i': b_i,
            'b_f': b_f,
            'b_c': b_c,
            'b_o': b_o
        }
        
        self.lstm_layers[layer_idx].forward_cell.set_weights(lstm_weights)
    
    def _load_bidirectional_lstm_weights(self, weights: List[np.ndarray], layer_idx: int):

        # Bidirectional LSTM has 6 weight matrices:
        # [forward_kernel, forward_recurrent, forward_bias, backward_kernel, backward_recurrent, backward_bias]
        forward_kernel, forward_recurrent, forward_bias = weights[:3]
        backward_kernel, backward_recurrent, backward_bias = weights[3:]
        
        # Load forward weights
        self._split_and_set_lstm_weights(
            forward_kernel, forward_recurrent, forward_bias, 
            self.lstm_layers[layer_idx].forward_cell, 'forward'
        )
        
        # Load backward weights
        self._split_and_set_lstm_weights(
            backward_kernel, backward_recurrent, backward_bias,
            self.lstm_layers[layer_idx].backward_cell, 'backward'
        )
    
    def _split_and_set_lstm_weights(self, kernel: np.ndarray, recurrent_kernel: np.ndarray, 
                                   bias: np.ndarray, cell: LSTMCell, prefix: str):
        hidden_size = kernel.shape[1] // 4
        
        # Split kernel into gates (input, forget, cell, output)
        W_i_x = kernel[:, :hidden_size]
        W_f_x = kernel[:, hidden_size:2*hidden_size]
        W_c_x = kernel[:, 2*hidden_size:3*hidden_size]
        W_o_x = kernel[:, 3*hidden_size:]
        
        # Split recurrent kernel into gates
        W_i_h = recurrent_kernel[:, :hidden_size]
        W_f_h = recurrent_kernel[:, hidden_size:2*hidden_size]
        W_c_h = recurrent_kernel[:, 2*hidden_size:3*hidden_size]
        W_o_h = recurrent_kernel[:, 3*hidden_size:]
        
        # Split bias into gates
        b_i = bias[:hidden_size]
        b_f = bias[hidden_size:2*hidden_size]
        b_c = bias[2*hidden_size:3*hidden_size]
        b_o = bias[3*hidden_size:]
        
        # Combine input and hidden weights for each gate
        lstm_weights = {
            'W_i': np.concatenate([W_i_x, W_i_h], axis=0),
            'W_f': np.concatenate([W_f_x, W_f_h], axis=0),
            'W_c': np.concatenate([W_c_x, W_c_h], axis=0),
            'W_o': np.concatenate([W_o_x, W_o_h], axis=0),
            'b_i': b_i,
            'b_f': b_f,
            'b_c': b_c,
            'b_o': b_o
        }
        
        cell.set_weights(lstm_weights)
    
    def create_padding_mask(self, input_ids: np.ndarray) -> np.ndarray:
        # Assume 0 is the padding token
        mask = (input_ids != 0).astype(np.float32)
        return mask
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        # Create padding mask
        mask = self.create_padding_mask(input_ids)
        
        # Embedding layer
        x = self.embedding.forward(input_ids)
        
        # LSTM layers
        for i, lstm_layer in enumerate(self.lstm_layers):
            x = lstm_layer.forward(x, mask)
        
        # Get last non-padded output for each sequence
        batch_size, seq_length, hidden_size = x.shape
        
        # Find last non-padded position for each sequence
        last_positions = np.sum(mask, axis=1).astype(int) - 1  # (batch_size,)
        
        # Extract last hidden states
        batch_indices = np.arange(batch_size)
        last_hidden = x[batch_indices, last_positions]  # (batch_size, hidden_size)
        
        # Dense layer
        x = self.dense.forward(last_hidden)
        
        # Output layer
        output = self.output_layer.forward(x)
        
        return output
    
    def predict(self, input_ids: np.ndarray, batch_size: int = 32) -> np.ndarray:
        num_samples = input_ids.shape[0]
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch_input = input_ids[i:i + batch_size]
            batch_pred = self.forward(batch_input)
            predictions.append(batch_pred)
        
        return np.concatenate(predictions, axis=0)
    
    def predict_classes(self, input_ids: np.ndarray, batch_size: int = 32) -> np.ndarray:
        probabilities = self.predict(input_ids, batch_size)
        return np.argmax(probabilities, axis=1)
    
    def evaluate(self, input_ids: np.ndarray, true_labels: np.ndarray, 
                batch_size: int = 32) -> Dict:

        print("Evaluating NumPy scratch model...")
        
        # Get predictions
        predicted_probs = self.predict(input_ids, batch_size)
        predicted_labels = np.argmax(predicted_probs, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_labels == true_labels)
        macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
        weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')
        
        # Classification report
        class_report = classification_report(
            true_labels, predicted_labels,
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        
        return {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'classification_report': class_report,
            'predictions': predicted_labels,
            'prediction_probabilities': predicted_probs
        }
    
    def save_weights(self, filepath: str):
        weights_dict = {
            'config': self.config,
            'embedding_weights': self.embedding.weights,
            'dense_weights': self.dense.weights,
            'dense_bias': self.dense.bias,
            'output_weights': self.output_layer.weights,
            'output_bias': self.output_layer.bias
        }
        
        # Save LSTM weights
        for i, lstm_layer in enumerate(self.lstm_layers):
            weights_dict[f'lstm_{i}_forward_weights'] = {
                'W_i': lstm_layer.forward_cell.W_i,
                'W_f': lstm_layer.forward_cell.W_f,
                'W_c': lstm_layer.forward_cell.W_c,
                'W_o': lstm_layer.forward_cell.W_o,
                'b_i': lstm_layer.forward_cell.b_i,
                'b_f': lstm_layer.forward_cell.b_f,
                'b_c': lstm_layer.forward_cell.b_c,
                'b_o': lstm_layer.forward_cell.b_o
            }
            
            if lstm_layer.bidirectional:
                weights_dict[f'lstm_{i}_backward_weights'] = {
                    'W_i': lstm_layer.backward_cell.W_i,
                    'W_f': lstm_layer.backward_cell.W_f,
                    'W_c': lstm_layer.backward_cell.W_c,
                    'W_o': lstm_layer.backward_cell.W_o,
                    'b_i': lstm_layer.backward_cell.b_i,
                    'b_f': lstm_layer.backward_cell.b_f,
                    'b_c': lstm_layer.backward_cell.b_c,
                    'b_o': lstm_layer.backward_cell.b_o
                }
        
        np.savez(filepath, **weights_dict)
        print(f"Scratch model weights saved to: {filepath}")

class ModelComparison:
    
    @staticmethod
    def compare_predictions(keras_model: tf.keras.Model,
                          scratch_model: LSTMScratchModel,
                          input_ids: np.ndarray,
                          true_labels: np.ndarray,
                          tolerance: float = 1e-5) -> Dict:
        
        print("Comparing Keras and NumPy scratch model predictions...")
        
        # Get Keras predictions
        keras_probs = keras_model.predict(input_ids, verbose=0)
        keras_labels = np.argmax(keras_probs, axis=1)
        
        # Get scratch model predictions
        scratch_probs = scratch_model.predict(input_ids)
        scratch_labels = np.argmax(scratch_probs, axis=1)
        
        # Calculate differences
        prob_diff = np.abs(keras_probs - scratch_probs)
        max_prob_diff = np.max(prob_diff)
        mean_prob_diff = np.mean(prob_diff)
        
        # Label agreement
        label_agreement = np.mean(keras_labels == scratch_labels)
        
        # Evaluate both models
        keras_metrics = {
            'accuracy': np.mean(keras_labels == true_labels),
            'macro_f1': f1_score(true_labels, keras_labels, average='macro')
        }
        
        scratch_metrics = {
            'accuracy': np.mean(scratch_labels == true_labels),
            'macro_f1': f1_score(true_labels, scratch_labels, average='macro')
        }
        
        # Check if models are equivalent within tolerance
        models_equivalent = max_prob_diff < tolerance
        
        results = {
            'models_equivalent': models_equivalent,
            'max_probability_difference': float(max_prob_diff),
            'mean_probability_difference': float(mean_prob_diff),
            'label_agreement': float(label_agreement),
            'tolerance': tolerance,
            'keras_metrics': keras_metrics,
            'scratch_metrics': scratch_metrics,
            'keras_predictions': keras_labels,
            'scratch_predictions': scratch_labels,
            'keras_probabilities': keras_probs,
            'scratch_probabilities': scratch_probs
        }
        
        return results
    
    @staticmethod
    def print_comparison_results(results: Dict):
        print("\n" + "="*60)
        print("KERAS vs NUMPY SCRATCH MODEL COMPARISON")
        print("="*60)
        
        print(f"Models equivalent (within tolerance): {results['models_equivalent']}")
        print(f"Tolerance: {results['tolerance']}")
        print(f"Max probability difference: {results['max_probability_difference']:.8f}")
        print(f"Mean probability difference: {results['mean_probability_difference']:.8f}")
        print(f"Label agreement: {results['label_agreement']:.4f}")
        
        print(f"\nKeras Model Performance:")
        print(f"  Accuracy: {results['keras_metrics']['accuracy']:.4f}")
        print(f"  Macro F1: {results['keras_metrics']['macro_f1']:.4f}")
        
        print(f"\nNumPy Scratch Model Performance:")
        print(f"  Accuracy: {results['scratch_metrics']['accuracy']:.4f}")
        print(f"  Macro F1: {results['scratch_metrics']['macro_f1']:.4f}")
        
        if results['models_equivalent']:
            print("\n✓ SUCCESS: Models produce equivalent results!")
        else:
            print(f"\n⚠ WARNING: Models differ by more than tolerance ({results['tolerance']})")

if __name__ == "__main__":
    # Example usage
    from datasets.data_loader import NusaXDataLoader
    from preprocessing.text_preprocessor import IndonesianTextPreprocessor
    from models.lstm_keras import LSTMSentimentModel, LSTMConfig
    
    # Load data
    loader = NusaXDataLoader()
    train_df, valid_df, test_df = loader.load_dataset()
    
    train_texts, train_labels = loader.get_texts_and_labels(train_df)
    test_texts, test_labels = loader.get_texts_and_labels(test_df)
    
    # Preprocess texts
    preprocessor = IndonesianTextPreprocessor(max_tokens=10000, max_sequence_length=128)
    preprocessor.build_vectorizer(train_texts)
    
    x_test = preprocessor.preprocess_texts(test_texts)
    
    # Load trained Keras model (example)
    config = LSTMConfig(
        vocab_size=preprocessor.get_vocabulary_size(),
        embedding_dim=128,
        lstm_units=64,
        num_lstm_layers=1,
        bidirectional=False
    )
    
    keras_model = LSTMSentimentModel(config)
    keras_model.build_model()
    # keras_model.load_model('path/to/model.h5', 'path/to/config.json')  # Load trained model
    
    # Create scratch model
    scratch_model = LSTMScratchModel(config.to_dict())
    scratch_model.load_weights_from_keras(keras_model.model)
    
    # Compare models
    comparison = ModelComparison.compare_predictions(
        keras_model.model, scratch_model, x_test, test_labels
    )
    
    ModelComparison.print_comparison_results(comparison)