import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, GRU, SimpleRNN, 
    Bidirectional, Dense, Dropout, 
    Input, GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class RNNSentimentClassifier:

    def __init__(self, vocab_size=5000, embedding_dim=128, max_sequence_length=55, num_classes=3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, 
                   num_rnn_layers=1,
                   rnn_units=64,
                   bidirectional=True,
                   dropout_rate=0.3,
                   dense_units=32,
                   learning_rate=0.001):

        model = Sequential()
        
        # Embedding Layer
        model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            mask_zero=True,
            name='embedding'
        ))
        
        # Initial Dropout
        model.add(Dropout(dropout_rate * 0.5, name='embedding_dropout'))
        
        # RNN Layers
        if isinstance(rnn_units, int):
            rnn_units = [rnn_units] * num_rnn_layers
        elif len(rnn_units) != num_rnn_layers:
            raise ValueError("Length of rnn_units must match num_rnn_layers")
        
        for i, units in enumerate(rnn_units):
            return_sequences = (i < num_rnn_layers - 1) 
            
            rnn_layer = SimpleRNN(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate * 0.5,  
                name=f'rnn_{i+1}'
            )
            
            if bidirectional:
                rnn_layer = Bidirectional(rnn_layer, name=f'bidirectional_rnn_{i+1}')
            
            model.add(rnn_layer)
            
            if i < num_rnn_layers - 1:
                model.add(Dropout(dropout_rate, name=f'rnn_dropout_{i+1}'))
        
        # Dense Layers
        if dense_units > 0:
            model.add(Dense(dense_units, activation='relu', name='dense'))
            model.add(Dropout(dropout_rate, name='dropout_dense'))
        
        # Output layer
        model.add(Dense(self.num_classes, activation='softmax', name='output'))
        
        # Compile
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, 
              train_dataset, 
              validation_dataset=None,
              epochs=50,
              callbacks=None,
              verbose=1):
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Default Callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if validation_dataset else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if validation_dataset else 'loss',
                    factor=0.5,
                    patience=5
                )
            ]
        
        # Train
        self.history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(self, test_dataset, y_true=None):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get Predictions
        predictions = self.model.predict(test_dataset)
        y_pred = np.argmax(predictions, axis=1)
        
        # True Labels
        if y_true is None:
            y_true = []
            for _, labels in test_dataset:
                y_true.extend(labels.numpy())
            y_true = np.array(y_true)
        
        # Calculate Metrics
        test_loss, test_accuracy = self.model.evaluate(test_dataset, verbose=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        metrics = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'macro_f1': macro_f1,
            'predictions': y_pred,
            'true_labels': y_true
        }
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot Training & Validation Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Training & Validation Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def get_model_summary(self):
        if self.model is None:
            return {"error": "Model not built yet"}
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.reduce_prod(var.shape) for var in self.model.trainable_variables])
        
        return {
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "layers": [layer.name for layer in self.model.layers],
            "model_summary": self.model.summary
        }
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if not filepath.endswith(('.keras', '.h5')):
            filepath += '.keras'
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def save_weights(self, filepath):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if not filepath.endswith('.weights.h5'):
            filepath += '.weights.h5'
        
        self.model.save_weights(filepath)
        print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.load_weights(filepath)
        print(f"Weights loaded from {filepath}")

class ModelComparator:
    def __init__(self):
        self.results = []
    
    def add_result(self, config_name, metrics, history=None):
        result = {
            'config_name': config_name,
            'metrics': metrics,
            'history': history
        }
        self.results.append(result)
    
    def compare_results(self):
        import pandas as pd
        
        comparison_data = []
        for result in self.results:
            comparison_data.append({
                'Configuration': result['config_name'],
                'Test Accuracy': result['metrics']['test_accuracy'],
                'Test Loss': result['metrics']['test_loss'],
                'Macro F1-Score': result['metrics']['macro_f1']
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Macro F1-Score', ascending=False)
    
    def plot_comparison(self, metric='macro_f1', save_path=None):
        configs = [result['config_name'] for result in self.results]
        values = [result['metrics'][metric] for result in self.results]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(configs, values)
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.xlabel('Configuration')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha='right')
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_training_curves_comparison(self, save_path=None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        for result in self.results:
            if result['history'] is not None:
                history = result['history'].history
                config_name = result['config_name']
                
                # Training Accuracy
                ax1.plot(history['accuracy'], label=f'{config_name} - Train')
                
                # Validation Accuracy
                if 'val_accuracy' in history:
                    ax2.plot(history['val_accuracy'], label=f'{config_name} - Val')
                
                # Training Loss
                ax3.plot(history['loss'], label=f'{config_name} - Train')
                
                # Validation Loss
                if 'val_loss' in history:
                    ax4.plot(history['val_loss'], label=f'{config_name} - Val')
        
        ax1.set_title('Training Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        ax3.set_title('Training Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        
        ax4.set_title('Validation Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()