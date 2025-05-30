import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import os
import json
from typing import Dict, List, Tuple, Optional
import seaborn as sns

class LSTMConfig:    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 128,
                 lstm_units: int = 64,
                 num_lstm_layers: int = 1,
                 bidirectional: bool = False,
                 dropout_rate: float = 0.3,
                 dense_units: int = 32,
                 num_classes: int = 3,
                 max_sequence_length: int = 128):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            lstm_units: Number of LSTM units per layer
            num_lstm_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout_rate: Dropout rate
            dense_units: Dense layer units before output
            num_classes: Number of output classes
            max_sequence_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
    
    def to_dict(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_lstm_layers': self.num_lstm_layers,
            'bidirectional': self.bidirectional,
            'dropout_rate': self.dropout_rate,
            'dense_units': self.dense_units,
            'num_classes': self.num_classes,
            'max_sequence_length': self.max_sequence_length
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(**config_dict)

class LSTMSentimentModel:    
    def __init__(self, config: LSTMConfig):
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:        
        # Input layer
        inputs = layers.Input(shape=(self.config.max_sequence_length,), name='input_ids')
        
        # Embedding layer
        x = layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=self.config.embedding_dim,
            input_length=self.config.max_sequence_length,
            mask_zero=True,
            name='embedding'
        )(inputs)
        
        # LSTM layers
        for i in range(self.config.num_lstm_layers):
            return_sequences = (i < self.config.num_lstm_layers - 1)  # Return sequences for all but last layer
            
            if self.config.bidirectional:
                x = layers.Bidirectional(
                    layers.LSTM(
                        self.config.lstm_units,
                        return_sequences=return_sequences,
                        dropout=self.config.dropout_rate,
                        name=f'lstm_{i}'
                    ),
                    name=f'bidirectional_lstm_{i}'
                )(x)
            else:
                x = layers.LSTM(
                    self.config.lstm_units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    name=f'lstm_{i}'
                )(x)
        
        # Dropout layer
        x = layers.Dropout(self.config.dropout_rate, name='dropout')(x)
        
        # Dense layer before output
        x = layers.Dense(
            self.config.dense_units, 
            activation='relu',
            name='dense'
        )(x)
        
        # Output layer
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='lstm_sentiment_model')
        
        self.model = model
        return model
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
                     metrics: List[str] = None):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if metrics is None:
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=metrics
        )
    
    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 20,
              batch_size: int = 32,
              verbose: int = 1,
              early_stopping_patience: int = 5) -> tf.keras.callbacks.History:
        if self.model is None:
            raise ValueError("Model not built and compiled.")
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Get predictions
        y_pred_probs = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        
        return {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'classification_report': class_report,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_probs
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: Optional[str] = None):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str, config_path: str):
        if self.model is None:
            raise ValueError("Model not built.")
        
        # Save model weights
        self.model.save_weights(model_path)
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"Model weights saved to: {model_path}")
        print(f"Model config saved to: {config_path}")
    
    def load_model(self, model_path: str, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        self.config = LSTMConfig.from_dict(config_dict)
        
        # Build and load model
        self.build_model()
        self.model.load_weights(model_path)
        
        print(f"Model loaded from: {model_path}")
        print(f"Config loaded from: {config_path}")
    
    def get_model_summary(self):
        if self.model is None:
            raise ValueError("Model not built.")
        
        print("=== Model Architecture ===")
        self.model.summary()
        print(f"\nModel Configuration:")
        for key, value in self.config.to_dict().items():
            print(f"  {key}: {value}")

class ExperimentRunner:    
    def __init__(self, 
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_val: np.ndarray,
                 y_val: np.ndarray,
                 x_test: np.ndarray,
                 y_test: np.ndarray,
                 vocab_size: int,
                 results_dir: str = 'results'):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.vocab_size = vocab_size
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        self.experiment_results = []
    
    def run_experiment(self, 
                      config: LSTMConfig,
                      experiment_name: str,
                      epochs: int = 20,
                      batch_size: int = 32) -> Dict:

        print(f"\n{'='*50}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*50}")
        
        # Create experiment directory
        exp_dir = os.path.join(self.results_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Initialize model
        model = LSTMSentimentModel(config)
        model.build_model()
        model.compile_model()
        
        # Print model info
        print(f"Model Configuration: {config.to_dict()}")
        model.get_model_summary()
        
        # Train model
        print(f"\nTraining model...")
        history = model.train(
            self.x_train, self.y_train,
            self.x_val, self.y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate model
        print(f"\nEvaluating model...")
        test_results = model.evaluate(self.x_test, self.y_test)
        val_results = model.evaluate(self.x_val, self.y_val)
        
        # Save results
        results = {
            'experiment_name': experiment_name,
            'config': config.to_dict(),
            'val_macro_f1': val_results['macro_f1'],
            'val_accuracy': val_results['test_accuracy'],
            'test_macro_f1': test_results['macro_f1'],
            'test_accuracy': test_results['test_accuracy'],
            'training_history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            }
        }
        
        # Save model
        model_path = os.path.join(exp_dir, 'model.weights.h5')
        config_path = os.path.join(exp_dir, 'model.config.json')
        model.save_model(model_path, config_path)
        
        # Save training plots
        plot_path = os.path.join(exp_dir, 'training_history.png')
        model.plot_training_history(save_path=plot_path)
        
        # Save confusion matrix
        cm_path = os.path.join(exp_dir, 'confusion_matrix.png')
        model.plot_confusion_matrix(
            self.y_test, 
            test_results['predictions'], 
            save_path=cm_path
        )
        
        # Save results as JSON
        results_path = os.path.join(exp_dir, 'results.json')
        
        # Convert numpy types to Python native types for JSON serialization
        json_safe_results = {
            'experiment_name': results['experiment_name'],
            'config': results['config'],
            'val_macro_f1': float(results['val_macro_f1']),
            'val_accuracy': float(results['val_accuracy']),
            'test_macro_f1': float(results['test_macro_f1']),
            'test_accuracy': float(results['test_accuracy']),
            'training_history': {
                'loss': [float(x) for x in results['training_history']['loss']],
                'val_loss': [float(x) for x in results['training_history']['val_loss']],
                'accuracy': [float(x) for x in results['training_history']['accuracy']],
                'val_accuracy': [float(x) for x in results['training_history']['val_accuracy']]
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        # Save detailed evaluation
        eval_path = os.path.join(exp_dir, 'evaluation.txt')
        with open(eval_path, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Configuration:\n")
            for key, value in config.to_dict().items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nValidation Results:\n")
            f.write(f"  Macro F1: {val_results['macro_f1']:.4f}\n")
            f.write(f"  Accuracy: {val_results['test_accuracy']:.4f}\n")
            f.write(f"\nTest Results:\n")
            f.write(f"  Macro F1: {test_results['macro_f1']:.4f}\n")
            f.write(f"  Accuracy: {test_results['test_accuracy']:.4f}\n")
            f.write(f"\nClassification Report:\n")
            f.write(str(test_results['classification_report']))
        
        self.experiment_results.append(results)
        
        print(f"\nExperiment completed!")
        print(f"Validation Macro F1: {val_results['macro_f1']:.4f}")
        print(f"Test Macro F1: {test_results['macro_f1']:.4f}")
        print(f"Results saved to: {exp_dir}")
        
        return results
    
    def run_all_experiments(self) -> List[Dict]:
        experiments = self.get_experiment_configs()
        
        for exp_name, config in experiments:
            self.run_experiment(config, exp_name)
        
        # Save summary
        self.save_experiment_summary()
        
        return self.experiment_results
    
    def get_experiment_configs(self) -> List[Tuple[str, LSTMConfig]]:
        experiments = []
        
        # Base configuration
        base_config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': 128,
            'dropout_rate': 0.3,
            'dense_units': 32,
            'num_classes': 3,
            'max_sequence_length': 128
        }
        
        # 1. Vary number of LSTM layers (3 configurations)
        for num_layers in [1, 2, 3]:
            config = LSTMConfig(
                lstm_units=64,
                num_lstm_layers=num_layers,
                bidirectional=False,
                **base_config
            )
            experiments.append((f"lstm_layers_{num_layers}", config))
        
        # 2. Vary number of cells per layer (3 configurations)
        for lstm_units in [32, 64, 128]:
            config = LSTMConfig(
                lstm_units=lstm_units,
                num_lstm_layers=1,
                bidirectional=False,
                **base_config
            )
            experiments.append((f"lstm_units_{lstm_units}", config))
        
        # 3. Compare bidirectional vs unidirectional (2 configurations)
        for bidirectional in [False, True]:
            direction = "bidirectional" if bidirectional else "unidirectional"
            config = LSTMConfig(
                lstm_units=64,
                num_lstm_layers=1,
                bidirectional=bidirectional,
                **base_config
            )
            experiments.append((f"lstm_{direction}", config))
        
        return experiments
    
    def save_experiment_summary(self):
        summary_path = os.path.join(self.results_dir, 'experiment_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("LSTM Sentiment Classification - Experiment Summary\n")
            f.write("="*60 + "\n\n")
            
            # Sort by test macro F1
            sorted_results = sorted(
                self.experiment_results, 
                key=lambda x: x['test_macro_f1'], 
                reverse=True
            )
            
            f.write("Results (sorted by Test Macro F1):\n")
            f.write("-" * 40 + "\n")
            
            for i, result in enumerate(sorted_results, 1):
                f.write(f"{i}. {result['experiment_name']}\n")
                f.write(f"   Val Macro F1: {result['val_macro_f1']:.4f}\n")
                f.write(f"   Test Macro F1: {result['test_macro_f1']:.4f}\n")
                f.write(f"   Test Accuracy: {result['test_accuracy']:.4f}\n")
                f.write(f"   Config: {result['config']}\n\n")
        
        print(f"Experiment summary saved to: {summary_path}")

if __name__ == "__main__":
    # Example usage
    from datasets.data_loader import NusaXDataLoader
    from preprocessing.text_preprocessor import IndonesianTextPreprocessor
    
    # Load data
    loader = NusaXDataLoader()
    train_df, valid_df, test_df = loader.load_dataset()
    
    train_texts, train_labels = loader.get_texts_and_labels(train_df)
    val_texts, val_labels = loader.get_texts_and_labels(valid_df)
    test_texts, test_labels = loader.get_texts_and_labels(test_df)
    
    # Preprocess texts
    preprocessor = IndonesianTextPreprocessor(max_tokens=10000, max_sequence_length=128)
    preprocessor.build_vectorizer(train_texts)
    
    x_train = preprocessor.preprocess_texts(train_texts)
    x_val = preprocessor.preprocess_texts(val_texts)
    x_test = preprocessor.preprocess_texts(test_texts)
    
    # Run experiments
    runner = ExperimentRunner(
        x_train, train_labels,
        x_val, val_labels,
        x_test, test_labels,
        vocab_size=preprocessor.get_vocabulary_size(),
        results_dir='results'
    )
    
    results = runner.run_all_experiments()
    print("All experiments completed!")
    