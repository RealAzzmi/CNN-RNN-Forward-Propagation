import os
import numpy as np
from tensorflow import keras
from sklearn.metrics import f1_score
from typing import Dict, Any
from config import MODELS_DIR, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE

class CNNModelTrainer:    
    def __init__(self, data_splits: Dict):
        self.x_train = data_splits['x_train']
        self.y_train = data_splits['y_train']
        self.x_val = data_splits['x_val']
        self.y_val = data_splits['y_val']
        self.x_test = data_splits['x_test']
        self.y_test = data_splits['y_test']
    
    def train_model(self, model: keras.Model, experiment_name: str, epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, Any]:
        print(f"\nTraining model: {experiment_name}")
        
        # Training the model using keras
        history = model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            verbose=1
        )
        
        # Evaluating the loss & accuracy on the test set
        test_loss, test_accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Evaluating the f1_macro on the test set
        y_pred = model.predict(self.x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1_macro = f1_score(self.y_test, y_pred_classes, average='macro')
        
        # Saving the model weights and architecture
        model_path = os.path.join(MODELS_DIR, f"{experiment_name}.weights.h5")
        architecture_path = os.path.join(MODELS_DIR, f"{experiment_name}.json")
        
        model.save_weights(model_path)
        with open(architecture_path, 'w') as f:
            f.write(model.to_json())
        
        results = {
            'history': history.history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'f1_macro': f1_macro,
            'model_path': model_path,
            'architecture_path': architecture_path
        }
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1-Score (macro): {f1_macro:.4f}")
        
        return results