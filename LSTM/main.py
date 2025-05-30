import os
import sys
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List
import argparse
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.data_loader import NusaXDataLoader
from preprocessing.text_preprocessor import IndonesianTextPreprocessor, TextStatistics
from models.lstm_keras import LSTMSentimentModel, LSTMConfig, ExperimentRunner
from models.lstm_scratch import LSTMScratchModel, ModelComparison

class ProjectPipeline:    
    def __init__(self, 
                 data_dir: str = 'datasets',
                 results_dir: str = 'results',
                 max_tokens: int = 10000,
                 max_sequence_length: int = 128):
        
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.max_tokens = max_tokens
        self.max_sequence_length = max_sequence_length
        
        # Create results directory structure
        self._create_directory_structure()
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.experiment_runner = None
        
        # Data storage
        self.train_texts = None
        self.train_labels = None
        self.val_texts = None
        self.val_labels = None
        self.test_texts = None
        self.test_labels = None
        
        self.x_train = None
        self.x_val = None
        self.x_test = None
    
    def _create_directory_structure(self):
        directories = [
            self.results_dir,
            os.path.join(self.results_dir, 'scratch_comparison')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"Created directory structure in: {self.results_dir}")
    
    def load_and_preprocess_data(self):
        print("\n" + "="*60)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        # Load data
        self.data_loader = NusaXDataLoader(self.data_dir)
        self.data_loader.print_dataset_info()
        
        train_df, valid_df, test_df = self.data_loader.load_dataset()
        
        # Extract texts and labels
        self.train_texts, self.train_labels = self.data_loader.get_texts_and_labels(train_df)
        self.val_texts, self.val_labels = self.data_loader.get_texts_and_labels(valid_df)
        self.test_texts, self.test_labels = self.data_loader.get_texts_and_labels(test_df)
        
        # Print text statistics
        TextStatistics.print_text_statistics(self.train_texts, "Training")
        TextStatistics.print_text_statistics(self.val_texts, "Validation")
        TextStatistics.print_text_statistics(self.test_texts, "Test")
        
        # Initialize and build preprocessor
        self.preprocessor = IndonesianTextPreprocessor(
            max_tokens=self.max_tokens,
            max_sequence_length=self.max_sequence_length
        )
        
        print(f"\nBuilding text vectorizer...")
        self.preprocessor.build_vectorizer(self.train_texts)
        
        # Preprocess texts
        print("Preprocessing texts...")
        self.x_train = self.preprocessor.preprocess_texts(self.train_texts)
        self.x_val = self.preprocessor.preprocess_texts(self.val_texts)
        self.x_test = self.preprocessor.preprocess_texts(self.test_texts)
        
        # Save preprocessor
        preprocessor_path = os.path.join(self.results_dir, 'text_vectorizer.pkl')
        self.preprocessor.save_vectorizer(preprocessor_path)
        
        # Print preprocessing info
        print(f"\nPreprocessing completed:")
        print(f"  Vocabulary size: {self.preprocessor.get_vocabulary_size()}")
        print(f"  Training shape: {self.x_train.shape}")
        print(f"  Validation shape: {self.x_val.shape}")
        print(f"  Test shape: {self.x_test.shape}")
        
        return True
    
    def run_keras_experiments(self):
        print("\n" + "="*60)
        print("STEP 2: RUNNING KERAS LSTM EXPERIMENTS")
        print("="*60)
        
        # Initialize experiment runner
        self.experiment_runner = ExperimentRunner(
            x_train=self.x_train,
            y_train=self.train_labels,
            x_val=self.x_val,
            y_val=self.val_labels,
            x_test=self.x_test,
            y_test=self.test_labels,
            vocab_size=self.preprocessor.get_vocabulary_size(),
            results_dir=self.results_dir
        )
        
        # Run all experiments
        experiment_results = self.experiment_runner.run_all_experiments()
        
        # Find best model based on validation macro F1
        best_experiment = max(experiment_results, key=lambda x: x['val_macro_f1'])
        
        print(f"\n" + "="*60)
        print("KERAS EXPERIMENTS SUMMARY")
        print("="*60)
        print(f"Best model: {best_experiment['experiment_name']}")
        print(f"Best validation macro F1: {best_experiment['val_macro_f1']:.4f}")
        print(f"Best test macro F1: {best_experiment['test_macro_f1']:.4f}")
        
        return best_experiment
    
    def run_scratch_implementation(self, best_experiment: Dict):
        print("\n" + "="*60)
        print("STEP 3: NUMPY SCRATCH IMPLEMENTATION")
        print("="*60)
        
        # Load best Keras model
        best_exp_name = best_experiment['experiment_name']
        model_path = os.path.join(self.results_dir, best_exp_name, 'model.weights.h5')
        config_path = os.path.join(self.results_dir, best_exp_name, 'model.config.json')
        
        print(f"Loading best model: {best_exp_name}")
        
        # Load Keras model
        keras_model = LSTMSentimentModel(LSTMConfig.from_dict(best_experiment['config']))
        keras_model.load_model(model_path, config_path)
        keras_model.compile_model()
        
        print("✓ Keras model loaded successfully")
        
        # Create scratch model
        print("Creating NumPy scratch model...")
        scratch_model = LSTMScratchModel(best_experiment['config'])
        
        # Load weights from Keras model
        scratch_model.load_weights_from_keras(keras_model.model)
        print("✓ Weights transferred to scratch model")
        
        # Compare models
        print("\nComparing Keras and NumPy scratch models...")
        comparison_results = ModelComparison.compare_predictions(
            keras_model=keras_model.model,
            scratch_model=scratch_model,
            input_ids=self.x_test,
            true_labels=self.test_labels,
            tolerance=1e-4
        )
        
        # Print comparison results
        ModelComparison.print_comparison_results(comparison_results)
        
        # Save scratch model evaluation
        scratch_eval = scratch_model.evaluate(self.x_test, self.test_labels)
        
        # Save comparison results
        comparison_dir = os.path.join(self.results_dir, 'scratch_comparison')
        
        # Save detailed comparison
        comparison_file = os.path.join(comparison_dir, 'model_comparison.json')
        with open(comparison_file, 'w') as f:
            # Convert numpy arrays and boolean objects to JSON serializable types
            comparison_json = {
                'models_equivalent': bool(comparison_results['models_equivalent']),
                'max_probability_difference': float(comparison_results['max_probability_difference']),
                'mean_probability_difference': float(comparison_results['mean_probability_difference']),
                'label_agreement': float(comparison_results['label_agreement']),
                'tolerance': float(comparison_results['tolerance']),
                'keras_metrics': {
                    'accuracy': float(comparison_results['keras_metrics']['accuracy']),
                    'macro_f1': float(comparison_results['keras_metrics']['macro_f1'])
                },
                'scratch_metrics': {
                    'accuracy': float(comparison_results['scratch_metrics']['accuracy']),
                    'macro_f1': float(comparison_results['scratch_metrics']['macro_f1'])
                }
            }
            json.dump(comparison_json, f, indent=2)
        
        # Save scratch model weights
        scratch_weights_file = os.path.join(comparison_dir, 'scratch_model_weights.npz')
        scratch_model.save_weights(scratch_weights_file)
        
        # Save detailed scratch evaluation
        scratch_eval_file = os.path.join(comparison_dir, 'scratch_evaluation.txt')
        with open(scratch_eval_file, 'w') as f:
            f.write("NumPy Scratch Model Evaluation\n")
            f.write("="*40 + "\n\n")
            f.write(f"Accuracy: {scratch_eval['accuracy']:.4f}\n")
            f.write(f"Macro F1: {scratch_eval['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {scratch_eval['weighted_f1']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(str(scratch_eval['classification_report']))
        
        print(f"\nScratch implementation results saved to: {comparison_dir}")
        
        return comparison_results, scratch_eval
    
    def generate_final_report(self, best_experiment: Dict, 
                            comparison_results: Dict, 
                            scratch_eval: Dict):
        print("\n" + "="*60)
        print("STEP 4: GENERATING FINAL REPORT")
        print("="*60)
        
        report_file = os.path.join(self.results_dir, 'FINAL_PROJECT_REPORT.txt')
        
        with open(report_file, 'w') as f:
            f.write("LSTM-BASED SENTIMENT CLASSIFICATION PROJECT REPORT\n")
            f.write("="*60 + "\n")
            f.write("IF3270 - Machine Learning\n")
            f.write("NusaX Sentiment (Indonesian) Dataset\n\n")
            
            # Dataset information
            f.write("1. DATASET INFORMATION\n")
            f.write("-"*30 + "\n")
            dataset_info = self.data_loader.get_dataset_info()
            f.write(f"Classes: {dataset_info['classes']}\n")
            f.write(f"Train samples: {dataset_info['train_size']}\n")
            f.write(f"Validation samples: {dataset_info['valid_size']}\n")
            f.write(f"Test samples: {dataset_info['test_size']}\n")
            f.write(f"Total samples: {sum([dataset_info['train_size'], dataset_info['valid_size'], dataset_info['test_size']])}\n\n")
            
            # Preprocessing information
            f.write("2. PREPROCESSING\n")
            f.write("-"*30 + "\n")
            preprocessing_info = self.preprocessor.get_preprocessing_info()
            f.write(f"Max tokens (vocabulary size): {preprocessing_info['max_tokens']}\n")
            f.write(f"Actual vocabulary size: {preprocessing_info['vocabulary_size']}\n")
            f.write(f"Max sequence length: {preprocessing_info['max_sequence_length']}\n\n")
            
            # Best model information
            f.write("3. BEST KERAS MODEL\n")
            f.write("-"*30 + "\n")
            f.write(f"Experiment name: {best_experiment['experiment_name']}\n")
            f.write(f"Configuration:\n")
            for key, value in best_experiment['config'].items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nPerformance:\n")
            f.write(f"  Validation Macro F1: {best_experiment['val_macro_f1']:.4f}\n")
            f.write(f"  Validation Accuracy: {best_experiment['val_accuracy']:.4f}\n")
            f.write(f"  Test Macro F1: {best_experiment['test_macro_f1']:.4f}\n")
            f.write(f"  Test Accuracy: {best_experiment['test_accuracy']:.4f}\n\n")
            
            # Scratch implementation results
            f.write("4. NUMPY SCRATCH IMPLEMENTATION\n")
            f.write("-"*30 + "\n")
            f.write(f"Models equivalent: {comparison_results['models_equivalent']}\n")
            f.write(f"Max probability difference: {comparison_results['max_probability_difference']:.8f}\n")
            f.write(f"Label agreement: {comparison_results['label_agreement']:.4f}\n")
            f.write(f"\nScratch Model Performance:\n")
            f.write(f"  Accuracy: {scratch_eval['accuracy']:.4f}\n")
            f.write(f"  Macro F1: {scratch_eval['macro_f1']:.4f}\n")
            f.write(f"  Weighted F1: {scratch_eval['weighted_f1']:.4f}\n\n")
            
            # All experiments summary
            f.write("5. ALL EXPERIMENTS SUMMARY\n")
            f.write("-"*30 + "\n")
            for result in sorted(self.experiment_runner.experiment_results, 
                               key=lambda x: x['test_macro_f1'], reverse=True):
                f.write(f"{result['experiment_name']}:\n")
                f.write(f"  Val F1: {result['val_macro_f1']:.4f}, Test F1: {result['test_macro_f1']:.4f}\n")
            
            f.write(f"\n6. PROJECT STRUCTURE\n")
            f.write("-"*30 + "\n")
            f.write("datasets/data_loader.py - Data loading utilities\n")
            f.write("preprocessing/text_preprocessor.py - Text preprocessing\n")
            f.write("models/lstm_keras.py - Keras LSTM implementation\n")
            f.write("models/lstm_scratch.py - NumPy scratch implementation\n")
            f.write("main.py - Main experiment runner\n")
            f.write("results/ - All experiment results and models\n")
        
        print(f"Final report generated: {report_file}")
    
    def run_complete_pipeline(self):
        print("STARTING LSTM SENTIMENT CLASSIFICATION PROJECT")
        print("="*60)
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Run Keras experiments
            best_experiment = self.run_keras_experiments()
            
            # Step 3: Run scratch implementation
            comparison_results, scratch_eval = self.run_scratch_implementation(best_experiment)
            
            # Step 4: Generate final report
            self.generate_final_report(best_experiment, comparison_results, scratch_eval)
            
            print("\n" + "="*60)
            print("PROJECT COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Results saved in: {self.results_dir}")
            print("Check FINAL_PROJECT_REPORT.txt for complete summary")
            
            return True
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description='LSTM Sentiment Classification Project')
    parser.add_argument('--data_dir', type=str, default='.', 
                       help='Directory containing train.csv, valid.csv, test.csv')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--max_tokens', type=int, default=10000,
                       help='Maximum vocabulary size')
    parser.add_argument('--max_sequence_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--skip_keras', action='store_true',
                       help='Skip Keras experiments (for testing scratch only)')
    parser.add_argument('--skip_scratch', action='store_true',
                       help='Skip scratch implementation (for Keras only)')
    
    args = parser.parse_args()
    
    # Validate data files exist
    required_files = ['train.csv', 'valid.csv', 'test.csv']
    for file in required_files:
        filepath = os.path.join(args.data_dir, file)
        if not os.path.exists(filepath):
            print(f"ERROR: Required file not found: {filepath}")
            return False
    
    # Initialize pipeline
    pipeline = ProjectPipeline(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        max_tokens=args.max_tokens,
        max_sequence_length=args.max_sequence_length
    )
    
    if args.skip_keras and args.skip_scratch:
        print("ERROR: Cannot skip both Keras and scratch implementations")
        return False
    
    # Run pipeline based on arguments
    if not args.skip_keras and not args.skip_scratch:
        # Run complete pipeline
        return pipeline.run_complete_pipeline()
    
    elif args.skip_scratch:
        # Run only Keras experiments
        pipeline.load_and_preprocess_data()
        best_experiment = pipeline.run_keras_experiments()
        print(f"\nBest model: {best_experiment['experiment_name']}")
        print(f"Test Macro F1: {best_experiment['test_macro_f1']:.4f}")
        return True
    
    elif args.skip_keras:
        # Run only scratch implementation (need existing Keras model)
        print("ERROR: Scratch implementation requires trained Keras model")
        print("Run with Keras experiments first, then use --skip_keras for testing")
        return False

class QuickTest:
    
    @staticmethod
    def test_data_loading():
        print("Testing data loading...")
        
        loader = NusaXDataLoader()
        loader.print_dataset_info()
        
        train_df, valid_df, test_df = loader.load_dataset()
        train_texts, train_labels = loader.get_texts_and_labels(train_df)
        
        print(f"Sample text: {train_texts[0]}")
        print(f"Sample label: {train_labels[0]}")
        
        return True
    
    @staticmethod
    def test_preprocessing():
        print("Testing text preprocessing...")
        
        loader = NusaXDataLoader()
        train_df, _, _ = loader.load_dataset()
        train_texts, _ = loader.get_texts_and_labels(train_df)
        
        preprocessor = IndonesianTextPreprocessor(max_tokens=1000, max_sequence_length=64)
        preprocessor.build_vectorizer(train_texts[:100])  # Use subset for testing
        
        vectorized = preprocessor.preprocess_texts(train_texts[:5])
        print(f"Vectorized shape: {vectorized.shape}")
        print(f"Sample vectorized: {vectorized[0][:10]}")
        
        return True
    
    @staticmethod
    def test_simple_model():
        print("Testing simple model...")
        
        # Load small subset of data
        loader = NusaXDataLoader()
        train_df, valid_df, _ = loader.load_dataset()
        
        # Use small subset for quick test
        train_subset = train_df.head(100)
        valid_subset = valid_df.head(20)
        
        train_texts, train_labels = loader.get_texts_and_labels(train_subset)
        val_texts, val_labels = loader.get_texts_and_labels(valid_subset)
        
        # Quick preprocessing
        preprocessor = IndonesianTextPreprocessor(max_tokens=1000, max_sequence_length=32)
        preprocessor.build_vectorizer(train_texts)
        
        x_train = preprocessor.preprocess_texts(train_texts)
        x_val = preprocessor.preprocess_texts(val_texts)
        
        # Simple model config
        config = LSTMConfig(
            vocab_size=preprocessor.get_vocabulary_size(),
            embedding_dim=32,
            lstm_units=16,
            num_lstm_layers=1,
            bidirectional=False,
            max_sequence_length=32
        )
        
        # Train simple model
        model = LSTMSentimentModel(config)
        model.build_model()
        model.compile_model()
        
        print("Training simple model...")
        model.train(x_train, train_labels, x_val, val_labels, epochs=2, verbose=1)
        
        # Test evaluation
        results = model.evaluate(x_val, val_labels)
        print(f"Test accuracy: {results['test_accuracy']:.4f}")
        print(f"Test macro F1: {results['macro_f1']:.4f}")
        
        return True

if __name__ == "__main__":
    # Check if this is a test run
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running quick tests...")
        
        if len(sys.argv) > 2:
            test_name = sys.argv[2]
            if test_name == 'data':
                QuickTest.test_data_loading()
            elif test_name == 'preprocess':
                QuickTest.test_preprocessing()
            elif test_name == 'model':
                QuickTest.test_simple_model()
            else:
                print(f"Unknown test: {test_name}")
                print("Available tests: data, preprocess, model")
        else:
            print("Running all tests...")
            QuickTest.test_data_loading()
            QuickTest.test_preprocessing()
            QuickTest.test_simple_model()
    else:
        # Run main pipeline
        success = main()
        sys.exit(0 if success else 1)
        