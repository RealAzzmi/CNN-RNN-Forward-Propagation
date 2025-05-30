import matplotlib.pyplot as plt
import os
from typing import Dict
from config import PLOTS_DIR

class ExperimentVisualizer:
    def plot_comparison(results: Dict, title: str, filename: str):
        plt.figure(figsize=(15, 5))
        
        # Plot training loss
        plt.subplot(1, 3, 1)
        for name, result in results.items():
            plt.plot(result['history']['loss'], label=f'{name} (train)')
        plt.title(f'{title} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot validation loss
        plt.subplot(1, 3, 2)
        for name, result in results.items():
            plt.plot(result['history']['val_loss'], label=f'{name} (val)')
        plt.title(f'{title} - Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot F1 scores comparison
        plt.subplot(1, 3, 3)
        names = list(results.keys())
        f1_scores = [results[name]['f1_macro'] for name in names]
        plt.bar(names, f1_scores)
        plt.title(f'{title} - F1 Score Comparison')
        plt.ylabel('F1 Score (Macro)')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f'{filename}_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training loss, validation loss, and f1 scores were already saved to: {plot_path}")
    
    def print_summary(results: Dict):
        print("EXPERIMENT RESULTS SUMMARY")
        
        for exp_name, exp_results in results.items():
            print(f"\n{exp_name.upper().replace('_', ' ')}:")
            for model_name, model_results in exp_results.items():
                print(f"{model_name:20} | F1: {model_results['f1_macro']:.4f} | Acc: {model_results['test_accuracy']:.4f}")