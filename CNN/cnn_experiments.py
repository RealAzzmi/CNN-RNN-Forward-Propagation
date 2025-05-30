import json
from compile_cnn_archictecture import CompileCNNArchitecture
from cnn_model_trainer import CNNModelTrainer
from experiment_visualizer import ExperimentVisualizer
from config import CONV_LAYERS_EXPERIMENTS, NUM_FILTERS_EXPERIMENTS, FILTER_SIZES_EXPERIMENTS, POOLING_TYPES_EXPERIMENTS, RESULTS_FILE

class CNNExperiments:
    def __init__(self, data_splits):
        self.trainer = CNNModelTrainer(data_splits)
        self.model_builder = CompileCNNArchitecture()
        self.visualizer = ExperimentVisualizer()
        self.results = {}
        
    def experiment_conv_layers(self):
        print("EXPERIMENT 1: Effect of Number of Convolutional Layers")
        
        results = {}
        for exp in CONV_LAYERS_EXPERIMENTS:
            model = self.model_builder.create_cnn_model(
                conv_layers=exp["conv_layers"],
                filters_per_layer=exp["filters"],
                filter_sizes=exp["filter_sizes"]
            )
            results[exp["name"]] = self.trainer.train_model(model, exp["name"])
        
        self.results["conv_layers_experiment"] = results
        self.visualizer.plot_comparison(results, "Number of Convolutional Layers", "conv_layers")
        
    def experiment_num_filters(self):
        print("EXPERIMENT 2: Effect of Number of Filters per Layer")
        
        results = {}
        for exp in NUM_FILTERS_EXPERIMENTS:
            model = self.model_builder.create_cnn_model(
                conv_layers=3,
                filters_per_layer=exp["filters"],
                filter_sizes=exp["filter_sizes"]
            )
            results[exp["name"]] = self.trainer.train_model(model, exp["name"])
        
        self.results["num_filters_experiment"] = results
        self.visualizer.plot_comparison(results, "Number of Filters per Layer", "num_filters")
        
    def experiment_filter_sizes(self):
        print("EXPERIMENT 3: Effect of Filter Sizes")
        
        results = {}
        for exp in FILTER_SIZES_EXPERIMENTS:
            model = self.model_builder.create_cnn_model(
                conv_layers=3,
                filters_per_layer=exp["filters"],
                filter_sizes=exp["filter_sizes"]
            )
            results[exp["name"]] = self.trainer.train_model(model, exp["name"])
        
        self.results["filter_sizes_experiment"] = results
        self.visualizer.plot_comparison(results, "Filter Sizes", "filter_sizes")
        
    def experiment_pooling_types(self):
        print("EXPERIMENT 4: Effect of Pooling Layer Types")
        
        results = {}
        for exp in POOLING_TYPES_EXPERIMENTS:
            model = self.model_builder.create_cnn_model(
                conv_layers=3,
                filters_per_layer=[32, 64, 128],
                filter_sizes=[3, 3, 3],
                pooling_type=exp["pooling_type"]
            )
            results[exp["name"]] = self.trainer.train_model(model, exp["name"])
        
        self.results["pooling_types_experiment"] = results
        self.visualizer.plot_comparison(results, "Pooling Layer Types", "pooling_types")
        
    def run_all_experiments(self):
        print("Starting the experiments...")
        
        self.experiment_conv_layers()
        self.experiment_num_filters()
        self.experiment_filter_sizes()
        self.experiment_pooling_types()
        
        self._save_results()
        self.visualizer.print_summary(self.results)
        
    def _save_results(self):
        serializable_results = {}
        for exp_name, exp_results in self.results.items():
            serializable_results[exp_name] = {}
            for model_name, model_results in exp_results.items():
                serializable_results[exp_name][model_name] = {
                    'test_accuracy': float(model_results['test_accuracy']),
                    'test_loss': float(model_results['test_loss']),
                    'f1_macro': float(model_results['f1_macro']),
                    'model_path': model_results['model_path'],
                    'architecture_path': model_results['architecture_path']
                }
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Experiment results saved to: {RESULTS_FILE}")