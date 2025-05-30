import numpy as np
import tensorflow as tf
from load_cifar10 import LoadCifar10
from cnn_experiments import CNNExperiments
from cnn_custom_forward_propagation import test_cnn_custom_forward_propagation
from config import RANDOM_SEED

# 0. Manually seed numpy and tenserflow for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 1. Load and prepare data
print("Step 1: Loading and preparing data...")
data_loader = LoadCifar10()
data_loader.load_and_prepare_data()
data_splits = data_loader.get_data_splits()

# 2. Run hyperparameter experiments
print("\nStep 2: Running hyperparameter experiments and saving the weights and architecture...")
experiments = CNNExperiments(data_splits)
experiments.run_all_experiments()

# 3. Test CNN custom forward propagation
print("\nStep 3: Testing CNN custom forward propagation...")
test_cnn_custom_forward_propagation()

print("Finished...")
print("- Models saved in 'models/' folder")
print("- Plots saved in 'plots/' folder")
print("- Results saved in 'experiment_results.json'")