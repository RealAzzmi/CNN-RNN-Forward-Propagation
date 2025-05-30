import os

RANDOM_SEED = 42

# Data info

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Include a third split according to spec
VALIDATION_SIZE = 10000

# Training config
# (running on kaggle/collab is necessary) 
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DROPOUT_RATE = 0.5

MODELS_DIR = "models"
PLOTS_DIR = "plots"
RESULTS_FILE = "experiment_results.json"

################################
# Experiment configurations    #
################################

# 1. The effect of the number of convolutional layers
CONV_LAYERS_EXPERIMENTS = [
    {"name": "2_conv_layers", "conv_layers": 2, "filters": [32, 64], "filter_sizes": [3, 3]},
    {"name": "3_conv_layers", "conv_layers": 3, "filters": [32, 64, 128], "filter_sizes": [3, 3, 3]},
    {"name": "4_conv_layers", "conv_layers": 4, "filters": [32, 64, 128, 256], "filter_sizes": [3, 3, 3, 3]}
]

# 2. The effect of the number of filters per convolutional layer
NUM_FILTERS_EXPERIMENTS = [
    {"name": "small_filters", "filters": [16, 32, 64], "filter_sizes": [3, 3, 3]},
    {"name": "medium_filters", "filters": [32, 64, 128], "filter_sizes": [3, 3, 3]},
    {"name": "large_filters", "filters": [64, 128, 256], "filter_sizes": [3, 3, 3]}
]

# 3. The effect of the filter sizes per convolutional layer
FILTER_SIZES_EXPERIMENTS = [
    {"name": "small_kernels", "filters": [32, 64, 128], "filter_sizes": [3, 3, 3]},
    {"name": "medium_kernels", "filters": [32, 64, 128], "filter_sizes": [5, 5, 5]},
    {"name": "mixed_kernels", "filters": [32, 64, 128], "filter_sizes": [3, 5, 7]}
]

# 4. The effect of max & avg pooling layer
POOLING_TYPES_EXPERIMENTS = [
    {"name": "max_pooling", "pooling_type": "max"},
    {"name": "average_pooling", "pooling_type": "average"}
]

# Checking for models/ & plots/ folder
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)