# LSTM-based Text Classification for Indonesian Sentiment Analysis

## Course Information
- **Course**: IF3270 - Machine Learning
- **Dataset**: NusaX-Sentiment (Indonesian)
- **Task**: Multi-class sentiment classification (negative, neutral, positive)

## Project Overview

This project implements a complete LSTM-based text classification system with two main components:
1. **Keras-based LSTM training** with comprehensive experiments
2. **NumPy-only forward propagation** implementation from scratch

## Dataset Structure

Your dataset should contain three CSV files:
- `train.csv` (500 samples)
- `valid.csv` (100 samples) 
- `test.csv` (400 samples)

Each file must have columns: `id`, `text`, `label`

## Project Structure

```
├── datasets/
│   └── data_loader.py          # Data loading utilities
├── preprocessing/
│   └── text_preprocessor.py    # Text preprocessing and vectorization
├── models/
│   ├── lstm_keras.py          # Keras LSTM implementation
│   └── lstm_scratch.py        # NumPy scratch implementation
├── main.py                    # Main experiment runner
├── requirements.txt           # Python dependencies
├── train.csv                  # Training data
├── valid.csv                  # Validation data
├── test.csv                   # Test data
└── results/                   # Generated results (created automatically)
    ├── experiment_summary.txt
    ├── FINAL_PROJECT_REPORT.txt
    ├── lstm_layers_1/         # Individual experiment results
    ├── lstm_layers_2/
    ├── lstm_layers_3/
    ├── lstm_units_32/
    ├── lstm_units_64/
    ├── lstm_units_128/
    ├── lstm_bidirectional/
    ├── lstm_unidirectional/
    └── scratch_comparison/
        ├── model_comparison.json
        ├── scratch_model_weights.npz
        └── scratch_evaluation.txt
```

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Complete Pipeline (Recommended)

Run all experiments and scratch implementation:
```bash
python main.py
```

With custom parameters:
```bash
python main.py --data_dir . --results_dir results --max_tokens 10000 --max_sequence_length 128
```

### Keras Experiments Only

```bash
python main.py --skip_scratch
```

## Experiments Configuration

The system automatically runs 8 different experiments:

### 1. LSTM Layers Variation (3 experiments)
- `lstm_layers_1`: Single LSTM layer
- `lstm_layers_2`: Two LSTM layers
- `lstm_layers_3`: Three LSTM layers

### 2. LSTM Units Variation (3 experiments)  
- `lstm_units_32`: 32 units per layer
- `lstm_units_64`: 64 units per layer
- `lstm_units_128`: 128 units per layer

### 3. Bidirectional Comparison (2 experiments)
- `lstm_unidirectional`: Standard LSTM
- `lstm_bidirectional`: Bidirectional LSTM

## Model Architecture

**Base Configuration:**
- Embedding dimension: 128
- Dropout rate: 0.3
- Dense layer units: 32
- Optimizer: Adam
- Loss function: SparseCategoricalCrossentropy
- Evaluation metric: Macro F1-score

## Output Verification
Check these files after completion:
- `results/experiment_summary.txt` - All experiment rankings
- `results/FINAL_PROJECT_REPORT.txt` - Complete project summary
- `results/scratch_comparison/model_comparison.json` - Implementation comparison
- Individual experiment folders with plots and metrics