# CNN CIFAR-10 Classification Project

Proyek ini mengimplementasikan Convolutional Neural Network (CNN) untuk klasifikasi dataset CIFAR-10 dengan eksperimen hyperparameter & arsitektur.

## Deskripsi Singkat

CNN/ berisi implementasi CNN yang melakukan:
- **Klasifikasi CIFAR-10**: Mengenali 10 kelas objek (pesawat, mobil, burung, kucing, rusa, anjing, katak, kuda, kapal, truk)
- **Eksperimen Hyperparameter**: Menguji efek dari jumlah layer konvolusi, jumlah filter, ukuran filter, dan tipe pooling
- **Custom Forward Propagation**: Implementasi manual forward pass CNN

### Eksperimen yang Dilakukan
1. **Jumlah Layer Konvolusi**: 2, 3, dan 4 layer
2. **Jumlah Filter**: Small (16,32,64), Medium (32,64,128), Large (64,128,256)
3. **Ukuran Filter**: Small kernels (3x3), Medium kernels (5x5), Mixed kernels (3x3, 5x5, 7x7)
4. **Tipe Pooling**: Max Pooling vs Average Pooling

## Setup Environment

### Requirements
- Python 3.10.16
- Conda package manager

### Instalasi

1. **Clone repository**
```bash
git clone <repository-url>
cd CNN
```

2. **Buat conda environment**
```bash
conda create -n cnn-project python=3.10.16
conda activate cnn-project
```

3. **Install dependencies**

**Menggunakan conda (recommended):**
```bash
# Core ML packages
conda install -c conda-forge tensorflow
conda install -c conda-forge scikit-learn
conda install -c conda-forge numpy

# Visualization
conda install -c conda-forge matplotlib

# Additional packages (jika tidak terinstall otomatis dengan tensorflow)
conda install -c conda-forge keras
```

**Alternatif menggunakan pip:**
```bash
pip install tensorflow
pip install scikit-learn
pip install matplotlib
pip install numpy
```


## Cara Menjalankan Program

### Menjalankan Semua Eksperimen
```bash
python main.py
```

Program akan menjalankan:
1. Loading dan preprocessing dataset CIFAR-10
2. Training model untuk semua eksperimen hyperparameter
3. Testing custom forward propagation implementation
4. Menyimpan hasil dan visualisasi

## Struktur Project

```
CNN/
├── main.py                           # Entry point utama
├── config.py                         # Konfigurasi dan hyperparameter
├── load_cifar10.py                   # Data loading dan preprocessing
├── compile_cnn_archictecture.py      # Pembuatan arsitektur CNN
├── cnn_model_trainer.py              # Training dan evaluasi model
├── cnn_experiments.py                # Orchestrator eksperimen
├── cnn_custom_forward_propagation.py # Implementasi custom forward pass
├── experiment_visualizer.py          # Visualisasi hasil eksperimen
├── experiment_results.json           # Hasil eksperimen (auto-generated)
├── models/                           # Model weights dan arsitektur (auto-generated)
│   ├── *.weights.h5                  # Saved weights
│   └── *.json                        # Model architecture
└── plots/                            # Grafik hasil eksperimen (auto-generated)
    └── *_comparison.png              # Training curves dan F1 score comparison
```

## Output dan Hasil

### Files yang Dihasilkan
- **`experiment_results.json`**: Hasil lengkap semua eksperimen (akurasi, loss, F1-score)
- **`models/`**: Model weights (.h5) dan arsitektur (.json) untuk setiap eksperimen
- **`plots/`**: Grafik perbandingan training loss, validation loss, dan F1 scores
