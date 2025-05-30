# CNN, RNN, dan LSTM

## Deskripsi
Repository ini berisi implementasi **CNN, RNN, dan LSTM** dalam memprediksi label dari suatu dataset dengan menggunakan library Keras maupun membuat model dari awal (untuk forward propagation) tanpa menggunakan library deep learning seperti TensorFlow atau PyTorch. Proyek ini bertujuan untuk memahami cara kerja CNN, RNN, dan LSTM, dan juga pengaruh berbagai parameter model dalam memprediksi dataset. 

Pada repository ini terdapat tiga implementasi, yaitu implementasi CNN untuk memprediksi dataset CIFAR-10 (gambar), serta implementasi Simple RNN dan LSTM untuk memprediksi label dataset kalimat NusaX-Sentiment (Bahasa Indonesia).
  
## Cara Setup dan Menjalankan Program

### Prasyarat
Pastikan Anda telah menginstal Python 3.x. Program ini juga membutuhkan beberapa package Python yang bisa diinstal melalui `pip`. Pastikan juga untuk memiliki berbagai library Machine Learning seperti Numpy, ScikitLearn, dan Matplotlib.

### Langkah-langkah Setup:
1. **Clone repository ini**:
   ```bash
   git clone https://github.com/RealAzzmi/CNN-RNN-Forward-Propagation.git
   cd CNN-RNN-Forward-Propagation
   ```

2. **Gunakan Virtual Environment**
   Saran: Gunakan virtual environment pada windows dan install library yang dibutuhkan.
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install numpy matplotlib scikit-learn tensorflow
   ```
3. **Jalankan program**:
   Untuk menjalankan model dan melakukan pelatihan, masuk ke folder CNN atau RNN. Dapat melakukan _run_ dari notebook maupun kode python tergantung implementasi

## Pembagian Tugas Anggota Kelompok

| **NIM**    | **Nama**                      | **Tugas**                 |
|-----------|-------------------------------|---------------------------|
| 13522069  | Nabila Shikoofa Muida         | Implementasi LSTM         |
| 13522087  | Shulha                        | Implementasi RNN          |
| 13522109  | Azmi Mahmud Bazeid            | Implementasi CNN          |
