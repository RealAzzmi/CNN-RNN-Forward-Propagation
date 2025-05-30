import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class TextPreprocessor:
    def __init__(self, vocab_size=5000, max_sequence_length=55, embedding_dim=128):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        
        self.vectorizer = TextVectorization(
            max_tokens=vocab_size,
            output_sequence_length=max_sequence_length,
            output_mode='int',
            split='whitespace',
            ngrams=None,
            standardize='lower_and_strip_punctuation'
        )
        
        self.label_encoder = LabelEncoder()
        
        self.vocab_size_actual = None
        self.num_classes = 3
        
    def fit_on_texts(self, texts, labels):
        cleaned_texts = [str(text).strip() for text in texts if str(text).strip()]
        
        self.vectorizer.adapt(cleaned_texts)
        
        vocab = self.vectorizer.get_vocabulary()
        self.vocab_size_actual = len(vocab)
        
        self.label_encoder.fit(labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"Vocabulary size: {self.vocab_size_actual}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"First 10 vocabulary words: {vocab[:10]}")
        
        if len(cleaned_texts) > 0:
            sample_text = cleaned_texts[0]
            sample_tokens = self.vectorizer([sample_text])
            print(f"\nSample tokenization:")
            print(f"Text: {sample_text}")
            print(f"Tokens: {sample_tokens.numpy()[0][:20]}...")
        
    def preprocess_texts(self, texts):
        return self.vectorizer(texts)
    
    def encode_labels(self, labels):
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels):
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_embedding_layer(self, mask_zero=True):
        return Embedding(
            input_dim=self.vocab_size_actual,
            output_dim=self.embedding_dim,
            mask_zero=mask_zero,
            name='embedding'
        )
    
    def load_nusax_data(self, train_path, test_path, valid_path):
        def read_csv_data(file_path):
            try:
                data = pd.read_csv(file_path)

                text_col = 'text'  
                label_col = 'label' 
                
                texts = data[text_col].astype(str).tolist()
                labels = data[label_col].astype(str).tolist()
                
                texts = [text.strip() for text in texts if text.strip() and text.strip() != 'nan']
                labels = [label.strip() for label in labels if label.strip() and label.strip() != 'nan']
                
                return texts, labels
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return [], []
        
        train_texts, train_labels = read_csv_data(train_path)
        valid_texts, valid_labels = read_csv_data(valid_path)
        test_texts, test_labels = read_csv_data(test_path)
        
        data_dict = {
            'train': {'texts': train_texts, 'labels': train_labels},
            'valid': {'texts': valid_texts, 'labels': valid_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }
        
        return data_dict
    
    def prepare_dataset(self, texts, labels, batch_size=32, shuffle=True):
        tokenized_texts = self.preprocess_texts(texts)
        encoded_labels = self.encode_labels(labels)
        
        dataset = tf.data.Dataset.from_tensor_slices((tokenized_texts, encoded_labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(texts))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_vocab_info(self):
        vocab = self.vectorizer.get_vocabulary()
        return {
            'vocab_size': len(vocab),
            'first_10_tokens': vocab[:10] if len(vocab) > 10 else vocab,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim
        }