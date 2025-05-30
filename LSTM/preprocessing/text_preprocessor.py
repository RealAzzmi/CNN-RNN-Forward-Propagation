import tensorflow as tf
import numpy as np
import re
from typing import List, Tuple, Optional
from tensorflow.keras.layers import TextVectorization
import pickle
import os

class IndonesianTextPreprocessor:    
    def __init__(self, 
                 max_tokens: int = 10000,
                 max_sequence_length: int = 128,
                 oov_token: str = "[UNK]"):
        self.max_tokens = max_tokens
        self.max_sequence_length = max_sequence_length
        self.oov_token = oov_token
        self.vectorizer = None
        
    def clean_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove non-alphabetic characters except spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vectorizer(self, texts: List[str]) -> TextVectorization:
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Create and configure vectorizer
        self.vectorizer = TextVectorization(
            max_tokens=self.max_tokens,
            output_sequence_length=self.max_sequence_length,
            output_mode='int',
            pad_to_max_tokens=False
        )
        
        # Adapt to training texts
        self.vectorizer.adapt(cleaned_texts)
        
        return self.vectorizer
    
    def preprocess_texts(self, texts: List[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise ValueError("Vectorizer not built. Call build_vectorizer first.")
        
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Vectorize
        vectorized = self.vectorizer(cleaned_texts)
        
        return vectorized.numpy()
    
    def get_vocabulary_size(self) -> int:
        if self.vectorizer is None:
            raise ValueError("Vectorizer not built.")
        return self.vectorizer.vocabulary_size()
    
    def get_vocabulary(self) -> List[str]:
        if self.vectorizer is None:
            raise ValueError("Vectorizer not built.")
        return self.vectorizer.get_vocabulary()
    
    def save_vectorizer(self, filepath: str):
        if self.vectorizer is None:
            raise ValueError("Vectorizer not built.")
            
        # Save vectorizer weights
        vectorizer_config = {
            'max_tokens': self.max_tokens,
            'max_sequence_length': self.max_sequence_length,
            'oov_token': self.oov_token,
            'vocabulary': self.get_vocabulary()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vectorizer_config, f)
            
        print(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath: str):
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        self.max_tokens = config['max_tokens']
        self.max_sequence_length = config['max_sequence_length']
        self.oov_token = config['oov_token']
        
        # Rebuild vectorizer
        self.vectorizer = tf.keras.utils.TextVectorization(
            max_tokens=self.max_tokens,
            output_sequence_length=self.max_sequence_length,
            output_mode='int',
            pad_to_max_tokens=False,
            vocabulary=config['vocabulary']
        )
        
        print(f"Vectorizer loaded from {filepath}")
    
    def get_preprocessing_info(self) -> dict:
        return {
            'max_tokens': self.max_tokens,
            'max_sequence_length': self.max_sequence_length,
            'vocabulary_size': self.get_vocabulary_size() if self.vectorizer else None,
            'oov_token': self.oov_token
        }

class TextStatistics:    
    @staticmethod
    def get_text_statistics(texts: List[str]) -> dict:
        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]
        
        return {
            'num_texts': len(texts),
            'avg_words': np.mean(word_counts),
            'min_words': np.min(word_counts),
            'max_words': np.max(word_counts),
            'std_words': np.std(word_counts),
            'avg_chars': np.mean(char_counts),
            'min_chars': np.min(char_counts),
            'max_chars': np.max(char_counts),
            'std_chars': np.std(char_counts)
        }
    
    @staticmethod
    def print_text_statistics(texts: List[str], dataset_name: str = "Dataset"):
        stats = TextStatistics.get_text_statistics(texts)
        
        print(f"\n=== {dataset_name} Text Statistics ===")
        print(f"Number of texts: {stats['num_texts']}")
        print(f"Average words per text: {stats['avg_words']:.2f}")
        print(f"Min/Max words: {stats['min_words']}/{stats['max_words']}")
        print(f"Word count std dev: {stats['std_words']:.2f}")
        print(f"Average characters per text: {stats['avg_chars']:.2f}")
        print(f"Min/Max characters: {stats['min_chars']}/{stats['max_chars']}")
        print(f"Character count std dev: {stats['std_chars']:.2f}")

if __name__ == "__main__":
    # Test the preprocessor
    from datasets.data_loader import NusaXDataLoader
    
    # Load data
    loader = NusaXDataLoader()
    train_df, valid_df, test_df = loader.load_dataset()
    
    train_texts, train_labels = loader.get_texts_and_labels(train_df)
    
    # Initialize preprocessor
    preprocessor = IndonesianTextPreprocessor(max_tokens=10000, max_sequence_length=128)
    
    # Build vectorizer
    vectorizer = preprocessor.build_vectorizer(train_texts)
    
    # Get statistics
    TextStatistics.print_text_statistics(train_texts, "Training")
    
    # Print preprocessing info
    print(f"\nPreprocessing info: {preprocessor.get_preprocessing_info()}")
    
    # Test preprocessing
    sample_texts = train_texts[:3]
    vectorized = preprocessor.preprocess_texts(sample_texts)
    print(f"\nSample vectorized shape: {vectorized.shape}")
    print(f"Sample vectorized text:\n{vectorized[0]}")
    