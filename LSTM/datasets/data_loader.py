import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os

class NusaXDataLoader:    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.num_classes = 3
        
    def load_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(self.data_dir, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        # Convert string labels to integers
        train_df['label'] = train_df['label'].map(self.label_to_id)
        valid_df['label'] = valid_df['label'].map(self.label_to_id)
        test_df['label'] = test_df['label'].map(self.label_to_id)
        
        return train_df, valid_df, test_df
    
    def get_texts_and_labels(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        texts = df['text'].tolist()
        labels = df['label'].values
        return texts, labels
    
    def get_dataset_info(self) -> Dict:
        train_df, valid_df, test_df = self.load_dataset()
        
        return {
            'train_size': len(train_df),
            'valid_size': len(valid_df),
            'test_size': len(test_df),
            'num_classes': self.num_classes,
            'classes': list(self.label_to_id.keys()),
            'train_distribution': train_df['label'].value_counts().to_dict(),
            'valid_distribution': valid_df['label'].value_counts().to_dict(),
            'test_distribution': test_df['label'].value_counts().to_dict()
        }
    
    def print_dataset_info(self):
        """Print dataset information"""
        info = self.get_dataset_info()
        print("=== NusaX Sentiment Dataset Information ===")
        print(f"Classes: {info['classes']}")
        print(f"Train samples: {info['train_size']}")
        print(f"Validation samples: {info['valid_size']}")
        print(f"Test samples: {info['test_size']}")
        print(f"Total samples: {info['train_size'] + info['valid_size'] + info['test_size']}")
        
        print(f"\nTrain distribution: {info['train_distribution']}")
        print(f"Valid distribution: {info['valid_distribution']}")
        print(f"Test distribution: {info['test_distribution']}")

if __name__ == "__main__":
    # Test the data loader
    loader = NusaXDataLoader()
    loader.print_dataset_info()
    
    # Load datasets for testing
    train_df, valid_df, test_df = loader.load_dataset()
    print(f"\nSample texts:")
    print(f"Train: {train_df['text'].iloc[0]}")
    print(f"Label: {train_df['label'].iloc[0]} ({loader.id_to_label[train_df['label'].iloc[0]]})")