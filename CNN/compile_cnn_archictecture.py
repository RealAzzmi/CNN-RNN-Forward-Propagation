from tensorflow import keras
from keras import layers
from typing import List
from config import INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE

class CompileCNNArchitecture:
    def create_cnn_model(conv_layers: List[int], filters_per_layer: List[int], filter_sizes: List[int], pooling_type: str = 'max') -> keras.Model:
        model = keras.Sequential()
        
        # Add input layer
        model.add(layers.Input(shape=INPUT_SHAPE))
        
        # Add convolutional layers
        for i, (num_filters, filter_size) in enumerate(zip(filters_per_layer, filter_sizes)):
            model.add(layers.Conv2D(num_filters, (filter_size, filter_size), activation='relu', padding='same'))
            

            # Skip pooling for later layers in deep networks to be more conservative
            # Also be careful of size mismatch
            if i < len(filters_per_layer) - 1 or len(filters_per_layer) <= 3:
                if pooling_type == 'max':
                    model.add(layers.MaxPooling2D((2, 2)))
                else:
                    model.add(layers.AveragePooling2D((2, 2)))
        
        # Trying global average pooling 
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(64, activation='relu'))
        # Dropout is usually added after dense non-linear activation (see https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network) 
        model.add(layers.Dropout(DROPOUT_RATE))
        model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
        
        # Compile
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model