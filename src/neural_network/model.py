"""Simple Keras model for demonstration (MLP)."""
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    m = build_model(5,5)
    m.summary()
    m.save('models/untrained_model.h5')
    print('Saved models/untrained_model.h5')