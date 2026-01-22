"""Simple Keras model for demonstration (MLP)."""
# -----------------------------------------------------------------------------
# Modul: src/neural_network/model.py
# Scop: definește un MLP simplu pentru date tabulare.
# Input: input_dim, num_classes.
# Output: model Keras compilat + posibilitate de salvare.
# Utilizare: import build_model(...) sau rulare directă.
# Pași principali:
#   1) Construire arhitectură MLP.
#   2) Compilare cu Adam + loss classification.
#   3) Salvare model neantrenat (opțional).
# Dependențe: tensorflow.keras.
# Parametri implicați: input_dim, num_classes.
# Fișiere scrise: models/untrained_model.h5 (doar în __main__).
# Observații: folosit ca schelet pentru etapa 4.
# -----------------------------------------------------------------------------
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