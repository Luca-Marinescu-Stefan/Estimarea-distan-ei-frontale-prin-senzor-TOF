# Modelul Neuronal (MLP)
import tensorflow as tf

def creare_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print("Modelul MLP a fost creat.")
    return model

if __name__ == "__main__":
    creare_model()
