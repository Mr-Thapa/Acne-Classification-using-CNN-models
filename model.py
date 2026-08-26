import tensorflow as tf
from config import IMG_SIZE


def create_model():
    model=tf.keras.Sequential([
        tf.keras.Input(shape=(*IMG_SIZE,3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(
            32,(3,3),activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2)
        ),
        tf.keras.layers.Conv2D(
            64,(3,3),activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            128,activation="relu"
        ),
        tf.keras.layers.Dense(5,activation="softmax")
    ])
    return model
if __name__ == "__main__":
    model = create_model()
    model.summary()