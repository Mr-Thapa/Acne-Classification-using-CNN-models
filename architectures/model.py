import tensorflow as tf
from src.config import IMG_SIZE


def create_model():
    model=tf.keras.Sequential([
        tf.keras.Input(shape=(*IMG_SIZE,3)),
        
        #Data augmentation
        #tf.keras.layers.RandomRotation(0.03),
        #tf.keras.layers.RandomFlip("horizontal"),
        #tf.keras.layers.RandomTranslation(height_factor=0.05,width_factor=0.05),
        #tf.keras.layers.RandomBrightness(factor=0.1),
        #Conv layers
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(
            32,(3,3),activation="relu"
        ),
        tf.keras.layers.Conv2D(
            32,(3,3),activation="relu"
            ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2)
        ),
        tf.keras.layers.Conv2D(
            64,(3,3),activation="relu"
        ),
        tf.keras.layers.Conv2D(
            64,(3,3),activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2)
        ),
         tf.keras.layers.Conv2D(
            128,(3,3),activation="relu"
        ),
        tf.keras.layers.Conv2D(
            128,(3,3),activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            128,
            activation="relu",
            #kernel_regularizer=tf.keras.regularizers.l2(0.0001)
                              ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5,activation="softmax")
    ])
    return model
if __name__ == "__main__":
    model = create_model()
    model.summary()