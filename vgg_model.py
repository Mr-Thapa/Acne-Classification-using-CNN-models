import tensorflow as tf
from config import IMG_SIZE

def create_model():
    vgg=tf.keras.applications.VGG16(
        input_shape=(*IMG_SIZE,3),
        weights="imagenet",
        include_top=False
    )
    vgg.trainable=False
    inputs=tf.keras.Input(shape=(*IMG_SIZE,3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x=vgg(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    output=tf.keras.layers.Dense(5,activation="softmax")(x)
    model=tf.keras.Model(inputs,output)

    return model
if __name__ == "__main__":
    model = create_model()
    model.summary()
    

    
