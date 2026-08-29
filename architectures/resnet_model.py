import tensorflow as tf
from src.config import IMG_SIZE



def create_model():
    input=tf.keras.layers.Input(shape=(*IMG_SIZE,3))
    resnet=tf.keras.applications.ResNet50(
        input_shape=(*IMG_SIZE,3),
        weights="imagenet",
        include_top=False
    )
    resnet.trainable=False
    x=tf.keras.applications.resnet50.preprocess_input(input)
    x=resnet(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    output=tf.keras.layers.Dense(5,activation="softmax")(x)
    model=tf.keras.Model(input,output)
    
    return model

if __name__=="__main__":
    model=create_model()
    print(model.summary())