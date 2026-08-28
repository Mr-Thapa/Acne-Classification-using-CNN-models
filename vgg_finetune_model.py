import tensorflow as tf
from config import IMG_SIZE

def create_model():
    model=tf.keras.models.load_model("vgg16_trained.keras")
    vgg=model.get_layer("vgg16")
    
    #Unfreeze Block 5
    for layer in vgg.layers:
        if layer.name.startswith("block5_"):
            layer.trainable=True
            
    if __name__=="__main__":
        for layer in vgg.layers:
            print(layer.name,":",layer.trainable)
    return model
if __name__ == "__main__":
    model = create_model()
    model.summary()
    

    
