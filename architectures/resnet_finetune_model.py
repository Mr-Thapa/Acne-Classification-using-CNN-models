import tensorflow as tf
from src.config import CHECKPOINT_DIR
def create_model():
    model=tf.keras.models.load_model(CHECKPOINT_DIR/"resnet_trained.keras")
    resnet=model.get_layer("resnet50")
    
    for layer in resnet.layers:
        if layer.name.startswith("conv5"):
            layer.trainable=True
            
    if __name__=="__main__":
        for layer in resnet.layers:
            print(layer.name,":",layer.trainable)
    return model
if __name__=="__main__":
    create_model()