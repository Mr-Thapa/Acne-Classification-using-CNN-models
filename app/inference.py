import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE,FINAL_MODEL

CLASS_NAMES=[
    "Blackheads",
    "Cyst",
    "Papules",
    "Pustules",
    "Whiteheads"
]

model=tf.keras.models.load_model(FINAL_MODEL)
class Invalid_Image_Exception(Exception):
    pass
def predict(image):
    try:
        image = tf.image.decode_image(image, channels=3,expand_animations=False)
    except tf.errors.InvalidArgumentError as e:
        raise Invalid_Image_Exception("Uploaded file is not a valid image") from e
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.expand_dims(image, axis=0)
    image = tf.ensure_shape(
    image,
    (1, IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    predictions=model.predict(image,verbose=0)
    predicted_index=np.argmax(predictions[0])
    confidence=predictions[0][predicted_index]
    return{
        "class":CLASS_NAMES[predicted_index],
        "confidence":float(confidence)
    }
if __name__=="__main__":
    import sys
    image=tf.io.read_file(sys.argv[1])
    result=predict(image)
    print(result)