import os
import tensorflow as tf
from config import Data_DIR,BATCH_SIZE, IMG_SIZE


train_path=f"{Data_DIR}/train"
valid_path=f"{Data_DIR}/valid"
test_path=f"{Data_DIR}/test"

train_dataset=tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset=tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

valid_dataset=tf.keras.utils.image_dataset_from_directory(
    valid_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
class_names = train_dataset.class_names
if __name__ == "__main__":
    print(train_dataset.class_names)
    for images,labels in train_dataset.take(1):
        print("Images shape:",images.shape)
        print("Labels shape:",labels.shape)
        print("Labels:",labels.numpy())
    
    import matplotlib.pyplot as plt
    for images,labels in train_dataset.take(1):
        plt.figure(figsize=(10,10))
        
        for i in range(9):
            ax=plt.subplot( 3,3,i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(train_dataset.class_names[labels[i]])
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    for images, labels in train_dataset.take(1):
        print(images[0])
    for images, labels in train_dataset.take(1):
        print("Shape:", images.shape)
        print("Minimum:", tf.reduce_min(images).numpy())
        print("Maximum:", tf.reduce_max(images).numpy())


    from collections import Counter

    def count_classes(dataset, class_names):
        counts = Counter()

        for images, labels in dataset:
            counts.update(labels.numpy())

        for class_id, count in sorted(counts.items()):
            print(class_names[class_id], ":", count)


    print("TRAIN")
    count_classes(train_dataset, class_names)

    print("\nVALIDATION")
    count_classes(valid_dataset, class_names)

    print("\nTEST")
    count_classes(test_dataset, class_names)

