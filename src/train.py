from src.DataLoad import train_dataset, valid_dataset
from src.config import CHECKPOINT_DIR

import tensorflow as tf
import sys

#choosing the model to train
choice=input('''Choose which model to train:
                    1. Baseline CNN
                    2. VGG-16
                    3. Finetune VGG-16
                    4. ResNet50
                    5. Finetune ResNet50
                    Enter any other value to exit
                 
                 ''')
if choice=="1":
    from architectures.model import create_model
    LR=0.001
    model_name="baseline.keras"
    EPOCHS=20

elif choice=="2":
    from architectures.vgg_model import create_model
    LR=0.0005
    model_name="vgg16_trained.keras"
    EPOCHS=20

elif choice=="3":
    from architectures.vgg_finetune_model import create_model
    LR=0.00005
    model_name="vgg_finetuned_trained.keras"
    EPOCHS=20
    
elif choice=="4":
    from architectures.resnet_model import create_model
    LR=0.0001
    model_name="resnet_trained.keras"
    EPOCHS=20
elif choice=="5":
    from architectures.resnet_finetune_model import create_model
    LR=0.00001
    model_name="resnet_finetuned_trained.keras"
    EPOCHS=20
else:
    print("Exiting....")
    sys.exit()

print(f"Training {model_name}")

#Model initialization
model=create_model()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    #optimizer = tf.keras.optimizers.SGD(
    #    learning_rate=1e-3,
    #    momentum=0.9,
    #    nesterov=True
    #), 
    metrics=["accuracy"]
)

#ensures best performance model is saved
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_DIR/model_name,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)
#ensures we stop early if the model stops improving 
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    mode="max",
    restore_best_weights=True
)
#train model
model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    verbose=2,
    callbacks=[checkpoint,early_stopping]
)
#explicit model save
model.save(CHECKPOINT_DIR/model_name)