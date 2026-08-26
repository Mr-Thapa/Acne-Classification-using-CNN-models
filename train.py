from DataLoad import train_dataset,test_dataset,valid_dataset
import tensorflow as tf
import sys

#choosing the model to train
choice=input('''Choose which model to train:
                    1. Baseline CNN
                    2. VGG-16
                    Enter any other value to exit
                 
                 ''')
if choice=="1":
    from model import create_model
    LR=0.001
    model_name="baseline.keras"
    
elif choice=="2":
    from vgg_model import create_model
    LR=0.0005
    model_name="vgg16_trained.keras"
    
else:
    print("Exiting....")
    sys.exit()

print(f"Training {model_name}")

#Model initialization
model=create_model()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    metrics=["accuracy"]
)

#ensures best performance model is saved
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_name,
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
    epochs=10,
    validation_data=valid_dataset,
    verbose=2,
    callbacks=[checkpoint,early_stopping]
)
#explicit model save
model.save(model_name)