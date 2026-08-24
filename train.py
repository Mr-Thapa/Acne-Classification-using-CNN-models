from DataLoad import train_dataset,test_dataset,valid_dataset
from model import create_model
import tensorflow as tf

model=create_model()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "baseline.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    mode="max",
    restore_best_weights=True
)
model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset,
    verbose=2,
    callbacks=[checkpoint,early_stopping]
)
model.save("baseline.keras")