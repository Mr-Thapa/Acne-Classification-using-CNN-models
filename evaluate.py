import tensorflow as tf
from DataLoad import test_dataset, class_names
from sklearn.metrics import classification_report, confusion_matrix


models = {
    "Baseline CNN": "baseline.keras",
    "VGG16": "vgg16_trained.keras",
    "ResNet50":"resnet_trained.keras"
}


# Collect true labels once.
# test_dataset has shuffle=False, so the order is fixed.
true_labels = []

for images, labels in test_dataset:
    true_labels.extend(labels.numpy())

true_labels = tf.convert_to_tensor(true_labels).numpy()


# Evaluate each model
for model_name, model_path in models.items():

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    model = tf.keras.models.load_model(model_path)

    # Loss and accuracy
    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=2
    )

    print("Loss =", loss)
    print("Accuracy =", accuracy)

    # Predictions
    predictions = model.predict(
        test_dataset,
        verbose=0
    )

    predicted_labels = tf.argmax(
        predictions,
        axis=1
    ).numpy()

    # Confusion matrix
    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )

    print("\nClassification Report:")
    print(report)
