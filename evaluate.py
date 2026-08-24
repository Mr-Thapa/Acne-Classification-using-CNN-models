import tensorflow as tf
from DataLoad import test_dataset,class_names
model=tf.keras.models.load_model("baseline.keras")
from sklearn.metrics import classification_report

loss,accuracy=model.evaluate(
    test_dataset,
    verbose=2
)

print("Loss=",loss)
print("Accuracy=",accuracy)

predictions = model.predict(test_dataset)
predicted_labels = tf.argmax(predictions, axis=1).numpy()
true_labels = []

for images, labels in test_dataset:
    true_labels.extend(labels.numpy())

true_labels = tf.convert_to_tensor(true_labels).numpy()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names
)

print(report)