import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from src.DataLoad import test_dataset, class_names
from src.config import CHECKPOINT_DIR
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# Configuration
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(
    style="whitegrid",
    context="notebook"
)


models = {
    "Baseline CNN": "baseline.keras",
    "VGG16": "vgg16_trained.keras",
    "ResNet50": "resnet_trained.keras",
    "VGG16 Fine Tuned": "vgg_finetuned_trained.keras",
    "ResNet50 Fine Tuned": "resnet_finetuned_trained.keras"
}


# Collect true labels once
# test_dataset uses shuffle=False, so the order is fixed.
true_labels = []

for images, labels in test_dataset:
    true_labels.extend(labels.numpy())
true_labels = np.array(true_labels)
print(f"Number of test images: {len(true_labels)}")
print(f"Classes: {class_names}")

results = []



# Evaluate each model
for model_name, model_path in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    model = tf.keras.models.load_model(
        CHECKPOINT_DIR / model_path
    )
    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=2
    )

    print(f"Loss     = {loss:.4f}")
    print(f"Accuracy = {accuracy:.4f}")
    predictions = model.predict(
        test_dataset,
        verbose=0
    )
    
    predicted_labels = np.argmax(
        predictions,
        axis=1
    )
    
    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True
    )

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names
        )
    )

    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    results.append({
        "Model": model_name,
        "Test Accuracy": accuracy,
        "Loss": loss,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })


#confusion matrix
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )

    plt.title(
        f"{model_name}\nConfusion Matrix",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel(
        "Predicted Label",
        fontsize=11
    )

    plt.ylabel(
        "True Label",
        fontsize=11
    )

    plt.xticks(rotation=30)
    plt.yticks(rotation=0)

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    #Per-Class F1 Score
    f1_scores = [
        report[class_name]["f1-score"]
        for class_name in class_names
    ]

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        class_names,
        f1_scores,
        color="#4C78A8"
    )

    plt.ylim(0, 1)

    plt.title(
        f"{model_name}\nPer-Class F1 Score",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel(
        "Class",
        fontsize=11
    )

    plt.ylabel(
        "F1 Score",
        fontsize=11
    )

    plt.xticks(rotation=20)

    # Add values above bars
    for bar, score in zip(bars, f1_scores):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_f1_scores.png"
    )

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    
# Results DataFrame
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# Save Results
results_df.to_csv(
    RESULTS_DIR / "model_results.csv",
    index=False
)


# Find Best Model
best_index = results_df["Test Accuracy"].idxmax()

best_model = results_df.loc[
    best_index,
    "Model"
]

best_accuracy = results_df.loc[
    best_index,
    "Test Accuracy"
]

best_f1 = results_df.loc[
    best_index,
    "Macro F1"
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model       : {best_model}")
print(f"Accuracy    : {best_accuracy:.4%}")
print(f"Macro F1    : {best_f1:.4f}")


# Figure 3: Test Accuracy Comparison
plt.figure(figsize=(11, 6))

colors = [
    "#E45756" if model == best_model else "#4C78A8"
    for model in results_df["Model"]
]

bars = plt.bar(
    results_df["Model"],
    results_df["Test Accuracy"],
    color=colors
)

plt.ylim(0, 1)

plt.title(
    "Test Accuracy Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Model",
    fontsize=11
)

plt.ylabel(
    "Test Accuracy",
    fontsize=11
)

plt.xticks(
    rotation=25,
    ha="right"
)

# Add accuracy values
for bar, value in zip(
    bars,
    results_df["Test Accuracy"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2%}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "model_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Figure 4: Accuracy vs Macro F1
x = np.arange(len(results_df))
width = 0.36

plt.figure(figsize=(12, 6))

accuracy_bars = plt.bar(
    x - width / 2,
    results_df["Test Accuracy"],
    width,
    label="Test Accuracy",
    color="#4C78A8"
)

f1_bars = plt.bar(
    x + width / 2,
    results_df["Macro F1"],
    width,
    label="Macro F1",
    color="#F58518"
)

plt.ylim(0, 1)

plt.title(
    "Model Performance Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Model",
    fontsize=11
)

plt.ylabel(
    "Score",
    fontsize=11
)

plt.xticks(
    x,
    results_df["Model"],
    rotation=25,
    ha="right"
)

plt.legend()


# Add values above bars
for bars in [accuracy_bars, f1_bars]:

    for bar in bars:

        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Finished
print("\n" + "=" * 60)
print("FIGURES GENERATED")
print("=" * 60)

print(f"Results directory: {RESULTS_DIR}")
print(f"Figures directory: {FIGURES_DIR}")

print("\nGenerated figures:")

for figure in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  - {figure.name}")
