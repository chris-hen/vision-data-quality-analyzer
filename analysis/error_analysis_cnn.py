# -------------------------------------------------------------------------
# CNN error analysis:
# -------------------------------------------------------------------------
#
# This script identifies all incorrectly classified images
# of the trained CNN model.
#
# The trained CNN predicts the degradation_type for every image
# directly from the generated image data.
#
# The script stores:
#
# - all CNN predictions
# - prediction confidence values
# - class-wise accuracy values
# - misclassified CNN predictions
# - classification report
# - CNN confusion matrix plot
#
# These outputs are later used for dashboard visualization,
# error analysis and model comparison.
# -------------------------------------------------------------------------


import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# paths for feature report, image folder, trained CNN model and output report
REPORT_PATH = "results/reports/image_quality_report.csv"
IMAGE_FOLDER = "data/augmented"
CNN_MODEL_PATH = "models/cnn_classifier.keras"
OUTPUT_PATH = "results/reports/cnn/cnn_misclassified_images.csv"

# fixed image size used during CNN training
IMAGE_SIZE = (128, 128)

# class order used for converting prediction indices back to labels
CLASS_NAMES = ["blur", "bright", "dark", "lowcontrast", "noise", "original"]

# paths for CNN evaluation outputs
CNN_CONFUSION_MATRIX_PLOT = "results/plots/cnn/cnn_confusion_matrix.svg"
CNN_CLASSIFICATION_REPORT = "results/reports/cnn/cnn_classification_report.txt"
CNN_PREDICTIONS_OUTPUT = "results/reports/cnn/cnn_predictions.csv"


# ------------------------------------------------------------
# Load feature report and trained CNN model
# ------------------------------------------------------------

# load extracted image quality report
df = pd.read_csv(REPORT_PATH)

# load trained CNN classifier
model = tf.keras.models.load_model(CNN_MODEL_PATH)


# ------------------------------------------------------------
# Load all images into memory
# ------------------------------------------------------------

# list for storing preprocessed image arrays
image_arrays = []

# iterate through every generated image from feature report
for filename in df["filename"]:

    # build full image path
    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    # load image and resize to cnn input size
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    # convert image to numerical array
    image_array = tf.keras.utils.img_to_array(image)

    # normalize pixel values to [0, 1]
    image_array = image_array / 255.0

    # store preprocessed image
    image_arrays.append(image_array)

# convert image list into numpy tensor batch
X = np.array(image_arrays)


# ------------------------------------------------------------
# Batch prediction
# ------------------------------------------------------------

# predict all images at once with trained cnn model
predictions = model.predict(
    X,
    batch_size=128,
    verbose=1
)

# get predicted class index with highest softmax probability
predicted_indices = np.argmax(
    predictions,
    axis=1
)

# get prediction confidence value for every image
prediction_confidences = np.max(
    predictions,
    axis=1
)

# convert predicted indices back to degradation class names
predicted_classes = [
    CLASS_NAMES[index]
    for index in predicted_indices
]

# store predicted cnn labels in dataframe
df["cnn_predicted_degradation_type"] = predicted_classes

# store prediction confidence values in dataframe
df["cnn_prediction_confidence"] = prediction_confidences

# compare predicted cnn label with known degradation type
df["cnn_prediction_correct"] = (
    df["cnn_predicted_degradation_type"] == df["degradation_type"]
)

# save complete cnn prediction report
df.to_csv(CNN_PREDICTIONS_OUTPUT, index=False)


# ------------------------------------------------------------
# Calculate class-wise CNN accuracy
# ------------------------------------------------------------

# output path for class accuracy report
CNN_CLASS_ACCURACY_OUTPUT = "results/reports/cnn/cnn_class_accuracy.csv"

# calculate mean prediction correctness for every degradation type
class_accuracy_df = (
    df.groupby("degradation_type")["cnn_prediction_correct"]
    .mean()
    .reset_index()
)

# rename columns for cleaner report
class_accuracy_df.columns = ["degradation_type", "accuracy"]

# save class-wise cnn accuracy report
class_accuracy_df.to_csv(CNN_CLASS_ACCURACY_OUTPUT, index=False)


# ------------------------------------------------------------
# Detect CNN prediction errors
# ------------------------------------------------------------

# keep only incorrect cnn predictions
misclassified_df = df[df["cnn_prediction_correct"] == False]

# save all misclassified images as csv report
misclassified_df.to_csv(OUTPUT_PATH, index=False)


# ------------------------------------------------------------
# Save CNN classification report
# ------------------------------------------------------------

# create text classification report
report = classification_report(
    df["degradation_type"],
    df["cnn_predicted_degradation_type"]
)

# save classification report as txt file
with open(CNN_CLASSIFICATION_REPORT, "w") as file:
    file.write(report)


# ------------------------------------------------------------
# Create CNN confusion matrix plot
# ------------------------------------------------------------

# fixed class label order for confusion matrix
classes = CLASS_NAMES

# create confusion matrix from actual and predicted labels
cm = confusion_matrix(
    df["degradation_type"],
    df["cnn_predicted_degradation_type"],
    labels=classes
)

# create confusion matrix figure
plt.figure(figsize=(8, 6))
plt.imshow(cm)

plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")

plt.colorbar()

# label confusion matrix axes with class names
plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
plt.yticks(range(len(classes)), classes)

# add prediction counts into matrix cells
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

# save confusion matrix plot
plt.tight_layout()
plt.savefig(CNN_CONFUSION_MATRIX_PLOT)
plt.close()


# print summary
print("CNN error analysis complete.")
print(f"Misclassified images: {len(misclassified_df)}")
print(f"Saved to: {OUTPUT_PATH}")