# ---------------------------------------------------------------------------
# CNN classifier model:
# ---------------------------------------------------------------------------
#
# This script trains a convolutional neural network on the
# generated image dataset.
#
# The model predicts the degradation_type 
# [original, blur, dark, bright, lowcontrast, noise]
# of each generated image variant directly from image data.
#
#
# A first simple cnn version with fewer filters, flattening and no
# regularization was tested before, but did not outperform the rf baseline 
# (accuracy of 0.77 for old cnn vs rf baseline of 0.80).
#
# Therefore, this improved architecture was designed with:
#
# - more convolution filters for stronger image feature extraction
# - BatchNormalization for more stable training
# - GlobalAveragePooling instead of Flatten to reduce overfitting
# - Dropout before the output layer for additional regularization
#
#
# The fixed train / validation / test folder structure is loaded from:
#
# data/model_dataset/
#
# Train and validation data are used during model training, while the fixed
# test set is used only for final evaluation.
# ---------------------------------------------------------------------------


import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

# paths to fixed dataset folders
TRAIN_DATASET_PATH = "data/model_dataset/train"
VALIDATION_DATASET_PATH = "data/model_dataset/validation"
TEST_DATASET_PATH = "data/model_dataset/test"

# image size and training settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30

# paths for generated model, reports and plots
CNN_MODEL_OUTPUT = "models/cnn_classifier.keras"
CNN_HISTORY_CSV = "results/reports/cnn/cnn_training_history.csv"
CNN_HISTORY_PLOT = "results/plots/cnn/cnn_training_history.svg"
CNN_RESULTS_OUTPUT = "results/reports/cnn/cnn_results.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("results/reports/cnn", exist_ok=True)
os.makedirs("results/plots/cnn", exist_ok=True)


# ------------------------------------------------------------
# load image datasets from fixed folder structure
# ------------------------------------------------------------

# load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
# load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
# load fixed test dataset
# shuffle is disabled for reproducible evaluation
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# store degradation class names from dataset folders
class_names = train_dataset.class_names

print("Classes:")
print(class_names)


# ------------------------------------------------------------
# normalize image values
# ------------------------------------------------------------

# scale pixel values from [0, 255] to [0, 1]
normalization_layer = layers.Rescaling(1.0 / 255)

# apply normalization to training dataset
train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)
# apply normalization to validation dataset
validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)
# apply normalization to test dataset
test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)


# ------------------------------------------------------------
# create improved cnn model
# ------------------------------------------------------------
#
# cnn architecture summary:
#
# input (128x128x3)
# -> cnn 32
# -> max pooling
# -> cnn 64
# -> max pooling
# -> cnn 128
# -> max pooling
# -> cnn 128
# -> global average pooling
# -> fc 128
# -> dropout 0.4
# -> softmax output (6 classes)
#
# cnn architecture:
# the model uses 4 convolution layers with progressing filter sizes of
# 32 -> 64 -> 128 -> 128 filters
# and 3 max pooling operations to progressively extract
# more detailed image degradation features.
#
# BatchNormalization layers are used after every convolution
# to stabilize training and improve convergence.
#
# GlobalAveragePooling2D replaces a traditional flatten layer
# to reduce the number of trainable parameters and lower overfitting.
#
# A fully connected dense layer with 128 neurons performs the final
# feature learning before classification.
#
# Dropout with a rate of 0.4 is applied before the output layer
# to improve generalization performance on unseen test data.
#
# The final softmax output layer predicts one of the 6
# degradation classes:
# [original, blur, dark, bright, lowcontrast, noise]


# sequential cnn architecture for degradation classification
model = models.Sequential([

    # input image shape:
    #
    # images are resized to 128x128 pixels
    # to reduce training time, as the model focuses on global degradation patterns
    layers.Input(shape=(128, 128, 3)),

    # first convolution block:
    #
    # detect simple image features such as
    # edges, blur patterns and brightness transitions
    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),

    # second convolution block:
    #
    # learn more detailed image structures
    # and degradation characteristics
    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),

    # third convolution block:
    #
    # learn more abstract degradation patterns
    # such as strong blur or image noise
    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False
    ),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2, 2)),

    # fourth convolution layer for additional feature extraction
    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False
    ),
    layers.BatchNormalization(),
    layers.ReLU(),

    # reduce feature maps into compact global feature representation
    layers.GlobalAveragePooling2D(),

    # fully connected classification layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # dropout for reduced overfitting
    layers.Dropout(0.4),

    # final output layer for degradation class prediction
    layers.Dense(
        len(class_names),
        activation="softmax"
    ) # softmax activation for multiclass classification task
])


# ------------------------------------------------------------
# compile cnn
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# early stopping to end training if validation accuracy does not improve
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True
)

# show model structure
model.summary()

# ------------------------------------------------------------
# train cnn
# ------------------------------------------------------------

# store start time to measure total training duration
start_time = time.time()
print("Starting CNN training...")


# train cnn model on training dataset
# and monitor validation accuracy during training
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# calculate total training time
end_time = time.time()
training_time = end_time - start_time


# ------------------------------------------------------------
# evaluate cnn
# ------------------------------------------------------------

# evaluate model on fixed test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)


# get final and best training values
final_training_accuracy = history.history["accuracy"][-1]
final_validation_accuracy = history.history["val_accuracy"][-1]
best_validation_accuracy = max(history.history["val_accuracy"])

# print final cnn training and evaluation results
print()
print(f"Training time: {training_time:.2f} seconds")
print(f"Final training accuracy: {final_training_accuracy:.3f}")
print(f"Final validation accuracy: {final_validation_accuracy:.3f}")
print(f"Best validation accuracy: {best_validation_accuracy:.3f}")
print(f"Final test accuracy: {test_accuracy:.3f}")


# ------------------------------------------------------------
# save cnn evaluation results and training history
# ------------------------------------------------------------

# save final cnn results
cnn_results_df = pd.DataFrame({
    "metric": [
        "final_training_accuracy",
        "final_validation_accuracy",
        "best_validation_accuracy",
        "test_accuracy",
        "training_time_seconds"
    ],
    "value": [
        final_training_accuracy,
        final_validation_accuracy,
        best_validation_accuracy,
        test_accuracy,
        training_time
    ]
})
# save cnn evaluation results as csv
cnn_results_df.to_csv(CNN_RESULTS_OUTPUT, index=False)


# convert keras training history into df
history_df = pd.DataFrame(history.history)

# add epoch numbers for plotting and analysis
history_df["epoch"] = range(1, len(history_df) + 1)

# save full training history as csv
history_df.to_csv(CNN_HISTORY_CSV, index=False)

# ------------------------------------------------------------
# create and save training plots and save model
# ------------------------------------------------------------

# save training history plot:
plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["accuracy"], label="Training Accuracy")
plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Training History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(CNN_HISTORY_PLOT)
plt.close()

print("CNN training history saved.")


# save trained model with restored best validation weights
model.save(CNN_MODEL_OUTPUT)

print("CNN training complete.")
print(f"Model saved to {CNN_MODEL_OUTPUT}")