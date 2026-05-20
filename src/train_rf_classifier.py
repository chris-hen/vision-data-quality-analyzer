# ---------------------------------------------------------------------------
# Random Forest classifier model:
# ---------------------------------------------------------------------------
#
# This script trains a Random Forest classifier on the
# extracted image quality values.
#
# The model predicts the degradation_type of each generated image variant
# using only the extracted
#
# - blur_score
# - brightness
# - contrast
#
# as input features.
#
# The dataset split report is loaded from results/reports/dataset_split.csv.
# Train and validation data are used for model training, while the fixed
# test set is used for final evaluation.
# ---------------------------------------------------------------------------

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# paths for both reports needed by the model:
# extracted features report
INPUT_CSV = "results/reports/image_quality_report.csv"
# split report for split label (train/validation/test)
SPLIT_CSV = "results/reports/dataset_split.csv"

# paths for generated model, reports and plots
MODEL_OUTPUT = "models/random_forest_model.pkl"
FEATURE_IMPORTANCE_OUTPUT = "results/reports/random_forest/feature_importance.csv"
CONFUSION_MATRIX_PLOT = "results/plots/random_forest/confusion_matrix.svg"
FEATURE_IMPORTANCE_PLOT = "results/plots/random_forest/feature_importance.svg"
RF_RESULTS_OUTPUT = "results/reports/random_forest/random_forest_results.csv"

# create rf output folder structure for model and plots
os.makedirs("models", exist_ok=True)
os.makedirs("results/reports/random_forest", exist_ok=True)
os.makedirs("results/plots/random_forest", exist_ok=True)


# ------------------------------------------------------------
# Load feature report and fixed dataset split
# ------------------------------------------------------------

# load extracted image features
df = pd.read_csv(INPUT_CSV)

# load fixed train / validation / test split df
split_df = pd.read_csv(SPLIT_CSV)

# features are extracted before the dataset split,
# the model needs both the extracted feature values and the split labels,
# so these two dataframes are merged in the next step.


# merge split information into the feature dataframe
# this adds the corresponding split label (train / validation / test)
# to every image entry

# add split information to feature report
df = df.merge(
    split_df[["augmented_filename", "split"]],
    left_on="filename",
    right_on="augmented_filename",
    how="left"
)

# remove duplicate filename column created during this merge
df = df.drop(columns=["augmented_filename"])

# use only known degradation classes; legacy safety check
df = df[df["degradation_type"] != "unknown"]

# input features used for rf training
features = ["blur_score", "brightness", "contrast"]

# use train + validation split for rf training
# as rf does not require a dedicated validation
# set during training like the cnn model
train_df = df[df["split"].isin(["train", "validation"])]

# use fixed test split for final rf evaluation
test_df = df[df["split"] == "test"]

print("Random Forest dataset split:")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print()

# create feature and target datasets
X_train = train_df[features]
y_train = train_df["degradation_type"]

X_test = test_df[features]
y_test = test_df["degradation_type"]


# ------------------------------------------------------------
# train rf classifier
# ------------------------------------------------------------

# rf consisting of 100 decision trees
# for stable classification importance + low training time

# set random state for reproducible results
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# train model on extracted features
model.fit(X_train, y_train)

# save trained model
joblib.dump(model, MODEL_OUTPUT)


# ------------------------------------------------------------
# evaluate rf classifier model:
# ------------------------------------------------------------

# predict degradation classes for the fixed test dataset
y_pred = model.predict(X_test)

# calculate overall classification accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model training complete.")
print(f"Accuracy: {accuracy:.3f}")

# store final rf accuracy
results_df = pd.DataFrame({
    "metric": ["accuracy"],
    "value": [accuracy]
})
results_df.to_csv(RF_RESULTS_OUTPUT, index=False)

# print classification metrics for quick safety check
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# analyze feature importance:
print()
print("Feature Importance:")

# higher importance means the feature has a stronger
# influence on the final rf predictions
for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance:.3f}")

# save feature importance as csv
importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

# sort features by importance for visualization
importance_df = importance_df.sort_values(
    by="importance",
    ascending=True
)
# save feature importance
importance_df.to_csv(FEATURE_IMPORTANCE_OUTPUT, index=False)

# create and save bar plot for feature performance
plt.figure(figsize=(7, 4))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(FEATURE_IMPORTANCE_PLOT)
plt.close()

# create and save confusion matrix plot from test prediction
cm = confusion_matrix(y_test, y_pred)
# cm plot
plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.colorbar()

# get sorted degradation class labels
classes = sorted(y_test.unique())

# label axis with classes
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
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PLOT)
plt.close()

# final output message
print()
print(f"Model saved to: {MODEL_OUTPUT}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print("Feature importance and Confusion matrix plot saved.")
print()