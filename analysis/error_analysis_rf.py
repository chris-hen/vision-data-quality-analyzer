# -------------------------------------------------------------------------
# Random Forest error analysis:
# -------------------------------------------------------------------------
#
# This script identifies all incorrectly classified images
# of the trained Random Forest model.
#
# The trained model predicts the degradation_type for every image
# stored in the extracted feature report.
#
# All prediction errors are stored separately for:
#
# - dashboard visualization
# - confusion analysis
# - inspection of difficult degradation classes
# - misclassified image inspection
# -------------------------------------------------------------------------


import joblib
import pandas as pd


# paths for feature report, trained model and output report
REPORT_PATH = "results/reports/image_quality_report.csv"
MODEL_PATH = "models/random_forest_model.pkl"
OUTPUT_PATH = "results/reports/random_forest/rf_misclassified_images.csv"

# extracted OpenCV features used for RF prediction
FEATURES = ["blur_score", "brightness", "contrast"]


# ------------------------------------------------------------
# Load feature report and trained RF model
# ------------------------------------------------------------

# load extracted image quality report
df = pd.read_csv(REPORT_PATH)

# load trained Random Forest classifier
model = joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# Predict degradation classes
# ------------------------------------------------------------

# use extracted feature values as RF input
X = df[FEATURES]

# predict degradation class for every image
df["predicted_degradation_type"] = model.predict(X)


# ------------------------------------------------------------
# Detect prediction errors
# ------------------------------------------------------------

# compare predicted degradation type with known target label
df["prediction_correct"] = (
    df["predicted_degradation_type"] == df["degradation_type"]
)

# keep only incorrect RF predictions
misclassified_df = df[df["prediction_correct"] == False]


# ------------------------------------------------------------
# Save RF prediction errors
# ------------------------------------------------------------

# save all misclassified images as csv report
misclassified_df.to_csv(OUTPUT_PATH, index=False)


# print summary
print("Error analysis complete.")
print(f"Misclassified images: {len(misclassified_df)}")
print(f"Saved to: {OUTPUT_PATH}")