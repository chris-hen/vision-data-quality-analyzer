# -------------------------------------------------------------------------
# Create Streamlit demo subset:
# -------------------------------------------------------------------------
#
# This script creates a lightweight demo subset for Streamlit deployment.
#
# It keeps the full training and evaluation results unchanged, but reduces
# large image-level reports and image folders to a smaller subset.
#
# This makes the dashboard suitable for online deployment while preserving
# the model metrics from the complete training pipeline.
# -------------------------------------------------------------------------


import os
import shutil
import pandas as pd


# number of original images used for dashboard demo subset
N_ORIGINAL_IMAGES = 30

# full pipeline outputs
IMAGE_REPORT_PATH = "results/reports/image_quality_report.csv"
AUGMENTATION_METADATA_PATH = "results/augmentation_metadata.csv"
RF_MISCLASSIFIED_PATH = "results/reports/random_forest/rf_misclassified_images.csv"
CNN_MISCLASSIFIED_PATH = "results/reports/cnn/cnn_misclassified_images.csv"
CNN_PREDICTIONS_PATH = "results/reports/cnn/cnn_predictions.csv"

# source image folders
RAW_IMAGE_FOLDER = "data/raw_images"
AUGMENTED_IMAGE_FOLDER = "data/augmented"

# lightweight demo output folders
DEMO_DATA_FOLDER = "data/streamlit_demo"
DEMO_RAW_IMAGE_FOLDER = "data/streamlit_demo/raw_images"
DEMO_IMAGE_FOLDER = "data/streamlit_demo/augmented"
DEMO_REPORT_FOLDER = "results/reports/streamlit_demo"


# ------------------------------------------------------------
# Prepare demo output folders
# ------------------------------------------------------------

# remove old demo subset if it already exists
if os.path.exists(DEMO_DATA_FOLDER):
    shutil.rmtree(DEMO_DATA_FOLDER)

if os.path.exists(DEMO_REPORT_FOLDER):
    shutil.rmtree(DEMO_REPORT_FOLDER)

# create fresh demo folders
os.makedirs(DEMO_RAW_IMAGE_FOLDER, exist_ok=True)
os.makedirs(DEMO_IMAGE_FOLDER, exist_ok=True)
os.makedirs(DEMO_REPORT_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# Load full image report and select demo originals
# ------------------------------------------------------------

# load full image quality report
df = pd.read_csv(IMAGE_REPORT_PATH)

# select first n original filenames for demo subset
demo_originals = (
    df["original_filename"]
    .drop_duplicates()
    .head(N_ORIGINAL_IMAGES)
    .tolist()
)


# alternatively for a more representative overview of images,
#but first 30 seems to work out for the demonstration right now

# select random original filenames for demo subset
# demo_originals = (
#     df["original_filename"]
#     .drop_duplicates()
#     .sample(
#         n=N_ORIGINAL_IMAGES,
#         random_state=42
#     )
#     .tolist()
# )

# keep all generated variants belonging to selected originals
demo_df = df[df["original_filename"].isin(demo_originals)].copy()


# ------------------------------------------------------------
# Save filtered image quality report
# ------------------------------------------------------------

# save lightweight image quality report for dashboard demo
demo_df.to_csv(
    f"{DEMO_REPORT_FOLDER}/image_quality_report.csv",
    index=False
)


# ------------------------------------------------------------
# Filter and save augmentation metadata
# ------------------------------------------------------------

# load full augmentation metadata
aug_df = pd.read_csv(AUGMENTATION_METADATA_PATH)

# keep metadata for selected demo originals
demo_aug_df = aug_df[
    aug_df["original_filename"].isin(demo_originals)
].copy()

# save lightweight augmentation metadata
demo_aug_df.to_csv(
    f"{DEMO_REPORT_FOLDER}/augmentation_metadata.csv",
    index=False
)


# ------------------------------------------------------------
# Copy required raw demo images
# ------------------------------------------------------------

# copy only original raw images used by the demo subset
for filename in demo_originals:

    source_path = os.path.join(
        RAW_IMAGE_FOLDER,
        filename
    )

    target_path = os.path.join(
        DEMO_RAW_IMAGE_FOLDER,
        filename
    )

    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)


# ------------------------------------------------------------
# Copy required augmented demo images
# ------------------------------------------------------------

# copy only augmented images used by the demo report
for filename in demo_df["filename"]:

    source_path = os.path.join(
        AUGMENTED_IMAGE_FOLDER,
        filename
    )

    target_path = os.path.join(
        DEMO_IMAGE_FOLDER,
        filename
    )

    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)


# ------------------------------------------------------------
# Filter image-level error reports
# ------------------------------------------------------------

# filter RF misclassified image report if available
if os.path.exists(RF_MISCLASSIFIED_PATH):

    rf_error_df = pd.read_csv(RF_MISCLASSIFIED_PATH)

    rf_demo_error_df = rf_error_df[
        rf_error_df["original_filename"].isin(demo_originals)
    ].copy()

    rf_demo_error_df.to_csv(
        f"{DEMO_REPORT_FOLDER}/rf_misclassified_images.csv",
        index=False
    )


# filter CNN misclassified image report if available
if os.path.exists(CNN_MISCLASSIFIED_PATH):

    cnn_error_df = pd.read_csv(CNN_MISCLASSIFIED_PATH)

    cnn_demo_error_df = cnn_error_df[
        cnn_error_df["original_filename"].isin(demo_originals)
    ].copy()

    cnn_demo_error_df.to_csv(
        f"{DEMO_REPORT_FOLDER}/cnn_misclassified_images.csv",
        index=False
    )


# filter CNN prediction report if available
if os.path.exists(CNN_PREDICTIONS_PATH):

    cnn_predictions_df = pd.read_csv(CNN_PREDICTIONS_PATH)

    cnn_demo_predictions_df = cnn_predictions_df[
        cnn_predictions_df["original_filename"].isin(demo_originals)
    ].copy()

    cnn_demo_predictions_df.to_csv(
        f"{DEMO_REPORT_FOLDER}/cnn_predictions.csv",
        index=False
    )


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print("Streamlit demo subset created.")
print(f"Original images selected: {len(demo_originals)}")
print(f"Raw demo images copied: {len(demo_originals)}")
print(f"Augmented demo images copied: {len(demo_df)}")
print(f"Demo raw image folder: {DEMO_RAW_IMAGE_FOLDER}")
print(f"Demo augmented image folder: {DEMO_IMAGE_FOLDER}")
print(f"Demo reports saved to: {DEMO_REPORT_FOLDER}")