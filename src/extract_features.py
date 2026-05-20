# -------------------------------------------------------------------------
# This script extracts image quality features
# from all generated augmented images using OpenCV.
#
# The extracted features and the assigned rule-based quality labels
# are saved as a structured CSV report for:
#
# - dataset analysis
# - dashboard visualization
# - Random Forest training
# - train/validation/test splitting with src/create_dataset_split.py
# -------------------------------------------------------------------------


import os
import cv2
import pandas as pd
import numpy as np

# for threshold based labeling
from quality_rules import classify_image_quality

# input folder containing only raw images
# INPUT_FOLDER = "data/raw_images"

# input folder containing augmented images
INPUT_FOLDER = "data/augmented"

# output file for extracted image features
OUTPUT_CSV = "results/reports/image_quality_report.csv"

# metadata created during image augmentation
# to assign degradation type and original filename
METADATA_PATH = "results/augmentation_metadata.csv"

# ------------------------------------------------------------
# Feature extraction functions
# for blur_score, brightness and contrast using OpenCV
# ------------------------------------------------------------

# estimate image sharpness using Laplacian variance
def calculate_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# calculate average brightness
def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


# calculate image contrast using standard deviation
def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)


# ------------------------------------------------------------
# Load augmentation metadata
# ------------------------------------------------------------

metadata_df = pd.read_csv(METADATA_PATH)

metadata_lookup = (
    metadata_df
    .set_index("augmented_filename")
    .to_dict("index")
)


# ------------------------------------------------------------
# Extract features from all generated images
# ------------------------------------------------------------
results = []

# iterate through all images in input folder
for filename in os.listdir(INPUT_FOLDER):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        path = os.path.join(INPUT_FOLDER, filename)

        image = cv2.imread(path)

        # skip invalid images
        if image is None:
            continue


        # read augmentation information from metadata
        metadata = metadata_lookup.get(filename, {})

        original_filename = metadata.get(
            "original_filename",
            "unknown"
        )

        degradation_type = metadata.get(
            "augmentation_type",
            "unknown"
        )

        # image resolution
        height, width = image.shape[:2]

        # extract quality values for image
        blur_score = calculate_blur_score(image)
        brightness = calculate_brightness(image)
        contrast = calculate_contrast(image)

        # use threshold-based rules to assign quality labels
        quality_issues, quality_status = classify_image_quality(
        blur_score,
        brightness,
        contrast
        )

        # store extracted features
        results.append({
            "filename": filename,

            # use original_filename for train/test splitting of dataset later
            "original_filename": original_filename,

            # degradation type now loaded from metadata instead
            "degradation_type": degradation_type,

            "width": width,
            "height": height,
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),

            "quality_issues": ", ".join(quality_issues) if quality_issues else "none",
            "quality_status": quality_status,
        })


# ------------------------------------------------------------
# Save feature report
# ------------------------------------------------------------

# convert results into dataframe
df = pd.DataFrame(results)

# create output directory if it does not exist
os.makedirs("results/reports", exist_ok=True)

# save extracted features as csv
df.to_csv(OUTPUT_CSV, index=False)

print("Feature extraction complete.")
print(df.head())
