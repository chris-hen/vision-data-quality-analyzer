# --------------------------------------------------------------------
# This script generates 5 augmentation variants for every input image. 
# These variants are stored with the original image and the metadata
# of every image is saved.
# 
#
# The generated augmented dataset and the metadata report
# is later split for model training with src\create_dataset_split.py 
# --------------------------------------------------------------------

import os
import cv2
import numpy as np
import random
import pandas as pd

INPUT_FOLDER = "data/raw_images"
OUTPUT_FOLDER = "data/augmented"

# metadata used for dashboard overview of augmentation process
METADATA_PATH = "results/augmentation_metadata.csv"

# set seed for reproducibility
random.seed(42)
np.random.seed(42)


# clear output folder if raw image dataset was changed
#
# if os.path.exists(OUTPUT_FOLDER):
#    shutil.rmtree(OUTPUT_FOLDER)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("results", exist_ok=True)

augmentation_records = []

# count raw images in input folder
raw_image_count = len([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(
    f"Generating augmented image variants for "
    f"{raw_image_count} raw images..."
)


# ------------------------------------------------------------
# 5 functions for image augmentation using OpenCV:
# ------------------------------------------------------------

# add blur with random kernel size
def add_blur(image):
    kernel_size = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
    augmented_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return augmented_image, {"kernel_size": kernel_size}


# darken image with random intensity
def darken_image(image):
    #alpha = random.uniform(0.4, 0.8)
    #beta = random.randint(-70, -20)
    alpha = random.uniform(0.65, 0.9)
    beta = random.randint(-20, -5)
    augmented_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return augmented_image, {"alpha": round(alpha, 2), "beta": beta}


# brighten image with random intensity
def brighten_image(image):
    alpha = random.uniform(1.1, 1.5)
    beta = random.randint(20, 80)
    augmented_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return augmented_image, {"alpha": round(alpha, 2), "beta": beta}


# reduce image contrast with random strength
def reduce_contrast(image):
    alpha = random.uniform(0.3, 0.7)
    mean = np.mean(image)
    augmented_image = cv2.convertScaleAbs(image, alpha=alpha, beta=mean * (1 - alpha))
    return augmented_image, {"alpha": round(alpha, 2), "mean": round(mean, 2)}


# add gaussian noise with random strength
def add_noise(image):
    # Lower sigma to avoid strongly affecting the Laplacian-based blur score
    sigma = random.randint(5, 12)
    noise = np.random.normal(0, sigma, image.shape).astype(np.int16)

    noisy_image = image.astype(np.int16) + noise
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image.astype(np.uint8), {"sigma": sigma}


# ----------------------------------------------------------------
# generate augmented variations for every image in input folder
# and save individual metadata 
# ----------------------------------------------------------------

for filename in os.listdir(INPUT_FOLDER):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        path = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(path)

        if image is None:
            continue

        base_name = os.path.splitext(filename)[0]

        # save original image
        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, f"{base_name}_original.jpg"),
            image
        )
        # also save metadata of original images
        augmentation_records.append({
        "original_filename": filename,
        "augmented_filename": f"{base_name}_original.jpg",
        "augmentation_type": "original",
        "parameters": {}
         })

        # save degraded versions and their metadata for 5 degradation types: 
        # blur, dark, bright, lowcontrast, noise
        augmented_image, parameters = add_blur(image)
        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, f"{base_name}_blur.jpg"),
            augmented_image
        ) # blur
        augmentation_records.append({
            "original_filename": filename,
            "augmented_filename": f"{base_name}_blur.jpg",
            "augmentation_type": "blur",
            "parameters": parameters
        })

        augmented_image, parameters = darken_image(image)
        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, f"{base_name}_dark.jpg"),
            augmented_image
        ) # dark
        augmentation_records.append({
            "original_filename": filename,
            "augmented_filename": f"{base_name}_dark.jpg",
            "augmentation_type": "dark",
            "parameters": parameters
        })

        augmented_image, parameters = brighten_image(image)
        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, f"{base_name}_bright.jpg"),
            augmented_image
        ) # bright
        augmentation_records.append({
            "original_filename": filename,
            "augmented_filename": f"{base_name}_bright.jpg",
            "augmentation_type": "bright",
            "parameters": parameters
        })

        augmented_image, parameters = reduce_contrast(image)
        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, f"{base_name}_lowcontrast.jpg"),
            augmented_image
        ) # lowcontrast
        augmentation_records.append({
            "original_filename": filename,
            "augmented_filename": f"{base_name}_lowcontrast.jpg",
            "augmentation_type": "lowcontrast",
            "parameters": parameters
        })

        augmented_image, parameters = add_noise(image)
        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, f"{base_name}_noise.jpg"),
            augmented_image
        ) # noise
        augmentation_records.append({
            "original_filename": filename,
            "augmented_filename": f"{base_name}_noise.jpg",
            "augmentation_type": "noise",
            "parameters": parameters
        })

# save augmentation metadata for all variants
pd.DataFrame(augmentation_records).to_csv(
    METADATA_PATH,
    index=False
)

# printing results:
total_dataset_size = len(augmentation_records)
generated_variant_count = total_dataset_size - raw_image_count

print(
    f"Generated {generated_variant_count} "
    f"augmented image variants with randomized degradation parameters "
    f"from {raw_image_count} original images."
)

print(
    f"The final dataset contains "
    f"{total_dataset_size} images in total."
)
print(f"Augmentation metadata saved to {METADATA_PATH}")