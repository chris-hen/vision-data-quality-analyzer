# ---------------------------------------------------------------------------
# This script is used to create a train/val/test split of the augmented data.
#
# The goal directory structure is something like this:
#
# data/model_dataset/
# ├── train/
# │   ├── original/
# │   ├── blur/
# │   ├── dark/
# │   ├── bright/
# │   ├── lowcontrast/
# │   └── noise/
# ├── validation/
# │   └── ...
# └── test/
#     └── ...

# as the cnn model/tensorflow needs a directory for every single class
# ---------------------------------------------------------------------------

import os
import shutil
import random
import pandas as pd

METADATA_PATH = "results/augmentation_metadata.csv"

INPUT_FOLDER = "data/augmented"
OUTPUT_FOLDER = "data/model_dataset"

SPLIT_REPORT_PATH = "results/reports/dataset_split.csv"

# split the augmented data 3 ways so there is a fixed validation set for the cnn
# the rf model later gets trained on the train + validation set (80% of total data)
# and both models are tested only on the remaining test set (20% of total data) for a fair comparison.
# This also prevents any data leakage between train and test data. 

# split ratios
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.2

# set seed for splitting
random.seed(42)

# load metadata report of augmented images
df = pd.read_csv(METADATA_PATH)


# ------------------------------------------------------------
# split original images
# ------------------------------------------------------------

# check df for original_filenames and discard duplicates
# => leaves only original image filenames 
#    to split the dataset
original_images = df["original_filename"].unique().tolist()

# produce random order of originals before splitting
random.shuffle(original_images)

# manual splitting of original image data in 3 fixed sets:
train_end = int(len(original_images) * TRAIN_RATIO)
validation_end = train_end + int(len(original_images) * VALIDATION_RATIO)

# original images now split into 3 datasets
train_originals = original_images[:train_end]
validation_originals = original_images[train_end:validation_end]
test_originals = original_images[validation_end:]


# ------------------------------------------------------------
# assign split labels for augmented images:
# ------------------------------------------------------------

# checks to which split an original image belongs
def get_split(original_filename):

    if original_filename in train_originals:
        return "train"

    if original_filename in validation_originals:
        return "validation"

    return "test"

# every augmented image variant receives the same
# split label as its corresponding original image.
#
# create a new "split" column inside the augmentation metadata df
df["split"] = df["original_filename"].apply(get_split)


# ----------------------------------------------------------------
# create final dataset folder structure as described in header:
# ----------------------------------------------------------------

# delete old dataset split folder if it already exists
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)

for split in ["train", "validation", "test"]:

    # create one folder for every degradation class in model_data
    # found in the augmentation metadata
    for degradation_type in df["augmentation_type"].unique():

        os.makedirs(
            os.path.join(
                OUTPUT_FOLDER,
                split,
                degradation_type
            ),
            exist_ok=True
        )


# ----------------------------------------------------------------
# now copy all augmented images into the final
# train / validation / test folder structure:
# ----------------------------------------------------------------

# iterate through every image entry stored in the metadata dataframe
for _, row in df.iterrows():

    # build the current source path of the image
    # inside the augmented dataset folder
    source_path = os.path.join(
        INPUT_FOLDER,
        row["augmented_filename"]
    )

    # build the final target path depending on:
    #
    # 1. assigned dataset split (train, validation, test)
    #
    # 2. degradation class (blur , dark , bright , lowcontrast , noise , original)
    #
    # i.e. data/model_dataset/train/blur/example_blur.jpg
    target_path = os.path.join(
        OUTPUT_FOLDER,
        row["split"],
        row["augmentation_type"],
        row["augmented_filename"]
    )

    # copy the image into the corresponding split/class folder
    shutil.copy(source_path, target_path)

# create output directory if it does not exist
os.makedirs("results/reports", exist_ok=True)

# save final dataset split information now including train / validation / test labels
df.to_csv(SPLIT_REPORT_PATH, index=False)


# print summary
print()
print(f"Model dataset created successfully at {OUTPUT_FOLDER}")
print()

print(df["split"].value_counts())
print()
print(f"Split report saved to {SPLIT_REPORT_PATH}")
print()