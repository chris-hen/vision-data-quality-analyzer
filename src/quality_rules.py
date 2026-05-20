# --------------------------------------------------------------------------------------------
# Rule-based image quality classification:
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# This script only contains the classify_image_quality function
# used in src\extract_features.py.
#
# The separation is done to make it easier to adjust or extent
# quality thresholds later without changing the main 
# feature extraction process.
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# The extracted image values:
#
#  - blur_score
#  - brightness
#  - contrast
#
# are evaluated using simple threshold-based rules to assign:
#
#     ==> 1. one or more quality_issues labels
#               [blurry, underexposed, overexposed, low_contrast]
#
#     ==> 2. a quality_status
#               [usable, not_usable]
#
# depending on whether any quality issues were found.
# --------------------------------------------------------------------------------------------



# assign quality labels for image:
def classify_image_quality(blur_score, brightness, contrast):

    labels = []

    # low blur score usually means the image is not sharp
    if blur_score < 100:
        labels.append("blurry")

    # very dark images are marked as underexposed
    if brightness < 60:
        labels.append("underexposed")

    # very bright images are marked as overexposed
    if brightness > 200:
        labels.append("overexposed")

    # low standard deviation means low contrast
    if contrast < 35:
        labels.append("low_contrast")

    # images with detected quality issues are marked as not usable
    if labels:
        quality_status = "not_usable"

    # images without detected issues are marked as usable
    else:
        quality_status = "usable"

    return labels, quality_status