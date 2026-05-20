# -------------------------------------------------------------------------
# Dataset analysis plots:
# -------------------------------------------------------------------------
#
# This script creates simple analysis plots from the
# extracted image quality report.
#
# The generated plots provide a quick overview of:
#
# - blur score distribution
# - brightness distribution
# - rule-based quality classification
#
# These plots are mainly used for:
#
# - dataset inspection
# - sanity checks
# - dashboard visualization
# -------------------------------------------------------------------------


import os
import pandas as pd
import matplotlib.pyplot as plt


# path to extracted image quality report
INPUT_CSV = "results/reports/image_quality_report.csv"


# ------------------------------------------------------------
# Load extracted image quality report
# ------------------------------------------------------------

# load extracted image feature data
df = pd.read_csv(INPUT_CSV)


# ------------------------------------------------------------
# Create output folder
# ------------------------------------------------------------

# create output directory for generated plots
os.makedirs("results/plots", exist_ok=True)


# ------------------------------------------------------------
# Blur score distribution
# ------------------------------------------------------------

# create histogram of extracted blur scores
plt.figure(figsize=(8, 5))

plt.hist(
    df["blur_score"],
    bins=30
)

plt.title("Blur Score Distribution")
plt.xlabel("Blur Score")
plt.ylabel("Number of Images")

# save blur score distribution plot
plt.savefig("results/plots/blur_distribution.png")
plt.close()


# ------------------------------------------------------------
# Brightness distribution
# ------------------------------------------------------------

# create histogram of extracted brightness values
plt.figure(figsize=(8, 5))

plt.hist(
    df["brightness"],
    bins=30
)

plt.title("Brightness Distribution")
plt.xlabel("Brightness")
plt.ylabel("Number of Images")

# save brightness distribution plot
plt.savefig("results/plots/brightness_distribution.png")
plt.close()


# ------------------------------------------------------------
# Rule-based quality classification distribution
# ------------------------------------------------------------

# count usable vs not usable images
quality_counts = df["quality_status"].value_counts()

# create bar plot for quality status distribution
plt.figure(figsize=(6, 5))

quality_counts.plot(kind="bar")

plt.title("Image Quality Classification")
plt.xlabel("Quality Status")
plt.ylabel("Count")

# save quality classification plot
plt.savefig("results/plots/quality_status_distribution.png")
plt.close()


# print summary
print("Analysis plots created.")