# -------------------------------------------------------------------------
# Full project pipeline runner:
# -------------------------------------------------------------------------
#
# This script runs the complete Vision Data Quality Analyzer pipeline
# in the correct order.
#
# It starts from the raw demo images and generates all data, reports,
# models and evaluation files needed by the dashboard.
#
# Pipeline steps:
#
# - generate augmented image variants
# - extract OpenCV image quality features
# - create fixed train / validation / test split
# - train Random Forest classifier
# - run Random Forest error analysis
# - train CNN classifier
# - run CNN error analysis
#
# Optionally, the Streamlit dashboard can be started automatically
# after the pipeline has finished.
# -------------------------------------------------------------------------


import subprocess
import sys


# commands executed in the required pipeline order
COMMANDS = [
    [sys.executable, "src/generate_degraded_images.py"],
    [sys.executable, "src/extract_features.py"],
    [sys.executable, "src/create_dataset_split.py"],
    [sys.executable, "src/train_rf_classifier.py"],
    [sys.executable, "graphics_code/visualize_tree.py"],
    [sys.executable, "analysis/error_analysis_rf.py"],
    [sys.executable, "src/train_cnn_classifier.py"],
    [sys.executable, "analysis/error_analysis_cnn.py"],
    [sys.executable, "src/create_streamlit_demo_subset.py"],

]


# ------------------------------------------------------------
# Run complete pipeline
# ------------------------------------------------------------

for command in COMMANDS:

    print()
    print("Running:", " ".join(command))
    print("-" * 60)

    subprocess.run(
        command,
        check=True
    )

# ------------------------------------------------------------
# Pipeline summary
# -----------------------------------------------------

print()
print("Full pipeline completed successfully.")

print()
print("=" * 60)
print("Pipeline completed successfully.")
print("=" * 60)

print()
print("Generated outputs:")

print()
print("[Dataset]")
print("data/augmented/")
print("data/model_dataset/")
print("data/streamlit_demo/")

print()
print("[Models]")
print("models/random_forest_model.pkl")
print("models/cnn_classifier.keras")

print()
print("[Reports]")
print("results/reports/")
print("results/reports/random_forest/")
print("results/reports/cnn/")
print("results/reports/streamlit_demo/")

print()
print("[Plots]")
print("results/plots/")
print("results/plots/random_forest/")
print("results/plots/cnn/")

print()
print("[Visualizations]")
print("results/visualizations/random_forest_tree_67.svg")
print("results/visualizations/random_forest_concept.svg")
print("results/visualizations/cnn_plotneuralnet_portfolio.svg")

print()
print("=" * 60)

# ------------------------------------------------------------
# Optional demo dataset dashboard start
# ------------------------------------------------------------

start_dashboard = input(
    "\nStart Streamlit demo version dashboard now? (y/n): "
).strip().lower()


if start_dashboard == "y":

    print()
    print("Starting Streamlit demo version dashboard...")
    print("-" * 60)

    subprocess.run([
        sys.executable,
        "src/run_dashboard.py"
    ])