import os
import pandas as pd
import streamlit as st
from PIL import Image
import joblib


# IMAGE_FOLDER = "data/augmented"

# MODEL_PATH = "models/random_forest_model.pkl"

# RF_RESULTS_PATH = "results/reports/random_forest_results.csv"
# CNN_RESULTS_PATH = "results/reports/cnn_results.csv"

# MISCLASSIFIED_PATH = "results/reports/misclassified_images.csv"
# CNN_MISCLASSIFIED_PATH = "results/reports/cnn_misclassified_images.csv"

# CNN_PREDICTIONS_PATH = "results/reports/cnn_predictions.csv"


def render_comparison_tab(df, filtered_df):

    # ------------------------------------------------------------
    # Model Comparison
    # ------------------------------------------------------------

    st.header("Model Comparison [work in progress]")

    st.write(
        "This section compares the feature-based Random Forest approach "
        "with the image-based CNN approach."
    )

