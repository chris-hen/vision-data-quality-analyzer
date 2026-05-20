# -------------------------------------------------------------------------
# This script starts the lightweight Streamlit demo dashboard for the
# Vision Data Quality Analyzer project.
#
# The demo dashboard uses a reduced image-level dataset for online deployment,
# while model metrics, plots and visualizations still come from the full
# training pipeline.
# -------------------------------------------------------------------------


import pandas as pd
import streamlit as st

import tabs.dataset_tab as dataset_tab
import tabs.random_forest_tab as random_forest_tab
import tabs.cnn_tab as cnn_tab

from tabs.about_tab import render_about_tab
from tabs.comparison_tab import render_comparison_tab


# lightweight demo reports
REPORT_PATH = "results/reports/streamlit_demo/image_quality_report.csv"
AUGMENTATION_METADATA_PATH = "results/reports/streamlit_demo/augmentation_metadata.csv"

# lightweight demo image folders
DEMO_RAW_IMAGE_FOLDER = "data/streamlit_demo/raw_images"
DEMO_IMAGE_FOLDER = "data/streamlit_demo/augmented"


# ------------------------------------------------------------
# Override image-level paths for demo dashboard
# ------------------------------------------------------------

dataset_tab.RAW_IMAGE_FOLDER = DEMO_RAW_IMAGE_FOLDER
dataset_tab.AUGMENTED_IMAGE_FOLDER = DEMO_IMAGE_FOLDER
dataset_tab.IMAGE_FOLDER = DEMO_IMAGE_FOLDER

random_forest_tab.IMAGE_FOLDER = DEMO_IMAGE_FOLDER
random_forest_tab.MISCLASSIFIED_PATH = (
    "results/reports/streamlit_demo/rf_misclassified_images.csv"
)

cnn_tab.IMAGE_FOLDER = DEMO_IMAGE_FOLDER
cnn_tab.CNN_MISCLASSIFIED_PATH = (
    "results/reports/streamlit_demo/cnn_misclassified_images.csv"
)
cnn_tab.CNN_PREDICTIONS_PATH = (
    "results/reports/streamlit_demo/cnn_predictions.csv"
)


# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Vision Data Quality Analyzer",
    layout="wide"
)

st.title("Vision Data Quality Analyzer")

st.write(
    "Interactive dashboard for image quality analysis, "
    "dataset augmentation, feature extraction and machine learning evaluation."
)

st.caption(
    """
    Demo version: This dashboard uses a reduced
    subset of the VisDrone dataset for GitHub
    deployment and interactive presentation.
    """
)

st.write("")


# ------------------------------------------------------------
# Load demo data
# ------------------------------------------------------------

try:
    df = pd.read_csv(REPORT_PATH)

except FileNotFoundError:
    st.error(
        f"""
        Demo report file not found:

        {REPORT_PATH}

        Run create_streamlit_demo_subset.py first.
        """
    )
    st.stop()

try:
    aug_df = pd.read_csv(AUGMENTATION_METADATA_PATH)

except FileNotFoundError:
    st.error(
        f"""
        Demo augmentation metadata file not found:

        {AUGMENTATION_METADATA_PATH}

        Run create_streamlit_demo_subset.py first.
        """
    )
    st.stop()


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------

st.sidebar.header("Filters")

quality_filter = st.sidebar.multiselect(
    "Quality status",
    options=sorted(df["quality_status"].unique()),
    default=sorted(df["quality_status"].unique())
)

degradation_filter = st.sidebar.multiselect(
    "Degradation type",
    options=sorted(df["degradation_type"].unique()),
    default=sorted(df["degradation_type"].unique())
)

filtered_df = df[
    (df["quality_status"].isin(quality_filter)) &
    (df["degradation_type"].isin(degradation_filter))
]


# ------------------------------------------------------------
# Dashboard tabs
# ------------------------------------------------------------

tab_about, tab_dataset, tab_random_forest, tab_cnn, tab_comparison = st.tabs(
    [
        "About",
        "Dataset & Image Quality",
        "Random Forest Model",
        "CNN Model",
        "Model Comparison"
    ]
)


with tab_about:
    render_about_tab()

with tab_dataset:
    dataset_tab.render_dataset_tab(df, filtered_df, aug_df)

with tab_random_forest:
    random_forest_tab.render_random_forest_tab(df, filtered_df)

with tab_cnn:
    cnn_tab.render_cnn_tab(df, filtered_df)

with tab_comparison:
    render_comparison_tab(df, filtered_df)