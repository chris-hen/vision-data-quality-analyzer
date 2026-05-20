# -------------------------------------------------------------------------
# This script starts the Streamlit dashboard for the
# Vision Data Quality Analyzer project.
#
# The dashboard loads the extracted image quality report and augmentation
# metadata and provides interactive tabs for:
#
# - project overview
# - dataset and image quality analysis
# - Random Forest model evaluation
# - CNN model evaluation
# - model comparison
#
# The sidebar filters are applied globally and passed to the dashboard tabs.
# -------------------------------------------------------------------------


import pandas as pd
import streamlit as st

from tabs.about_tab import render_about_tab
from tabs.dataset_tab import render_dataset_tab
from tabs.random_forest_tab import render_random_forest_tab
from tabs.cnn_tab import render_cnn_tab
from tabs.comparison_tab import render_comparison_tab


# path to extracted image quality feature report
REPORT_PATH = "results/reports/image_quality_report.csv"

# path to augmentation metadata created during image augmentation
AUGMENTATION_METADATA_PATH = "results/augmentation_metadata.csv"


# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------

# configure dashboard page title and layout
st.set_page_config(
    page_title="Vision Data Quality Analyzer",
    layout="wide"
)

# main dashboard title
st.title("Vision Data Quality Analyzer")

# short dashboard description
st.write(
    "Interactive dashboard for image quality analysis, "
    "dataset augmentation, feature extraction and machine learning evaluation."
)

st.write("")


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

# load extracted image quality report
try:
    df = pd.read_csv(REPORT_PATH)

# stop dashboard if feature report is missing
except FileNotFoundError:
    st.error(
        f"""
        Report file not found:

        {REPORT_PATH}

        Run the image quality pipeline first.
        """
    )
    st.stop()

# load augmentation metadata report
try:
    aug_df = pd.read_csv(AUGMENTATION_METADATA_PATH)

# stop dashboard if augmentation metadata is missing
except FileNotFoundError:
    st.error(
        f"""
        Augmentation metadata file not found:

        {AUGMENTATION_METADATA_PATH}

        Run the augmentation pipeline first.
        """
    )
    st.stop()


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------

# sidebar title for global dashboard filters
st.sidebar.header("Filters")

# filter images by rule-based quality status
quality_filter = st.sidebar.multiselect(
    "Quality status",
    options=sorted(df["quality_status"].unique()),
    default=sorted(df["quality_status"].unique())
)

# filter images by known degradation class
degradation_filter = st.sidebar.multiselect(
    "Degradation type",
    options=sorted(df["degradation_type"].unique()),
    default=sorted(df["degradation_type"].unique())
)

# apply selected sidebar filters to the feature dataframe
filtered_df = df[
    (df["quality_status"].isin(quality_filter)) &
    (df["degradation_type"].isin(degradation_filter))
]


# ------------------------------------------------------------
# Dashboard tabs
# ------------------------------------------------------------

# create dashboard tabs for project explanation,
# dataset analysis and model evaluation
tab_about, tab_dataset, tab_random_forest, tab_cnn, tab_comparison = st.tabs(
    [
        "About",
        "Dataset & Image Quality",
        "Random Forest Model",
        "CNN Model",
        "Model Comparison"
    ]
)


# render project overview tab
with tab_about:
    render_about_tab()

# render dataset and image quality analysis tab
with tab_dataset:
    render_dataset_tab(df, filtered_df, aug_df)

# render Random Forest model evaluation tab
with tab_random_forest:
    render_random_forest_tab(df, filtered_df)

# render CNN model evaluation tab
with tab_cnn:
    render_cnn_tab(df, filtered_df)

# render model comparison tab
with tab_comparison:
    render_comparison_tab(df, filtered_df)