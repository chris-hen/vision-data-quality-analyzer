# -------------------------------------------------------------------------
# Streamlit dashboard runner:
# -------------------------------------------------------------------------
#
# This script starts the Streamlit dashboard of the
# Vision Data Quality Analyzer project.
#
# It is used after the pipeline has generated all required reports,
# plots, models and visualizations.
# -------------------------------------------------------------------------


import subprocess


# ------------------------------------------------------------
# Start Streamlit dashboard (demo version)
# ------------------------------------------------------------

subprocess.run([
    "streamlit",
    "run",
    "app_with_tabs/dashboard_demo_version.py"
])

# change to dashboard_complete.py to include all images used in the dashboard