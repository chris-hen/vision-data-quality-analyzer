# -------------------------------------------------------------------------
# This script renders the Random Forest dashboard tab of the
# Vision Data Quality Analyzer project.
#
# The dashboard tab visualizes:
#
# - the Random Forest classification approach
# - extracted OpenCV feature inputs
# - model evaluation metrics
# - feature importance
# - confusion matrix analysis
# - prediction errors and misclassified images
#
# The Random Forest model acts as the classical feature-based
# baseline for degradation classification.
# -------------------------------------------------------------------------


import os
import pandas as pd
import streamlit as st
from PIL import Image


# folder containing generated augmented images
IMAGE_FOLDER = "data/augmented"

# Random Forest evaluation reports
RF_RESULTS_PATH = "results/reports/random_forest/random_forest_results.csv"
FEATURE_IMPORTANCE_PATH = "results/reports/random_forest/feature_importance.csv"
CONFUSION_MATRIX_PATH = "results/plots/random_forest/confusion_matrix.svg"
MISCLASSIFIED_PATH = "results/reports/random_forest/rf_misclassified_images.csv"

# visualizations used in the dashboard
RANDOM_FOREST_GRAPHIC_PATH = "results/visualizations/random_forest_concept.svg"
DECISION_TREE_GRAPHIC_PATH = "results/visualizations/random_forest_tree_example_from_model.svg"


# ------------------------------------------------------------
# Render Random Forest dashboard tab
# ------------------------------------------------------------

def render_random_forest_tab(df, filtered_df):

    # ------------------------------------------------------------
    # A. Random Forest model overview
    # ------------------------------------------------------------

    st.header("Random Forest Model")

    st.write(
        """
        This section shows how a classical machine learning model is used
        to classify generated image degradation types from extracted image
        quality features.
        """
    )

    st.caption(
        """
        The Random Forest model acts as a feature-based baseline.
        The following process overview illustrates how extracted image quality
        features are processed by the Random Forest classifier.
        """
    )

    # display Random Forest ensemble visualization
    with st.container(border=True):

        if os.path.exists(RANDOM_FOREST_GRAPHIC_PATH):
            st.image(
                RANDOM_FOREST_GRAPHIC_PATH,
                caption="Random Forest ensemble concept.",
                use_container_width=True
            )

        else:
            st.info("Random Forest concept graphic not found.")

        st.caption(
            """
            The Random Forest combines multiple decision trees. Each tree predicts
            a degradation class from the extracted image quality features. The final
            output is selected by majority voting across all trees.
            """
        )

    # expandable example tree visualization
    with st.expander("Show example decision tree from the trained model"):

        if os.path.exists(DECISION_TREE_GRAPHIC_PATH):

            st.image(
                DECISION_TREE_GRAPHIC_PATH,
                caption="Example decision tree from the trained Random Forest model",
                use_container_width=True
            )

        else:
            st.info("Decision tree graphic not found.")

        st.caption(
            """
            The example tree shows real decision thresholds from the trained model.
            It illustrates how feature values such as blur_score, brightness and
            contrast are used to separate degradation classes.
            """
        )

    # ------------------------------------------------------------
    # B. Feature-based classification setup
    # ------------------------------------------------------------

    st.subheader("Feature-Based Classification")

    # left side = feature overview table
    # right side = classification explanation
    feature_col1, spacer, feature_col2 = st.columns([1.1, 0.05, 0.9])

    with feature_col1:

        st.write("### Model Inputs and Target")

        # overview of RF input features and target labels
        feature_table = {
            "Role": [
                "Input feature",
                "Input feature",
                "Input feature",
                "Target label"
            ],
            "Value": [
                "blur_score",
                "brightness",
                "contrast",
                "degradation_type"
            ],
            "Meaning": [
                "Sharpness estimate based on Laplacian variance",
                "Average image brightness",
                "Pixel intensity variation",
                "Known augmentation class to be predicted"
            ]
        }

        # display feature overview table
        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True
        )

    with feature_col2:

        # explanation of feature-based RF classification
        with st.container(border=True):

            st.markdown(
                """
                The Random Forest model is used as the classical feature-based
                baseline for degradation classification.

                Instead of learning directly from raw image pixels, the model operates
                on structured OpenCV-derived image quality metrics extracted during
                the dataset analysis stage.

                Each image is represented by a compact numerical feature vector
                containing:

                - blur_score
                - brightness
                - contrast
                """
            )

        st.caption(
            """
            The degradation classification task is treated as a structured
            feature-based machine learning problem.
            """
        )

    # ------------------------------------------------------------
    # C. Model overview metrics
    # ------------------------------------------------------------

    st.subheader("Model Overview")

    # default fallback values
    rf_accuracy = None
    error_count = 0

    # continue only if RF evaluation report exists
    if os.path.exists(RF_RESULTS_PATH):

        # load stored RF evaluation metrics
        rf_results_df = pd.read_csv(RF_RESULTS_PATH)

        # extract final RF accuracy
        rf_accuracy = rf_results_df[
            rf_results_df["metric"] == "accuracy"
        ]["value"].iloc[0]

        # load misclassified samples if available
        if os.path.exists(MISCLASSIFIED_PATH):
            error_df = pd.read_csv(MISCLASSIFIED_PATH)
            error_count = len(error_df)

        # display compact RF overview metrics
        with st.container(border=True):

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Model", "Random Forest")

            with col2:
                st.metric("Input Features", 3)

            with col3:
                st.metric(
                    "Target Classes",
                    df["degradation_type"].nunique()
                )

            with col4:
                st.metric(
                    "Accuracy",
                    f"{rf_accuracy:.3f}"
                )

            with col5:
                st.metric(
                    "Misclassified",
                    error_count
                )

        st.caption(
            """
            The accuracy value summarizes the prediction performance on the
            test split. Misclassified samples are inspected below to understand
            where the feature-based model reaches its limits.
            """
        )

    else:
        st.info(
            "Random Forest results file not found. "
            "Run train_classifier.py first."
        )

    # ------------------------------------------------------------
    # D. Random Forest evaluation
    # ------------------------------------------------------------

    st.subheader("Random Forest Evaluation")

    st.caption(
        """
        Feature importance and the confusion matrix are used to evaluate how
        the model makes use of the extracted features and which classes
        are difficult to separate.
        """
    )

    # left side = feature importance
    # right side = confusion matrix
    col_eval1, col_eval2 = st.columns(2)

    with col_eval1:

        st.write("### Feature Importance")

        # load stored feature importance values
        if os.path.exists(FEATURE_IMPORTANCE_PATH):

            importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

            # display feature importance chart
            st.bar_chart(
                importance_df.set_index("feature")["importance"]
            )

            st.caption(
                """
                Feature importance shows which extracted image quality values
                contribute most to the Random Forest prediction.
                """
            )

        else:
            st.info(
                "Feature importance file not found. "
                "Run train_classifier.py first."
            )

    with col_eval2:

        st.write("### Confusion Matrix")

        # display RF confusion matrix visualization
        if os.path.exists(CONFUSION_MATRIX_PATH):

            st.image(
                CONFUSION_MATRIX_PATH,
                use_container_width=True
            )

            st.caption(
                """
                The confusion matrix compares actual degradation classes with
                predicted degradation classes.
                """
            )

        else:
            st.info(
                "Confusion matrix plot not found. "
                "Run train_classifier.py first."
            )

    # ------------------------------------------------------------
    # E. Random Forest error analysis
    # ------------------------------------------------------------

    st.subheader("Random Forest Error Analysis")

    st.caption(
        """
        Misclassified samples are analyzed to identify which degradation types
        are difficult for the feature-based model.
        """
    )

    # continue only if misclassification report exists
    if os.path.exists(MISCLASSIFIED_PATH):

        # load misclassified RF predictions
        error_df = pd.read_csv(MISCLASSIFIED_PATH)

        # display class-wise RF prediction errors
        col_error1, col_error2 = st.columns(2)

        with col_error1:

            st.write("### Errors by Actual Class")

            st.bar_chart(
                error_df["degradation_type"].value_counts()
            )

        with col_error2:

            st.write("### Errors by Predicted Class")

            st.bar_chart(
                error_df["predicted_degradation_type"].value_counts()
            )

        # filter RF prediction errors by actual degradation class
        error_class_filter = st.multiselect(
            "Filter errors by actual class",
            options=sorted(error_df["degradation_type"].unique()),
            default=sorted(error_df["degradation_type"].unique()),
            key="rf_error_class_filter"
        )

        # apply selected error filters
        filtered_error_df = error_df[
            error_df["degradation_type"].isin(error_class_filter)
        ]

        # display filtered RF prediction errors
        st.dataframe(
            filtered_error_df[
                [
                    "filename",
                    "degradation_type",
                    "predicted_degradation_type",
                    "blur_score",
                    "brightness",
                    "contrast"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------------------------
        # F. Misclassified image inspection
        # ------------------------------------------------------------

        st.subheader("RF Misclassified Image Inspection")

        st.caption(
            """
            Individual prediction errors of the RF model can be inspected 
            together with the corresponding image and extracted feature values.
            """
        )

        # continue only if filtered RF errors exist
        if len(filtered_error_df) > 0:

            # select individual RF prediction error
            selected_error_file = st.selectbox(
                "Select misclassified image",
                options=filtered_error_df["filename"].tolist(),
                key="rf_misclassified_image_select"
            )

            # build image path for selected prediction error
            error_image_path = os.path.join(
                IMAGE_FOLDER,
                selected_error_file
            )

            # load selected error row
            error_row = filtered_error_df[
                filtered_error_df["filename"] == selected_error_file
            ].iloc[0]

            # left side = image preview
            # right side = extracted feature values
            col_error_img, col_error_info = st.columns([2.2, 0.8])

            with col_error_img:

                # display misclassified image
                if os.path.exists(error_image_path):

                    error_image = Image.open(error_image_path)

                    st.image(
                        error_image,
                        caption=selected_error_file,
                        use_container_width=True
                    )

                else:
                    st.warning("Image file not found.")

            with col_error_info:

                # display extracted RF prediction information
                with st.container(border=True):

                    st.write("### Prediction Error")

                    st.write(
                        f"**Actual type:** "
                        f"{error_row['degradation_type']}"
                    )

                    st.write(
                        f"**Predicted type:** "
                        f"{error_row['predicted_degradation_type']}"
                    )

                    st.markdown(" ")

                    st.write(
                        f"**Blur score:** "
                        f"{error_row['blur_score']}"
                    )

                    st.write(
                        f"**Brightness:** "
                        f"{error_row['brightness']}"
                    )

                    st.write(
                        f"**Contrast:** "
                        f"{error_row['contrast']}"
                    )

        else:
            st.info("No errors match the selected filter.")

    else:
        st.info(
            "Misclassified images file not found. "
            "Run error_analysis.py first."
        )