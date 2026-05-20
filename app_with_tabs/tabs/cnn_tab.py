# -------------------------------------------------------------------------
# This script renders the CNN dashboard tab of the
# Vision Data Quality Analyzer project.
#
# The tab visualizes:
#
# - the CNN architecture and classification approach
# - image-based model inputs
# - CNN training history
# - model evaluation results
# - prediction confidence behavior
# - confusion matrix analysis
# - CNN prediction errors and misclassified images
#
# Unlike the Random Forest baseline, the CNN learns degradation-related
# image patterns directly from image pixels instead of extracted
# OpenCV feature vectors.
# -------------------------------------------------------------------------


import os
import pandas as pd
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components


# folder containing generated augmented images
IMAGE_FOLDER = "data/augmented"

# cnn training and evaluation reports
CNN_HISTORY_PATH = "results/reports/cnn/cnn_training_history.csv"
CNN_RESULTS_PATH = "results/reports/cnn/cnn_results.csv"
CNN_MISCLASSIFIED_PATH = "results/reports/cnn/cnn_misclassified_images.csv"
CNN_CONFUSION_MATRIX_PATH = "results/plots/cnn/cnn_confusion_matrix.svg"
CNN_PREDICTIONS_PATH = "results/reports/cnn/cnn_predictions.csv"

# cnn architecture visualization
CNN_GRAPHIC_PATH = "results/visualizations/cnn_plotneuralnet_portfolio.svg"


# ------------------------------------------------------------
# Render CNN dashboard tab
# ------------------------------------------------------------

def render_cnn_tab(df, filtered_df):

    # ------------------------------------------------------------
    # A. CNN model overview
    # ------------------------------------------------------------

    st.header("CNN Model")

    st.write(
        """
        This section shows how a convolutional neural network is used
        to classify generated image degradation types directly from image pixels.
        """
    )

    st.caption(
        """
        Unlike the Random Forest baseline, the CNN does not rely on manually
        extracted OpenCV features. It receives the resized image itself as input
        and learns visual degradation patterns automatically during training.
        """
    )

    # display CNN architecture visualization
    with st.container():

        if os.path.exists(CNN_GRAPHIC_PATH):

            # load svg visualization
            with open(CNN_GRAPHIC_PATH, "r", encoding="utf-8") as file:
                svg_content = file.read()

            # inject responsive svg styling
            svg_content = svg_content.replace(
                "<svg",
                '<svg style="width:100%; height:auto; display:block; background:white;"',
                1
            )

            # render svg inside styled html container
            components.html(
                f"""
                <div style="
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    line-height: 0;
                    overflow: hidden;
                ">
                    {svg_content}
                </div>
                """,
                height=700,
                scrolling=False
            )

        else:
            st.info("CNN architecture graphic not found.")

        st.caption(
            """
            The CNN processes the image through convolutional feature extraction,
            spatial downsampling, global average pooling and a dense classification
            head. The final softmax layer predicts one of the generated degradation
            classes.
            """
        )


    # ------------------------------------------------------------
    # B. Image-based classification setup
    # ------------------------------------------------------------

    st.subheader("Image-Based Classification")

    # left side = cnn input overview
    # right side = cnn classification explanation
    cnn_col1, spacer, cnn_col2 = st.columns([1.1, 0.05, 0.9])

    with cnn_col1:

        st.write("### Model Inputs and Target")

        # cnn input overview table
        cnn_input_table = {
            "Role": [
                "Input data",
                "Input size",
                "Input channels",
                "Target label"
            ],
            "Value": [
                "image pixels",
                "128 × 128",
                "RGB",
                "degradation_type"
            ],
            "Meaning": [
                "Resized image data used directly by the CNN",
                "Fixed spatial input size for training and inference",
                "Three color channels per image",
                "Known augmentation class to be predicted"
            ]
        }

        # display cnn input table
        st.dataframe(
            cnn_input_table,
            use_container_width=True,
            hide_index=True
        )

    with cnn_col2:

        st.write("### Why CNN?")

        # explanation of image-based cnn classification
        with st.container(border=True):

            st.markdown(
            """
            The CNN model operates directly on image data instead of extracted
            numerical feature vectors.

            Each image is processed as a resized RGB input tensor containing
            the underlying spatial image information.

            The model learns degradation-related visual patterns directly 
            from image pixels and approaches degradation classification directly
            from image data instead of extracted feature vectors.
            """
            )

        st.caption(
            """
            This model setup turns the degradation classification task into an end-to-end
            computer vision problem.
            """
        )


    # ------------------------------------------------------------
    # C. Model overview metrics
    # ------------------------------------------------------------

    st.subheader("Model Overview")

    # fallback values
    cnn_accuracy = None
    cnn_error_count = 0

    # continue only if cnn evaluation report exists
    if os.path.exists(CNN_RESULTS_PATH):

        # load cnn evaluation results
        cnn_results_df = pd.read_csv(CNN_RESULTS_PATH)

        # use test accuracy if available
        if "test_accuracy" in cnn_results_df["metric"].values:

            cnn_accuracy = cnn_results_df[
                cnn_results_df["metric"] == "test_accuracy"
            ]["value"].iloc[0]

        # fallback to validation accuracy if needed
        elif "validation_accuracy" in cnn_results_df["metric"].values:

            cnn_accuracy = cnn_results_df[
                cnn_results_df["metric"] == "validation_accuracy"
            ]["value"].iloc[0]

        # load cnn prediction errors if available
        if os.path.exists(CNN_MISCLASSIFIED_PATH):

            cnn_error_df = pd.read_csv(CNN_MISCLASSIFIED_PATH)
            cnn_error_count = len(cnn_error_df)

        # display compact cnn overview metrics
        with st.container(border=True):

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Model", "CNN")

            with col2:
                st.metric("Input", "Image Data")

            with col3:
                st.metric(
                    "Target Classes",
                    df["degradation_type"].nunique()
                )

            with col4:

                if cnn_accuracy is not None:

                    st.metric(
                        "Accuracy",
                        f"{cnn_accuracy:.3f}"
                    )

                else:
                    st.metric("Accuracy", "n/a")

            with col5:
                st.metric(
                    "Misclassified",
                    cnn_error_count
                )

        st.caption(
            """
            The accuracy value summarizes the CNN performance on the test split. 
            Misclassified samples are analyzed separately to identify
            which degradation types remain difficult for the image-based model.
            """
        )

    else:
        st.info("CNN results file not found. Run train_cnn.py first.")


    # ------------------------------------------------------------
    # D. CNN training history
    # ------------------------------------------------------------

    st.subheader("CNN Training History")

    # continue only if cnn training history exists
    if os.path.exists(CNN_HISTORY_PATH):

        # load cnn training history
        cnn_history_df = pd.read_csv(CNN_HISTORY_PATH)

        # left side = accuracy curves
        # right side = loss curves
        col_history1, col_history2 = st.columns(2)

        with col_history1:

            st.write("### Accuracy by Epoch")

            # display training and validation accuracy
            st.line_chart(
                cnn_history_df.set_index("epoch")[["accuracy", "val_accuracy"]]
            )

        with col_history2:

            st.write("### Loss by Epoch")

            # display training and validation loss
            st.line_chart(
                cnn_history_df.set_index("epoch")[["loss", "val_loss"]]
            )

        st.caption(
            """
            The training curves show how well the CNN learns from the training data
            and how stable the validation performance remains during training.
            Minor fluctuations between epochs are expected because the model is 
            trained on randomly shuffled mini-batches and a relatively small dataset.
            """
        )

    else:
        st.info("CNN training history not found. Run train_cnn.py first.")

    # ------------------------------------------------------------
    # E. CNN evaluation
    # ------------------------------------------------------------

    st.subheader("CNN Evaluation")

    # left side = prediction confidence
    # right side = confusion matrix
    col_cnn_eval1, col_cnn_eval2 = st.columns(2)

    with col_cnn_eval1:

        st.write("### CNN Confidence Distribution")

        # continue only if cnn prediction report exists
        if os.path.exists(CNN_PREDICTIONS_PATH):

            # load cnn predictions
            cnn_predictions_df = pd.read_csv(CNN_PREDICTIONS_PATH)

            # split confidence values into intervals
            confidence_bins = pd.cut(
                cnn_predictions_df["cnn_prediction_confidence"],
                bins=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
                labels=[
                    "0.0  - 0.5",
                    "0.5  - 0.6",
                    "0.6  - 0.7",
                    "0.7  - 0.8",
                    "0.8  - 0.9",
                    "0.9  - 0.95",
                    "0.95 - 1.0"
                ],
                include_lowest=True
            )

            # count predictions per confidence interval
            confidence_counts = (
                confidence_bins
                .value_counts()
                .sort_index()
                .reset_index()
            )

            confidence_counts.columns = ["confidence_range", "count"]

            # display cnn confidence distribution
            st.bar_chart(
                confidence_counts,
                x="confidence_range",
                y="count"
            )

            st.caption(
            """
            The confidence distribution and confusion matrix provide insight into
            prediction confidence and class-level classification behavior of the CNN.
            """
           )

        else:
            st.info(
                "CNN predictions file not found. "
                "Run error_analysis_cnn.py first."
            )

    with col_cnn_eval2:

        st.write("### CNN Confusion Matrix")

        # display cnn confusion matrix
        if os.path.exists(CNN_CONFUSION_MATRIX_PATH):

            st.image(
                CNN_CONFUSION_MATRIX_PATH,
                use_container_width=True
            )

        else:
            st.info(
                "CNN confusion matrix not found. "
                "Run error_analysis_cnn.py first."
            )

    # ------------------------------------------------------------
    # F. CNN error analysis
    # ------------------------------------------------------------

    st.subheader("CNN Error Analysis")

    st.caption(
        """
        Misclassified samples are analyzed to identify which degradation types
        remain difficult for the CNN image classifier.
        """
    )

    # continue only if cnn error report exists
    if os.path.exists(CNN_MISCLASSIFIED_PATH):

        # load cnn prediction errors
        cnn_error_df = pd.read_csv(CNN_MISCLASSIFIED_PATH)

        # left side = actual classes
        # right side = predicted classes
        col_cnn_error1, col_cnn_error2 = st.columns(2)

        with col_cnn_error1:

            st.write("### CNN Errors by Actual Class")

            # display actual error class distribution
            st.bar_chart(
                cnn_error_df["degradation_type"].value_counts()
            )

        with col_cnn_error2:

            st.write("### CNN Errors by Predicted Class")

            # display predicted error class distribution
            st.bar_chart(
                cnn_error_df["cnn_predicted_degradation_type"].value_counts()
            )

        # filter cnn errors by actual degradation class
        cnn_error_filter = st.multiselect(
            "Filter CNN errors by actual class",
            options=sorted(cnn_error_df["degradation_type"].unique()),
            default=sorted(cnn_error_df["degradation_type"].unique()),
            key="cnn_error_class_filter"
        )

        # apply cnn error filters
        filtered_cnn_error_df = cnn_error_df[
            cnn_error_df["degradation_type"].isin(cnn_error_filter)
        ]

        # display filtered cnn prediction errors
        st.dataframe(
            filtered_cnn_error_df[
                [
                    "filename",
                    "degradation_type",
                    "cnn_predicted_degradation_type",
                    "blur_score",
                    "brightness",
                    "contrast"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------------------------
        # G. CNN misclassified image inspection
        # ------------------------------------------------------------

        st.subheader("CNN Misclassified Image Inspection")

        st.caption(
            """
            Individual CNN prediction errors can be inspected together with the
            corresponding image and extracted feature values.
            """
        )

        # continue only if filtered cnn errors exist
        if len(filtered_cnn_error_df) > 0:

            # select individual cnn prediction error
            selected_cnn_error_file = st.selectbox(
                "Select CNN misclassified image",
                options=filtered_cnn_error_df["filename"].tolist(),
                key="cnn_misclassified_image_inspection_select"
            )

            # build full image path for selected cnn prediction error
            cnn_error_image_path = os.path.join(
                IMAGE_FOLDER,
                selected_cnn_error_file
            )

            # select dataframe row for selected prediction error
            cnn_error_row = filtered_cnn_error_df[
                filtered_cnn_error_df["filename"] == selected_cnn_error_file
            ].iloc[0]

            # left side = image preview
            # right side = prediction information
            col_cnn_error_img, col_cnn_error_info = st.columns([2.2, 0.8])

            with col_cnn_error_img:

                # display selected misclassified image
                if os.path.exists(cnn_error_image_path):

                    cnn_error_image = Image.open(cnn_error_image_path)

                    st.image(
                        cnn_error_image,
                        caption=selected_cnn_error_file,
                        use_container_width=True
                    )

                else:
                    st.warning("Image file not found.")

            with col_cnn_error_info:

                # display cnn prediction information
                with st.container(border=True):

                    st.write("### CNN Prediction Error")

                    st.write(
                        f"**Actual type:** "
                        f"{cnn_error_row['degradation_type']}"
                    )

                    st.write(
                        f"**Predicted type:** "
                        f"{cnn_error_row['cnn_predicted_degradation_type']}"
                    )

                    st.markdown(" ")

                    st.write(
                        f"**Blur score:** "
                        f"{cnn_error_row['blur_score']}"
                    )

                    st.write(
                        f"**Brightness:** "
                        f"{cnn_error_row['brightness']}"
                    )

                    st.write(
                        f"**Contrast:** "
                        f"{cnn_error_row['contrast']}"
                    )

        else:
            st.info("No CNN errors match the selected filter.")

    else:
        st.info(
            "CNN misclassified images file not found. "
            "Run error_analysis_cnn.py first."
        )