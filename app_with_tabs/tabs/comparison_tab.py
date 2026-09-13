# -------------------------------------------------------------------------
# This script renders the Model Comparison dashboard tab of the
# Vision Data Quality Analyzer project.
#
# The tab compares both classification approaches directly:
#
# - overall accuracy and error counts
# - per-class performance differences
# - cases in which both models disagree
# - prediction confidence behavior
# - feature ranges in which predictions fail
# - confusion patterns of both models
# - confidence threshold simulation
# - difficult images and individual case inspection
#
# Unlike the single model tabs, this tab combines the Random Forest
# feature predictions and the CNN image predictions in one shared
# dataframe to make both approaches directly comparable.
# -------------------------------------------------------------------------


import os
import pandas as pd
import streamlit as st
from PIL import Image
import joblib


# folder containing generated augmented images
IMAGE_FOLDER = "data/augmented"

# trained Random Forest model used for live feature predictions
MODEL_PATH = "models/random_forest_model.pkl"

# evaluation reports of both models
RF_RESULTS_PATH = "results/reports/random_forest/random_forest_results.csv"
CNN_RESULTS_PATH = "results/reports/cnn/cnn_results.csv"

# misclassified image reports of both models
RF_MISCLASSIFIED_PATH = "results/reports/random_forest/rf_misclassified_images.csv"
CNN_MISCLASSIFIED_PATH = "results/reports/cnn/cnn_misclassified_images.csv"

# CNN predictions including extracted features of every test image
CNN_PREDICTIONS_PATH = "results/reports/cnn/cnn_predictions.csv"


# ------------------------------------------------------------
# Render Model Comparison dashboard tab
# ------------------------------------------------------------

def render_comparison_tab(df, filtered_df):

    # ------------------------------------------------------------
    # Model comparison overview
    # ------------------------------------------------------------

    st.header("Model Comparison")

    st.write(
        """
        This section compares the feature-based Random Forest approach
        with the image-based CNN approach on the same test dataset.
        """
    )

    st.caption(
        """
        Both models are evaluated on identical images, which allows a
        direct comparison of accuracy, per-class behavior, prediction
        confidence and the cases in which both approaches disagree.
        """
    )

    rf_accuracy = None
    cnn_accuracy = None
    rf_model = None

    if os.path.exists(RF_RESULTS_PATH):
        rf_results_df = pd.read_csv(RF_RESULTS_PATH)

        if "accuracy" in rf_results_df["metric"].values:

            rf_accuracy = rf_results_df[
                rf_results_df["metric"] == "accuracy"
            ]["value"].iloc[0]

    if os.path.exists(CNN_RESULTS_PATH):
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

    if os.path.exists(MODEL_PATH):
        rf_model = joblib.load(MODEL_PATH)

    # ------------------------------------------------------------
    # A. Overall model performance
    # ------------------------------------------------------------

    st.subheader("Overall Model Performance")

    st.caption(
        """
        Overall accuracy and the number of misclassified images give a first
        indication of how both approaches perform on the same test dataset.
        """
    )


    rf_error_count = 0
    cnn_error_count = 0

    if os.path.exists(RF_MISCLASSIFIED_PATH):
        rf_error_df = pd.read_csv(RF_MISCLASSIFIED_PATH)
        rf_error_count = len(rf_error_df)

    if os.path.exists(CNN_MISCLASSIFIED_PATH):
        cnn_error_df = pd.read_csv(CNN_MISCLASSIFIED_PATH)
        cnn_error_count = len(cnn_error_df)

    if rf_accuracy is not None and cnn_accuracy is not None:

        accuracy_difference = abs(rf_accuracy - cnn_accuracy)

        if rf_accuracy > cnn_accuracy:
            better_model = "Random Forest"
        elif cnn_accuracy > rf_accuracy:
            better_model = "CNN"
        else:
            better_model = "Both models"

        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)

        col_perf1.metric(
            "Random Forest accuracy",
            f"{rf_accuracy:.3f}"
        )

        col_perf2.metric(
            "CNN accuracy",
            f"{cnn_accuracy:.3f}"
        )

        col_perf3.metric(
            "Accuracy difference",
            f"{accuracy_difference:.3f}"
        )

        col_perf4.metric(
            "Better model",
            better_model
        )

        col_info1, col_info2, _, _ = st.columns(4)

        col_info1.write(
            f"{rf_error_count} misclassified images"
        )

        col_info2.write(
            f"{cnn_error_count} misclassified images"
        )

    else:
        st.info("Model accuracy results not available yet.")

    # ------------------------------------------------------------
    # Main model comparison dataframe
    # ------------------------------------------------------------

    if os.path.exists(CNN_PREDICTIONS_PATH) and rf_model is not None:

        model_comparison_df = pd.read_csv(CNN_PREDICTIONS_PATH)

        rf_features = model_comparison_df[
            ["blur_score", "brightness", "contrast"]
        ]

        model_comparison_df["rf_predicted_degradation_type"] = rf_model.predict(
            rf_features
        )

        rf_probabilities = rf_model.predict_proba(rf_features)

        model_comparison_df["rf_confidence"] = rf_probabilities.max(axis=1)

        model_comparison_df["rf_prediction_correct"] = (
            model_comparison_df["rf_predicted_degradation_type"] ==
            model_comparison_df["degradation_type"]
        )

        # ------------------------------------------------------------
        # B. Per-class performance
        # ------------------------------------------------------------

        st.subheader("Per-Class Performance")

        st.caption(
            """
            A higher overall accuracy does not necessarily mean better behavior on
            every degradation type. The per-class view shows where each approach
            is actually stronger.
            """
        )


        rf_class_accuracy = (
            model_comparison_df
            .groupby("degradation_type")["rf_prediction_correct"]
            .mean()
        )

        cnn_class_accuracy = (
            model_comparison_df
            .groupby("degradation_type")["cnn_prediction_correct"]
            .mean()
        )

        class_accuracy_comparison = pd.DataFrame({
            "degradation_type": rf_class_accuracy.index,
            "Random Forest": rf_class_accuracy.values,
            "CNN": cnn_class_accuracy.values
        })

        st.write("### Accuracy per Class")

        st.dataframe(
            class_accuracy_comparison,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            class_accuracy_comparison,
            x="degradation_type",
            y=["Random Forest", "CNN"],
            stack=False
        )

        # ------------------------------------------------------------
        # Hardest classes
        # ------------------------------------------------------------

        st.write("### Hardest Classes")

        st.caption(
            """
            Classes with the lowest accuracy indicate which degradation types are
            most difficult to separate for the respective model.
            """
        )


        hardest_classes_df = class_accuracy_comparison.copy()

        hardest_classes_df["RF error rate"] = (
            1 - hardest_classes_df["Random Forest"]
        )

        hardest_classes_df["CNN error rate"] = (
            1 - hardest_classes_df["CNN"]
        )

        hardest_classes_df["average error rate"] = (
            hardest_classes_df["RF error rate"] +
            hardest_classes_df["CNN error rate"]
        ) / 2

        hardest_classes_df = hardest_classes_df.sort_values(
            by="average error rate",
            ascending=False
        )

        st.dataframe(
            hardest_classes_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            hardest_classes_df,
            x="degradation_type",
            y=["RF error rate", "CNN error rate"],
            stack=False
        )

        # ------------------------------------------------------------
        # Model strength by class
        # ------------------------------------------------------------

        st.write("### Stronger Model per Class")

        st.caption(
            """
            Comparing both models class by class shows whether the advantage of one
            approach is consistent or limited to individual degradation types.
            """
        )


        model_strength_df = class_accuracy_comparison.copy()

        def get_stronger_model(row):
            if row["Random Forest"] > row["CNN"]:
                return "Random Forest"
            if row["CNN"] > row["Random Forest"]:
                return "CNN"
            return "Tie"

        model_strength_df["stronger_model"] = model_strength_df.apply(
            get_stronger_model,
            axis=1
        )

        model_strength_df["accuracy_difference"] = abs(
            model_strength_df["Random Forest"] -
            model_strength_df["CNN"]
        )

        st.dataframe(
            model_strength_df,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------------------------
        # C. Model disagreement analysis
        # ------------------------------------------------------------

        st.subheader("Model Disagreement Analysis")

        st.caption(
            """
            Images on which both models predict different classes are especially
            informative, because they show where feature-based and image-based
            classification diverge.
            """
        )


        def get_agreement_case(row):

            if (
                row["cnn_prediction_correct"] and
                not row["rf_prediction_correct"]
            ):
                return "CNN only correct"

            if (
                row["rf_prediction_correct"] and
                not row["cnn_prediction_correct"]
            ):
                return "RF only correct"

            if (
                not row["rf_prediction_correct"] and
                not row["cnn_prediction_correct"]
            ):
                return "Both incorrect"

            return "Both correct"

        model_comparison_df["case_type"] = model_comparison_df.apply(
            get_agreement_case,
            axis=1
        )

        disagreement_df = model_comparison_df[
            model_comparison_df["case_type"] != "Both correct"
        ]

        st.write("### Overall Disagreement Summary")

        full_case_summary = (
            model_comparison_df["case_type"]
            .value_counts()
            .reset_index()
        )

        full_case_summary.columns = ["case_type", "count"]

        graph_case_summary = full_case_summary[
            full_case_summary["case_type"] != "Both correct"
        ]

        col_summary_table, col_summary_chart = st.columns([1, 1.4])

        with col_summary_table:

            st.dataframe(
                full_case_summary,
                use_container_width=True,
                hide_index=True
            )

        with col_summary_chart:

            if len(graph_case_summary) > 0:

                chart_df = graph_case_summary.set_index("case_type")

                st.bar_chart(
                    chart_df,
                    height=200,
                    horizontal=True
                )

            else:
                st.info("No disagreement cases found.")

        st.write("### Disagreement by Actual Class")

        st.caption(
            """
            Disagreements are not distributed evenly across classes. Degradation
            types with many disagreements are candidates for further inspection.
            """
        )


        if len(disagreement_df) > 0:

            disagreement_by_class = (
                disagreement_df
                .groupby(["degradation_type", "case_type"])
                .size()
                .reset_index(name="count")
            )

            disagreement_pivot = disagreement_by_class.pivot(
                index="degradation_type",
                columns="case_type",
                values="count"
            ).fillna(0)

            st.bar_chart(
                disagreement_pivot,
                stack=False
            )

        else:
            st.info("No class-level disagreement cases available.")

        # ------------------------------------------------------------
        # D. Confidence analysis
        # ------------------------------------------------------------

        st.subheader("Confidence Analysis")

        st.caption(
            """
            Besides the predicted class, both models also provide a confidence value.
            Comparing confidence and correctness shows how reliable these values are.
            """
        )


        st.write("### Average Confidence by Disagreement Type")

        if len(disagreement_df) > 0:

            confidence_by_case = (
                disagreement_df
                .groupby("case_type")[
                    ["rf_confidence", "cnn_prediction_confidence"]
                ]
                .mean()
                .reset_index()
            )

            confidence_by_case = confidence_by_case.rename(
                columns={
                    "rf_confidence": "Random Forest",
                    "cnn_prediction_confidence": "CNN"
                }
            )

            st.dataframe(
                confidence_by_case,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                confidence_by_case,
                x="case_type",
                y=["Random Forest", "CNN"],
                stack=False
            )

        else:
            st.info("No disagreement cases found for confidence analysis.")

        st.write("### Confidence by Correctness")

        st.caption(
            """
            A well behaved model should be noticeably less confident on its
            own incorrect predictions. On this dataset the CNN shows exactly
            this behavior, which makes its confidence value usable as a
            rejection criterion.
            """
        )


        rf_confidence_correctness = (
            model_comparison_df
            .groupby("rf_prediction_correct")["rf_confidence"]
            .mean()
            .reset_index()
        )

        rf_confidence_correctness.columns = [
            "prediction_correct",
            "Random Forest"
        ]

        cnn_confidence_correctness = (
            model_comparison_df
            .groupby("cnn_prediction_correct")["cnn_prediction_confidence"]
            .mean()
            .reset_index()
        )

        cnn_confidence_correctness.columns = [
            "prediction_correct",
            "CNN"
        ]

        confidence_correctness_df = pd.merge(
            rf_confidence_correctness,
            cnn_confidence_correctness,
            on="prediction_correct",
            how="outer"
        )

        confidence_correctness_df["prediction_correct"] = (
            confidence_correctness_df["prediction_correct"]
            .astype(str)
        )

        st.dataframe(
            confidence_correctness_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            confidence_correctness_df,
            x="prediction_correct",
            y=["Random Forest", "CNN"],
            stack=False
        )

        st.write("### Average Confidence by Actual Class")

        confidence_by_class = (
            model_comparison_df
            .groupby("degradation_type")[
                ["rf_confidence", "cnn_prediction_confidence"]
            ]
            .mean()
            .reset_index()
        )

        confidence_by_class = confidence_by_class.rename(
            columns={
                "rf_confidence": "Random Forest",
                "cnn_prediction_confidence": "CNN"
            }
        )

        st.dataframe(
            confidence_by_class,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            confidence_by_class,
            x="degradation_type",
            y=["Random Forest", "CNN"],
            stack=False
        )

        # ------------------------------------------------------------
        # Feature Correlation with Errors
        # ------------------------------------------------------------

        st.subheader("Feature Correlation with Prediction Errors")

        st.caption(
            """
            Because the extracted OpenCV features are available for every image,
            prediction errors can be related directly to measurable image properties.
            """
        )


        feature_error_analysis = model_comparison_df.copy()

        feature_error_analysis["rf_error"] = (
            feature_error_analysis["rf_prediction_correct"] == False
        ).astype(int)

        feature_error_analysis["cnn_error"] = (
            feature_error_analysis["cnn_prediction_correct"] == False
        ).astype(int)

        rf_feature_correlation = feature_error_analysis[
            [
                "blur_score",
                "brightness",
                "contrast",
                "rf_error"
            ]
        ].corr()["rf_error"].drop("rf_error")

        cnn_feature_correlation = feature_error_analysis[
            [
                "blur_score",
                "brightness",
                "contrast",
                "cnn_error"
            ]
        ].corr()["cnn_error"].drop("cnn_error")

        feature_correlation_df = pd.DataFrame({
            "feature": rf_feature_correlation.index,
            "RF error correlation": rf_feature_correlation.values,
            "CNN error correlation": cnn_feature_correlation.values
        })

        st.write("### Correlation between Features and Prediction Errors")

        st.dataframe(
            feature_correlation_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            feature_correlation_df,
            x="feature",
            y=["RF error correlation", "CNN error correlation"],
            stack=False
        )

        # ------------------------------------------------------------
        # Error Rate by Feature Range
        # ------------------------------------------------------------

        st.subheader("Error Rate by Feature Range")

        st.caption(
            """
            Splitting the feature values into ranges shows whether errors
            accumulate in specific brightness, contrast or blur regions.
            Errors of both models are concentrated in the value ranges in
            which the untouched original images overlap with the darker and
            lower contrast classes.
            """
        )


        selected_feature = st.selectbox(
            "Select feature for error rate analysis",
            ["blur_score", "brightness", "contrast"],
            key="feature_error_range_analysis"
        )

        feature_range_df = model_comparison_df.copy()

        feature_range_df["feature_bin"] = pd.cut(
            feature_range_df[selected_feature],
            bins=5
        )

        rf_accuracy_by_range = (
            feature_range_df
            .groupby("feature_bin")["rf_prediction_correct"]
            .mean()
        )

        cnn_accuracy_by_range = (
            feature_range_df
            .groupby("feature_bin")["cnn_prediction_correct"]
            .mean()
        )

        error_rate_range_df = pd.DataFrame({
            "feature_range": rf_accuracy_by_range.index.astype(str),
            "RF accuracy": rf_accuracy_by_range.values,
            "CNN accuracy": cnn_accuracy_by_range.values
        })

        error_rate_range_df["RF error rate"] = (
            1 - error_rate_range_df["RF accuracy"]
        )

        error_rate_range_df["CNN error rate"] = (
            1 - error_rate_range_df["CNN accuracy"]
        )

        st.write(f"### Error Rate by {selected_feature}")

        st.dataframe(
            error_rate_range_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            error_rate_range_df,
            x="feature_range",
            y=["RF error rate", "CNN error rate"],
            stack=False
        )

        # ------------------------------------------------------------
        # E. Confusion analysis
        # ------------------------------------------------------------

        st.subheader("Confusion Analysis")

        st.caption(
            """
            Most confused class pairs reveal which degradation types are visually or
            numerically similar enough to be mixed up by the models.
            """
        )


        st.write("### CNN Most Confused Class Pairs")

        cnn_confused_pairs = model_comparison_df[
            model_comparison_df["cnn_prediction_correct"] == False
        ]

        if len(cnn_confused_pairs) > 0:

            cnn_confused_pairs = (
                cnn_confused_pairs
                .groupby(
                    [
                        "degradation_type",
                        "cnn_predicted_degradation_type"
                    ]
                )
                .size()
                .reset_index(name="count")
                .sort_values(by="count", ascending=False)
            )

            cnn_confused_pairs["confusion_pair"] = (
                cnn_confused_pairs["degradation_type"] +
                " → " +
                cnn_confused_pairs["cnn_predicted_degradation_type"]
            )

            st.dataframe(
                cnn_confused_pairs,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                cnn_confused_pairs,
                x="confusion_pair",
                y="count"
            )

        else:
            st.info("No CNN confusion pairs found.")

        st.write("### Random Forest Most Confused Class Pairs")

        rf_confused_pairs = model_comparison_df[
            model_comparison_df["rf_prediction_correct"] == False
        ]

        if len(rf_confused_pairs) > 0:

            rf_confused_pairs = (
                rf_confused_pairs
                .groupby(
                    [
                        "degradation_type",
                        "rf_predicted_degradation_type"
                    ]
                )
                .size()
                .reset_index(name="count")
                .sort_values(by="count", ascending=False)
            )

            rf_confused_pairs["confusion_pair"] = (
                rf_confused_pairs["degradation_type"] +
                " → " +
                rf_confused_pairs["rf_predicted_degradation_type"]
            )

            st.dataframe(
                rf_confused_pairs,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                rf_confused_pairs,
                x="confusion_pair",
                y="count"
            )

        else:
            st.info("No Random Forest confusion pairs found.")

        # ------------------------------------------------------------
        # F. Confidence threshold simulation
        # ------------------------------------------------------------

        st.subheader("Confidence Threshold Simulation")

        st.caption(
            """
            Rejecting low confidence predictions increases accuracy on the remaining
            predictions, but reduces how many images can be classified automatically.
            The simulation shows this trade-off.
            """
        )


        threshold_values = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

        threshold_results = []

        total_images = len(model_comparison_df)

        for threshold in threshold_values:

            rf_accepted = model_comparison_df[
                model_comparison_df["rf_confidence"] >= threshold
            ]

            cnn_accepted = model_comparison_df[
                model_comparison_df["cnn_prediction_confidence"] >= threshold
            ]

            if len(rf_accepted) > 0:
                rf_threshold_accuracy = rf_accepted[
                    "rf_prediction_correct"
                ].mean()
            else:
                rf_threshold_accuracy = 0

            if len(cnn_accepted) > 0:
                cnn_threshold_accuracy = cnn_accepted[
                    "cnn_prediction_correct"
                ].mean()
            else:
                cnn_threshold_accuracy = 0

            threshold_results.append({
                "threshold": threshold,
                "RF coverage": len(rf_accepted) / total_images,
                "CNN coverage": len(cnn_accepted) / total_images,
                "RF accepted accuracy": rf_threshold_accuracy,
                "CNN accepted accuracy": cnn_threshold_accuracy
            })

        threshold_df = pd.DataFrame(threshold_results)

        st.write("### Coverage by Confidence Threshold")

        st.dataframe(
            threshold_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            threshold_df,
            x="threshold",
            y=["RF coverage", "CNN coverage"],
            stack=False
        )

        st.write("### Accuracy of Accepted Predictions by Threshold")

        st.bar_chart(
            threshold_df,
            x="threshold",
            y=["RF accepted accuracy", "CNN accepted accuracy"],
            stack=False
        )

        # ------------------------------------------------------------
        # G. Top difficult images
        # ------------------------------------------------------------

        st.subheader("Top Difficult Images")

        st.caption(
            """
            Images that both models classify incorrectly or with low confidence are
            the most useful starting point for further dataset improvements.
            """
        )


        difficult_images_df = model_comparison_df.copy()

        difficult_images_df["average_confidence"] = (
            difficult_images_df["rf_confidence"] +
            difficult_images_df["cnn_prediction_confidence"]
        ) / 2

        difficult_images_df["both_wrong"] = (
            (difficult_images_df["rf_prediction_correct"] == False) &
            (difficult_images_df["cnn_prediction_correct"] == False)
        )

        difficult_images_df = difficult_images_df.sort_values(
            by=["both_wrong", "average_confidence"],
            ascending=[False, False]
        )

        st.dataframe(
            difficult_images_df[
                [
                    "filename",
                    "degradation_type",
                    "rf_predicted_degradation_type",
                    "cnn_predicted_degradation_type",
                    "rf_prediction_correct",
                    "cnn_prediction_correct",
                    "rf_confidence",
                    "cnn_prediction_confidence",
                    "average_confidence",
                    "both_wrong"
                ]
            ].head(20),
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------------------------
        # H. Interesting case explorer
        # ------------------------------------------------------------

        st.subheader("Interesting Case Explorer")

        st.caption(
            """
            Individual disagreement cases can be inspected together with the image
            itself to understand what caused the different predictions.
            """
        )


        if len(disagreement_df) > 0:

            st.dataframe(
                disagreement_df[
                    [
                        "filename",
                        "degradation_type",
                        "rf_predicted_degradation_type",
                        "cnn_predicted_degradation_type",
                        "case_type",
                        "rf_confidence",
                        "cnn_prediction_confidence"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            selected_interesting_file = st.selectbox(
                "Select disagreement case",
                options=disagreement_df["filename"].tolist(),
                key="model_disagreement_case_selectbox_final"
            )

            if selected_interesting_file:

                interesting_row = disagreement_df[
                    disagreement_df["filename"] == selected_interesting_file
                ].iloc[0]

                interesting_image_path = os.path.join(
                    IMAGE_FOLDER,
                    selected_interesting_file
                )

                col_case_img, col_case_info = st.columns([2, 1])

                with col_case_img:

                    if os.path.exists(interesting_image_path):
                        case_image = Image.open(interesting_image_path)

                        st.image(
                            case_image,
                            caption=selected_interesting_file,
                            use_container_width=True
                        )
                    else:
                        st.warning("Image file not found.")

                with col_case_info:

                    st.write("### Case Details")

                    st.write(
                        f"**Actual class:** "
                        f"{interesting_row['degradation_type']}"
                    )

                    st.write(
                        f"**Case type:** "
                        f"{interesting_row['case_type']}"
                    )

                    st.write(
                        f"**RF prediction:** "
                        f"{interesting_row['rf_predicted_degradation_type']}"
                    )

                    st.write(
                        f"**RF confidence:** "
                        f"{interesting_row['rf_confidence']:.3f}"
                    )

                    st.write(
                        f"**CNN prediction:** "
                        f"{interesting_row['cnn_predicted_degradation_type']}"
                    )

                    st.write(
                        f"**CNN confidence:** "
                        f"{interesting_row['cnn_prediction_confidence']:.3f}"
                    )

                    if interesting_row["rf_prediction_correct"]:
                        st.success("RF prediction correct")
                    else:
                        st.error("RF prediction incorrect")

                    if interesting_row["cnn_prediction_correct"]:
                        st.success("CNN prediction correct")
                    else:
                        st.error("CNN prediction incorrect")

        else:
            st.info(
                "No disagreement cases found. "
                "Both models agree on all predictions."
            )

        # ------------------------------------------------------------
        # I. Result interpretation
        # ------------------------------------------------------------

        st.divider()

        st.subheader("Result Interpretation")

        interpretation_col, _ = st.columns([0.7, 0.3])

        with interpretation_col:

            with st.container(border=True):

                st.markdown(
                    """
                    Both approaches solve the same classification task
                    with fundamentally different inputs.

                    The Random Forest uses three explicitly extracted
                    OpenCV features, while the CNN learns degradation
                    patterns directly from image pixels.

                    On this dataset the feature-based approach is clearly
                    ahead. The three extracted features describe the
                    generated degradations almost directly, which is a
                    strong advantage on a dataset of this size.
                    """
                )

                if rf_accuracy is not None and cnn_accuracy is not None:

                    st.markdown(
                        f"""
                        On this dataset version the better overall accuracy
                        was reached by **{better_model}**, with an accuracy
                        difference of {accuracy_difference:.3f}.
                        """
                    )

                st.markdown(
                    """
                    The per-class results show that both models fail in
                    different ways.

                    Errors of the Random Forest are almost entirely limited
                    to the classes original, dark and lowcontrast. These
                    classes overlap in their brightness and contrast values,
                    so the decision boundaries between them are the weak
                    point of the feature-based approach.

                    The CNN reaches comparable accuracy on blur, bright,
                    dark and lowcontrast, but breaks down on noise. Most
                    noisy images are predicted as original or dark, which in
                    turn reduces the precision of the original class. Added
                    noise changes the pixel statistics of an image without
                    changing its global structure, and the CNN does not
                    learn this difference reliably from the available
                    training data. The Laplacian variance used by the Random
                    Forest reacts to exactly this change, which explains the
                    different error behavior.

                    The training history supports this. The validation
                    accuracy of the CNN fluctuates strongly and ends far
                    below its best value, which indicates that the amount of
                    training data is the limiting factor rather than the
                    architecture itself. On larger dataset versions the CNN
                    showed noticeable improvements.
                    """
                )

                st.markdown(
                    """
                    The advantage of the feature-based model also follows
                    partly from the experimental setup. The degradations are
                    generated under controlled conditions, and the extracted
                    features measure almost exactly the image properties that
                    are modified during augmentation. The Random Forest
                    therefore works on values that describe the target classes
                    directly, while the CNN has to learn the same relation
                    from image data.

                    The comparison shows how both approaches behave on this
                    specific task rather than which approach is generally
                    better suited for image quality classification.
                    """
                )

                st.caption(
                    """
                    The results shown here are based on the reduced demo
                    dataset and on a fixed train, validation and test split
                    in which all augmentations of one source image stay
                    within the same split. The comparison is therefore
                    intended to illustrate the evaluation workflow and the
                    different error behavior of both approaches rather than
                    to provide a final benchmark of either model.
                    """
                )

    else:
        st.info(
            "CNN predictions file or Random Forest model not found. "
            "Run error_analysis_cnn.py and train_classifier.py first."
        )
