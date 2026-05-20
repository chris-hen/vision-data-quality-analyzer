# -------------------------------------------------------------------------
# This script renders the About dashboard tab of the
# Vision Data Quality Analyzer project.
#
# The tab explains the project idea, dataset background,
# classification approaches and used methods.
#
# It gives a compact overview of:
#
# - the project purpose
# - generated image degradations
# - VisDrone as source dataset
# - Random Forest and CNN classification
# - tools and methods used in the project
# - the scope of the dashboard demo version
# -------------------------------------------------------------------------


import streamlit as st


# ------------------------------------------------------------
# Render About dashboard tab
# ------------------------------------------------------------

def render_about_tab():

    # ------------------------------------------------------------
    # A. Project overview
    # ------------------------------------------------------------

    st.header("About This Project")

    # create compact intro column on the left side
    intro_col, _ = st.columns([0.45, 0.55])

    with intro_col:

        # short project summary
        st.write(
            """
            This project analyzes and classifies image quality
            issues in computer vision datasets using classical
            machine learning and deep learning methods.
            """
        )

        # explain project focus and generated degradation data
        st.markdown(
            """
            It focuses on common problems in drone imagery
            such as blur, low contrast, brightness changes
            and image noise.

            To simulate realistic degradation effects,
            artificial image degradations are generated
            under controlled conditions for image analysis,
            feature extraction and model evaluation.

            The resulting image data and extracted features
            are then used to compare different approaches
            for automated image quality classification.
            """
        )

        # explain dashboard purpose
        st.caption(
            """
            To visualize and explore these concepts interactively,
            this Streamlit dashboard provides tools for dataset
            inspection, image analysis, model evaluation and
            prediction visualization.
            """
        )

    st.divider()

    # ------------------------------------------------------------
    # B. Purpose + Classification
    # ------------------------------------------------------------

    # left side = project purpose
    # right side = classification approaches
    purpose_col, spacer, approach_col = st.columns([1, 0.05, 1])

    with purpose_col:

        # project purpose information card
        with st.container(border=True):

            st.write("### Project Purpose")

            st.markdown(
                """
                The project investigates how image quality problems
                in drone imagery can be analyzed and classified
                automatically before images are used for further
                computer vision tasks.

                It focuses on questions such as:

                - How can image quality degradations be analyzed automatically?
                - Which image quality features are most useful for classification?
                - Can different degradation types be classified reliably?
                - How do classical ML and CNN-based approaches compare on this task?
                """
            )

    with approach_col:

        # classification approach information card
        with st.container(border=True):

            st.write("### Classification Approaches")

            st.markdown(
                """
                The project compares two different classification
                approaches:

                - Random Forest based on extracted image quality features
                - CNN based directly on image pixels

                This allows comparison between feature-based machine
                learning and end-to-end computer vision models.
                """
            )

    # ------------------------------------------------------------
    # C. Dataset
    # ------------------------------------------------------------

    # explain source dataset and generated degradation data
    with st.container():

        st.subheader("VisDrone Dataset")

        st.markdown(
            """
            The project uses images from the
            [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset),
            a large-scale benchmark for drone-based computer vision research
            introduced by Zhu et al. in *Detection and Tracking Meet Drones Challenge*
            (ECCV/ICCV Workshop Series, 2018–2020).

            The dataset contains thousands of drone-captured images and video frames
            recorded across diverse urban environments and was designed for challenging
            tasks such as object detection, object tracking and video analysis under
            real-world conditions.

            VisDrone highlights several important challenges in aerial imagery,
            including motion blur, scale variation, occlusions, illumination changes
            and small object detection which strongly influence the
            performance and reliability of computer vision models.

            In this project, artificial image degradations are generated from the
            original VisDrone images to simulate realistic quality issues such as
            blur, brightness changes, darkness, low contrast and noise. The resulting
            dataset is then used to analyze and classify image quality automatically.


            **Reference:**  
            Zhu, P., Wen, L., Du, D., Bian, X., Fan, H., Hu, Q., & Ling, H.  
            [*Detection and Tracking Meet Drones Challenge*](https://arxiv.org/abs/2001.06303),  
            arXiv:2001.06303, 2021.
            """
        )

    # ------------------------------------------------------------
    # D. Methods + Demo
    # ------------------------------------------------------------

    # left side = methods and tools
    # right side = demo version information
    methods_col, spacer, demo_col = st.columns([0.7, 0.05, 1])

    with methods_col:

        # methods and tools information card
        with st.container(border=True):

            st.subheader("Methods & Tools")

            st.markdown(
                """
                The project combines computer vision, machine learning
                and visualization techniques for image quality analysis.

                Methods and tools used throughout the project include:
                """
            )

            st.markdown(
                """
                - **Computer Vision:** OpenCV
                - **Machine Learning:** scikit-learn
                - **Deep Learning:** TensorFlow / Keras
                - **Data Analysis:** pandas
                - **Visualization:** matplotlib, Graphviz, PlotNeuralNetworks
                - **Dashboard Frontend:** Streamlit, HTML/CSS
                """
            )

    with demo_col:

        # explain scope of current dashboard demo version
        with st.container(border=True):

            st.subheader("Demo Version")

            st.markdown(
                """
                This published project is a compact demo version based on
                a small subset of the VisDrone dataset.
                
                The reduced dataset version keeps the repository
                size manageable and suitable for GitHub deployment.

                The demo dataset contains 300 original images
                that were augmented with 5 different image quality degradations,
                resulting in a total of 1800 images.

                To make the project easier to share and deploy,
                this dashboard version only includes a fixed
                subset of 30 of these images together
                with their corresponding augmentations.

                This allows the complete image quality
                analysis workflow to be explored interactively
                within the dashboard, including dataset inspection
                and model predictions.
                """
            )
        st.caption(
            """
            Because of the relatively small dataset size,
            the Random Forest and CNN models are not trained
            and optimized as extensively as they could be
            on a larger dataset version. This especially affected 
            the CNN model, which showed noticeable improvements on
            larger dataset sizes.

            """
            )