# -------------------------------------------------------------------------
# This script renders the dataset and image quality dashboard tab
# of the Vision Data Quality Analyzer project.
#
# The tab shows how raw images are transformed into a structured
# image quality dataset using:
#
# - image augmentation
# - extracted OpenCV quality values
# - rule-based quality labels
# - degradation type labels for model training
# - interactive image inspection
#
# The extracted values and labels are used for dataset analysis,
# dashboard visualization and later RF/CNN model training.
# -------------------------------------------------------------------------


import os
import streamlit as st
from PIL import Image
import pandas as pd


# general image folder used for image inspection and previews
IMAGE_FOLDER = "data/augmented"

# folder containing the original raw input images
RAW_IMAGE_FOLDER = "data/raw_images"

# folder containing generated augmented image variants
AUGMENTED_IMAGE_FOLDER = "data/augmented"


# ------------------------------------------------------------
# Show augmentation parameters
# ------------------------------------------------------------

# display the augmentation parameters for a selected augmented image
def show_augmentation_parameters(aug_df, augmented_filename):

    # select the metadata row that belongs to the displayed augmented image
    metadata_row = aug_df[
        aug_df["augmented_filename"] == augmented_filename
    ]

    # if no metadata is available, show fallback message
    if metadata_row.empty:
        st.caption("Parameters not available")
        return

    # read stored augmentation parameter string from metadata
    parameters = metadata_row.iloc[0]["parameters"]

    # clean and format parameter string for better dashboard readability
    parameters = (
        parameters
        .replace("{", "")
        .replace("}", "")
        .replace("'", "")
        .replace("np.float64(", "")
        .replace(")", "")
        .replace(",", " |")
        .replace("_", " ")
    )

    # rename parameter keys for cleaner display
    parameters = parameters.replace("kernel size", "Kernel size")
    parameters = parameters.replace("alpha", "Alpha")
    parameters = parameters.replace("beta", "Beta")
    parameters = parameters.replace("mean", "Mean")
    parameters = parameters.replace("sigma", "Sigma")

    # display formatted parameters below the image
    st.markdown(
        f"""
        <p style='
            font-size:10px;
            margin-top:-22px;
            text-align:center;
            color:rgba(220,220,220,0.65);
        '>
            {parameters}
        </p>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# Render dataset dashboard tab
# ------------------------------------------------------------

def render_dataset_tab(df, filtered_df, aug_df):
    
    # ------------------------------------------------------------
    # Main section title
    # ------------------------------------------------------------

    st.header("Dataset & Image Quality Analysis")

    st.write(
    """
    This section shows how the dataset is generated and how image quality
    features are extracted.
    
    """
)

    # ------------------------------------------------------------
    # A. Dataset generation and augmentation preview
    # ------------------------------------------------------------

    st.subheader("Dataset Generation and Image Augmentation")

    st.caption(
        """
        Each raw image is used to generate controlled image quality variations.
        This creates a labeled dataset for image quality analysis and model
        training.
        """
    )
    
    # load all supported image files from raw image folder
    raw_images = [
        f for f in os.listdir(RAW_IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # select original image for augmentation preview
    selected_original = st.selectbox(
        "Select original image",
        options=raw_images,
        key="augmentation_preview_select"
    )

    # continue only if an original image was selected
    if selected_original:

        # extract filename without extension
        # used to build generated augmentation filenames
        base_name = os.path.splitext(selected_original)[0]

        # build full path to selected original image
        original_path = os.path.join(
            RAW_IMAGE_FOLDER,
            selected_original
        )

        # left side = processing pipeline
        # right side = generated augmentation samples
        col_info, spacer, col_grid = st.columns([0.85, 0.10, 2])

        with col_info:

            # display the processing workflow of the project
            st.write("### Processing Pipeline")

            # pipeline steps shown in dashboard
            pipeline_steps = [
                ("1", "Image Augmentation", "Generate artificial image quality degradations"),
                ("2", "Feature Extraction", "Extract blur, brightness and contrast values"),
                ("3", "Quality Checks", "Apply rule-based thresholds to assign quality issues"),
                ("4", "Dataset Split", "Create train/test datasets"),
                ("5", "Model Training", "Train RF/CNN classifiers"),
                ("6", "Model Evaluation", "Evaluate classification performance and model behavior"),
                ("7", "Error Analysis", "Inspect misclassified images and difficult degradation classes")
            ]

            # render every pipeline step as small information card
            for number, title, description in pipeline_steps:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid rgba(250,250,250,0.12);
                        border-radius: 10px;
                        padding: 10px 12px;
                        margin-bottom: 8px;
                        background-color: rgba(250,250,250,0.03);
                    ">
                        <div style="font-size: 13px; font-weight: 600;">
                            {number}. {title}
                        </div>
                        <div style="font-size: 12px; color: rgba(250,250,250,0.65); margin-top: 3px;">
                            {description}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col_grid:

            # display generated image variants for selected original image
            st.write("### Generated Augmented Samples")

            # create 2x3 grid layout for original and augmented samples
            top_col1, top_col2, top_col3 = st.columns(3)
            bottom_col1, bottom_col2, bottom_col3 = st.columns(3)

            # show original image as reference
            with top_col1:
                st.image(
                    Image.open(original_path),
                    caption="Original",
                    use_container_width=True
                )

            # show blur image variant if it exists
            with top_col2:
                blur_filename = f"{base_name}_blur.jpg"
                blur_path = os.path.join(AUGMENTED_IMAGE_FOLDER, blur_filename)

                if os.path.exists(blur_path):
                    st.image(
                        Image.open(blur_path),
                        caption="Gaussian Blur",
                        use_container_width=True
                    )
                    show_augmentation_parameters(aug_df, blur_filename)

            # show dark image variant if it exists
            with top_col3:
                dark_filename = f"{base_name}_dark.jpg"
                dark_path = os.path.join(AUGMENTED_IMAGE_FOLDER, dark_filename)

                if os.path.exists(dark_path):
                    st.image(
                        Image.open(dark_path),
                        caption="Underexposed",
                        use_container_width=True
                    )
                    show_augmentation_parameters(aug_df, dark_filename)

            # show bright image variant if it exists
            with bottom_col1:
                bright_filename = f"{base_name}_bright.jpg"
                bright_path = os.path.join(AUGMENTED_IMAGE_FOLDER, bright_filename)

                if os.path.exists(bright_path):
                    st.image(
                        Image.open(bright_path),
                        caption="Overexposed",
                        use_container_width=True
                    )
                    show_augmentation_parameters(aug_df, bright_filename)

            # show low contrast image variant if it exists
            with bottom_col2:
                contrast_filename = f"{base_name}_lowcontrast.jpg"
                contrast_path = os.path.join(
                    AUGMENTED_IMAGE_FOLDER,
                    contrast_filename
                )

                if os.path.exists(contrast_path):
                    st.image(
                        Image.open(contrast_path),
                        caption="Low Contrast",
                        use_container_width=True
                    )
                    show_augmentation_parameters(aug_df, contrast_filename)

            # show noise image variant if it exists
            with bottom_col3:
                noise_filename = f"{base_name}_noise.jpg"
                noise_path = os.path.join(AUGMENTED_IMAGE_FOLDER, noise_filename)

                if os.path.exists(noise_path):
                    st.image(
                        Image.open(noise_path),
                        caption="Gaussian Noise",
                        use_container_width=True
                    )
                    show_augmentation_parameters(aug_df, noise_filename)

            # explain how generated variants are used later
            st.caption(
            """
            Each generated variation is assigned to one of six degradation types
            (original, blur, dark, bright, lowcontrast or noise), which are later
            used for dataset analysis and RF/CNN model training.
            """
            )
    # spacer
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # B. Extracted image quality features
    # ------------------------------------------------------------

    st.subheader("Extracted Values and Quality Labels")

    st.caption(
        """
        For each generated image, blur, brightness and contrast values are extracted using OpenCV. These values are then used for rule-based quality checks.
        """
    )

    # left side = stored feature overview
    # right side = rule-based label explanation
    feature_col1, spacer, feature_col2 = st.columns([1.45,0.05, 0.75])

    with feature_col1:

        # display stored values for every generated image
        st.write("### What gets stored for each image")

        # feature documentation table shown in dashboard
        feature_explanation = {
            "Value": [
                "blur_score",
                "brightness",
                "contrast",
                "width / height",
                "quality_issues",
                "quality_status",
                "degradation_type"
            ],
            "Category": [
                "Extracted",
                "Extracted",
                "Extracted",
                "Metadata",
                "Rule-based",
                "Rule-based",
                "ML label"
            ],
            "Description": [
                "Sharpness estimate based on the variance of the Laplacian filter",
                "Average brightness of the image",
                "Contrast estimate based on the standard deviation of pixel intensities",
                "Image resolution stored for reference",
                "Detected quality issues based on rule-based thresholds",
                "Final rule-based usability result",
                "Known augmentation class used as target label for model training"
            ]
        }

        # render feature explanation table
        st.dataframe(
            feature_explanation,
            use_container_width=True,
            hide_index=True
        )

    with feature_col2:

        # display rule-based quality classification logic
        st.write("### Rule-Based Quality Labels")

        # create bordered container for threshold overview
        with st.container(border=True):

            # left = threshold condition
            # middle = arrow mapping
            # right = assigned quality label
            left_col, arrows, right_col = st.columns([0.6,0.15, 0.45])

            with left_col:

                # display applied threshold checks
                st.caption("Quality check")
                st.markdown("""
                `blur_score < 100`  
                `brightness < 60`  
                `brightness > 200`  
                `contrast < 35`  

                `no detected issues`  
                `one or more issues`
                """)
            
            with arrows:

                # visual connection between thresholds and labels
                st.caption("---→")
                st.markdown("""
                ---→   
                ---→      
                ---→      
                ---→     

                ---→      
                ---→    
                """)

            with right_col:

                # display assigned rule-based labels
                st.caption("Assigned labels")
                st.markdown("""
                blurry  
                underexposed  
                overexposed  
                low_contrast  

                usable  
                not_usable
                """)
            
        # explain difference between quality labels and ML labels
        st.caption(
        """
        The usable / not usable result is only used for the rule-based quality overview.
        RF and CNN training is performed separately using the degradation type labels.
        """
        )

    # ------------------------------------------------------------
    # C. Dataset overview
    # ------------------------------------------------------------

    st.subheader("Dataset Overview")

    # count original raw input images
    original_count = len(raw_images)

    # count generated augmentation types without original class
    augmentation_types = (
        filtered_df.loc[
            filtered_df["degradation_type"] != "original",
            "degradation_type"
        ]
        .dropna()
        .nunique()
    )

    # count images classified as usable by rule-based quality logic
    usable_count = len(
        filtered_df[filtered_df["quality_status"] == "usable"]
    )

    # count images classified as not usable by rule-based quality logic
    not_usable_count = len(
        filtered_df[filtered_df["quality_status"] == "not_usable"]
    )

    # calculate percentage of currently usable images
    usable_ratio = (
        usable_count / len(filtered_df) * 100
        if len(filtered_df) > 0 else 0
    )

    # display main dataset metrics
    with st.container(border=True):

        # create seven-column KPI overview
        overview_col1, overview_col2, overview_col3, \
        overview_col4, overview_col5, overview_col6, \
        overview_col7 = st.columns(7)

        # number of original source images
        with overview_col1.container():
            st.metric(
                "Originals",
                original_count
            )

        # number of generated augmentation classes
        with overview_col2.container():
            st.metric(
                "Augmented Variants",
                augmentation_types
            )

        # number of degradation classes
        with overview_col3.container():
            st.metric(
                "Classes",
                filtered_df["degradation_type"].nunique()
            )

        # number of images after current filter selection
        with overview_col4.container():
            st.metric(
                "Total Images",
                len(filtered_df)
            )

        # number of usable images
        with overview_col5.container():
            st.metric(
                "Usable",
                usable_count
            )

        # number of not usable images
        with overview_col6.container():
            st.metric(
                "Not Usable",
                not_usable_count
            )

        # percentage of usable images
        with overview_col7.container():
            st.metric(
                "Usability Ratio",
                f"{usable_ratio:.1f}%"
            )

    # total number of images before filtering
    total_count = len(df)


    st.caption(
        f"""
        Each raw image is augmented with every degradation type using randomized parameters. 
        This creates a balanced training dataset with {original_count} images per class and {total_count} images in total.
        The usability ratio acts as a simple sanity check for the rule-based
        quality logic and reflects how many generated images pass the current
        quality thresholds. It is only used for dataset inspection in the
        dashboard and is not used for model training.
        """
    )


    st.write("### Average Quality Values by Degradation Type")

    # calculate mean blur, brightness and contrast values
    # for every degradation class
    feature_summary = (
        filtered_df
        .groupby("degradation_type")[
            ["blur_score", "brightness", "contrast"]
        ]
        .mean()
        .round(2)
    )

    # display aggregated feature statistics
    st.dataframe(
        feature_summary,
        use_container_width=True
    )

    st.caption(
        """
        This table is used as a quick sanity check for the generated image
        degradations. Each degradation type should affect at least one of the
        extracted values in a visible way, for example blur_score for blurred
        images, brightness for dark or bright images, and contrast for
        low-contrast images.

        These measurable differences create distinct characteristics between the
        generated classes, which can later be used for RF/CNN model training.       
        """
    )

    # ------------------------------------------------------------
    # CSV download
    # ------------------------------------------------------------

    # convert filtered dataframe into csv format
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    # create download button for filtered dataset report
    st.download_button(
        label="Download filtered report as CSV",
        data=csv_data,
        file_name="filtered_image_quality_report.csv",
        mime="text/csv"
    )

    # ------------------------------------------------------------
    # D. Image inspection
    # ------------------------------------------------------------

    st.subheader("Image Inspection")

    st.caption(
        """
        Inspect generated images together with their extracted values and labels.
        """
    )

    # select generated image from filtered dataset
    selected_file = st.selectbox(
        "Select image",
        options=filtered_df["filename"].tolist(),
        key="dataset_image_preview_select"
    )

    # continue only if generated image was selected
    if selected_file:

        # build full image path for selected file
        image_path = os.path.join(
            IMAGE_FOLDER,
            selected_file
        )

        # select dataframe row containing all values for selected image
        selected_row = filtered_df[
            filtered_df["filename"] == selected_file
        ].iloc[0]

        # left side = image preview
        # right side = extracted image information
        col_img, col_info = st.columns([2.2, 0.7])

        with col_img:

            # load and display selected image
            image = Image.open(image_path)

            st.image(
                image,
                caption=selected_file,
                use_container_width=True
            )

        with col_info:

            # display extracted image details inside bordered container
            with st.container(border=True):

                st.write("### Image Details")

                # display assigned degradation class
                st.write(
                    f"**Degradation type:** "
                    f"{selected_row['degradation_type']}"
                )

                # display rule-based quality status
                st.write(
                    f"**Quality status:** "
                    f"{selected_row['quality_status']}"
                )

                # display detected rule-based quality issues
                st.write(
                    f"**Quality labels:** "
                    f"{selected_row['quality_issues']}"
                )
                
                # small visual spacing
                st.markdown(" ")

                # display blur score
                st.write(
                    f"**Blur score:** "
                    f"{selected_row['blur_score']}"
                )

                # display brightness value
                st.write(
                    f"**Brightness:** "
                    f"{selected_row['brightness']}"
                )

                # display contrast value
                st.write(
                    f"**Contrast:** "
                    f"{selected_row['contrast']}"
                )

                # display image resolution
                st.write(
                    f"**Resolution:** "
                    f"{selected_row['width']} × "
                    f"{selected_row['height']}"
                )