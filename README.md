# Vision Data Quality Analyzer

Automated image quality analysis and degradation classification
for computer vision datasets using OpenCV, Random Forest
and CNN-based approaches.

---

## Project Overview

This project analyzes and classifies image quality issues
in computer vision datasets using classical machine learning
and deep learning methods.

It focuses on common problems in drone imagery such as
blur, low contrast, brightness changes and image noise.

To simulate realistic image quality problems,
artificial image degradations are generated under
controlled conditions for image analysis, feature
extraction and model evaluation.

The resulting image data and extracted features are
then used to compare different approaches for automated
image quality classification.

A Streamlit dashboard is used to visualize and explore
the generated datasets, extracted quality metrics,
model predictions and evaluation results interactively.

---

## Project Purpose

The project investigates how image quality degradations
can be analyzed and classified automatically before
images are used for further computer vision tasks.

It focuses on questions such as:

- How can image quality degradations be analyzed automatically?
- Which image quality features are most useful for classification?
- Can different degradation types be classified reliably?
- How do classical ML and CNN-based approaches compare on this task?

---

## Dataset

The project uses images from the
[VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset),
a large-scale benchmark dataset for drone-based computer vision research.

The dataset was introduced by:

> Zhu, P. et al.  
> *Vision Meets Drones: A Challenge*  
> ICCV / ECCV Workshop Series, 2018–2020

ArXiv publication:

https://arxiv.org/abs/2001.06303

VisDrone highlights several important challenges in aerial
imagery such as motion blur, scale variation, occlusions,
illumination changes and small object detection.

These characteristics make the dataset well suited for
image quality analysis and degradation classification tasks.

The original VisDrone dataset is not part of this repository.
A small demonstration subset is included for dashboard visualization
and interactive project presentation.

Please refer to the official VisDrone repository for the
complete dataset and additional information.

---

## Processing Pipeline

1. **Image Augmentation**  
   Generate artificial image degradations

2. **Feature Extraction**  
   Extract blur, brightness and contrast features

3. **Quality Checks**  
   Apply rule-based thresholds to assign quality labels

4. **Dataset Split**  
   Create train and test datasets

5. **Model Training**  
   Train Random Forest and CNN classifiers

6. **Model Evaluation**  
   Evaluate classification performance and model behavior

7. **Error Analysis**  
   Inspect misclassified images and difficult degradation classes

---

## Generated Image Degradations

The following degradation types are generated automatically
for every original image:

- Gaussian Blur
- Underexposure
- Overexposure
- Low Contrast
- Gaussian Noise

Each generated image is assigned to one of six classes:

- original
- blur
- dark
- bright
- lowcontrast
- noise

The generated degradations are used for:

- dataset analysis
- OpenCV feature extraction
- Random Forest training
- CNN training
- model evaluation
- error analysis

---

## Feature Extraction

Image quality features are extracted using OpenCV.

The extracted features currently include:

- blur score (Laplacian variance)
- brightness
- contrast
- image resolution

Rule-based thresholds are additionally used to assign:

- blurry
- underexposed
- overexposed
- low_contrast
- usable
- not_usable

The extracted feature reports are stored as CSV files
for later model training and dashboard visualization.

---

## Dataset Split

The generated dataset is divided into fixed
train, validation and test splits before
model training.

The split is created once and reused for both
the Random Forest and CNN classifiers to ensure
consistent training conditions and comparable
evaluation results.

Images derived from the same original source image
are kept within the same split to avoid data leakage
between training and evaluation datasets.

---

## Random Forest Classifier

The Random Forest model uses extracted OpenCV features
as input for degradation classification.

Input features:

- blur_score
- brightness
- contrast

The model predicts the degradation type of each image.

Additional evaluation outputs include:

- classification accuracy
- feature importance analysis
- confusion matrix visualization
- misclassified image inspection

Custom Graphviz visualizations are used to illustrate
the Random Forest classification process and decision trees.

---

## CNN Classifier

The CNN model predicts image degradation types directly
from image data.

The architecture consists of two main parts:

### Convolutional Feature Extraction

- convolution layers
- batch normalization
- max pooling

### Classification Layers

- global average pooling
- dense classification layer
- dropout regularization

Training history, evaluation metrics and prediction
reports are exported automatically for dashboard
visualization and analysis.

The CNN architecture was designed as an improved version
after earlier baseline experiments did not outperform the
feature-based Random Forest model.

---

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard
for dataset inspection and model evaluation.

Dashboard sections include:

- project overview
- dataset exploration
- augmentation preview
- OpenCV feature analysis
- Random Forest evaluation
- CNN evaluation
- prediction visualization
- error analysis
- model comparison and result interpretation [Work in Progress]

The dashboard also includes a lightweight demo dataset
for easier deployment and presentation.

---

## Methods & Tools

- **Computer Vision:** OpenCV
- **Machine Learning:** scikit-learn
- **Deep Learning:** TensorFlow / Keras
- **Data Analysis:** pandas
- **Visualization:** matplotlib, Graphviz, PlotNeuralNetworks
- **Dashboard Frontend:** Streamlit, HTML/CSS

---

## Project Structure

```text
vision-data-quality-analyzer/
│
├── analysis/          # evaluation and error analysis scripts
├── app_with_tabs/     # Streamlit dashboard and dashboard tabs
├── data/              # datasets and demo subsets
├── graphics_code/     # Graphviz and model visualizations
├── models/            # trained RF and CNN models
├── results/           # reports, plots and exported outputs
├── src/               # core data processing and training pipeline
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Run the Full Pipeline

```bash
python src/run_pipeline.py
```

The pipeline automatically runs:

- image augmentation
- feature extraction
- dataset split generation
- Random Forest training
- CNN training
- evaluation and error analysis
- Streamlit demo subset creation
- Streamlit dashboard launch

The full pipeline workflow is defined in
`src/run_pipeline.py`.

---

## Start the Dashboard

```bash
streamlit run app_with_tabs/dashboard_demo_version.py
```

---

## Demo Version

The published dashboard version uses a smaller subset
of 300 images from the VisDrone dataset together with
generated image degradations to provide a compact
demonstration of the project.

The included images are intended to demonstrate image
quality analysis, dataset inspection and model
predictions within the dashboard.

The Random Forest and CNN models were trained
and evaluated on the complete augmented dataset
generated from the selected VisDrone image subset.
