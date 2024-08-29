# Music Track Classifier - Machine Learning Final Project

This repository contains the final project for a Machine Learning course, which focuses on classifying music tracks based on various audio features. The project involves several steps, including data preprocessing, feature extraction, model training, and evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Project Files](#project-files)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Clustering Analysis](#clustering-analysis)


## Overview

The goal of this project is to classify music tracks based on various audio features. The process includes extracting features from audio files, training several machine learning models, and evaluating their performance. Additionally, clustering techniques are applied to analyze patterns within the data.

## Project Files

- **`ml_final_project.ipynb`**: A Jupyter notebook containing the code for data preprocessing, feature extraction, model training, and evaluation.
- **`ml_final_project.html`**: An HTML export of the Jupyter notebook for easier viewing.
- **`ML-FinalProject-Report-810197648-810197535.pdf`**: The final project report in PDF format, detailing the methods, results, and conclusions of the project.

## Data Preprocessing

**Objective**: The initial step involves loading the audio data, handling corrupted files, and preparing the data for feature extraction.

**Key Steps**:
- **Data Loading**: The dataset is loaded, and corrupted files are identified and removed.
- **Visualization**: Basic visualizations, such as waveform, spectrogram, and spectrum, are created using libraries like `librosa` and `matplotlib` to understand the data distribution.
- **Normalization**: Data is normalized to ensure consistency across different audio tracks.
- **Label Encoding**: The target labels are encoded using `LabelEncoder` from `sklearn` to convert them into a format suitable for model training.

## Feature Extraction

**Objective**: Extract relevant features from the audio files that will be used for training machine learning models.

**Key Features**:
- **Zero Crossing Rate**: The rate at which the signal changes from positive to negative or back, which is a simple feature to detect the noisiness of the signal.
- **Spectral Centroid**: Indicates where the center of mass of the spectrum is located and is a measure of the brightness of a sound.
- **Spectral Roll-off**: The frequency below which a specified percentage of the total spectral energy lies.
- **Mel-Frequency Cepstral Coefficients (MFCCs)**: A representation of the short-term power spectrum of a sound, often used in speech and audio processing.
- **Chroma Frequencies**: Represents the energy distribution across the 12 different pitch classes, useful in identifying harmonic and melodic features.

## Model Training and Evaluation

**Objective**: Train various machine learning models on the extracted features and evaluate their performance in classifying music tracks.

**Models Used**:
- **K-Nearest Neighbors (KNN)**: A non-parametric method used for classification by finding the most similar instances in the feature space.
- **Support Vector Machines (SVM)**: A supervised learning model that finds the hyperplane that best separates the data into classes.
- **Polynomial Logistic Regression**: A regression model that extends logistic regression to model more complex relationships between features and the target variable.

**Evaluation**:
- **Grid Search**: Used to find the optimal parameters for each model.
- **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score are calculated to evaluate the models. Confusion matrices are also generated to visualize the classification performance.

## Clustering Analysis

**Objective**: Perform clustering on the dataset to identify patterns and group similar tracks together.

**Techniques Used**:
- **K-Means Clustering**: A common clustering method that partitions the data into K clusters by minimizing the variance within each cluster.
- **K-Medoids**: Similar to K-Means but uses medoids to define clusters, which are more robust to outliers.
- **Hierarchical Clustering**: Builds a tree of clusters by iteratively merging or splitting existing clusters.

**Evaluation**:
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters.
- **Purity Score**: Evaluates the extent to which clusters contain a single class of data points.
