# Exploring Password Complexity: A Comparative Analysis of Supervised and Unsupervised Approaches for Password Strength Classification

This project presents a comprehensive analysis of password complexity using both supervised and unsupervised machine learning techniques. By leveraging advanced algorithms, this project aims to enhance password strength classification and contribute to the development of more robust password policies for enhanced user security.

This is a senior thesis completed in partial fulfilment for the requirements of the degree of Bachelor of Science in Statistics and 
Data Science at Yale University.

## Table of Contents
- [Abstract](#abstract)
- [Summary](#summary)
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Supervised Models](#supervised-models)
- [Unsupervised Model](#unsupervised-model)
- [Password Analysis](#password-analysis)
- [Usage](#usage)
- [Results and Insights](#results-and-insights)
- [Future Directions](#future-directions)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Abstract
Within the first four months of 2023, 17 major companies reported data breaches. Although many companies store password encryptions instead of raw passwords, these breaches still pose a threat to account holders. Attackers can access personal information if they are able to crack the password. Companies use algorithms to determine password strength and derive their password policy accordingly. In 2012, Dropbox released zxcvbn, their own algorithm for classifying passwords [2]. In this project, we examined password complexity as defined by zxcvbn through the use of supervised and unsupervised methods. 700,000 passwords were randomly sampled from the 2015 000webhost data breach. Their strengths were classified by the zxcvbn algorithm, and we extract several features to capture the underlying pattern of the data, including n-grams, Shannon entropy, Levenshtein distance, and character repetition weight sum. For the supervised methods, we used a Multilayer Perceptron (MLP) and a Convolution Neural Network (CNN). For the unsupervised methods, we used k-means clustering at different clustering levels. By comparing the performance of these models, we aimed to uncover patterns in password strength and identify the most influential features in predicting password complexity. This analysis offers valuable insights into password complexity and its relationship with the zxcvbn algorithm, contributing to the development of more robust password policies and enhanced user security.

## Summary
This project delves into the realm of password complexity analysis using both supervised and unsupervised machine learning techniques. By dissecting a sample of 700,000 passwords from the 2015 000webhost data breach, we leverage the zxcvbn algorithm for password strength classification. Our exploration encompasses n-grams, Shannon entropy, Levenshtein distance, and character repetition weight sum as key features.

The project involves:
- Applying supervised techniques, including a Multilayer Perceptron (MLP) and a Convolution Neural Network (CNN), to classify password strengths.
- Employing K-means clustering for an unsupervised exploration of underlying password patterns.
- Identifying significant features in predicting password complexity.

Supervised models showcase promising performance, achieving an accuracy of 70%. Both MLP and CNN models highlight Levenshtein distance and Shannon entropy as influential features. Unsupervised K-means clustering offers insights into the password feature space's structure, revealing nuances in password strength patterns.

## Project Overview
- **Supervised and Unsupervised Methods:** We explore password complexity using both supervised (MLP and CNN) and unsupervised (K-means) machine learning methods.
- **Feature Extraction:** Extracted features include n-grams, Shannon entropy, Levenshtein distance, and character repetition weight sum.
- **Password Strength Classification:** The zxcvbn algorithm is employed for classifying password strengths.
- **Performance Evaluation:** Models are evaluated based on accuracy and feature importance.

## Methodology
1. **Data Collection:** A dataset of 700,000 passwords from the 2015 000webhost breach is used.
2. **Feature Extraction:** Features such as n-grams, Shannon entropy, Levenshtein distance, and character repetition weight sum are extracted.
3. **Supervised Learning:** Multilayer Perceptron (MLP) and Convolution Neural Network (CNN) models are trained for password strength classification.
4. **Unsupervised Learning:** K-means clustering explores underlying patterns in password strength.
5. **Performance Analysis:** Models are evaluated, and feature importance is assessed.

## Data Preprocessing
The `data-preprocessing` directory contains scripts to preprocess the raw password data before model input:

- `sample_passwords.py`: Selects a subset of passwords from the dataset.
- `classify_passwords.py`: Classifies passwords using Dropbox's zxcvbn algorithm.
- `feature_extraction.py`: Extracts features from passwords for model input.

To preprocess the data, follow these examples:

- Subset a dataset:
`./Preprocess subset [file_name] [sample_size]`

- Classify password strengths:
`./Preprocess classify [file_name]`

- Extract features:
`./Preprocess extract [file_name]`

## Supervised Models
The supervised models, a Multilayer Perceptron (MLP) and a Convolution Neural Network (CNN), are implemented in Jupyter notebooks for password strength classification. Check out the notebooks for detailed code and analysis:

- `models/CNN.ipynb`: Notebook containing the CNN model and its analysis.
- `models/MLP.ipynb`: Notebook containing the MLP model and its analysis.

## Unsupervised Model
The unsupervised k-means clustering model is explored in the `models/K-Means Clustering.ipynb` notebook. This technique helps identify patterns and relationships within the password dataset.

## Password Analysis
The `other/password-analysis.ipynb` notebook performs an insightful analysis of the password dataset, examining various aspects such as classification, distribution, and more.

## Usage
To replicate or build upon this project, follow these steps:

1. Clone the repository: `git clone https://github.com/briannaschuh/s-ds492.git`
2. Install required dependencies: `pip install -r requirements.txt`
3. Preprocess the data using the scripts in `data-preprocessing`.
4. Explore and analyze the supervised and unsupervised models in the `models` directory.
5. Gain insights from the password analysis in `other/password-analysis.ipynb`.

## Results and Insights
This analysis provides valuable insights into password complexity, classification, and relationships. The supervised models achieved an accuracy of 70%, with Levenshtein distance and Shannon entropy emerging as influential features. K-means clustering reveals underlying patterns within the data, revealing non-trivial relationships.

## Future Directions
Future research could focus on understanding the misclassification of strong passwords by supervised models, exploring the use of CNNs for password evaluation, and addressing potential security concerns related to password data analysis.

## License


