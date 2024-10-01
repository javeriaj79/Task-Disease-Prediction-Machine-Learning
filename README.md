# Disease Prediction Using Machine Learning

In this project, machine learning algorithms are implemented on a comprehensive Disease Prediction Dataset to accurately classify health conditions based on features extracted from patient medical records. This initiative applies various algorithms: Support Vector Classifier (SVC), Gaussian Naive Bayes, and Random Forest—to improve diagnostic accuracy.

## Table of Contents
1. [Project Overview](#task-disease-prediction-machine-learning)
2. [Algorithm Used](#algorithm-used)
   [Support Vector Classifier (SVC)](#1-support-vector-classifier-svc)
   [Naive Bayes](#2-naive-bayes)
   [Random Forest](#3-random-forest)
3. [Usage](#usage)
4. [Prerequisites](#prerequisites)
5. [Results](#results)
6. [Feedback](#feedback)
7. [Contributing](#contributing)

## Project Overview
Millions worldwide face health issues requiring timely intervention, with diagnosed cases projected to rise. Integrating Artificial Intelligence (AI) into medical practice is crucial for disease prediction and management. This project utilizes machine learning algorithms on a comprehensive Disease Prediction Dataset to improve patient outcomes.

## Algorithm Used

### 1. Support Vector Classifier (SVC):
It works by finding the optimal hyperplane that separates classes in a dataset. It is effective in high-dimensional spaces and is used here to classify disease outcomes, applicable in both linear and non-linear problems.

### 2. Naive Bayes:
It is a probabilistic classifier based on applying Bayes' theorem with the assumption of independence between features. This algorithm is particularly fast, works well with large datasets, and is well-suited for categorical data.

### 3. Random Forest:
It is an ensemble method that creates multiple decision trees and merges their results to improve accuracy and reduce overfitting. It handles both classification and regression tasks well, and provides insights into feature importance.

## Usage
To use this project:

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/disease-prediction.git
    cd disease-prediction
    ```
2. Install the necessary dependencies using:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the main script to train the models and make predictions:
    ```bash
    python disease_prediction.py
    ```

## Prerequisites:
- Python 3.x
- Libraries: scikit-learn, pandas, numpy, matplotlib (optional for plotting)

## Results

The performance of each algorithm is summarized below:

| Algorithm               | Training Accuracy | Test Accuracy   |
|-------------------------|-------------------|-----------------|
| **SVC**                 | 89.65%            | 92.86%          |
| **Naive Bayes**          | 100.00%           | 100.00%         |
| **Random Forest**        | 100.00%           | 100.00%         |

## Feedback
If you have any feedback, contact me at [your.email@example.com].

## Contributing
Your contributions are appreciated! Please fork the repository and open a pull request to propose your enhancements.
