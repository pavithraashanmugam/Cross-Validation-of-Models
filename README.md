# README: Heart Disease Prediction - Model Comparison

## Overview

This project focuses on building and evaluating different machine learning models for heart disease prediction using the **Heart Disease Dataset**. The dataset is used to predict the likelihood of a person having heart disease based on various health attributes such as age, cholesterol level, blood pressure, etc. 

The models implemented and compared in this project include:
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

### Goal

The primary goal of this project is to compare two different techniques for evaluating model performance:
1. **Train-Test Split**: This method splits the dataset into a training set and a testing set and evaluates the models on the test data.
2. **Cross-Validation**: This technique splits the dataset into multiple parts and evaluates the model multiple times using different training and test sets. It provides a more robust and reliable performance evaluation.

In this README, you’ll find the details of how the models were trained, evaluated, and compared using both techniques.

---

## Data Description

The dataset used is the **Heart Disease Dataset**. It contains health-related attributes of individuals and a target variable that indicates whether the individual has heart disease (`1`) or not (`0`).

### Columns in the dataset:
- **age**: Age of the individual
- **sex**: Gender of the individual (1 = male, 0 = female)
- **cp**: Chest pain type (4 types)
- **trestbps**: Resting blood pressure (in mmHg)
- **chol**: Serum cholesterol level (in mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: Depression induced by exercise relative to rest
- **slope**: Slope of peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy
- **thal**: Thalassemia (3 types)
- **target**: Heart disease (1 = defective heart, 0 = healthy heart)

---

## Dependencies

The following Python libraries are used in this project:

- `numpy`: For numerical computations.
- `pandas`: For data manipulation and analysis.
- `sklearn`: For building and evaluating machine learning models.
    - `train_test_split`: To split the dataset into training and testing sets.
    - `cross_val_score`: For cross-validation of models.
    - `accuracy_score`: To calculate model accuracy.
    - `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `RandomForestClassifier`: For machine learning models.
  
---

## Steps and Implementation

### 1. Data Collection and Processing

- The heart disease dataset is read using `pandas.read_csv()` from the provided CSV file path.
- Missing values are checked, and it is confirmed that there are no missing values.
- The target variable `target` is separated from the feature set, and the features are stored in `X`, while the target variable is stored in `Y`.

### 2. Splitting the Data

- The dataset is split into training and testing sets using the `train_test_split()` function. The data is split in an 80-20 ratio (80% training, 20% testing).
- Stratification is applied to ensure that the proportions of the target variable (defective heart vs healthy heart) are preserved in both the training and test sets.

### 3. Model Evaluation Using Train-Test Split

- Four machine learning models are trained using the training data: 
    - Logistic Regression
    - Support Vector Classifier (SVC)
    - K-Nearest Neighbors (KNN)
    - Random Forest Classifier

- Each model is evaluated by making predictions on the test set and calculating the accuracy score.

### 4. Model Evaluation Using Cross-Validation

- Cross-validation is applied using the `cross_val_score()` function, which divides the dataset into 5 folds (default) and computes the accuracy for each fold.
- The mean accuracy across the 5 folds is computed for each model.

### 5. Model Comparison

- The performance of each model is compared based on the accuracy scores obtained from both the train-test split method and cross-validation.
- The results show that cross-validation gives more reliable and consistent results, yielding higher accuracy than the train-test split.

---

## Code Structure

The main code is structured as follows:

1. **Importing Dependencies**: All necessary libraries are imported.
2. **Data Collection & Processing**: The dataset is loaded and preprocessed.
3. **Model Evaluation Using Train-Test Split**: The models are trained and tested on the split dataset, and the results are printed.
4. **Model Evaluation Using Cross-Validation**: Each model is evaluated using cross-validation, and the mean accuracy is printed.
5. **Comparison Function**: A function `compare_models_cross_val_score()` is created to compare all models using cross-validation.

### Key Functions

- `compare_models_train_test_split()`: Compares the performance of the models using train-test split and prints their accuracy scores.
- `compare_models_cross_val_score()`: Compares the performance of the models using cross-validation and prints their mean accuracy scores.

---

## Results

### Train-Test Split Evaluation:

The accuracy scores obtained using the train-test split method are as follows:

- **Logistic Regression**: 77.05%
- **SVC**: 77.05%
- **KNeighbors**: 65.57%
- **Random Forest Classifier**: 80.33%

### Cross-Validation Evaluation:

The mean accuracy scores obtained using cross-validation are:

- **Logistic Regression**: 82.83%
- **SVC**: 82.83%
- **KNeighbors**: 64.39%
- **Random Forest Classifier**: 83.16%

As observed, **Random Forest Classifier** yields the highest accuracy among all models, while **cross-validation** provides more stable and higher accuracy than **train-test split**.

---

## Conclusion

- Cross-validation provides a more reliable estimate of model performance compared to a simple train-test split.
- Random Forest Classifier and SVC (both with linear kernel) perform the best in this case, with Random Forest slightly outperforming the others.
- K-Nearest Neighbors performed the worst in terms of accuracy.

---

## Usage

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   ```
   
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Run the Python script:
   ```bash
   python heart_disease_prediction.py
   ```

---
