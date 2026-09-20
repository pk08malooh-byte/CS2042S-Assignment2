# CSC2042S Assignment 2 — Multinomial Logistic Regression

## Overview

This repository contains my **CSC2042S Assignment 2** implementation of multinomial logistic regression for **MasakhaNEWS** text classification.

The assignment investigates how a multinomial logistic regression model performs when classifying news articles across three languages:

* English (eng)
* isiXhosa (xho)
* chiShona (sna)

The experiments use two feature representations:

* **Bag-of-Words (CountVectorizer)**
* **TF-IDF (TfidfVectorizer)**

### Assignment Notebook

The complete assignment can be viewed here:

[**Logistic Regression Assignment 2 Notebook**](./Logistic_Regression_Assignment_2.ipynb)


## Main Tasks

### 1. Data Preparation

The MasakhaNEWS data is loaded for English, isiXhosa and chiShona.

The headline and article text are combined to create the input text used for classification.

Basic text preprocessing is applied before feature extraction.

### 2. Feature Extraction

Two feature representations are investigated:

* **Bag-of-Words using CountVectorizer**
* **TF-IDF using TfidfVectorizer**

The vectorizers are fitted using the training data and then applied to the development and test sets.

### 3. Multinomial Logistic Regression

A MultinomialLogisticRegression class is implemented using PyTorch.

The implementation includes:

* Forward propagation
* Probability computation
* Prediction
* Cross-entropy loss
* Mini-batch training
* Validation evaluation
* Early stopping

### 4. Baseline Experiments

Baseline models are trained and evaluated for:

* English
* isiXhosa
* chiShona

Both CountVectorizer and TF-IDF representations are compared.

### 5. Hyperparameter Tuning

The notebook investigates different combinations of:

* Learning rate
* Batch size
* Vocabulary size
* Feature representation

Validation performance is used to select configurations for further experiments.

### 6. Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Micro-averaged metrics
* Macro-averaged metrics
* Confusion matrices

### 7. Weight Analysis

The learned weights of the best English model are examined to identify words that contribute strongly to predictions for different news categories.

### 8. isiXhosa Class Imbalance

The isiXhosa training data is investigated for class imbalance.

Two sampling approaches are compared:

* Upsampling using `WeightedRandomSampler`
* Downsampling majority classes

These results are compared with the original training distribution.

### 9. Bilingual Training

The notebook investigates whether combining languages during training can improve isiXhosa classification.

Two bilingual combinations are evaluated:

* **isiXhosa + chiShona**
* **isiXhosa + English**

The bilingual models are compared with the monolingual isiXhosa baseline.

---

## Reproducibility

Random seeds are explicitly controlled during the experiments to make the results reproducible.

The notebook defines a reproducibility function that sets seeds for:

* Python's `random` module
* NumPy
* PyTorch

To reproduce the experiments:

1. Clone or download this repository.
2. Open `Logistic_Regression_Assignment_2.ipynb`.
3. Ensure the required Python packages are installed.
4. Ensure the dataset is available in the expected directory.
5. Run the notebook cells from top to bottom.

The notebook is intended to run in its existing order without requiring manual modification of the implementation.

---

## Requirements

The assignment uses Python and the following main libraries:

* Python
* NumPy
* pandas
* scikit-learn
* PyTorch
* Matplotlib

---

## Results and Analysis

The notebook contains the results of the:

* Baseline experiments
* Hyperparameter tuning
* Weight analysis
* isiXhosa class-imbalance experiments
* Bilingual training experiments

Results are presented using tables, evaluation metrics, confusion matrices and model comparisons.

For the complete implementation and numerical results, see the assignment notebook:

[**Open the Assignment Notebook**](./Logistic_Regression_Assignment_2.ipynb)



## Author
Phumla Khumalo

