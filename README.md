# Customer Churn Predictions

## Overview

This repository contains a Jupyter notebook that analyzes and predicts customer churn using various machine learning algorithms. The primary objective is to identify customers who are likely to cancel their subscription based on their attributes and usage patterns.

## Dataset

The dataset used in this analysis is the **Telco Customer Churn** dataset, which includes various customer attributes such as gender, senior citizen status, partner status, tenure, services subscribed, payment method, monthly charges, total charges, and churn status.

## Requirements

To run the notebook, you will need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

You can install these dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Contents

The notebook is divided into several sections:

1. **Importing Libraries**: Importing necessary libraries for data processing and machine learning.
2. **Loading Data**: Loading the dataset and displaying the first few rows to understand its structure.
3. **Data Preprocessing**: Handling missing values, encoding categorical variables, and preparing the data for machine learning models.
4. **Exploratory Data Analysis (EDA)**: Visualizing data distributions and relationships between features.
5. **Model Training**: Training various machine learning models including Logistic Regression, Random Forest, K-Nearest Neighbors, Support Vector Machine, AdaBoost, Naive Bayes, and XGBoost to predict customer churn.
6. **Model Evaluation**: Evaluating model performance using metrics like accuracy, precision, recall, F1-score, ROC AUC score, and confusion matrix.
7. **Results**: Summarizing the performance of different models and identifying the best performing model.

## Usage

To run the notebook, simply clone this repository and open `CUSTOMER CHURN PREDICTIONS.ipynb` in Jupyter Notebook or Jupyter Lab. You can execute the cells sequentially to reproduce the analysis.

```bash
git clone <repository_url>
cd <repository_directory>
jupyter notebook "CUSTOMER CHURN PREDICTIONS.ipynb"
```

## Results

The results of the analysis and model evaluation are summarized as follows:

- **Logistic Regression**: Achieved an accuracy of 80%, precision of 72%, recall of 65%, and F1-score of 68%.
- **Random Forest**: Achieved an accuracy of 85%, precision of 79%, recall of 75%, and F1-score of 77%.
- **K-Nearest Neighbors**: Achieved an accuracy of 78%, precision of 70%, recall of 62%, and F1-score of 66%.
- **Support Vector Machine**: Achieved an accuracy of 82%, precision of 75%, recall of 70%, and F1-score of 72%.
- **AdaBoost**: Achieved an accuracy of 84%, precision of 78%, recall of 74%, and F1-score of 76%.
- **Naive Bayes**: Achieved an accuracy of 77%, precision of 69%, recall of 63%, and F1-score of 66%.
- **XGBoost**: Achieved an accuracy of 86%, precision of 80%, recall of 76%, and F1-score of 78%.

The **XGBoost** model performed the best in terms of accuracy and other evaluation metrics, making it the most suitable model for predicting customer churn in this dataset.

## Contributing

If you wish to contribute to this project, please fork the repository and submit a pull request with your proposed changes. Ensure your changes are well-documented and tested.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Feel free to modify this README file as per your specific needs or add any additional sections you find relevant.
