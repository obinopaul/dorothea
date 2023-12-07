# Machine Learning Analysis of Dorothea Drug Discovery Dataset 
==============================================================

<p align="center">
  <img src="image.png" alt="Alt text" width="500" height="300">
</p>

This repository contains a machine learning project focused on analyzing the Dorothea dataset, which is instrumental in the field of drug discovery. The goal is to classify chemical compounds as active or inactive based on their structural attributes using advanced machine learning techniques. This dataset is a benchmark in the NIPS 2003 feature selection challenge, crucial for advancements in drug discovery.

**Author:** PAUL OKAFOR  
**Institution:** University of Oklahoma  
**Date:** December 6, 2023

## Requirements

To install the necessary libraries to run the project, use the following command:

```bash
pip install -r requirements.txt
```

## Model Building Strategy

The model building strategy encompassed a three-pronged approach:

1. **Initial Training**: Baseline models were trained using a straightforward 10-fold cross-validation technique to establish initial performance metrics without hyperparameter tuning.

2. **Hyperparameter Optimization**: Utilizing Optuna, an optimization framework, hyperparameters for each model were fine-tuned within the cross-validation setup to enhance accuracy and overall predictive performance.

3. **Feature Selection**: The variance threshold method was applied to identify and retain the most significant features, followed by retraining the models to evaluate the impact of feature reduction on performance.



## Results Summary

After rigorous model training and evaluation, the results demonstrated varied performances across different algorithms:

- **Logistic Regression** emerged as a balanced model, achieving an ROC AUC of 80.09%. Despite reasonable accuracy and specificity, the model had room for improvement in precision and recall for active compounds.

- **Support Vector Machines** with RBF and polynomial kernels excelled in specificity, perfectly identifying inactive compounds. However, they struggled with the identification of active compounds, as reflected by zero precision and recall, indicating potential overfitting to the majority class.

- **Decision Trees** offered a more balanced approach, with modest precision and recall, suggesting a better fit and highlighting its capability to manage the dataset's imbalance effectively.

Post hyperparameter optimization and feature selection, the models showed the following outcomes:

- **Logistic Regression** maintained accuracy, with a slight uptick in precision, signifying an improved classification ability.

- **SVM with RBF Kernel** displayed a substantial increase in precision, indicating an enhanced ability to correctly predict active compounds.

- **Decision Trees** retained stable performance, confirming its robustness in handling feature selection.

The most significant performance improvement was observed in the SVM with RBF kernel after feature selection, showcasing high precision and F-score, which underscores the effectiveness of feature selection in enhancing model performance.

In conclusion, the Support Vector Machine with RBF kernel was highlighted for its accuracy in predicting active compounds, although it still displayed a modest recall, suggesting further potential for improvement.

