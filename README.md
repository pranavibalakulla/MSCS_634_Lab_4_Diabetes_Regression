# MSCS 634 Lab 4: Regression Model Comparison

## Overview

This lab focuses on developing, evaluating, and comparing regression models using Python and the Diabetes dataset from Scikit-learn. The objective is to understand how different regression techniques perform, evaluate their prediction errors, and investigate the effects of model complexity and regularization on predictive performance.

The models evaluated include Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, Ridge Regression, and Lasso Regression.

## Objectives

- Load and prepare the Diabetes dataset for regression analysis.
- Build and evaluate different regression models.
- Compare model performance using MAE, MSE, RMSE, and R².
- Investigate the effects of polynomial degree on model performance.
- Examine how Ridge and Lasso regularization affect prediction accuracy and overfitting.
- Visualize model performance and interpret the results.

## Dataset

The project uses the Diabetes dataset available through `sklearn.datasets`. The dataset contains predictor variables related to diabetes measurements and a quantitative target variable representing disease progression.

The dataset is divided into training and testing sets to evaluate how well the models generalize to unseen observations.

## Technologies and Libraries

- Python
- Google Colab or Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Models Implemented

### 1. Simple Linear Regression

A basic regression model using Body Mass Index (BMI) as the predictor variable to estimate the target.

### 2. Multiple Linear Regression

A regression model using multiple available predictor variables to estimate the target.

### 3. Polynomial Regression

Polynomial regression models are evaluated using degrees 1, 2, 3, 5, and 10 to examine whether increasing model complexity improves prediction accuracy.

### 4. Ridge Regression

Ridge Regression applies L2 regularization to penalize large coefficients. Different alpha values are tested to evaluate their effects on model performance.

### 5. Lasso Regression

Lasso Regression applies L1 regularization, which can shrink coefficients and set some coefficients to zero. Multiple alpha values are evaluated to identify a suitable model configuration.

## Evaluation Metrics

The models are compared using the following metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures the square root of the average squared prediction error.
- **R² Score:** Measures how much variation in the target is explained by the model.

Lower MAE, MSE, and RMSE values indicate better predictive performance, while a higher R² generally indicates a better fit.

## Results and Findings

Based on the experimental results, Lasso Regression with alpha = 1 achieved the lowest test RMSE of **53.1467** and the highest test R² of **0.4669** among the reported models. Multiple Linear Regression achieved the lowest test MAE of **42.7941**, while Ridge Regression with alpha = 10 achieved a test RMSE of **53.6263** and an R² of **0.4572**.

Polynomial Regression performed less effectively than the linear and regularized models. The degree-10 polynomial model recorded the highest test RMSE of **65.6927** and an R² of **0.1855**. The increase in testing error at higher polynomial complexity suggests that the model may be overfitting the training data.

Overall, Lasso Regression with alpha = 1 was the strongest model based on RMSE and R², while Multiple Linear Regression performed best according to MAE. These results demonstrate the importance of comparing multiple evaluation metrics rather than relying on a single measure.

## Visualizations

The notebook includes visualizations to support the analysis:

- Training versus testing RMSE across polynomial degrees.
- Test RMSE comparison between Ridge and Lasso Regression across alpha values.
- Test R² comparison between Ridge and Lasso Regression across alpha values.
- Actual versus predicted target values for the selected Ridge and Lasso models.
- Comparative charts showing model error metrics and R² scores.

These visualizations help illustrate differences in prediction accuracy, model complexity, and generalization performance.

## How to Run the Notebook

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Ensure the required Python libraries are available.
3. Run the notebook cells in order, starting with dataset loading and preparation.
4. Execute the model training and evaluation sections.
5. Review the generated tables, graphs, and written analysis.
6. Compare the results with the findings described in this README.

## Conclusion

This lab demonstrates how regression techniques can be applied to a real-world health-related dataset. The findings show that regularized regression models can outperform the tested polynomial models and that increasing model complexity does not necessarily improve predictions. Lasso Regression with alpha = 1 achieved the best overall results based on test RMSE and R², while Multiple Linear Regression achieved the lowest MAE. Further work could investigate cross-validation, additional hyperparameter tuning, and alternative regression techniques to improve predictive performance.

## Author

**Course:** MSCS 634  
**Lab:** Lab 4 – Regression Model Comparison
