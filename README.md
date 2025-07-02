# Multiclass Logistic Regression
A Python implementation of multiclass logistic regression with focus on the statistical analysis of the coefficients.

## 1 - Objective
In this project we have implemented a Python-based module of the multiclass logistic regression method. Its purpose is purely pedagogical and to some extend it mirrors [scikit-learn's LogisticRegression class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). We give some emphasis to the statistical analysis of the logistic regression coefficients. In particular, the code computes the standard errors and $Z$-statistics for the coefficients and their $p$-value against the hypothesis that there is no relation between dependent and preditor variables. These quantities are relevant to assess the accuracy for the estimation of the regression coefficients, the (linear) dependence of the *logit* with the predictors $X$ and the overall quality of the fit with the data. The module also produces the multiclass confusion matrix and some relevant classification scores.

## 2 - Repo organisation
**`logistic_regression/`: The linear regression modules**
- `logistic_regression.py`: The multiclass logistic regression module with its methods. See also Module architecture below.
- `_optimiser.py`: The optimisation module used to estimate the regression coefficients.
- `_utilities.py`: Functions used in the logistic regression and optimiser modules.

**`notebooks/:` Notebooks demonstrating the modules**
- `LogisticRegression.ipynb`: Notebook discussing the basics of the logistic regression 

## 3 - Module architecture
Description of the `logistic_regression` module architecture.
- `logistic_regression/__init__.py`
  - Initialises the module.
  - Imports the `LogisticRegressor` class, for multiclass logistic regression.

- `logistic_regression/logistic_regression.py`: defines the LogisticRegressor class with the methods
  - `fit`;
  - `predict_proba`;
  - `predict`; 
  - `scores`;
  - `confusion_matrix`;
  - `get_params`;
  - `regression_report`.

 - `logistic_regression/_optimiser.py`: defines the NewtonOptimiser class with the methods
  - `optimise`.

 - `logistic_regression/_utilities.py`: defines a number of functions used in the modules above.

## 4 - Features
- The `LogisticRegressor` class:
  - performs logistic regression in multiclass problems by maximising the log-likelyhood function using the Newton-Raphson method assuming that the probabilities are modelled by the softmax function;
  - predicts the estimated probabilities for all classes for samples in the predictor values $X$;
  - predicts the class labels for samples in the predictor values $X$;
  - computes a number of classification scores and produces the confusion matrix for all classes;
  - produces a statistical analysis of the coefficient estimates;
  - has the following methods:
    - `fit` fits the logistic regression model,
    - `predict_proba` predicts the estimated probabilities, 
    - `predict` predicts the class labels using model,
    - `scores` returns the prevalence, precision, recall, true/false positive rates and accuracy scores,
    - `confusion_matrix` returns the confusion matrix for the classification problem,
    - `get_params` returns the estimation of the regression coefficients,
    - `regression_report` returns a report on the statistical analysis of the estimators's coefficients, with standard error, $Z$-statistics and $p$-value against the null hypothesis.

- The `NewtonOptimiser` class:
  - finds the maximum of the log-likelyhood function by searching the zero of the norm of the gradient vector;
  - has the following methods:
    - `optimise` executes the Newton-Raphson method.

## 5 - Results

To test and illustrate the `LogisticRegressor`, we use the well-known [*Iris dataset*](https://en.wikipedia.org/wiki/Iris_flower_data_set). 

TO DO

## 6 - Bibliography

- G. James, D. Witten, T. Hastie and R. Tibshirani, *An Introduction to Statistical Learning*, Springer (2017).
- T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning: Data mining, Inference, and Prediction*, Springer (2017).
- M.N. Magalhães and A.C. Pedroso de Lima, *Noções de Probabilidade e Estatística*, Edusp (2023).
- R. A. Fisher, ["Iris", UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) (1936).
