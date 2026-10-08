# Polynomial Regression (IMT2024055)

Two polynomial regression problems: predicting y from 6 features (var1) and from 3 features (var2).

## Final models
| Problem | Degree | Model | CV MSE | CV R2 |
|---|---|---|---|---|
| var1 | 5 | StandardScaler + Lasso | 0.338 | 0.966 |
| var2 | 10 | Ridge (raw polynomial terms) | 0.280 | 0.993 |

## How it works
Features are expanded with scikit-learn PolynomialFeatures, then fit with a regularised linear model (Lasso or Ridge). Degree and regularisation strength were chosen by 5-fold cross-validation, with a 20% holdout split as a final check.

## Files
- `IMT2024055_polyreg.ipynb`: all code (data checks, degree search, final training, prediction)
- `IMT2024055_pred_var1.csv`, `IMT2024055_pred_var2.csv`: test predictions
- data CSVs and plots used in the report

## How to run
Open the notebook in Google Colab, upload the 5 CSV files, and run all cells (Runtime > Run all). The last training cell writes the two prediction files.
